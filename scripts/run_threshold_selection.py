"""Choose the operating threshold from a cost matrix, not from the model.

Run it with::

    uv run python scripts/run_threshold_selection.py

**Why the threshold is not a modelling decision.** ADR-0002 discards F1 as the decision
metric for exactly this reason: it fixes a threshold by an internal criterion of the
classifier, and the classifier has no way of knowing what a mistake costs. PR-AUC evaluates
the model at every threshold precisely so that the choice of one can be deferred to the
point where the costs are known. This script is that point, and the input it needs is a
business assumption, stated out loud.

**The assumption, and why it is stated as a ratio.** A false negative - lending to a client
who defaults - costs the principal. A false positive - refusing a client who would have
paid - costs the margin that was not earned. The default assumption here is that a false
negative costs **5 times** a false positive. Only the ratio matters: the threshold that
minimises `5*FN + 1*FP` is the same as the one that minimises `5000*FN + 1000*FP`, so
nothing here depends on knowing the absolute size of a loan.

**Why the sensitivity analysis is not optional.** The ratio is an assumption, and an
assumption that moves the answer a lot deserves more scrutiny than one that does not. The
script reports the threshold at 3:1 and 10:1 alongside 5:1 so the reader can see how much of
the recommendation is data and how much is the guess.

**Where the probabilities come from.** Out-of-fold, from `cross_val_probabilities`: every
row is scored by a model that was not fitted on it. Choosing a threshold on in-sample
probabilities would tune it to rows the model had already memorised, and the confusion
matrix printed here would be a description of the training set.

Exit code 0 when the sweep completed and was recorded, 1 when the tracking server could not
be configured.
"""

import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_copilot.config import settings  # noqa: E402
from credit_copilot.console import enable_unicode_console  # noqa: E402
from credit_copilot.data.loader import load_dataset  # noqa: E402
from credit_copilot.models.estimators import (  # noqa: E402
    PRODUCTION_CALIBRATION_METHOD,
    build_calibrated_forest,
)
from credit_copilot.models.evaluation import (  # noqa: E402
    DEFAULT_N_SPLITS,
    cross_val_probabilities,
    split_features_and_target,
)
from credit_copilot.models.tracking import (  # noqa: E402
    LEAKAGE_CHECK_TAG,
    MLflowConfigurationError,
    ensure_experiment,
)

THRESHOLD_TAG_VALUE: Final[str] = "threshold-selection"
"""Marks the run that fixes the operating threshold."""

PRIMARY_COST_RATIO: Final[float] = 5.0
"""How many false positives one false negative is worth. The project's stated assumption."""

SENSITIVITY_RATIOS: Final[tuple[float, ...]] = (3.0, 5.0, 10.0)
"""Ratios the sweep is repeated at, so the reader sees how far the answer travels."""

THRESHOLD_GRID: Final[np.ndarray] = np.round(np.arange(0.0, 1.0005, 0.005), 4)
"""Thresholds swept, from 0 to 1 in steps of 0.005.

Two hundred points is finer than the decision can possibly need - moving the cut by 0.005
moves a handful of clients out of 30,000 - and coarse enough to print. The grid starts at 0
and ends at 1 so that the degenerate ends, refuse everybody and accept everybody, are
included rather than assumed away.
"""

SWEEP_ARTEFACT: Final[str] = "threshold_sweep.csv"
"""Filename of the full sweep attached to the run."""

CURVE_ARTEFACT: Final[str] = "threshold_cost_curve.png"
"""Filename of the cost curve attached to the run."""

RULE: Final[str] = "=" * 100


@dataclass(frozen=True)
class Operating:
    """One threshold and everything it implies, in counts rather than rates.

    Attributes:
        ratio: Cost of a false negative in units of a false positive.
        threshold: Probability at or above which a client is refused.
        true_positives: Defaulters correctly refused.
        false_positives: Payers wrongly refused.
        true_negatives: Payers correctly accepted.
        false_negatives: Defaulters wrongly accepted.
        expected_cost: `ratio * false_negatives + false_positives`, in false-positive units.
    """

    ratio: float
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    expected_cost: float

    @property
    def refused(self) -> int:
        """How many clients are refused in total.

        Returns:
            The count of rows scored at or above the threshold.
        """
        return self.true_positives + self.false_positives

    @property
    def recall(self) -> float:
        """Share of the defaulters that the threshold catches.

        Returns:
            Recall in [0, 1].
        """
        caught = self.true_positives + self.false_negatives
        return self.true_positives / caught if caught else 0.0

    @property
    def precision(self) -> float:
        """Share of the refused clients that would really have defaulted.

        Returns:
            Precision in [0, 1], and 0 when nobody is refused.
        """
        return self.true_positives / self.refused if self.refused else 0.0


def sweep(target: pd.Series, probabilities: pd.Series, ratio: float) -> pd.DataFrame:
    """Evaluate every threshold on the grid under one cost ratio.

    Args:
        target: True labels.
        probabilities: Out-of-fold probability of default, aligned to `target`.
        ratio: Cost of a false negative in units of a false positive.

    Returns:
        One row per threshold, with the confusion counts and the expected cost.
    """
    labels = target.to_numpy()
    scores = probabilities.to_numpy()
    positives = labels == 1

    rows = []
    for threshold in THRESHOLD_GRID:
        refused = scores >= threshold
        true_positives = int(np.sum(refused & positives))
        false_positives = int(np.sum(refused & ~positives))
        false_negatives = int(np.sum(~refused & positives))
        true_negatives = int(np.sum(~refused & ~positives))
        rows.append(
            {
                "threshold": float(threshold),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
                "expected_cost": ratio * false_negatives + false_positives,
            }
        )
    return pd.DataFrame(rows)


def best_operating_point(table: pd.DataFrame, ratio: float) -> Operating:
    """Pick the cheapest threshold, breaking ties toward the higher cut.

    Ties are broken toward the **higher** threshold, which refuses fewer clients. When two
    cuts cost the same, the one that intervenes less is the one to prefer: refusing a client
    is an action with consequences the cost matrix does not capture, and doing it to fewer
    people for the same expected loss is strictly better.

    Args:
        table: The output of `sweep`.
        ratio: The ratio the table was computed under.

    Returns:
        The chosen operating point.
    """
    cheapest = table["expected_cost"].min()
    candidates = table.loc[table["expected_cost"] == cheapest]
    row = candidates.iloc[-1]
    return Operating(
        ratio=ratio,
        threshold=float(row["threshold"]),
        true_positives=int(row["true_positives"]),
        false_positives=int(row["false_positives"]),
        true_negatives=int(row["true_negatives"]),
        false_negatives=int(row["false_negatives"]),
        expected_cost=float(row["expected_cost"]),
    )


def _draw_cost_curves(
    tables: Mapping[float, pd.DataFrame], points: Sequence[Operating], destination: Path
) -> None:
    """Draw the expected cost against the threshold, one line per ratio.

    Args:
        tables: Ratio -> its sweep.
        points: The chosen operating point of each ratio, marked on its line.
        destination: File to write the PNG to.
    """
    figure, axis = plt.subplots(figsize=(11, 6))
    for ratio, table in tables.items():
        axis.plot(table["threshold"], table["expected_cost"], label=f"FN:FP = {ratio:g}:1")
    for point in points:
        axis.axvline(point.threshold, linestyle=":", alpha=0.6)
        axis.annotate(
            f"{point.threshold:.3f}",
            xy=(point.threshold, point.expected_cost),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xlabel("threshold: refuse the client at or above this probability")
    axis.set_ylabel("expected cost, in false-positive units")
    axis.set_title("Expected cost against the operating threshold, out-of-fold on 30,000 clients")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(destination, dpi=130)
    plt.close(figure)


def _print_business_reading(point: Operating, total: int) -> None:
    """Translate one operating point into the sentences a credit committee would use.

    Args:
        point: The chosen operating point.
        total: Number of clients scored.
    """
    defaulters = point.true_positives + point.false_negatives
    payers = point.false_positives + point.true_negatives
    print(f"\nWhat the {point.ratio:g}:1 threshold means for {total:,} clients")
    print("-" * 100)
    print(
        f"  Refuse           {point.refused:>7,} clients ({point.refused / total:.1%} of the book)"
    )
    print(
        f"  Of those refused {point.true_positives:>7,} would really have defaulted "
        f"({point.precision:.1%} of the refusals were right)"
    )
    print(
        f"  and              {point.false_positives:>7,} would have paid - good clients lost "
        f"({point.false_positives / payers:.1%} of all payers)"
    )
    print(
        f"  Defaulters caught{point.true_positives:>7,} of {defaulters:,} "
        f"({point.recall:.1%} - the rest are approved and default)"
    )
    print(f"  Defaulters missed{point.false_negatives:>7,}")
    print(f"  Expected cost    {point.expected_cost:>7,.0f} false-positive units")


def main() -> int:
    """Sweep the thresholds, choose one, test its sensitivity, and record it.

    Returns:
        0 if the sweep completed and was recorded, 1 if MLflow could not be configured.
    """
    enable_unicode_console()
    try:
        context = ensure_experiment()
    except MLflowConfigurationError as error:
        print(f"MLflow is not configured:\n{error}", file=sys.stderr)
        return 1

    print(RULE)
    print("OPERATING THRESHOLD - chosen from a cost matrix, not from the model")
    print(RULE)
    print(f"Tracking server : {context.tracking_uri}")
    print(f"Experiment      : {context.name} (id {context.experiment_id})")

    frame = load_dataset()
    features, target = split_features_and_target(frame)
    total = len(frame)
    print(f"Rows            : {total:,}")
    print(f"Prevalence      : {target.mean():.6f}")
    print(f"Model           : production forest + {PRODUCTION_CALIBRATION_METHOD} calibration")
    print(
        f"Probabilities   : out-of-fold, StratifiedKFold({DEFAULT_N_SPLITS}, shuffle=True), ",
        end="",
    )
    print(f"random_state={settings.random_state}")
    print(f"Grid            : {len(THRESHOLD_GRID)} thresholds from 0.000 to 1.000")
    print(f"Cost assumption : one false negative costs {PRIMARY_COST_RATIO:g} false positives")

    print("\nScoring every row by a model that never saw it...")
    probabilities = cross_val_probabilities(
        build_calibrated_forest(PRODUCTION_CALIBRATION_METHOD), features, target
    )
    print(f"   scored {len(probabilities):,} rows, mean probability {probabilities.mean():.6f}")

    tables = {ratio: sweep(target, probabilities, ratio) for ratio in SENSITIVITY_RATIOS}
    points = [best_operating_point(tables[ratio], ratio) for ratio in SENSITIVITY_RATIOS]
    primary = next(point for point in points if point.ratio == PRIMARY_COST_RATIO)

    print("\n" + RULE)
    print(f"THE CHOSEN THRESHOLD, at {PRIMARY_COST_RATIO:g}:1")
    print(RULE)
    print(f"  threshold      {primary.threshold:.3f}")
    print(f"  expected cost  {primary.expected_cost:,.0f} false-positive units")
    print("\n  Confusion matrix, absolute counts, out-of-fold")
    print("  " + "-" * 60)
    print(f"  {'':<22}{'predicted pay':>18}{'predicted default':>20}")
    print(f"  {'actually paid':<22}{primary.true_negatives:>18,}{primary.false_positives:>20,}")
    print(
        f"  {'actually defaulted':<22}{primary.false_negatives:>18,}{primary.true_positives:>20,}"
    )
    print(f"\n  recall {primary.recall:.4f}   precision {primary.precision:.4f}")

    _print_business_reading(primary, total)

    print("\n" + RULE)
    print("SENSITIVITY - how far the answer travels with the assumption")
    print(RULE)
    print(
        f"{'FN:FP':<10}{'threshold':>12}{'refused':>12}{'caught':>10}{'missed':>10}"
        f"{'good lost':>12}{'recall':>10}{'precision':>12}"
    )
    for point in points:
        print(
            f"{point.ratio:g}:1{'':<6}{point.threshold:>12.3f}{point.refused:>12,}"
            f"{point.true_positives:>10,}{point.false_negatives:>10,}"
            f"{point.false_positives:>12,}{point.recall:>10.4f}{point.precision:>12.4f}"
        )

    span = max(point.threshold for point in points) - min(point.threshold for point in points)
    refusal_span = max(point.refused for point in points) - min(point.refused for point in points)
    print(
        f"\nAcross 3:1 to 10:1 the threshold moves {span:.3f} and the number of refusals moves "
        f"{refusal_span:,}\nclients, {refusal_span / total:.1%} of the book. That span is the part "
        "of the recommendation that\ncomes from the assumption rather than from the data."
    )

    with mlflow.start_run(
        experiment_id=context.experiment_id, run_name="threshold-selection"
    ) as run:
        mlflow.set_tags(
            {
                LEAKAGE_CHECK_TAG: THRESHOLD_TAG_VALUE,
                "phase": "02-modeling",
                "calibration_method": PRODUCTION_CALIBRATION_METHOD,
                "probabilities": "out-of-fold",
            }
        )
        mlflow.log_params(
            {
                "cost_ratio_primary": str(PRIMARY_COST_RATIO),
                "cost_ratios_tested": str(list(SENSITIVITY_RATIOS)),
                "grid_points": str(len(THRESHOLD_GRID)),
                "random_state": str(settings.random_state),
                "n_splits": str(DEFAULT_N_SPLITS),
            }
        )
        mlflow.log_metrics(
            {
                "operating_threshold": primary.threshold,
                "expected_cost": primary.expected_cost,
                "recall_at_threshold": primary.recall,
                "precision_at_threshold": primary.precision,
                "refused_clients": float(primary.refused),
                "defaulters_caught": float(primary.true_positives),
                "defaulters_missed": float(primary.false_negatives),
                "good_clients_lost": float(primary.false_positives),
                **{f"threshold_at_{point.ratio:g}to1": point.threshold for point in points},
            }
        )
        with tempfile.TemporaryDirectory() as staging:
            directory = Path(staging)
            combined = pd.concat(
                [table.assign(cost_ratio=ratio) for ratio, table in tables.items()]
            )
            combined.to_csv(directory / SWEEP_ARTEFACT, index=False)
            _draw_cost_curves(tables, points, directory / CURVE_ARTEFACT)
            mlflow.log_artifact(str(directory / SWEEP_ARTEFACT))
            mlflow.log_artifact(str(directory / CURVE_ARTEFACT))
        run_id = run.info.run_id

    print("\n" + RULE)
    print("MLFLOW")
    print(RULE)
    print(f"Experiment: {context.url}")
    print(f"    threshold-selection    run_id {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
