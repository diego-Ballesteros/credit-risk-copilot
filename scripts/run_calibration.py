"""Compare calibration methods on the production forest, fitted inside the folds.

Run it with::

    uv run python scripts/run_calibration.py

**What calibration is for, and why it is a separate question from ranking.** PR-AUC, ROC-AUC,
KS, Gini and both precision-at-top metrics read only the *order* the scores induce. Brier
reads their *values*. A model can order clients perfectly and still say "38%" about a group
that defaults 22% of the time, and ADR-0002 keeps Brier because the declared business use -
expected loss, and therefore pricing and the operating threshold - is computed from the
value and not from the order.

**Why the calibrator has to be fitted inside the cross-validation.** A calibration map
fitted on the same rows the model was fitted on learns to correct errors it has already
seen, and the resulting reliability curve sits on the diagonal for a reason that has nothing
to do with the model being well calibrated. Here the map is fitted by
`CalibratedClassifierCV`'s own internal split of whatever data it is given, and that data is
one training fold, so no row is ever used to calibrate the score it is later judged by. The
curve itself is drawn from **out-of-fold** probabilities for the same reason.

**The two methods, and what they trade.**

- **Sigmoid (Platt scaling)** fits a two-parameter logistic map. It cannot bend into an
  arbitrary shape, which is exactly why it is hard to overfit; it is the right choice when
  the distortion is a smooth monotone squeeze.
- **Isotonic** fits a free monotone step function. It can correct any monotone distortion,
  and with that freedom it can also fit noise - it is the method that needs data, and the
  one whose advantage on a training set is least likely to survive.

Ranking is expected to be untouched: both maps are monotone, and a monotone transform
cannot change an order. The script reports PR-AUC anyway, because if it *does* move, the
assumption is wrong somewhere and that is worth knowing before anything is deployed.

Exit code 0 when every method was measured and recorded, 1 when the tracking server could
not be configured.
"""

import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

from credit_copilot.config import settings  # noqa: E402
from credit_copilot.console import enable_unicode_console  # noqa: E402
from credit_copilot.data.loader import load_dataset  # noqa: E402
from credit_copilot.models.estimators import (  # noqa: E402
    PRODUCTION_CALIBRATION_CV,
    build_calibrated_forest,
    build_production_forest,
)
from credit_copilot.models.evaluation import (  # noqa: E402
    DEFAULT_N_SPLITS,
    PRACTICAL_SIGNIFICANCE_THRESHOLD,
    CrossValidationResult,
    compare_to_reference,
    cross_val_probabilities,
    evaluate_and_log,
    format_comparison_table,
    split_features_and_target,
)
from credit_copilot.models.metrics import DECISION_METRIC, METRIC_NAMES  # noqa: E402
from credit_copilot.models.tracking import (  # noqa: E402
    LEAKAGE_CHECK_TAG,
    MLflowConfigurationError,
    ensure_experiment,
)

CALIBRATION_TAG_VALUE = "calibration"
"""Marks a run as one arm of the calibration comparison."""

REFERENCE_LABEL = "uncalibrated"
"""The arm the calibrated ones are judged against: the raw forest."""

CALIBRATION_METRIC = "brier"
"""The metric this script is about. Lower is better."""

N_CURVE_BINS = 10
"""Bins of the reliability curve.

Ten deciles of predicted probability, which is enough resolution to see a systematic bend
and coarse enough that each bin still holds ~3,000 rows, so a bin's observed frequency is
not itself noise.
"""

CURVE_ARTEFACT = "calibration_curves.png"
"""Filename of the reliability diagram attached to the run."""

CURVE_DATA_ARTEFACT = "calibration_curves.csv"
"""Filename of the numbers behind the diagram, so the picture can be audited."""

RULE = "=" * 100


def _arms() -> dict[str, tuple[object, str]]:
    """The three arms, in the order the report reads them.

    Returns:
        Label -> (estimator, one-line description).
    """
    return {
        REFERENCE_LABEL: (
            build_production_forest(),
            "the raw forest - a vote share, not a probability",
        ),
        "sigmoid": (
            build_calibrated_forest("sigmoid"),
            "Platt scaling: a two-parameter logistic map, hard to overfit",
        ),
        "isotonic": (
            build_calibrated_forest("isotonic"),
            "a free monotone step function: flexible, and able to fit noise",
        ),
    }


def _reliability_table(
    target: pd.Series,
    probabilities: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Build the reliability curve of every arm from out-of-fold probabilities.

    Args:
        target: The true labels.
        probabilities: Arm label -> out-of-fold probability of the positive class.

    Returns:
        One row per (arm, bin) with the mean predicted probability and the observed
        frequency of defaults in that bin.
    """
    rows = []
    for label, scores in probabilities.items():
        observed, predicted = calibration_curve(
            target.to_numpy(), scores.to_numpy(), n_bins=N_CURVE_BINS, strategy="quantile"
        )
        for index, (mean_predicted, frequency) in enumerate(zip(predicted, observed, strict=True)):
            rows.append(
                {
                    "arm": label,
                    "bin": index + 1,
                    "mean_predicted": float(mean_predicted),
                    "observed_frequency": float(frequency),
                    "gap": float(mean_predicted - frequency),
                }
            )
    return pd.DataFrame(rows)


def _draw_curves(table: pd.DataFrame, destination: Path) -> None:
    """Draw the reliability diagram of every arm on one pair of axes.

    Args:
        table: The output of `_reliability_table`.
        destination: File to write the PNG to.
    """
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfectly calibrated")
    for label, group in table.groupby("arm", sort=False):
        axes[0].plot(
            group["mean_predicted"], group["observed_frequency"], marker="o", label=str(label)
        )
    axes[0].set_xlabel("mean predicted probability of default")
    axes[0].set_ylabel("observed frequency of default")
    axes[0].set_title(f"Reliability, out-of-fold, {N_CURVE_BINS} quantile bins")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for label, group in table.groupby("arm", sort=False):
        axes[1].plot(group["bin"], group["gap"], marker="o", label=str(label))
    axes[1].axhline(0.0, linestyle="--", color="grey")
    axes[1].set_xlabel("bin (increasing predicted probability)")
    axes[1].set_ylabel("predicted minus observed")
    axes[1].set_title("Calibration gap: above zero is over-prediction")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(destination, dpi=130)
    plt.close(figure)


def main() -> int:
    """Measure the arms, draw their reliability curves, and record everything.

    Returns:
        0 if every arm was measured and recorded, 1 if MLflow could not be configured.
    """
    enable_unicode_console()
    try:
        context = ensure_experiment()
    except MLflowConfigurationError as error:
        print(f"MLflow is not configured:\n{error}", file=sys.stderr)
        return 1

    print(RULE)
    print("CALIBRATION - two methods against the raw forest")
    print(RULE)
    print(f"Tracking server : {context.tracking_uri}")
    print(f"Experiment      : {context.name} (id {context.experiment_id})")

    frame = load_dataset()
    features, target = split_features_and_target(frame)
    prevalence = float(target.mean())
    print(f"Rows            : {len(frame):,}")
    print(f"Prevalence      : {prevalence:.6f}  <- what a calibrated mean must reproduce")
    print("Model           : build_production_forest(), identical in all three arms")
    print(f"Validation      : StratifiedKFold({DEFAULT_N_SPLITS}, shuffle=True), ", end="")
    print(f"random_state={settings.random_state} from config.py")
    print(f"Calibrator      : fitted inside each fold, on {PRODUCTION_CALIBRATION_CV} ", end="")
    print("internal splits of that fold's training rows only")

    results: dict[str, CrossValidationResult] = {}
    run_ids: dict[str, str] = {}
    probabilities: dict[str, pd.Series] = {}

    for label, (estimator, description) in _arms().items():
        print(f"\n-> {label}: {description}")
        result, run_id = evaluate_and_log(
            estimator,
            features,
            target,
            run_name=f"calibration-{label}",
            context=context,
            tags={
                LEAKAGE_CHECK_TAG: CALIBRATION_TAG_VALUE,
                "phase": "02-modeling",
                "calibration_method": label,
                "base_model": "RandomForestClassifier",
            },
        )
        results[label] = result
        run_ids[label] = run_id
        probabilities[label] = cross_val_probabilities(estimator, features, target)
        print(
            f"   {CALIBRATION_METRIC} = {result.means[CALIBRATION_METRIC]:.6f} "
            f"± {result.stds[CALIBRATION_METRIC]:.6f}   "
            f"{DECISION_METRIC} = {result.means[DECISION_METRIC]:.4f}   run {run_id}"
        )

    print("\n" + RULE)
    print(f"THE SEVEN METRICS OF ADR-0002, mean ± std over {DEFAULT_N_SPLITS} folds")
    print(RULE)
    print(format_comparison_table(results, METRIC_NAMES))

    reference = results[REFERENCE_LABEL]
    print(f"\nCalibration: {CALIBRATION_METRIC} against the uncalibrated forest (lower is better)")
    print("-" * 100)
    print(
        f"{'arm':<16}{'brier':>12}{'std':>10}{'vs raw':>11}"
        f"{'mean predicted':>17}{'prevalence':>13}{'  bias':<10}"
    )
    for label, result in results.items():
        delta = result.means[CALIBRATION_METRIC] - reference.means[CALIBRATION_METRIC]
        mean_predicted = float(probabilities[label].mean())
        print(
            f"{label:<16}{result.means[CALIBRATION_METRIC]:>12.6f}"
            f"{result.stds[CALIBRATION_METRIC]:>10.6f}{delta:>+11.6f}"
            f"{mean_predicted:>17.6f}{prevalence:>13.6f}  {mean_predicted - prevalence:>+.6f}"
        )
    print(
        "\n'mean predicted' against 'prevalence' is the first-order check: a calibrated model\n"
        "predicts, on average, the rate the population actually has. It is necessary and not\n"
        "sufficient - a model can get the mean right and still be wrong in every bin."
    )

    print(f"\nRanking: {DECISION_METRIC} against the uncalibrated forest")
    print("-" * 100)
    for label, result in results.items():
        if label == REFERENCE_LABEL:
            continue
        item = compare_to_reference(label, result, reference)
        print(
            f"  {item.label:<14}{item.value:>10.4f}  vs {item.reference_value:.4f}  "
            f"difference {item.difference:+.4f}   {item.verdict}"
        )
    print(
        "\nBoth maps are monotone, so the order should not move at all. A difference here\n"
        f"larger than {PRACTICAL_SIGNIFICANCE_THRESHOLD:.2f} would mean the assumption is wrong."
    )

    table = _reliability_table(target, probabilities)
    print(f"\nReliability, out-of-fold, {N_CURVE_BINS} quantile bins")
    print("-" * 100)
    pivot = table.pivot(index="bin", columns="arm", values="gap")
    print("gap = mean predicted - observed frequency (positive means over-prediction)")
    print(pivot.to_string(float_format=lambda value: f"{value:+.4f}"))
    print("\nWorst absolute gap in any bin:")
    for label in probabilities:
        worst = table.loc[table["arm"] == label, "gap"].abs().max()
        print(f"    {label:<14}{worst:.4f}")

    with mlflow.start_run(experiment_id=context.experiment_id, run_name="calibration-curves"):
        mlflow.set_tags(
            {
                LEAKAGE_CHECK_TAG: CALIBRATION_TAG_VALUE,
                "phase": "02-modeling",
                "artefact_only": "true",
            }
        )
        with tempfile.TemporaryDirectory() as staging:
            directory = Path(staging)
            _draw_curves(table, directory / CURVE_ARTEFACT)
            table.to_csv(directory / CURVE_DATA_ARTEFACT, index=False)
            mlflow.log_artifact(str(directory / CURVE_ARTEFACT))
            mlflow.log_artifact(str(directory / CURVE_DATA_ARTEFACT))
        mlflow.log_metrics(
            {
                f"worst_gap_{label}": float(table.loc[table["arm"] == label, "gap"].abs().max())
                for label in probabilities
            }
        )

    best = min(results, key=lambda label: results[label].means[CALIBRATION_METRIC])
    print("\n" + RULE)
    print("VERDICT")
    print(RULE)
    print(f"Best {CALIBRATION_METRIC}: {best} ({results[best].means[CALIBRATION_METRIC]:.6f})")
    improvement = reference.means[CALIBRATION_METRIC] - results[best].means[CALIBRATION_METRIC]
    print(f"Improvement over the raw forest: {improvement:+.6f} in {CALIBRATION_METRIC}")
    ranking_cost = results[best].means[DECISION_METRIC] - reference.means[DECISION_METRIC]
    print(f"Ranking cost: {ranking_cost:+.4f} in {DECISION_METRIC}")

    print("\n" + RULE)
    print("MLFLOW")
    print(RULE)
    print(f"Experiment: {context.url}")
    for label, run_id in run_ids.items():
        print(f"    {label:<16} run_id {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
