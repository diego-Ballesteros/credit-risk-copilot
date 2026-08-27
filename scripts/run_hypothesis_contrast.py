"""Contrast the project's main hypothesis: payment behaviour against demography.

Run it with::

    uv run python scripts/run_hypothesis_contrast.py

**The hypothesis, as `docs/ROADMAP.md` states it.** Recent payment behaviour - the
delinquency trajectory, the evolution of credit utilisation and the payment-to-balance ratio
- predicts default better than the client's static demographic attributes. The roadmap also
states how it is falsified: train one model on demography alone and one on behaviour alone,
and compare.

**What makes this a contrast and not three unrelated measurements.** The three arms share
one estimator built by one constructor, one preprocessor fitted once per fold, one splitter
and one seed. The single thing that differs is the argument naming which group of columns
survives to the model, and that argument is the only difference the code permits: the
estimator comes from `build_logistic_regression()`, which takes no parameters precisely so
that two arms cannot drift apart. If the metric moves, the columns are the only thing that
could have moved it.

**How the columns are restricted, and why not earlier.** Selection happens on the
preprocessor's *output*, not on the raw table. The preprocessor builds the behaviour features
out of the raw blocks and addresses every source column by name, so a table with the
demographic columns removed would not merely be smaller - it would fail, or it would change
what the surviving features are worth. See `models/feature_groups` for the full reasoning and
for the one honest caveat: the utilisation ratios are built on `LIMIT_BAL`, which is counted
as demographic, so the two groups partition the columns without perfectly partitioning the
information.

Exit code 0 when the three arms were measured and recorded, 1 when the tracking server could
not be configured.
"""

import sys
from collections.abc import Mapping, Sequence

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from credit_copilot.config import settings
from credit_copilot.data.loader import load_dataset
from credit_copilot.models.estimators import build_logistic_regression
from credit_copilot.models.evaluation import (
    DEFAULT_N_SPLITS,
    CrossValidationResult,
    cross_validate_estimator,
    evaluate_and_log,
    format_comparison_table,
    format_fold_table,
    split_features_and_target,
)
from credit_copilot.models.feature_groups import (
    ALL_SOURCE_COLUMNS,
    BEHAVIOURAL_SOURCE_COLUMNS,
    DEMOGRAPHIC_SOURCE_COLUMNS,
    SelectFeatureGroup,
)
from credit_copilot.models.metrics import DECISION_METRIC, METRIC_NAMES
from credit_copilot.models.tracking import (
    LEAKAGE_CHECK_TAG,
    MLflowConfigurationError,
    ensure_experiment,
)

CONTRAST_TAG_VALUE = "hypothesis-contrast"
"""Marks a run as one arm of the main-hypothesis contrast, not a candidate model."""

RULE = "=" * 96


def build_arm(source_columns: Sequence[str]) -> Pipeline:
    """Assemble one arm of the contrast: select a column group, then the canonical model.

    Every arm is built by this one function, so the three differ in exactly one argument.
    The estimator is constructed rather than passed in, which removes the only other way two
    arms could come to disagree.

    Args:
        source_columns: The source columns whose matrix columns reach the model.

    Returns:
        An unfitted two-step pipeline. It is handed to `cross_validate_estimator` as the
        estimator, so the preprocessor still sits outside it and is fitted once per fold.
    """
    return Pipeline(
        [
            ("select", SelectFeatureGroup(source_columns)),
            ("model", build_logistic_regression()),
        ]
    )


def _arms() -> dict[str, tuple[Sequence[str], str]]:
    """The three arms, in the order the report reads them.

    Returns:
        Label -> (source columns, one-line description).
    """
    return {
        "demography-only": (
            DEMOGRAPHIC_SOURCE_COLUMNS,
            "what the client is: sex, education, marriage, age, credit limit",
        ),
        "behaviour-only": (
            BEHAVIOURAL_SOURCE_COLUMNS,
            "what the client did: repayment status, bills, payments, derived features",
        ),
        "all-columns": (
            ALL_SOURCE_COLUMNS,
            "both groups together - what demography adds on top of behaviour",
        ),
    }


def _print_distance_table(
    results: Mapping[str, CrossValidationResult],
    trivial: CrossValidationResult,
) -> None:
    """Print each arm's distance from the trivial baseline, in metric units and in folds.

    The second column is the one that decides whether a difference is real. A gap smaller
    than the spread between folds is a gap this protocol cannot resolve, and reporting it as
    an improvement would be reporting noise.

    Args:
        results: The three arms, keyed by label.
        trivial: The majority-class baseline, measured on the same folds.
    """
    floor = trivial.means[DECISION_METRIC]
    print(f"\nDistance from the trivial baseline, on {DECISION_METRIC}")
    print("-" * 96)
    print(
        f"{'arm':<20}{'PR-AUC':>12}{'std':>10}{'floor':>10}"
        f"{'lift':>12}{'lift / std':>14}{'columns':>12}"
    )
    for label, result in results.items():
        value = result.means[DECISION_METRIC]
        spread = result.stds[DECISION_METRIC]
        lift = value - floor
        print(
            f"{label:<20}{value:>12.4f}{spread:>10.4f}{floor:>10.4f}"
            f"{lift:>+12.4f}{lift / spread:>14.1f}{result.n_features:>12}"
        )


def _print_increment(results: Mapping[str, CrossValidationResult]) -> None:
    """Print what each group adds on top of the other, which is the hypothesis itself.

    Args:
        results: The three arms, keyed by label.
    """
    behaviour = results["behaviour-only"]
    demography = results["demography-only"]
    everything = results["all-columns"]

    print("\nWhat each group adds on top of the other")
    print("-" * 96)
    pairs = (
        (
            "demography adds to behaviour",
            everything.means[DECISION_METRIC] - behaviour.means[DECISION_METRIC],
            behaviour.stds[DECISION_METRIC],
        ),
        (
            "behaviour adds to demography",
            everything.means[DECISION_METRIC] - demography.means[DECISION_METRIC],
            demography.stds[DECISION_METRIC],
        ),
    )
    for label, increment, spread in pairs:
        verdict = "within fold noise" if abs(increment) <= spread else "larger than fold noise"
        print(f"  {label:<32}{increment:>+10.4f}   ({increment / spread:>5.1f} std)   {verdict}")


def main() -> int:
    """Measure the three arms, record them, and print the contrast.

    Returns:
        0 if every arm was measured and recorded, 1 if MLflow could not be configured.
    """
    try:
        context = ensure_experiment()
    except MLflowConfigurationError as error:
        print(f"MLflow is not configured:\n{error}", file=sys.stderr)
        return 1

    print(RULE)
    print("MAIN HYPOTHESIS - payment behaviour against demographic attributes")
    print(RULE)
    print(f"Tracking server : {context.tracking_uri}")
    print(f"Experiment      : {context.name} (id {context.experiment_id})")

    frame = load_dataset()
    features, target = split_features_and_target(frame)
    print(f"Rows            : {len(frame):,}")
    print(f"Prevalence      : {target.mean():.6f}")
    print(f"Validation      : StratifiedKFold({DEFAULT_N_SPLITS}, shuffle=True), ", end="")
    print(f"random_state={settings.random_state} from config.py")
    print("Estimator       : build_logistic_regression() - identical in all three arms")
    print("Preprocessing   : one preprocessor, fitted inside each fold; selection is on its")
    print("                  output, so a kept column carries the same value in every arm")

    print("\n-> majority-class (the baseline every arm is read against)")
    trivial = cross_validate_estimator(DummyClassifier(strategy="most_frequent"), features, target)
    print(f"   {DECISION_METRIC} = {trivial.means[DECISION_METRIC]:.4f}")
    print("   already recorded in MLflow by run_baselines.py; measured here, not logged again")

    results: dict[str, CrossValidationResult] = {}
    run_ids: dict[str, str] = {}

    for label, (source_columns, description) in _arms().items():
        print(f"\n-> {label}: {description}")
        result, run_id = evaluate_and_log(
            build_arm(source_columns),
            features,
            target,
            run_name=label,
            context=context,
            tags={
                LEAKAGE_CHECK_TAG: CONTRAST_TAG_VALUE,
                "phase": "02-modeling",
                "feature_group": label,
                "n_source_columns": str(len(source_columns)),
            },
        )
        results[label] = result
        run_ids[label] = run_id
        print(
            f"   {DECISION_METRIC} = {result.means[DECISION_METRIC]:.4f} "
            f"± {result.stds[DECISION_METRIC]:.4f}   "
            f"{result.n_features} matrix columns   run {run_id}"
        )

    print("\n" + RULE)
    print(f"THE SEVEN METRICS OF ADR-0002, mean ± std over {DEFAULT_N_SPLITS} folds")
    print(RULE)
    print(format_comparison_table({"majority-class": trivial, **results}, METRIC_NAMES))

    _print_distance_table(results, trivial)
    _print_increment(results)

    for label, result in results.items():
        print(f"\nPer fold - {label}")
        print("-" * 96)
        print(format_fold_table(result, METRIC_NAMES))

    print("\n" + RULE)
    print("MLFLOW")
    print(RULE)
    print(f"Experiment: {context.url}")
    for label, run_id in run_ids.items():
        print(f"    {label:<20} run_id {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
