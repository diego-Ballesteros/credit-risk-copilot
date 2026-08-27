"""Measure the floor of the project: three baselines, recorded in MLflow.

Run it with::

    uv run python scripts/run_baselines.py

Nothing measured after this turn means anything without these numbers. Section 6.1 of
`docs/METHODOLOGY.md` puts it as a rule - every metric is reported with its baseline beside
it - and this script is what produces the right-hand side of that comparison.

**The three baselines are three different floors, and that is the point.**

1. **Majority class.** The absolute floor. It answers "does not default" to everybody, and
   its accuracy is the number ADR-0002 refuses to decide on. Printed here next to its
   PR-AUC so the argument is visible rather than quoted.
2. **Stratified random.** The floor of a *ranking* metric. The first baseline has no
   ordering at all; this one has an ordering built from the class prior and no information,
   which is what a ranking metric should score at chance.
3. **L2 logistic regression, balanced class weights.** Not a floor - the cheapest real
   model. It is the number every later model has to beat to justify its complexity, and if
   a gradient boosting ends up two points above it, that is what the two points cost.

**Where the data comes from, and where it deliberately does not.** The table is loaded
through `loader.load_dataset` and the preprocessor is fitted inside each fold.
`data/processed/features.parquet` is never read: it was fitted on all 30,000 rows, and
evaluating on it would measure rows whose scaling had already seen them.

Exit code 0 when every baseline was measured and recorded, 1 when the tracking server could
not be configured. A run that prints metrics it failed to record is worse than one that
stops, because the metrics look like evidence and are not.
"""

import sys

from sklearn.dummy import DummyClassifier

from credit_copilot.config import settings
from credit_copilot.data.loader import load_dataset
from credit_copilot.models.estimators import build_logistic_regression
from credit_copilot.models.evaluation import (
    DEFAULT_N_SPLITS,
    CrossValidationResult,
    evaluate_and_log,
    format_comparison_table,
    format_fold_table,
    null_reference,
    split_features_and_target,
)
from credit_copilot.models.metrics import (
    ACCURACY_METRIC,
    DECISION_METRIC,
    LOWER_IS_BETTER,
    METRIC_NAMES,
)
from credit_copilot.models.tracking import (
    BASELINE_TAG_VALUE,
    LEAKAGE_CHECK_TAG,
    MLflowConfigurationError,
    ensure_experiment,
)

RULE = "=" * 92


def _build_baselines() -> dict[str, tuple[object, str]]:
    """The three estimators to measure, in the order the report reads them.

    Returns:
        Label -> (estimator, one-line description of what floor it represents).
    """
    return {
        "majority-class": (
            DummyClassifier(strategy="most_frequent"),
            "always answers 'does not default' - the absolute floor",
        ),
        "stratified-random": (
            DummyClassifier(strategy="stratified", random_state=settings.random_state),
            "orders by the class prior alone - the floor of a ranking metric",
        ),
        "logistic-l2-balanced": (
            build_logistic_regression(),
            "the cheapest real model - what any complex model has to beat",
        ),
    }


def _print_accuracy_argument(results: dict[str, CrossValidationResult]) -> None:
    """Print accuracy next to PR-AUC, which is the whole case ADR-0002 makes against it.

    Args:
        results: The measurements, keyed by label.
    """
    print("\nWhy ADR-0002 does not decide on accuracy")
    print("-" * 92)
    print(f"{'model':<24}{'accuracy':>16}{'PR-AUC':>16}{'PR-AUC floor':>18}{'lift over floor':>18}")
    for label, result in results.items():
        floor = result.positive_rate
        decision = result.means[DECISION_METRIC]
        print(
            f"{label:<24}{result.means[ACCURACY_METRIC]:>16.4f}{decision:>16.4f}"
            f"{floor:>18.4f}{decision - floor:>+18.4f}"
        )
    print(
        "\nThe majority-class baseline wins on accuracy and identifies no defaulter at all: "
        "its\nPR-AUC sits on the floor, which is the prevalence. Accuracy is the one metric "
        "here that\ncannot fail, and a metric that cannot fail proves nothing."
    )


def main() -> int:
    """Load, measure the three baselines, record them, and print the comparison.

    Returns:
        0 if every baseline was measured and recorded, 1 if MLflow could not be configured.
    """
    try:
        context = ensure_experiment()
    except MLflowConfigurationError as error:
        print(f"MLflow is not configured:\n{error}", file=sys.stderr)
        return 1

    print(RULE)
    print("BASELINES - the floor every later measurement is read against")
    print(RULE)
    print(f"Tracking server : {context.tracking_uri}")
    print(f"Experiment      : {context.name} (id {context.experiment_id})")

    frame = load_dataset()
    features, target = split_features_and_target(frame)
    print(f"Rows            : {len(frame):,}")
    print(f"Raw columns     : {features.shape[1]} predictors, target held out")
    print(f"Prevalence      : {target.mean():.6f}  <- the floor of PR-AUC and of both p@top-k%")
    print(f"Validation      : StratifiedKFold({DEFAULT_N_SPLITS}, shuffle=True), ", end="")
    print(f"random_state={settings.random_state} from config.py")
    print("Preprocessing   : fitted inside each fold, on that fold's training rows only")

    results: dict[str, CrossValidationResult] = {}
    run_ids: dict[str, str] = {}

    for label, (estimator, description) in _build_baselines().items():
        print(f"\n-> {label}: {description}")
        result, run_id = evaluate_and_log(
            estimator,
            features,
            target,
            run_name=label,
            context=context,
            tags={
                LEAKAGE_CHECK_TAG: BASELINE_TAG_VALUE,
                "phase": "02-modeling",
                "baseline_kind": description,
            },
        )
        results[label] = result
        run_ids[label] = run_id
        print(
            f"   {DECISION_METRIC} = {result.means[DECISION_METRIC]:.4f} "
            f"± {result.stds[DECISION_METRIC]:.4f}   run {run_id}"
        )

    print("\n" + RULE)
    print(f"THE SEVEN METRICS OF ADR-0002, mean ± std over {DEFAULT_N_SPLITS} folds")
    print(RULE)
    print(format_comparison_table(results, METRIC_NAMES))

    floors = null_reference(next(iter(results.values())))
    print("\nThe no-signal floor of each metric, for reading the table above")
    print("-" * 92)
    for name in METRIC_NAMES:
        note = "lower is better" if name in LOWER_IS_BETTER else ""
        print(f"{name:<24}{floors[name]:>10.4f}   {note}")

    _print_accuracy_argument(results)

    for label, result in results.items():
        print(f"\nPer fold - {label}")
        print("-" * 92)
        print(format_fold_table(result, METRIC_NAMES))
        rates = ", ".join(f"{rate:.4f}" for rate in result.fold_positive_rates)
        print(f"validation-fold positive rate: {rates}")

    print("\n" + RULE)
    print("MLFLOW")
    print(RULE)
    print(f"Experiment: {context.url}")
    for label, run_id in run_ids.items():
        print(f"    {label:<24} run_id {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
