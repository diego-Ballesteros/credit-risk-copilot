"""Compare the candidate models against the logistic regression, on one partition.

Run it with::

    uv run python scripts/run_model_comparison.py

**What is being asked.** The logistic regression reached PR-AUC 0.540173 +/- 0.010295. The
question of this script is not "which model is best" but "does any model beat that by
enough that this protocol can tell". Those are different questions and only the second one
has an answer at five folds.

**The threshold is fixed before the results, not after.** A difference below
`PRACTICAL_SIGNIFICANCE_THRESHOLD` - 0.02 in PR-AUC, about twice the measured fold spread -
is a difference this measurement cannot resolve. Every row of the output carries its
standard deviation and its verdict, because a comparison quoted without them invites being
settled later by whichever reading is convenient.

**Hyperparameters are untuned defaults, chosen to be defensible rather than good.** They
live in `models/estimators` with the reasoning for each. Tuning comes in `run_tuning.py`,
after the imbalance strategy is measured, because tuning a model before knowing how it
should treat the class imbalance tunes it around the wrong problem.

**The gradient booster is `HistGradientBoostingClassifier`, not LightGBM.** LightGBM was
intended and does not load on this machine: its wheel ships `lib_lightgbm.dll`, which needs
the Microsoft Visual C++ runtime, and only the .NET-bundled variants are installed here.
The substitute is the same family - histogram-based gradient boosting - from a library the
project already depends on. Installing the runtime and swapping the constructor back is a
one-line change in `models/estimators`.

Exit code 0 when every model was measured and recorded, 1 when the tracking server could
not be configured.
"""

import sys

from credit_copilot.config import settings
from credit_copilot.data.loader import load_dataset
from credit_copilot.models.estimators import (
    build_hist_gradient_boosting,
    build_logistic_regression,
    build_random_forest,
)
from credit_copilot.models.evaluation import (
    DEFAULT_N_SPLITS,
    PRACTICAL_SIGNIFICANCE_THRESHOLD,
    CrossValidationResult,
    compare_to_reference,
    evaluate_and_log,
    format_comparison_table,
    format_comparison_verdicts,
    format_fold_table,
    split_features_and_target,
)
from credit_copilot.models.metrics import DECISION_METRIC, METRIC_NAMES
from credit_copilot.models.tracking import (
    LEAKAGE_CHECK_TAG,
    MLflowConfigurationError,
    ensure_experiment,
)

COMPARISON_TAG_VALUE = "model-comparison"
"""Marks a run as a candidate model measured with untuned defaults."""

REFERENCE_LABEL = "logistic-l2-balanced"
"""The arm every other one is judged against."""

RULE = "=" * 100


def _candidates() -> dict[str, tuple[object, str]]:
    """The models to compare, in the order the report reads them.

    Returns:
        Label -> (estimator, one-line description).
    """
    return {
        REFERENCE_LABEL: (
            build_logistic_regression(),
            "the reference - linear, and the number to beat",
        ),
        "random-forest": (
            build_random_forest(),
            "bagged trees, 300 estimators, leaves of at least 20 rows",
        ),
        "hist-gradient-boosting": (
            build_hist_gradient_boosting(),
            "boosted trees, 300 rounds at learning rate 0.05 (stands in for LightGBM)",
        ),
    }


def main() -> int:
    """Measure every candidate on one partition, record them, and print the comparison.

    Returns:
        0 if every model was measured and recorded, 1 if MLflow could not be configured.
    """
    try:
        context = ensure_experiment()
    except MLflowConfigurationError as error:
        print(f"MLflow is not configured:\n{error}", file=sys.stderr)
        return 1

    print(RULE)
    print("MODEL COMPARISON - untuned defaults, one partition, one seed")
    print(RULE)
    print(f"Tracking server : {context.tracking_uri}")
    print(f"Experiment      : {context.name} (id {context.experiment_id})")

    frame = load_dataset()
    features, target = split_features_and_target(frame)
    print(f"Rows            : {len(frame):,}")
    print(f"Prevalence      : {target.mean():.6f}  <- the floor of PR-AUC")
    print(f"Validation      : StratifiedKFold({DEFAULT_N_SPLITS}, shuffle=True), ", end="")
    print(f"random_state={settings.random_state} from config.py")
    print("Preprocessing   : fitted inside each fold, on that fold's training rows only")
    print(f"Decision metric : {DECISION_METRIC}, per ADR-0002")
    print(
        f"Threshold       : {PRACTICAL_SIGNIFICANCE_THRESHOLD:.2f} in {DECISION_METRIC}, ", end=""
    )
    print("fixed before these results were computed")

    results: dict[str, CrossValidationResult] = {}
    run_ids: dict[str, str] = {}

    for label, (estimator, description) in _candidates().items():
        print(f"\n-> {label}: {description}")
        result, run_id = evaluate_and_log(
            estimator,
            features,
            target,
            run_name=label,
            context=context,
            tags={
                LEAKAGE_CHECK_TAG: COMPARISON_TAG_VALUE,
                "phase": "02-modeling",
                "tuned": "false",
                "imbalance_strategy": "class_weight=balanced",
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

    reference = results[REFERENCE_LABEL]
    comparisons = [
        compare_to_reference(label, result, reference)
        for label, result in results.items()
        if label != REFERENCE_LABEL
    ]

    print(f"\nAgainst {REFERENCE_LABEL}, on {DECISION_METRIC}")
    print("-" * 100)
    print(format_comparison_verdicts(comparisons, DECISION_METRIC))
    print(
        f"\nA difference is called a difference only at {PRACTICAL_SIGNIFICANCE_THRESHOLD:.2f} "
        f"or more in {DECISION_METRIC}. Below that, five folds cannot separate the model\n"
        "from the partition: the fold-to-fold spread is of the same size as the gap."
    )

    winner = max(results, key=lambda label: results[label].means[DECISION_METRIC])
    resolved = [item for item in comparisons if item.clears_threshold and item.difference > 0]
    print(f"\nHighest {DECISION_METRIC}: {winner} ({results[winner].means[DECISION_METRIC]:.4f})")
    if resolved:
        print(
            "Beats the reference by more than the threshold: "
            + ", ".join(i.label for i in resolved)
        )
    else:
        print(
            "No model beats the reference by more than the threshold. The highest score is\n"
            "still the highest score, but this protocol cannot call it a better model."
        )

    for label, result in results.items():
        print(f"\nPer fold - {label}")
        print("-" * 100)
        print(format_fold_table(result, METRIC_NAMES))

    print("\n" + RULE)
    print("MLFLOW")
    print(RULE)
    print(f"Experiment: {context.url}")
    for label, run_id in run_ids.items():
        print(f"    {label:<26} run_id {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
