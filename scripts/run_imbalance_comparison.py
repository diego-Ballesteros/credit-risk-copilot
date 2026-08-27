"""Measure what each imbalance strategy does, to ranking and to calibration.

Run it with::

    uv run python scripts/run_imbalance_comparison.py

**Why this is measured and not assumed.** Section 6.2 of `docs/METHODOLOGY.md` uses this
exact case as its example of the difference between a result and an opinion: "SMOTE should
help" is not a finding. The three arms differ in one thing - how the 22% minority is
handled during training - and share the model, its hyperparameters, the partition and the
seed.

**Why Brier score is reported next to PR-AUC and not as an afterthought.** Resampling and
reweighting both change the *proportion* of positives the model is fitted on, so both move
the predicted probabilities away from the rate the real population has. Ranking can survive
that untouched while calibration does not, and ADR-0002 keeps Brier precisely because the
business use - expected loss, and therefore pricing - needs a probability rather than an
order. A strategy that improves PR-AUC and wrecks Brier has not obviously helped.

**How SMOTE is prevented from touching a validation fold.** The sampler sits inside an
`imblearn.pipeline.Pipeline`, which calls `fit_resample` during `fit` and skips samplers
entirely during `predict`. That pipeline is handed to `cross_validate_estimator` as the
estimator, so the ordinary preprocessor-inside-the-fold machinery is unchanged and the
sampler runs strictly after it, on the training part only. This was verified rather than
assumed: a spy subclass recorded five `fit_resample` calls, each on exactly 24,000 rows -
the training fold - and never on 30,000.

**One honest caveat about SMOTE on this matrix.** SMOTE interpolates between neighbouring
rows, and 74 of the 110 columns here are one-hot indicators. Interpolating those produces
fractional values such as 0.37 for a column that means "this client's education level is
university", which corresponds to no client that could exist. `SMOTENC` is the variant
built for mixed data; it is not used here because the spec asked for SMOTE, and the effect
is visible in the result rather than hidden.

Exit code 0 when every strategy was measured and recorded, 1 when the tracking server could
not be configured.
"""

import sys

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancePipeline

from credit_copilot.config import settings
from credit_copilot.data.loader import load_dataset
from credit_copilot.models.estimators import build_random_forest
from credit_copilot.models.evaluation import (
    DEFAULT_N_SPLITS,
    PRACTICAL_SIGNIFICANCE_THRESHOLD,
    CrossValidationResult,
    compare_to_reference,
    evaluate_and_log,
    format_comparison_table,
    format_comparison_verdicts,
    split_features_and_target,
)
from credit_copilot.models.metrics import DECISION_METRIC, METRIC_NAMES
from credit_copilot.models.tracking import (
    LEAKAGE_CHECK_TAG,
    MLflowConfigurationError,
    ensure_experiment,
)

IMBALANCE_TAG_VALUE = "imbalance-comparison"
"""Marks a run as one arm of the imbalance-strategy comparison."""

REFERENCE_LABEL = "no-treatment"
"""The arm the other two are judged against: the model left alone."""

CALIBRATION_METRIC = "brier"
"""The metric that says whether the probabilities still mean anything. Lower is better."""

RULE = "=" * 100


def _strategies() -> dict[str, tuple[object, str]]:
    """The three imbalance strategies, in the order the report reads them.

    The model is the same in all three - `build_random_forest`, the winner of the model
    comparison - and only the treatment of the class imbalance changes. The SMOTE arm
    builds the forest with `class_weight=None` on purpose: resampling and reweighting are
    two ways of doing the same thing, and applying both would measure neither.

    Returns:
        Label -> (estimator, one-line description).
    """
    return {
        REFERENCE_LABEL: (
            build_random_forest(class_weight=None),
            "the model left alone - 22% positives, no correction",
        ),
        "class-weight-balanced": (
            build_random_forest(class_weight="balanced"),
            "reweight the loss by inverse class frequency - no rows invented",
        ),
        "smote": (
            ImbalancePipeline(
                [
                    ("smote", SMOTE(random_state=settings.random_state)),
                    ("model", build_random_forest(class_weight=None)),
                ]
            ),
            "synthesise minority rows until the training fold is 50/50",
        ),
    }


def _print_calibration(results: dict[str, CrossValidationResult]) -> None:
    """Print ranking and calibration side by side, which is the point of this script.

    Args:
        results: The three arms, keyed by label.
    """
    reference = results[REFERENCE_LABEL]
    print("\nRanking against calibration")
    print("-" * 100)
    print(
        f"{'strategy':<26}{DECISION_METRIC:>12}{'std':>10}{'vs ref':>11}"
        f"{CALIBRATION_METRIC:>12}{'std':>10}{'vs ref':>11}{'  calibration':<16}"
    )
    for label, result in results.items():
        pr_delta = result.means[DECISION_METRIC] - reference.means[DECISION_METRIC]
        brier_delta = result.means[CALIBRATION_METRIC] - reference.means[CALIBRATION_METRIC]
        if label == REFERENCE_LABEL:
            note = "reference"
        elif brier_delta > 0:
            note = "worse (higher)"
        else:
            note = "better (lower)"
        print(
            f"{label:<26}{result.means[DECISION_METRIC]:>12.4f}"
            f"{result.stds[DECISION_METRIC]:>10.4f}{pr_delta:>+11.4f}"
            f"{result.means[CALIBRATION_METRIC]:>12.4f}"
            f"{result.stds[CALIBRATION_METRIC]:>10.4f}{brier_delta:>+11.4f}  {note:<16}"
        )
    print(
        "\nBrier is the only reported metric where lower is better. A strategy that lifts\n"
        "PR-AUC while raising Brier has bought ordering with probabilities that no longer\n"
        "match the population - and expected loss is computed from the probabilities."
    )


def main() -> int:
    """Measure the three strategies on one partition, record them, and print the result.

    Returns:
        0 if every strategy was measured and recorded, 1 if MLflow could not be configured.
    """
    try:
        context = ensure_experiment()
    except MLflowConfigurationError as error:
        print(f"MLflow is not configured:\n{error}", file=sys.stderr)
        return 1

    print(RULE)
    print("IMBALANCE STRATEGIES - measured on the model that won the comparison")
    print(RULE)
    print(f"Tracking server : {context.tracking_uri}")
    print(f"Experiment      : {context.name} (id {context.experiment_id})")

    frame = load_dataset()
    features, target = split_features_and_target(frame)
    print(f"Rows            : {len(frame):,}")
    print(f"Prevalence      : {target.mean():.6f}  <- what every strategy is correcting for")
    print("Model           : build_random_forest(), identical in all three arms")
    print(f"Validation      : StratifiedKFold({DEFAULT_N_SPLITS}, shuffle=True), ", end="")
    print(f"random_state={settings.random_state} from config.py")
    print("SMOTE           : inside an imblearn Pipeline, so it resamples training folds only")

    results: dict[str, CrossValidationResult] = {}
    run_ids: dict[str, str] = {}

    for label, (estimator, description) in _strategies().items():
        print(f"\n-> {label}: {description}")
        result, run_id = evaluate_and_log(
            estimator,
            features,
            target,
            run_name=f"imbalance-{label}",
            context=context,
            tags={
                LEAKAGE_CHECK_TAG: IMBALANCE_TAG_VALUE,
                "phase": "02-modeling",
                "tuned": "false",
                "imbalance_strategy": label,
                "base_model": "RandomForestClassifier",
            },
        )
        results[label] = result
        run_ids[label] = run_id
        print(
            f"   {DECISION_METRIC} = {result.means[DECISION_METRIC]:.4f} "
            f"± {result.stds[DECISION_METRIC]:.4f}   "
            f"{CALIBRATION_METRIC} = {result.means[CALIBRATION_METRIC]:.4f}   run {run_id}"
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

    _print_calibration(results)

    resolved = [item for item in comparisons if item.clears_threshold]
    print(f"\nVerdict at the {PRACTICAL_SIGNIFICANCE_THRESHOLD:.2f} threshold")
    print("-" * 100)
    if resolved:
        for item in resolved:
            direction = "helps" if item.difference > 0 else "hurts"
            print(f"  {item.label}: {direction} by {item.difference:+.4f} in {DECISION_METRIC}")
    else:
        print(
            "  No strategy moves PR-AUC by more than the threshold. On this dataset the\n"
            "  imbalance treatment is not what decides the ranking quality - which is a\n"
            "  result, and the reason it was measured instead of assumed."
        )

    print("\n" + RULE)
    print("MLFLOW")
    print(RULE)
    print(f"Experiment: {context.url}")
    for label, run_id in run_ids.items():
        print(f"    {label:<26} run_id {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
