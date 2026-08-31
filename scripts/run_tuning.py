"""Tune the winning model with Optuna, inside a nested cross-validation.

Run it with::

    uv run python scripts/run_tuning.py

**Why nested, and what goes wrong without it.** Tuning searches for the hyperparameters
that score best on a partition. Reporting that best score on the same partition reports the
maximum of thirty noisy draws as if it were the expected performance of one model - an
optimistic bias whose size depends on how many trials were run and on how noisy the metric
is, which is to say a bias nobody can bound afterwards. The nested design separates the two
jobs: an **inner** cross-validation on each outer training fold chooses the
hyperparameters, and the **outer** fold, which no trial ever saw, scores the result. What
the outer loop estimates is not "the best hyperparameters" but the honest performance of
*the whole tuning procedure*, which is the thing that would actually be deployed.

**Why there are two studies and only one of them is an estimate.** The nested loop produces
five sets of best hyperparameters, one per outer fold, and no single deployable model. A
final study over all the data produces the set to deploy. Its own cross-validated score is
reported too, and it is **not** an unbiased estimate - it is the number the nested loop
exists to correct. Both are printed, labelled, so the gap between them is visible instead
of being a thing to know.

**The imbalance strategy is fixed, and it was measured.** `run_imbalance_comparison.py`
found no treatment ahead of both alternatives: PR-AUC identical within noise, and Brier
better by 0.040 against `class_weight="balanced"` and 0.024 against SMOTE. The tuning
therefore searches over a forest with `class_weight=None`. Tuning a model around a
correction the data says it does not need would tune it around the wrong problem.

Exit code 0 when the study completed and was recorded, 1 when the tracking server could not
be configured.
"""

import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import mlflow
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from credit_copilot.config import settings
from credit_copilot.console import enable_unicode_console
from credit_copilot.data.loader import load_dataset
from credit_copilot.models.estimators import build_random_forest
from credit_copilot.models.evaluation import (
    DEFAULT_N_SPLITS,
    PRACTICAL_SIGNIFICANCE_THRESHOLD,
    Comparison,
    cross_validate_estimator,
    evaluate_and_log,
    fit_and_score,
    split_features_and_target,
)
from credit_copilot.models.metrics import DECISION_METRIC, METRIC_NAMES
from credit_copilot.models.tracking import (
    LEAKAGE_CHECK_TAG,
    MLflowConfigurationError,
    ensure_experiment,
)

TUNING_TAG_VALUE: Final[str] = "tuning"
"""Marks a run produced by the tuning study."""

N_TRIALS: Final[int] = 30
"""Optuna trials per study.

**Chosen against a measured cost, not picked round.** The search has four dimensions, and
the whole nested procedure costs `outer folds x trials x inner folds` model fits - at 30
trials that is 5 x 30 x 3 = 450 fits, plus the final study's 90, and a forest fit on this
data takes roughly two seconds. Thirty trials is enough for TPE to leave its random
start-up phase and exploit in four dimensions, and small enough that the whole thing runs
in the tens of minutes rather than overnight.

It is emphatically not enough to *exhaust* the space. That matters less than it sounds
here: the finding this script is set up to be able to report is that the gain does not
clear the 0.02 threshold, and a larger budget moves the best trial by less than the fold
noise once the metric has plateaued.
"""

INNER_N_SPLITS: Final[int] = 3
"""Folds of the inner cross-validation that scores each trial.

Three rather than five, and the trade is deliberate: the inner loop runs `trials` times per
outer fold, so its cost multiplies through the whole procedure. Three folds make each trial
noisier, which makes the *search* less precise - but the number this script reports comes
from the outer loop, which is unaffected, so the cost of the compromise falls on the
hyperparameters chosen and not on the honesty of the estimate.
"""

RULE: Final[str] = "=" * 100


def suggest_parameters(trial: optuna.Trial) -> dict[str, Any]:
    """Draw one hyperparameter configuration for the random forest.

    The space covers the four things that decide how a forest fits, and deliberately not
    more: a wider space with this trial budget would sample each dimension worse without
    reaching anything new.

    - `n_estimators` - how much averaging. More is never worse for a forest, only slower,
      so the floor is set where the variance has already largely settled.
    - `max_depth` and `min_samples_leaf` - the two ways a tree stops growing, and together
      the main control on overfitting. `min_samples_leaf` is drawn on a log scale because
      the difference between 1 and 5 matters far more than between 60 and 80.
    - `max_features` - how decorrelated the trees are, which is the mechanism that makes a
      forest better than one tree.

    Args:
        trial: The Optuna trial being sampled.

    Returns:
        Keyword arguments for `RandomForestClassifier`.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 500, step=100),
        "max_depth": trial.suggest_int("max_depth", 6, 24, step=2),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 80, log=True),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
    }


def build_tuned_forest(parameters: Mapping[str, Any]) -> RandomForestClassifier:
    """Build a forest from a sampled configuration, with everything else held fixed.

    `class_weight=None` is not a tuned choice: `run_imbalance_comparison.py` measured it as
    the best of the three strategies on both PR-AUC and Brier, and it is held fixed so that
    this study varies only the structural hyperparameters.

    Args:
        parameters: The sampled hyperparameters.

    Returns:
        An unfitted estimator.
    """
    return RandomForestClassifier(
        class_weight=None,
        n_jobs=-1,
        random_state=settings.random_state,
        **parameters,
    )


def run_study(
    features: pd.DataFrame,
    target: pd.Series,
    label: str,
) -> optuna.Study:
    """Search the hyperparameter space against an inner cross-validation.

    Args:
        features: Predictors the search is allowed to see. In the nested loop this is one
            outer *training* fold, never the whole dataset.
        target: Matching labels.
        label: Name of the study, for the log line.

    Returns:
        The completed study.
    """
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.random_state),
        study_name=label,
    )

    def objective(trial: optuna.Trial) -> float:
        estimator = build_tuned_forest(suggest_parameters(trial))
        result = cross_validate_estimator(estimator, features, target, n_splits=INNER_N_SPLITS)
        return result.means[DECISION_METRIC]

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study


def nested_cross_validate(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    """Run the full nested cross-validation: tune inside, score outside.

    Args:
        features: All predictors.
        target: All labels.

    Returns:
        The outer-fold metrics, the best parameters chosen in each outer fold, and every
        trial of every inner study.
    """
    splitter = StratifiedKFold(
        n_splits=DEFAULT_N_SPLITS, shuffle=True, random_state=settings.random_state
    )
    rows: list[Mapping[str, float]] = []
    chosen: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []

    for fold, (train_index, validation_index) in enumerate(
        splitter.split(features, target), start=1
    ):
        train_features = features.iloc[train_index]
        train_target = target.iloc[train_index]

        study = run_study(train_features, train_target, f"outer-fold-{fold}")
        best = dict(study.best_params)
        chosen.append({"fold": fold, "inner_best_pr_auc": study.best_value, **best})

        for trial in study.trials:
            trials.append(
                {
                    "outer_fold": fold,
                    "trial": trial.number,
                    "inner_pr_auc": trial.value,
                    **trial.params,
                }
            )

        metrics = fit_and_score(
            build_tuned_forest(best),
            train_features,
            train_target,
            features.iloc[validation_index],
            target.iloc[validation_index],
        )
        rows.append(metrics)
        print(
            f"   outer fold {fold}: inner best {study.best_value:.4f} -> "
            f"outer {DECISION_METRIC} {metrics[DECISION_METRIC]:.4f}   {best}",
            flush=True,
        )

    outer = pd.DataFrame(rows)
    outer.index = pd.Index(range(1, DEFAULT_N_SPLITS + 1), name="fold")
    return outer, chosen, pd.DataFrame(trials)


def main() -> int:
    """Tune the forest, estimate the procedure honestly, and record everything.

    Returns:
        0 if the study completed and was recorded, 1 if MLflow could not be configured.
    """
    enable_unicode_console()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        context = ensure_experiment()
    except MLflowConfigurationError as error:
        print(f"MLflow is not configured:\n{error}", file=sys.stderr)
        return 1

    print(RULE)
    print("HYPERPARAMETER TUNING - Optuna inside a nested cross-validation")
    print(RULE)
    print(f"Tracking server : {context.tracking_uri}")
    print(f"Experiment      : {context.name} (id {context.experiment_id})")

    frame = load_dataset()
    features, target = split_features_and_target(frame)
    print(f"Rows            : {len(frame):,}")
    print("Model           : RandomForestClassifier, class_weight=None (measured best)")
    print(f"Outer loop      : StratifiedKFold({DEFAULT_N_SPLITS}, shuffle=True) - scores only")
    print(f"Inner loop      : StratifiedKFold({INNER_N_SPLITS}, shuffle=True) - chooses only")
    print(f"Trials          : {N_TRIALS} per study, TPE, ", end="")
    print(f"seeded from config.py ({settings.random_state})")
    print(f"Model fits       : {DEFAULT_N_SPLITS * N_TRIALS * INNER_N_SPLITS} nested, ", end="")
    print(f"plus {N_TRIALS * INNER_N_SPLITS} for the final study")
    print("Search space    : n_estimators, max_depth, min_samples_leaf, max_features")

    print(
        f"\n-> untuned reference: build_random_forest(class_weight=None), {DEFAULT_N_SPLITS} folds"
    )
    untuned = cross_validate_estimator(build_random_forest(class_weight=None), features, target)
    print(
        f"   {DECISION_METRIC} = {untuned.means[DECISION_METRIC]:.4f} "
        f"± {untuned.stds[DECISION_METRIC]:.4f}"
    )

    print(f"\n-> nested cross-validation ({DEFAULT_N_SPLITS} outer folds, a study inside each)")
    outer, chosen, trials = nested_cross_validate(features, target)

    nested_mean = float(outer[DECISION_METRIC].mean())
    nested_std = float(outer[DECISION_METRIC].std(ddof=1))

    print(f"\n-> final study on all {len(frame):,} rows, for the deployable hyperparameters")
    final_study = run_study(features, target, "final")
    best_parameters = dict(final_study.best_params)
    print(f"   best inner {DECISION_METRIC}: {final_study.best_value:.4f}")
    print(f"   best parameters: {best_parameters}")

    print("\n-> the tuned model, scored the ordinary (optimistic) way for comparison")
    tuned, tuned_run_id = evaluate_and_log(
        build_tuned_forest(best_parameters),
        features,
        target,
        run_name="tuned-random-forest",
        context=context,
        tags={
            LEAKAGE_CHECK_TAG: TUNING_TAG_VALUE,
            "phase": "02-modeling",
            "tuned": "true",
            "imbalance_strategy": "no-treatment",
            "estimate_is_unbiased": "false",
            "note": "hyperparameters chosen on this same partition; see the nested estimate",
        },
    )
    print(
        f"   {DECISION_METRIC} = {tuned.means[DECISION_METRIC]:.4f} "
        f"± {tuned.stds[DECISION_METRIC]:.4f}   run {tuned_run_id}"
    )

    with mlflow.start_run(experiment_id=context.experiment_id, run_name="tuning-nested-cv") as run:
        mlflow.set_tags(
            {
                LEAKAGE_CHECK_TAG: TUNING_TAG_VALUE,
                "phase": "02-modeling",
                "tuned": "true",
                "estimate_is_unbiased": "true",
                "note": "nested CV: the honest estimate of the tuning procedure",
            }
        )
        mlflow.log_params(
            {
                "n_trials": str(N_TRIALS),
                "outer_n_splits": str(DEFAULT_N_SPLITS),
                "inner_n_splits": str(INNER_N_SPLITS),
                "sampler": "TPESampler",
                "random_state": str(settings.random_state),
                "search_space": "n_estimators, max_depth, min_samples_leaf, max_features",
                **{f"final_best__{k}": str(v) for k, v in best_parameters.items()},
            }
        )
        mlflow.log_metrics(
            {
                **{f"{name}_mean": float(outer[name].mean()) for name in outer.columns},
                **{f"{name}_std": float(outer[name].std(ddof=1)) for name in outer.columns},
                "final_study_inner_pr_auc": float(final_study.best_value),
            }
        )
        with tempfile.TemporaryDirectory() as staging:
            directory = Path(staging)
            outer.to_csv(directory / "nested_outer_fold_metrics.csv")
            pd.DataFrame(chosen).to_csv(directory / "per_outer_fold_best_params.csv", index=False)
            trials.to_csv(directory / "all_trials.csv", index=False)
            final_study.trials_dataframe().to_csv(directory / "final_study_trials.csv", index=False)
            for name in (
                "nested_outer_fold_metrics.csv",
                "per_outer_fold_best_params.csv",
                "all_trials.csv",
                "final_study_trials.csv",
            ):
                mlflow.log_artifact(str(directory / name))
        nested_run_id = run.info.run_id

    print("\n" + RULE)
    print("RESULT")
    print(RULE)
    print(f"{'estimate':<44}{DECISION_METRIC:>12}{'std':>10}   what it means")
    print("-" * 100)
    print(
        f"{'untuned forest, 5-fold':<44}{untuned.means[DECISION_METRIC]:>12.4f}"
        f"{untuned.stds[DECISION_METRIC]:>10.4f}   the reference"
    )
    print(
        f"{'tuned forest, nested CV':<44}{nested_mean:>12.4f}{nested_std:>10.4f}"
        "   unbiased: what tuning is worth"
    )
    print(
        f"{'tuned forest, ordinary 5-fold':<44}{tuned.means[DECISION_METRIC]:>12.4f}"
        f"{tuned.stds[DECISION_METRIC]:>10.4f}   OPTIMISTIC - same partition as the search"
    )

    honest = Comparison(
        label="tuning (nested estimate)",
        value=nested_mean,
        reference_value=untuned.means[DECISION_METRIC],
        difference=nested_mean - untuned.means[DECISION_METRIC],
        spread=nested_std,
        threshold=PRACTICAL_SIGNIFICANCE_THRESHOLD,
        clears_threshold=abs(nested_mean - untuned.means[DECISION_METRIC])
        >= PRACTICAL_SIGNIFICANCE_THRESHOLD,
    )
    optimism = tuned.means[DECISION_METRIC] - nested_mean

    print(f"\nWhat tuning bought, judged at the {PRACTICAL_SIGNIFICANCE_THRESHOLD:.2f} threshold")
    print("-" * 100)
    print(
        f"  nested minus untuned : {honest.difference:+.4f}   "
        f"(fold spread {honest.spread:.4f})   {honest.verdict}"
    )
    print(
        f"  optimism of the naive estimate: {optimism:+.4f}   "
        "= ordinary 5-fold minus nested, the bias nesting removes"
    )

    print("\nOuter-fold metrics (the honest estimate, fold by fold)")
    print("-" * 100)
    summary = outer[list(METRIC_NAMES)].copy()
    summary.loc["mean"] = summary.mean()
    summary.loc["std"] = outer[list(METRIC_NAMES)].std(ddof=1)
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))

    print("\nHyperparameters chosen in each outer fold")
    print("-" * 100)
    print(pd.DataFrame(chosen).to_string(index=False))
    spread = {key: sorted({row[key] for row in chosen}) for key in best_parameters}
    print(f"\nDistinct values chosen across the five folds: {spread}")
    print(
        "A parameter the five folds disagree about is a parameter the data does not\n"
        "constrain: the search is picking between configurations the metric cannot separate."
    )

    print("\n" + RULE)
    print("MLFLOW")
    print(RULE)
    print(f"Experiment: {context.url}")
    print(f"    tuning-nested-cv       run_id {nested_run_id}  (the honest estimate)")
    print(f"    tuned-random-forest    run_id {tuned_run_id}  (optimistic, tagged as such)")
    print(f"    final best parameters: {best_parameters}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
