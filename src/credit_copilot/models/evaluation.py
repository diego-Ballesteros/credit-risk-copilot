"""Cross-validation with the preprocessor inside it. The routine every measurement uses.

**The one thing this module exists to make impossible.** `data/processed/features.parquet`
was produced by fitting the pipeline on all 30,000 rows. Training or evaluating on it would
mean every validation row was scaled, clipped and imputed with statistics that had already
seen that row, and the resulting metric would be optimistic by an amount nobody can bound
afterwards. This module never reads that file. It takes the canonical table, and inside
each fold it builds a *fresh* preprocessor and fits it on that fold's training part alone.

**Why the preprocessor arrives as a factory and not as an object.** Handing in a built
`Pipeline` would invite reusing one instance across folds, and a reused instance is a
fitted instance: fold 2 would be transformed with fold 1's medians. A factory cannot be
misused that way, because each fold calls it and gets an object nobody has fitted. The same
seam is what lets `tests/test_evaluation.py` insert a counter and prove that the fit happens
once per fold, on the training rows only - the claim in this paragraph is checked, not
asserted.

**Why the metric functions are not in this module.** `metrics.py` owns the seven numbers of
ADR-0002 and this module owns how the data is split. Keeping them apart means a change to
the validation protocol cannot quietly change what a metric means.

**What reaches MLflow, and what does not.** `evaluate_and_log` records the model's
parameters, every metric with its mean and standard deviation across folds, the number of
folds, and the per-fold table as an artefact. Credentials are read by `tracking.py` into the
process environment and never become a parameter, a tag or a line of output.
"""

import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from credit_copilot.config import settings
from credit_copilot.data import schema
from credit_copilot.data.preprocessor import build_preprocessor
from credit_copilot.models.metrics import REPORTED_METRIC_NAMES, compute_metrics
from credit_copilot.models.tracking import ExperimentContext

DEFAULT_N_SPLITS: Final[int] = 5
"""Folds used by every measurement in the project, so results stay comparable."""

POSITIVE_LABEL: Final[int] = 1
"""The class whose probability is scored: the client defaults next month."""

PREPROCESSOR_STEP: Final[str] = "preprocess"
"""Name of the preprocessing step inside the per-fold pipeline."""

MODEL_STEP: Final[str] = "model"
"""Name of the estimator step inside the per-fold pipeline."""

FOLD_METRICS_ARTEFACT: Final[str] = "fold_metrics.csv"
"""Filename of the per-fold table attached to every run."""

PreprocessorFactory = Callable[[], Pipeline]
"""Builds an unfitted preprocessing pipeline. Called once per fold, never reused."""


class EvaluationInputError(ValueError):
    """The data handed to the cross-validation cannot support the protocol asked for."""


@dataclass(frozen=True)
class CrossValidationResult:
    """Everything one cross-validated measurement produced, per fold and summarised.

    The per-fold table is kept rather than only its summary, because a mean of five folds
    hides the case that matters most: four folds agreeing and one disagreeing loudly. The
    standard deviation is what makes a difference between two models readable as a real
    difference or as noise.

    Attributes:
        estimator_name: Class name of the estimator that was evaluated.
        n_splits: Number of folds.
        n_samples: Rows in the full dataset the folds were cut from.
        positive_rate: Share of the positive class over the full dataset. This is the floor
            of PR-AUC and of both precision-at-top metrics, so no value in `fold_metrics`
            can be read without it.
        random_state: Seed handed to the splitter.
        fold_metrics: One row per fold, one column per name in `REPORTED_METRIC_NAMES`,
            indexed by a 1-based fold number.
        fold_positive_rates: Share of the positive class in each validation fold. Present so
            that "the split was stratified" is a number the reader can check.
    """

    estimator_name: str
    n_splits: int
    n_samples: int
    positive_rate: float
    random_state: int
    fold_metrics: pd.DataFrame
    fold_positive_rates: tuple[float, ...]

    @property
    def means(self) -> Mapping[str, float]:
        """Mean of each metric across folds.

        Returns:
            Metric name -> mean, in `REPORTED_METRIC_NAMES` order.
        """
        return {name: float(self.fold_metrics[name].mean()) for name in self.fold_metrics.columns}

    @property
    def stds(self) -> Mapping[str, float]:
        """Sample standard deviation of each metric across folds.

        Uses `ddof=1`: the five folds are a sample of the possible splits, not the
        population of them.

        Returns:
            Metric name -> standard deviation, in `REPORTED_METRIC_NAMES` order.
        """
        return {
            name: float(self.fold_metrics[name].std(ddof=1)) for name in self.fold_metrics.columns
        }

    def summary_frame(self) -> pd.DataFrame:
        """Per-fold table with the mean and standard deviation appended as rows.

        Returns:
            A copy of `fold_metrics` with two extra rows, `mean` and `std`. It is a copy:
            the fold table is the record of the measurement and is not modified in place.
        """
        summary = self.fold_metrics.copy()
        summary.loc["mean"] = pd.Series(self.means)
        summary.loc["std"] = pd.Series(self.stds)
        return summary


def split_features_and_target(
    frame: pd.DataFrame,
    target_column: str = schema.TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the canonical table into predictors and label.

    The preprocessor's `ColumnTransformer` already drops the target by not selecting it, so
    removing the column here is a second, independent guarantee rather than the only one.
    Two mechanisms have to fail together for the label to reach the matrix.

    Args:
        frame: Canonical table as `loader.load_dataset` returns it.
        target_column: Name of the label column.

    Returns:
        The predictors without the label, and the label.

    Raises:
        EvaluationInputError: If the label column is not in the frame.
    """
    if target_column not in frame.columns:
        raise EvaluationInputError(
            f"The frame has no column {target_column!r}. Load it through "
            "loader.load_dataset, which applies the canonical names."
        )
    return frame.drop(columns=[target_column]), frame[target_column]


def _positive_class_column(classes: Sequence[Any]) -> int:
    """Locate the column of `predict_proba` that holds the positive class.

    Reading column 1 by position is right for almost every estimator and wrong in exactly
    the case that is hardest to notice, because the numbers stay in [0, 1] and the metrics
    stay plausible - they simply describe the wrong class. The column is therefore looked
    up by label.

    Args:
        classes: The estimator's `classes_`, in `predict_proba` column order.

    Returns:
        Index of the column for `POSITIVE_LABEL`.

    Raises:
        EvaluationInputError: If the estimator never saw the positive class.
    """
    labels = [int(label) for label in classes]
    if POSITIVE_LABEL not in labels:
        raise EvaluationInputError(
            f"The estimator was fitted without the positive class {POSITIVE_LABEL}; "
            f"it knows {labels}. A fold with one class means the split is not stratified."
        )
    return labels.index(POSITIVE_LABEL)


def build_fold_pipeline(
    estimator: BaseEstimator,
    preprocessor_factory: PreprocessorFactory = build_preprocessor,
) -> Pipeline:
    """Assemble the full pipeline for one fold: fresh preprocessor, then a clone of the model.

    Both halves are new objects. `clone` copies the estimator's parameters and discards
    anything it learned, so a caller can pass the same configured estimator to several
    evaluations without one run's fitted state leaking into another.

    Args:
        estimator: A scikit-learn classifier exposing `predict_proba`.
        preprocessor_factory: Builds the unfitted preprocessing pipeline.

    Returns:
        An unfitted `Pipeline` of two steps, `preprocess` then `model`.
    """
    return Pipeline(
        [
            (PREPROCESSOR_STEP, preprocessor_factory()),
            (MODEL_STEP, clone(estimator)),
        ]
    )


def cross_validate_estimator(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int | None = None,
    preprocessor_factory: PreprocessorFactory = build_preprocessor,
) -> CrossValidationResult:
    """Run stratified k-fold cross-validation with the preprocessor fitted inside each fold.

    The protocol, fixed for the whole project so that two measurements are comparable:
    `StratifiedKFold` with shuffling on and the seed from `config.py`. Stratification keeps
    the 22% positive rate steady in every fold, which matters more here than usual because
    PR-AUC's floor *is* that rate - a fold with a different prevalence would be scored
    against a different floor.

    For each fold, in order: build a new preprocessor, clone the estimator, fit the pair on
    the training rows only, then score the held-out rows. Nothing computed on a validation
    row ever reaches the object that transforms it.

    Args:
        estimator: A scikit-learn classifier exposing `predict_proba`.
        features: Predictors, without the label. See `split_features_and_target`.
        target: Binary label sharing the index of `features`.
        n_splits: Number of folds.
        random_state: Seed for the splitter. Defaults to `settings.random_state`.
        preprocessor_factory: Builds the unfitted preprocessor for each fold.

    Returns:
        Per-fold metrics plus everything needed to read them: prevalence, fold count, fold
        sizes and seed.

    Raises:
        EvaluationInputError: If the estimator has no `predict_proba`, if `features` and
            `target` disagree in length, or if `n_splits` is below 2.
    """
    if not hasattr(estimator, "predict_proba"):
        raise EvaluationInputError(
            f"{type(estimator).__name__} has no predict_proba. Six of the seven metrics of "
            "ADR-0002 are computed on a score, not on a label."
        )
    if len(features) != len(target):
        raise EvaluationInputError(
            f"features has {len(features)} rows and target has {len(target)}."
        )
    if n_splits < 2:
        raise EvaluationInputError(f"n_splits must be at least 2; got {n_splits}.")

    seed = settings.random_state if random_state is None else random_state
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows: list[Mapping[str, float]] = []
    fold_positive_rates: list[float] = []

    for train_index, validation_index in splitter.split(features, target):
        train_features = features.iloc[train_index]
        train_target = target.iloc[train_index]
        validation_features = features.iloc[validation_index]
        validation_target = target.iloc[validation_index]

        pipeline = build_fold_pipeline(estimator, preprocessor_factory)
        pipeline.fit(train_features, train_target)

        column = _positive_class_column(pipeline.named_steps[MODEL_STEP].classes_)
        scores = pipeline.predict_proba(validation_features)[:, column]

        rows.append(compute_metrics(validation_target.to_numpy(), scores))
        fold_positive_rates.append(float(validation_target.mean()))

    fold_metrics = pd.DataFrame(rows, columns=list(REPORTED_METRIC_NAMES))
    fold_metrics.index = pd.Index(range(1, n_splits + 1), name="fold")

    return CrossValidationResult(
        estimator_name=type(estimator).__name__,
        n_splits=n_splits,
        n_samples=len(target),
        positive_rate=float(target.mean()),
        random_state=seed,
        fold_metrics=fold_metrics,
        fold_positive_rates=tuple(fold_positive_rates),
    )


def _loggable_params(estimator: BaseEstimator) -> Mapping[str, str]:
    """Flatten an estimator's parameters into strings MLflow will accept.

    Every value is stringified rather than filtered by type: a parameter MLflow could not
    serialise would otherwise vanish from the run, and a run that silently records fewer
    parameters than the model has is not a reproducible record of it.

    Args:
        estimator: The estimator whose configuration is being recorded.

    Returns:
        Prefixed parameter name -> value as text.
    """
    return {f"model__{name}": str(value) for name, value in sorted(estimator.get_params().items())}


def evaluate_and_log(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    run_name: str,
    context: ExperimentContext,
    tags: Mapping[str, str] | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int | None = None,
    preprocessor_factory: PreprocessorFactory = build_preprocessor,
) -> tuple[CrossValidationResult, str]:
    """Cross-validate an estimator and record the whole measurement as one MLflow run.

    Recorded as parameters: the estimator's full configuration, the validation protocol and
    the seed. As metrics: every reported metric's mean and standard deviation across folds,
    plus the prevalence, so a reader of the run alone still has PR-AUC's floor. As an
    artefact: the per-fold table, because a mean without its folds cannot be audited.

    Args:
        estimator: A scikit-learn classifier exposing `predict_proba`.
        features: Predictors, without the label.
        target: Binary label sharing the index of `features`.
        run_name: Name of the run, as it appears in the MLflow interface.
        context: Experiment the run belongs to, from `tracking.ensure_experiment`.
        tags: Extra tags. Used to mark a run as a baseline or as a diagnostic.
        n_splits: Number of folds.
        random_state: Seed for the splitter. Defaults to `settings.random_state`.
        preprocessor_factory: Builds the unfitted preprocessor for each fold.

    Returns:
        The measurement, and the identifier of the run that now holds it.
    """
    result = cross_validate_estimator(
        estimator,
        features,
        target,
        n_splits=n_splits,
        random_state=random_state,
        preprocessor_factory=preprocessor_factory,
    )

    with mlflow.start_run(experiment_id=context.experiment_id, run_name=run_name) as run:
        mlflow.set_tags(dict(tags or {}))
        mlflow.log_params(
            {
                **_loggable_params(estimator),
                "estimator": result.estimator_name,
                "cv_strategy": "StratifiedKFold(shuffle=True)",
                "cv_n_splits": str(result.n_splits),
                "random_state": str(result.random_state),
                "n_samples": str(result.n_samples),
                "n_raw_columns": str(features.shape[1]),
                "preprocessor_fitted": "once per fold, on the training part only",
            }
        )

        means, stds = result.means, result.stds
        mlflow.log_metrics(
            {
                **{f"{name}_mean": means[name] for name in REPORTED_METRIC_NAMES},
                **{f"{name}_std": stds[name] for name in REPORTED_METRIC_NAMES},
                "positive_rate": result.positive_rate,
            }
        )

        with tempfile.TemporaryDirectory() as staging:
            artefact = Path(staging) / FOLD_METRICS_ARTEFACT
            result.summary_frame().to_csv(artefact)
            mlflow.log_artifact(str(artefact))

        return result, run.info.run_id


def format_comparison_table(
    results: Mapping[str, CrossValidationResult],
    metric_names: Sequence[str] = REPORTED_METRIC_NAMES,
) -> str:
    """Render several measurements side by side as `mean ± std`, one column per model.

    Side by side and not one table each, because the methodology's rule is that a metric is
    never reported without its baseline: a layout that puts them in separate tables makes it
    possible to quote one without the other.

    Args:
        results: Column label -> measurement. Insertion order is column order.
        metric_names: Metrics to show, in row order.

    Returns:
        A plain-text table, without a trailing newline.
    """
    labels = list(results)
    width = max((len(label) for label in labels), default=0)
    column_width = max(width, 17)
    header = "metric".ljust(24) + "".join(label.rjust(column_width + 2) for label in labels)
    lines = [header, "-" * len(header)]

    for name in metric_names:
        cells = []
        for label in labels:
            result = results[label]
            cells.append(
                f"{result.means[name]:.4f} ± {result.stds[name]:.4f}".rjust(column_width + 2)
            )
        lines.append(name.ljust(24) + "".join(cells))

    return "\n".join(lines)


def format_fold_table(result: CrossValidationResult, metric_names: Sequence[str]) -> str:
    """Render one measurement's per-fold values, so a mean can be audited against its folds.

    Args:
        result: The measurement.
        metric_names: Metrics to show, in column order.

    Returns:
        A plain-text table, without a trailing newline.
    """
    frame = result.summary_frame()[list(metric_names)]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        return str(frame.round(6).to_string(float_format=lambda value: f"{value:.6f}"))


def null_reference(result: CrossValidationResult) -> Mapping[str, float]:
    """The value each metric takes when the scores carry no information about the label.

    This is the floor every number in `result` is read against, and it is not the same
    constant for every metric: ranking metrics that use precision sit at the prevalence,
    ROC-AUC sits at 0.5 whatever the balance, and a Brier score from a constant prediction
    of the prevalence is `p(1-p)`.

    Args:
        result: A measurement, read only for its prevalence and nothing else.

    Returns:
        Metric name -> its value under a ranking that carries no signal.
    """
    prevalence = result.positive_rate
    return {
        "pr_auc": prevalence,
        "roc_auc": 0.5,
        "ks": 0.0,
        "gini": 0.0,
        "brier": float(prevalence * (1.0 - prevalence)),
        "precision_at_top_10pct": prevalence,
        "precision_at_top_5pct": prevalence,
        "accuracy": float(np.max([prevalence, 1.0 - prevalence])),
    }
