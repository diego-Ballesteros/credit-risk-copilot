"""Modelling subpackage: how a result is measured, and where it is recorded.

Three responsibilities, kept apart on purpose. `metrics` owns *what* a number means and is
the only implementation of the seven metrics ADR-0002 fixes. `evaluation` owns *how* the
data is split, and its whole reason for existing is that the preprocessor is fitted inside
each fold rather than before the split. `tracking` owns *where* the run is written, and is
the only place a credential moves.

Nothing here trains a production model or reads `data/processed/features.parquet`; that
matrix was fitted on the full dataset and evaluating on it would be the leakage this
subpackage is built to prevent.
"""

from credit_copilot.models.evaluation import (
    CrossValidationResult,
    EvaluationInputError,
    build_fold_pipeline,
    cross_validate_estimator,
    evaluate_and_log,
    format_comparison_table,
    format_fold_table,
    null_reference,
    split_features_and_target,
)
from credit_copilot.models.metrics import (
    DECISION_METRIC,
    METRIC_NAMES,
    REPORTED_METRIC_NAMES,
    MetricInputError,
    accuracy_at_threshold,
    brier_score,
    compute_metrics,
    gini,
    ks_statistic,
    pr_auc,
    precision_at_top_percent,
    roc_auc,
)
from credit_copilot.models.tracking import (
    ExperimentContext,
    MLflowConfigurationError,
    configure_mlflow,
    ensure_experiment,
)

__all__ = [
    "DECISION_METRIC",
    "METRIC_NAMES",
    "REPORTED_METRIC_NAMES",
    "CrossValidationResult",
    "EvaluationInputError",
    "ExperimentContext",
    "MLflowConfigurationError",
    "MetricInputError",
    "accuracy_at_threshold",
    "brier_score",
    "build_fold_pipeline",
    "compute_metrics",
    "configure_mlflow",
    "cross_validate_estimator",
    "ensure_experiment",
    "evaluate_and_log",
    "format_comparison_table",
    "format_fold_table",
    "gini",
    "ks_statistic",
    "null_reference",
    "pr_auc",
    "precision_at_top_percent",
    "roc_auc",
    "split_features_and_target",
]
