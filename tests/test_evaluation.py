"""Tests of the cross-validation routine.

Three of these check properties that have no visible symptom when they break, which is why
they are the ones the prompt asked for by name:

- `test_the_preprocessor_is_fitted_once_per_fold_and_never_before` is the **leakage** test
  at the level of the routine. A preprocessor fitted on all 30,000 rows, or fitted once and
  reused across folds, produces metrics that are simply better - nothing raises, nothing
  looks wrong. The recording preprocessor makes each `fit` leave a trace, so "once per fold,
  on the training part only" becomes a count and a row number rather than a claim.
- `test_every_validation_fold_carries_the_stratified_share_of_positives` matters more here
  than in an ordinary project, because PR-AUC's floor *is* the prevalence: a fold with a
  different positive rate is scored against a different floor, and averaging five of those
  averages five incomparable numbers.
- `test_the_same_seed_produces_identical_metrics` is what makes the reported figures
  reproducible by a third party, which the methodology requires to be verified rather than
  assumed.

The recording preprocessor stands in for the real one in the first two, so that the routine
is what is under test and 30,000 rows of real preprocessing are not. The reproducibility
test uses the real `build_preprocessor`, so the two halves are covered.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from credit_copilot.config import RANDOM_STATE
from credit_copilot.data import schema
from credit_copilot.models.estimators import build_logistic_regression
from credit_copilot.models.evaluation import (
    DEFAULT_N_SPLITS,
    MODEL_STEP,
    PREPROCESSOR_STEP,
    EvaluationInputError,
    build_fold_pipeline,
    cross_validate_estimator,
    format_comparison_table,
    null_reference,
    split_features_and_target,
)
from credit_copilot.models.metrics import REPORTED_METRIC_NAMES

# ---------------------------------------------------------------------------
# Synthetic data shaped like the canonical table
# ---------------------------------------------------------------------------

NUMERIC_STAND_IN = ("LIMIT_BAL", "AGE", *schema.BILL_AMOUNT_COLUMNS, *schema.PAY_AMOUNT_COLUMNS)


def synthetic_frame(n_rows: int = 300, positive_rate: float = 0.22) -> pd.DataFrame:
    """Build a canonical-shaped frame whose target carries some real signal.

    The signal is deliberately weak and driven by `PAY_STATUS_1`, so a logistic regression
    lands somewhere between chance and perfect. A separable target would make every fold
    score 1.0 and the reproducibility test would pass without discriminating anything.
    """
    draw = np.random.default_rng(RANDOM_STATE)
    delinquency = draw.integers(-2, 4, size=n_rows)
    propensity = 1.0 / (1.0 + np.exp(-(delinquency - 1.0)))
    ranked = np.argsort(propensity + draw.normal(0.0, 0.6, size=n_rows))
    target = np.zeros(n_rows, dtype=int)
    target[ranked[-int(round(n_rows * positive_rate)) :]] = 1

    return pd.DataFrame(
        {
            "LIMIT_BAL": draw.integers(10_000, 800_000, size=n_rows),
            "SEX": draw.integers(1, 3, size=n_rows),
            "EDUCATION": draw.integers(1, 5, size=n_rows),
            "MARRIAGE": draw.integers(1, 4, size=n_rows),
            "AGE": draw.integers(21, 70, size=n_rows),
            "PAY_STATUS_1": delinquency,
            **{c: draw.integers(-2, 4, size=n_rows) for c in schema.PAY_STATUS_COLUMNS[1:]},
            **{c: draw.integers(-5_000, 400_000, size=n_rows) for c in schema.BILL_AMOUNT_COLUMNS},
            **{c: draw.integers(0, 90_000, size=n_rows) for c in schema.PAY_AMOUNT_COLUMNS},
            schema.TARGET_COLUMN: target,
        }
    )


# ---------------------------------------------------------------------------
# A preprocessor that leaves a trace every time it is fitted
# ---------------------------------------------------------------------------


@dataclass
class FitCall:
    """One observed `fit` of a preprocessor."""

    instance_id: int
    n_rows: int
    n_positives: int
    saw_target: bool


@dataclass
class Journal:
    """Everything the recording preprocessors saw, in the order they saw it."""

    calls: list[FitCall] = field(default_factory=list)
    factory_calls: int = 0


class RecordingPreprocessor(BaseEstimator, TransformerMixin):
    """Stands in for the real preprocessor and records what each `fit` was handed.

    It also *learns* something - the medians it imputes with - so that "fitted on this
    fold" is a real statement about state and not just a counted call.
    """

    def __init__(self, journal: Journal, columns: Sequence[str]) -> None:
        self.journal = journal
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RecordingPreprocessor":  # noqa: N803
        self.journal.calls.append(
            FitCall(
                instance_id=id(self),
                n_rows=len(X),
                n_positives=-1 if y is None else int(np.sum(np.asarray(y))),
                saw_target=y is not None,
            )
        )
        self.medians_ = X[list(self.columns)].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return X[list(self.columns)].fillna(self.medians_)


def recording_factory(journal: Journal) -> object:
    """Build a factory that yields a fresh recording preprocessor and counts the calls."""

    def factory() -> Pipeline:
        journal.factory_calls += 1
        return Pipeline([("record", RecordingPreprocessor(journal, NUMERIC_STAND_IN))])

    return factory


# ---------------------------------------------------------------------------
# split_features_and_target
# ---------------------------------------------------------------------------


def test_splitting_removes_the_target_from_the_predictors() -> None:
    frame = synthetic_frame(60)
    features, target = split_features_and_target(frame)
    assert schema.TARGET_COLUMN not in features.columns
    assert features.shape == (60, frame.shape[1] - 1)
    assert target.name == schema.TARGET_COLUMN
    assert list(features.index) == list(target.index)


def test_a_frame_without_the_target_is_refused() -> None:
    frame = synthetic_frame(20).drop(columns=[schema.TARGET_COLUMN])
    with pytest.raises(EvaluationInputError, match=schema.TARGET_COLUMN):
        split_features_and_target(frame)


# ---------------------------------------------------------------------------
# The leakage property: one fit per fold, on the training part only
# ---------------------------------------------------------------------------


def test_the_preprocessor_is_fitted_once_per_fold_and_never_before() -> None:
    frame = synthetic_frame(200)
    features, target = split_features_and_target(frame)
    journal = Journal()
    factory = recording_factory(journal)

    assert journal.factory_calls == 0, "building the factory must not build a preprocessor"
    assert journal.calls == [], "nothing may be fitted before the split exists"

    n_splits = 4
    cross_validate_estimator(
        DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        features,
        target,
        n_splits=n_splits,
        preprocessor_factory=factory,  # type: ignore[arg-type]
    )

    assert journal.factory_calls == n_splits
    assert len(journal.calls) == n_splits, "one fit per fold, no more and no fewer"


def test_each_fold_gets_its_own_preprocessor_instance() -> None:
    # A reused instance is a fitted instance: fold 2 would be transformed with fold 1's
    # medians. Distinct identities are what rule that out.
    frame = synthetic_frame(200)
    features, target = split_features_and_target(frame)
    journal = Journal()

    cross_validate_estimator(
        DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        features,
        target,
        n_splits=4,
        preprocessor_factory=recording_factory(journal),  # type: ignore[arg-type]
    )

    identities = [call.instance_id for call in journal.calls]
    assert len(set(identities)) == len(identities)


def test_the_preprocessor_never_sees_the_full_dataset() -> None:
    # This is the assertion that would fail if the preprocessor were fitted before the
    # split, or on features.parquet, which was fitted on all 30,000 rows.
    n_rows, n_splits = 200, 5
    frame = synthetic_frame(n_rows)
    features, target = split_features_and_target(frame)
    journal = Journal()

    cross_validate_estimator(
        DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        features,
        target,
        n_splits=n_splits,
        preprocessor_factory=recording_factory(journal),  # type: ignore[arg-type]
    )

    expected_train_rows = n_rows - n_rows // n_splits
    for call in journal.calls:
        assert call.n_rows == expected_train_rows
        assert call.n_rows < n_rows


def test_the_preprocessor_is_fitted_as_part_of_the_supervised_fit() -> None:
    # If the target never reached the preprocessor's fit, the step would be sitting outside
    # the supervised pipeline and its position relative to the split would be accidental.
    frame = synthetic_frame(120)
    features, target = split_features_and_target(frame)
    journal = Journal()

    cross_validate_estimator(
        DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        features,
        target,
        n_splits=3,
        preprocessor_factory=recording_factory(journal),  # type: ignore[arg-type]
    )

    assert all(call.saw_target for call in journal.calls)


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


def test_every_validation_fold_carries_the_stratified_share_of_positives() -> None:
    n_rows, n_splits = 300, 5
    frame = synthetic_frame(n_rows, positive_rate=0.22)
    features, target = split_features_and_target(frame)

    result = cross_validate_estimator(
        DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        features,
        target,
        n_splits=n_splits,
        preprocessor_factory=recording_factory(Journal()),  # type: ignore[arg-type]
    )

    total_positives = int(target.sum())
    fold_size = n_rows // n_splits
    counts = [round(rate * fold_size) for rate in result.fold_positive_rates]

    # StratifiedKFold divides the positives as evenly as they go: every fold holds either
    # floor(p/k) or ceil(p/k) of them, and together they account for all of them.
    assert sum(counts) == total_positives
    assert set(counts) <= {total_positives // n_splits, -(-total_positives // n_splits)}
    assert result.positive_rate == pytest.approx(total_positives / n_rows)


def test_the_training_folds_are_stratified_too() -> None:
    n_rows, n_splits = 300, 5
    frame = synthetic_frame(n_rows, positive_rate=0.22)
    features, target = split_features_and_target(frame)
    journal = Journal()

    cross_validate_estimator(
        DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        features,
        target,
        n_splits=n_splits,
        preprocessor_factory=recording_factory(journal),  # type: ignore[arg-type]
    )

    overall = float(target.mean())
    for call in journal.calls:
        assert call.n_positives / call.n_rows == pytest.approx(overall, abs=1.0 / call.n_rows)


def test_stratification_holds_even_when_the_target_arrives_sorted() -> None:
    # The contrast that proves shuffling and stratification are actually switched on: on a
    # target sorted 0s-then-1s, a plain KFold hands back single-class folds, and a metric
    # computed on one of those is undefined. The routine's folds stay balanced.
    n_rows, n_splits = 250, 5
    frame = synthetic_frame(n_rows).sort_values(schema.TARGET_COLUMN).reset_index(drop=True)
    features, target = split_features_and_target(frame)

    naive = [
        int(target.iloc[validation].nunique())
        for _, validation in KFold(n_splits=n_splits).split(features)
    ]
    assert min(naive) == 1, "the sorted frame must actually be adversarial for a plain KFold"

    result = cross_validate_estimator(
        DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        features,
        target,
        n_splits=n_splits,
        preprocessor_factory=recording_factory(Journal()),  # type: ignore[arg-type]
    )
    overall = float(target.mean())
    for rate in result.fold_positive_rates:
        assert rate == pytest.approx(overall, abs=0.02)


# ---------------------------------------------------------------------------
# Reproducibility - against the real preprocessor
# ---------------------------------------------------------------------------


def test_the_same_seed_produces_identical_metrics() -> None:
    frame = synthetic_frame(240)
    features, target = split_features_and_target(frame)
    estimator = build_logistic_regression()

    first = cross_validate_estimator(estimator, features, target, random_state=RANDOM_STATE)
    second = cross_validate_estimator(estimator, features, target, random_state=RANDOM_STATE)

    pd.testing.assert_frame_equal(first.fold_metrics, second.fold_metrics)
    assert first.means == second.means
    assert first.stds == second.stds


def test_a_different_seed_moves_the_metrics() -> None:
    # The mirror of the test above. If the seed changed nothing, the reproducibility test
    # would be passing because the routine ignores its seed, which is a different fact.
    frame = synthetic_frame(240)
    features, target = split_features_and_target(frame)
    estimator = build_logistic_regression()

    first = cross_validate_estimator(estimator, features, target, random_state=RANDOM_STATE)
    other = cross_validate_estimator(estimator, features, target, random_state=RANDOM_STATE + 1)

    assert not first.fold_metrics.equals(other.fold_metrics)


def test_a_reused_estimator_object_does_not_carry_state_between_evaluations() -> None:
    frame = synthetic_frame(200)
    features, target = split_features_and_target(frame)
    estimator = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)

    first = cross_validate_estimator(estimator, features, target, n_splits=3)
    assert not hasattr(estimator, "coef_"), "the caller's object must not be fitted in place"
    second = cross_validate_estimator(estimator, features, target, n_splits=3)
    pd.testing.assert_frame_equal(first.fold_metrics, second.fold_metrics)


# ---------------------------------------------------------------------------
# The shape of the result, and the guard rails
# ---------------------------------------------------------------------------


def test_the_result_carries_every_reported_metric_for_every_fold() -> None:
    frame = synthetic_frame(150)
    features, target = split_features_and_target(frame)
    result = cross_validate_estimator(
        DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        features,
        target,
        n_splits=3,
        preprocessor_factory=recording_factory(Journal()),  # type: ignore[arg-type]
    )

    assert list(result.fold_metrics.columns) == list(REPORTED_METRIC_NAMES)
    assert list(result.fold_metrics.index) == [1, 2, 3]
    assert result.n_splits == 3
    assert result.n_samples == 150
    assert result.random_state == RANDOM_STATE


def test_the_summary_appends_mean_and_std_without_touching_the_fold_table() -> None:
    frame = synthetic_frame(150)
    features, target = split_features_and_target(frame)
    result = cross_validate_estimator(
        DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        features,
        target,
        n_splits=3,
        preprocessor_factory=recording_factory(Journal()),  # type: ignore[arg-type]
    )

    summary = result.summary_frame()
    assert list(summary.index) == [1, 2, 3, "mean", "std"]
    assert summary.loc["mean", "pr_auc"] == pytest.approx(result.means["pr_auc"])
    assert list(result.fold_metrics.index) == [1, 2, 3], "the fold table was modified in place"


def test_the_fold_pipeline_puts_the_preprocessor_before_the_model() -> None:
    pipeline = build_fold_pipeline(DummyClassifier())
    assert list(pipeline.named_steps) == [PREPROCESSOR_STEP, MODEL_STEP]


def test_an_estimator_without_predict_proba_is_refused() -> None:
    from sklearn.svm import LinearSVC

    frame = synthetic_frame(60)
    features, target = split_features_and_target(frame)
    with pytest.raises(EvaluationInputError, match="predict_proba"):
        cross_validate_estimator(LinearSVC(), features, target)


def test_fewer_than_two_folds_is_refused() -> None:
    frame = synthetic_frame(60)
    features, target = split_features_and_target(frame)
    with pytest.raises(EvaluationInputError, match="n_splits"):
        cross_validate_estimator(DummyClassifier(), features, target, n_splits=1)


def test_mismatched_lengths_are_refused() -> None:
    frame = synthetic_frame(60)
    features, target = split_features_and_target(frame)
    with pytest.raises(EvaluationInputError, match="rows"):
        cross_validate_estimator(DummyClassifier(), features, target.iloc[:10])


def test_the_default_protocol_is_five_stratified_folds() -> None:
    assert DEFAULT_N_SPLITS == 5


# ---------------------------------------------------------------------------
# Reading aids
# ---------------------------------------------------------------------------


def test_the_no_signal_floor_is_not_the_same_number_for_every_metric() -> None:
    frame = synthetic_frame(200, positive_rate=0.25)
    features, target = split_features_and_target(frame)
    result = cross_validate_estimator(
        DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        features,
        target,
        n_splits=4,
        preprocessor_factory=recording_factory(Journal()),  # type: ignore[arg-type]
    )

    floors = null_reference(result)
    prevalence = result.positive_rate
    assert floors["pr_auc"] == pytest.approx(prevalence)
    assert floors["precision_at_top_10pct"] == pytest.approx(prevalence)
    assert floors["roc_auc"] == pytest.approx(0.5)
    assert floors["gini"] == pytest.approx(0.0)
    assert floors["brier"] == pytest.approx(prevalence * (1 - prevalence))
    assert floors["accuracy"] == pytest.approx(1 - prevalence)


def test_the_comparison_table_reports_every_metric_with_its_spread() -> None:
    frame = synthetic_frame(150)
    features, target = split_features_and_target(frame)
    result = cross_validate_estimator(
        DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        features,
        target,
        n_splits=3,
        preprocessor_factory=recording_factory(Journal()),  # type: ignore[arg-type]
    )

    table = format_comparison_table({"only-model": result})
    assert "only-model" in table
    for name in REPORTED_METRIC_NAMES:
        assert name in table
    assert table.count("±") == len(REPORTED_METRIC_NAMES), "a mean without its std"
