"""Tests of the preprocessing pipeline.

Most of these run on hand-built frames, for the reason given in `test_features.py`: a test
that asserts against 30,000 measured rows can only restate what the code computed.

Four of them are different in kind and worth naming, because they check properties that
have no visible symptom when they break:

- `test_two_folds_learn_different_parameters` is the **leakage** test. Every statistic the
  pipeline learns is extracted and compared between two fits on disjoint subsets. If they
  came out identical, the pipeline would be learning from something that is not the fold.
- `test_permuting_the_target_does_not_move_the_matrix` checks the label cannot reach the
  matrix, by moving the label and requiring the matrix not to move.
- `test_a_joblib_round_trip_through_disk_produces_the_same_matrix` exercises serialisation
  through the filesystem, not through an object kept in memory, because the API will load
  the artefact and not inherit it.
- `test_the_three_branches_partition_every_column_the_pipeline_is_given` is what makes
  `remainder="drop"` safe: it proves no source column is silently discarded.

The column groups below are spelled out literally rather than imported from the module
under test. A test that derives its expectations from the code it checks agrees with that
code by construction, including when the code is wrong.
"""

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from credit_copilot.config import RANDOM_STATE
from credit_copilot.data import schema
from credit_copilot.data.preprocessor import (
    AttachPaymentBehaviourFeatures,
    CollapseEducation,
    PercentileClipper,
    PreprocessorInputError,
    build_preprocessor,
    learned_parameters,
)

EXPECTED_CATEGORICAL = (
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_STATUS_1",
    "PAY_STATUS_2",
    "PAY_STATUS_3",
    "PAY_STATUS_4",
    "PAY_STATUS_5",
    "PAY_STATUS_6",
)
EXPECTED_NUMERIC_SOURCE = (
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
)
EXPECTED_INDICATORS = (
    "PAYMENT_RATIO_NOT_COMPUTABLE_M2",
    "PAYMENT_RATIO_NOT_COMPUTABLE_M3",
    "PAYMENT_RATIO_NOT_COMPUTABLE_M4",
    "PAYMENT_RATIO_NOT_COMPUTABLE_M5",
    "UTILIZATION_NOT_COMPUTABLE",
    "IS_DELINQUENT_MOST_RECENT_M1",
)

_DEFAULT_ROW: dict[str, int] = {
    "LIMIT_BAL": 200_000,
    "SEX": 1,
    "EDUCATION": 1,
    "MARRIAGE": 1,
    "AGE": 30,
    **dict.fromkeys(schema.PAY_STATUS_COLUMNS, -1),
    **dict.fromkeys(schema.BILL_AMOUNT_COLUMNS, 40_000),
    **dict.fromkeys(schema.PAY_AMOUNT_COLUMNS, 4_000),
    schema.TARGET_COLUMN: 0,
}


def make_frame(*rows: dict[str, int], index: Sequence[int] | None = None) -> pd.DataFrame:
    """Build a canonical-looking frame, overriding only the columns a test cares about."""
    material = rows if rows else ({},)
    return pd.DataFrame([{**_DEFAULT_ROW, **row} for row in material], index=index)


def varied_frame(n: int = 60) -> pd.DataFrame:
    """Build a frame with enough spread that the learned statistics are not degenerate."""
    draw = np.random.default_rng(RANDOM_STATE)
    rows = [
        {
            "LIMIT_BAL": int(draw.integers(10_000, 800_000)),
            "AGE": int(draw.integers(21, 70)),
            "SEX": int(draw.integers(1, 3)),
            "EDUCATION": int(draw.integers(1, 5)),
            "MARRIAGE": int(draw.integers(1, 4)),
            **{c: int(draw.integers(-2, 4)) for c in schema.PAY_STATUS_COLUMNS},
            **{c: int(draw.integers(-5_000, 400_000)) for c in schema.BILL_AMOUNT_COLUMNS},
            **{c: int(draw.integers(0, 90_000)) for c in schema.PAY_AMOUNT_COLUMNS},
            schema.TARGET_COLUMN: int(draw.integers(0, 2)),
        }
        for _ in range(n)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The three branches, and what remainder="drop" is allowed to drop
# ---------------------------------------------------------------------------


def test_the_three_branches_partition_every_column_the_pipeline_is_given() -> None:
    """`remainder="drop"` must be silent about nothing except the target.

    Every source predictor lands in exactly one branch, and every derived feature too. If
    a column were added to the contract and to no branch, it would vanish from the matrix
    without any error - which is the failure this test exists to make impossible.
    """
    from credit_copilot.data.preprocessor import (
        CATEGORICAL_COLUMNS,
        INDICATOR_COLUMNS,
        NUMERIC_COLUMNS,
    )
    from credit_copilot.features.builder import FEATURE_NAMES

    routed = set(CATEGORICAL_COLUMNS) | set(NUMERIC_COLUMNS) | set(INDICATOR_COLUMNS)
    predictors = set(schema.WORKING_COLUMNS) - {schema.TARGET_COLUMN}

    assert predictors | set(FEATURE_NAMES) == routed, "a column has no declared destination"
    assert schema.TARGET_COLUMN not in routed, "the target reaches a branch"

    assert set(CATEGORICAL_COLUMNS) == set(EXPECTED_CATEGORICAL)
    assert set(INDICATOR_COLUMNS) == set(EXPECTED_INDICATORS)
    assert set(EXPECTED_NUMERIC_SOURCE) <= set(NUMERIC_COLUMNS)

    for group_a, group_b in (
        (CATEGORICAL_COLUMNS, NUMERIC_COLUMNS),
        (CATEGORICAL_COLUMNS, INDICATOR_COLUMNS),
        (NUMERIC_COLUMNS, INDICATOR_COLUMNS),
    ):
        assert not set(group_a) & set(group_b), "a column is routed to two branches"


def test_the_repayment_codes_are_one_hot_encoded_and_never_scaled() -> None:
    """ADR-0004 decision 2 calls this consequence direct and non-negotiable."""
    from credit_copilot.data.preprocessor import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

    for column in schema.PAY_STATUS_COLUMNS:
        assert column in CATEGORICAL_COLUMNS
        assert column not in NUMERIC_COLUMNS


# ---------------------------------------------------------------------------
# Names: the matrix has to be readable, or SHAP explains the wrong variable
# ---------------------------------------------------------------------------


def test_get_feature_names_out_matches_the_columns_of_the_matrix() -> None:
    frame = varied_frame()
    pipeline = build_preprocessor()
    matrix = pipeline.fit_transform(frame)

    names = list(pipeline.get_feature_names_out())
    assert names == list(matrix.columns)
    assert len(names) == matrix.shape[1]
    assert len(names) == len(set(names)), "a duplicated output name makes SHAP ambiguous"


def test_the_matrix_carries_no_source_column_under_its_own_name() -> None:
    """Every column is the output of a branch, so a raw name surviving means passthrough."""
    frame = varied_frame()
    matrix = build_preprocessor().fit_transform(frame)
    for column in EXPECTED_CATEGORICAL:
        assert column not in matrix.columns, f"{column} was not encoded"


# ---------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------


def test_the_target_never_reaches_the_matrix() -> None:
    frame = varied_frame()
    matrix = build_preprocessor().fit_transform(frame)
    assert schema.TARGET_COLUMN not in matrix.columns


def test_permuting_the_target_does_not_move_the_matrix() -> None:
    """The label moves; the matrix must not. Both fits see the same column names."""
    frame = varied_frame()
    flipped = frame.assign(**{schema.TARGET_COLUMN: 1 - frame[schema.TARGET_COLUMN]})

    original = build_preprocessor().fit_transform(frame)
    permuted = build_preprocessor().fit_transform(flipped)
    pd.testing.assert_frame_equal(permuted, original)


def test_passing_y_to_fit_changes_nothing() -> None:
    frame = varied_frame()
    target = frame[schema.TARGET_COLUMN]

    without_y = build_preprocessor().fit_transform(frame)
    with_y = build_preprocessor().fit_transform(frame, target)
    with_wrong_y = build_preprocessor().fit_transform(frame, 1 - target)

    pd.testing.assert_frame_equal(with_y, without_y)
    pd.testing.assert_frame_equal(with_wrong_y, without_y)


# ---------------------------------------------------------------------------
# Cross-validation: fit on one subset, transform another
# ---------------------------------------------------------------------------


def test_fit_on_one_subset_and_transform_another_produces_the_full_matrix() -> None:
    """The condition for entering a cross-validation at all."""
    frame = varied_frame(80)
    train, test = frame.iloc[:50], frame.iloc[50:]

    pipeline = build_preprocessor().fit(train)
    transformed = pipeline.transform(test)

    assert list(transformed.columns) == list(pipeline.get_feature_names_out())
    assert len(transformed) == len(test)
    assert list(transformed.index) == list(test.index)


def test_two_folds_learn_different_parameters() -> None:
    """The leakage test: what `fit` learns must be a property of the fold it saw.

    Identical parameters from two disjoint subsets would mean the statistics came from
    somewhere other than the training data - a module-level constant computed at import
    time, a cached value, a full-dataset statistic. None of those has a symptom in the
    output, so the only way to see it is to compare what two fits learned.
    """
    frame = varied_frame(120)
    first = build_preprocessor().fit(frame.iloc[:60])
    second = build_preprocessor().fit(frame.iloc[60:])

    left, right = learned_parameters(first), learned_parameters(second)
    assert set(left) == set(right)

    for step in left:
        differing = [key for key in left[step] if left[step][key] != right[step][key]]
        assert differing, f"the {step!r} step learned identical parameters from two folds"


def test_the_pipeline_survives_a_real_cross_validation() -> None:
    """Not a claim that it *can* be cross-validated: an actual 5-fold run of it."""
    frame = varied_frame(120)
    target = frame[schema.TARGET_COLUMN]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_validate(
        build_preprocessor(),
        frame,
        target,
        cv=splitter,
        scoring=lambda estimator, X, y: float(estimator.transform(X).shape[1]),  # noqa: ARG005, N803
    )
    widths = set(scores["test_score"])
    assert len(widths) == 1, f"the matrix changed width between folds: {widths}"


# ---------------------------------------------------------------------------
# Unknown categories
# ---------------------------------------------------------------------------


def test_a_category_not_seen_in_fit_does_not_break_transform() -> None:
    """A rare but documented code absent from a training fold must still be servable.

    Measured on the real dataset with `StratifiedKFold(5, shuffle=True,
    random_state=RANDOM_STATE)`, two folds contain a `PAY_STATUS` level their own training
    part never saw. `handle_unknown="error"` would crash the project's own cross-validation
    on legitimate data.
    """
    train = make_frame(*({"PAY_STATUS_1": code} for code in (-1, 0, 1)))
    unseen = make_frame({"PAY_STATUS_1": 8})

    pipeline = build_preprocessor().fit(train)
    transformed = pipeline.transform(unseen)

    assert list(transformed.columns) == list(pipeline.get_feature_names_out())
    assert transformed.notna().all().all()


def test_an_unseen_level_is_encoded_as_all_zeros_and_not_as_a_known_level() -> None:
    """Silently mapping the unknown onto a known level would be worse than dropping it."""
    train = make_frame(*({"PAY_STATUS_1": code} for code in (-1, 0, 1)))
    pipeline = build_preprocessor().fit(train)

    block = [name for name in pipeline.get_feature_names_out() if name.startswith("PAY_STATUS_1_")]
    assert block, "the repayment codes of month 1 produced no one-hot block"

    unseen = pipeline.transform(make_frame({"PAY_STATUS_1": 8}))
    known = pipeline.transform(make_frame({"PAY_STATUS_1": 0}))
    assert unseen[block].to_numpy().sum() == 0
    assert known[block].to_numpy().sum() == 1


def test_the_undocumented_education_codes_are_collapsed_onto_level_four() -> None:
    """ADR-0004 decision 4. `MARRIAGE` is deliberately not given the same treatment."""
    frame = make_frame(*({"EDUCATION": code} for code in (0, 5, 6, 4, 1)))
    collapsed = CollapseEducation().fit(frame).transform(frame)
    assert list(collapsed["EDUCATION"]) == [4, 4, 4, 4, 1]


def test_marriage_is_not_collapsed() -> None:
    frame = make_frame(*({"MARRIAGE": code} for code in (0, 1, 2, 3)))
    collapsed = CollapseEducation().fit(frame).transform(frame)
    assert list(collapsed["MARRIAGE"]) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# The clipping step
# ---------------------------------------------------------------------------


def test_the_clipper_learns_its_threshold_in_fit_and_applies_it_in_transform() -> None:
    values = pd.DataFrame({"RATIO": [float(v) for v in range(101)], "OTHER": [1.0] * 101})
    clipper = PercentileClipper(columns=("RATIO",), percentile=90.0).fit(values)

    assert clipper.upper_bounds_["RATIO"] == pytest.approx(90.0)
    clipped = clipper.transform(values)
    assert clipped["RATIO"].max() == pytest.approx(90.0)
    assert clipped["OTHER"].equals(values["OTHER"]), "an unconfigured column was touched"


def test_the_clipper_caps_only_the_upper_side() -> None:
    values = pd.DataFrame({"RATIO": [-5.0, 0.0, 1.0, 500.0]})
    clipped = PercentileClipper(columns=("RATIO",), percentile=75.0).fit(values).transform(values)
    assert clipped["RATIO"].min() == pytest.approx(-5.0), "the lower side was clipped"


def test_the_clipper_leaves_missing_values_missing() -> None:
    """Capping an unknown would be an imputation, and this step does not impute."""
    values = pd.DataFrame({"RATIO": [1.0, np.nan, 2.0, 900.0]})
    clipped = PercentileClipper(columns=("RATIO",), percentile=50.0).fit(values).transform(values)
    assert pd.isna(clipped.loc[1, "RATIO"])
    assert clipped["RATIO"].notna().sum() == 3


def test_an_all_missing_column_gets_no_cap_instead_of_a_nan_cap() -> None:
    """A `NaN` threshold would turn every row of the column into `NaN` without a word."""
    values = pd.DataFrame({"RATIO": [np.nan, np.nan], "OTHER": [1.0, 2.0]})
    clipper = PercentileClipper(columns=("RATIO",), percentile=99.5).fit(values)
    assert clipper.upper_bounds_["RATIO"] == float("inf")
    assert clipper.transform(values)["OTHER"].notna().all()


def test_the_clip_threshold_is_a_property_of_the_training_fold_only() -> None:
    small = pd.DataFrame({"RATIO": [float(v) for v in range(10)]})
    large = pd.DataFrame({"RATIO": [float(v) for v in range(1_000)]})

    fitted_on_small = PercentileClipper(columns=("RATIO",), percentile=99.5).fit(small)
    fitted_on_large = PercentileClipper(columns=("RATIO",), percentile=99.5).fit(large)
    assert fitted_on_small.upper_bounds_ != fitted_on_large.upper_bounds_

    # Transforming the large frame with the small fold's threshold must use that threshold.
    assert fitted_on_small.transform(large)["RATIO"].max() == pytest.approx(
        fitted_on_small.upper_bounds_["RATIO"]
    )


def test_an_out_of_range_percentile_fails_loudly() -> None:
    values = pd.DataFrame({"RATIO": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        PercentileClipper(columns=("RATIO",), percentile=120.0).fit(values)


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


def test_the_matrix_carries_no_missing_value() -> None:
    """Deliberate: the `NaN` are imputed, and the fact that they were is a column.

    ADR-0005 makes every non-computable value a `NaN` with an indicator beside it. The
    imputation replaces the value and keeps the indicator, so nothing about the absence is
    lost; see `test_the_absence_survives_the_imputation`.
    """
    frame = varied_frame(80)
    matrix = build_preprocessor().fit_transform(frame)
    assert int(matrix.isna().sum().sum()) == 0


def test_the_absence_survives_the_imputation() -> None:
    """The imputed value is gone; the fact that it was unknown is still in the matrix."""
    computable = make_frame({"PAY_AMT2": 5_000, "BILL_AMT3": 40_000})
    unknown = make_frame({"PAY_AMT2": 5_000, "BILL_AMT3": 0})
    frame = pd.concat([computable, unknown], ignore_index=True)

    matrix = build_preprocessor().fit_transform(frame)
    assert matrix["PAYMENT_RATIO_NOT_COMPUTABLE_M2"].tolist() == [0, 1]
    assert matrix["PAYMENT_RATIO_M2"].notna().all()


def test_the_imputation_median_is_learned_and_not_a_constant() -> None:
    first = build_preprocessor().fit(varied_frame(60))
    second = build_preprocessor().fit(varied_frame(60).assign(LIMIT_BAL=999_999))
    assert (
        learned_parameters(first)["impute"]["LIMIT_BAL"]
        != learned_parameters(second)["impute"]["LIMIT_BAL"]
    )


# ---------------------------------------------------------------------------
# Serialisation and the sklearn contract
# ---------------------------------------------------------------------------


def test_a_joblib_round_trip_through_disk_produces_the_same_matrix(tmp_path: Path) -> None:
    """Through the filesystem, not through an object kept in memory: the API loads a file."""
    frame = varied_frame(80)
    pipeline = build_preprocessor().fit(frame)
    before = pipeline.transform(frame)

    artefact = tmp_path / "preprocessor.joblib"
    joblib.dump(pipeline, artefact)
    restored = joblib.load(artefact)

    pd.testing.assert_frame_equal(restored.transform(frame), before)
    assert list(restored.get_feature_names_out()) == list(pipeline.get_feature_names_out())


def test_a_single_row_round_trips_through_the_loaded_artefact(tmp_path: Path) -> None:
    """The production path: one request, one loaded artefact, one row of the matrix."""
    frame = varied_frame(80)
    pipeline = build_preprocessor().fit(frame)

    artefact = tmp_path / "preprocessor.joblib"
    joblib.dump(pipeline, artefact)
    restored = joblib.load(artefact)

    one_row = frame.iloc[[7]]
    served = restored.transform(one_row)
    assert len(served) == 1
    pd.testing.assert_frame_equal(served, pipeline.transform(frame).iloc[[7]])


def test_build_preprocessor_returns_an_unfitted_pipeline() -> None:
    pipeline = build_preprocessor()
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["behaviour", "education", "clip", "columns"]
    with pytest.raises(Exception, match="(?i)not fitted|fit"):
        pipeline.transform(make_frame())


def test_the_clip_percentile_is_a_parameter_with_a_default() -> None:
    default = build_preprocessor()
    custom = build_preprocessor(clip_percentile=90.0)
    assert default.named_steps["clip"].percentile == 99.5
    assert custom.named_steps["clip"].percentile == 90.0


def test_the_index_of_the_input_is_preserved() -> None:
    frame = make_frame({}, {}, index=[17, 42])
    matrix = build_preprocessor().fit_transform(frame)
    assert list(matrix.index) == [17, 42]


def test_no_step_mutates_its_input() -> None:
    frame = varied_frame(40)
    untouched = frame.copy()
    build_preprocessor().fit_transform(frame)
    pd.testing.assert_frame_equal(frame, untouched)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_a_missing_source_column_fails_loudly() -> None:
    frame = varied_frame(20).drop(columns=["BILL_AMT3"])
    with pytest.raises(Exception, match="BILL_AMT3"):
        build_preprocessor().fit(frame)


def test_an_input_without_column_names_fails_loudly() -> None:
    with pytest.raises(TypeError):
        AttachPaymentBehaviourFeatures().fit(np.zeros((3, 24)))


def test_attaching_over_an_existing_derived_name_fails_loudly() -> None:
    """A duplicated column name makes every later selection silently ambiguous."""
    frame = make_frame().assign(PAYMENT_RATIO_M2=0.5)
    with pytest.raises(PreprocessorInputError, match="PAYMENT_RATIO_M2"):
        AttachPaymentBehaviourFeatures().fit(frame)


def test_the_attach_step_keeps_the_source_columns_and_adds_the_derived_ones() -> None:
    """`PaymentBehaviourFeatures` alone returns only the derived block; this keeps both."""
    from credit_copilot.features.builder import FEATURE_NAMES

    frame = make_frame()
    attached = AttachPaymentBehaviourFeatures().fit(frame).transform(frame)
    assert list(attached.columns) == [*frame.columns, *FEATURE_NAMES]
    pd.testing.assert_frame_equal(attached[list(frame.columns)], frame)
