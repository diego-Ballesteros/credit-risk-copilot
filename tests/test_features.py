"""Tests of the payment-behaviour features, on hand-built frames only.

The real dataset never appears here. A test that asserts against 30,000 measured rows can
only restate what the code computed; these assert values worked out by hand from the
definition, so they can disagree with the code.

Two of them deserve a note on method, because "we did not mix the two repayment scales"
is the kind of claim that is easy to write and easy to fake:

- `test_block_features_are_computable_without_the_month_one_columns` is **structural**. It
  hands the block builder a frame from which the month-1 columns have been deleted
  outright. If any block feature read one, the lookup would raise. This proves the block
  does not *need* month 1.
- `test_block_features_are_invariant_to_month_one` is **behavioural**. It perturbs only
  the month-1 columns, over many draws from a generator seeded with the project's
  `RANDOM_STATE`, and requires every block column to come back bit-identical. This proves
  the block does not *use* month 1 on the inputs tested.

Together they are strong but not a proof over every possible input: the second is an
empirical check at finitely many points, and only a static analysis of the data flow would
close that gap. The pair is reported as what it is rather than dressed up as a proof.

The lists of month-1 and block source columns below are spelled out literally instead of
imported from the module under test. A test that derives its expectations from the code it
checks agrees with that code by construction, including when the code is wrong.
"""

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from credit_copilot.config import RANDOM_STATE
from credit_copilot.data import schema
from credit_copilot.features import builder
from credit_copilot.features.builder import (
    FEATURE_NAMES,
    MissingSourceColumnsError,
    PaymentBehaviourFeatures,
)

MONTH_ONE_SOURCE_COLUMNS = ("PAY_STATUS_1", "BILL_AMT1", "PAY_AMT1")
BLOCK_ONLY_SOURCE_COLUMNS = (
    "PAY_STATUS_2",
    "PAY_STATUS_3",
    "PAY_STATUS_4",
    "PAY_STATUS_5",
    "PAY_STATUS_6",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
)
MONTH_ONE_FEATURES = ("UTILIZATION_MOST_RECENT_M1", "IS_DELINQUENT_MOST_RECENT_M1")
BLOCK_FEATURES = tuple(name for name in FEATURE_NAMES if name not in MONTH_ONE_FEATURES)

_DEFAULT_ROW: dict[str, int] = {
    "LIMIT_BAL": 1_000,
    "SEX": 1,
    "EDUCATION": 1,
    "MARRIAGE": 1,
    "AGE": 30,
    **dict.fromkeys(schema.PAY_STATUS_COLUMNS, -1),
    **dict.fromkeys(schema.BILL_AMOUNT_COLUMNS, 100),
    **dict.fromkeys(schema.PAY_AMOUNT_COLUMNS, 10),
    schema.TARGET_COLUMN: 0,
}


def make_frame(*rows: dict[str, int], index: Sequence[int] | None = None) -> pd.DataFrame:
    """Build a canonical-looking frame, overriding only the columns a test cares about."""
    material = rows if rows else ({},)
    return pd.DataFrame([{**_DEFAULT_ROW, **row} for row in material], index=index)


def features(*rows: dict[str, int]) -> pd.DataFrame:
    """Run the transformer over hand-built rows and return the derived frame."""
    return PaymentBehaviourFeatures().fit_transform(make_frame(*rows))


# ---------------------------------------------------------------------------
# Utilisation
# ---------------------------------------------------------------------------


def test_utilization_is_the_balance_over_the_granted_limit() -> None:
    out = features({"LIMIT_BAL": 1_000, "BILL_AMT2": 250, "BILL_AMT6": 900})
    assert out.loc[0, "UTILIZATION_M2"] == pytest.approx(0.25)
    assert out.loc[0, "UTILIZATION_M6"] == pytest.approx(0.90)


def test_month_one_utilization_uses_the_month_one_balance() -> None:
    out = features({"LIMIT_BAL": 1_500, "BILL_AMT1": 750})
    assert out.loc[0, "UTILIZATION_MOST_RECENT_M1"] == pytest.approx(0.5)


def test_utilization_trend_is_the_slope_in_points_per_month() -> None:
    """Utilisation 0.1, 0.2, 0.3, 0.4, 0.5 from April to August rises 0.1 per month."""
    rising = {
        "LIMIT_BAL": 1_000,
        "BILL_AMT6": 100,
        "BILL_AMT5": 200,
        "BILL_AMT4": 300,
        "BILL_AMT3": 400,
        "BILL_AMT2": 500,
    }
    falling = {
        "LIMIT_BAL": 1_000,
        "BILL_AMT6": 500,
        "BILL_AMT5": 400,
        "BILL_AMT4": 300,
        "BILL_AMT3": 200,
        "BILL_AMT2": 100,
    }
    out = features(rising, falling, {})
    assert out.loc[0, "UTILIZATION_TREND_M2_M6"] == pytest.approx(0.1)
    assert out.loc[1, "UTILIZATION_TREND_M2_M6"] == pytest.approx(-0.1)
    assert out.loc[2, "UTILIZATION_TREND_M2_M6"] == pytest.approx(0.0)


def test_utilization_trend_is_not_the_endpoint_difference() -> None:
    """Four flat months at 0.2 and a final 0.6 give a slope of 0.08, not a jump of 0.4.

    This is the test that makes the choice of definition falsifiable rather than merely
    documented: the endpoint difference would return 0.4 here, and the mean of consecutive
    differences 0.1.
    """
    out = features(
        {
            "LIMIT_BAL": 1_000,
            "BILL_AMT6": 200,
            "BILL_AMT5": 200,
            "BILL_AMT4": 200,
            "BILL_AMT3": 200,
            "BILL_AMT2": 600,
        }
    )
    assert out.loc[0, "UTILIZATION_TREND_M2_M6"] == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# Payment ratio and the index direction
# ---------------------------------------------------------------------------


def test_payment_ratio_divides_by_the_previous_month_which_carries_the_higher_index() -> None:
    """`PAYMENT_RATIO_M2` is `PAY_AMT2 / BILL_AMT3`, never `/ BILL_AMT1`.

    The decoy in `BILL_AMT1` is a hundred times the real denominator: reading the index in
    the intuitive direction would produce 0.001 instead of 0.25.
    """
    out = features({"PAY_AMT2": 100, "BILL_AMT3": 400, "BILL_AMT1": 40_000})
    assert out.loc[0, "PAYMENT_RATIO_M2"] == pytest.approx(0.25)


def test_payment_ratio_ignores_the_balance_of_its_own_month() -> None:
    same_month_changed = features(
        {"PAY_AMT2": 100, "BILL_AMT3": 400, "BILL_AMT2": 100},
        {"PAY_AMT2": 100, "BILL_AMT3": 400, "BILL_AMT2": 999_999},
    )
    assert same_month_changed.loc[0, "PAYMENT_RATIO_M2"] == pytest.approx(0.25)
    assert same_month_changed.loc[1, "PAYMENT_RATIO_M2"] == pytest.approx(0.25)


def test_every_payment_ratio_month_lands_on_its_own_denominator() -> None:
    out = features(
        {
            "PAY_AMT2": 200,
            "PAY_AMT3": 150,
            "PAY_AMT4": 50,
            "PAY_AMT5": 100,
            "BILL_AMT3": 400,
            "BILL_AMT4": 300,
            "BILL_AMT5": 200,
            "BILL_AMT6": 100,
        }
    )
    assert out.loc[0, "PAYMENT_RATIO_M2"] == pytest.approx(0.5)
    assert out.loc[0, "PAYMENT_RATIO_M3"] == pytest.approx(0.5)
    assert out.loc[0, "PAYMENT_RATIO_M4"] == pytest.approx(0.25)
    assert out.loc[0, "PAYMENT_RATIO_M5"] == pytest.approx(1.0)


def test_the_oldest_block_month_has_no_payment_ratio() -> None:
    """Month 6's previous month is outside the homogeneous window, so it is not built."""
    assert builder.PAYMENT_RATIO_MONTHS == (2, 3, 4, 5)
    assert "PAYMENT_RATIO_M6" not in FEATURE_NAMES
    assert "PAYMENT_RATIO_M1" not in FEATURE_NAMES


# ---------------------------------------------------------------------------
# The zero-denominator policy
# ---------------------------------------------------------------------------


def test_a_zero_or_negative_previous_balance_makes_the_ratio_unknown_not_zero() -> None:
    out = features(
        {"PAY_AMT2": 100, "BILL_AMT3": 400},
        {"PAY_AMT2": 100, "BILL_AMT3": 0},
        {"PAY_AMT2": 100, "BILL_AMT3": -500},
    )
    assert out.loc[0, "PAYMENT_RATIO_M2"] == pytest.approx(0.25)
    assert out.loc[0, "PAYMENT_RATIO_NOT_COMPUTABLE_M2"] == 0

    for row in (1, 2):
        assert pd.isna(out.loc[row, "PAYMENT_RATIO_M2"]), "unknown was filled with a number"
        assert out.loc[row, "PAYMENT_RATIO_NOT_COMPUTABLE_M2"] == 1


def test_a_negative_previous_balance_never_produces_a_negative_ratio() -> None:
    """Dividing by a credit balance would flip the sign and read as excellent behaviour."""
    out = features({"PAY_AMT2": 5_000, "BILL_AMT3": -2_000})
    assert pd.isna(out.loc[0, "PAYMENT_RATIO_M2"])
    assert not (out["PAYMENT_RATIO_M2"].fillna(0) < 0).any()


def test_a_non_positive_credit_limit_makes_every_utilization_unknown() -> None:
    out = features({"LIMIT_BAL": 0}, {"LIMIT_BAL": -100}, {"LIMIT_BAL": 1_000})
    utilization_columns = [*builder.UTILIZATION_FEATURES, "UTILIZATION_MOST_RECENT_M1"]

    for row in (0, 1):
        assert out.loc[row, utilization_columns].isna().all()
        assert pd.isna(out.loc[row, "UTILIZATION_TREND_M2_M6"]), "a flat 0.0 trend was invented"
        assert out.loc[row, "UTILIZATION_NOT_COMPUTABLE"] == 1

    assert out.loc[2, utilization_columns].notna().all()
    assert out.loc[2, "UTILIZATION_NOT_COMPUTABLE"] == 0


def test_the_non_computable_indicators_are_columns_of_their_own() -> None:
    """The policy has to be readable by a model, not buried in a sentinel value."""
    indicators = ("UTILIZATION_NOT_COMPUTABLE", *builder.PAYMENT_RATIO_NOT_COMPUTABLE_FEATURES)
    assert set(indicators) <= set(FEATURE_NAMES)

    out = features({"LIMIT_BAL": 0, "BILL_AMT3": 0})
    for indicator in indicators:
        assert out[indicator].dtype == "int64"
        assert set(out[indicator].unique()) <= {0, 1}


# ---------------------------------------------------------------------------
# Delinquency
# ---------------------------------------------------------------------------


def test_delinquency_streak_counts_back_from_the_most_recent_month_of_the_block() -> None:
    """Months 2 and 3 in arrears, month 4 clean: the run that reaches the end is 2 long."""
    out = features(
        {
            "PAY_STATUS_2": 2,
            "PAY_STATUS_3": 1,
            "PAY_STATUS_4": -1,
            "PAY_STATUS_5": 3,
            "PAY_STATUS_6": 4,
        }
    )
    assert out.loc[0, "DELINQUENCY_STREAK_M2_M6"] == 2


def test_delinquency_streak_is_zero_when_the_most_recent_block_month_is_clean() -> None:
    """A client who fell behind and recovered scores zero, however bad the earlier run."""
    out = features(
        {
            "PAY_STATUS_2": -1,
            "PAY_STATUS_3": 5,
            "PAY_STATUS_4": 5,
            "PAY_STATUS_5": 5,
            "PAY_STATUS_6": 5,
        }
    )
    assert out.loc[0, "DELINQUENCY_STREAK_M2_M6"] == 0
    assert out.loc[0, "MAX_DELINQUENCY_M2_M6"] == 5


def test_delinquency_streak_saturates_at_the_block_length() -> None:
    out = features(dict.fromkeys(schema.PAY_STATUS_HOMOGENEOUS_COLUMNS, 3))
    assert out.loc[0, "DELINQUENCY_STREAK_M2_M6"] == 5


@pytest.mark.parametrize("code", [-2, -1, 0])
def test_the_codes_adr_0004_declares_not_to_be_arrears_break_the_streak(code: int) -> None:
    out = features(
        {
            "PAY_STATUS_2": code,
            "PAY_STATUS_3": 4,
            "PAY_STATUS_4": 4,
            "PAY_STATUS_5": 4,
            "PAY_STATUS_6": 4,
        }
    )
    assert out.loc[0, "DELINQUENCY_STREAK_M2_M6"] == 0


def test_max_delinquency_floors_the_non_arrears_codes_instead_of_ranking_them() -> None:
    """`-2`, `-1` and `0` all read as "not in arrears"; none of them outranks another.

    ADR-0004 measured that the numeric order of these codes is not an order of severity, so
    the maximum is taken only over the region where the code is a delay count.
    """
    out = features(
        dict.fromkeys(schema.PAY_STATUS_HOMOGENEOUS_COLUMNS, -2),
        dict.fromkeys(schema.PAY_STATUS_HOMOGENEOUS_COLUMNS, 0),
        {
            "PAY_STATUS_2": -2,
            "PAY_STATUS_3": -1,
            "PAY_STATUS_4": 0,
            "PAY_STATUS_5": -1,
            "PAY_STATUS_6": -2,
        },
    )
    assert (out["MAX_DELINQUENCY_M2_M6"] == 0).all()


def test_max_delinquency_is_the_worst_delay_anywhere_in_the_block() -> None:
    out = features(
        {
            "PAY_STATUS_2": 1,
            "PAY_STATUS_3": 0,
            "PAY_STATUS_4": 8,
            "PAY_STATUS_5": -2,
            "PAY_STATUS_6": 2,
        }
    )
    assert out.loc[0, "MAX_DELINQUENCY_M2_M6"] == 8


@pytest.mark.parametrize(("code", "expected"), [(-2, 0), (-1, 0), (0, 0), (1, 1), (2, 1), (8, 1)])
def test_month_one_delinquency_indicator_uses_the_adr_threshold(code: int, expected: int) -> None:
    out = features({"PAY_STATUS_1": code})
    assert out.loc[0, "IS_DELINQUENT_MOST_RECENT_M1"] == expected


# ---------------------------------------------------------------------------
# Volatility and months without payment
# ---------------------------------------------------------------------------


def test_bill_volatility_is_the_population_standard_deviation_of_the_block() -> None:
    """Balances 100, 0, 0, 0, 0: mean 20, variance 8000/5 = 1600, deviation exactly 40."""
    out = features(
        {
            "BILL_AMT2": 100,
            "BILL_AMT3": 0,
            "BILL_AMT4": 0,
            "BILL_AMT5": 0,
            "BILL_AMT6": 0,
        }
    )
    assert out.loc[0, "BILL_VOLATILITY_M2_M6"] == pytest.approx(40.0)


def test_bill_volatility_is_zero_for_a_perfectly_stable_balance() -> None:
    out = features(dict.fromkeys(schema.BILL_AMOUNT_COLUMNS, 5_000))
    assert out.loc[0, "BILL_VOLATILITY_M2_M6"] == pytest.approx(0.0)


def test_months_without_payment_counts_the_zero_payments_of_the_block_only() -> None:
    """`PAY_AMT1` is zero here too and must not be counted: it is outside the block."""
    out = features(
        {
            "PAY_AMT1": 0,
            "PAY_AMT2": 0,
            "PAY_AMT3": 0,
            "PAY_AMT4": 5,
            "PAY_AMT5": 0,
            "PAY_AMT6": 7,
        }
    )
    assert out.loc[0, "MONTHS_WITHOUT_PAYMENT_M2_M6"] == 3


# ---------------------------------------------------------------------------
# The isolation of month 1 - structural, then behavioural
# ---------------------------------------------------------------------------


def test_the_two_source_groups_share_only_the_columns_that_have_no_month() -> None:
    shared = set(builder.BLOCK_SOURCE_COLUMNS) & set(builder.ISOLATED_SOURCE_COLUMNS)
    assert shared == {"LIMIT_BAL"}
    assert not set(builder.BLOCK_SOURCE_COLUMNS) & set(MONTH_ONE_SOURCE_COLUMNS)
    assert not set(builder.ISOLATED_SOURCE_COLUMNS) & set(BLOCK_ONLY_SOURCE_COLUMNS)


def test_block_features_are_computable_without_the_month_one_columns() -> None:
    """Structural half: the block builder is handed a frame where month 1 does not exist.

    A block feature that reached for `PAY_STATUS_1`, `BILL_AMT1` or `PAY_AMT1` would raise
    a lookup error here instead of silently averaging two scales together.
    """
    frame = make_frame({"BILL_AMT2": 500, "PAY_AMT2": 250, "PAY_STATUS_2": 2})
    stripped = frame.drop(columns=list(MONTH_ONE_SOURCE_COLUMNS))
    assert not set(MONTH_ONE_SOURCE_COLUMNS) & set(stripped.columns)

    produced = builder._block_features(stripped[list(builder.BLOCK_SOURCE_COLUMNS)])
    assert set(produced) == set(BLOCK_FEATURES) - {"UTILIZATION_NOT_COMPUTABLE"}


def _perturb(
    frame: pd.DataFrame, columns: Sequence[str], draw: np.random.Generator
) -> pd.DataFrame:
    """Replace the given columns with plausible but arbitrary values."""
    noisy = frame.copy()
    for column in columns:
        if column.startswith("PAY_STATUS"):
            noisy[column] = draw.integers(-2, 9, size=len(frame))
        else:
            noisy[column] = draw.integers(-50_000, 500_000, size=len(frame))
    return noisy


def test_block_features_are_invariant_to_month_one() -> None:
    """Behavioural half: only month 1 moves, and every block column must stay identical.

    The generator is seeded from the project's `RANDOM_STATE` so a failure is reproducible
    rather than a story about a run that once went red.
    """
    transformer = PaymentBehaviourFeatures()
    base = make_frame(*({"AGE": 25 + step} for step in range(20)))
    baseline = transformer.fit_transform(base)[list(BLOCK_FEATURES)]

    draw = np.random.default_rng(RANDOM_STATE)
    for _ in range(25):
        perturbed = transformer.fit_transform(_perturb(base, MONTH_ONE_SOURCE_COLUMNS, draw))
        pd.testing.assert_frame_equal(perturbed[list(BLOCK_FEATURES)], baseline)


def test_month_one_features_are_invariant_to_the_block() -> None:
    transformer = PaymentBehaviourFeatures()
    base = make_frame(*({"AGE": 25 + step} for step in range(20)))
    baseline = transformer.fit_transform(base)[list(MONTH_ONE_FEATURES)]

    draw = np.random.default_rng(RANDOM_STATE)
    for _ in range(25):
        perturbed = transformer.fit_transform(_perturb(base, BLOCK_ONLY_SOURCE_COLUMNS, draw))
        pd.testing.assert_frame_equal(perturbed[list(MONTH_ONE_FEATURES)], baseline)


def test_no_feature_name_claims_a_span_that_crosses_the_boundary() -> None:
    """A block feature naming month 1, or a month-1 feature naming the block, is a bug."""
    for name in BLOCK_FEATURES:
        assert "M1" not in name.removeprefix("UTILIZATION_NOT_COMPUTABLE"), name
    for name in MONTH_ONE_FEATURES:
        assert name.endswith("M1"), name


# ---------------------------------------------------------------------------
# The target is never seen
# ---------------------------------------------------------------------------


def test_the_target_is_not_among_the_columns_the_transformer_reads() -> None:
    assert schema.TARGET_COLUMN not in builder.REQUIRED_COLUMNS


def test_the_output_does_not_move_when_the_target_is_permuted_or_removed() -> None:
    transformer = PaymentBehaviourFeatures()
    base = make_frame(*({"AGE": 25 + step, "BILL_AMT2": 100 * step} for step in range(10)))

    with_target = transformer.fit_transform(base)
    flipped = base.assign(**{schema.TARGET_COLUMN: 1 - base[schema.TARGET_COLUMN]})
    without_target = base.drop(columns=[schema.TARGET_COLUMN])

    pd.testing.assert_frame_equal(transformer.fit_transform(flipped), with_target)
    pd.testing.assert_frame_equal(transformer.fit_transform(without_target), with_target)


def test_fit_stores_no_statistic_of_the_training_data() -> None:
    """Nothing carried out of `fit` means nothing can leak from one fold into another."""
    transformer = PaymentBehaviourFeatures().fit(make_frame({"BILL_AMT2": 999_999}))
    assert set(vars(transformer)) == {"n_features_in_", "feature_names_in_"}


def test_what_fit_saw_does_not_change_what_transform_produces() -> None:
    target = make_frame({"BILL_AMT2": 400, "PAY_AMT2": 100})
    fitted_on_one = PaymentBehaviourFeatures().fit(make_frame({"LIMIT_BAL": 1}))
    fitted_on_other = PaymentBehaviourFeatures().fit(make_frame({"LIMIT_BAL": 900_000}))
    pd.testing.assert_frame_equal(
        fitted_on_one.transform(target), fitted_on_other.transform(target)
    )


# ---------------------------------------------------------------------------
# Contract: names, serialisation, pipeline, failure modes
# ---------------------------------------------------------------------------


def test_get_feature_names_out_matches_the_columns_transform_produces() -> None:
    transformer = PaymentBehaviourFeatures()
    out = transformer.fit_transform(make_frame())
    assert list(transformer.get_feature_names_out()) == list(out.columns)
    assert list(out.columns) == list(FEATURE_NAMES)


def test_the_declared_feature_names_are_unique_and_complete() -> None:
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert len(FEATURE_NAMES) == 21


def test_a_joblib_round_trip_produces_the_same_frame(tmp_path: Path) -> None:
    """Serialisation is exercised through the disk, not through a copy kept in memory."""
    sample = make_frame(
        {"BILL_AMT2": 400, "PAY_AMT2": 100, "PAY_STATUS_2": 3},
        {"LIMIT_BAL": 0, "BILL_AMT3": -20},
    )
    transformer = PaymentBehaviourFeatures().fit(sample)
    before = transformer.transform(sample)

    artefact = tmp_path / "payment_behaviour_features.joblib"
    joblib.dump(transformer, artefact)
    restored = joblib.load(artefact)

    pd.testing.assert_frame_equal(restored.transform(sample), before)
    assert list(restored.get_feature_names_out()) == list(FEATURE_NAMES)


def test_the_transformer_runs_inside_a_sklearn_pipeline() -> None:
    """The reason this is an estimator at all: it has to be a step in the shared pipeline."""
    frame = make_frame({"BILL_AMT2": 400}, {"BILL_AMT2": 800})
    pipeline = Pipeline([("payment_behaviour", PaymentBehaviourFeatures())])
    out = pipeline.fit_transform(frame, frame[schema.TARGET_COLUMN])
    assert list(out.columns) == list(FEATURE_NAMES)
    assert len(out) == 2


def test_the_transformer_can_be_cloned() -> None:
    clone(PaymentBehaviourFeatures())


def test_the_index_of_the_input_is_preserved() -> None:
    frame = make_frame({}, {}, index=[17, 42])
    out = PaymentBehaviourFeatures().fit_transform(frame)
    assert list(out.index) == [17, 42]


def test_transform_does_not_mutate_its_input() -> None:
    frame = make_frame({"LIMIT_BAL": 0, "BILL_AMT3": -1})
    untouched = frame.copy()
    PaymentBehaviourFeatures().fit_transform(frame)
    pd.testing.assert_frame_equal(frame, untouched)


def test_a_missing_source_column_fails_loudly_and_names_all_of_them() -> None:
    frame = make_frame().drop(columns=["PAY_AMT3", "BILL_AMT5"])
    with pytest.raises(MissingSourceColumnsError) as failure:
        PaymentBehaviourFeatures().fit(frame)
    assert "PAY_AMT3" in str(failure.value)
    assert "BILL_AMT5" in str(failure.value)


def test_an_input_without_column_names_fails_loudly() -> None:
    with pytest.raises(TypeError):
        PaymentBehaviourFeatures().fit(np.zeros((3, 24)))


def test_extra_columns_and_a_different_column_order_are_accepted() -> None:
    frame = make_frame({"BILL_AMT2": 400})
    shuffled = frame[sorted(frame.columns)].assign(SOMETHING_ELSE=1)
    pd.testing.assert_frame_equal(
        PaymentBehaviourFeatures().fit_transform(shuffled),
        PaymentBehaviourFeatures().fit_transform(frame),
    )
