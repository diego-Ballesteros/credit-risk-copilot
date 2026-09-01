"""The split that the main hypothesis is measured with.

Entry 003 of `docs/EVALUATION.md` contrasts a demography-only model against a
behaviour-only one and reports **12 demographic and 98 behavioural columns, disjoint and
covering the 110**. If that split silently stopped covering the matrix, the contrast would
compare fewer columns than it claims to and the headline result of the project would be
wrong with nothing failing. These tests are what makes that impossible.
"""

import numpy as np
import pandas as pd
import pytest

from credit_copilot.data.preprocessor import build_preprocessor
from credit_copilot.models.feature_groups import (
    ALL_SOURCE_COLUMNS,
    BEHAVIOURAL_SOURCE_COLUMNS,
    DEMOGRAPHIC_SOURCE_COLUMNS,
    FeatureGroupError,
    SelectFeatureGroup,
    _resolve_owner,
    group_columns_by_source,
)
from tests.test_preprocessor import varied_frame

# ---------------------------------------------------------------------------
# The two groups, as declared
# ---------------------------------------------------------------------------


def test_the_two_groups_are_disjoint() -> None:
    """A column in both would be counted twice and neither model would be what it says."""
    assert not set(DEMOGRAPHIC_SOURCE_COLUMNS) & set(BEHAVIOURAL_SOURCE_COLUMNS)


def test_the_two_groups_cover_every_declared_source() -> None:
    """`ALL_SOURCE_COLUMNS` must be exactly the union, not a third hand-kept list."""
    assert set(ALL_SOURCE_COLUMNS) == set(DEMOGRAPHIC_SOURCE_COLUMNS) | set(
        BEHAVIOURAL_SOURCE_COLUMNS
    )


def test_the_granted_limit_is_counted_as_demographic() -> None:
    """`LIMIT_BAL` is an attribute of the account as underwritten, before any conduct.

    The choice is declared rather than obvious, and entry 003 states the caveat it creates:
    the utilisation ratios divide by it, so the two groups split the *columns* without
    perfectly splitting the *information*.
    """
    assert "LIMIT_BAL" in DEMOGRAPHIC_SOURCE_COLUMNS
    assert "LIMIT_BAL" not in BEHAVIOURAL_SOURCE_COLUMNS


# ---------------------------------------------------------------------------
# Tracing a matrix column back to its source
# ---------------------------------------------------------------------------


def test_a_column_that_is_its_own_source_resolves_to_itself() -> None:
    """A passthrough or scaled numeric column keeps the source name."""
    assert _resolve_owner("AGE", ALL_SOURCE_COLUMNS) == "AGE"


def test_a_one_hot_column_resolves_to_the_column_it_expands() -> None:
    """`EDUCATION_2` is one of the four levels `EDUCATION` expands into."""
    assert _resolve_owner("EDUCATION_2", ALL_SOURCE_COLUMNS) == "EDUCATION"


def test_a_repayment_code_with_a_negative_level_still_resolves() -> None:
    """The one-hot vocabulary includes `-2` and `-1`, so the suffix carries a minus sign."""
    assert _resolve_owner("PAY_STATUS_1_-2", ALL_SOURCE_COLUMNS) == "PAY_STATUS_1"


def test_an_untraceable_column_is_refused_by_name() -> None:
    """A feature added to the pipeline and to neither group must fail loudly here.

    Silently dropping it is what would make the contrast compare fewer columns than it
    reports, which is the failure this module exists to prevent.
    """
    with pytest.raises(FeatureGroupError) as error:
        _resolve_owner("SOMETHING_NEW", ALL_SOURCE_COLUMNS)

    assert "SOMETHING_NEW" in str(error.value)


def test_an_ambiguous_column_is_refused_rather_than_assigned_arbitrarily() -> None:
    """Two sources could claim it, so the naming convention is no longer well defined.

    Picking the first match would put the column in whichever group the declaration order
    happened to favour, which is a coin toss dressed as a rule.
    """
    with pytest.raises(FeatureGroupError) as error:
        _resolve_owner("A_B_1", ["A", "A_B"])

    assert "A_B_1" in str(error.value)


def test_grouping_collects_every_matrix_column_under_its_source() -> None:
    """The map is what lets a caller ask "which columns did EDUCATION produce?"."""
    grouped = group_columns_by_source(["AGE", "EDUCATION_1", "EDUCATION_2", "SEX_1"])

    assert grouped == {
        "AGE": ["AGE"],
        "EDUCATION": ["EDUCATION_1", "EDUCATION_2"],
        "SEX": ["SEX_1"],
    }


# ---------------------------------------------------------------------------
# SelectFeatureGroup
# ---------------------------------------------------------------------------


def test_a_numpy_array_is_refused_because_the_step_addresses_columns_by_name() -> None:
    """Losing `set_output('pandas')` upstream must fail here rather than select by position."""
    selector = SelectFeatureGroup(DEMOGRAPHIC_SOURCE_COLUMNS)

    with pytest.raises(TypeError) as error:
        selector.fit(np.zeros((3, 4)))  # type: ignore[arg-type]

    assert "set_output" in str(error.value)


def test_a_group_that_matches_nothing_is_refused() -> None:
    """A model with no features cannot be measured, and would otherwise train on an empty
    matrix and report a number."""
    frame = pd.DataFrame({"AGE": [30, 40], "SEX_1": [1, 0]})

    with pytest.raises(FeatureGroupError) as error:
        SelectFeatureGroup(["PAY_AMT1"]).fit(frame)

    assert "no features cannot be measured" in str(error.value)


def test_selection_keeps_only_the_declared_group_and_preserves_order() -> None:
    """Reordering columns between fit and transform would silently rewire the model."""
    frame = pd.DataFrame({"AGE": [30], "PAY_AMT1": [100], "SEX_1": [1], "BILL_AMT1": [7]})

    kept = SelectFeatureGroup(DEMOGRAPHIC_SOURCE_COLUMNS).fit(frame).transform(frame)

    assert list(kept.columns) == ["AGE", "SEX_1"]


def test_an_untraceable_column_anywhere_in_the_matrix_fails_the_fit() -> None:
    """Every column is traced, not only the ones being kept.

    Tracing all of them is what turns "the two groups cover the matrix" from a comment into
    a checked invariant that fires on the fold where it stops being true.
    """
    frame = pd.DataFrame({"AGE": [30], "MYSTERY_COLUMN": [1]})

    with pytest.raises(FeatureGroupError) as error:
        SelectFeatureGroup(DEMOGRAPHIC_SOURCE_COLUMNS).fit(frame)

    assert "MYSTERY_COLUMN" in str(error.value)


# ---------------------------------------------------------------------------
# The invariant behind the headline result
# ---------------------------------------------------------------------------


def test_the_two_groups_partition_the_real_matrix_exactly() -> None:
    """Against the real preprocessor, not a hand-written column list.

    The counts are the ones entry 003 of `docs/EVALUATION.md` reports for the full dataset.
    This frame is a small synthetic one, so the totals differ - the one-hot vocabulary
    depends on which repayment codes appear. What must hold at any size is that the two
    selections are disjoint and together account for every column.
    """
    frame = varied_frame(80)
    matrix = build_preprocessor().fit_transform(frame)

    demographic = SelectFeatureGroup(DEMOGRAPHIC_SOURCE_COLUMNS).fit(matrix).transform(matrix)
    behavioural = SelectFeatureGroup(BEHAVIOURAL_SOURCE_COLUMNS).fit(matrix).transform(matrix)

    assert not set(demographic.columns) & set(behavioural.columns)
    assert set(demographic.columns) | set(behavioural.columns) == set(matrix.columns)
    assert demographic.shape[1] + behavioural.shape[1] == matrix.shape[1]
