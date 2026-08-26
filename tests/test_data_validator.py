"""Tests of the validator, on tables built by hand.

The real dataset is deliberately not used here. It already breaks the contract in several
places, so a test written against it could not tell "the validator found the undocumented
education codes" apart from "the validator found something". Each test starts from a
table that produces zero findings, injects exactly one defect, and asserts that exactly
that defect comes back.
"""

import pandas as pd
import pytest

from credit_copilot.data import schema
from credit_copilot.data.validator import (
    DataContractError,
    Severity,
    validate_dataframe,
    validate_or_raise,
)


def _clean_frame() -> pd.DataFrame:
    """Build a three-row table that honours the contract in every respect.

    Values are written out rather than derived from `schema`, so a defect introduced in
    the contract cannot travel into the fixture and cancel itself out.

    Returns:
        A 3 x 25 table with canonical column names and `int64` throughout.
    """
    columns: dict[str, list[int]] = {
        "ID": [1, 2, 3],
        "LIMIT_BAL": [20_000, 120_000, 90_000],
        "SEX": [1, 2, 2],
        "EDUCATION": [1, 2, 3],
        "MARRIAGE": [1, 2, 3],
        "AGE": [24, 35, 51],
    }
    for month in range(1, 7):
        columns[f"PAY_STATUS_{month}"] = [2, -1, 1]
        columns[f"BILL_AMT{month}"] = [3_913, -1_200, 50_000]
        columns[f"PAY_AMT{month}"] = [0, 689, 1_500]
    columns[schema.TARGET_COLUMN] = [1, 0, 0]

    frame = pd.DataFrame(columns).astype("int64")
    return frame[list(schema.CANONICAL_COLUMNS)]


def _blocking_checks(frame: pd.DataFrame) -> set[str]:
    """Names of the checks that produced a blocking finding.

    Args:
        frame: Table to validate.

    Returns:
        The set of blocking check names, so a test can assert that one defect produced
        one kind of finding and nothing else.
    """
    return {issue.check for issue in validate_dataframe(frame).blocking}


# ---------------------------------------------------------------------------
# The fixture itself must be clean, or every assertion below is unattributable
# ---------------------------------------------------------------------------


def test_clean_frame_produces_no_findings() -> None:
    """The baseline table passes, so any finding in the tests below is the injected one."""
    result = validate_dataframe(_clean_frame())
    assert result.issues == (), f"Baseline is not clean: {[i.check for i in result.issues]}"
    assert result.is_valid
    assert result.n_rows == 3
    assert result.n_columns == 25


def test_null_counts_are_reported_for_every_column() -> None:
    """Zero nulls is a measurement that gets shown, not an assumption."""
    result = validate_dataframe(_clean_frame())
    assert set(result.null_counts) == set(schema.CANONICAL_COLUMNS)
    assert sum(result.null_counts.values()) == 0


# ---------------------------------------------------------------------------
# One defect per test
# ---------------------------------------------------------------------------


def test_detects_a_missing_column() -> None:
    """A contract column absent from the table is blocking and names itself."""
    frame = _clean_frame().drop(columns=["AGE"])

    assert _blocking_checks(frame) == {"missing_columns"}
    issue = validate_dataframe(frame).blocking[0]
    assert "AGE" in issue.message
    assert issue.severity is Severity.BLOCKING


def test_detects_an_unknown_category() -> None:
    """A value outside the levels UCI documents is blocking and reports its frequency."""
    frame = _clean_frame()
    frame.loc[0, "EDUCATION"] = 0

    assert _blocking_checks(frame) == {"unknown_category"}
    issue = validate_dataframe(frame).blocking[0]
    assert issue.column == "EDUCATION"
    assert dict(issue.counts) == {"0": 1}


def test_detects_a_value_out_of_range() -> None:
    """A value below the plausible minimum is blocking and says which side it fell off."""
    frame = _clean_frame()
    frame.loc[0, "AGE"] = 7

    assert _blocking_checks(frame) == {"out_of_range"}
    issue = validate_dataframe(frame).blocking[0]
    assert issue.column == "AGE"
    assert dict(issue.counts) == {"below_minimum": 1, "above_maximum": 0}


def test_detects_a_duplicated_row() -> None:
    """A row repeated across every column, identifier included, is blocking."""
    base = _clean_frame()
    frame = pd.concat([base, base.iloc[[0]]], ignore_index=True)

    assert _blocking_checks(frame) == {"duplicate_rows"}
    issue = validate_dataframe(frame).blocking[0]
    assert dict(issue.counts) == {"extra_copies": 1}


# ---------------------------------------------------------------------------
# Severity boundary and raising behaviour
# ---------------------------------------------------------------------------


def test_rows_differing_only_by_identifier_are_informative_not_blocking() -> None:
    """Two clients with the same attributes are ordinary; the count is still reported."""
    base = _clean_frame()
    twin = base.iloc[[0]].copy()
    twin.loc[:, "ID"] = 99
    frame = pd.concat([base, twin], ignore_index=True)

    result = validate_dataframe(frame)
    assert result.is_valid
    assert [issue.check for issue in result.informative] == ["duplicate_rows_ignoring_id"]


def test_detects_an_unexpected_column() -> None:
    """A column the contract does not know is blocking, not ignored.

    It breaks no arithmetic, but it has the shape of a leakage vector, and the project
    makes leakage impossible rather than remembering not to cause it.
    """
    frame = _clean_frame()
    frame["REPAID_IN_FULL_NEXT_MONTH"] = [0, 1, 1]

    assert _blocking_checks(frame) == {"unexpected_columns"}
    issue = validate_dataframe(frame).blocking[0]
    assert "REPAID_IN_FULL_NEXT_MONTH" in issue.message


def test_range_check_is_skipped_loudly_when_the_dtype_is_not_numeric() -> None:
    """A check that cannot run says so; it never passes by staying quiet."""
    frame = _clean_frame()
    frame["AGE"] = frame["AGE"].astype(str)

    result = validate_dataframe(frame)
    assert _blocking_checks(frame) == {"dtype_mismatch"}
    assert [issue.check for issue in result.informative] == ["range_check_skipped"]


def test_accumulates_every_problem_instead_of_stopping_at_the_first() -> None:
    """Three unrelated defects come back in one run, not one run at a time.

    One numeric column and one categorical column are dropped, so the checks that iterate
    the contract have to tolerate an absent column instead of raising on it.
    """
    frame = _clean_frame().drop(columns=["AGE", "SEX"])
    frame.loc[0, "MARRIAGE"] = 0
    frame.loc[1, "LIMIT_BAL"] = -5

    assert _blocking_checks(frame) == {"missing_columns", "unknown_category", "out_of_range"}


def test_validate_or_raise_raises_on_a_blocking_finding() -> None:
    """The raising entry point carries the full report, not just the first problem."""
    frame = _clean_frame()
    frame.loc[0, "SEX"] = 9

    with pytest.raises(DataContractError) as excinfo:
        validate_or_raise(frame)
    assert not excinfo.value.result.is_valid
    assert "SEX" in str(excinfo.value)


def test_validate_or_raise_returns_the_result_when_the_contract_holds() -> None:
    """A clean table passes through and returns the same complete result."""
    result = validate_or_raise(_clean_frame())
    assert result.is_valid
    assert result.issues == ()


def test_validator_does_not_modify_the_frame() -> None:
    """It reports and nothing else: no imputation, no clipping, no dropping."""
    frame = _clean_frame()
    frame.loc[0, "EDUCATION"] = 0
    frame.loc[1, "AGE"] = 500
    before = frame.copy(deep=True)

    validate_dataframe(frame)

    pd.testing.assert_frame_equal(frame, before)


def test_report_renders_the_verdict_in_both_directions() -> None:
    """The console report states which way it went, so nobody has to infer it."""
    assert "PASS" in validate_dataframe(_clean_frame()).report()

    frame = _clean_frame()
    frame.loc[0, "MARRIAGE"] = 0
    assert "FAIL" in validate_dataframe(frame).report()
