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
    """Build a three-row working table that honours the contract in every respect.

    The baseline carries the 24 columns `loader.load_dataset` returns, identifier
    excluded, because that is the table the project actually validates end to end. The
    two tests that exercise the identifier branch add it back through
    `_frame_with_identifier`.

    Values are written out rather than derived from `schema`, so a defect introduced in
    the contract cannot travel into the fixture and cancel itself out.

    Returns:
        A 3 x 24 table with canonical column names and `int64` throughout.
    """
    columns: dict[str, list[int]] = {
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
    return frame[list(schema.WORKING_COLUMNS)]


def _frame_with_identifier() -> pd.DataFrame:
    """Build the same clean table with `ID` restored, as the raw file carries it.

    The validator has to be right about both shapes: the raw-renamed table, where a
    repeated identifier means a broken extract, and the working table, where the check
    cannot be run at all. This fixture covers the first.

    Returns:
        A 3 x 25 table with canonical column names, in the file's column order.
    """
    frame = _clean_frame()
    frame[schema.ID_COLUMN] = [1, 2, 3]
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
    """The baseline working table passes, so any finding below is the injected one.

    One informative finding is expected and only one: without `ID` the exact-duplicate
    check cannot run, and the validator says so instead of staying quiet. Nothing blocks.
    """
    result = validate_dataframe(_clean_frame())
    assert result.blocking == (), f"Baseline blocks: {[i.check for i in result.blocking]}"
    assert [issue.check for issue in result.informative] == ["exact_duplicate_check_skipped"]
    assert result.is_valid
    assert result.n_rows == 3
    assert result.n_columns == 24


def test_clean_frame_with_the_identifier_also_produces_no_findings() -> None:
    """The raw-renamed shape validates too: `ID` is allowed, just not required."""
    result = validate_dataframe(_frame_with_identifier())
    assert result.issues == (), f"Baseline is not clean: {[i.check for i in result.issues]}"
    assert result.n_columns == 25


def test_null_counts_are_reported_for_every_column() -> None:
    """Zero nulls is a measurement that gets shown, not an assumption."""
    result = validate_dataframe(_clean_frame())
    assert set(result.null_counts) == set(schema.WORKING_COLUMNS)
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
    """A code in neither contract map is blocking and reports its frequency.

    `7` is chosen deliberately: it is undocumented by UCI *and* absent from
    `OBSERVED_CODES_ACCEPTED`, which is the combination that still has to stop the run.
    """
    frame = _clean_frame()
    frame.loc[0, "EDUCATION"] = 7

    assert _blocking_checks(frame) == {"unknown_category"}
    issue = validate_dataframe(frame).blocking[0]
    assert issue.column == "EDUCATION"
    assert dict(issue.counts) == {"7": 1}


def test_a_code_accepted_by_an_adr_is_informative_and_not_blocking() -> None:
    """An undocumented code an ADR accepted stops blocking but does not go quiet.

    The finding names the ADR, so the reader of a run can reach the evidence without
    knowing beforehand that a decision exists.
    """
    frame = _clean_frame()
    frame.loc[0, "EDUCATION"] = 5

    result = validate_dataframe(frame)
    assert result.is_valid
    assert result.blocking == ()

    accepted = [
        issue for issue in result.informative if issue.check == "accepted_undocumented_category"
    ]
    assert len(accepted) == 1
    assert accepted[0].column == "EDUCATION"
    assert dict(accepted[0].counts) == {"5": 1}
    assert schema.ADR_UNDOCUMENTED_CODES in accepted[0].message


def test_accepted_and_unknown_codes_in_one_column_are_reported_separately() -> None:
    """One accepted code and one unknown code produce two findings, not one verdict.

    Collapsing them would let an accepted code launder an unknown one through the same
    column, which is the exact failure the two-map split exists to prevent.
    """
    frame = _clean_frame()
    frame.loc[0, "MARRIAGE"] = 0
    frame.loc[1, "MARRIAGE"] = 9

    result = validate_dataframe(frame)
    assert not result.is_valid

    blocking = [issue for issue in result.blocking if issue.column == "MARRIAGE"]
    assert len(blocking) == 1
    assert dict(blocking[0].counts) == {"9": 1}

    informative = [
        issue
        for issue in result.informative
        if issue.check == "accepted_undocumented_category" and issue.column == "MARRIAGE"
    ]
    assert len(informative) == 1
    assert dict(informative[0].counts) == {"0": 1}


def test_the_repayment_status_codes_the_dataset_is_made_of_do_not_block() -> None:
    """-2 and 0 cover most of the real dataset; after ADR-0004 they must not stop a run."""
    frame = _clean_frame()
    for month in range(1, 7):
        frame.loc[0, f"PAY_STATUS_{month}"] = -2
        frame.loc[1, f"PAY_STATUS_{month}"] = 0

    result = validate_dataframe(frame)
    assert result.is_valid
    accepted = [
        issue.column
        for issue in result.informative
        if issue.check == "accepted_undocumented_category"
    ]
    assert accepted == list(schema.PAY_STATUS_COLUMNS)


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
    base = _frame_with_identifier()
    frame = pd.concat([base, base.iloc[[0]]], ignore_index=True)

    assert _blocking_checks(frame) == {"duplicate_rows"}
    issue = validate_dataframe(frame).blocking[0]
    assert dict(issue.counts) == {"extra_copies": 1}


def test_a_repeated_row_without_the_identifier_is_informative_not_blocking() -> None:
    """Without `ID` a repeated row cannot be called a broken extract, so it is not.

    This is the shape the project actually works with, and it is why dropping the
    identifier does not turn the 35 look-alike rows of the real dataset into 35 blockers:
    with no identifier to be repeated, the blocking condition cannot even be stated.
    """
    base = _clean_frame()
    frame = pd.concat([base, base.iloc[[0]]], ignore_index=True)

    result = validate_dataframe(frame)
    assert result.is_valid
    assert result.blocking == ()

    checks = [issue.check for issue in result.informative]
    assert "duplicate_rows_ignoring_id" in checks
    assert "exact_duplicate_check_skipped" in checks
    duplicates = next(i for i in result.informative if i.check == "duplicate_rows_ignoring_id")
    assert dict(duplicates.counts) == {"extra_copies": 1}


# ---------------------------------------------------------------------------
# Severity boundary and raising behaviour
# ---------------------------------------------------------------------------


def test_rows_differing_only_by_identifier_are_informative_not_blocking() -> None:
    """Two clients with the same attributes are ordinary; the count is still reported."""
    base = _frame_with_identifier()
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
    assert "range_check_skipped" in [issue.check for issue in result.informative]


def test_accumulates_every_problem_instead_of_stopping_at_the_first() -> None:
    """Three unrelated defects come back in one run, not one run at a time.

    One numeric column and one categorical column are dropped, so the checks that iterate
    the contract have to tolerate an absent column instead of raising on it.
    """
    frame = _clean_frame().drop(columns=["AGE", "SEX"])
    frame.loc[0, "MARRIAGE"] = 9
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
    """A clean table passes through and returns the same complete result.

    Informative findings do not stop it; only blocking ones do.
    """
    result = validate_or_raise(_clean_frame())
    assert result.is_valid
    assert result.blocking == ()


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
    frame.loc[0, "MARRIAGE"] = 9
    assert "FAIL" in validate_dataframe(frame).report()
