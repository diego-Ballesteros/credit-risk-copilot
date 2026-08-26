"""Tests of the declarative contract itself, with no data involved.

These check properties of the maps in `schema.py`. They are cheap and they catch the
whole class of bug where a column is added to one map and forgotten in another - the
"two-step process where the second step gets skipped" failure, applied to the contract.
"""

import pytest

from credit_copilot.data import schema


def test_rename_map_covers_every_column_the_source_delivers() -> None:
    """The file has 25 columns: the identifier, 23 predictors and the target.

    ADR-0001 and the roadmap both say "24 columns". They count 23 predictors plus the
    target and leave out `ID`, which is nonetheless a column in the delivered file. The
    contract has to describe the file, so it carries 25 entries. The assertion is written
    as 1 + 23 + 1 rather than as a bare 25 so that the arithmetic is visible.
    """
    assert len(schema.RAW_TO_CANONICAL) == 25
    assert len(schema.CANONICAL_COLUMNS) == 25

    predictors = set(schema.CANONICAL_COLUMNS) - {schema.ID_COLUMN, schema.TARGET_COLUMN}
    assert len(predictors) == 23


def test_rename_map_is_injective() -> None:
    """No two raw columns collapse onto the same canonical name."""
    canonical = list(schema.RAW_TO_CANONICAL.values())
    assert len(set(canonical)) == len(canonical), "Two raw columns map to the same canonical name"


def test_rename_map_is_bijective() -> None:
    """The map is invertible: inverting it twice returns the original map."""
    inverse = {canonical: raw for raw, canonical in schema.RAW_TO_CANONICAL.items()}
    assert len(inverse) == len(schema.RAW_TO_CANONICAL)
    assert {canonical: raw for raw, canonical in inverse.items()} == dict(schema.RAW_TO_CANONICAL)


def test_expected_dtypes_keys_match_canonical_names() -> None:
    """`EXPECTED_DTYPES` is keyed by canonical names, not raw ones, and covers them all."""
    assert set(schema.EXPECTED_DTYPES) == set(schema.RAW_TO_CANONICAL.values())


def test_pay_status_block_is_iterable_without_special_cases() -> None:
    """The renaming exists to make this loop possible; the source names break it."""
    assert len(schema.PAY_STATUS_COLUMNS) == 6
    for month, column in enumerate(schema.PAY_STATUS_COLUMNS, start=1):
        assert column == f"PAY_STATUS_{month}"
        assert column in schema.EXPECTED_DTYPES


def test_pay_status_renaming_preserves_the_month_alignment() -> None:
    """PAY_0 is September 2005, the same month as BILL_AMT1 and PAY_AMT1."""
    assert schema.RAW_TO_CANONICAL["PAY_0"] == "PAY_STATUS_1"
    assert schema.RAW_TO_CANONICAL["BILL_AMT1"] == "BILL_AMT1"
    assert schema.RAW_TO_CANONICAL["PAY_AMT1"] == "PAY_AMT1"


def test_canonical_names_are_valid_identifiers() -> None:
    """Nothing canonical carries a space, so every column can be an attribute."""
    invalid = [name for name in schema.CANONICAL_COLUMNS if not name.isidentifier()]
    assert not invalid, f"Canonical names that are not valid identifiers: {invalid}"


def test_target_and_id_are_canonical_columns() -> None:
    """The named special columns exist in the contract."""
    assert schema.TARGET_COLUMN in schema.CANONICAL_COLUMNS
    assert schema.ID_COLUMN in schema.CANONICAL_COLUMNS


def test_uci_legend_matches_the_raw_names() -> None:
    """The frozen UCI legend produces exactly the raw names the rename map consumes."""
    assert set(schema.UCI_CODE_TO_RAW.values()) == set(schema.RAW_TO_CANONICAL)


def test_categorical_and_numeric_maps_partition_the_columns() -> None:
    """Every column is covered by exactly one of the two value contracts."""
    categorical = set(schema.CATEGORICAL_LEVELS)
    numeric = set(schema.NUMERIC_RANGES)
    assert not categorical & numeric, f"Columns in both maps: {sorted(categorical & numeric)}"
    uncovered = set(schema.CANONICAL_COLUMNS) - categorical - numeric
    assert not uncovered, f"Columns with no value contract: {sorted(uncovered)}"


def test_undocumented_codes_are_absent_from_categorical_levels() -> None:
    """The levels map states what UCI documents, never what the data happens to contain.

    0 in EDUCATION and MARRIAGE and 0 and -2 in the repayment-status block are real values
    in the file and undocumented at the source. Adding them here would silence the
    validator instead of answering the question.
    """
    assert 0 not in schema.CATEGORICAL_LEVELS["EDUCATION"]
    assert 0 not in schema.CATEGORICAL_LEVELS["MARRIAGE"]
    for column in schema.PAY_STATUS_COLUMNS:
        assert 0 not in schema.CATEGORICAL_LEVELS[column]
        assert -2 not in schema.CATEGORICAL_LEVELS[column]


def test_bill_amount_ranges_admit_negative_values() -> None:
    """Overpayments leave the statement in credit; a floor of zero would reject them."""
    for column in schema.BILL_AMOUNT_COLUMNS:
        minimum = schema.NUMERIC_RANGES[column].minimum
        assert minimum is not None and minimum < 0


def test_payment_amount_ranges_do_not_admit_negative_values() -> None:
    """Money returned to a client is a credit on the statement, never a negative payment."""
    for column in schema.PAY_AMOUNT_COLUMNS:
        assert schema.NUMERIC_RANGES[column].minimum == 0


def test_every_numeric_range_documents_its_bounds() -> None:
    """A bound with no stated reason cannot be argued with, so it is not allowed."""
    undocumented = [
        column
        for column, bounds in schema.NUMERIC_RANGES.items()
        if len(bounds.rationale.strip()) < 40
    ]
    assert not undocumented, f"Ranges without a real rationale: {undocumented}"


def test_contract_maps_are_read_only() -> None:
    """A consumer cannot mutate the contract; the tool forbids it, not a comment asking."""
    for mapping in (
        schema.UCI_CODE_TO_RAW,
        schema.RAW_TO_CANONICAL,
        schema.EXPECTED_DTYPES,
        schema.CATEGORICAL_LEVELS,
        schema.NUMERIC_RANGES,
    ):
        with pytest.raises(TypeError):
            mapping["INJECTED"] = "anything"  # type: ignore[index]
