"""Tests of the loader, on a raw CSV written for each test.

The real download is never touched here. Every test writes a small file with the source's
own column names, so the assertions are about what the loader does to a file rather than
about whether a particular file happens to be on disk.
"""

from pathlib import Path

import pandas as pd
import pytest

from credit_copilot.data import schema
from credit_copilot.data.loader import (
    RawDataUnavailableError,
    load_dataset,
    load_raw_dataframe,
)


def _write_raw_csv(directory: Path) -> Path:
    """Write a two-row CSV carrying the source's raw column names.

    Args:
        directory: Where to write the file.

    Returns:
        Path to the written CSV.
    """
    columns: dict[str, list[int]] = {
        "ID": [1, 2],
        "LIMIT_BAL": [20_000, 120_000],
        "SEX": [1, 2],
        "EDUCATION": [1, 2],
        "MARRIAGE": [1, 2],
        "AGE": [24, 35],
    }
    for suffix in ("0", "2", "3", "4", "5", "6"):
        columns[f"PAY_{suffix}"] = [2, -1]
    for month in range(1, 7):
        columns[f"BILL_AMT{month}"] = [3_913, -1_200]
    for month in range(1, 7):
        columns[f"PAY_AMT{month}"] = [0, 689]
    columns["default payment next month"] = [1, 0]

    path = directory / schema.RAW_FILENAME
    pd.DataFrame(columns).to_csv(path, index=False)
    return path


def test_load_dataset_does_not_return_the_identifier(tmp_path: Path) -> None:
    """`ID` is gone from the working table, so it cannot become a feature.

    This is the whole point of ADR-0004's sixth decision. "Must not be used as a feature"
    written in a document holds while somebody remembers it; a column that is not there
    holds regardless.
    """
    frame = load_dataset(_write_raw_csv(tmp_path))

    assert schema.ID_COLUMN not in frame.columns
    assert list(frame.columns) == list(schema.WORKING_COLUMNS)
    assert frame.shape == (2, 24)


def test_load_raw_dataframe_still_carries_the_identifier(tmp_path: Path) -> None:
    """The drop happens on load, not on disk: traceability to the source is intact."""
    raw = load_raw_dataframe(_write_raw_csv(tmp_path))

    assert schema.ID_COLUMN in raw.columns
    assert raw.shape == (2, 25)


def test_load_dataset_applies_the_canonical_renaming(tmp_path: Path) -> None:
    """The repayment block is addressable by month and the target name is an identifier."""
    frame = load_dataset(_write_raw_csv(tmp_path))

    for column in schema.PAY_STATUS_COLUMNS:
        assert column in frame.columns
    assert schema.TARGET_COLUMN in frame.columns
    assert "PAY_0" not in frame.columns
    assert "default payment next month" not in frame.columns


def test_load_dataset_changes_no_value(tmp_path: Path) -> None:
    """Renaming and dropping are the only transformations; nothing is imputed or clipped."""
    path = _write_raw_csv(tmp_path)
    raw = load_raw_dataframe(path)
    frame = load_dataset(path)

    expected = raw.rename(columns=dict(schema.RAW_TO_CANONICAL)).drop(columns=[schema.ID_COLUMN])
    pd.testing.assert_frame_equal(frame, expected)


def test_missing_raw_file_names_the_command_that_fixes_it(tmp_path: Path) -> None:
    """The error tells the reader what to run instead of only what went wrong."""
    with pytest.raises(RawDataUnavailableError) as excinfo:
        load_dataset(tmp_path / "not-here.csv")
    assert "scripts/download_dataset.py" in str(excinfo.value)
