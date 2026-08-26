"""Data subpackage: the contract, the door into the data, and the check on both.

`schema` declares what the data is supposed to be, `loader` is the only way to read it,
and `validator` says whether the two agree. Nothing here cleans or imputes.
"""

from credit_copilot.data.loader import (
    DownloadOutcome,
    RawDataUnavailableError,
    SourceContractError,
    download_raw_dataset,
    load_dataset,
    load_raw_dataframe,
    raw_dataset_path,
)
from credit_copilot.data.validator import (
    DataContractError,
    Severity,
    ValidationIssue,
    ValidationResult,
    validate_dataframe,
    validate_or_raise,
)

__all__ = [
    "DataContractError",
    "DownloadOutcome",
    "RawDataUnavailableError",
    "Severity",
    "SourceContractError",
    "ValidationIssue",
    "ValidationResult",
    "download_raw_dataset",
    "load_dataset",
    "load_raw_dataframe",
    "raw_dataset_path",
    "validate_dataframe",
    "validate_or_raise",
]
