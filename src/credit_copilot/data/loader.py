"""Acquisition and loading of the raw dataset. The only door into the data.

Two responsibilities, kept apart on purpose:

- `download_raw_dataset` talks to UCI and writes `data/raw/`. It is idempotent: an
  existing file is left alone unless forcing is explicit, so the exploratory work of the
  next turns neither depends on the network nor hammers the source server.
- `load_dataset` reads that file and applies the canonical renaming. Every consumer in
  the project - notebook, training script, API - goes through it. Reading the CSV
  directly anywhere else would fork the naming contract in two, and the two copies would
  drift.

Neither function cleans, imputes or corrects anything. What arrives is what the source
sent, renamed and nothing else; judging it is the validator's job and deciding what to do
about it is the human's.
"""

import shutil
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from ucimlrepo import fetch_ucirepo

from credit_copilot.config import settings
from credit_copilot.data import schema


class RawDataUnavailableError(FileNotFoundError):
    """The raw file is not on disk and no download was requested."""


class SourceContractError(RuntimeError):
    """The UCI payload no longer matches the legend frozen in `schema.UCI_CODE_TO_RAW`."""


class DownloadOutcome(NamedTuple):
    """Result of a download attempt.

    Attributes:
        path: Location of the raw CSV on disk.
        downloaded: `True` if the network was hit and the file written, `False` if an
            existing file was reused.
    """

    path: Path
    downloaded: bool


def raw_dataset_path() -> Path:
    """Canonical location of the raw CSV.

    Returns:
        Absolute path to `data/raw/default_of_credit_card_clients.csv`.
    """
    return settings.raw_data_dir / schema.RAW_FILENAME


def _decode_source_columns(dataset: object) -> pd.DataFrame:
    """Rebuild the documented column header from the `ucimlrepo` payload.

    The API labels the columns `X1..X23`/`Y` and ships the documented names in a side
    table. This restores the header the source itself publishes, so the raw file matches
    the dataset as it is documented and as every published example of it is written.
    The legend is not trusted blindly: it is checked against the frozen copy in `schema`,
    and any divergence is a hard failure rather than a silent change of meaning.

    A column the source describes with no text - `ID` is the only one - takes its own code
    as its name.

    Args:
        dataset: Object returned by `ucimlrepo.fetch_ucirepo`.

    Returns:
        The 30,000 x 25 table with the documented header, columns in the frozen order.

    Raises:
        SourceContractError: If the delivered columns or their documented names differ
            from `schema.UCI_CODE_TO_RAW`.
    """
    original: pd.DataFrame = dataset.data.original  # type: ignore[attr-defined]
    variables: pd.DataFrame = dataset.variables  # type: ignore[attr-defined]

    delivered = set(original.columns)
    expected = set(schema.UCI_CODE_TO_RAW)
    if delivered != expected:
        raise SourceContractError(
            "The UCI payload does not carry the expected column codes. "
            f"Missing: {sorted(expected - delivered)}. Unexpected: {sorted(delivered - expected)}."
        )

    live_legend = {
        str(code): (str(description) if isinstance(description, str) else str(code))
        for code, description in zip(variables["name"], variables["description"], strict=True)
    }
    divergences = {
        code: (frozen, live_legend.get(code))
        for code, frozen in schema.UCI_CODE_TO_RAW.items()
        if live_legend.get(code) != frozen
    }
    if divergences:
        detail = "; ".join(
            f"{code}: frozen={frozen!r} live={live!r}"
            for code, (frozen, live) in divergences.items()
        )
        raise SourceContractError(
            "The names UCI documents no longer match the legend frozen in schema.py. "
            f"{detail}. Do not edit the CSV: decide what the change means first."
        )

    renamed: pd.DataFrame = original.rename(columns=dict(schema.UCI_CODE_TO_RAW))
    return renamed[list(schema.UCI_CODE_TO_RAW.values())]


def download_raw_dataset(
    *,
    force: bool = False,
    destination: Path | None = None,
) -> DownloadOutcome:
    """Fetch the dataset from UCI and write it to `data/raw/`, once.

    Idempotent by default: if the file already exists the network is not touched. The
    file is written through a temporary neighbour and moved into place, so an interrupted
    download cannot leave a half-written CSV that later looks like a valid cached copy.

    Args:
        force: Download and overwrite even if the file already exists.
        destination: Where to write. Defaults to `raw_dataset_path()`.

    Returns:
        The path written or reused, and whether the network was hit.

    Raises:
        SourceContractError: If the UCI payload no longer matches the frozen legend.
    """
    target = destination if destination is not None else raw_dataset_path()
    if target.is_file() and not force:
        return DownloadOutcome(path=target, downloaded=False)

    frame = _decode_source_columns(fetch_ucirepo(id=schema.UCI_DATASET_ID))

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".partial")
    frame.to_csv(staging, index=False)
    shutil.move(str(staging), str(target))
    return DownloadOutcome(path=target, downloaded=True)


def load_raw_dataframe(path: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV with the source's own column names, unchanged.

    Dtypes are inferred by pandas rather than forced. Forcing them here would silence the
    validator's dtype check, which exists precisely to notice when the file stops holding
    what the contract says it holds.

    Args:
        path: File to read. Defaults to `raw_dataset_path()`.

    Returns:
        The table exactly as stored, raw column names included.

    Raises:
        RawDataUnavailableError: If the file is not on disk.
    """
    source = path if path is not None else raw_dataset_path()
    if not source.is_file():
        raise RawDataUnavailableError(
            f"Raw dataset not found at {source}. "
            "Run `uv run python scripts/download_dataset.py` to fetch it."
        )
    return pd.read_csv(source)


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the dataset with canonical column names. The project's single entry point.

    Renaming is the only transformation applied. A column the contract does not know is
    passed through under its original name instead of raising: aborting here would hide
    every other problem in the file behind the first one, and the validator is built to
    report the whole picture in one run.

    Args:
        path: Raw CSV to read. Defaults to `raw_dataset_path()`.

    Returns:
        The table with canonical column names, in the order the file stores them.

    Raises:
        RawDataUnavailableError: If the file is not on disk.
    """
    return load_raw_dataframe(path).rename(columns=dict(schema.RAW_TO_CANONICAL))
