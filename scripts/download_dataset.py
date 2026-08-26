"""Fetch the raw dataset, load it, validate the contract and print the whole report.

Run it with:

    uv run python scripts/download_dataset.py
    uv run python scripts/download_dataset.py --force

The exit code is the verdict: 0 when the table honours the contract, 1 when at least one
blocking finding needs a human decision. It exits non-zero even when the findings are
already known and expected, because a script that reports a problem and returns success
teaches everyone to stop reading its output.
"""

import argparse
import sys

from credit_copilot.data.loader import download_raw_dataset, load_dataset
from credit_copilot.data.validator import validate_dataframe


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        Parsed arguments with a single `force` flag.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-download and overwrite the raw file even if it already exists. "
            "Off by default so repeated runs neither need the network nor hit the UCI "
            "server again."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Download, load, validate and report.

    Returns:
        0 if the data honours the contract, 1 if any blocking finding was produced.
    """
    args = parse_args()

    outcome = download_raw_dataset(force=args.force)
    action = "downloaded from UCI" if outcome.downloaded else "reused from disk (already present)"
    print(f"Raw dataset {action}:\n  {outcome.path}\n")

    frame = load_dataset()
    print(f"Loaded with canonical names: {frame.shape[0]:,} rows x {frame.shape[1]} columns\n")

    result = validate_dataframe(frame)
    print(result.report())

    if not result.is_valid:
        print(
            "\nThe blocking findings above are measurements, not crashes. The validator "
            "reports\nthem and stops; deciding what they mean belongs in an ADR, not in a "
            "function."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
