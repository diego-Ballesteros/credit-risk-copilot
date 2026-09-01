"""Score applicants with the pinned production model from the MLflow Model Registry.

Run it with::

    uv run python scripts/run_prediction.py --applicant-row 8842
    uv run python scripts/run_prediction.py --applicant-row 1 --applicant-row 2
    uv run python scripts/run_prediction.py --applicant-file applicants.json
    uv run python scripts/run_prediction.py --applicant-row 8842 --output scores.csv

===========================================================================
IT DOES NOT IMPUTE. A MISSING FIELD IS AN ERROR, NEVER A ZERO
===========================================================================

Every one of the 23 attributes is required and none has a default, because **a default is
an imputation with better manners**. `PAY_AMT3 = 0` means *"paid nothing in July"*, which is
a business fact; an absent `PAY_AMT3` means *"we do not know"*, which is not. Writing the
first over the second manufactures a fact, and no downstream check can detect it because the
row arrives well-formed and says something false about the world. Section 2.3 of the internal
credit policy sends the unknown case to full manual evaluation, and the only way to make that
path reachable is for this script to refuse.

The refusal is `models.applicant.ApplicantRecord` validated in **strict mode**, which is the
same contract `api/schemas.ApplicantAttributes` puts in front of the HTTP endpoint. Strict
mode matters beyond missing fields: in lax mode Pydantic turns the JSON literal `true` into
`1`, and `PAY_STATUS_1 = 1` means *one month in arrears*. That is a business fact fabricated
from a type error, which is imputation through another door.

Four ways in, and each stops before the artefact is touched:

    missing field       -> refused, naming the field
    explicit null       -> refused, naming the field
    unknown category    -> refused by `require_known_values`, naming the code
    out of plausible range -> refused, naming the column and the bound

**Nothing is scored partially.** Validation runs over the whole batch first; if any applicant
is refused, no probability is produced for any of them. A file half-scored and half-refused
invites the half that worked being used.

===========================================================================
WHAT IT LOADS, AND WHY IT NEVER REBUILDS ANYTHING
===========================================================================

`models.registry.load_registered_model` downloads `models:/credit-risk-default-probability/1`
exactly as `scripts/run_training.py` left it - the whole `Pipeline`, preprocessor included -
and never refits it. That is what makes the probability this script prints the same
probability `/predict` returns and the same one `docs/MODEL_CARD.md` describes. A script that
rebuilt the preprocessing would be the divergence section 6.3 of `docs/METHODOLOGY.md`
exists to prevent.

The version is **pinned**, not "latest", for the same reason ADR-0010 pins it in both
containers: a floating pointer means the number in a credit file cannot be reproduced later.
`--model-version` overrides it for deliberate comparisons.

===========================================================================
WHAT IT PRINTS, AND WHY THE CAVEATS TRAVEL WITH THE NUMBER
===========================================================================

Per applicant: the calibrated probability, the decision the operating threshold recommends,
and the threshold itself. Then, once per run, the cost assumption and the decision caveat -
the same `DecisionContext` the API attaches to every response.

They are not decoration. **The threshold is not a property of the model**: 0.160 comes from
assuming a false negative costs five times a false positive, an assumption this dataset
cannot support with exposure, recovery or margin data. At 3:1 it would be 0.220 and at 10:1,
0.105, and moving between those two moves 48.5% of the book. A probability printed without
that sentence reads like a verdict, and it is a recommendation.

**Only JSON is accepted for `--applicant-file`**, an object for one applicant or an array for
many. CSV is deliberately not supported: the strict contract above is defined over JSON
types, and reading integers back out of text would need coercion rules that are a decision
with alternatives rather than a move.

Exit code 0 when every applicant was scored, 1 when the input could not be read or an
applicant was refused, 2 when the registry could not serve the artefact.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final

import pandas as pd
from pydantic import ValidationError

from credit_copilot.api.schemas import DecisionContext
from credit_copilot.console import enable_unicode_console
from credit_copilot.data import schema
from credit_copilot.data.loader import RawDataUnavailableError, load_raw_dataframe
from credit_copilot.models.applicant import ApplicantRecord
from credit_copilot.models.decision import decide
from credit_copilot.models.registry import (
    PREDICTOR_COLUMNS,
    PRODUCTION_MODEL_NAME,
    PRODUCTION_MODEL_VERSION,
    ModelUnavailableError,
    UnknownValueError,
    load_registered_model,
)

RULE: Final[str] = "=" * 78
SUBRULE: Final[str] = "-" * 78


def parse_args() -> argparse.Namespace:
    """Read which applicants to score and which registry version to score them with.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--applicant-row",
        type=int,
        action="append",
        default=None,
        metavar="ID",
        help="Score the applicant with this source `ID` from data/raw/. Repeatable.",
    )
    parser.add_argument(
        "--applicant-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="JSON with one applicant (object) or many (array), keyed by canonical column.",
    )
    parser.add_argument(
        "--model-name",
        default=PRODUCTION_MODEL_NAME,
        help=f"Registered model name (default: {PRODUCTION_MODEL_NAME}).",
    )
    parser.add_argument(
        "--model-version",
        default=PRODUCTION_MODEL_VERSION,
        help=f"Registry version. Pinned by default ({PRODUCTION_MODEL_VERSION}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also write the scores to this CSV.",
    )
    return parser.parse_args()


def load_applicants_from_dataset(client_ids: list[int]) -> list[dict[str, Any]]:
    """Read the raw attributes of the given clients from the downloaded dataset.

    Args:
        client_ids: Values of the source `ID` column.

    Returns:
        One mapping of the 23 canonical predictor columns per identifier, in the order
        asked for.

    Raises:
        RawDataUnavailableError: The raw file has not been downloaded.
        LookupError: No row carries one of those identifiers.
    """
    frame = load_raw_dataframe().rename(columns=dict(schema.RAW_TO_CANONICAL))
    indexed = frame.set_index(schema.ID_COLUMN)
    applicants: list[dict[str, Any]] = []
    for client_id in client_ids:
        if client_id not in indexed.index:
            raise LookupError(
                f"No row carries {schema.ID_COLUMN}={client_id}. The dataset holds "
                f"{len(frame):,} rows, with identifiers from {frame[schema.ID_COLUMN].min()} "
                f"to {frame[schema.ID_COLUMN].max()}."
            )
        row = indexed.loc[client_id]
        applicants.append({column: int(row[column]) for column in PREDICTOR_COLUMNS})
    return applicants


def load_applicants_from_file(path: Path) -> list[dict[str, Any]]:
    """Read applicants from a JSON file: one object, or an array of objects.

    Values are passed through **exactly as written**. Nothing is filled in and nothing is
    coerced here; the record contract refuses what is wrong, which is where that refusal
    belongs.

    Args:
        path: File holding one JSON object or an array of them.

    Returns:
        The payloads, unmodified.

    Raises:
        ValueError: The file is not a JSON object or array of objects.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(
            f"{path} must hold a JSON object with one applicant or an array of them; it "
            f"holds a {type(payload).__name__}."
        )
    return list(payload)


def validate_all(payloads: list[dict[str, Any]]) -> list[ApplicantRecord]:
    """Validate every applicant in strict mode before any of them is scored.

    Args:
        payloads: Raw mappings, one per applicant.

    Returns:
        The validated records, in input order.

    Raises:
        SystemExit: Any applicant was refused. The message names the applicant and the
            field, and no probability is produced for the batch.
    """
    records: list[ApplicantRecord] = []
    problems: list[str] = []
    for position, payload in enumerate(payloads, start=1):
        try:
            records.append(ApplicantRecord.model_validate(payload, strict=True))
        except ValidationError as error:
            for issue in error.errors():
                field = ".".join(str(part) for part in issue["loc"]) or "(record)"
                problems.append(f"  applicant {position}: {field}: {issue['msg']}")
        except UnknownValueError as error:
            problems.append(f"  applicant {position}: {error}")

    if problems:
        print(f"\n{len(problems)} applicant(s) refused. Nothing was scored.", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        print(
            "\nNo value was filled in on purpose: an absent field means 'unknown', and a "
            "zero would mean 'did not pay'. Section 2.3 of the internal credit policy sends "
            "the unknown case to full manual evaluation.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return records


def main() -> int:
    """Load the pinned artefact, score the applicants, and print the decisions.

    Returns:
        0 when every applicant was scored, 1 on bad input, 2 when the registry could not
        serve the artefact.
    """
    enable_unicode_console()
    args = parse_args()

    if not args.applicant_row and not args.applicant_file:
        print(
            "Nothing to score. Pass --applicant-row ID or --applicant-file PATH.",
            file=sys.stderr,
        )
        return 1

    payloads: list[dict[str, Any]] = []
    labels: list[str] = []
    try:
        if args.applicant_row:
            payloads += load_applicants_from_dataset(args.applicant_row)
            labels += [f"ID {client_id}" for client_id in args.applicant_row]
        if args.applicant_file:
            from_file = load_applicants_from_file(args.applicant_file)
            payloads += from_file
            labels += [f"{args.applicant_file.name}#{i}" for i in range(1, len(from_file) + 1)]
    except (RawDataUnavailableError, LookupError, ValueError, json.JSONDecodeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1

    records = validate_all(payloads)

    print(RULE)
    print("SCORING WITH THE PINNED PRODUCTION MODEL")
    print(RULE)
    try:
        model = load_registered_model(name=args.model_name, version=args.model_version)
    except ModelUnavailableError as error:
        print(f"The registry could not serve the artefact:\n{error}", file=sys.stderr)
        return 2

    print(f"  artefact    {model.uri}")
    print(f"  applicants  {len(records)}")
    print("  imputation  none - every field was required and present")

    frame = pd.DataFrame(
        [record.model_dump() for record in records], columns=list(PREDICTOR_COLUMNS)
    )
    probabilities = model.pipeline.predict_proba(frame)[:, 1]
    context = DecisionContext.current()

    print("\n" + SUBRULE)
    print(f"{'applicant':<22}{'P(default)':>14}{'decision':>12}{'threshold':>12}")
    print(SUBRULE)
    scores = []
    for label, probability in zip(labels, probabilities, strict=True):
        decision = decide(float(probability))
        print(f"{label:<22}{probability:>14.6f}{decision:>12}{context.threshold:>12.3f}")
        scores.append(
            {
                "applicant": label,
                "probability_of_default": float(probability),
                "decision": decision,
                "threshold": context.threshold,
                "cost_ratio_fn_to_fp": context.cost_ratio_fn_to_fp,
            }
        )
    print(SUBRULE)

    print("\n" + RULE)
    print("WHAT THE NUMBER DOES NOT SAY")
    print(RULE)
    print(f"  {context.cost_assumption}")
    print()
    print(f"  {context.decision_caveat}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(scores).to_csv(args.output, index=False)
        print(f"\nScores written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
