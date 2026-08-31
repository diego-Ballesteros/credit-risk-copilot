"""Verification of the data contract. Reports, never repairs.

The validator answers one question - does this table hold what `schema.py` says it holds -
and it answers it completely. It does not impute, correct, clip or drop anything. A value
the contract did not expect is a fact about the data, and turning it into a different
value inside a function would destroy the finding and replace it with an invented one.
What to do about a finding is decided by a human and recorded in an ADR.

Two design choices are worth stating, because both are easy to get wrong:

**Accumulate, do not abort.** Every check runs and every problem is collected before
anything is raised. Failing on the first one turns a full picture into a queue of single
findings discovered one run at a time, and the second finding often changes what the first
one means.

**Severity is about who can absorb the problem, not about how alarming it looks.**

- `BLOCKING`: proceeding requires a decision that no default can make correctly. A missing
  column cannot be computed around. A column the contract does not know has the exact
  shape of a leakage vector. A wrong dtype breaks every arithmetic downstream. A null in a
  source documented as having none demands an imputation policy, and a silent one turns
  "unknown" into a false business fact. A category that appears in **neither** the levels
  the source declares nor the codes an ADR accepts is a value nobody has looked at yet. A
  value outside a plausible range is a unit error or a corrupt row. None of these can be
  settled by a library default.
- `INFORMATIVE`: a measured fact worth knowing that does not invalidate the contract. Two
  clients with identical attributes and different identifiers are unremarkable in a
  30,000-row extract; the number is still reported, because it matters for deduplication
  decisions later. An undocumented code that an ADR accepted is reported here too: the
  decision removed the blocker, not the fact, and the reading behind it is an inference
  that a future reader has a right to see restated on every run. A check that could not
  run at all is reported here as well, because a silent check reads as a passing one.

`validate_dataframe` never raises on findings. `validate_or_raise` wraps it for callers
that must stop. Both return the same complete result.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import pandas as pd

from credit_copilot.data import schema

_NO_COUNTS: Final[Mapping[str, int]] = MappingProxyType({})
"""Empty frequency table, for issues that do not have one."""

_MAX_LISTED_VALUES: Final[int] = 20
"""Cap on how many offending values a single issue message spells out."""


class Severity(StrEnum):
    """How much a finding costs the caller."""

    BLOCKING = "blocking"
    INFORMATIVE = "informative"


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found in the data.

    Attributes:
        check: Machine-readable name of the check that produced it.
        severity: Whether the finding blocks downstream work.
        column: Canonical column involved, or `None` for table-wide findings.
        message: What was found, in enough detail to act on without rerunning anything.
        counts: Offending value or column -> number of rows, when the check measures one.
    """

    check: str
    severity: Severity
    column: str | None
    message: str
    # `default_factory`, not `default`: dataclasses rejects any default whose `__hash__`
    # is None, and `mappingproxy` is unhashable even though it is read-only.
    counts: Mapping[str, int] = field(default_factory=lambda: _NO_COUNTS)


@dataclass(frozen=True)
class ValidationResult:
    """Everything the validator measured in one pass.

    Attributes:
        n_rows: Number of rows inspected.
        n_columns: Number of columns inspected.
        null_counts: Null count for every column present, zeros included. Reported
            unconditionally: "no nulls" is a claim that has to be shown, not assumed.
        issues: Findings, in the order the checks ran.
    """

    n_rows: int
    n_columns: int
    null_counts: Mapping[str, int]
    issues: tuple[ValidationIssue, ...]

    @property
    def blocking(self) -> tuple[ValidationIssue, ...]:
        """Findings that stop downstream work.

        Returns:
            The subset of `issues` with `BLOCKING` severity.
        """
        return tuple(issue for issue in self.issues if issue.severity is Severity.BLOCKING)

    @property
    def informative(self) -> tuple[ValidationIssue, ...]:
        """Findings worth knowing that do not stop anything.

        Returns:
            The subset of `issues` with `INFORMATIVE` severity.
        """
        return tuple(issue for issue in self.issues if issue.severity is Severity.INFORMATIVE)

    @property
    def is_valid(self) -> bool:
        """Whether the table honours the contract.

        Returns:
            `True` when there is no blocking finding. Informative findings do not
            invalidate the table.
        """
        return not self.blocking

    def report(self) -> str:
        """Render the full result as console text.

        Returns:
            A multi-section report: shape, null counts per column, blocking findings,
            informative findings, and a verdict.
        """
        lines: list[str] = [
            "=" * 78,
            "DATA CONTRACT VALIDATION",
            "=" * 78,
            f"Rows: {self.n_rows:,}    Columns: {self.n_columns}",
            "",
            "-- Null count per column ---------------------------------------------------",
        ]
        lines.extend(f"  {column:<28} {count:>10,}" for column, count in self.null_counts.items())
        total_nulls = sum(self.null_counts.values())
        lines.append(f"  {'TOTAL':<28} {total_nulls:>10,}")

        for severity, heading in (
            (Severity.BLOCKING, "Blocking findings"),
            (Severity.INFORMATIVE, "Informative findings"),
        ):
            found = [issue for issue in self.issues if issue.severity is severity]
            lines.extend(["", f"-- {heading} {'-' * (74 - len(heading))}"])
            if not found:
                lines.append("  none")
                continue
            for issue in found:
                target = issue.column if issue.column is not None else "<table>"
                lines.append(f"  [{issue.check}] {target}")
                lines.append(f"      {issue.message}")
                if issue.counts:
                    detail = ", ".join(f"{key}={value:,}" for key, value in issue.counts.items())
                    lines.append(f"      counts: {detail}")

        verdict = (
            "PASS - the table honours the contract"
            if self.is_valid
            else f"FAIL - {len(self.blocking)} blocking finding(s) need a human decision"
        )
        lines.extend(["", "=" * 78, verdict, "=" * 78])
        return "\n".join(lines)


class DataContractError(ValueError):
    """The table breaks the contract declared in `schema.py`."""

    def __init__(self, result: ValidationResult) -> None:
        """Build the exception from a completed validation.

        Args:
            result: The result whose blocking findings triggered the failure.
        """
        self.result = result
        super().__init__(result.report())


def _sorted_counts(counts: Mapping[object, int]) -> Mapping[str, int]:
    """Order a frequency table by descending frequency, then by value, and cap it.

    Args:
        counts: Raw value -> row count.

    Returns:
        A read-only, string-keyed frequency table of at most `_MAX_LISTED_VALUES` entries.
    """
    ordered = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
    return MappingProxyType({str(key): int(value) for key, value in ordered[:_MAX_LISTED_VALUES]})


def _check_columns(frame: pd.DataFrame) -> list[ValidationIssue]:
    """Compare the set of columns against the canonical contract.

    Two different sets do two different jobs. A column is **required** when
    `schema.WORKING_COLUMNS` lists it: that is the table `loader.load_dataset` produces
    and everything downstream consumes. A column is **allowed** when
    `schema.CANONICAL_COLUMNS` lists it, which is the wider set the raw file carries. The
    gap between them is `schema.DROPPED_ON_LOAD`, so the same validator can be pointed at
    the raw-renamed table or at the working table and be right about both, instead of
    reporting the identifier as missing from one and unexpected in the other.

    Args:
        frame: Table with canonical column names.

    Returns:
        One issue for missing columns and one for unexpected columns, when either exists.
    """
    present = list(frame.columns)
    missing = [column for column in schema.WORKING_COLUMNS if column not in present]
    unexpected = [column for column in present if column not in schema.CANONICAL_COLUMNS]

    issues: list[ValidationIssue] = []
    if missing:
        issues.append(
            ValidationIssue(
                check="missing_columns",
                severity=Severity.BLOCKING,
                column=None,
                message=(
                    f"{len(missing)} contract column(s) absent from the table: {missing}. "
                    "Nothing downstream can compute around a column that is not there."
                ),
            )
        )
    if unexpected:
        issues.append(
            ValidationIssue(
                check="unexpected_columns",
                severity=Severity.BLOCKING,
                column=None,
                message=(
                    f"{len(unexpected)} column(s) not in the contract: {unexpected}. "
                    "Treated as blocking rather than ignored: an unrecognised column has "
                    "the exact shape of a leakage vector, and the project's policy is to "
                    "make leakage impossible rather than to remember not to cause it."
                ),
            )
        )
    return issues


def _check_dtypes(frame: pd.DataFrame) -> list[ValidationIssue]:
    """Compare each present column's dtype against `schema.EXPECTED_DTYPES`.

    Args:
        frame: Table with canonical column names.

    Returns:
        One issue per column whose dtype differs from the contract.
    """
    return [
        ValidationIssue(
            check="dtype_mismatch",
            severity=Severity.BLOCKING,
            column=column,
            message=(
                f"Expected dtype {expected!r} but found {str(frame[column].dtype)!r}. "
                "Every arithmetic and every comparison downstream reads this column."
            ),
        )
        for column, expected in schema.EXPECTED_DTYPES.items()
        if column in frame.columns and str(frame[column].dtype) != expected
    ]


def _check_nulls(frame: pd.DataFrame) -> tuple[Mapping[str, int], list[ValidationIssue]]:
    """Count nulls in every column and flag the ones that have any.

    Args:
        frame: Table with canonical column names.

    Returns:
        The per-column null count for every column present, zeros included, and one
        issue per column that holds at least one null.
    """
    counts = {str(column): int(frame[column].isna().sum()) for column in frame.columns}
    issues = [
        ValidationIssue(
            check="null_values",
            severity=Severity.BLOCKING,
            column=column,
            message=(
                f"{count:,} null value(s) in a source documented as having none. "
                "Blocking because continuing means choosing an imputation, and a silent "
                "one turns 'unknown' into a business fact that is not true."
            ),
            counts=MappingProxyType({"nulls": count}),
        )
        for column, count in counts.items()
        if count > 0
    ]
    return MappingProxyType(counts), issues


def _check_categories(frame: pd.DataFrame) -> list[ValidationIssue]:
    """Compare each categorical column against both value contracts.

    Two maps are consulted, and the order matters. `schema.CATEGORICAL_LEVELS` says what
    the source declares. `schema.OBSERVED_CODES_ACCEPTED` says what this project accepted
    afterwards, on its own measurements, with an ADR behind each entry. A value in neither
    is a code nobody has looked at, and that is what stays blocking.

    The accepted codes are still reported. Silence would be cheaper and wrong: these are
    the majority of the dataset in the repayment-status block, the reading behind them is
    an inference and not a documented fact, and a run that mentions them keeps that
    visible to whoever reads the output next.

    Args:
        frame: Table with canonical column names.

    Returns:
        One blocking issue per column holding a value in neither map, and one informative
        issue per column holding an accepted code, each with its frequencies.
    """
    issues: list[ValidationIssue] = []
    for column, declared in schema.CATEGORICAL_LEVELS.items():
        if column not in frame.columns:
            continue
        accepted = schema.OBSERVED_CODES_ACCEPTED.get(column, {})
        observed = frame[column].value_counts(dropna=True)
        undeclared = {value: count for value, count in observed.items() if value not in declared}

        unknown = {value: count for value, count in undeclared.items() if value not in accepted}
        if unknown:
            affected = sum(unknown.values())
            issues.append(
                ValidationIssue(
                    check="unknown_category",
                    severity=Severity.BLOCKING,
                    column=column,
                    message=(
                        f"{len(unknown)} value(s) appear in the data that neither the "
                        "official UCI documentation declares nor any ADR accepts, across "
                        f"{affected:,} rows ({affected / len(frame):.2%}). Documented "
                        f"levels: {sorted(declared)}. Accepted codes: "
                        f"{sorted(accepted)}. These codes mean something nobody has "
                        "written down; what to do with them is a decision, not a default."
                    ),
                    counts=_sorted_counts(unknown),
                )
            )

        known = {value: count for value, count in undeclared.items() if value in accepted}
        if known:
            affected = sum(known.values())
            # Codes that share a reading are listed against it once. Three EDUCATION codes
            # were accepted by one measurement, and printing that measurement three times
            # buries the finding it belongs to.
            grouped: dict[str, list[int]] = {}
            for code in sorted(known, key=int):
                grouped.setdefault(accepted[code].meaning, []).append(int(code))
            readings = " ".join(
                f"[{', '.join(str(code) for code in codes)}] {meaning}"
                for meaning, codes in grouped.items()
            )
            adrs = sorted({accepted[code].adr for code in known})
            issues.append(
                ValidationIssue(
                    check="accepted_undocumented_category",
                    severity=Severity.INFORMATIVE,
                    column=column,
                    message=(
                        f"{len(known)} value(s) the official UCI documentation does not "
                        f"declare, across {affected:,} rows ({affected / len(frame):.2%}), "
                        f"accepted by {', '.join(adrs)} on measured evidence. Not blocking, "
                        "and not silent either: the reading is an inference, so if the "
                        f"source ever documents these codes the ADR is revisited. {readings}"
                    ),
                    counts=_sorted_counts(known),
                )
            )
    return issues


def _check_numeric_ranges(frame: pd.DataFrame) -> list[ValidationIssue]:
    """Find values outside the plausible ranges declared in `schema.NUMERIC_RANGES`.

    A column whose dtype is not numeric cannot be compared, so its range check is skipped
    and the skip is reported rather than passed over in silence.

    Args:
        frame: Table with canonical column names.

    Returns:
        One issue per column with out-of-range values, plus one per skipped column.
    """
    issues: list[ValidationIssue] = []
    for column, bounds in schema.NUMERIC_RANGES.items():
        if column not in frame.columns:
            continue
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            issues.append(
                ValidationIssue(
                    check="range_check_skipped",
                    severity=Severity.INFORMATIVE,
                    column=column,
                    message=(
                        f"Range check not run: dtype is {str(series.dtype)!r} and cannot "
                        "be compared numerically. The dtype mismatch is reported "
                        "separately as blocking."
                    ),
                )
            )
            continue

        below = 0 if bounds.minimum is None else int((series < bounds.minimum).sum())
        above = 0 if bounds.maximum is None else int((series > bounds.maximum).sum())
        if not below and not above:
            continue
        low = "-inf" if bounds.minimum is None else f"{bounds.minimum:,}"
        high = "+inf" if bounds.maximum is None else f"{bounds.maximum:,}"
        issues.append(
            ValidationIssue(
                check="out_of_range",
                severity=Severity.BLOCKING,
                column=column,
                message=(
                    f"{below + above:,} value(s) outside the plausible range "
                    f"[{low}, {high}]. Observed span: {series.min():,} to {series.max():,}. "
                    f"Bounds rationale: {bounds.rationale}"
                ),
                counts=MappingProxyType({"below_minimum": below, "above_maximum": above}),
            )
        )
    return issues


def _check_duplicates(frame: pd.DataFrame) -> list[ValidationIssue]:
    """Count repeated rows, with and without the identifier column.

    The two counts answer different questions. A row repeated across every column,
    identifier included, is a broken extract. A row repeated across every column except
    the identifier is two distinct clients with the same attributes, which a 30,000-row
    sample of the 24 remaining, mostly coarse, variables produces on its own.

    **The blocking half of this check needs the identifier.** Since ADR-0004 the loader
    drops `ID`, so the working table cannot support it: with no identifier present, every
    repeated row is by construction "identical on every column except the identifier",
    which is the case this validator has always called informative. Keeping it blocking
    would not be strictness, it would be a different claim than the evidence supports -
    the 35 such rows in the real dataset were informative when `ID` was there and nothing
    about them changed when the column left. The unavailable check is reported as skipped
    rather than passed over, for the same reason the range check reports its own skips: a
    check that stays quiet reads as a check that succeeded.

    Args:
        frame: Table with canonical column names.

    Returns:
        A blocking issue for exact duplicates and an informative one for duplicates that
        differ only by identifier, when either exists; and, when the identifier is absent,
        an informative issue recording that the blocking half could not run.
    """
    issues: list[ValidationIssue] = []
    has_identifier = schema.ID_COLUMN in frame.columns

    if not has_identifier:
        issues.append(
            ValidationIssue(
                check="exact_duplicate_check_skipped",
                severity=Severity.INFORMATIVE,
                column=None,
                message=(
                    f"Exact-duplicate detection not run: {schema.ID_COLUMN} is not in this "
                    "table, and without an identifier a broken extract cannot be told "
                    "apart from two distinct clients with identical attributes. The "
                    f"loader drops {schema.ID_COLUMN} by decision "
                    f"({schema.ADR_UNDOCUMENTED_CODES}), so this is the expected state of "
                    "the working table, not a fault. Run the check on "
                    "`loader.load_raw_dataframe`, which still carries the column."
                ),
            )
        )

    exact = int(frame.duplicated(keep="first").sum()) if has_identifier else 0
    if exact:
        issues.append(
            ValidationIssue(
                check="duplicate_rows",
                severity=Severity.BLOCKING,
                column=None,
                message=(
                    f"{exact:,} row(s) are an exact copy of an earlier row, identifier "
                    "included. An identifier is supposed to identify; a repeated one "
                    "means the extract is broken, not that a client appears twice."
                ),
                counts=MappingProxyType({"extra_copies": exact}),
            )
        )

    # A degenerate frame - no rows, or no columns left after dropping the identifier -
    # needs no guard: pandas reports zero duplicates for both, and a table that shape has
    # already produced a blocking `missing_columns` finding upstream.
    comparable = frame.drop(columns=[schema.ID_COLUMN]) if has_identifier else frame
    without_id = int(comparable.duplicated(keep="first").sum())
    if without_id:
        provenance = (
            f"{exact:,} of those are exact duplicates reported separately as blocking"
            if has_identifier
            else f"{schema.ID_COLUMN} is not in this table, so every column present was compared"
        )
        issues.append(
            ValidationIssue(
                check="duplicate_rows_ignoring_id",
                severity=Severity.INFORMATIVE,
                column=None,
                message=(
                    f"{without_id:,} row(s) match an earlier row on every column "
                    f"except {schema.ID_COLUMN}; {provenance}. Not blocking on its "
                    "own: distinct clients with identical attributes are expected at "
                    "this sample size. Reported because it bounds how much of the "
                    "data is genuinely independent."
                ),
                counts=MappingProxyType({"extra_copies": without_id}),
            )
        )
    return issues


def validate_dataframe(frame: pd.DataFrame) -> ValidationResult:
    """Run every contract check and collect all findings, raising nothing.

    Args:
        frame: Table with canonical column names, as returned by `loader.load_dataset`.

    Returns:
        The complete result: shape, per-column null counts, and every finding.
    """
    null_counts, null_issues = _check_nulls(frame)
    checks: Sequence[list[ValidationIssue]] = (
        _check_columns(frame),
        _check_dtypes(frame),
        null_issues,
        _check_categories(frame),
        _check_numeric_ranges(frame),
        _check_duplicates(frame),
    )
    return ValidationResult(
        n_rows=int(len(frame)),
        n_columns=int(frame.shape[1]),
        null_counts=null_counts,
        issues=tuple(issue for group in checks for issue in group),
    )


def validate_or_raise(frame: pd.DataFrame) -> ValidationResult:
    """Run every contract check and refuse to continue if anything blocks.

    Args:
        frame: Table with canonical column names.

    Returns:
        The complete result, when no blocking finding was produced.

    Raises:
        DataContractError: If at least one blocking finding was produced. The full report,
            informative findings included, is the exception message.
    """
    result = validate_dataframe(frame)
    if not result.is_valid:
        raise DataContractError(result)
    return result
