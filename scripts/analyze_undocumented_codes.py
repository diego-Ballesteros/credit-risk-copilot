"""Measure what the undocumented repayment-status codes look like in the real data.

The data validator reports eight blocking `unknown_category` findings: the
repayment-status block holds codes the official UCI documentation never declares. This
script does not decide what those codes mean. It does not touch the data, the schema or
the validator; it only reads, so that the decision is taken over evidence instead of over
secondary literature.

Three hypotheses are put in front of the data. They come from a financial reading of the
product, not from the source documentation:

- **H1.** Code `-2` marks a month with no consumption. If so, the rows carrying it should
  show a bill statement near zero.
- **H2.** Code `0` marks revolving credit: the card was used, something was paid, and a
  balance is carried without being in arrears. If so, those rows should show a high bill
  statement and a positive payment smaller than the balance.
- **H3.** The scale of `PAY_STATUS_1` differs from the scale of `PAY_STATUS_2..6`. The
  signal that suggests it: code `1`, which the source does document, appears thousands of
  times in month 1 and almost never in months 2 to 6.

A fourth block, outside the hypotheses, profiles the undocumented codes of `EDUCATION`
and `MARRIAGE` against the documented levels of the same column.

Nothing here prints a conclusion. A measurement may be flagged as consistent or
inconsistent with a hypothesis; what a code *means* is not asserted anywhere.

**Panel direction.** Index 1 is the most recent month (September 2005) and index 6 the
oldest (April 2005), so month `m + 1` is chronologically *earlier* than month `m`. Every
measurement that crosses two months states which direction it reads, because reading it
backwards in silence would invert the meaning of the result.

Run it with:

    uv run python scripts/analyze_undocumented_codes.py

Output is a set of markdown tables, aligned so they are readable in a console and
copyable verbatim into `docs/analysis/undocumented-codes-evidence.md`. Retyping figures
into a document is how a document ends up holding numbers that no rerun reproduces.
"""

import sys
from collections.abc import Sequence
from typing import Final

import pandas as pd

from credit_copilot.data import schema
from credit_copilot.data.loader import load_dataset

_MONTHS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)
"""Panel month indices, most recent first."""

_OLDEST_MONTH: Final[int] = max(_MONTHS)
"""Index of the oldest month in the panel. It has no predecessor inside the dataset."""

_QUANTILES: Final[tuple[float, float, float]] = (0.25, 0.50, 0.75)
"""Quantiles reported for every amount column."""

_RULE_WIDTH: Final[int] = 100
"""Width of the rules printed between sections."""

_MISSING: Final[str] = "n/a"
"""Cell content for a statistic that cannot be computed, never a zero standing in for it."""

_ALL_ROWS_LABEL: Final[str] = "all"
"""Label of the whole-column row added to every per-code table.

Every subset statistic is printed next to the same statistic over the full column. A
subset default rate without the population rate beside it cannot be read: 22% looks high
or low depending on a baseline the reader would otherwise have to hold in memory.
"""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a table as an aligned markdown pipe table.

    Cells are padded to a common width, which makes the output readable in a console and
    valid markdown at the same time. That duality is the point: the evidence document
    quotes these blocks verbatim instead of retyping the figures.

    Args:
        headers: Column headers, one per column.
        rows: Rows of pre-formatted cells. Every row must have as many cells as headers.

    Returns:
        The table as a multi-line string, without a trailing newline.
    """
    widths = [
        max([len(header), *[len(row[index]) for row in rows]])
        for index, header in enumerate(headers)
    ]

    def line(cells: Sequence[str]) -> str:
        padded = " | ".join(cell.rjust(width) for cell, width in zip(cells, widths, strict=True))
        return f"| {padded} |"

    separator = "| " + " | ".join("-" * (width - 1) + ":" for width in widths) + " |"
    return "\n".join([line(headers), separator, *(line(row) for row in rows)])


def heading(text: str) -> str:
    """Render a section heading surrounded by rules.

    Args:
        text: Heading text.

    Returns:
        A three-line block: rule, text, rule.
    """
    rule = "=" * _RULE_WIDTH
    return f"\n{rule}\n{text}\n{rule}"


def _fmt_amount(value: float) -> str:
    """Format a NT$ amount for a table cell.

    Args:
        value: Amount, possibly a fractional quantile of integer data.

    Returns:
        The value rounded to whole NT$, with thousands separators.
    """
    return f"{value:,.0f}"


def _fmt_count(value: int) -> str:
    """Format a row count for a table cell.

    Args:
        value: Number of rows.

    Returns:
        The count with thousands separators.
    """
    return f"{value:,}"


def _fmt_pct(value: float) -> str:
    """Format a proportion as a percentage for a table cell.

    Args:
        value: Proportion in `[0, 1]`.

    Returns:
        The proportion as a percentage with two decimals and a `%` sign.
    """
    return f"{value * 100:.2f}%"


def _fmt_ratio(value: float) -> str:
    """Format a dimensionless ratio for a table cell.

    Args:
        value: The ratio.

    Returns:
        The ratio with three decimals.
    """
    return f"{value:.3f}"


# ---------------------------------------------------------------------------
# Shared measurement helpers
# ---------------------------------------------------------------------------


def code_masks(frame: pd.DataFrame, column: str) -> list[tuple[str, pd.Series]]:
    """Build one row mask per code present in a column, plus one for the whole column.

    Codes are taken from the data, not from the contract: the undocumented ones are
    exactly what has to be measured, so filtering by the declared levels would drop the
    subject of the analysis.

    Args:
        frame: Table with canonical column names.
        column: Column whose codes are enumerated.

    Returns:
        Pairs of label and boolean mask, codes in ascending order, `all` last.
    """
    codes = sorted(int(code) for code in frame[column].unique())
    masks: list[tuple[str, pd.Series]] = [(str(code), frame[column] == code) for code in codes]
    masks.append((_ALL_ROWS_LABEL, pd.Series(True, index=frame.index)))
    return masks


def payment_coverage(frame: pd.DataFrame, month: int, mask: pd.Series) -> tuple[str, str]:
    """Measure how much of the previous month's balance a month's payment covers.

    The denominator is the bill statement of the **previous** month, not of the same
    month. A billing cycle closes and only then is paid: the money recorded in
    `PAY_AMT{m}` settles the statement the client received at the close of the previous
    cycle, so dividing it by `BILL_AMT{m}` would compare a payment against a statement
    issued after that payment was made. Such a ratio would not measure coverage of
    anything - it would fold the new consumption of month `m` into the denominator and
    make a client who paid their balance in full look like a partial payer whenever they
    used the card again.

    Because index 1 is the most recent month, the chronologically previous month is
    `m + 1`, not `m - 1`. The oldest month has no predecessor inside the dataset, so its
    ratio does not exist and is reported as missing rather than as zero.

    Rows whose denominator is not positive are excluded: a zero or negative previous
    balance leaves nothing to cover, and dividing by it would manufacture an infinity or
    a sign flip. How many rows that removes is reported next to the median, because a
    median computed over a tenth of a subset is a different claim than one computed over
    all of it.

    Args:
        frame: Table with canonical column names.
        month: Panel month index of the payment.
        mask: Rows of the subset being profiled.

    Returns:
        The median coverage ratio and the share of subset rows excluded for a
        non-positive denominator, both pre-formatted, or `n/a` for the oldest month.
    """
    if month == _OLDEST_MONTH:
        return _MISSING, _MISSING

    subset = frame.loc[mask]
    previous_bill = subset[f"BILL_AMT{month + 1}"]
    usable = previous_bill > 0
    excluded = _fmt_pct(float((~usable).mean()))
    if not bool(usable.any()):
        return _MISSING, excluded

    ratio = subset.loc[usable, f"PAY_AMT{month}"] / previous_bill[usable]
    return _fmt_ratio(float(ratio.median())), excluded


def spearman(left: pd.Series, right: pd.Series) -> float:
    """Compute the Spearman rank correlation between two series.

    Spearman is Pearson computed over ranks, and it is computed that way here rather than
    through a statistics package: the project has no such dependency, adding one for a
    two-line formula would not be justified, and ranking explicitly makes the treatment of
    ties visible. Ties take the average rank, which is the standard definition and the one
    that matters here, because these columns hold a handful of repeated integer codes
    spread over thirty thousand rows.

    Args:
        left: First series.
        right: Second series, aligned on the same index.

    Returns:
        The rank correlation coefficient.
    """
    return float(left.rank().corr(right.rank()))


# ---------------------------------------------------------------------------
# H1 and H2 - what the amounts look like under each repayment-status code
# ---------------------------------------------------------------------------


def bill_profile(frame: pd.DataFrame, month: int) -> str:
    """Profile the bill statement of a month under each of its repayment-status codes.

    Args:
        frame: Table with canonical column names.
        month: Panel month index.

    Returns:
        A markdown table: one row per code, plus the whole-column baseline row.
    """
    bill = frame[f"BILL_AMT{month}"]
    target = frame[schema.TARGET_COLUMN]
    total_rows = len(frame)

    rows: list[list[str]] = []
    for label, mask in code_masks(frame, f"PAY_STATUS_{month}"):
        count = int(mask.sum())
        values = bill[mask]
        quantiles = values.quantile(list(_QUANTILES))
        rows.append(
            [
                label,
                _fmt_count(count),
                _fmt_pct(count / total_rows),
                _fmt_pct(float(target[mask].mean())),
                _fmt_amount(float(values.min())),
                _fmt_amount(float(quantiles.iloc[0])),
                _fmt_amount(float(quantiles.iloc[1])),
                _fmt_amount(float(quantiles.iloc[2])),
                _fmt_amount(float(values.max())),
                _fmt_pct(float((values == 0).mean())),
                _fmt_pct(float((values <= 0).mean())),
            ]
        )

    return render_table(
        [
            f"PAY_STATUS_{month}",
            "rows",
            "share",
            "default",
            f"BILL_AMT{month} min",
            "p25",
            "p50",
            "p75",
            "max",
            "= 0",
            "<= 0",
        ],
        rows,
    )


def payment_profile(frame: pd.DataFrame, month: int) -> str:
    """Profile the payment of a month under each of its repayment-status codes.

    Args:
        frame: Table with canonical column names.
        month: Panel month index.

    Returns:
        A markdown table: one row per code, plus the whole-column baseline row.
    """
    paid = frame[f"PAY_AMT{month}"]

    rows: list[list[str]] = []
    for label, mask in code_masks(frame, f"PAY_STATUS_{month}"):
        values = paid[mask]
        quantiles = values.quantile(list(_QUANTILES))
        median_coverage, excluded = payment_coverage(frame, month, mask)
        rows.append(
            [
                label,
                _fmt_amount(float(values.min())),
                _fmt_amount(float(quantiles.iloc[0])),
                _fmt_amount(float(quantiles.iloc[1])),
                _fmt_amount(float(quantiles.iloc[2])),
                _fmt_amount(float(values.max())),
                _fmt_pct(float((values == 0).mean())),
                median_coverage,
                excluded,
            ]
        )

    return render_table(
        [
            f"PAY_STATUS_{month}",
            f"PAY_AMT{month} min",
            "p25",
            "p50",
            "p75",
            "max",
            "= 0",
            "coverage p50",
            "excluded",
        ],
        rows,
    )


def section_amount_profiles(frame: pd.DataFrame) -> str:
    """Render the H1 and H2 block: amount profiles per code, for the six months.

    Args:
        frame: Table with canonical column names.

    Returns:
        The whole block as printable text.
    """
    parts: list[str] = [
        heading("H1 / H2 - BILL_AMT and PAY_AMT under every code of PAY_STATUS_m"),
        "",
        "One row per code present in the column, documented or not, plus an `all` row",
        "holding the same statistic over the whole column as a baseline.",
        "",
        "`default` is the share of the subset whose target is 1.",
        "`= 0` and `<= 0` are shares of the subset, not of the table.",
        "`coverage p50` is the median of PAY_AMT{m} / BILL_AMT{m+1} over the rows whose",
        "denominator is positive; `excluded` is the share of the subset left out of that",
        "median. Month m + 1 is the chronologically previous month, so the oldest month",
        "has no denominator and reports n/a.",
    ]
    for month in _MONTHS:
        label = f"-- Month {month} "
        parts.extend(
            [
                "\n" + label + "-" * (_RULE_WIDTH - len(label)),
                "",
                bill_profile(frame, month),
                "",
                payment_profile(frame, month),
            ]
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# H3 - is the scale of PAY_STATUS_1 the scale of PAY_STATUS_2..6
# ---------------------------------------------------------------------------


def code_frequency_matrix(frame: pd.DataFrame) -> str:
    """Cross-tabulate every repayment-status code against the six months, in one view.

    Args:
        frame: Table with canonical column names.

    Returns:
        Two markdown tables, absolute counts and shares of the table, one row per code.
    """
    counts = pd.DataFrame(
        {column: frame[column].value_counts() for column in schema.PAY_STATUS_COLUMNS}
    ).fillna(0)
    counts = counts.sort_index()
    shares = counts / len(frame)

    headers = ["code", *schema.PAY_STATUS_COLUMNS]
    count_rows = [
        [str(int(code)), *[_fmt_count(int(value)) for value in row]]
        for code, row in counts.iterrows()
    ]
    share_rows = [
        [str(int(code)), *[_fmt_pct(float(value)) for value in row]]
        for code, row in shares.iterrows()
    ]
    return "\n\n".join(
        [
            "Absolute frequency of each code, per month:",
            render_table(headers, count_rows),
            "Share of the table, per month:",
            render_table(headers, share_rows),
        ]
    )


def transition_matrix(frame: pd.DataFrame, month: int) -> str:
    """Distribute the code of month `m + 1` conditioned on the code of month `m`.

    Rows are the code observed in month `m`, columns the code observed in month `m + 1`,
    and every row sums to 100%. Because index 1 is the most recent month, this reads
    backwards in time: it conditions on the newer month and distributes over the older
    one. All five matrices read in the same direction, which is what comparing them
    requires; a matrix that read the other way could not be placed beside the rest.

    Args:
        frame: Table with canonical column names.
        month: Panel month index of the conditioning column.

    Returns:
        A markdown table with the conditioning row count and the row-normalised
        distribution.
    """
    current = frame[f"PAY_STATUS_{month}"]
    following = frame[f"PAY_STATUS_{month + 1}"]
    counts = pd.crosstab(current, following)
    shares = pd.crosstab(current, following, normalize="index")

    headers = [f"m={month}", "rows", *[str(int(code)) for code in shares.columns]]
    rows = [
        [
            str(int(code)),
            _fmt_count(int(counts.loc[code].sum())),
            *[_fmt_pct(float(value)) for value in shares.loc[code]],
        ]
        for code in shares.index
    ]
    return render_table(headers, rows)


def rank_correlations(frame: pd.DataFrame) -> str:
    """Correlate each repayment-status column with the target and with its neighbour.

    Args:
        frame: Table with canonical column names.

    Returns:
        Two markdown tables of Spearman coefficients.
    """
    target = frame[schema.TARGET_COLUMN]
    target_rows = [
        [column, _fmt_ratio(spearman(frame[column], target))]
        for column in schema.PAY_STATUS_COLUMNS
    ]
    neighbour_rows = [
        [
            f"PAY_STATUS_{month} ~ PAY_STATUS_{month + 1}",
            _fmt_ratio(spearman(frame[f"PAY_STATUS_{month}"], frame[f"PAY_STATUS_{month + 1}"])),
        ]
        for month in _MONTHS[:-1]
    ]
    return "\n\n".join(
        [
            f"Spearman of each column with {schema.TARGET_COLUMN}:",
            render_table(["column", "spearman"], target_rows),
            "Spearman between contiguous columns:",
            render_table(["pair", "spearman"], neighbour_rows),
        ]
    )


def section_scale_comparison(frame: pd.DataFrame) -> str:
    """Render the H3 block: frequencies, transitions and rank correlations.

    Args:
        frame: Table with canonical column names.

    Returns:
        The whole block as printable text.
    """
    parts: list[str] = [
        heading("H3 - is the scale of PAY_STATUS_1 the scale of PAY_STATUS_2..6"),
        "",
        code_frequency_matrix(frame),
        "",
        "Transition matrices. Rows: code in month m. Columns: distribution of the code in",
        "month m + 1, the chronologically previous month. Every row sums to 100%.",
    ]
    for month in _MONTHS[:-1]:
        parts.extend(
            [
                f"\nFrom PAY_STATUS_{month} to PAY_STATUS_{month + 1}:",
                "",
                transition_matrix(frame, month),
            ]
        )
    parts.extend(["", rank_correlations(frame)])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Outside the hypotheses - the undocumented codes of EDUCATION and MARRIAGE
# ---------------------------------------------------------------------------


def category_profile(frame: pd.DataFrame, column: str) -> str:
    """Profile every code of a categorical column against the documented levels.

    The undocumented codes are derived by subtracting the levels `schema` declares from
    the values the data holds, never listed by hand: a hardcoded list would keep printing
    the same codes after the data or the contract had changed.

    Args:
        frame: Table with canonical column names.
        column: Categorical column to profile.

    Returns:
        A markdown table with one row per code, one row aggregating the documented
        levels, one aggregating the undocumented ones, and the whole-column baseline.
    """
    declared = schema.CATEGORICAL_LEVELS[column]
    target = frame[schema.TARGET_COLUMN]
    total_rows = len(frame)
    present = sorted(int(code) for code in frame[column].unique())
    undocumented = [code for code in present if code not in declared]

    def row(label: str, documented: str, mask: pd.Series) -> list[str]:
        count = int(mask.sum())
        return [
            label,
            documented,
            _fmt_count(count),
            _fmt_pct(count / total_rows),
            _fmt_pct(float(target[mask].mean())) if count else _MISSING,
        ]

    rows = [
        row(str(code), "yes" if code in declared else "NO", frame[column] == code)
        for code in present
    ]
    rows.append(row("documented levels", "yes", frame[column].isin(sorted(declared))))
    rows.append(row("undocumented codes", "NO", frame[column].isin(undocumented)))
    rows.append(row(_ALL_ROWS_LABEL, "-", pd.Series(True, index=frame.index)))

    return "\n\n".join(
        [
            f"{column} - documented levels: {sorted(declared)} - "
            f"undocumented codes present: {undocumented}",
            render_table([column, "documented", "rows", "share", "default"], rows),
        ]
    )


def section_undocumented_demographics(frame: pd.DataFrame) -> str:
    """Render the block on the undocumented codes of EDUCATION and MARRIAGE.

    Args:
        frame: Table with canonical column names.

    Returns:
        The whole block as printable text.
    """
    return "\n".join(
        [
            heading("Outside the hypotheses - undocumented codes of EDUCATION and MARRIAGE"),
            "",
            "`default` is the share of each subset whose target is 1. The documented",
            "levels are aggregated into one row so that the undocumented codes can be",
            "read against them instead of against nothing.",
            "",
            category_profile(frame, "EDUCATION"),
            "",
            category_profile(frame, "MARRIAGE"),
        ]
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Load the dataset and print every measurement.

    Returns:
        Always 0. This script measures; it is not a gate and has nothing to fail on. The
        gate is `scripts/download_dataset.py`, which still exits 1 on the blocking
        findings these measurements describe.
    """
    frame = load_dataset()
    target = frame[schema.TARGET_COLUMN]

    print(heading("UNDOCUMENTED CODES - EVIDENCE"))
    print()
    print(f"Rows: {len(frame):,}    Columns: {frame.shape[1]}")
    print(
        f"Target {schema.TARGET_COLUMN}: {int(target.sum()):,} positives "
        f"({_fmt_pct(float(target.mean()))}) - the baseline every default rate below is "
        "read against."
    )
    print(
        "\nThis script measures. It decides nothing, modifies nothing, and nowhere "
        "asserts\nwhat a code means."
    )

    print(section_amount_profiles(frame))
    print(section_scale_comparison(frame))
    print(section_undocumented_demographics(frame))
    return 0


if __name__ == "__main__":
    sys.exit(main())
