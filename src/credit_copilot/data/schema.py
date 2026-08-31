"""Declarative data contract for "Default of Credit Card Clients" (UCI id 350).

This module holds facts, never behaviour. It declares four separate things that are
easy to confuse and must not be merged:

1. What the source **delivers** (`UCI_CODE_TO_RAW`): the `ucimlrepo` API labels the
   columns `X1..X23`/`Y` and ships the documented names in a side table. The raw file
   is written with the documented names, so this legend is frozen here and checked
   against the live payload at download time.
2. What the project **calls** each column (`RAW_TO_CANONICAL`).
3. What the source **says** each column contains (`EXPECTED_DTYPES`,
   `CATEGORICAL_LEVELS`, `NUMERIC_RANGES`).
4. What the project **accepts** beyond that, on its own measured evidence
   (`OBSERVED_CODES_ACCEPTED`, `EDUCATION_COLLAPSE_MAP`,
   `PAY_STATUS_HOMOGENEOUS_COLUMNS`), each entry pointing at the ADR that decided it.

`CATEGORICAL_LEVELS` deserves an explicit warning, because the temptation to "fix" it is
the single most damaging edit that can be made to this module: it records the levels the
official UCI documentation **declares**, not the values the data **contains**. The real
data holds codes the source never documented, and those codes are now accepted - but they
are accepted in the *fourth* map, never by being added to the third one. Merging the two
would erase the distinction between a fact about the source and a decision of ours, which
is the one thing that has to stay legible if UCI ever publishes documentation for them.
ADR-0004 holds the evidence and `docs/analysis/undocumented-codes-evidence.md` the
measurements behind it.

Every map is exposed as a read-only mapping. A downstream module that mutated the
contract would poison every consumer in silence, so the tool forbids it rather than a
document asking for it not to happen.

Source documentation quoted throughout:
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

UCI_DATASET_ID: Final[int] = 350
"""Identifier of the dataset in the UCI Machine Learning Repository."""

RAW_FILENAME: Final[str] = "default_of_credit_card_clients.csv"
"""Name of the raw CSV inside `data/raw/`."""

_MONTH_INDICES: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)
"""Panel month index: 1 is the most recent month (September 2005), 6 the oldest (April 2005)."""

# ---------------------------------------------------------------------------
# 1 - What the source delivers
# ---------------------------------------------------------------------------

UCI_CODE_TO_RAW: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ID": "ID",
        "X1": "LIMIT_BAL",
        "X2": "SEX",
        "X3": "EDUCATION",
        "X4": "MARRIAGE",
        "X5": "AGE",
        "X6": "PAY_0",
        "X7": "PAY_2",
        "X8": "PAY_3",
        "X9": "PAY_4",
        "X10": "PAY_5",
        "X11": "PAY_6",
        "X12": "BILL_AMT1",
        "X13": "BILL_AMT2",
        "X14": "BILL_AMT3",
        "X15": "BILL_AMT4",
        "X16": "BILL_AMT5",
        "X17": "BILL_AMT6",
        "X18": "PAY_AMT1",
        "X19": "PAY_AMT2",
        "X20": "PAY_AMT3",
        "X21": "PAY_AMT4",
        "X22": "PAY_AMT5",
        "X23": "PAY_AMT6",
        "Y": "default payment next month",
    }
)
"""Legend published by UCI: opaque column code -> documented column name.

Frozen here rather than trusted blindly from the API on every run, so that a change on
the source side becomes a loud failure at download time instead of a silent change of
meaning. The values are the names the raw CSV carries and the names every published
example of this dataset uses; they are never translated or normalised.
"""

# ---------------------------------------------------------------------------
# 2 - What the project calls each column
# ---------------------------------------------------------------------------

RAW_TO_CANONICAL: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ID": "ID",
        "LIMIT_BAL": "LIMIT_BAL",
        "SEX": "SEX",
        "EDUCATION": "EDUCATION",
        "MARRIAGE": "MARRIAGE",
        "AGE": "AGE",
        # The source numbers the repayment-status block PAY_0, PAY_2..PAY_6: the index
        # skips 1, so no loop can address it without a special case. The block is also
        # prefix-ambiguous with PAY_AMT1..6, which is a different variable entirely. The
        # canonical suffix is the panel month index already used by BILL_AMT* and
        # PAY_AMT*, so the three blocks line up: PAY_STATUS_1, BILL_AMT1 and PAY_AMT1 all
        # describe September 2005.
        "PAY_0": "PAY_STATUS_1",
        "PAY_2": "PAY_STATUS_2",
        "PAY_3": "PAY_STATUS_3",
        "PAY_4": "PAY_STATUS_4",
        "PAY_5": "PAY_STATUS_5",
        "PAY_6": "PAY_STATUS_6",
        "BILL_AMT1": "BILL_AMT1",
        "BILL_AMT2": "BILL_AMT2",
        "BILL_AMT3": "BILL_AMT3",
        "BILL_AMT4": "BILL_AMT4",
        "BILL_AMT5": "BILL_AMT5",
        "BILL_AMT6": "BILL_AMT6",
        "PAY_AMT1": "PAY_AMT1",
        "PAY_AMT2": "PAY_AMT2",
        "PAY_AMT3": "PAY_AMT3",
        "PAY_AMT4": "PAY_AMT4",
        "PAY_AMT5": "PAY_AMT5",
        "PAY_AMT6": "PAY_AMT6",
        # Spaces make the source name unusable as an identifier. The transformation is
        # the minimum that fixes it: whitespace to underscore, uppercased like the rest.
        # No word is added and none is dropped.
        "default payment next month": "DEFAULT_PAYMENT_NEXT_MONTH",
    }
)
"""Raw column name -> canonical column name, applied once on load.

A name changes only when the source name makes correct code impossible or fragile. Two
blocks qualify; the other eighteen columns keep the source name verbatim, because every
paper, tutorial and published notebook on this dataset uses that vocabulary and the cost
of diverging from it is paid on every future read.
"""

TARGET_COLUMN: Final[str] = "DEFAULT_PAYMENT_NEXT_MONTH"
"""Canonical name of the target: 1 if the client defaults next month, 0 otherwise."""

ID_COLUMN: Final[str] = "ID"
"""Canonical name of the row identifier. Carries no business meaning; never a feature."""

PAY_STATUS_COLUMNS: Final[tuple[str, ...]] = tuple(f"PAY_STATUS_{i}" for i in _MONTH_INDICES)
"""Repayment-status block, most recent month first."""

BILL_AMOUNT_COLUMNS: Final[tuple[str, ...]] = tuple(f"BILL_AMT{i}" for i in _MONTH_INDICES)
"""Bill-statement block, most recent month first."""

PAY_AMOUNT_COLUMNS: Final[tuple[str, ...]] = tuple(f"PAY_AMT{i}" for i in _MONTH_INDICES)
"""Previous-payment block, most recent month first."""

CANONICAL_COLUMNS: Final[tuple[str, ...]] = tuple(RAW_TO_CANONICAL.values())
"""The 25 canonical columns, in source order: the identifier, 23 predictors, the target."""

DROPPED_ON_LOAD: Final[tuple[str, ...]] = (ID_COLUMN,)
"""Columns the loader removes, so that no consumer can reach them by accident.

`ID` carries no business meaning and must never become a feature. That rule used to live
only in the data dictionary, which is the weakest kind of guarantee there is - it holds
while somebody remembers it. Removing the column at load time turns "must not be used"
into "cannot be used", which is the only version that survives a distracted afternoon.
Decided in ADR-0004; the raw CSV keeps the column and `loader.load_raw_dataframe` still
returns it, so traceability back to the source is not lost.
"""

WORKING_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in CANONICAL_COLUMNS if column not in DROPPED_ON_LOAD
)
"""The 24 columns `loader.load_dataset` returns: 23 predictors and the target.

`CANONICAL_COLUMNS` describes the *file*; this describes the *table the project works
with*. The two differ by `DROPPED_ON_LOAD`, and keeping both is what lets the validator
check either one without pretending the difference does not exist.
"""

# ---------------------------------------------------------------------------
# 3 - What the source says each column contains
# ---------------------------------------------------------------------------

EXPECTED_DTYPES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ID": "int64",
        "LIMIT_BAL": "int64",
        "SEX": "int64",
        "EDUCATION": "int64",
        "MARRIAGE": "int64",
        "AGE": "int64",
        **dict.fromkeys(PAY_STATUS_COLUMNS, "int64"),
        **dict.fromkeys(BILL_AMOUNT_COLUMNS, "int64"),
        **dict.fromkeys(PAY_AMOUNT_COLUMNS, "int64"),
        TARGET_COLUMN: "int64",
    }
)
"""Canonical column -> expected pandas dtype.

Every column of this dataset is an integer at the source, the monetary ones included:
amounts are whole NT dollars. The loader deliberately does not force these dtypes when
reading the CSV. It lets pandas infer and lets the validator compare, because forcing
them would make the dtype check tautological and would hide exactly the corruption the
check exists to find.
"""

CATEGORICAL_LEVELS: Final[Mapping[str, frozenset[int]]] = MappingProxyType(
    {
        # "Gender (1 = male; 2 = female)."
        "SEX": frozenset({1, 2}),
        # "Education (1 = graduate school; 2 = university; 3 = high school; 4 = others)."
        "EDUCATION": frozenset({1, 2, 3, 4}),
        # "Marital status (1 = married; 2 = single; 3 = others)."
        "MARRIAGE": frozenset({1, 2, 3}),
        # "The measurement scale for the repayment status is: -1 = pay duly; 1 = payment
        # delay for one month; ...; 9 = payment delay for nine months and above."
        # The documentation declares neither 0 nor -2.
        **dict.fromkeys(PAY_STATUS_COLUMNS, frozenset({-1, 1, 2, 3, 4, 5, 6, 7, 8, 9})),
        # "a binary variable, default payment (Yes = 1, No = 0)"
        TARGET_COLUMN: frozenset({0, 1}),
    }
)
"""Canonical categorical column -> levels the official UCI documentation declares.

Read the module docstring before editing this map. It states the source's claim, not the
dataset's content, and the difference between the two is what the validator reports.
"""

# ---------------------------------------------------------------------------
# 4 - What the project accepts beyond what the source says
# ---------------------------------------------------------------------------

ADR_UNDOCUMENTED_CODES: Final[str] = "ADR-0004"
"""Identifier of the decision that accepted the undocumented codes.

Carried as a constant rather than typed into each message, so that a reader who wants the
evidence has one string to search for and the validator cannot cite a different ADR from
the one this map implements.
"""


@dataclass(frozen=True)
class AcceptedCode:
    """An undocumented code the project accepts, and the decision that accepted it.

    Attributes:
        meaning: What the code is taken to mean, in one line.
        adr: The ADR that accepted it and holds the measurements behind that reading.
    """

    meaning: str
    adr: str


OBSERVED_CODES_ACCEPTED: Final[Mapping[str, Mapping[int, AcceptedCode]]] = MappingProxyType(
    {
        "EDUCATION": MappingProxyType(
            {
                code: AcceptedCode(
                    meaning=(
                        "Undocumented education code. Grouped, the three of them default at "
                        "7.54% against 5.69% for documented level 4 ('others') and 19.23% to "
                        "25.16% for levels 1 to 3, so they are collapsed onto level 4."
                    ),
                    adr=ADR_UNDOCUMENTED_CODES,
                )
                for code in (0, 5, 6)
            }
        ),
        "MARRIAGE": MappingProxyType(
            {
                0: AcceptedCode(
                    meaning=(
                        "Undocumented marital-status code, kept as a level of its own. It "
                        "defaults at 9.26% against 26.01% for documented level 3 ('others'), "
                        "so it is not collapsed onto it."
                    ),
                    adr=ADR_UNDOCUMENTED_CODES,
                )
            }
        ),
        **dict.fromkeys(
            PAY_STATUS_COLUMNS,
            MappingProxyType(
                {
                    -2: AcceptedCode(
                        meaning=(
                            "No consumption during the month. Median payment coverage of the "
                            "previous balance is 1.000, and 25.15% to 61.08% of these rows "
                            "carry a bill statement of exactly zero."
                        ),
                        adr=ADR_UNDOCUMENTED_CODES,
                    ),
                    0: AcceptedCode(
                        meaning=(
                            "Revolving credit: the card was used, part of the balance was "
                            "paid, and the rest is carried without being in arrears. Median "
                            "payment coverage is 0.042 to 0.057, against 1.000 for -1 and -2."
                        ),
                        adr=ADR_UNDOCUMENTED_CODES,
                    ),
                }
            ),
        ),
    }
)
"""Canonical categorical column -> code the data holds that the source never declared.

**This is deliberately a second map and not an addition to `CATEGORICAL_LEVELS`.** That
one records what the source *says*; this one records what this project *accepts* after
measuring. Merging them would save a few lines and destroy the only thing that makes
either one worth having: the day UCI publishes documentation for these codes, the
difference between "declared by the source" and "accepted by us on our own evidence" has
to still be readable. A code that lives in one map is a fact about the source; a code that
lives in the other is a decision with an ADR behind it, and the two are not
interchangeable.

The validator consults both. A code in this map stops being blocking and is reported as
informative, naming the ADR. A code in neither map is still blocking, because it is a
value nobody has looked at yet.
"""

EDUCATION_COLLAPSE_MAP: Final[Mapping[int, int]] = MappingProxyType({0: 4, 5: 4, 6: 4})
"""`EDUCATION` code -> documented level it is collapsed onto, per ADR-0004.

Declared here and applied nowhere in this module: `schema` holds facts, and the collapse
is a transformation that belongs in the preprocessing pipeline, where it can be fitted,
serialised and versioned like every other step.

`MARRIAGE` has no equivalent map on purpose. Its undocumented `0` is *not* collapsed onto
the documented "others" level, because the measurement pointed the other way - 9.26%
default against 26.01%. The absence of a map here is the decision, not an oversight.
"""

PAY_STATUS_HOMOGENEOUS_COLUMNS: Final[tuple[str, ...]] = PAY_STATUS_COLUMNS[1:]
"""The repayment-status months that share one scale: 2 to 6, August back to April.

Trajectory features are built over these five columns and not over all six. ADR-0004
measured three independent signs that month 1 does not share their low-end scale: code 1
appears 3,688 times in month 1 and 28, 4, 2, 0 and 0 times afterwards; contiguous Spearman
is 0.799 to 0.822 among the four pairs that exclude month 1 and 0.627 for the pair that
includes it; and in the month 1 to 2 transition matrix the code 0 row sends 0.00% of its
14,737 rows to code 2, against 3.48% to 5.32% in the other four matrices.

This constant exists so that the feature-engineering step cannot assume homogeneity by
accident. A loop over `PAY_STATUS_COLUMNS` that computes a trend is silently wrong; the
same loop over this tuple is right. The wrong version has to be spelled out to happen.
"""

PAY_STATUS_ISOLATED_COLUMN: Final[str] = PAY_STATUS_COLUMNS[0]
"""The repayment-status month treated on its own: month 1, September 2005.

Not excluded - it is the single most predictive column in the block, with a Spearman of
0.292 against the target versus 0.143 to 0.217 for the rest. It is kept as a variable of
its own rather than folded into a trajectory that assumes a scale it does not share.
"""


@dataclass(frozen=True)
class NumericRange:
    """Plausible closed interval for a numeric column, with the reason for its bounds.

    A bound is not the observed minimum or maximum. Fitting the interval to the data
    would produce a check that can never fire. A bound answers a different question: past
    this point the value is no longer a business fact but a data error - a unit change, a
    corrupt row, a shifted column.

    Attributes:
        minimum: Lowest admissible value, or `None` when no lower bound is asserted.
        maximum: Highest admissible value, or `None` when no upper bound is asserted.
        rationale: Why these bounds and not others. Quoted in the data dictionary.
    """

    minimum: int | None
    maximum: int | None
    rationale: str


_AMOUNT_CEILING: Final[int] = 5_000_000
"""Ceiling shared by every NT$ amount, about three times the largest value at the source.

Wide enough never to fire on a legitimate amount, tight enough that a currency-unit change
- NT dollars read as cents multiplies every amount by one hundred - breaks it immediately.
"""

NUMERIC_RANGES: Final[Mapping[str, NumericRange]] = MappingProxyType(
    {
        "ID": NumericRange(
            minimum=1,
            maximum=None,
            rationale=(
                "Row identifier assigned by the source, 1-based. No upper bound is "
                "asserted: the identifier carries no business meaning, so any ceiling "
                "would be arbitrary and a larger extract would legitimately exceed it."
            ),
        ),
        "LIMIT_BAL": NumericRange(
            minimum=1,
            maximum=_AMOUNT_CEILING,
            rationale=(
                "Granted credit in NT dollars, individual plus family supplementary. "
                "Zero or negative is not a credit limit but an absent or corrupt record. "
                "Observed at the source: 10,000 to 1,000,000."
            ),
        ),
        "AGE": NumericRange(
            minimum=18,
            maximum=120,
            rationale=(
                "Years. A card holder is an adult, and 18 is the most permissive "
                "adult threshold in use anywhere, so the floor cannot reject a "
                "legitimate record; 120 is beyond any recorded human lifespan. The "
                "exact legal minimum in Taiwan in 2005 was not verified, so it is "
                "not claimed here. Observed at the source: 21 to 79."
            ),
        ),
        **{
            column: NumericRange(
                minimum=-_AMOUNT_CEILING,
                maximum=_AMOUNT_CEILING,
                rationale=(
                    "Bill statement in NT dollars. The lower bound is deliberately "
                    "negative and symmetric: a negative statement is legitimate and "
                    "frequent, because an overpayment or a refund leaves the account in "
                    "credit. Measured at the source, not assumed: 3,932 negative "
                    "statements spread over 1,930 rows, about 2% of rows per month, "
                    "down to -339,603. A floor of zero here would reject all of them."
                ),
            )
            for column in BILL_AMOUNT_COLUMNS
        },
        **{
            column: NumericRange(
                minimum=0,
                maximum=_AMOUNT_CEILING,
                rationale=(
                    "Amount paid in NT dollars. Unlike the bill statement this floor is "
                    "zero, and the asymmetry is the point: money moving back to the "
                    "client shows up as a credit on the statement, never as a negative "
                    "payment. Verified on the source data, not assumed: zero negative "
                    "values across the six months."
                ),
            )
            for column in PAY_AMOUNT_COLUMNS
        },
    }
)
"""Canonical numeric column -> plausible range.

Together with `CATEGORICAL_LEVELS` this partitions the 25 canonical columns: every column
is covered by exactly one of the two maps, never by both and never by neither.
"""
