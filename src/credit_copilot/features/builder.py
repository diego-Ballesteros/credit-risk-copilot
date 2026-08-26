"""Payment-behaviour features: the derived variables the main hypothesis is tested with.

The project's main hypothesis, in `docs/ROADMAP.md`, is that recent payment behaviour
predicts default better than static demographic attributes. None of the variables that
would test it exist in the source table: the source ships six monthly repayment codes,
six bill statements and six payments, and every behavioural quantity - how much of the
limit is being used, whether that is rising, how much of the bill is actually paid, how
long the client has been in arrears - has to be derived. This module derives them.

**Why a transformer and not a function.** The same transformation has to run in the
exploratory notebook, in the training script and behind the API. A loose function gets
reimplemented at the second call site and the two copies drift, which is the failure
described in section 6.3 of `docs/METHODOLOGY.md`: everything green in the notebook,
garbage predictions in production. As a `Pipeline` step there is one object, serialised
once and loaded by all three.

**Why month 1 is kept apart.** ADR-0004 section 3 measured three independent signs that
`PAY_STATUS_1` does not share the low-end scale of months 2 to 6, so no feature here
treats the six columns as one homogeneous panel. The block features read
`schema.PAY_STATUS_HOMOGENEOUS_COLUMNS`; month 1 gets two variables of its own. The
separation is structural rather than remembered: `_block_features` is handed a frame that
does not contain the month-1 columns at all, so a block feature that reached for one
would raise `KeyError` instead of quietly mixing two scales.

**Why nothing is learned here.** Every feature is a row-wise arithmetic function of that
same row. `fit` stores no statistic, which means this step cannot carry information from
the training fold into the validation fold no matter how a cross-validation is wired.
That is a level-1 guarantee in the sense of section 6.5 of `docs/METHODOLOGY.md` - the
tool makes the leak impossible - and it is the reason `fit` never reads `y`.

**Why non-computable is a column and not a value.** A ratio whose denominator is not
usable is not zero and it is not "no usage": it is unknown. Writing a number there would
turn an absence into a business fact, which section 7.1 of `docs/METHODOLOGY.md` calls
the silent imputation. The ratio is left as `NaN` and the fact that it could not be
computed is emitted as its own indicator column, so a downstream model can use the
absence as the signal it usually is. **No imputation happens in this module.** What
counts as usable is `MINIMUM_DENOMINATOR_NTD`, decided in ADR-0005.
"""

import re
from collections.abc import Iterable, Mapping
from typing import Final

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_copilot.data import schema

# ---------------------------------------------------------------------------
# Month indices, derived from the contract and never written out by hand
# ---------------------------------------------------------------------------

_MONTH_SUFFIX: Final[re.Pattern[str]] = re.compile(r"(\d+)$")


def _by_month(columns: Iterable[str]) -> Mapping[int, str]:
    """Index a monthly column block by its month number.

    Args:
        columns: Canonical column names ending in the panel month index.

    Returns:
        Month index -> column name.

    Raises:
        ValueError: If a name does not end in a month index.
    """
    indexed: dict[int, str] = {}
    for column in columns:
        match = _MONTH_SUFFIX.search(column)
        if match is None:
            raise ValueError(f"Column {column!r} does not end in a panel month index.")
        indexed[int(match.group(1))] = column
    return indexed


_PAY_STATUS_BY_MONTH: Final[Mapping[int, str]] = _by_month(schema.PAY_STATUS_COLUMNS)
_BILL_BY_MONTH: Final[Mapping[int, str]] = _by_month(schema.BILL_AMOUNT_COLUMNS)
_PAY_AMOUNT_BY_MONTH: Final[Mapping[int, str]] = _by_month(schema.PAY_AMOUNT_COLUMNS)

BLOCK_MONTHS: Final[tuple[int, ...]] = tuple(
    sorted(_by_month(schema.PAY_STATUS_HOMOGENEOUS_COLUMNS))
)
"""Months of the homogeneous repayment block, most recent first: 2 to 6, August to April.

Read off `schema.PAY_STATUS_HOMOGENEOUS_COLUMNS` rather than typed here, so that the one
place ADR-0004 is encoded stays the one place it can be changed.
"""

ISOLATED_MONTH: Final[int] = next(iter(_by_month([schema.PAY_STATUS_ISOLATED_COLUMN])))
"""The month treated on its own: 1, September 2005, the most recent."""

_CHRONOLOGICAL_BLOCK_MONTHS: Final[tuple[int, ...]] = tuple(sorted(BLOCK_MONTHS, reverse=True))
"""The block in the order time runs: oldest month first, because index 6 is April."""

PAYMENT_RATIO_MONTHS: Final[tuple[int, ...]] = tuple(
    month for month in BLOCK_MONTHS if month + 1 in BLOCK_MONTHS
)
"""Block months whose *previous* month is also inside the block: 2 to 5.

The payment ratio divides what was paid in a month by the balance of the month before it,
and the month before is the one with the **higher** index. Month 6 is the oldest month of
the block, so its predecessor lies outside the block and its ratio is not built. Reaching
one month further back would be reaching outside the homogeneous window on the balance
side while staying inside it on the payment side, and the window is not decoration.
"""

MINIMUM_DENOMINATOR_NTD: Final[int] = 100
"""Floor a denominator must exceed, in NT dollars, for its ratio to be computable.

Decided in ADR-0005. **A denominator has to be greater than this floor, not merely
different from zero**, and the rule is strict: exactly 100 NT$ is *not* usable, so the
floor is the largest excluded value. Strictness is the same shape the previous `> 0`
policy had, with the excluded value moved; a floor that admitted its own value would
make the constant read as "the smallest usable denominator", which is not what it is.

The unit is in the name because the number is only meaningful with it. 100 NT$ is a
trivial fraction of the smallest credit limit in this dataset, which is 10,000 NT$, so
the threshold cannot discard meaningful behaviour - what it discards is arithmetic that
is correct and says nothing, such as a payment of 10,002 NT$ against a previous balance
of 2 NT$ scoring 5,001 on a scale where every other value is a coverage fraction.

Deliberately a module constant and not a constructor parameter, for the same reason as
`DELINQUENCY_THRESHOLD`: it transcribes a decision with an ADR behind it, and a
hyperparameter is something a tuner is allowed to move.
"""

DELINQUENCY_THRESHOLD: Final[int] = 1
"""Lowest repayment code that counts as being in arrears, per ADR-0004.

Codes -2, -1 and 0 are *not* arrears: -2 is no consumption, -1 is paid in full and 0 is
revolving credit carried without delay. Deliberately a module constant and not a
constructor parameter: it transcribes a decision with measured evidence behind it, and a
hyperparameter is something a tuner is allowed to move.
"""

# ---------------------------------------------------------------------------
# Source columns, split by the boundary ADR-0004 draws
# ---------------------------------------------------------------------------

SHARED_SOURCE_COLUMNS: Final[tuple[str, ...]] = ("LIMIT_BAL",)
"""Columns that belong to no month. The credit limit is a property of the account."""

BLOCK_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    *SHARED_SOURCE_COLUMNS,
    *(_PAY_STATUS_BY_MONTH[m] for m in BLOCK_MONTHS),
    *(_BILL_BY_MONTH[m] for m in BLOCK_MONTHS),
    *(_PAY_AMOUNT_BY_MONTH[m] for m in BLOCK_MONTHS),
)
"""Everything the homogeneous-block features are allowed to read. Month 1 is absent."""

ISOLATED_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    *SHARED_SOURCE_COLUMNS,
    _PAY_STATUS_BY_MONTH[ISOLATED_MONTH],
    _BILL_BY_MONTH[ISOLATED_MONTH],
)
"""Everything the month-1 features are allowed to read. The block is absent.

`PAY_AMT1` is missing on purpose, and its absence is the isolation rule costing something
measurable: a month-1 payment ratio would need the month-2 balance as its denominator,
which crosses the boundary this module exists to hold.
"""

REQUIRED_COLUMNS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys((*BLOCK_SOURCE_COLUMNS, *ISOLATED_SOURCE_COLUMNS))
)
"""Every column `transform` reads, in a stable order and without repetition.

The target is not here and cannot be: `transform` selects this tuple out of its input and
looks at nothing else, so the transformer is structurally unable to see the label even
when it is handed a frame that still carries it.
"""

# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

_BLOCK_SPAN: Final[str] = f"M{BLOCK_MONTHS[0]}_M{BLOCK_MONTHS[-1]}"

UTILIZATION_FEATURES: Final[tuple[str, ...]] = tuple(f"UTILIZATION_M{m}" for m in BLOCK_MONTHS)
UTILIZATION_TREND_FEATURE: Final[str] = f"UTILIZATION_TREND_{_BLOCK_SPAN}"
PAYMENT_RATIO_FEATURES: Final[tuple[str, ...]] = tuple(
    f"PAYMENT_RATIO_M{m}" for m in PAYMENT_RATIO_MONTHS
)
PAYMENT_RATIO_NOT_COMPUTABLE_FEATURES: Final[tuple[str, ...]] = tuple(
    f"PAYMENT_RATIO_NOT_COMPUTABLE_M{m}" for m in PAYMENT_RATIO_MONTHS
)
DELINQUENCY_STREAK_FEATURE: Final[str] = f"DELINQUENCY_STREAK_{_BLOCK_SPAN}"
MAX_DELINQUENCY_FEATURE: Final[str] = f"MAX_DELINQUENCY_{_BLOCK_SPAN}"
BILL_VOLATILITY_FEATURE: Final[str] = f"BILL_VOLATILITY_{_BLOCK_SPAN}"
UTILIZATION_VOLATILITY_FEATURE: Final[str] = f"UTILIZATION_VOLATILITY_{_BLOCK_SPAN}"
MONTHS_WITHOUT_PAYMENT_FEATURE: Final[str] = f"MONTHS_WITHOUT_PAYMENT_{_BLOCK_SPAN}"
UTILIZATION_MOST_RECENT_FEATURE: Final[str] = f"UTILIZATION_MOST_RECENT_M{ISOLATED_MONTH}"
IS_DELINQUENT_MOST_RECENT_FEATURE: Final[str] = f"IS_DELINQUENT_MOST_RECENT_M{ISOLATED_MONTH}"
UTILIZATION_NOT_COMPUTABLE_FEATURE: Final[str] = "UTILIZATION_NOT_COMPUTABLE"

FEATURE_NAMES: Final[tuple[str, ...]] = (
    *UTILIZATION_FEATURES,
    UTILIZATION_TREND_FEATURE,
    *PAYMENT_RATIO_FEATURES,
    *PAYMENT_RATIO_NOT_COMPUTABLE_FEATURES,
    DELINQUENCY_STREAK_FEATURE,
    MAX_DELINQUENCY_FEATURE,
    BILL_VOLATILITY_FEATURE,
    UTILIZATION_VOLATILITY_FEATURE,
    MONTHS_WITHOUT_PAYMENT_FEATURE,
    UTILIZATION_MOST_RECENT_FEATURE,
    IS_DELINQUENT_MOST_RECENT_FEATURE,
    UTILIZATION_NOT_COMPUTABLE_FEATURE,
)
"""The 22 columns `transform` produces, in the order it produces them.

Declared once and used by both `transform` and `get_feature_names_out`, because a
labelled column that does not hold what the label says is worse than an unlabelled one:
SHAP would attribute the explanation to the wrong variable and nothing would look broken.
"""

# ---------------------------------------------------------------------------
# The denominator-floor policy, written once
# ---------------------------------------------------------------------------


def _ratio_over_usable_denominator(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide, declaring the result unknown wherever the denominator is not usable.

    **The policy is `denominator > MINIMUM_DENOMINATOR_NTD`, not `denominator != 0`, and
    the two steps away from the naive rule are both measured.** ADR-0005 holds the
    evidence; the summary is that a denominator can be unusable in two different ways.

    *It can point the wrong way.* A previous bill statement of exactly zero means there
    was nothing to cover, so "what fraction was covered" has no answer. A *negative*
    previous statement means the account was in credit - an overpayment or a refund,
    measured in 3,932 rows of this dataset - and dividing by it flips the sign, so a
    client who paid 5,000 against a balance of -2,000 would score -2.5 on a scale where
    every other value is a coverage fraction. That is not a small ratio, it is a
    different quantity wearing the same column, and a model would read it as extreme
    good behaviour.

    *It can be positive and still trivial.* The extreme case measured on this file is a
    payment of 10,002 NT$ against a previous balance of 2 NT$, which divides to 5,001.
    Arithmetically correct, and as a statement about repayment behaviour it says nothing
    at all - it says the client had almost no bill. A rule of `> 0` admits it.

    The unknown is written as `NaN` and never as `0`. Zero is a measurement - "paid
    nothing" - and this is the absence of one. The caller emits the companion indicator
    column that makes the absence visible to the model.

    Args:
        numerator: Amount on top of the ratio, in NT$.
        denominator: Amount underneath it, in NT$.

    Returns:
        The elementwise ratio, `NaN` wherever the denominator does not clear the floor.
    """
    usable = denominator.where(denominator > MINIMUM_DENOMINATOR_NTD)
    quotient: pd.Series = numerator.div(usable)
    return quotient


def _is_unusable_denominator(values: pd.Series) -> pd.Series:
    """Flag the rows where a denominator makes its ratio non-computable.

    The complement of the condition in `_ratio_over_usable_denominator`, and written next
    to it so the indicator column cannot drift away from the rule it reports.

    Args:
        values: The denominator column, in NT$.

    Returns:
        1 where the value does not clear `MINIMUM_DENOMINATOR_NTD`, 0 otherwise, as
        `int64`.
    """
    flag: pd.Series = (values <= MINIMUM_DENOMINATOR_NTD).astype("int64")
    return flag


# ---------------------------------------------------------------------------
# The three feature groups
# ---------------------------------------------------------------------------


def _shared_features(shared: pd.DataFrame) -> dict[str, pd.Series]:
    """Build the features that belong to no month.

    `UTILIZATION_NOT_COMPUTABLE` marks the accounts whose credit limit does not clear
    `MINIMUM_DENOMINATOR_NTD`, which makes **every** utilisation ratio in this module
    unknown - the five block months and month 1 alike, because all six share one
    denominator. It is a single column rather than one per month precisely because
    `LIMIT_BAL` has no month: duplicating it per month would emit perfectly collinear
    columns and would suggest a time dimension the underlying fact does not have. For the
    same reason it does not cross the month-1 boundary - there is no month in it to cross
    with.

    In business terms: an account with no meaningful granted limit is not an account with
    zero usage, it is an account whose usage cannot be expressed as a fraction of
    anything. `schema.NUMERIC_RANGES` declares `LIMIT_BAL >= 1` and the smallest limit in
    the file is 10,000 NT$, so a row that trips this flag is a row the validator should
    already have rejected; the flag exists so that the pipeline states the assumption
    instead of relying on it. ADR-0005 decision 5 records why it is kept despite having
    zero variance on this dataset.

    Args:
        shared: Frame containing `SHARED_SOURCE_COLUMNS`.

    Returns:
        Feature name -> column.
    """
    return {UTILIZATION_NOT_COMPUTABLE_FEATURE: _is_unusable_denominator(shared["LIMIT_BAL"])}


def _utilization_trend(utilization: Mapping[str, pd.Series]) -> pd.Series:
    """Slope of the ordinary least-squares line through the block's utilisation.

    **The chosen definition.** The five monthly utilisations are regressed on time and the
    slope is kept, in utilisation points per month, oriented so that a positive value
    means the client is using more of the limit as time runs forward. Because the months
    are equally spaced the slope has a closed form - a fixed weighted sum of the five
    values over a constant - so nothing is fitted, nothing iterates, and the constant
    denominator can never be zero. This definition therefore adds **no** new
    non-computable case: the trend is unknown exactly when its inputs are, which is when
    the credit limit does not clear `MINIMUM_DENOMINATOR_NTD`. ADR-0005 decision 1
    records the choice and the three definitions it was chosen over.

    **Why not the endpoint difference, `UTILIZATION_M2 - UTILIZATION_M6`.** It is
    determined by two months and throws the other three away, and the two it keeps are the
    worst pair to depend on: they are the block's edges, where a single large purchase in
    April or one settlement in August moves the whole feature. The regression lets the
    three middle months contradict a freak endpoint.

    **Why not the mean of consecutive differences.** It telescopes to the endpoint
    difference divided by four. Same defect, different units - it looks like it uses all
    five months and does not.

    **Why not the ratio of last to first.** It reintroduces a zero denominator, and a
    frequent one: utilisation is exactly 0 whenever the statement is 0, which is the
    normal state under repayment code -2. It is also unbounded, so a client going from 1%
    to 10% would outrank one going from 40% to 95%, which inverts the business reading.

    Args:
        utilization: The block's monthly utilisation columns, keyed by feature name.

    Returns:
        Slope in utilisation points per month, `NaN` wherever the inputs are unknown.
    """
    chronological = [utilization[f"UTILIZATION_M{month}"] for month in _CHRONOLOGICAL_BLOCK_MONTHS]
    positions = list(range(len(chronological)))
    centre = sum(positions) / len(positions)
    deviations = [position - centre for position in positions]
    spread = sum(deviation**2 for deviation in deviations)

    frame = pd.concat(chronological, axis=1)
    weights = pd.Series(deviations, index=frame.columns, dtype="float64")
    # `skipna=False` is the whole point: where the limit is not positive every term is
    # NaN, and a skipping sum would silently report a perfectly flat trend of 0.0.
    trend: pd.Series = frame.mul(weights, axis=1).sum(axis=1, skipna=False) / spread
    return trend


def _utilization_volatility(utilization: Mapping[str, pd.Series]) -> pd.Series:
    """Population standard deviation of the block's utilisation, in utilisation points.

    **Why this exists next to `BILL_VOLATILITY_M2_M6` rather than instead of it.**
    Volatility measured in NT$ is confounded with the size of the credit limit: a client
    with a 500,000 NT$ limit whose statement swings by 50,000 and a client with a 50,000
    NT$ limit whose statement swings by 5,000 are running the *same* habit, and the NT$
    version ranks the first one ten times more erratic. Dividing by the limit before
    taking the deviation removes the scale, so what is left is how erratic the client is
    relative to the room they were given.

    That does not make the NT$ version wrong. An absolute swing of 50,000 NT$ is a larger
    amount of money at risk than one of 5,000 whatever the limit says, so the two columns
    answer different questions and the useful one is an empirical matter. **Both are
    built and the choice is deferred to the modelling turn, where it can be made on
    measured predictive contribution instead of on the argument above.** Dropping one now
    would be settling with a story what should be settled with evidence.

    The divisor is the population one, matching `BILL_VOLATILITY_M2_M6`: the five months
    are the whole window being described, not a sample drawn from a larger one. Keeping
    the two divisors equal is what makes the pair comparable at all.

    Args:
        utilization: The block's monthly utilisation columns, keyed by feature name.

    Returns:
        Standard deviation in utilisation points, `NaN` wherever the inputs are unknown.
    """
    frame = pd.concat([utilization[name] for name in UTILIZATION_FEATURES], axis=1)
    # Explicit rather than relying on the default skipping behaviour: where the limit
    # does not clear the floor every month is NaN, and a skipping deviation would report
    # a perfectly stable 0.0 for a client whose usage is not known at all.
    volatility: pd.Series = frame.std(axis=1, ddof=0).where(frame.notna().all(axis=1))
    return volatility.astype("float64")


def _block_features(block: pd.DataFrame) -> dict[str, pd.Series]:
    """Build every feature of the homogeneous repayment block, months 2 to 6.

    The frame this receives contains `BLOCK_SOURCE_COLUMNS` and nothing else, so month 1
    is not merely unused here - it is unreachable.

    Features, and what each one measures about the client:

    - **`UTILIZATION_M2..M6`** - the share of the granted limit the balance occupies in
      each month. A client sitting near the top of the limit has no headroom left, which
      is the ordinary shape of stress before a missed payment. Unknown where the limit
      does not clear the floor; see `UTILIZATION_NOT_COMPUTABLE`.
    - **`UTILIZATION_TREND_M2_M6`** - whether that occupation is filling up or draining,
      in utilisation points per month. Positive means the client is consuming more of the
      limit as time runs forward; negative means recovering. Level answers *how deep*,
      trend answers *which way*, and a client at 40% on the way up is a different risk
      from one at 40% on the way down.
    - **`PAYMENT_RATIO_M2..M5`** - what fraction of the previous month's statement the
      client actually paid. This is the sharpest behavioural separator ADR-0004 measured:
      a median of 1.000 for the codes that settle the statement against 0.042 to 0.057 for
      revolving credit. Paying the whole bill and paying the minimum are different
      businesses, and the raw amounts cannot tell them apart because they do not know the
      size of the bill.
    - **`DELINQUENCY_STREAK_M2_M6`** - how many consecutive months the client was in
      arrears counting back from the most recent month of the block. Persistence, not
      severity: one late month is an accident, four in a row is a trajectory. The count
      stops at the first month that is not in arrears, so a client who fell behind and
      recovered scores zero however bad the earlier stretch was - that stretch is what
      `MAX_DELINQUENCY_M2_M6` is for.
    - **`MAX_DELINQUENCY_M2_M6`** - the worst delay reached anywhere in the block, in
      months. Codes below the arrears threshold are floored at zero first, so `-2`, `-1`
      and `0` all read as "not in arrears" instead of being ranked against each other.
      **This is not a claim that the repayment codes are ordinal.** ADR-0004 section 2
      showed they are not, in the low zone: code 0 defaults at 12.81% against 16.78% for
      code -1. The same ADR measured that the zone at and above the threshold *is*
      consistent across months and does read as a delay count, and the maximum is taken
      only there.
    - **`BILL_VOLATILITY_M2_M6`** - how erratic the balance is across the block, in NT$,
      as a population standard deviation. A client whose statement barely moves is running
      a stable habit; one whose statement swings is either a heavy irregular user or an
      account under strain. The population divisor is deliberate: the five months are the
      whole window being described, not a sample drawn from a larger one.
    - **`UTILIZATION_VOLATILITY_M2_M6`** - the same erraticism measured in utilisation
      points instead of NT$, so that it does not carry the size of the credit limit
      inside it. It coexists with the NT$ version on purpose; see
      `_utilization_volatility` for why neither one is dropped yet.
    - **`MONTHS_WITHOUT_PAYMENT_M2_M6`** - how many of the five months recorded no payment
      at all. Distinct from the repayment codes, which describe the *state* of the
      account; this counts a concrete event, and a zero payment is the event that precedes
      the state.

    Args:
        block: Frame containing `BLOCK_SOURCE_COLUMNS`.

    Returns:
        Feature name -> column.
    """
    limit = block["LIMIT_BAL"]

    utilization = {
        name: _ratio_over_usable_denominator(block[_BILL_BY_MONTH[month]], limit)
        for name, month in zip(UTILIZATION_FEATURES, BLOCK_MONTHS, strict=True)
    }
    payment_ratio = {
        name: _ratio_over_usable_denominator(
            block[_PAY_AMOUNT_BY_MONTH[month]],
            block[_BILL_BY_MONTH[month + 1]],
        )
        for name, month in zip(PAYMENT_RATIO_FEATURES, PAYMENT_RATIO_MONTHS, strict=True)
    }
    payment_ratio_unknown = {
        name: _is_unusable_denominator(block[_BILL_BY_MONTH[month + 1]])
        for name, month in zip(
            PAYMENT_RATIO_NOT_COMPUTABLE_FEATURES, PAYMENT_RATIO_MONTHS, strict=True
        )
    }

    status = block[[_PAY_STATUS_BY_MONTH[month] for month in BLOCK_MONTHS]]
    in_arrears = (status >= DELINQUENCY_THRESHOLD).astype("int64")
    bills = block[[_BILL_BY_MONTH[month] for month in BLOCK_MONTHS]]
    payments = block[[_PAY_AMOUNT_BY_MONTH[month] for month in BLOCK_MONTHS]]

    return {
        **utilization,
        UTILIZATION_TREND_FEATURE: _utilization_trend(utilization),
        **payment_ratio,
        **payment_ratio_unknown,
        # `cumprod` along the row turns the leading run of 1s into 1s and everything from
        # the first 0 onwards into 0, so the sum is the length of that leading run. The
        # columns are ordered most recent first, so the run counted is the one that
        # reaches the end of the block.
        DELINQUENCY_STREAK_FEATURE: in_arrears.cumprod(axis=1).sum(axis=1).astype("int64"),
        MAX_DELINQUENCY_FEATURE: status.clip(lower=0).max(axis=1).astype("int64"),
        BILL_VOLATILITY_FEATURE: bills.std(axis=1, ddof=0).astype("float64"),
        UTILIZATION_VOLATILITY_FEATURE: _utilization_volatility(utilization),
        MONTHS_WITHOUT_PAYMENT_FEATURE: (payments == 0).sum(axis=1).astype("int64"),
    }


def _isolated_features(isolated: pd.DataFrame) -> dict[str, pd.Series]:
    """Build the two features of month 1, the most recent month, on their own.

    The frame this receives contains `ISOLATED_SOURCE_COLUMNS` and nothing else, so the
    block is unreachable from here in the same way month 1 is unreachable from the block.

    Month 1 is not excluded from the model - ADR-0004 records it as the single most
    predictive repayment column, with a Spearman of 0.292 against the target against 0.143
    to 0.217 for the rest. It is kept as variables of its own because it does not share
    the low-end scale of the other five, so folding it into a trajectory would average two
    different scales together.

    Features:

    - **`UTILIZATION_MOST_RECENT_M1`** - the share of the limit occupied in September
      2005, the month immediately before the one the target is about. The same quantity as
      the block's utilisations and deliberately not averaged with them: it is the closest
      observation to the outcome and the one a credit officer would read first.
    - **`IS_DELINQUENT_MOST_RECENT_M1`** - whether the client was in arrears in that last
      observed month, by the ADR-0004 threshold. Binary rather than the raw code, because
      the raw code is categorical: its numeric order is not an order of severity, and
      handing the model the integer would teach it a monotonicity the data denies. The
      code itself is not lost - it goes to one-hot encoding elsewhere in the pipeline.

    Args:
        isolated: Frame containing `ISOLATED_SOURCE_COLUMNS`.

    Returns:
        Feature name -> column.
    """
    utilization = _ratio_over_usable_denominator(
        isolated[_BILL_BY_MONTH[ISOLATED_MONTH]],
        isolated["LIMIT_BAL"],
    )
    in_arrears: pd.Series = (
        isolated[_PAY_STATUS_BY_MONTH[ISOLATED_MONTH]] >= DELINQUENCY_THRESHOLD
    ).astype("int64")
    return {
        UTILIZATION_MOST_RECENT_FEATURE: utilization,
        IS_DELINQUENT_MOST_RECENT_FEATURE: in_arrears,
    }


# ---------------------------------------------------------------------------
# The transformer
# ---------------------------------------------------------------------------


class MissingSourceColumnsError(ValueError):
    """The input frame lacks columns the payment-behaviour features are built from."""


class PaymentBehaviourFeatures(BaseEstimator, TransformerMixin):
    """Derive the payment-behaviour features from the canonical dataset.

    A `Pipeline` step, not a helper: the notebook, the training script and the API all
    load the same fitted object, so the transformation cannot be reimplemented differently
    at a second call site.

    The estimator has no hyperparameters. Everything it does is fixed by ADR-0004,
    ADR-0005 and the contract in `schema`, and a decision with measured evidence behind
    it is not something a tuner should be able to move.

    `transform` returns only the derived columns, never the columns it read. In a
    `ColumnTransformer` the raw columns are routed to their own branches - one-hot for the
    categorical blocks, scaling for the numeric ones - and passing them through here as
    well would feed them into the matrix twice.

    Attributes:
        n_features_in_: Number of columns seen during `fit`.
        feature_names_in_: Column names seen during `fit`.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "PaymentBehaviourFeatures":  # noqa: ARG002, N803
        """Check that the input carries what the features are built from. Learn nothing.

        No statistic is estimated here, and that is a property worth stating rather than
        an omission: with nothing carried out of `fit`, this step cannot move information
        from a training fold into a validation fold.

        `y` is accepted because the `Pipeline` API passes it positionally, and it is never
        read, stored or inspected. Ignoring it is not a convention here - a feature of this
        kind that consulted the label would be leakage by construction.

        Args:
            X: Canonical table, as returned by `loader.load_dataset`.
            y: Ignored. Present for API compatibility only.

        Returns:
            The fitted transformer.

        Raises:
            TypeError: If `X` is not a `pandas.DataFrame`.
            MissingSourceColumnsError: If a required column is absent.
        """
        self._check_frame(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Build the derived features for every row.

        Args:
            X: Canonical table carrying `REQUIRED_COLUMNS`. Extra columns are ignored, and
                column order does not matter: every column is addressed by name.

        Returns:
            A frame of `FEATURE_NAMES`, in that order, sharing the index of `X`.

        Raises:
            TypeError: If `X` is not a `pandas.DataFrame`.
            MissingSourceColumnsError: If a required column is absent.
            RuntimeError: If the produced columns do not match `FEATURE_NAMES`.
        """
        self._check_frame(X)
        features: dict[str, pd.Series] = {
            **_block_features(X[list(BLOCK_SOURCE_COLUMNS)]),
            **_isolated_features(X[list(ISOLATED_SOURCE_COLUMNS)]),
            **_shared_features(X[list(SHARED_SOURCE_COLUMNS)]),
        }
        if set(features) != set(FEATURE_NAMES):
            raise RuntimeError(
                "The produced columns do not match the declared feature names. "
                f"Missing: {sorted(set(FEATURE_NAMES) - set(features))}. "
                f"Unexpected: {sorted(set(features) - set(FEATURE_NAMES))}."
            )
        return pd.DataFrame(features, index=X.index)[list(FEATURE_NAMES)]

    def get_feature_names_out(self, input_features: object = None) -> pd.Index:  # noqa: ARG002
        """Name the columns `transform` produces, in the order it produces them.

        Without this a `ColumnTransformer` labels the block positionally and SHAP has
        nothing to attribute an explanation to. It does not depend on `input_features`,
        because the output of this step is fixed by the contract and not by what the step
        was handed.

        Args:
            input_features: Ignored. Present for API compatibility only.

        Returns:
            The names in `FEATURE_NAMES`.
        """
        return pd.Index(FEATURE_NAMES, dtype=object)

    @staticmethod
    def _check_frame(frame: pd.DataFrame) -> None:
        """Fail loudly and completely on an input the features cannot be built from.

        Every missing column is listed at once rather than the first one found: a caller
        who has renamed a block gets the whole picture in one run instead of one column
        per attempt.

        Args:
            frame: The candidate input.

        Raises:
            TypeError: If it is not a `pandas.DataFrame`.
            MissingSourceColumnsError: If a required column is absent.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                "PaymentBehaviourFeatures needs a pandas DataFrame: it addresses columns "
                f"by name and a {type(frame).__name__} has none."
            )
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise MissingSourceColumnsError(
                f"Missing columns required to build the payment-behaviour features: {missing}. "
                "Load the data through loader.load_dataset, which applies the canonical names."
            )
