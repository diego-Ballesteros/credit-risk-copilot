"""The single preprocessing `Pipeline`: one raw row in, one model-ready matrix out.

**Why this module is one object and not a sequence of helper calls.** Section 6.3 of
`docs/METHODOLOGY.md` names the most treacherous failure mode in production ML: the
notebook's preprocessing and the serving script's preprocessing drift apart, everything
stays green in the notebook, and the API returns garbage. The defence is structural
rather than remembered - there is exactly one `Pipeline` object, it is serialised once,
and the notebook, the training script and the API all load *that artefact*. A second
implementation of "the same" arithmetic cannot drift from the first if it does not exist.

**Why every statistic is learned inside `fit`.** Four steps here learn something from the
data: the clipping thresholds, the imputation medians, the scaler's median and IQR, and
the one-hot vocabulary. All four learn it in `fit` and apply it in `transform`, which is
what lets the whole object sit inside a cross-validation without carrying information
from a training fold into a validation fold. Section 6.5 of the methodology calls this a
level-1 guarantee: the tool makes the leak impossible instead of a document asking for it
not to happen. Nothing in this module reads the full dataset at import time or at build
time.

**Why the target cannot reach the matrix.** The final `ColumnTransformer` selects its
columns by name and drops everything it was not asked for, and the target is in none of
the three lists. A frame that still carries the label can be handed to `fit` and the label
still cannot become a feature - it is dropped, not ignored by convention.

**Why every column can be named.** `get_feature_names_out` works end to end, through the
custom steps as well as the sklearn ones. Without that, a `ColumnTransformer` labels its
output positionally and SHAP attributes an explanation to whichever variable happens to
sit at that index - an error that produces a plausible, well-formatted, wrong story about
why a client was refused credit.

The decisions with alternatives behind them - the clipping percentile, the scaler, the
imputation policy and the unknown-category policy - are recorded with their measurements
in the docstrings below, and their reasoning belongs in an ADR the Architect writes.
"""

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from credit_copilot.data import schema
from credit_copilot.features.builder import (
    FEATURE_NAMES,
    IS_DELINQUENT_MOST_RECENT_FEATURE,
    PAYMENT_RATIO_FEATURES,
    PAYMENT_RATIO_NOT_COMPUTABLE_FEATURES,
    UTILIZATION_NOT_COMPUTABLE_FEATURE,
    PaymentBehaviourFeatures,
)

# ---------------------------------------------------------------------------
# The three column groups, derived from the contract and never typed by hand
# ---------------------------------------------------------------------------

CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    *schema.PAY_STATUS_COLUMNS,
)
"""Columns that go to one-hot encoding and **never** to a scaler.

The six `PAY_STATUS_*` columns are here because ADR-0004 decision 2 measured that their
numeric order is not an order of severity - code `0` defaults at 12.81% against 16.78%
for `-1` and 13.23% for `-2`, so the sequence `-2, -1, 0` is not monotone in risk. Handing
a model the integer would teach it a monotonicity the data denies. The ADR calls this
consequence "direct and non-negotiable", and this tuple is where it is enforced.
"""

NUMERIC_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "LIMIT_BAL",
    "AGE",
    *schema.BILL_AMOUNT_COLUMNS,
    *schema.PAY_AMOUNT_COLUMNS,
)
"""Source columns that are genuine quantities: money in NT$ and one age in years."""

INDICATOR_COLUMNS: Final[tuple[str, ...]] = (
    *PAYMENT_RATIO_NOT_COMPUTABLE_FEATURES,
    UTILIZATION_NOT_COMPUTABLE_FEATURE,
    IS_DELINQUENT_MOST_RECENT_FEATURE,
)
"""Derived columns that are already 0/1 and pass through untouched.

Scaling a binary column is not wrong, it is pointless: it rescales two values by their own
spread and costs the only property that made them readable, which is that a 1 means the
thing happened. `UTILIZATION_NOT_COMPUTABLE` is constant at 0 on this dataset, and both
`StandardScaler` and `RobustScaler` fall back to a scale of 1 on a degenerate column, so
even the arithmetic argument for scaling it is empty.
"""

NUMERIC_DERIVED_COLUMNS: Final[tuple[str, ...]] = tuple(
    name for name in FEATURE_NAMES if name not in INDICATOR_COLUMNS
)
"""The payment-behaviour features that carry a magnitude rather than a flag."""

NUMERIC_COLUMNS: Final[tuple[str, ...]] = (*NUMERIC_SOURCE_COLUMNS, *NUMERIC_DERIVED_COLUMNS)
"""Everything the numeric branch imputes and scales: 14 source columns and 16 derived."""

DEFAULT_CLIP_PERCENTILE: Final[float] = 99.5
"""Upper percentile at which the payment ratios are capped, learned per column in `fit`.

**Why the ratios need capping at all.** ADR-0005 decision 2 put a floor of 100 NT$ under
every denominator and measured that it removes the pathological case and not the tail: the
maximum of `PAYMENT_RATIO_M4` stayed at 129.71 on a previous balance of 780 NT$, and
`PAYMENT_RATIO_M5` still reaches 447.74 on a balance of 291 NT$. A denominator can clear
100 NT$ and still be small enough that the ratio measures the size of the bill rather than
the discipline of the client.

**Why 99.5 and not another number.** Measured on the 30,000 rows, the 99.5th percentile of
the four ratios is 2.0000, 2.0000, 2.0004 and 1.9984 - four independent columns landing on
"paid twice the previous statement" to four decimal places. The cut is therefore not an
arbitrary quantile that happens to be convenient: it coincides with a business threshold,
and above it the magnitude stops being about repayment behaviour. The rows above that cap
have a median previous balance of 1,483 NT$, against a dataset median statement in the
tens of thousands.

It costs 108 to 129 rows per column, 0.4% to 0.5% of the computable rows, and it takes the
worst robust-scaled value of the block from 464.8 to 2.58.

**Why not 99.** Its cap is 1.32 to 1.35, which sits *inside* behaviour that is still
interpretable - paying one and a half times a statement is settling an old balance, not an
artefact - and 89.6% to 92.8% of rows are already at or below 1.0, so the region being
flattened is exactly the informative part of the upper half.

**Why not 99.9.** Its cap is 5.2 to 6.1 and it touches 26 rows. It leaves the 2x-to-6x band
intact, and that band is where the magnitude is driven by how small the denominator was.

A parameter with a default rather than a constant, because unlike the 100 NT$ floor this
one is a modelling knob: it can legitimately be tuned against a metric, and the pipeline
has to let a later turn do that without editing the module.
"""


class PreprocessorInputError(ValueError):
    """The frame handed to the preprocessor is missing columns it is built from."""


def _require_columns(frame: pd.DataFrame, required: Sequence[str], step: str) -> None:
    """Fail loudly and completely when a step cannot find what it reads.

    Every missing column is listed at once rather than the first one found, so a caller
    who renamed a block sees the whole picture in one run.

    Args:
        frame: The candidate input.
        required: Columns the step addresses by name.
        step: Name of the step, used in the message.

    Raises:
        TypeError: If `frame` is not a `pandas.DataFrame`.
        PreprocessorInputError: If a required column is absent.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{step} needs a pandas DataFrame: it addresses columns by name and a "
            f"{type(frame).__name__} has none."
        )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise PreprocessorInputError(
            f"{step} cannot find the columns it reads: {missing}. "
            "Load the data through loader.load_dataset, which applies the canonical names."
        )


class AttachPaymentBehaviourFeatures(BaseEstimator, TransformerMixin):
    """Add the payment-behaviour features to the frame, keeping the source columns.

    `PaymentBehaviourFeatures.transform` deliberately returns **only** its 22 derived
    columns, because inside a `ColumnTransformer` the source columns are routed to their
    own branches and passing them through twice would put them in the matrix twice. That
    contract is right for a branch and wrong for a step: the categorical branch still
    needs `SEX` and the six repayment codes, and they have to survive this step to reach
    it. This class holds the two together.

    **It delegates and does not reimplement.** The arithmetic lives in one place; a second
    copy of it is the exact failure this pipeline exists to prevent. What is added here is
    the concatenation and nothing else.

    Nothing is learned. `fit` records the input columns so the output can be named, and
    stores no statistic of the data.

    Attributes:
        n_features_in_: Number of columns seen during `fit`.
        feature_names_in_: Column names seen during `fit`.
    """

    def __init__(self) -> None:
        """Build the step and the delegate it wraps."""
        self.features = PaymentBehaviourFeatures()

    def fit(self, X: pd.DataFrame, y: object = None) -> "AttachPaymentBehaviourFeatures":  # noqa: N803
        """Record the input columns and check the delegate can run. Learn nothing.

        Args:
            X: Canonical table, as returned by `loader.load_dataset`.
            y: Ignored. Present for API compatibility only.

        Returns:
            The fitted step.

        Raises:
            TypeError: If `X` is not a `pandas.DataFrame`.
            PreprocessorInputError: If a derived name already exists in the input.
        """
        _require_columns(X, (), "AttachPaymentBehaviourFeatures")
        collision = [name for name in FEATURE_NAMES if name in X.columns]
        if collision:
            raise PreprocessorInputError(
                "The input already carries columns named like the derived features: "
                f"{collision}. Attaching them would produce duplicated column names, and a "
                "duplicated name silently makes downstream selection ambiguous."
            )
        self.features.fit(X, y)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return the input columns followed by the 22 derived features.

        Args:
            X: Canonical table carrying the columns `fit` saw.

        Returns:
            A frame of `get_feature_names_out()`, in that order, sharing the index of `X`.

        Raises:
            TypeError: If `X` is not a `pandas.DataFrame`.
            RuntimeError: If the produced columns do not match the declared names.
        """
        _require_columns(X, (), "AttachPaymentBehaviourFeatures")
        attached = pd.concat([X, self.features.transform(X)], axis=1)
        expected = list(self.get_feature_names_out())
        if list(attached.columns) != expected:
            raise RuntimeError(
                "The attached frame does not carry the declared column order. "
                f"Produced: {list(attached.columns)}. Declared: {expected}."
            )
        return attached

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:  # noqa: ARG002
        """Name the columns `transform` produces: the input names, then the derived ones.

        Args:
            input_features: Ignored. The names come from what `fit` saw.

        Returns:
            Input column names followed by `FEATURE_NAMES`.
        """
        return np.asarray([*self.feature_names_in_, *FEATURE_NAMES], dtype=object)


class CollapseEducation(BaseEstimator, TransformerMixin):
    """Fold the undocumented `EDUCATION` codes onto the documented "others" level.

    ADR-0004 decision 4 measured that codes `0`, `5` and `6`, grouped, default at 7.54%
    against 5.69% for the documented level `4` ("others") and 19.23% to 25.16% for levels
    `1` to `3`. They behave like the residual level and not like an education level, so
    collapsing them onto `4` puts them where the measurement puts them.

    **`MARRIAGE` gets no equivalent step, and that absence is the decision.** ADR-0004
    decision 5 measured its undocumented `0` at 9.26% against 26.01% for its own documented
    "others" level - 2.8 times less risky, not more - so collapsing it would introduce a
    bias of known direction. The two columns are treated differently because the
    measurement came out differently.

    The map itself lives in `schema.EDUCATION_COLLAPSE_MAP`, which declares it and applies
    it nowhere: `schema` holds facts, and the application belongs in the pipeline where it
    can be fitted, serialised and versioned. This step is that application.

    Nothing is learned. A code the map does not mention passes through untouched and is
    dealt with by the one-hot policy downstream; failing loudly on an unknown category is
    the validator's job, at the door, and duplicating it here would put the same guarantee
    in two places that can disagree.

    Attributes:
        n_features_in_: Number of columns seen during `fit`.
        feature_names_in_: Column names seen during `fit`.
    """

    COLUMN: Final[str] = "EDUCATION"
    """The one column this step rewrites."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "CollapseEducation":  # noqa: ARG002, N803
        """Check the column is present and record the input names. Learn nothing.

        Args:
            X: Frame carrying `EDUCATION`.
            y: Ignored. Present for API compatibility only.

        Returns:
            The fitted step.

        Raises:
            TypeError: If `X` is not a `pandas.DataFrame`.
            PreprocessorInputError: If `EDUCATION` is absent.
        """
        _require_columns(X, (self.COLUMN,), "CollapseEducation")
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Rewrite `EDUCATION` in a copy, leaving every other column alone.

        Args:
            X: Frame carrying `EDUCATION`.

        Returns:
            A copy with the collapsed codes. The input is not mutated.

        Raises:
            TypeError: If `X` is not a `pandas.DataFrame`.
            PreprocessorInputError: If `EDUCATION` is absent.
        """
        _require_columns(X, (self.COLUMN,), "CollapseEducation")
        collapsed = X.copy()
        collapsed[self.COLUMN] = collapsed[self.COLUMN].replace(dict(schema.EDUCATION_COLLAPSE_MAP))
        return collapsed

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:  # noqa: ARG002
        """Name the columns `transform` produces: the same ones it was given.

        Args:
            input_features: Ignored. The names come from what `fit` saw.

        Returns:
            The input column names, unchanged.
        """
        return np.asarray(self.feature_names_in_, dtype=object)


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Cap the named columns at a percentile **learned in `fit`**, upper side only.

    This is the step that makes the whole pipeline worth being a `Pipeline`. The threshold
    is a statistic of the data, so computing it once over the full table and reusing it
    would move information from the validation fold into the training fold - a leak with
    no symptom, because the resulting matrix looks entirely normal. Learned in `fit`, the
    threshold is a property of the training fold and of nothing else.

    **Why only the upper side.** `schema.NUMERIC_RANGES` declares `PAY_AMT* >= 0`, verified
    on the source with zero negative payments, and ADR-0005 puts a positive floor under the
    denominator. A payment ratio therefore cannot be negative, and its observed minimum is
    exactly 0. A lower bound would clip nothing and would only invite the reader to believe
    something was being defended against.

    **Why `nanpercentile`.** The payment ratios carry `NaN` by design - between 11.98% and
    15.94% of rows per month - and those absences are informative, with their own indicator
    column beside them. A percentile that counted them as values would be a percentile of a
    different quantity. `NaN` passes through `transform` untouched: capping a value the
    module does not know would be an imputation, and this step does not impute.

    Attributes:
        upper_bounds_: Column -> cap learned in `fit`.
        n_features_in_: Number of columns seen during `fit`.
        feature_names_in_: Column names seen during `fit`.
    """

    def __init__(
        self,
        columns: Sequence[str] = PAYMENT_RATIO_FEATURES,
        percentile: float = DEFAULT_CLIP_PERCENTILE,
    ) -> None:
        """Configure which columns are capped and where.

        Args:
            columns: Columns to cap. Every other column passes through untouched.
            percentile: Upper percentile, in [0, 100]. See `DEFAULT_CLIP_PERCENTILE` for
                the measurements behind the default.
        """
        self.columns = columns
        self.percentile = percentile

    def fit(self, X: pd.DataFrame, y: object = None) -> "PercentileClipper":  # noqa: ARG002, N803
        """Learn one cap per configured column, from this frame only.

        A column whose values are all `NaN` in this fold gets a cap of positive infinity
        rather than a `NaN` cap. The difference is not cosmetic: clipping against `NaN`
        turns the whole column into `NaN` silently, which would convert "no evidence in
        this fold" into "no data at all" for every row.

        Args:
            X: Frame carrying the configured columns.
            y: Ignored. Present for API compatibility only.

        Returns:
            The fitted step.

        Raises:
            TypeError: If `X` is not a `pandas.DataFrame`.
            PreprocessorInputError: If a configured column is absent.
            ValueError: If `percentile` is outside [0, 100].
        """
        _require_columns(X, self.columns, "PercentileClipper")
        if not 0.0 <= self.percentile <= 100.0:
            raise ValueError(f"percentile must lie in [0, 100], got {self.percentile}.")

        bounds: dict[str, float] = {}
        for column in self.columns:
            values = X[column].to_numpy(dtype="float64")
            observed = values[~np.isnan(values)]
            bounds[column] = (
                float(np.percentile(observed, self.percentile)) if observed.size else float("inf")
            )
        self.upper_bounds_ = bounds
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Apply the caps learned in `fit`, leaving `NaN` and other columns alone.

        Args:
            X: Frame carrying the configured columns.

        Returns:
            A copy with the configured columns capped. The input is not mutated.

        Raises:
            TypeError: If `X` is not a `pandas.DataFrame`.
            PreprocessorInputError: If a configured column is absent.
        """
        _require_columns(X, self.columns, "PercentileClipper")
        clipped = X.copy()
        for column, bound in self.upper_bounds_.items():
            clipped[column] = clipped[column].clip(upper=bound)
        return clipped

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:  # noqa: ARG002
        """Name the columns `transform` produces: the same ones it was given.

        Args:
            input_features: Ignored. The names come from what `fit` saw.

        Returns:
            The input column names, unchanged.
        """
        return np.asarray(self.feature_names_in_, dtype=object)


def _categorical_branch() -> OneHotEncoder:
    """One-hot encode the categorical block, tolerating a level absent from the fold.

    **`handle_unknown="ignore"` is required, not preferred, and the measurement says so.**
    With `StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)` on this dataset, two
    of the five folds contain a `PAY_STATUS` level that its own training part never saw -
    `PAY_STATUS_5 = 8` and `PAY_STATUS_2 = 8`, one row each, both codes the source
    documents. With `handle_unknown="error"` the project's own cross-validation would crash
    on legitimate data. The same holds for serving: a single request carrying a rare but
    documented code must produce a score, not a 500.

    **This does not weaken the rule that an unknown category fails loudly.** That guarantee
    lives in `data.validator`, at the single door into the data, where `validate_or_raise`
    refuses a code that appears in neither `schema.CATEGORICAL_LEVELS` nor
    `schema.OBSERVED_CODES_ACCEPTED`. Putting the same check here as well would place one
    guarantee in two components that can disagree, and the one that fires second is the one
    that gets deleted. The division is deliberate: the validator decides whether the *data*
    is admissible, the encoder decides what to do with a *fold* that happens not to contain
    every admissible value.

    An ignored level is encoded as all zeros across that column's block, which is
    distinguishable from every known level - each of those sets exactly one indicator.

    **Why no reference level is dropped.** `drop="first"` removes the collinearity that
    only an unregularised linear model suffers from, and costs something every model pays:
    the dropped level stops having a column, so SHAP has nothing to attribute its
    contribution to and every explanation becomes relative to a baseline the reader cannot
    see. The project's stated deliverable is an explainable score.

    Returns:
        The configured encoder.
    """
    return OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    )


def _numeric_branch() -> Pipeline:
    """Impute the missing values, then put every column on a comparable scale.

    **The imputation: median, learned in `fit`, with no indicator added.** ADR-0005 makes
    every non-computable value an explicit `NaN` with its own indicator column beside it,
    and that pairing was verified exactly on the 30,000 rows: for all twelve columns that
    can be missing, `value.isna()` equals `indicator == 1` row for row, with no exception.
    The absence is therefore already in the matrix as a first-class feature, and imputing
    the value destroys nothing - the model can still learn "this ratio was unknown" from
    the column that says so. `add_indicator=True` would emit a perfectly collinear
    duplicate of a column that already exists.

    The median rather than the mean, for the same reason the scaler is robust: the columns
    that carry `NaN` are the payment ratios, whose mean is a statistic of their tail.

    `keep_empty_features=True` so that a fold in which a column is entirely missing still
    produces that column. Without it the matrix would silently change width between folds,
    and a matrix whose shape depends on the fold cannot be cross-validated.

    **The scaler: `RobustScaler`, chosen by measurement.** Over the 29 numeric columns with
    a non-degenerate interquartile range, measured after clipping:

    | | `StandardScaler` | `RobustScaler` |
    | --- | --- | --- |
    | width of the central 50% band, min | 0.181 | 1.000 |
    | width of the central 50% band, max | 1.963 | 1.000 |
    | **widest / narrowest** | **10.86x** | **1.00x** |
    | worst absolute value reached | 72.8 | 403.7 |

    Scaling exists so that no feature dominates a penalty or a distance merely because of
    the units it is written in. `StandardScaler` fails at that here: eleven of the thirty
    columns have an absolute skew above 5 and twelve have a kurtosis above 50, so their
    standard deviation is a statistic of their outliers, and dividing by it compresses the
    region where the data actually lives into a band eleven times narrower than another
    column's. A regularised model would then penalise `PAY_AMT2` an order of magnitude more
    heavily than `UTILIZATION_MOST_RECENT_M1` for the same amount of real signal.
    `RobustScaler` divides by the interquartile range, which the tail does not move, and
    the central band comes out at exactly 1.0 for all 29.

    **The cost is declared, not hidden.** The extreme values reach further - 403.7 against
    72.8, both in `PAY_AMT2`. The clipping step removes that problem for the payment ratios
    and does not touch the raw amounts, so `PAY_AMT*` keeps a long tail on purpose: whether
    those columns should be clipped or log-transformed is a modelling decision that has not
    been taken and is not taken here.

    **`QuantileTransformer` was considered and discarded.** It would flatten every tail by
    construction, and it does it by replacing each value with its rank, which destroys the
    distances between values. A score built on it cannot be explained in the units a credit
    analyst reads - "utilisation went from 40% to 95%" stops being recoverable from the
    feature - and explainability is a stated deliverable rather than a nicety.

    Returns:
        The two-step branch: impute, then scale.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", RobustScaler()),
        ]
    )


def build_preprocessor(
    *,
    clip_percentile: float = DEFAULT_CLIP_PERCENTILE,
) -> Pipeline:
    """Build the preprocessing `Pipeline`: canonical table in, model-ready matrix out.

    The four steps, in order, and what each one is for:

    1. **`behaviour`** - attaches the 22 payment-behaviour features to the source columns,
       by delegating to `PaymentBehaviourFeatures`. Learns nothing.
    2. **`education`** - collapses the undocumented `EDUCATION` codes onto level 4, per
       ADR-0004 decision 4. Learns nothing. `MARRIAGE` is deliberately untouched.
    3. **`clip`** - caps the four payment ratios at a percentile **learned from the
       training fold**. See `DEFAULT_CLIP_PERCENTILE`.
    4. **`columns`** - the `ColumnTransformer` with three branches: one-hot for the
       categorical block, impute-then-scale for the numeric one, passthrough for the
       already-binary indicators. `remainder="drop"`, which is what removes the target.

    **The three branches partition the input exactly, and a test asserts it.** The 23
    source predictors are split 9 categorical / 14 numeric with none left over, and the 22
    derived features 16 numeric / 6 indicator. `remainder="drop"` is therefore silent about
    nothing: every column either has a declared destination or is the target.

    The output is a `pandas.DataFrame` with named columns, because `set_output` is set on
    the whole pipeline. A positional matrix would make `get_feature_names_out` an
    unverifiable claim rather than the thing the frame is actually built from.

    Args:
        clip_percentile: Upper percentile for the payment-ratio cap, in [0, 100].

    Returns:
        An unfitted `Pipeline`. Fitting it on a frame that still carries the target is
        safe - the target reaches no branch - but `fit` and `transform` must be handed the
        same set of columns, as with any sklearn estimator.
    """
    columns = ColumnTransformer(
        transformers=[
            ("categorical", _categorical_branch(), list(CATEGORICAL_COLUMNS)),
            ("numeric", _numeric_branch(), list(NUMERIC_COLUMNS)),
            ("indicators", "passthrough", list(INDICATOR_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        [
            ("behaviour", AttachPaymentBehaviourFeatures()),
            ("education", CollapseEducation()),
            ("clip", PercentileClipper(percentile=clip_percentile)),
            ("columns", columns),
        ]
    )
    return pipeline.set_output(transform="pandas")


def learned_parameters(pipeline: Pipeline) -> Mapping[str, Mapping[str, float]]:
    """Extract everything the fitted pipeline learned from its training fold.

    Exists so that "this step learned something, and something different on a different
    fold" is a checkable statement rather than a claim in a docstring. The leakage test
    compares two fits through this function.

    Args:
        pipeline: A pipeline returned by `build_preprocessor` and already fitted.

    Returns:
        Step name -> parameter name -> value, for the three steps that learn a statistic.
    """
    clipper: PercentileClipper = pipeline.named_steps["clip"]
    numeric: Pipeline = pipeline.named_steps["columns"].named_transformers_["numeric"]
    imputer: SimpleImputer = numeric.named_steps["impute"]
    scaler: RobustScaler = numeric.named_steps["scale"]
    return {
        "clip": dict(clipper.upper_bounds_),
        "impute": dict(zip(NUMERIC_COLUMNS, imputer.statistics_, strict=True)),
        "scale_center": dict(zip(NUMERIC_COLUMNS, scaler.center_, strict=True)),
        "scale_iqr": dict(zip(NUMERIC_COLUMNS, scaler.scale_, strict=True)),
    }
