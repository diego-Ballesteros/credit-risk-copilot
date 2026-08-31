"""The column groups the main hypothesis contrasts, and the step that selects one.

**What the contrast needs to be a valid experiment.** The project's main hypothesis is that
recent payment behaviour predicts default better than static demographic attributes. Testing
it means running the *same* estimator, with the *same* hyperparameters, inside the *same*
cross-validation, changing nothing but the set of columns the model is allowed to see. If
anything else moved, the difference in the metric would have two possible causes and the
experiment would answer neither.

**Why the selection happens after the preprocessor and not before it.** The obvious approach
- drop the unwanted columns from the raw table - does not work and would quietly corrupt the
comparison. The preprocessor computes the behaviour features *from* the raw blocks, and its
`ColumnTransformer` addresses every source column by name; a table missing `EDUCATION` fails
loudly, and a table missing `BILL_AMT3` would change what the surviving features are worth.
Selecting from the preprocessor's *output* keeps one identical preprocessor in all three
arms, fitted once per fold as always.

That this preserves the values is a property of the preprocessor, not an assumption: every
statistic it learns - the clipping bound, the imputation median, the scaler's centre and
IQR, the one-hot vocabulary - is learned **per column**. A column's transformed value does
not depend on which other columns are present, so `LIMIT_BAL` carries the same number in the
demographic arm and in the full one.

**One honest caveat about the split.** The two groups partition the columns, but they do not
partition the *information*. `UTILIZATION_*` is a bill statement divided by `LIMIT_BAL`, and
`LIMIT_BAL` is counted as demographic while the utilisation features are counted as
behavioural. The behavioural arm therefore carries the credit limit inside a ratio, and
since it also holds the bill amounts, the limit is in principle recoverable from it. A
linear model cannot perform that division, so the leakage of information between arms is
limited in practice - but the contrast measures "these columns against those columns", not
"demography against behaviour" as pure and separable concepts, and the reading has to say so.
"""

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_copilot.data import schema
from credit_copilot.features.builder import FEATURE_NAMES

DEMOGRAPHIC_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "LIMIT_BAL",
)
"""What the client *is*: five attributes that do not move with repayment conduct.

`LIMIT_BAL` sits here rather than with behaviour because the granted limit is an attribute
of the account as underwritten, decided before any of the six months of conduct in this
panel. See the caveat in the module docstring about the utilisation ratios built on it.
"""

BEHAVIOURAL_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    *schema.PAY_STATUS_COLUMNS,
    *schema.BILL_AMOUNT_COLUMNS,
    *schema.PAY_AMOUNT_COLUMNS,
    *FEATURE_NAMES,
)
"""What the client *did*: the six months of repayment status, bills and payments, plus the
22 features ADR-0005 derives from them."""

ALL_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    *DEMOGRAPHIC_SOURCE_COLUMNS,
    *BEHAVIOURAL_SOURCE_COLUMNS,
)
"""Every source the matrix can carry. The two groups are disjoint and cover it exactly."""


class FeatureGroupError(ValueError):
    """A matrix column cannot be traced back to exactly one source column."""


def _resolve_owner(column: str, sources: Sequence[str]) -> str:
    """Trace one matrix column back to the source column it came from.

    The preprocessor names its output by one of two rules: a numeric or indicator column
    keeps its source name unchanged, and a categorical column becomes `<source>_<level>`.
    Resolution follows that, preferring an exact match so that `PAYMENT_RATIO_M2` is never
    mistaken for a level of some shorter name.

    The ambiguous and the unowned cases both raise instead of guessing. A column nobody
    claims means the two groups have stopped covering the matrix - a new feature was added
    to the pipeline and to neither group - and the contrast would then silently compare
    fewer columns than it says it does.

    Args:
        column: A column name produced by the preprocessor.
        sources: Every source column name to resolve against.

    Returns:
        The source column that produced `column`.

    Raises:
        FeatureGroupError: If no source claims the column, or more than one does.
    """
    if column in sources:
        return column

    claimants = [source for source in sources if column.startswith(f"{source}_")]
    if len(claimants) == 1:
        return claimants[0]
    if not claimants:
        raise FeatureGroupError(
            f"The matrix column {column!r} traces back to no declared source column. "
            "Either a feature was added to the pipeline and to neither group in "
            "feature_groups, or the preprocessor changed how it names its output. Both "
            "would make the hypothesis contrast compare fewer columns than it reports."
        )
    raise FeatureGroupError(
        f"The matrix column {column!r} could have come from any of {claimants}. "
        "The naming convention has become ambiguous and the group split is no longer "
        "well defined."
    )


def group_columns_by_source(columns: Sequence[str]) -> Mapping[str, list[str]]:
    """Map each source column to the matrix columns it produced.

    Args:
        columns: The preprocessor's output column names.

    Returns:
        Source column -> its matrix columns, in the order they appear.

    Raises:
        FeatureGroupError: If any column cannot be traced to exactly one source.
    """
    grouped: dict[str, list[str]] = {}
    for column in columns:
        grouped.setdefault(_resolve_owner(column, ALL_SOURCE_COLUMNS), []).append(column)
    return grouped


class SelectFeatureGroup(BaseEstimator, TransformerMixin):
    """Keep only the matrix columns that came from a declared group of source columns.

    Learns nothing from the data. What it resolves in `fit` is which output columns exist,
    which is a property of the fold's one-hot vocabulary rather than a statistic: with
    `handle_unknown="ignore"`, a fold whose training part never saw `PAY_STATUS_5 = 8`
    produces no column for it, and a hard-coded column list would break on that fold.

    Attributes:
        source_columns: The source columns whose output is kept.
    """

    def __init__(self, source_columns: Sequence[str]) -> None:
        """Declare which source columns' output this step keeps.

        Args:
            source_columns: The source columns whose matrix columns survive.
        """
        self.source_columns = source_columns

    def fit(self, X: pd.DataFrame, y: object = None) -> "SelectFeatureGroup":  # noqa: ARG002, N803
        """Resolve which of the matrix's columns belong to the declared group.

        Every column of the incoming matrix is traced to its source, not only the ones being
        kept. Tracing all of them is what turns "the two groups cover the matrix" into a
        checked invariant that fires on the fold where it stops being true.

        Args:
            X: The preprocessor's output.
            y: Ignored; present for the scikit-learn API.

        Returns:
            The fitted selector.

        Raises:
            TypeError: If `X` is not a `pandas.DataFrame`.
            FeatureGroupError: If a column traces to no source or to more than one, or if
                the declared group matched nothing at all.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "SelectFeatureGroup addresses columns by name and was handed a "
                f"{type(X).__name__}. The preprocessor must keep set_output('pandas')."
            )

        wanted = set(self.source_columns)
        owners = [_resolve_owner(column, ALL_SOURCE_COLUMNS) for column in X.columns]
        self.selected_ = [
            column for column, owner in zip(X.columns, owners, strict=True) if owner in wanted
        ]

        if not self.selected_:
            raise FeatureGroupError(
                f"None of the {len(X.columns)} matrix columns came from the declared group "
                f"{sorted(wanted)}. A model with no features cannot be measured."
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return the matrix restricted to the selected columns.

        Args:
            X: The preprocessor's output.

        Returns:
            The same rows with only the group's columns, in their original order.

        Raises:
            FeatureGroupError: If a column resolved in `fit` is absent here.
        """
        missing = [column for column in self.selected_ if column not in X.columns]
        if missing:
            raise FeatureGroupError(
                f"Columns selected during fit are absent from this matrix: {missing}."
            )
        return X[self.selected_]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:  # noqa: ARG002
        """Names of the columns this step emits.

        Args:
            input_features: Ignored; present for the scikit-learn API.

        Returns:
            The selected column names.
        """
        return np.asarray(self.selected_, dtype=object)
