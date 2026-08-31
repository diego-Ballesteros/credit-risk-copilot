"""Distribution drift between the data the model was fitted on and the data it is served.

**Which test, and why this one is the primary.** The reported statistic is the **Population
Stability Index**, with the two-sample **Kolmogorov-Smirnov** test beside it as a secondary
check. The choice is not a preference:

- **PSI is threshold-based, and its thresholds do not move with sample size.** It is the
  standard instrument in credit risk for exactly this comparison, and the bands below are the
  ones the industry reads. A monitoring number whose meaning is stable as traffic grows is
  the one that can sit on a dashboard.
- **KS reports a p-value, and a p-value is a statement about sample size as much as about
  distance.** With 24,000 reference rows against 6,000 served rows, a shift far too small to
  move a single decision comes out "significant"; with fifty served rows, a shift that
  matters does not. So the KS **statistic** is reported as a distance and its p-value is
  reported next to it with that warning attached, never as the trigger.

Reporting both is deliberate. They see different things: PSI is a binned comparison that
answers *"how much mass moved between these buckets"*, and KS is the largest gap between the
two cumulative distributions, which catches a shift that leaves the bucket masses alone. A
feature that moves on one and not on the other is informative, and collapsing them to one
number would hide it.

**What counts as signal.** The `DriftBand` thresholds: below 0.10 is noise, 0.10 to 0.25 is
a moderate shift worth watching, and 0.25 or above is a shift that should be acted on before
the model's numbers are quoted. These are conventional cutoffs and they are **declared, not
derived from this dataset** - the same status as the 5:1 cost ratio, and worth remembering
before a PSI of 0.11 is treated as a measurement of anything.

**What this module deliberately cannot tell you.** PSI over a feature says the input moved.
It says nothing about whether the model's *performance* moved, and the two come apart in both
directions: a feature can drift a long way inside a region where the model is flat, and a
model can degrade badly while every marginal distribution stays put, because what moved was
the relationship between the features and the target. Feature drift is an early warning that
costs nothing to compute because it needs no labels. It is not a substitute for a metric.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import ks_2samp

__all__ = [
    "DEFAULT_BINS",
    "MODERATE_PSI",
    "SIGNIFICANT_PSI",
    "DriftBand",
    "DriftReport",
    "FeatureDrift",
    "compare_frames",
    "population_stability_index",
]

DEFAULT_BINS: Final[int] = 10
"""Quantile bins a numeric feature is compared over.

Ten because the PSI thresholds below are the ones published for decile binning, and a
statistic read against a threshold calibrated for a different bin count is a different
statistic wearing the same name.
"""

MODERATE_PSI: Final[float] = 0.10
"""Below this, the shift is treated as noise. **A declared convention, not a measurement.**"""

SIGNIFICANT_PSI: Final[float] = 0.25
"""At or above this, the shift is treated as actionable. **Declared, not measured.**"""

_EPSILON: Final[float] = 1e-6
"""Floor applied to an empty bin before taking its logarithm.

A bin that is populated in one sample and empty in the other makes the PSI term infinite,
which would report "infinitely drifted" for what is often one absent category in a small
sample. The floor bounds that term instead, and the bound is the honest reading: the evidence
says *this bucket is empty here*, not *this feature has diverged without limit*.
"""


class DriftBand(StrEnum):
    """How much of a shift a PSI value represents."""

    NONE = "none"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"


def _band(psi: float) -> DriftBand:
    """Place a PSI value in its band."""
    if psi >= SIGNIFICANT_PSI:
        return DriftBand.SIGNIFICANT
    if psi >= MODERATE_PSI:
        return DriftBand.MODERATE
    return DriftBand.NONE


@dataclass(frozen=True)
class FeatureDrift:
    """What one feature's distribution did between the reference and the served sample.

    Attributes:
        feature: Column name.
        psi: Population Stability Index. Zero means the binned distributions match.
        band: Where `psi` falls against the declared thresholds.
        ks_statistic: Largest gap between the two empirical cumulative distributions.
        ks_p_value: Its p-value. **Read the module docstring before using it as a trigger**:
            it is a function of sample size as much as of distance.
        is_categorical: Whether the comparison was made over declared categories rather than
            over quantile bins.
        n_bins: How many buckets the comparison used.
        reference_mean: Mean of the feature in the reference sample, for context.
        served_mean: Mean of the feature in the served sample.
    """

    feature: str
    psi: float
    band: DriftBand
    ks_statistic: float
    ks_p_value: float
    is_categorical: bool
    n_bins: int
    reference_mean: float
    served_mean: float


@dataclass(frozen=True)
class DriftReport:
    """Every feature compared, and the summary a person reads first.

    Attributes:
        features: One entry per compared column, ordered by descending PSI so the worst is
            first. A report read from the top is then read in the order that matters.
        n_reference: Rows in the reference sample.
        n_served: Rows in the served sample.
    """

    features: tuple[FeatureDrift, ...]
    n_reference: int
    n_served: int

    @property
    def max_psi(self) -> float:
        """The largest PSI across features.

        Returns:
            The worst feature's PSI, or 0.0 when no feature was compared.
        """
        return max((item.psi for item in self.features), default=0.0)

    @property
    def drifted(self) -> tuple[FeatureDrift, ...]:
        """Features whose shift is at least moderate.

        Returns:
            The subset above `MODERATE_PSI`, worst first.
        """
        return tuple(item for item in self.features if item.band is not DriftBand.NONE)

    def report(self) -> str:
        """Render the comparison as console text.

        Returns:
            A table of every feature with its PSI, band and KS, followed by a verdict that
            names the declared thresholds rather than leaving them implicit.
        """
        lines = [
            "-" * 86,
            f"{'feature':<28}{'PSI':>9}{'band':>14}{'KS':>9}{'KS p':>11}{'ref mean':>15}",
            "-" * 86,
        ]
        lines.extend(
            f"{item.feature:<28}{item.psi:>9.4f}{item.band.value:>14}"
            f"{item.ks_statistic:>9.4f}{item.ks_p_value:>11.2e}{item.reference_mean:>15,.1f}"
            for item in self.features
        )
        drifted = self.drifted
        verdict = (
            f"NO SIGNAL - every feature below PSI {MODERATE_PSI:.2f}"
            if not drifted
            else f"{len(drifted)} feature(s) at or above PSI {MODERATE_PSI:.2f}: "
            + ", ".join(f"{item.feature} ({item.psi:.4f}, {item.band.value})" for item in drifted)
        )
        lines.extend(
            [
                "-" * 86,
                f"reference n={self.n_reference:,}   served n={self.n_served:,}   "
                f"max PSI={self.max_psi:.4f}",
                verdict,
                f"Thresholds are DECLARED conventions: <{MODERATE_PSI} noise, "
                f"{MODERATE_PSI}-{SIGNIFICANT_PSI} moderate, >={SIGNIFICANT_PSI} actionable.",
            ]
        )
        return "\n".join(lines)


def _edges(reference: npt.NDArray[np.float64], bins: int) -> npt.NDArray[np.float64]:
    """Build bin edges from the reference sample's quantiles.

    The edges come from the **reference** and are then applied to both samples. Re-binning
    each sample by its own quantiles would put the same proportion of mass in every bin by
    construction, and the PSI would be zero however far the distribution had moved.

    Args:
        reference: The reference values.
        bins: Requested number of bins.

    Returns:
        Unique, sorted edges with infinite tails, so a served value outside the reference's
        observed range lands in an end bin rather than being dropped.
    """
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        edges = np.array([reference[0] - 0.5, reference[0] + 0.5])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(
    reference: Sequence[float] | npt.NDArray[np.float64],
    served: Sequence[float] | npt.NDArray[np.float64],
    *,
    bins: int = DEFAULT_BINS,
    categories: Sequence[int] | None = None,
) -> tuple[float, int]:
    """Compare two samples of one feature and return their PSI.

    Args:
        reference: Values the model was fitted on.
        served: Values the model is being asked about.
        bins: Quantile bins, when the feature is numeric.
        categories: Declared levels, when the feature is categorical. Given, the comparison
            is made over these instead of over quantiles, because binning a code by
            quantiles compares buckets that have no meaning.

    Returns:
        The PSI and the number of buckets the comparison used.

    Raises:
        ValueError: Either sample is empty. A PSI against nothing is not zero drift, it is
            an absent measurement, and returning 0.0 would report the second as the first.
    """
    reference_values = np.asarray(reference, dtype=np.float64)
    served_values = np.asarray(served, dtype=np.float64)
    if reference_values.size == 0 or served_values.size == 0:
        raise ValueError(
            f"PSI needs both samples: reference has {reference_values.size} values and "
            f"served has {served_values.size}. An empty sample is a missing measurement, "
            "not an absence of drift."
        )

    if categories is not None:
        levels = np.asarray(sorted(set(categories) | set(np.unique(reference_values).tolist())))
        reference_share = np.array([(reference_values == c).mean() for c in levels])
        served_share = np.array([(served_values == c).mean() for c in levels])
        n_buckets = int(levels.size)
    else:
        edges = _edges(reference_values, bins)
        reference_share = np.histogram(reference_values, bins=edges)[0] / reference_values.size
        served_share = np.histogram(served_values, bins=edges)[0] / served_values.size
        n_buckets = int(edges.size - 1)

    safe_reference = np.clip(reference_share, _EPSILON, None)
    safe_served = np.clip(served_share, _EPSILON, None)
    psi = float(np.sum((safe_served - safe_reference) * np.log(safe_served / safe_reference)))
    return psi, n_buckets


def compare_frames(
    reference: pd.DataFrame,
    served: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    categorical_levels: Mapping[str, Sequence[int]] | None = None,
    bins: int = DEFAULT_BINS,
) -> DriftReport:
    """Compare every shared column of two frames, feature by feature.

    Args:
        reference: The distribution the model was fitted on.
        served: The distribution it is being asked about.
        columns: Columns to compare. Defaults to the columns both frames share.
        categorical_levels: Column name to its declared levels, for the columns that are
            codes rather than magnitudes. Passing `schema.CATEGORICAL_LEVELS` here is what
            keeps `EDUCATION` from being compared over quantiles of an arbitrary integer.
        bins: Quantile bins for numeric columns.

    Returns:
        One entry per column, worst PSI first.

    Raises:
        ValueError: The two frames share no column to compare.
    """
    levels = dict(categorical_levels or {})
    shared = [column for column in (columns or reference.columns) if column in served.columns]
    if not shared:
        raise ValueError(
            "The reference and the served sample share no column, so nothing can be "
            "compared. Check that both carry the canonical column names."
        )

    entries: list[FeatureDrift] = []
    for column in shared:
        reference_values = reference[column].to_numpy(dtype=np.float64)
        served_values = served[column].to_numpy(dtype=np.float64)
        declared = levels.get(column)
        psi, n_buckets = population_stability_index(
            reference_values,
            served_values,
            bins=bins,
            categories=list(declared) if declared is not None else None,
        )
        ks = ks_2samp(reference_values, served_values)
        entries.append(
            FeatureDrift(
                feature=str(column),
                psi=psi,
                band=_band(psi),
                ks_statistic=float(ks.statistic),
                ks_p_value=float(ks.pvalue),
                is_categorical=declared is not None,
                n_bins=n_buckets,
                reference_mean=float(reference_values.mean()),
                served_mean=float(served_values.mean()),
            )
        )
    entries.sort(key=lambda item: -item.psi)
    return DriftReport(features=tuple(entries), n_reference=len(reference), n_served=len(served))
