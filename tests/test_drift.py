"""Tests of the drift instrument, built around one rule: a detector that cannot fail is useless.

Section 6.1 of `docs/METHODOLOGY.md` applies to a monitoring instrument as much as to a model
metric. A drift detector that reports "no drift" is only informative if it is known to report
drift when there is some, so every negative control here has a positive control beside it: the
same function, the same call, a distribution that moved.

The second family pins the two ways this statistic is quietly wrong. Binning each sample by
its **own** quantiles puts the same mass in every bin by construction and reports zero for any
shift; and an empty bin makes the PSI term infinite unless it is floored, which turns one
absent category into "infinitely drifted".
"""

import numpy as np
import pandas as pd
import pytest

from credit_copilot.monitoring.drift import (
    MODERATE_PSI,
    SIGNIFICANT_PSI,
    DriftBand,
    compare_frames,
    population_stability_index,
)

SEED = 42


def normal(n: int, loc: float = 0.0, scale: float = 1.0, seed: int = SEED) -> np.ndarray:
    """Draw a reproducible normal sample."""
    return np.random.default_rng(seed).normal(loc, scale, n)


# ---------------------------------------------------------------------------
# The detector fires, and does not fire, when it should
# ---------------------------------------------------------------------------


def test_identical_samples_have_zero_drift() -> None:
    values = normal(5_000)

    psi, _ = population_stability_index(values, values)

    assert psi == pytest.approx(0.0, abs=1e-12)


def test_a_fresh_draw_from_the_same_distribution_is_below_the_noise_threshold() -> None:
    """The negative control. Two samples of one distribution must not look like drift."""
    reference = normal(20_000, seed=1)
    served = normal(5_000, seed=2)

    psi, _ = population_stability_index(reference, served)

    assert psi < MODERATE_PSI


def test_a_shifted_distribution_is_detected() -> None:
    """The positive control. Without this, the test above proves only that the code runs."""
    reference = normal(20_000, seed=1)
    served = normal(5_000, loc=0.8, seed=2)

    psi, _ = population_stability_index(reference, served)

    assert psi >= SIGNIFICANT_PSI


def test_drift_grows_with_the_size_of_the_shift() -> None:
    reference = normal(20_000, seed=1)
    values = [
        population_stability_index(reference, normal(5_000, loc=shift, seed=2))[0]
        for shift in (0.0, 0.25, 0.5, 1.0)
    ]

    assert values == sorted(values), f"PSI must be monotone in the shift; got {values}"


# ---------------------------------------------------------------------------
# The two ways this statistic goes quietly wrong
# ---------------------------------------------------------------------------


def test_the_bin_edges_come_from_the_reference_and_not_from_each_sample() -> None:
    """Re-binning each sample by its own quantiles would report 0.0 for any shift at all."""
    reference = normal(20_000, seed=1)
    served = normal(5_000, loc=2.0, seed=2)

    psi, _ = population_stability_index(reference, served)

    assert psi > 1.0, (
        "a two-sigma shift produced almost no PSI, which is what self-binning looks like"
    )


def test_a_category_absent_from_the_served_sample_is_bounded_and_not_infinite() -> None:
    reference = np.array([1] * 500 + [2] * 500 + [3] * 500, dtype=float)
    served = np.array([1] * 500 + [2] * 500, dtype=float)

    psi, buckets = population_stability_index(reference, served, categories=[1, 2, 3])

    assert np.isfinite(psi)
    assert psi > SIGNIFICANT_PSI, "a third of the mass vanishing is a real shift"
    assert buckets == 3


def test_a_categorical_column_is_compared_over_its_declared_levels() -> None:
    """A code binned by quantiles compares buckets that mean nothing."""
    reference = np.array([1] * 600 + [2] * 300 + [3] * 100, dtype=float)
    served = np.array([1] * 100 + [2] * 300 + [3] * 600, dtype=float)

    psi, buckets = population_stability_index(reference, served, categories=[1, 2, 3])

    assert buckets == 3
    assert psi >= SIGNIFICANT_PSI


def test_an_empty_sample_is_a_missing_measurement_and_not_zero_drift() -> None:
    with pytest.raises(ValueError, match="not an absence of drift"):
        population_stability_index(normal(100), [])


# ---------------------------------------------------------------------------
# The frame-level report
# ---------------------------------------------------------------------------


def test_compare_frames_ranks_the_worst_feature_first_and_names_it() -> None:
    rng = np.random.default_rng(SEED)
    reference = pd.DataFrame({"quiet": rng.normal(0, 1, 20_000), "moved": rng.normal(0, 1, 20_000)})
    served = pd.DataFrame({"quiet": rng.normal(0, 1, 5_000), "moved": rng.normal(1.2, 1, 5_000)})

    report = compare_frames(reference, served)

    assert report.features[0].feature == "moved"
    assert report.features[0].band is DriftBand.SIGNIFICANT
    assert report.features[-1].feature == "quiet"
    assert report.features[-1].band is DriftBand.NONE
    assert report.max_psi == report.features[0].psi
    assert [item.feature for item in report.drifted] == ["moved"]
    assert report.n_reference == 20_000
    assert report.n_served == 5_000


def test_the_rendered_report_names_the_thresholds_it_judged_against() -> None:
    rng = np.random.default_rng(SEED)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 5_000)})
    served = pd.DataFrame({"x": rng.normal(0, 1, 1_000)})

    text = compare_frames(reference, served).report()

    assert "DECLARED conventions" in text
    assert str(MODERATE_PSI) in text
    assert str(SIGNIFICANT_PSI) in text
    assert "NO SIGNAL" in text


def test_frames_with_no_shared_column_are_refused() -> None:
    with pytest.raises(ValueError, match="share no column"):
        compare_frames(pd.DataFrame({"a": [1.0, 2.0]}), pd.DataFrame({"b": [1.0, 2.0]}))


def test_the_project_raw_columns_survive_a_round_trip_against_themselves() -> None:
    """The exact call `run_online_simulation.py` makes, on the real data contract."""
    from credit_copilot.data import schema
    from credit_copilot.models.registry import PREDICTOR_COLUMNS

    rng = np.random.default_rng(SEED)
    frame = pd.DataFrame(
        {column: rng.integers(1, 4, 2_000).astype(float) for column in PREDICTOR_COLUMNS}
    )
    levels = {
        column: sorted(set(schema.CATEGORICAL_LEVELS[column]))
        for column in schema.CATEGORICAL_LEVELS
        if column in PREDICTOR_COLUMNS
    }

    report = compare_frames(
        frame, frame, columns=list(PREDICTOR_COLUMNS), categorical_levels=levels
    )

    assert len(report.features) == len(PREDICTOR_COLUMNS)
    assert report.max_psi == pytest.approx(0.0, abs=1e-12)
    assert not report.drifted
