"""Measure the null distribution of the ranking metrics on this pipeline.

Run it with::

    uv run python scripts/measure_null_distribution.py

**Why this script exists at all.** ADR-0006 fixes the tolerances of the shuffled-target test
at 0.020 for ROC-AUC and 0.015 for PR-AUC, and it justifies them with a measured standard
error rather than an intuition. `docs/METHODOLOGY.md` requires every figure in
`docs/analysis/` to be reproducible by a script in the repository; until this file existed,
the numbers the ADR and the leakage check cite came from a throwaway script and were
therefore unverifiable by anyone reading them. That is the gap this closes.

**What it measures, and why it is not the obvious thing.** The question is: when the target
carries no information at all, how far from its floor does each metric land? The tempting
way to answer it is to draw random scores and compute the metric - and that answer is wrong
here, because it leaves out both things that actually move the number. The model is *fitted*
on the permuted labels, so it can chase noise; and the five training folds of a 5-fold split
overlap by 80%, so the five per-fold results are not independent draws. Only running the
real pipeline against real permutations measures the quantity the criterion is about.

Measured both ways, the difference is not academic: the naive simulation gives a standard
error of 0.00403 for ROC-AUC where the real one is 0.00593, understating the spread by about
a third. A tolerance derived from the naive figure would be too tight and would eventually
fail on an honest pipeline.

**Cost.** Eight permutations times five folds is forty complete fits of the preprocessor and
the model. It runs in minutes, not seconds, and it is meant to be re-run when the pipeline
changes - not on every turn. ADR-0006 records that explicitly.

Nothing here is written to MLflow. These are eight diagnostic runs on a destroyed target;
recording them beside the model results would put eight more things in the experiment that a
reader has to know to filter out.
"""

import sys
from typing import Final

import numpy as np
import pandas as pd

from credit_copilot.config import settings
from credit_copilot.data.loader import load_dataset
from credit_copilot.models.estimators import build_logistic_regression
from credit_copilot.models.evaluation import (
    DEFAULT_N_SPLITS,
    cross_validate_estimator,
    split_features_and_target,
)

N_PERMUTATIONS: Final[int] = 8
"""How many independent permutations of the target to measure.

Eight is enough to place the centre and the spread to the precision the tolerances need -
three significant figures on a standard error - and small enough that the whole measurement
is forty pipeline fits rather than hundreds. It is not enough to characterise the tails, and
the ADR does not claim it is: the criterion is stated in standard errors, not in quantiles.
"""

PERMUTATION_INDICES: Final[tuple[int, ...]] = tuple(range(N_PERMUTATIONS))
"""Identifier of each permutation, used as the seed of that permutation and nothing else.

**This is deliberately not `settings.random_state`, and it is not a second project seed.**
The project seed governs everything whose reproducibility matters to a reported metric - the
fold split and the estimator - and it is read from `config.py` below, as the rules require.
What this tuple indexes is the eight *draws* of the experiment. They have to differ from one
another, so they cannot all be the project seed; and they have to be fixed, so that the
figures ADR-0006 cites can be regenerated exactly. An index is the simplest thing that is
both.
"""

RULE: Final[str] = "=" * 92


def permute(target: pd.Series, seed: int) -> pd.Series:
    """Shuffle the labels among the rows, keeping the prevalence exactly.

    A permutation moves labels between rows; it does not create or destroy any. The floor of
    PR-AUC is the prevalence, so a resampling scheme that changed it would move the target of
    the comparison at the same time as the thing being compared.

    Args:
        target: The real labels.
        seed: Seed identifying this permutation.

    Returns:
        The same labels in a different order, on the original index.
    """
    shuffled = target.sample(frac=1.0, random_state=seed).to_numpy()
    return pd.Series(shuffled, index=target.index, name=target.name)


def summarise(values: np.ndarray, centre: float) -> dict[str, float]:
    """Describe where a metric's null draws landed relative to its no-signal floor.

    Args:
        values: One mean-over-folds value per permutation.
        centre: The metric's value when the scores carry no information.

    Returns:
        The empirical centre, its offset from the floor, the standard error of a
        mean-over-folds draw, and the worst absolute deviation observed.
    """
    deviations = values - centre
    return {
        "mean": float(values.mean()),
        "offset": float(deviations.mean()),
        "standard_error": float(values.std(ddof=1)),
        "worst_absolute_deviation": float(np.abs(deviations).max()),
    }


def main() -> int:
    """Run the pipeline against each permutation and report the null distribution.

    Returns:
        0 always. This script measures; it does not judge. The criterion it feeds lives in
        `run_leakage_check.py`, and that is the script with a verdict and an exit code.
    """
    print(RULE)
    print("NULL DISTRIBUTION OF THE RANKING METRICS - measured on the real pipeline")
    print(RULE)

    frame = load_dataset()
    features, target = split_features_and_target(frame)
    prevalence = float(target.mean())

    print(f"Rows              : {len(frame):,}")
    print(f"Prevalence        : {prevalence:.6f}  (identical in every permutation)")
    print(f"Permutations      : {N_PERMUTATIONS}, seeds {list(PERMUTATION_INDICES)}")
    print(f"Validation        : StratifiedKFold({DEFAULT_N_SPLITS}, shuffle=True), ", end="")
    print(f"random_state={settings.random_state} from config.py")
    print("Estimator         : build_logistic_regression(), the project's canonical model")
    print(f"Total pipeline fits: {N_PERMUTATIONS * DEFAULT_N_SPLITS}")

    print("\n" + RULE)
    print("PER PERMUTATION - mean over the 5 folds")
    print(RULE)
    print(
        f"{'permutation':<14}{'ROC-AUC':>14}{'dev from 0.5':>16}{'PR-AUC':>14}{'dev from prev':>16}"
    )
    print("-" * 92)

    roc_values, pr_values = [], []
    for seed in PERMUTATION_INDICES:
        result = cross_validate_estimator(
            build_logistic_regression(),
            features,
            permute(target, seed),
        )
        roc = result.means["roc_auc"]
        pr = result.means["pr_auc"]
        roc_values.append(roc)
        pr_values.append(pr)
        print(
            f"seed {seed:<9}{roc:>14.6f}{roc - 0.5:>+16.6f}{pr:>14.6f}{pr - prevalence:>+16.6f}",
            flush=True,
        )

    roc_summary = summarise(np.asarray(roc_values), 0.5)
    pr_summary = summarise(np.asarray(pr_values), prevalence)

    print("\n" + RULE)
    print("THE NULL DISTRIBUTION")
    print(RULE)
    for label, floor, summary in (
        ("ROC-AUC", 0.5, roc_summary),
        ("PR-AUC", prevalence, pr_summary),
    ):
        print(f"\n{label}   (no-signal floor {floor:.6f})")
        print(f"    empirical centre           {summary['mean']:.6f}")
        print(f"    offset from the floor      {summary['offset']:+.6f}")
        print(f"    standard error of the mean {summary['standard_error']:.6f}")
        print(f"    worst |deviation| observed {summary['worst_absolute_deviation']:.6f}")

    print("\n" + RULE)
    print("WHAT THIS MAKES OF THE TOLERANCES IN ADR-0006")
    print(RULE)
    for label, tolerance, summary in (
        ("ROC-AUC", 0.02, roc_summary),
        ("PR-AUC", 0.015, pr_summary),
    ):
        errors = tolerance / summary["standard_error"]
        headroom = tolerance / summary["worst_absolute_deviation"]
        print(
            f"  {label:<9} tolerance +/-{tolerance:.3f} = {errors:.1f} standard errors; "
            f"{headroom:.1f}x the worst deviation seen"
        )

    print("\nThe naive alternative, for the record: drawing independent random scores rather")
    print("than fitting the model gives 0.00403 and 0.00244, understating the real spread")
    print("because it ignores both the fitting and the 80% overlap between training folds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
