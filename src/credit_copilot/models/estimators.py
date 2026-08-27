"""The estimator configurations the project measures with, defined once each.

**Why this module exists.** Before it, the logistic regression was written out three times
- in the baselines script, in the leakage check and in the tests - and the hypothesis
contrast needed a fourth copy. Four copies of "the same" configuration is a promise that
somebody keeps them equal, and the methodology's own hierarchy calls that the weakest kind
of guarantee there is. The contrast in particular is only a valid experiment if its three
arms differ in *nothing* but the columns they see; with a single constructor that is a
property of the code, not a thing to check by reading.

**Why `build_logistic_regression` takes no arguments.** Any parameter would be a way for
two call sites to drift apart, which is exactly what the module exists to prevent. When a
later turn needs a second configuration - the roadmap names an L1 variant for the model
comparison - it gets its own named constructor here, so that the difference between two
measured models is visible as two function names rather than hidden in an argument.

**Why the tree constructors do take arguments, and only these.** The imbalance comparison
is an experiment *about* `class_weight`, and the tuning study is an experiment about the
structural hyperparameters; a constructor that refused to vary them would force those
scripts to build their own estimators and reintroduce exactly the duplication this module
removes. The arguments are therefore the ones a measurement legitimately varies, each with
the default that the untuned comparison used, so calling the constructor with no arguments
still reproduces the recorded baseline configuration exactly.

**Why there is no LightGBM here.** It was the intended third model and it does not import
on this machine: the wheel ships `lib_lightgbm.dll`, but loading it needs the Microsoft
Visual C++ runtime, and this system carries only the .NET-bundled `*_clr0400` variants.
`HistGradientBoostingClassifier` stands in - the same family of histogram-based gradient
boosting, from a library already installed, with no native dependency to resolve. The
substitution is recorded in the report of the turn that made it, not decided here.
"""

from typing import Final

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from credit_copilot.config import settings

LOGISTIC_MAX_ITER: Final[int] = 2000
"""Iteration budget for lbfgs.

High enough that the solver converges rather than being stopped mid-descent. A model that
hit its iteration cap reports the metric of an unfinished fit as if it were the metric of
the model, and nothing in the output distinguishes the two.
"""

LOGISTIC_L1_RATIO: Final[float] = 0.0
"""Pure L2 regularisation, in the parameterisation scikit-learn is moving to.

`l1_ratio=0` is L2, `l1_ratio=1` is L1, and anything between is elastic net. The project
used `penalty="l2"` until this was migrated; scikit-learn deprecated that spelling in 1.8
and removes it in 1.10. The two were verified to be **bit-for-bit identical** on this
dataset before the change - all five folds, all seven metrics, to twelve decimal places -
so the migration moved the API and not the model.
"""


def build_logistic_regression() -> LogisticRegression:
    """Build the project's canonical logistic regression: L2, balanced class weights.

    `class_weight="balanced"` reweights each class by the inverse of its frequency, so the
    22% minority stops being something the loss can afford to ignore. It changes what the
    model optimises, not what it is measured against: the metrics of ADR-0002 are computed
    on the untouched validation fold either way.

    Returns:
        An unfitted estimator. Every call returns an equivalent, independent object, so two
        measurements can never share fitted state.
    """
    return LogisticRegression(
        l1_ratio=LOGISTIC_L1_RATIO,
        class_weight="balanced",
        max_iter=LOGISTIC_MAX_ITER,
        random_state=settings.random_state,
    )


# ---------------------------------------------------------------------------
# Tree ensembles - untuned defaults for the model comparison
# ---------------------------------------------------------------------------

RANDOM_FOREST_N_ESTIMATORS: Final[int] = 300
"""Trees in the forest. Enough that the averaging has converged, so the comparison is not
measuring how noisy a small forest is; a forest's error falls monotonically with this
number and then flattens, so it is not a parameter that can flatter the model."""

RANDOM_FOREST_MIN_SAMPLES_LEAF: Final[int] = 20
"""Minimum rows in a leaf. The one deliberate piece of regularisation in the untuned
forest: with the default of 1 a forest grows leaves holding a single row, memorises the
training fold and reports a validation metric dominated by variance. 20 rows on 24,000 is
mild - about 0.08% of the fold - and it is the knob the tuning study is free to move."""

HIST_GB_MAX_ITER: Final[int] = 300
"""Boosting rounds, matched to the forest's tree count so the two ensembles are given a
comparable budget rather than one being handicapped."""

HIST_GB_LEARNING_RATE: Final[float] = 0.05
"""Shrinkage per round. Low enough that 300 rounds is a real fit rather than a few large
steps, which is the pairing that makes the round count meaningful."""

HIST_GB_MAX_LEAF_NODES: Final[int] = 31
"""Leaves per tree. The long-standing default of this family of boosters."""


def build_random_forest(class_weight: str | None = "balanced") -> RandomForestClassifier:
    """Build the project's untuned random forest.

    Args:
        class_weight: Reweighting strategy. `"balanced"` is the configuration the model
            comparison measured; the imbalance comparison varies it, which is what this
            argument exists for.

    Returns:
        An unfitted estimator.
    """
    return RandomForestClassifier(
        n_estimators=RANDOM_FOREST_N_ESTIMATORS,
        min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=settings.random_state,
    )


def build_hist_gradient_boosting(
    class_weight: str | None = "balanced",
) -> HistGradientBoostingClassifier:
    """Build the project's untuned histogram gradient boosting.

    **`early_stopping` is switched off deliberately.** Its default of `"auto"` turns it on
    above 10,000 rows, which would carve an internal validation split out of each training
    fold and stop at a different round in every fold. That is a legitimate way to fit a
    model and a bad way to run a comparison: the number of rounds would stop being a fixed
    quantity and two arms would differ by something other than the thing under test. Fixing
    it here makes the round count exactly `HIST_GB_MAX_ITER` in every fold.

    Args:
        class_weight: Reweighting strategy, varied by the imbalance comparison.

    Returns:
        An unfitted estimator.
    """
    return HistGradientBoostingClassifier(
        max_iter=HIST_GB_MAX_ITER,
        learning_rate=HIST_GB_LEARNING_RATE,
        max_leaf_nodes=HIST_GB_MAX_LEAF_NODES,
        class_weight=class_weight,
        early_stopping=False,
        random_state=settings.random_state,
    )
