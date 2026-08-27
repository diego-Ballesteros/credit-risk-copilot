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

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from sklearn.calibration import CalibratedClassifierCV
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
    """Build the project's canonical logistic regression: L2, no class reweighting.

    **`class_weight="balanced"` was removed from this default, and the evidence is entry
    005 of `docs/EVALUATION.md`.** Measured on the same folds and the same seed, reweighting
    by inverse class frequency bought **nothing** in ranking - PR-AUC moved by -0.0009,
    inside the fold spread - and cost **+0.0404 in Brier score**, which is twenty times the
    fold-to-fold spread of that metric. The mechanism is direct: reweighting the positive
    class makes the model predict probabilities above the rate the real population has, and
    Brier measures exactly that distance. ADR-0002 keeps Brier because the declared business
    use needs a probability, so a default that damages it is the wrong default.

    **What this changes about the numbers already recorded.** Entries 001 and 003 of
    `docs/EVALUATION.md` - the phase-2 baselines and the main-hypothesis contrast - were
    measured with the previous default and are **not** invalidated: every arm inside each of
    those comparisons carried the same setting, so each comparison stays internally
    consistent. What they are is measured on a configuration this project no longer ships.
    Re-running either script now will produce different absolute numbers, and that is the
    expected consequence of the change rather than a regression.

    Returns:
        An unfitted estimator. Every call returns an equivalent, independent object, so two
        measurements can never share fitted state.
    """
    return LogisticRegression(
        l1_ratio=LOGISTIC_L1_RATIO,
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


# ---------------------------------------------------------------------------
# The production model - one definition, so four scripts cannot disagree
# ---------------------------------------------------------------------------

PRODUCTION_FOREST_PARAMS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 18,
        "max_features": 0.3,
    }
)
"""Hyperparameters of the production forest: the final study of entry 006.

**These are the tuned values, and the honest caveat is that tuning bought almost nothing.**
Entry 006 measured the gain at +0.0028 in PR-AUC against a fold spread of 0.0080 - inside
the noise, and well below the 0.02 threshold. Choosing the tuned configuration over the
untuned default is therefore choosing between two options the data cannot separate, and the
untuned `build_random_forest(class_weight=None)` would be an equally defensible production
model.

The reason the tuned set is used anyway is the one dimension where the data *did* speak:
`max_features=0.3` was chosen by all five outer folds and by the final study, while the
untuned default uses `"sqrt"` - about 0.1 of 110 columns. That parameter is constrained by
the evidence; the other three are not, and entry 006 records that the five folds disagreed
about every one of them.

**No `class_weight`.** Entry 005 measured that reweighting costs 0.0404 of Brier and buys
no ranking, and this model exists to be calibrated.
"""

PRODUCTION_CALIBRATION_CV: Final[int] = 3
"""Inner folds `CalibratedClassifierCV` uses to fit the calibration map.

Three rather than five keeps the cost of the calibrated model at three forest fits instead
of five, and the calibrator is a one- or two-parameter map that does not need a large
sample to place. The split is over the *training* data only, wherever the calibrated model
is fitted, which is what keeps the calibration curve honest.
"""


def build_production_forest() -> RandomForestClassifier:
    """Build the uncalibrated ranking model: the forest the project decided on.

    This is the object SHAP explains. Calibration is applied on top of it and is a monotone
    transform of the score, so it moves the probabilities without moving the order - which
    is why an explanation of this model is also an explanation of the calibrated one.

    Returns:
        An unfitted estimator.
    """
    return RandomForestClassifier(
        class_weight=None,
        n_jobs=-1,
        random_state=settings.random_state,
        **PRODUCTION_FOREST_PARAMS,
    )


def build_calibrated_forest(
    method: str, cv: int = PRODUCTION_CALIBRATION_CV
) -> CalibratedClassifierCV:
    """Wrap the production forest in a calibration map fitted by internal cross-validation.

    **`ensemble=False` is a deliberate choice with two consequences that both matter here.**
    With the default `ensemble=True`, scikit-learn keeps one (model, calibrator) pair per
    internal fold and averages their predictions, so the deployed object holds three forests
    and the thing SHAP would have to explain is an average of three different models. With
    `ensemble=False` the calibrator is fitted on out-of-fold predictions and then **one**
    forest is refitted on all the training data, so the deployed object holds a single
    ranking function - explainable, and cheaper to serve.

    Wherever this estimator is handed to `cross_validate_estimator`, its internal splitting
    happens strictly inside whatever data it was given, which in a fold is that fold's
    training part. The calibration map therefore never sees the rows it is scored on.

    Args:
        method: `"sigmoid"` for Platt scaling, a two-parameter logistic map, or
            `"isotonic"` for a free monotone step function.
        cv: Internal folds used to produce the out-of-fold predictions the map is fitted on.

    Returns:
        An unfitted estimator.
    """
    return CalibratedClassifierCV(
        estimator=build_production_forest(),
        method=method,
        cv=cv,
        ensemble=False,
    )


PRODUCTION_CALIBRATION_METHOD: Final[str] = "sigmoid"
"""Calibration map the production model carries.

**The measurement says calibration was not needed, and this constant records that tension
rather than hiding it.** Entry 007 of `docs/EVALUATION.md` compared both methods against
the raw forest on out-of-fold probabilities: the **uncalibrated** forest scored the best
Brier of the three (0.133228, against 0.134009 for sigmoid and 0.133551 for isotonic) and
had the smallest gap in every decile of predicted probability, worst bin 0.0106 against
0.0481 and 0.0157. A forest of 300 trees averaging leaf frequencies is already producing a
probability, not a vote share, and there was little left to correct.

Sigmoid is kept because the cost of keeping it is measured and negligible - **+0.0008 in
Brier, well inside the fold spread of 0.002, and exactly 0.0000 in PR-AUC** - and because a
two-parameter map is cheap insurance if the score distribution ever shifts. Dropping the
calibration step is a defensible alternative and the evidence for it is in entry 007.

**Isotonic is not used, and the reason is a real finding rather than a preference.** It cost
0.0133 of PR-AUC. A monotone map cannot reorder anything, but isotonic regression is only
*non-decreasing*: it collapses ranges of score into a single constant, which creates ties,
and ties are what precision-at-top-k and average precision are computed from. Sigmoid is
strictly increasing and left every ranking metric bit-identical.
"""


PRODUCTION_OPERATING_THRESHOLD: Final[float] = 0.160
"""Probability at or above which the production model recommends refusing a client.

**Chosen from a cost matrix, not from the model.** Entry 008 of `docs/EVALUATION.md` swept
201 thresholds against out-of-fold probabilities and minimised `5*FN + FP`, under the
project's stated assumption that a false negative - lending to someone who defaults - costs
five times a false positive. At this cut, on the 30,000 clients of the dataset: 11,635
refusals, 4,769 defaulters caught of 6,636, and 6,866 paying clients turned away.

**It is an assumption-heavy number and the sensitivity is the headline.** At 3:1 the cut is
0.220 and refuses 7,768 clients; at 10:1 it is 0.105 and refuses 22,329. Moving the ratio
across that range moves 48.5% of the book. Anything downstream that treats this constant as
a property of the model rather than of the cost assumption is reading it wrong.
"""
