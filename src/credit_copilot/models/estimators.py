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
"""

from typing import Final

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
