"""Feature subpackage: the variables the project builds, as pipeline steps.

Nothing here is a loose function. Every transformation is an estimator that a `Pipeline`
can hold, serialise and hand to the notebook, the training script and the API alike, so
that the three consume one object instead of three copies of the same arithmetic.
"""

from credit_copilot.features.builder import (
    FEATURE_NAMES,
    REQUIRED_COLUMNS,
    MissingSourceColumnsError,
    PaymentBehaviourFeatures,
)

__all__ = [
    "FEATURE_NAMES",
    "REQUIRED_COLUMNS",
    "MissingSourceColumnsError",
    "PaymentBehaviourFeatures",
]
