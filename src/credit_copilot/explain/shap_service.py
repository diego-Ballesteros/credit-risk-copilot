"""Local SHAP attribution for one applicant, with the direction taken from that applicant.

**The mistake this module exists to prevent.** `scripts/run_shap_analysis.py` reports, for
each feature, a mean absolute SHAP value and the *mean signed* value beside it, and reading
the sign of that mean as "what this feature does" is wrong twice over. It is a population
average, so it says nothing about one applicant; and it is an average over rows that push in
opposite directions, so it can be near zero for a feature that moves every single client a
lot. A feature whose population mean is negative can raise this applicant's score, and a
sentence built on the population sign would be fluent, specific and false.

**How it is resolved here.** The direction reported for a feature is the sign of **this
row's** SHAP value and nothing else. No population statistic is read, computed or carried:
`explain()` takes exactly one row, and there is no code path in this module that averages
over rows. That is deliberate - the guarantee is structural rather than a rule somebody has
to remember, which is the hierarchy of section 6.5 of `docs/METHODOLOGY.md`.

**What is explained, and why it is the right object.** The forest inside the calibrated
wrapper, reached through `models.registry`. The production calibration is a sigmoid, which
is strictly increasing: it moves the value of the probability without moving the order of
two clients or the sign of any contribution to that order. `TreeExplainer` also cannot see
into a calibrated wrapper at all, so this is the only defined option. The consequence has to
be stated rather than hidden: **the SHAP values add up to the forest's score, not to the
calibrated probability the decision is taken on**, and `LocalExplanation` carries both
numbers so the difference is visible instead of inferred.

**Why the feature names are checked before anything is attributed.** A `ColumnTransformer`
that lost its names would label its output positionally and SHAP would attribute this
applicant's refusal to whichever variable happened to sit at that index. The failure is not
a missing explanation - it is a readable, confident, wrong one. The same check that guards
the global analysis guards this one.
"""

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd
import shap

from credit_copilot.models.registry import (
    POSITIVE_CLASS_INDEX,
    PREDICTOR_COLUMNS,
    RegisteredModel,
)

__all__ = [
    "DEFAULT_TOP_FEATURES",
    "Direction",
    "ExplanationError",
    "FeatureEffect",
    "LocalExplanation",
    "ShapLocalExplainer",
]

DEFAULT_TOP_FEATURES: Final[int] = 5
"""Features reported by default.

Five because that is what section 4.2 of the internal credit policy requires a model-assisted
credit file to record: *"las cinco variables de mayor contribución a la estimación, con su
dirección y magnitud"*. The number is a policy requirement, not a display preference.
"""

Direction = Literal["raises_risk", "lowers_risk"]
"""Which way a feature moved **this** applicant's score. Never a population statistic."""


class ExplanationError(RuntimeError):
    """The explanation could not be produced in a form that can be trusted."""


@dataclass(frozen=True)
class FeatureEffect:
    """One feature's contribution to one applicant's score.

    Attributes:
        feature: Name of the transformed feature, as the pipeline declares it.
        shap_value: Signed contribution of this feature for this applicant, in the forest's
            output units. Positive pushes towards default.
        magnitude: Absolute value of `shap_value`, which is what ranks the features.
        direction: Sign of `shap_value` for **this applicant**. See the module docstring.
        feature_value: The value this feature takes for this applicant, after preprocessing.
    """

    feature: str
    shap_value: float
    magnitude: float
    direction: Direction
    feature_value: float


@dataclass(frozen=True)
class LocalExplanation:
    """Everything the explanation of a single applicant consists of.

    Attributes:
        probability_of_default: The calibrated probability the decision is taken on.
        forest_score: The uncalibrated forest score the SHAP values decompose.
        base_value: The forest's expected output; `base_value + sum(all shap values)`
            reconstructs `forest_score`, not `probability_of_default`.
        effects: The requested features, in descending order of magnitude.
        features_considered: How many features the attribution ran over in total.
    """

    probability_of_default: float
    forest_score: float
    base_value: float
    effects: tuple[FeatureEffect, ...]
    features_considered: int


class ShapLocalExplainer:
    """A `TreeExplainer` bound to one registered artefact, reused across applicants.

    Building the explainer walks 300 trees, so it is done once per instance and never per
    request. The instance holds no applicant state, which is what makes reuse safe.
    """

    def __init__(self, model: RegisteredModel) -> None:
        """Bind an explainer to the forest inside a registered pipeline.

        Args:
            model: The pinned artefact whose forest gets explained.

        Raises:
            ModelUnavailableError: The artefact does not carry an explainable forest.
        """
        self._model = model
        self._forest = model.uncalibrated_forest()
        self._explainer = shap.TreeExplainer(self._forest)

    @property
    def model(self) -> RegisteredModel:
        """The artefact this explainer explains.

        Returns:
            The registered model it was built from.
        """
        return self._model

    def explain(
        self, applicant: pd.DataFrame, top_n: int = DEFAULT_TOP_FEATURES
    ) -> LocalExplanation:
        """Attribute one applicant's score to the features that produced it.

        Args:
            applicant: Exactly one row, carrying the raw canonical columns.
            top_n: How many features to report, ranked by absolute contribution.

        Returns:
            The applicant's probability, the forest score the attribution decomposes, and
            the `top_n` strongest contributions with the direction each took **for this
            applicant**.

        Raises:
            ExplanationError: More or fewer than one row was given, `top_n` is not positive,
                the feature names do not line up across the pipeline, the forest and the
                explanation, or SHAP returned a shape this code cannot read without
                guessing which axis is which.
            MissingColumnsError: The row is missing raw columns. Nothing is imputed.
        """
        if len(applicant) != 1:
            raise ExplanationError(
                f"An explanation is about one applicant; {len(applicant)} rows were given. "
                "Averaging them would produce exactly the population statistic this module "
                "exists to keep out of an individual explanation."
            )
        if top_n <= 0:
            raise ExplanationError(f"top_n={top_n} is not positive.")

        probability = float(self._model.probability_of_default(applicant)[0])
        matrix = self._transform(applicant)
        values, base_value = self._attribute(matrix)

        order = np.argsort(-np.abs(values))[:top_n]
        effects = tuple(
            FeatureEffect(
                feature=str(matrix.columns[index]),
                shap_value=float(values[index]),
                magnitude=float(abs(values[index])),
                direction="raises_risk" if values[index] > 0 else "lowers_risk",
                feature_value=float(matrix.iloc[0, index]),
            )
            for index in order
        )
        return LocalExplanation(
            probability_of_default=probability,
            forest_score=float(base_value + values.sum()),
            base_value=float(base_value),
            effects=effects,
            features_considered=int(matrix.shape[1]),
        )

    def _transform(self, applicant: pd.DataFrame) -> pd.DataFrame:
        """Run the fitted preprocessor and refuse to continue if the names do not line up."""
        matrix = self._model.preprocessor.transform(applicant[list(PREDICTOR_COLUMNS)])
        declared = [str(name) for name in self._model.preprocessor.get_feature_names_out()]
        fitted_on = [str(name) for name in getattr(self._forest, "feature_names_in_", [])]
        if list(matrix.columns) != declared or declared != fitted_on:
            raise ExplanationError(
                "The feature names do not line up across the preprocessor's output "
                f"({len(matrix.columns)}), what it declares ({len(declared)}) and what the "
                f"forest was fitted on ({len(fitted_on)}). Refusing to attribute anything: "
                "an explanation pinned to the wrong variable is worse than no explanation, "
                "because it is readable and confident."
            )
        return matrix

    def _attribute(self, matrix: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Run SHAP over the single row and return its values for the positive class."""
        explanation = self._explainer(matrix)
        values = np.asarray(explanation.values)
        base = np.asarray(explanation.base_values)
        if values.ndim == 3:
            return values[0, :, POSITIVE_CLASS_INDEX], float(
                base.reshape(base.shape[0], -1)[0, POSITIVE_CLASS_INDEX]
            )
        if values.ndim == 2:
            return values[0, :], float(np.ravel(base)[0])
        raise ExplanationError(
            f"SHAP returned values of shape {values.shape}, which this code cannot read "
            "without guessing which axis is the class. Refusing to guess."
        )
