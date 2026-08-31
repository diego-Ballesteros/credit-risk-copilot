"""Loading the production artefact from the MLflow Model Registry, and scoring with it.

**Why the copilot loads a registry version instead of building a pipeline.** Every other
entry point of this project can call `build_production_pipeline()` and fit it; the copilot
must not. A refitted pipeline is a *different object* from the one phase 2 measured, even
when the code that builds it is identical - the forest is stochastic and the calibration map
is refitted, so the probability a client is told about would not be the probability any
recorded metric describes. `models:/credit-risk-default-probability/1` is a pinned,
immutable artefact, and pinning it is what makes the number the agent reports the same
number `docs/MODEL_CARD.md` documents.

**Why the version is pinned rather than resolved to "latest".** An alias that follows the
newest version would silently change what the agent says the day somebody registers version
2, and no test would fail. The version is a constant here and a parameter of the loader, so
moving to a new one is an edit somebody makes on purpose.

**Why missing columns raise instead of being filled.** The pipeline would raise a `KeyError`
from inside a `ColumnTransformer`, which reaches the caller as a message about an internal
step. Worse, a caller who "fixed" it by inserting zeros would turn *"we do not know this
applicant's September bill"* into *"this applicant's September bill was zero"*, which is a
business fact and a false one - section 7.1 of `docs/METHODOLOGY.md`, and section 2.3 of the
internal credit policy, which sends exactly that case to full manual evaluation. The check
happens here, up front, and names the columns.

**Why the uncalibrated forest is reachable from outside.** `explain/shap_service.py` needs a
tree ensemble, and `CalibratedClassifierCV` is not one. Reaching into
`calibrated_classifiers_` is intimate with scikit-learn's internals, so it is done in one
place, verified rather than assumed, and it fails loudly if the shape ever changes - the
alternative is each consumer reaching in, and one of them guessing wrong in silence.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from credit_copilot.data import schema
from credit_copilot.models.evaluation import MODEL_STEP, PREPROCESSOR_STEP
from credit_copilot.models.tracking import MLflowConfigurationError, configure_mlflow

__all__ = [
    "POSITIVE_CLASS_INDEX",
    "PREDICTOR_COLUMNS",
    "PRODUCTION_MODEL_NAME",
    "PRODUCTION_MODEL_VERSION",
    "MissingColumnsError",
    "ModelUnavailableError",
    "RegisteredModel",
    "UnknownValueError",
    "load_registered_model",
    "require_known_values",
    "require_predictor_columns",
]

PRODUCTION_MODEL_NAME: Final[str] = "credit-risk-default-probability"
"""Name the production pipeline carries in the MLflow Model Registry."""

PRODUCTION_MODEL_VERSION: Final[str] = "1"
"""Registry version the copilot is pinned to. Never `latest`; see the module docstring."""

POSITIVE_CLASS_INDEX: Final[int] = 1
"""Column of `predict_proba` that holds the probability of default."""

PREDICTOR_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in schema.WORKING_COLUMNS if column != schema.TARGET_COLUMN
)
"""The 23 raw columns the registered signature declares, in source order.

Derived from `schema.WORKING_COLUMNS` rather than written out, so a change to the data
contract reaches the copilot instead of leaving two lists to be kept equal by hand.
"""


class ModelUnavailableError(RuntimeError):
    """The registered model could not be loaded from the registry."""


class MissingColumnsError(ValueError):
    """The frame to score does not carry every raw column the model needs.

    Raised instead of imputing. An absent column is an absence of knowledge, and filling it
    would convert that absence into a value the model reads as a fact.
    """


def require_predictor_columns(columns: Sequence[str] | pd.Index) -> None:
    """Refuse a frame that is missing any raw column the model was fitted on.

    Args:
        columns: The column names available.

    Raises:
        MissingColumnsError: One or more of `PREDICTOR_COLUMNS` is absent. The message names
            every missing column. Nothing is ever filled in: an absent column is not a zero.
    """
    available = set(columns)
    missing = [column for column in PREDICTOR_COLUMNS if column not in available]
    if missing:
        raise MissingColumnsError(
            f"The applicant is missing {len(missing)} of the {len(PREDICTOR_COLUMNS)} raw "
            f"columns the model needs: {', '.join(missing)}. Nothing is imputed - a missing "
            "value means 'unknown', and writing a zero into it would turn that into a "
            "business fact that is false. Section 2.3 of the internal credit policy sends "
            "this case to full manual evaluation instead."
        )


class UnknownValueError(ValueError):
    """An applicant attribute carries a value the data contract does not recognise.

    Raised rather than tolerated, per the hard rule of `CLAUDE.md`: an unknown category
    fails loudly. A forest asked about a code it never saw does not fail - it routes the row
    down whichever branch the comparison happens to take and returns a plausible number.
    """


def require_known_values(values: Mapping[str, int]) -> None:
    """Refuse applicant attribute values the data contract does not recognise.

    Two contracts are consulted, in the order `schema.py` keeps them apart. A categorical
    value is accepted when the source documents it (`schema.CATEGORICAL_LEVELS`) **or** when
    ADR-0004 accepted it on measured evidence (`schema.OBSERVED_CODES_ACCEPTED`); anything
    else is a code nobody has looked at. A numeric value is accepted inside the plausible
    interval `schema.NUMERIC_RANGES` declares, whose bounds mark where a value stops being a
    business fact and becomes a data error.

    Args:
        values: Column name to value. Columns outside the predictor set are ignored, so the
            same check works on a whole applicant and on a partial scenario.

    Raises:
        UnknownValueError: A value is outside both contracts. The message names the column,
            the value, and what was admissible.
    """
    for column, value in values.items():
        if column not in PREDICTOR_COLUMNS:
            continue
        levels = schema.CATEGORICAL_LEVELS.get(column)
        if levels is not None:
            accepted = set(levels) | set(schema.OBSERVED_CODES_ACCEPTED.get(column, {}))
            if value not in accepted:
                raise UnknownValueError(
                    f"{column}={value} is not a level this project recognises. Documented by "
                    f"the source: {sorted(levels)}; accepted on measured evidence by "
                    f"{schema.ADR_UNDOCUMENTED_CODES}: "
                    f"{sorted(schema.OBSERVED_CODES_ACCEPTED.get(column, {}))}. An "
                    "unrecognised code is not scored, because the model would route it down "
                    "some branch anyway and return a number that looks like the others."
                )
            continue
        span = schema.NUMERIC_RANGES.get(column)
        if span is None:
            continue
        if (span.minimum is not None and value < span.minimum) or (
            span.maximum is not None and value > span.maximum
        ):
            raise UnknownValueError(
                f"{column}={value} falls outside the plausible range "
                f"[{span.minimum}, {span.maximum}]. {span.rationale}"
            )


@dataclass(frozen=True)
class RegisteredModel:
    """One pinned version of the production pipeline, ready to score raw applicant rows.

    Attributes:
        name: Registered name in the model registry.
        version: Registry version this instance loaded.
        pipeline: The whole artefact: preprocessor plus calibrated forest.
    """

    name: str
    version: str
    pipeline: Pipeline

    @property
    def uri(self) -> str:
        """Registry URI this artefact was loaded from.

        Returns:
            A `models:/name/version` URI, quotable in a report or in a credit file.
        """
        return f"models:/{self.name}/{self.version}"

    @property
    def preprocessor(self) -> Pipeline:
        """The fitted preprocessing half of the artefact.

        Returns:
            The `preprocess` step of the pipeline.

        Raises:
            ModelUnavailableError: The artefact does not carry that step.
        """
        return self._step(PREPROCESSOR_STEP)

    @property
    def calibrated_classifier(self) -> CalibratedClassifierCV:
        """The calibrated classifier the artefact serves probabilities from.

        Returns:
            The `model` step of the pipeline.

        Raises:
            ModelUnavailableError: The artefact does not carry that step.
        """
        step = self._step(MODEL_STEP)
        if not isinstance(step, CalibratedClassifierCV):
            raise ModelUnavailableError(
                f"{self.uri} carries a {type(step).__name__} as its model step rather than "
                "a CalibratedClassifierCV."
            )
        return step

    def uncalibrated_forest(self) -> RandomForestClassifier:
        """Reach the tree ensemble underneath the calibration map.

        The sigmoid calibration of ADR-0007 is strictly increasing, so it moves the value of
        a probability without moving the order of two clients or the sign of any feature's
        contribution to that order. Explaining the forest is therefore explaining the ranking
        the deployed artefact serves, and it is also the only option that is defined at all:
        `TreeExplainer` cannot see into a calibrated wrapper.

        Returns:
            The `RandomForestClassifier` fitted inside the calibrated wrapper.

        Raises:
            ModelUnavailableError: The wrapper does not have the shape this project
                registered - exactly one internal classifier holding a fitted forest. The
                check is explicit because `calibrated_classifiers_` is a scikit-learn
                internal, and a silent change of its shape would otherwise surface as an
                explanation attributed to the wrong object.
        """
        calibrated = self.calibrated_classifier
        internals = getattr(calibrated, "calibrated_classifiers_", None)
        if not internals:
            raise ModelUnavailableError(
                f"{self.uri} carries a calibrated classifier with no fitted internal "
                "classifier. The artefact cannot be explained."
            )
        if len(internals) != 1:
            raise ModelUnavailableError(
                f"{self.uri} carries {len(internals)} internal calibrated classifiers; this "
                "project registered exactly one, fitted with ensemble=False. Refusing to "
                "pick one of them arbitrarily."
            )
        estimator = getattr(internals[0], "estimator", None)
        if not isinstance(estimator, RandomForestClassifier):
            raise ModelUnavailableError(
                f"{self.uri} wraps a {type(estimator).__name__} rather than a "
                "RandomForestClassifier. SHAP's TreeExplainer would not be valid on it."
            )
        return estimator

    def probability_of_default(self, applicants: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Score raw applicant rows with the registered artefact.

        Args:
            applicants: One row per applicant, carrying the raw canonical columns. Extra
                columns are ignored; missing ones are refused.

        Returns:
            The probability of default of each row, in the order given.

        Raises:
            MissingColumnsError: Any of the 23 raw columns is absent. Nothing is imputed.
        """
        require_predictor_columns(applicants.columns)
        probabilities = self.pipeline.predict_proba(applicants[list(PREDICTOR_COLUMNS)])
        return np.asarray(probabilities, dtype=np.float64)[:, POSITIVE_CLASS_INDEX]

    def _step(self, name: str) -> object:
        """Return a named step of the pipeline, or say which one is missing."""
        step = self.pipeline.named_steps.get(name)
        if step is None:
            raise ModelUnavailableError(
                f"{self.uri} has no `{name}` step; it carries "
                f"{sorted(self.pipeline.named_steps)}. This is not the artefact "
                "`scripts/register_production_model.py` registers."
            )
        return step


def load_registered_model(
    name: str = PRODUCTION_MODEL_NAME,
    version: str = PRODUCTION_MODEL_VERSION,
) -> RegisteredModel:
    """Load one pinned version of the production pipeline from the MLflow registry.

    The artefact is never rebuilt and never refitted: it is downloaded exactly as
    `scripts/register_production_model.py` left it, which is what makes the probability the
    copilot reports the same probability the model card describes.

    Args:
        name: Registered model name.
        version: Registry version. Pinned by default; see the module docstring.

    Returns:
        The loaded artefact, with the registry coordinates it came from.

    Raises:
        ModelUnavailableError: MLflow is not configured, the registry has no such version,
            or the artefact could not be deserialised.
    """
    try:
        configure_mlflow()
    except MLflowConfigurationError as error:
        raise ModelUnavailableError(
            f"The copilot cannot load `{name}` version {version}: {error}"
        ) from error

    uri = f"models:/{name}/{version}"
    try:
        pipeline = mlflow.sklearn.load_model(uri)
    except Exception as error:  # noqa: BLE001 - the registry raises many unrelated types
        raise ModelUnavailableError(
            f"Could not load {uri}: {type(error).__name__}: {error}. Run "
            "`uv run python scripts/register_production_model.py` if that version does not "
            "exist yet."
        ) from error

    if not isinstance(pipeline, Pipeline):
        raise ModelUnavailableError(
            f"{uri} deserialised to a {type(pipeline).__name__} rather than a Pipeline. The "
            "copilot expects the whole preprocessor-plus-model artefact."
        )
    return RegisteredModel(name=name, version=version, pipeline=pipeline)
