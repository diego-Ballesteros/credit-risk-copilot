"""What the model would say about an applicant with different attributes. Not what would happen.

**The sentence this module is allowed to produce, and the one it is not.** Section 4.3 of
the internal credit policy draws the line in the corpus, and it is repeated here because the
wrong sentence reads perfectly well:

> *"si este cliente presentara una utilización de cupo del 30% en lugar del 85%, el modelo
> estimaría una probabilidad de 0,12 en lugar de 0,31"* - allowed, and verifiable by rerunning
> this function.
>
> *"reducir la utilización hará que el cliente deje de incumplir"* - **not** allowed. That is
> a causal claim, and it is not supported by observational data that never intervened on
> anything.

The distinction is not pedantry about wording. The model learned that clients who look a
certain way default at a certain rate; it never observed what happens when a client is
*changed* into looking that way. `docs/MODEL_CARD.md` lists inferring causality among the
uses the model does not support, and this module is where the temptation is strongest,
because a delta looks exactly like an effect.

**Why the original applicant is never mutated.** A scenario that edited its input in place
would leave the caller holding a row that silently stopped describing the applicant, and the
baseline probability recomputed afterwards would be the scenario's. The copy is not a
defensive habit here; it is what makes the baseline and the scenario two different things.

**Why a scenario's values are checked against the data contract.** A change is a value
somebody invented, which is exactly where an unrecognised code enters. `PAY_STATUS_1 = 15`
does not fail on its own - the forest routes the row down some branch and returns a number
shaped like every other number. The check is `models.registry.require_known_values`, the same
one the applicant record itself passes through, so a scenario cannot reach the model through
a door the applicant could not.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from credit_copilot.models.registry import (
    PREDICTOR_COLUMNS,
    require_known_values,
    require_predictor_columns,
)

__all__ = [
    "ProbabilityScorer",
    "ScenarioError",
    "ScenarioOutcome",
    "apply_scenario",
    "evaluate_scenario",
]


class ProbabilityScorer(Protocol):
    """Anything that turns raw applicant rows into probabilities of default.

    Declared structurally rather than as `RegisteredModel` so that a test can drive a
    scenario with a scorer built by hand, and check the no-mutation guarantee without a
    registry, a network and a 300-tree forest.
    """

    def probability_of_default(
        self, applicants: pd.DataFrame
    ) -> npt.NDArray[np.float64]:  # pragma: no cover - structural declaration
        """Score raw applicant rows."""


class ScenarioError(ValueError):
    """The requested scenario cannot be applied to this applicant."""


@dataclass(frozen=True)
class ScenarioOutcome:
    """The two probabilities a scenario produces, and the distance between them.

    It deliberately carries no decision and no recommendation: turning a probability into a
    decision is the policy's job, and mixing the two here would make it easy to read the
    delta as a consequence rather than as a difference between two model outputs.

    Attributes:
        changes: The attributes that were changed, and the values they were changed to.
        baseline_probability: What the model says about the applicant as given.
        scenario_probability: What the model says about the modified applicant.
        delta: `scenario_probability - baseline_probability`. A difference between two
            model outputs, never an estimated effect of making the change.
    """

    changes: Mapping[str, int]
    baseline_probability: float
    scenario_probability: float

    @property
    def delta(self) -> float:
        """Distance between the two model outputs.

        Returns:
            `scenario_probability - baseline_probability`. Negative means the model would
            score the modified applicant lower, and says nothing about causation.
        """
        return self.scenario_probability - self.baseline_probability


def apply_scenario(applicant: pd.DataFrame, changes: Mapping[str, int]) -> pd.DataFrame:
    """Build a modified copy of an applicant. The original is never touched.

    Args:
        applicant: Exactly one row, carrying the raw canonical columns.
        changes: Raw column name to the value it takes in the scenario.

    Returns:
        A new frame, equal to `applicant` except in the changed columns.

    Raises:
        ScenarioError: More or fewer than one row was given, no change was requested, or a
            change names a column that is not one of the 23 raw predictors.
        UnknownValueError: A proposed value is outside the data contract.
        MissingColumnsError: The applicant is missing raw columns. Nothing is imputed.
    """
    if len(applicant) != 1:
        raise ScenarioError(f"A scenario is about one applicant; {len(applicant)} rows were given.")
    if not changes:
        raise ScenarioError(
            "No change was requested. An empty scenario returns the baseline twice and a "
            "delta of zero, which reads like a measured result and is not one."
        )
    require_predictor_columns(applicant.columns)

    unknown = [column for column in changes if column not in PREDICTOR_COLUMNS]
    if unknown:
        raise ScenarioError(
            f"The scenario changes {unknown}, which the model does not read. Its inputs are "
            f"the {len(PREDICTOR_COLUMNS)} raw columns: {', '.join(PREDICTOR_COLUMNS)}. "
            "Derived features such as the utilisation ratios are computed by the pipeline "
            "and cannot be set directly - changing one would describe an applicant whose "
            "bills and limit do not add up to it."
        )
    require_known_values(changes)

    modified = applicant.copy(deep=True)
    for column, value in changes.items():
        modified[column] = value
    return modified


def evaluate_scenario(
    model: ProbabilityScorer, applicant: pd.DataFrame, changes: Mapping[str, int]
) -> ScenarioOutcome:
    """Score an applicant as given and as modified, and report both.

    Both probabilities come from the same pinned registry artefact and the same call path,
    so the difference between them is the change and nothing else.

    Args:
        model: Anything that scores raw applicant rows; in production the pinned artefact.
        applicant: Exactly one row, carrying the raw canonical columns.
        changes: Raw column name to the value it takes in the scenario.

    Returns:
        The baseline probability, the scenario probability, and the changes applied.

    Raises:
        ScenarioError: The scenario cannot be applied; see `apply_scenario`.
        UnknownValueError: A proposed value is outside the data contract.
        MissingColumnsError: The applicant is missing raw columns. Nothing is imputed.
    """
    modified = apply_scenario(applicant, changes)
    return ScenarioOutcome(
        changes=dict(changes),
        baseline_probability=float(model.probability_of_default(applicant)[0]),
        scenario_probability=float(model.probability_of_default(modified)[0]),
    )
