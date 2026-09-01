"""The registry loader and the two contracts that stand in front of the artefact.

These tests exercise the refusals rather than the happy path, because the refusals are the
guarantees. A model asked about a code it never saw does not fail: it routes the row down
whichever branch the comparison happens to take and returns a plausible number. Everything
here checks that such a row never reaches the forest.
"""

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from credit_copilot.data import schema
from credit_copilot.models import registry
from credit_copilot.models.registry import (
    PREDICTOR_COLUMNS,
    MissingColumnsError,
    ModelUnavailableError,
    RegisteredModel,
    UnknownValueError,
    load_registered_model,
    require_known_values,
    require_predictor_columns,
)
from credit_copilot.models.tracking import MLflowConfigurationError

# ---------------------------------------------------------------------------
# require_predictor_columns
# ---------------------------------------------------------------------------


def test_the_full_predictor_set_is_accepted() -> None:
    """The contract must not reject what the model was actually fitted on."""
    require_predictor_columns(list(PREDICTOR_COLUMNS))


def test_a_missing_column_is_named_and_nothing_is_imputed() -> None:
    """The message has to name the column: a caller cannot act on "some column"."""
    columns = [column for column in PREDICTOR_COLUMNS if column != "PAY_AMT3"]

    with pytest.raises(MissingColumnsError) as error:
        require_predictor_columns(columns)

    assert "PAY_AMT3" in str(error.value)
    assert "imputed" in str(error.value)


def test_every_missing_column_is_reported_not_just_the_first() -> None:
    """Reporting one at a time turns one fix into as many round trips as there are gaps."""
    columns = [column for column in PREDICTOR_COLUMNS if column not in {"AGE", "SEX", "PAY_AMT1"}]

    with pytest.raises(MissingColumnsError) as error:
        require_predictor_columns(columns)

    message = str(error.value)
    assert "AGE" in message
    assert "SEX" in message
    assert "PAY_AMT1" in message


def test_a_pandas_index_is_accepted_as_well_as_a_list() -> None:
    """Callers hand this `frame.columns`; it must not need conversion first."""
    require_predictor_columns(pd.DataFrame(columns=list(PREDICTOR_COLUMNS)).columns)


# ---------------------------------------------------------------------------
# require_known_values
# ---------------------------------------------------------------------------


def test_a_documented_category_is_accepted() -> None:
    """`EDUCATION = 2` is a level the source itself declares."""
    require_known_values({"EDUCATION": 2})


def test_a_code_adr_0004_accepted_on_measured_evidence_is_accepted() -> None:
    """`EDUCATION = 5` is undocumented by UCI and accepted by ADR-0004. Both contracts count.

    If this ever starts failing, the two maps in `schema.py` have been merged into one and
    the difference between *what the source declares* and *what this project accepted by
    measuring* has been lost.
    """
    assert 5 in schema.OBSERVED_CODES_ACCEPTED["EDUCATION"]
    require_known_values({"EDUCATION": 5})


def test_an_unknown_category_is_refused_and_says_what_was_admissible() -> None:
    """Naming only the offending value leaves the caller guessing what to send instead."""
    with pytest.raises(UnknownValueError) as error:
        require_known_values({"EDUCATION": 99})

    message = str(error.value)
    assert "EDUCATION=99" in message
    assert schema.ADR_UNDOCUMENTED_CODES in message


def test_a_value_below_its_plausible_range_is_refused_with_the_bound() -> None:
    """A range bound marks where a value stops being a business fact and becomes an error."""
    with pytest.raises(UnknownValueError) as error:
        require_known_values({"AGE": 3})

    assert "AGE=3" in str(error.value)


def test_a_value_above_its_plausible_range_is_refused() -> None:
    """The check has to fire on both bounds, not only the one someone happened to test."""
    with pytest.raises(UnknownValueError):
        require_known_values({"AGE": 500})


def test_a_column_outside_the_predictor_set_is_ignored() -> None:
    """The same check runs over a whole applicant and over a partial scenario.

    A scenario names two or three columns; refusing it for not being an applicant would make
    the function unusable for half its callers.
    """
    require_known_values({"NOT_A_PREDICTOR": -12345})


def test_a_numeric_column_inside_its_range_is_accepted() -> None:
    """The plausible interval must admit the data the model was fitted on."""
    require_known_values({"LIMIT_BAL": 50_000, "AGE": 35})


# ---------------------------------------------------------------------------
# RegisteredModel
# ---------------------------------------------------------------------------


def _pipeline_with(steps: dict[str, object]) -> Pipeline:
    """Build a pipeline carrying the given named steps."""
    return Pipeline(list(steps.items()))


def test_the_uri_is_quotable_in_a_credit_file() -> None:
    """`models:/name/version` is what makes a score reproducible months later."""
    model = RegisteredModel(
        name="credit-risk-default-probability",
        version="1",
        pipeline=_pipeline_with({"preprocess": DummyClassifier()}),
    )

    assert model.uri == "models:/credit-risk-default-probability/1"


def test_an_artefact_without_the_expected_steps_is_refused_by_name() -> None:
    """A pipeline with other steps is not the artefact the training script registers.

    Accepting it would mean the copilot explaining a model whose shape it guessed.
    """
    model = RegisteredModel(
        name="x", version="1", pipeline=_pipeline_with({"something_else": DummyClassifier()})
    )

    with pytest.raises(ModelUnavailableError) as error:
        _ = model.preprocessor

    assert "something_else" in str(error.value)


# ---------------------------------------------------------------------------
# load_registered_model - the three ways it refuses
# ---------------------------------------------------------------------------


def test_an_unconfigured_mlflow_becomes_a_model_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller handles one exception type, not MLflow's whole taxonomy.

    The message must still carry the configuration problem: swallowing it would send whoever
    reads it looking at the registry instead of at their `.env`.
    """

    def refuse() -> str:
        raise MLflowConfigurationError("MLFLOW_TRACKING_URI missing or blank")

    monkeypatch.setattr(registry, "configure_mlflow", refuse)

    with pytest.raises(ModelUnavailableError) as error:
        load_registered_model()

    assert "MLFLOW_TRACKING_URI" in str(error.value)


def test_a_registry_failure_names_the_uri_and_the_command_that_fixes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Version not found" is only actionable if the reader is told how to create it."""
    monkeypatch.setattr(registry, "configure_mlflow", lambda: "https://redacted")

    def explode(_uri: str) -> object:
        raise RuntimeError("Model Version not found")

    monkeypatch.setattr(registry.mlflow.sklearn, "load_model", explode)

    with pytest.raises(ModelUnavailableError) as error:
        load_registered_model(name="some-model", version="7")

    message = str(error.value)
    assert "models:/some-model/7" in message
    assert "scripts/register_production_model.py" in message


def test_an_artefact_that_is_not_a_pipeline_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole preprocessor-plus-model object is the guarantee; a bare estimator is not.

    A classifier alone would score a raw applicant row by silently misreading 23 raw columns
    as the 110-column matrix, which is the divergence the single-artefact rule exists to make
    impossible.
    """
    monkeypatch.setattr(registry, "configure_mlflow", lambda: "https://redacted")
    monkeypatch.setattr(registry.mlflow.sklearn, "load_model", lambda _uri: DummyClassifier())

    with pytest.raises(ModelUnavailableError) as error:
        load_registered_model()

    assert "DummyClassifier" in str(error.value)
    assert "Pipeline" in str(error.value)


def test_a_pipeline_is_returned_with_the_coordinates_it_came_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loader must report which version it loaded, not just hand back an object."""
    pipeline = _pipeline_with({"preprocess": DummyClassifier(), "model": DummyClassifier()})
    monkeypatch.setattr(registry, "configure_mlflow", lambda: "https://redacted")
    monkeypatch.setattr(registry.mlflow.sklearn, "load_model", lambda _uri: pipeline)

    model = load_registered_model(name="credit-risk-default-probability", version="1")

    assert model.version == "1"
    assert model.uri == "models:/credit-risk-default-probability/1"
    assert model.pipeline is pipeline
