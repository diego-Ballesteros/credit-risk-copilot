"""Tests of both HTTP surfaces, driven with `TestClient` and hand-built services.

Four families live here and they defend different things.

The **refusal** tests are the reason this file exists. Section 7.5 of `docs/MODEL_CARD.md`
recorded an open risk: the registered signature declares the 23 columns as integers, and an
integer cannot represent a missing value. These tests pin the resolution - a missing field, an
explicit null, a value outside its plausible range and a category no ADR accepts are all
refused with 422, the message names the field, and **the scorer is never called**. The last
clause is the one that matters: a refusal that still reached the model would mean the model
saw an applicant nobody validated, and every stub here counts its calls so that the assertion
is about behaviour rather than about a status code.

The **isolation** tests drive both applications with doubles - a scorer that computes a
readable function of the row, an explainer with a fixed attribution, a runner that replays a
recorded copilot run. No registry, no network, no API key, no embedding model. That is what
makes every assertion about the API and not about scikit-learn or Anthropic.

The **contract** tests check what a response must always carry: a probability never travels
without the threshold and the cost assumption behind it, and a citation never travels without
the identifier of the fragment it came from.

The **isolation-of-the-service** test is the one that cannot be written inside this process.
`test_model_app_does_not_import_the_agent_stack` starts a fresh interpreter, imports the model
application there, and asserts that `torch`, `chromadb`, `anthropic`, `langgraph` and
`sentence_transformers` are absent from `sys.modules`. The two-service split is a property of
the import graph, and an innocent convenience import restores the coupling in silence - so it
is a test rather than a note in a document.
"""

import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_copilot.api import agent_app, model_app
from credit_copilot.api.dependencies import (
    REQUEST_ID_HEADER,
    AgentServiceState,
    AppState,
    ModelService,
    ModelServiceState,
    ValidationMetrics,
    get_agent_state,
    get_model_state,
)
from credit_copilot.api.schemas import ApplicantAttributes
from credit_copilot.explain.shap_service import FeatureEffect, LocalExplanation
from credit_copilot.models.decision import COST_ASSUMPTION, OPERATING_THRESHOLD
from credit_copilot.models.registry import PREDICTOR_COLUMNS, MissingColumnsError

# ---------------------------------------------------------------------------
# One applicant, written out in full. Every value is inside the data contract.
# ---------------------------------------------------------------------------

APPLICANT: dict[str, int] = {
    "LIMIT_BAL": 120_000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 34,
    "PAY_STATUS_1": 1,
    "PAY_STATUS_2": 0,
    "PAY_STATUS_3": 0,
    "PAY_STATUS_4": -1,
    "PAY_STATUS_5": -1,
    "PAY_STATUS_6": -2,
    "BILL_AMT1": 45_000,
    "BILL_AMT2": 43_000,
    "BILL_AMT3": 41_000,
    "BILL_AMT4": 12_000,
    "BILL_AMT5": 8_000,
    "BILL_AMT6": 0,
    "PAY_AMT1": 2_000,
    "PAY_AMT2": 3_000,
    "PAY_AMT3": 12_000,
    "PAY_AMT4": 8_000,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0,
}

REGISTRY_METRICS: dict[str, float] = {
    "cv_pr_auc_mean": 0.564230,
    "cv_pr_auc_std": 0.007962,
    "cv_roc_auc_mean": 0.786279,
    "cv_brier_mean": 0.133408,
    "cv_baseline_trivial_pr_auc": 0.221200,
    "cv_baseline_logistic_pr_auc": 0.540173,
}
"""The shape `scripts/register_production_model.py` attaches to the run, not a live read."""


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class StubScorer:
    """A scorer whose probability is a readable function of the row, and that counts calls.

    The counter is the point. Several tests assert that a rejected request never reached the
    model, and "the response was a 422" does not establish that on its own.
    """

    name = "stub-model"
    version = "0"

    def __init__(self) -> None:
        self.calls = 0

    @property
    def uri(self) -> str:
        """Registry URI this double pretends to come from."""
        return f"models:/{self.name}/{self.version}"

    def probability_of_default(self, applicants: pd.DataFrame) -> np.ndarray:
        """Score rows, refusing any that is missing a raw column."""
        missing = [column for column in PREDICTOR_COLUMNS if column not in applicants.columns]
        if missing:
            raise MissingColumnsError(f"missing {missing}")
        self.calls += 1
        limit = applicants["LIMIT_BAL"].to_numpy(dtype=float)
        return np.asarray((300_000.0 - limit) / 1_000_000.0, dtype=np.float64)


class StubExplainer:
    """An explainer with a fixed attribution whose signs deliberately disagree."""

    def __init__(self, scorer: StubScorer) -> None:
        self._scorer = scorer
        self.calls = 0

    def explain(self, applicant: pd.DataFrame, top_n: int = 5) -> LocalExplanation:
        """Return a fixed attribution for the row, truncated to `top_n`."""
        self.calls += 1
        effects = (
            FeatureEffect("PAY_STATUS_1_1", 0.081, 0.081, "raises_risk", 1.0),
            FeatureEffect("LIMIT_BAL", -0.042, 0.042, "lowers_risk", 0.3),
            FeatureEffect("UTILIZATION_M2", 0.013, 0.013, "raises_risk", 0.9),
        )
        return LocalExplanation(
            probability_of_default=float(self._scorer.probability_of_default(applicant)[0]),
            forest_score=0.301,
            base_value=0.249,
            effects=effects[:top_n],
            features_considered=110,
        )


class StubRunner:
    """A copilot run, replayed. No graph, no API key, no index."""

    def __init__(self, final: dict[str, Any]) -> None:
        self._final = final
        self.seen: list[tuple[str, Any, int]] = []

    def run(self, query: str, applicant: Any, max_iterations: int) -> dict[str, Any]:
        """Record what was asked and return the recorded run."""
        self.seen.append((query, applicant, max_iterations))
        return dict(self._final)


def recorded_run(**overrides: Any) -> dict[str, Any]:
    """Build the shape `agent.graph.run_query` returns, as plain mappings.

    Plain mappings rather than the agent's Pydantic models on purpose: `/chat` reads the run
    structurally, and driving it with dictionaries is what proves the endpoint does not depend
    on the agent package being importable.

    Args:
        **overrides: Fields to replace in the recorded run.

    Returns:
        A final graph state.
    """
    final: dict[str, Any] = {
        "query": "¿El puntaje dio 0,19: qué banda es y qué debo hacer?",
        "answer": "El valor 0,19 cae en la banda D.",
        "outcome": "answered",
        "gap": "",
        "llm_calls": 5,
        "iterations": 2,
        "tool_records": [
            {
                "call_id": "toolu_01",
                "name": "consultar_politica",
                "arguments": {"question": "banda de decisión", "probability_of_default": 0.19},
                "ok": True,
                "result": {"band": {"code": "D"}},
                "error": None,
            },
            {
                "call_id": "toolu_02",
                "name": "score_solicitante",
                "arguments": {},
                "ok": False,
                "result": None,
                "error": "`score_solicitante` necesita un solicitante y no hay ninguno.",
            },
        ],
        "citations": [
            {
                "chunk_id": "politica-interna-credito::005-2-1-tabla-de-bandas",
                "citation": "Política Interna de Otorgamiento, sección 2.1",
                "document_id": "politica-interna-credito",
                "location": "2. Bandas > 2.1 Tabla de bandas",
                "is_synthetic": True,
                "integrity_notice": "DOCUMENTO SINTÉTICO: redactado para este proyecto.",
            },
            {
                "chunk_id": "politica-interna-credito::005-2-1-tabla-de-bandas",
                "citation": "Política Interna de Otorgamiento, sección 2.1",
                "document_id": "politica-interna-credito",
                "location": "2. Bandas > 2.1 Tabla de bandas",
                "is_synthetic": True,
                "integrity_notice": "DOCUMENTO SINTÉTICO: redactado para este proyecto.",
            },
        ],
        "token_usage": [
            {
                "node": "plan",
                "model": "claude-haiku-4-5",
                "input_tokens": 1200,
                "output_tokens": 80,
                "cache_read_tokens": 0,
            }
        ],
    }
    final.update(overrides)
    return final


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def start_model_load_with(holder: AppState, load: Any) -> Any:
    """Start a background load with a substituted loader.

    `start_model_load` is `_start_load` bound to the real loader; the tests need the same
    machinery bound to a double, so they call the same private starter rather than
    reimplementing thread management they would then be testing instead of the real one.
    """
    from credit_copilot.api.dependencies import _start_load

    return _start_load(holder, "model", load, "model")


@pytest.fixture
def scorer() -> StubScorer:
    """A scorer that counts how many times the model was actually reached."""
    return StubScorer()


@pytest.fixture
def model_client(scorer: StubScorer) -> Any:
    """A client over the model application, wired to the doubles instead of the registry."""
    service = ModelService(
        model=scorer,  # type: ignore[arg-type]  # a double standing in for RegisteredModel
        explainer=StubExplainer(scorer),  # type: ignore[arg-type]
        validation=ValidationMetrics(run_id="run-stub", metrics=dict(REGISTRY_METRICS)),
        tracking_uri="https://dagshub.example/mlflow",
    )
    state = ModelServiceState(phase="ready", service=service, elapsed_seconds=1.5)
    app = model_app.create_app(AppState(model=state))
    app.dependency_overrides[get_model_state] = lambda: state
    with TestClient(app) as client:
        yield client


@pytest.fixture
def unloaded_model_client() -> Any:
    """A client over a model application whose registry could not be reached at start-up."""
    state = ModelServiceState(
        phase="degraded", error="Could not load models:/x/1: ConnectionError", elapsed_seconds=263.2
    )
    app = model_app.create_app(AppState(model=state))
    app.dependency_overrides[get_model_state] = lambda: state
    with TestClient(app) as client:
        yield client


def agent_client(runner: StubRunner) -> TestClient:
    """Build a client over the agent application, wired to a replayed run."""
    state = AgentServiceState(
        phase="ready", runner=runner, default_max_iterations=3, model_loaded=True
    )
    app = agent_app.create_app(AppState(agent=state))
    app.dependency_overrides[get_agent_state] = lambda: state
    return TestClient(app)


# ---------------------------------------------------------------------------
# The contract: a probability never travels alone
# ---------------------------------------------------------------------------


def test_predict_returns_probability_threshold_and_cost_assumption(
    model_client: TestClient,
) -> None:
    response = model_client.post("/predict", json={"applicant": APPLICANT})

    assert response.status_code == 200
    body = response.json()
    assert body["probability_of_default"] == pytest.approx(0.18)
    assert body["decision"] == "refuse"
    context = body["decision_context"]
    assert context["threshold"] == OPERATING_THRESHOLD
    assert context["cost_ratio_fn_to_fp"] == 5
    assert context["cost_assumption"] == COST_ASSUMPTION
    assert "no medido" in context["cost_assumption"]
    assert context["decision_caveat"]
    assert body["model"]["version"] == "0"


def test_predict_below_the_threshold_recommends_approving(model_client: TestClient) -> None:
    response = model_client.post(
        "/predict", json={"applicant": {**APPLICANT, "LIMIT_BAL": 250_000}}
    )

    body = response.json()
    assert body["probability_of_default"] == pytest.approx(0.05)
    assert body["decision"] == "approve"


def test_every_endpoint_that_reports_a_probability_reports_the_threshold_too(
    model_client: TestClient,
) -> None:
    payloads = {
        "/predict": {"applicant": APPLICANT},
        "/explain": {"applicant": APPLICANT, "top_n": 3},
        "/simulate": {"applicant": APPLICANT, "changes": {"AGE": 45}},
    }
    for path, payload in payloads.items():
        body = model_client.post(path, json=payload).json()
        assert body["decision_context"]["threshold"] == OPERATING_THRESHOLD, path
        assert body["decision_context"]["cost_assumption"] == COST_ASSUMPTION, path


# ---------------------------------------------------------------------------
# The refusals. None of these may reach the model.
# ---------------------------------------------------------------------------


def test_a_missing_field_is_refused_by_name_and_never_reaches_the_model(
    model_client: TestClient, scorer: StubScorer
) -> None:
    incomplete = {key: value for key, value in APPLICANT.items() if key != "PAY_AMT3"}

    response = model_client.post("/predict", json={"applicant": incomplete})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["type"] == "invalid_request"
    assert "PAY_AMT3" in error["fields"]
    assert "PAY_AMT3" in error["message"]
    assert scorer.calls == 0
    assert "probability_of_default" not in response.text


def test_an_explicit_null_is_refused_and_never_becomes_a_zero(
    model_client: TestClient, scorer: StubScorer
) -> None:
    response = model_client.post("/predict", json={"applicant": {**APPLICANT, "PAY_AMT3": None}})

    assert response.status_code == 422
    assert "PAY_AMT3" in response.json()["error"]["fields"]
    assert scorer.calls == 0


def test_a_value_outside_its_plausible_range_is_refused(
    model_client: TestClient, scorer: StubScorer
) -> None:
    response = model_client.post("/predict", json={"applicant": {**APPLICANT, "AGE": 7}})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["fields"] == ("AGE",) or error["fields"] == ["AGE"]
    assert "18" in error["message"]
    assert scorer.calls == 0


def test_an_unknown_category_is_refused_and_names_what_was_admissible(
    model_client: TestClient, scorer: StubScorer
) -> None:
    response = model_client.post("/predict", json={"applicant": {**APPLICANT, "PAY_STATUS_1": 15}})

    assert response.status_code == 422
    error = response.json()["error"]
    assert "PAY_STATUS_1" in error["fields"]
    assert "-1, 1, 2, 3, 4, 5, 6, 7, 8, 9" in error["message"]
    assert scorer.calls == 0


def test_an_unexpected_attribute_is_refused_rather_than_ignored(
    model_client: TestClient, scorer: StubScorer
) -> None:
    response = model_client.post("/predict", json={"applicant": {**APPLICANT, "SALARY": 1_000_000}})

    assert response.status_code == 422
    assert "SALARY" in response.json()["error"]["fields"]
    assert scorer.calls == 0


def test_a_boolean_is_not_coerced_into_a_repayment_status(
    model_client: TestClient, scorer: StubScorer
) -> None:
    """`true` read as `PAY_STATUS_1 = 1` would invent a month of arrears out of a type error."""
    response = model_client.post(
        "/predict", json={"applicant": {**APPLICANT, "PAY_STATUS_1": True}}
    )

    assert response.status_code == 422
    assert "PAY_STATUS_1" in response.json()["error"]["fields"]
    assert scorer.calls == 0


def test_a_refusal_does_not_echo_the_applicant_back(model_client: TestClient) -> None:
    """The caller is told what was wrong, not what they sent. They already have that."""
    response = model_client.post("/predict", json={"applicant": {**APPLICANT, "AGE": 7}})

    body = response.text
    assert "AGE" in body
    assert "67521" not in body, "the bill statements must not travel back in an error body"
    assert "120000" not in body


def test_a_scenario_that_changes_a_derived_feature_is_refused(model_client: TestClient) -> None:
    response = model_client.post(
        "/simulate", json={"applicant": APPLICANT, "changes": {"UTILIZATION_M2": 1}}
    )

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_scenario"


def test_an_empty_scenario_is_refused(model_client: TestClient) -> None:
    response = model_client.post("/simulate", json={"applicant": APPLICANT, "changes": {}})

    assert response.status_code == 422
    assert "changes" in response.json()["error"]["fields"]


# ---------------------------------------------------------------------------
# What the other model endpoints must carry
# ---------------------------------------------------------------------------


def test_explain_reports_individual_direction_and_says_what_it_decomposes(
    model_client: TestClient,
) -> None:
    body = model_client.post("/explain", json={"applicant": APPLICANT, "top_n": 3}).json()

    assert len(body["top_features"]) == 3
    assert body["top_features"][0]["feature"] == "PAY_STATUS_1_1"
    assert body["top_features"][0]["direction"] == "raises_risk"
    assert body["top_features"][1]["direction"] == "lowers_risk"
    assert "PARA ESTE SOLICITANTE" in body["direction_note"]
    assert body["forest_score"] != body["probability_of_default"]
    assert body["features_considered"] == 110


def test_simulate_declares_that_it_speaks_about_the_model_and_not_about_the_world(
    model_client: TestClient,
) -> None:
    body = model_client.post(
        "/simulate", json={"applicant": APPLICANT, "changes": {"LIMIT_BAL": 250_000}}
    ).json()

    assert body["claim_type"] == "about_the_model"
    assert "no sobre el mundo" in body["causal_note"]
    assert body["baseline_probability"] == pytest.approx(0.18)
    assert body["scenario_probability"] == pytest.approx(0.05)
    assert body["delta"] == pytest.approx(-0.13)
    assert body["baseline_decision"] == "refuse"
    assert body["scenario_decision"] == "approve"


def test_model_info_reports_the_primary_metric_with_its_baselines(
    model_client: TestClient,
) -> None:
    payload = model_client.get("/model-info").json()

    metrics = {row["name"]: row for row in payload["validation"]["metrics"]}
    assert metrics["pr_auc"]["is_primary"] is True
    assert metrics["pr_auc"]["value"] == pytest.approx(0.564230)
    assert metrics["pr_auc"]["std"] == pytest.approx(0.007962)
    assert metrics["pr_auc"]["baselines"]["trivial"] == pytest.approx(0.221200)
    assert metrics["pr_auc"]["baselines"]["logistic_l2"] == pytest.approx(0.540173)
    assert metrics["roc_auc"]["baselines"] == {}
    assert metrics["roc_auc"]["baseline_note"]
    assert payload["validation"]["run_id"] == "run-stub"
    assert payload["decision_context"]["threshold"] == OPERATING_THRESHOLD
    assert "model_card" in payload["documentation"]
    assert any("fuera de tiempo" in limit for limit in payload["limitations"])


# ---------------------------------------------------------------------------
# Health, with and without the artefact
# ---------------------------------------------------------------------------


def test_health_answers_without_the_model_loaded_and_says_so(
    unloaded_model_client: TestClient,
) -> None:
    response = unloaded_model_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "model"
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert "ConnectionError" in body["detail"]


def test_scoring_without_the_model_loaded_is_503_and_never_a_probability(
    unloaded_model_client: TestClient,
) -> None:
    response = unloaded_model_client.post("/predict", json={"applicant": APPLICANT})

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "model_unavailable"
    assert "probability_of_default" not in response.text


def test_health_reports_ok_when_the_artefact_is_loaded(model_client: TestClient) -> None:
    body = model_client.get("/health").json()

    assert body["status"] == "ok"
    assert body["phase"] == "ready"
    assert body["model_loaded"] is True
    assert body["detail"] is None
    assert body["load_seconds"] == 1.5


# ---------------------------------------------------------------------------
# The three load phases, which is what ADR-0010 decision 3 bought
# ---------------------------------------------------------------------------


def _client_in_phase(state: ModelServiceState) -> TestClient:
    """Build a model application pinned to one loader phase."""
    app = model_app.create_app(AppState(model=state))
    app.dependency_overrides[get_model_state] = lambda: state
    return TestClient(app)


def test_health_answers_200_while_still_loading_because_loading_is_not_unhealthy() -> None:
    """The container healthcheck reads this. A restart here would restart the load."""
    with _client_in_phase(ModelServiceState(phase="loading")) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "starting"
    assert body["phase"] == "loading"
    assert body["model_loaded"] is False
    assert body["detail"] is None, "there is no failure to report while it is still working"
    assert body["load_seconds"] is None


def test_loading_and_degraded_are_two_different_503s() -> None:
    """'Not yet' and 'it could not' are different instructions to whoever is on call."""
    with _client_in_phase(ModelServiceState(phase="loading")) as client:
        loading = client.post("/predict", json={"applicant": APPLICANT})
    degraded_state = ModelServiceState(phase="degraded", error="ConnectionError to the registry")
    with _client_in_phase(degraded_state) as client:
        degraded = client.post("/predict", json={"applicant": APPLICANT})

    assert loading.status_code == 503
    assert loading.json()["error"]["type"] == "model_loading"
    assert loading.headers["Retry-After"] == "5"
    assert "reintenta" in loading.json()["error"]["message"]

    assert degraded.status_code == 503
    assert degraded.json()["error"]["type"] == "model_unavailable"
    assert "Retry-After" not in degraded.headers, (
        "a service that has finished failing must not invite a client to poll it"
    )
    assert "ConnectionError" in degraded.json()["error"]["message"]

    for response in (loading, degraded):
        assert "probability_of_default" not in response.text


def test_ready_is_the_gate_and_health_is_the_report() -> None:
    phases = {
        "loading": ModelServiceState(phase="loading"),
        "degraded": ModelServiceState(phase="degraded", error="registry down"),
    }
    for name, state in phases.items():
        with _client_in_phase(state) as client:
            assert client.get("/health").status_code == 200, name
            assert client.get("/ready").status_code == 503, name


def test_ready_answers_200_once_the_artefact_is_in_memory(model_client: TestClient) -> None:
    response = model_client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["phase"] == "ready"
    assert body["service"] == "model"


def test_the_agent_reports_the_same_three_phases() -> None:
    runner = StubRunner(recorded_run())
    states = {
        "loading": AgentServiceState(phase="loading"),
        "degraded": AgentServiceState(phase="degraded", error="no API key"),
        "ready": AgentServiceState(phase="ready", runner=runner, model_loaded=True),
    }
    seen = {}
    for name, state in states.items():
        app = agent_app.create_app(AppState(agent=state))
        app.dependency_overrides[get_agent_state] = lambda state=state: state
        with TestClient(app) as client:
            seen[name] = (client.get("/health").json(), client.get("/ready").status_code)

    assert seen["loading"][0]["status"] == "starting"
    assert seen["degraded"][0]["status"] == "degraded"
    assert seen["ready"][0]["status"] == "ok"
    assert (seen["loading"][1], seen["degraded"][1], seen["ready"][1]) == (503, 503, 200)


def test_the_lifespan_does_not_block_on_the_load() -> None:
    """The regression test for the 263-second start-up: the server must serve immediately.

    The loader is replaced with one that sleeps far longer than any acceptable start-up. If
    the load were still inside the lifespan, `TestClient.__enter__` would not return until it
    finished and this test would hang rather than fail - which is why the sleep is bounded and
    the assertion is on elapsed time.
    """
    import time as _time

    started = _time.perf_counter()

    def _slow_load() -> ModelServiceState:
        _time.sleep(30)
        return ModelServiceState(phase="degraded", error="never reached")

    holder = AppState()
    thread = start_model_load_with(holder, _slow_load)
    app = model_app.create_app(holder)
    with TestClient(app) as client:
        elapsed = _time.perf_counter() - started
        body = client.get("/health").json()

    assert elapsed < 5.0, f"start-up blocked for {elapsed:.1f}s; the load is not in the background"
    assert body["phase"] == "loading"
    assert thread.is_alive(), "the loader is still working while the service answers"


def test_the_background_loader_publishes_its_outcome() -> None:
    holder = AppState()
    expected = ModelServiceState(phase="degraded", error="registry refused the connection")

    thread = start_model_load_with(holder, lambda: expected)
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert holder.model is expected
    assert holder.model.phase == "degraded"


# ---------------------------------------------------------------------------
# Correlation identifiers
# ---------------------------------------------------------------------------


def test_every_response_carries_a_correlation_identifier(model_client: TestClient) -> None:
    response = model_client.get("/health")

    assert response.headers[REQUEST_ID_HEADER]
    assert response.json()["request_id"] == response.headers[REQUEST_ID_HEADER]


def test_a_well_formed_incoming_correlation_identifier_is_kept(
    model_client: TestClient,
) -> None:
    response = model_client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc.123"})

    assert response.headers[REQUEST_ID_HEADER] == "trace-abc.123"


def test_a_correlation_identifier_that_could_forge_a_log_line_is_discarded(
    model_client: TestClient,
) -> None:
    response = model_client.get("/health", headers={REQUEST_ID_HEADER: "abc def"})

    assert response.headers[REQUEST_ID_HEADER] != "abc def"
    assert response.json()["request_id"] == response.headers[REQUEST_ID_HEADER]


# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------


def test_chat_returns_citations_with_their_fragment_identifier_and_integrity_mark() -> None:
    runner = StubRunner(recorded_run())
    with agent_client(runner) as client:
        body = client.post("/chat", json={"query": "El puntaje dio 0,19, ¿qué hago?"}).json()

    assert body["citations"], "the recorded run used a citation and it must reach the caller"
    for citation in body["citations"]:
        assert citation["chunk_id"], "no citation may arrive without a fragment identifier"
        assert citation["citation"]
    first = body["citations"][0]
    assert first["is_synthetic"] is True
    assert first["has_integrity_notice"] is True
    assert "SINTÉTICO" in first["integrity_notice"]
    assert len(body["citations"]) == 1, "the same fragment cited twice is reported once"


def test_chat_reports_every_tool_call_including_the_one_that_refused() -> None:
    runner = StubRunner(recorded_run())
    with agent_client(runner) as client:
        body = client.post("/chat", json={"query": "El puntaje dio 0,19, ¿qué hago?"}).json()

    names = [call["name"] for call in body["tools_invoked"]]
    assert names == ["consultar_politica", "score_solicitante"]
    refused = body["tools_invoked"][1]
    assert refused["ok"] is False
    assert "no hay ninguno" in refused["error"]
    assert body["tools_invoked"][0]["arguments"]["probability_of_default"] == 0.19


def test_chat_stands_on_its_own() -> None:
    """The response carries what it costs, how it ended and how to read it."""
    runner = StubRunner(recorded_run())
    with agent_client(runner) as client:
        body = client.post("/chat", json={"query": "El puntaje dio 0,19, ¿qué hago?"}).json()

    assert body["llm_calls"] == 5
    assert body["iterations"] == 2
    assert body["max_iterations"] == 3
    assert body["outcome"] == "answered"
    assert body["outcome_meaning"]
    assert body["token_usage"][0]["model"] == "claude-haiku-4-5"
    assert body["decision_context"]["threshold"] == OPERATING_THRESHOLD
    assert "politica_sintetica" in body["reading_notes"]
    assert "que_es_una_cita" in body["reading_notes"]


def test_chat_explains_an_outcome_with_gaps() -> None:
    runner = StubRunner(
        recorded_run(outcome="answered_with_gaps", gap="Falta el artículo sobre garantías.")
    )
    with agent_client(runner) as client:
        body = client.post("/chat", json={"query": "¿Qué dice sobre garantías?"}).json()

    assert body["outcome"] == "answered_with_gaps"
    assert "tope de ciclos" in body["outcome_meaning"]
    assert body["unresolved_gap"] == "Falta el artículo sobre garantías."


def test_chat_passes_the_applicant_through_the_same_data_contract() -> None:
    runner = StubRunner(recorded_run())
    with agent_client(runner) as client:
        rejected = client.post(
            "/chat",
            json={"query": "¿Apruebo a este solicitante?", "applicant": {**APPLICANT, "AGE": 7}},
        )
        accepted = client.post(
            "/chat", json={"query": "¿Apruebo a este solicitante?", "applicant": APPLICANT}
        )

    assert rejected.status_code == 422
    assert "AGE" in rejected.json()["error"]["fields"]
    assert accepted.status_code == 200
    assert runner.seen == [("¿Apruebo a este solicitante?", APPLICANT, 3)]


def test_chat_without_the_copilot_loaded_is_503() -> None:
    state = AgentServiceState(phase="degraded", error="CopilotConfigurationError: no API key")
    app = agent_app.create_app(AppState(agent=state))
    app.dependency_overrides[get_agent_state] = lambda: state
    with TestClient(app) as client:
        response = client.post("/chat", json={"query": "hola qué tal"})
        health = client.get("/health")

    assert response.status_code == 503
    assert health.json()["status"] == "degraded"
    assert health.json()["model_loaded"] is False
    assert "no API key" in health.json()["detail"]


def test_a_query_too_short_to_be_a_question_is_refused() -> None:
    runner = StubRunner(recorded_run())
    with agent_client(runner) as client:
        response = client.post("/chat", json={"query": "?"})

    assert response.status_code == 422
    assert "query" in response.json()["error"]["fields"]


# ---------------------------------------------------------------------------
# Drift, and the two-service split
# ---------------------------------------------------------------------------


def test_the_http_applicant_contract_is_the_project_data_contract() -> None:
    """The API restates none of the 23 columns: it inherits them from `ApplicantRecord`."""
    assert set(ApplicantAttributes.model_fields) == set(PREDICTOR_COLUMNS)


def test_the_reading_notes_name_documents_the_corpus_actually_holds() -> None:
    """A warning about a document that is not in the corpus warns about nothing."""
    from credit_copilot.api.agent_app import _READING_NOTES
    from credit_copilot.config import settings

    present = {path.stem for path in settings.corpus_dir.glob("*.md")}

    assert "politica-interna-credito" in present
    assert "circular-basica-contable-sfc-cap-ii" in present
    assert "politica-interna-credito" in _READING_NOTES["politica_sintetica"]
    assert "circular-basica-contable-sfc-cap-ii" in _READING_NOTES["capitulo_derogado"]


_IMPORT_PURITY_PROBE = """
import sys
import credit_copilot.api.model_app  # noqa: F401
forbidden = ["torch", "chromadb", "anthropic", "langgraph", "sentence_transformers"]
print(",".join(name for name in forbidden if name in sys.modules))
"""


def test_model_app_does_not_import_the_agent_stack() -> None:
    """The two-service split is a property of the import graph, so it is checked as one.

    A fresh interpreter is the only place this can be established: inside the test process
    every one of these modules is already imported by the tests of the agent.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PURITY_PROBE],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "", (
        f"the model service must not carry the agent stack; it imported: {completed.stdout.strip()}"
    )
