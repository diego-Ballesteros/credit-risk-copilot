"""Tests of the four tools, each one in isolation, against hand-built collaborators.

Three families live here and they defend different things.

The **contract** tests build inputs by hand and check that the tool refuses what it should
refuse. They matter most for the refusals: a tool that silently accepts a half-filled
applicant does not fail, it produces a probability, and a probability is exactly the kind of
output nobody inspects twice.

The **isolation** tests drive the tools with stub collaborators - a scorer that returns a
number computed from the row it was handed, an explainer with a fixed attribution, a
retriever over three fragments written in this file. No registry, no network, no embedding
model, no 300-tree forest. That is what makes every assertion below about the tool and not
about scikit-learn.

The **drift** tests compare two copies of one fact. `ApplicantRecord` restates the 23 columns
of `schema.py`, and `POLICY_BANDS` restates the band table of the corpus; both duplications
are deliberate and both would rot in silence, so each has a test that fails when the copies
disagree.
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from credit_copilot.agent.tools import (
    OPERATING_THRESHOLD,
    POLICY_BANDS,
    POLICY_BANDS_CHUNK_ID,
    ApplicantRecord,
    CorpusRetriever,
    ExplainInput,
    PolicyInput,
    RetrievedFragment,
    ScoreInput,
    SimulateInput,
    ToolContext,
    ToolExecutionError,
    _fragment_from_chunk,
    _fragment_from_search,
    anthropic_tool_definitions,
    consultar_politica,
    execute,
    explicar_decision,
    resolve_band,
    score_solicitante,
    simular_escenario,
)
from credit_copilot.config import settings
from credit_copilot.explain.shap_service import FeatureEffect, LocalExplanation
from credit_copilot.models.registry import PREDICTOR_COLUMNS, MissingColumnsError
from credit_copilot.rag.chunking import chunk_corpus
from credit_copilot.rag.documents import load_corpus
from credit_copilot.rag.vectorstore import SearchResult

# ---------------------------------------------------------------------------
# One applicant, written out in full. Every value is inside the data contract:
# SEX 1-2, EDUCATION 1-4 plus 0/5/6, MARRIAGE 1-3 plus 0, PAY_STATUS -2..9.
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


class StubScorer:
    """A scorer whose probability is a readable function of the row it was handed.

    The formula is arbitrary and that is the point: the tests assert relationships between
    calls - the baseline differs from the scenario, the original frame is unchanged - not a
    number that came out of a model.
    """

    name = "stub-model"
    version = "0"

    def __init__(self) -> None:
        self.seen: list[pd.DataFrame] = []

    def probability_of_default(self, applicants: pd.DataFrame) -> pd.Series:
        missing = [column for column in PREDICTOR_COLUMNS if column not in applicants.columns]
        if missing:
            raise MissingColumnsError(f"missing {missing}")
        self.seen.append(applicants.copy(deep=True))
        limit = applicants["LIMIT_BAL"].to_numpy(dtype=float)
        return (300_000.0 - limit) / 1_000_000.0


class StubExplainer:
    """An explainer with a fixed attribution whose signs deliberately disagree."""

    def __init__(self) -> None:
        self.effects = (
            FeatureEffect("PAY_STATUS_1_1", 0.081, 0.081, "raises_risk", 1.0),
            FeatureEffect("LIMIT_BAL", -0.042, 0.042, "lowers_risk", 0.3),
            FeatureEffect("UTILIZATION_M2", 0.013, 0.013, "raises_risk", 0.9),
        )

    def explain(self, applicant: pd.DataFrame, top_n: int) -> LocalExplanation:
        assert len(applicant) == 1
        return LocalExplanation(
            probability_of_default=0.19,
            forest_score=0.21,
            base_value=0.156,
            effects=self.effects[:top_n],
            features_considered=110,
        )


BANDS_FRAGMENT = RetrievedFragment(
    chunk_id=POLICY_BANDS_CHUNK_ID,
    citation="Política Interna de Crédito (documento sintético), 2.1. Tabla de bandas",
    document_id="politica-interna-credito",
    location="2. Bandas de decisión por score > 2.1. Tabla de bandas",
    text="AVISO: DOCUMENTO SINTÉTICO.\n\nBanda D: 0,160 <= PD < 0,300. Rechazar.",
    is_synthetic=True,
    integrity_notice="DOCUMENTO SINTÉTICO.",
)

LAW_FRAGMENT = RetrievedFragment(
    chunk_id="ley-1266-2008-habeas-data::013-art-13",
    citation="Ley 1266 de 2008, artículo 13",
    document_id="ley-1266-2008-habeas-data",
    location="Artículo 13",
    text="La información de carácter positivo permanecerá de manera indefinida.",
    score=0.87,
)


class StubRetriever:
    """A retriever over the fragments declared in this file, with no index behind it."""

    def __init__(self, fragments: tuple[RetrievedFragment, ...] = (LAW_FRAGMENT,)) -> None:
        self.fragments = fragments
        self.by_id = {BANDS_FRAGMENT.chunk_id: BANDS_FRAGMENT}
        self.queries: list[str] = []

    def retrieve(self, query, top_k, document_ids):  # noqa: ANN001, ANN201
        self.queries.append(query)
        if document_ids is not None:
            unknown = set(document_ids) - {fragment.document_id for fragment in self.fragments}
            if unknown:
                raise ToolExecutionError(f"unknown documents {sorted(unknown)}")
        return self.fragments[:top_k]

    def fragment_by_chunk_id(self, chunk_id):  # noqa: ANN001, ANN201
        return self.by_id.get(chunk_id)


@pytest.fixture
def context() -> ToolContext:
    """A tool context with no model, no index and no network behind it."""
    return ToolContext(scorer=StubScorer(), explainer=StubExplainer(), retriever=StubRetriever())


@pytest.fixture(scope="module")
def corpus_fragments() -> dict[str, RetrievedFragment]:
    """Every chunk of the real corpus, converted to a fragment."""
    chunks = chunk_corpus(load_corpus(settings.corpus_dir))
    return {chunk.chunk_id: _fragment_from_chunk(chunk) for chunk in chunks}


# ---------------------------------------------------------------------------
# The applicant contract: it refuses rather than imputes
# ---------------------------------------------------------------------------


def test_applicant_record_declares_exactly_the_model_input_columns():
    # The record restates schema.py. If the two ever disagree, the copilot would be
    # validating a contract the model does not have.
    assert set(ApplicantRecord.model_fields) == set(PREDICTOR_COLUMNS)
    assert len(PREDICTOR_COLUMNS) == 23


def test_score_fails_loudly_when_a_column_is_missing_and_imputes_nothing(context):
    incomplete = {key: value for key, value in APPLICANT.items() if key != "PAY_AMT3"}

    with pytest.raises(ToolExecutionError) as error:
        execute("score_solicitante", {}, context, applicant=incomplete)

    assert "PAY_AMT3" in str(error.value)
    # The scorer was never reached: nothing was filled in and then scored.
    assert context.scorer.seen == []


def test_score_fails_when_several_columns_are_missing_and_names_all_of_them(context):
    incomplete = {"LIMIT_BAL": 120_000, "SEX": 2, "AGE": 34}

    with pytest.raises(ToolExecutionError) as error:
        execute("score_solicitante", {}, context, applicant=incomplete)

    message = str(error.value)
    assert "EDUCATION" in message
    assert "PAY_STATUS_1" in message
    assert context.scorer.seen == []


def test_applicant_record_refuses_an_undocumented_category():
    # PAY_STATUS runs -2..9. A 15 is a code nobody has looked at, and the forest would
    # route it down some branch and return a number shaped like every other number.
    with pytest.raises(ValidationError) as error:
        ApplicantRecord.model_validate({**APPLICANT, "PAY_STATUS_1": 15})

    assert "PAY_STATUS_1" in str(error.value)


def test_applicant_record_refuses_an_extra_column():
    with pytest.raises(ValidationError):
        ApplicantRecord.model_validate({**APPLICANT, "SALARY": 1_000})


def test_tools_that_need_an_applicant_refuse_when_there_is_none(context):
    for name, arguments in (
        ("score_solicitante", {}),
        ("explicar_decision", {}),
        ("simular_escenario", {"changes": {"LIMIT_BAL": 200_000}}),
    ):
        with pytest.raises(ToolExecutionError) as error:
            execute(name, arguments, context, applicant=None)
        assert "solicitante" in str(error.value).lower()


def test_the_model_is_never_offered_a_schema_that_carries_applicant_attributes():
    # The rule "ninguna herramienta inventa un número" is enforced by the shape of the
    # schema, not by the prompt: the attributes are simply not askable.
    for definition in anthropic_tool_definitions():
        properties = set(definition["input_schema"].get("properties", {}))
        assert not properties & set(PREDICTOR_COLUMNS)
        assert "applicant" not in properties


# ---------------------------------------------------------------------------
# score_solicitante
# ---------------------------------------------------------------------------


def test_score_reports_the_threshold_and_the_cost_assumption_behind_it(context):
    output = score_solicitante(ScoreInput(applicant=ApplicantRecord(**APPLICANT)), context)

    # (300000 - 120000) / 1e6 = 0.18, above the 0.160 threshold.
    assert output.probability_of_default == pytest.approx(0.18)
    assert output.decision == "refuse"
    assert output.threshold == OPERATING_THRESHOLD
    assert "5" in output.cost_assumption
    assert "0,160" in output.cost_assumption
    assert output.decision_caveat


def test_score_approves_below_the_threshold_and_the_boundary_refuses(context):
    # 300000 - LIMIT_BAL = 160000 gives exactly the threshold, which the policy puts in
    # band D: the comparison is >=, not >.
    at_threshold = score_solicitante(
        ScoreInput(applicant=ApplicantRecord(**{**APPLICANT, "LIMIT_BAL": 140_000})), context
    )
    below = score_solicitante(
        ScoreInput(applicant=ApplicantRecord(**{**APPLICANT, "LIMIT_BAL": 250_000})), context
    )

    assert at_threshold.probability_of_default == pytest.approx(OPERATING_THRESHOLD)
    assert at_threshold.decision == "refuse"
    assert below.decision == "approve"


# ---------------------------------------------------------------------------
# explicar_decision
# ---------------------------------------------------------------------------


def test_explanation_direction_is_the_sign_of_this_applicants_value(context):
    output = explicar_decision(
        ExplainInput(applicant=ApplicantRecord(**APPLICANT), top_n=3), context
    )

    # Two features push up and one pushes down. Each direction follows its own sign,
    # never a majority, an average or a population statistic.
    by_name = {feature.feature: feature for feature in output.top_features}
    assert by_name["PAY_STATUS_1_1"].direction == "raises_risk"
    assert by_name["LIMIT_BAL"].direction == "lowers_risk"
    assert by_name["UTILIZATION_M2"].direction == "raises_risk"
    for feature in output.top_features:
        assert (feature.shap_value > 0) == (feature.direction == "raises_risk")


def test_explanation_ships_both_warnings_and_says_what_the_values_add_up_to(context):
    output = explicar_decision(
        ExplainInput(applicant=ApplicantRecord(**APPLICANT), top_n=2), context
    )

    assert len(output.top_features) == 2
    assert "media poblacional" in output.direction_note
    assert "causal" in output.causal_note
    # The values decompose the forest score, not the calibrated probability.
    assert output.forest_score != output.probability_of_default


def test_explanation_honours_top_n_through_the_llm_facing_path(context):
    output = execute("explicar_decision", {"top_n": 1}, context, applicant=APPLICANT)

    assert len(output.top_features) == 1


def test_explanation_refuses_a_top_n_outside_the_contract(context):
    with pytest.raises(ToolExecutionError):
        execute("explicar_decision", {"top_n": 0}, context, applicant=APPLICANT)


# ---------------------------------------------------------------------------
# simular_escenario
# ---------------------------------------------------------------------------


def test_simulation_does_not_modify_the_original_applicant(context):
    before = dict(APPLICANT)
    record = ApplicantRecord(**APPLICANT)
    original_frame = record.to_frame()

    output = simular_escenario(
        SimulateInput(applicant=record, changes={"LIMIT_BAL": 260_000}), context
    )

    # The dictionary the caller holds, the validated record and the frame built from it
    # all still describe the applicant that was given.
    assert before == APPLICANT
    assert record.LIMIT_BAL == 120_000
    assert original_frame.loc[0, "LIMIT_BAL"] == 120_000
    # And the two scored frames differ in exactly the changed column.
    baseline_frame, scenario_frame = context.scorer.seen[-2:]
    assert baseline_frame.loc[0, "LIMIT_BAL"] == 120_000
    assert scenario_frame.loc[0, "LIMIT_BAL"] == 260_000
    assert output.baseline_probability == pytest.approx(0.18)
    assert output.scenario_probability == pytest.approx(0.04)
    assert output.delta == pytest.approx(-0.14)
    assert output.baseline_decision == "refuse"
    assert output.scenario_decision == "approve"


def test_simulation_declares_it_is_not_a_causal_claim(context):
    output = simular_escenario(
        SimulateInput(applicant=ApplicantRecord(**APPLICANT), changes={"AGE": 45}), context
    )

    assert "MODELO" in output.causal_note
    assert "causal" in output.causal_note


def test_simulation_refuses_a_column_the_model_does_not_read(context):
    with pytest.raises(ToolExecutionError) as error:
        execute(
            "simular_escenario",
            {"changes": {"UTILIZATION_M2": 0}},
            context,
            applicant=APPLICANT,
        )

    assert "UTILIZATION_M2" in str(error.value)


def test_simulation_refuses_a_value_outside_the_data_contract(context):
    with pytest.raises(ToolExecutionError) as error:
        execute("simular_escenario", {"changes": {"EDUCATION": 9}}, context, applicant=APPLICANT)

    assert "EDUCATION" in str(error.value)


def test_simulation_refuses_an_empty_scenario(context):
    with pytest.raises(ToolExecutionError):
        execute("simular_escenario", {"changes": {}}, context, applicant=APPLICANT)


# ---------------------------------------------------------------------------
# consultar_politica: the band, resolved by comparing numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.000, "A"),
        (0.059, "A"),
        (0.0599, "A"),
        (0.060, "B"),  # lower bound of B is inclusive
        (0.119, "B"),
        (0.1199, "B"),
        (0.120, "C"),  # lower bound of C is inclusive
        (0.159, "C"),
        (0.1599, "C"),
        (0.160, "D"),  # the operating threshold opens band D
        (0.299, "D"),
        (0.2999, "D"),
        (0.300, "E"),  # lower bound of E is inclusive
        (0.999, "E"),
        (1.000, "E"),
    ],
)
def test_band_edges_resolve_to_the_half_open_interval_that_contains_them(probability, expected):
    assert resolve_band(probability).code == expected


def test_band_d_opens_exactly_at_the_operating_threshold():
    band_d = next(band for band in POLICY_BANDS if band.code == "D")
    assert band_d.lower_inclusive == OPERATING_THRESHOLD


def test_the_bands_cover_the_whole_unit_interval_without_a_gap():
    assert POLICY_BANDS[0].lower_inclusive is None
    assert POLICY_BANDS[-1].upper_exclusive is None
    for lower, upper in zip(POLICY_BANDS[:-1], POLICY_BANDS[1:], strict=True):
        assert lower.upper_exclusive == upper.lower_inclusive


def test_band_boundaries_in_code_match_the_boundaries_written_in_the_corpus(corpus_fragments):
    # The band table is transcribed into POLICY_BANDS so that an inequality can be
    # evaluated, which retrieval cannot do. Two copies of one fact drift apart unless
    # something forbids it; this is what forbids it.
    text = corpus_fragments[POLICY_BANDS_CHUNK_ID].text
    boundaries = {
        value
        for band in POLICY_BANDS
        for value in (band.lower_inclusive, band.upper_exclusive)
        if value is not None
    }
    for value in sorted(boundaries):
        assert f"{value:.3f}".replace(".", ",") in text


def test_a_probability_outside_the_unit_interval_is_refused():
    with pytest.raises(ToolExecutionError):
        resolve_band(19.0)
    with pytest.raises(ToolExecutionError):
        resolve_band(-0.01)


def test_policy_query_with_a_probability_returns_the_band_and_the_fragment_backing_it(context):
    output = consultar_politica(
        PolicyInput(
            question="Me dio 0,19 en un solicitante, ¿qué hago?", probability_of_default=0.19
        ),
        context,
    )

    assert output.band is not None
    assert output.band.code == "D"
    assert output.band.probability_source == "query"
    assert output.band_fragment is not None
    assert output.band_fragment.citation
    assert output.retrieval_caveat


def test_a_probability_the_system_computed_overrides_the_one_the_model_proposed(context):
    output = execute(
        "consultar_politica",
        {"question": "¿qué hago con 0,42?", "probability_of_default": 0.42},
        context,
        applicant=APPLICANT,
        scored_probability=0.13,
    )

    assert output.band.code == "C"
    assert output.band.probability_of_default == pytest.approx(0.13)
    assert output.band.probability_source == "model"


def test_policy_query_without_a_probability_returns_no_band(context):
    output = consultar_politica(
        PolicyInput(question="¿Qué exige la norma sobre la capacidad de pago?"), context
    )

    assert output.band is None
    assert output.band_fragment is None
    assert output.fragments


def test_policy_query_can_be_restricted_to_a_document(context):
    output = execute(
        "consultar_politica",
        {"question": "¿qué dice la ley?", "document_ids": ["ley-1266-2008-habeas-data"]},
        context,
        applicant=None,
    )

    assert output.documents_searched == ("ley-1266-2008-habeas-data",)


def test_policy_query_refuses_a_document_the_corpus_does_not_have(context):
    with pytest.raises(ToolExecutionError):
        execute(
            "consultar_politica",
            {"question": "¿y esto?", "document_ids": ["decreto-inventado"]},
            context,
            applicant=None,
        )


# ---------------------------------------------------------------------------
# No fragment without a citation, in any of the three ways one can be built
# ---------------------------------------------------------------------------


def test_a_fragment_cannot_be_built_without_a_citation():
    with pytest.raises(ValidationError):
        RetrievedFragment(chunk_id="x", citation="", document_id="d", text="algo que dice la norma")


def test_a_search_hit_without_a_citation_is_refused_rather_than_used():
    hit = SearchResult(
        chunk_id="ley-1266-2008-habeas-data::013-art-13",
        text="La información de carácter positivo permanecerá de manera indefinida.",
        metadata={"document_id": "ley-1266-2008-habeas-data"},
        score=0.9,
        distance=0.1,
    )

    with pytest.raises(ToolExecutionError) as error:
        _fragment_from_search(hit)

    assert "cita" in str(error.value)


def test_every_fragment_a_policy_query_returns_carries_a_citation(context):
    output = consultar_politica(
        PolicyInput(question="¿Qué dice la ley?", probability_of_default=0.19), context
    )

    fragments = [*output.fragments, output.band_fragment]
    assert fragments
    for fragment in fragments:
        assert fragment is not None
        assert fragment.citation.strip()
    for citation in output.citations():
        assert citation.citation.strip()


def test_every_chunk_of_the_real_corpus_produces_a_fragment_with_a_citation(corpus_fragments):
    # Against the corpus as it is on disk, not a fixture that resembles it.
    assert len(corpus_fragments) == 89
    for chunk_id, fragment in corpus_fragments.items():
        assert fragment.citation.strip(), chunk_id


def test_the_band_table_is_reachable_by_identifier_in_the_real_corpus(corpus_fragments):
    # The band's citation must not depend on a search succeeding: the three numeric
    # questions of the retrieval evaluation are exactly the ones search fails on.
    fragment = corpus_fragments[POLICY_BANDS_CHUNK_ID]
    assert fragment.is_synthetic
    assert "SINTÉTICO" in fragment.integrity_notice.upper()
    assert fragment.score is None


def test_the_synthetic_notice_travels_with_the_policy_fragment(context):
    output = consultar_politica(
        PolicyInput(question="¿banda?", probability_of_default=0.19), context
    )

    assert output.band_fragment is not None
    citation = output.band_fragment.to_citation()
    assert citation.is_synthetic
    assert citation.integrity_notice


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_an_unknown_tool_is_refused_by_name(context):
    with pytest.raises(ToolExecutionError) as error:
        execute("borrar_expediente", {}, context, applicant=APPLICANT)

    assert "borrar_expediente" in str(error.value)


def test_arguments_that_break_the_contract_are_refused_before_anything_runs(context):
    with pytest.raises(ToolExecutionError):
        execute("consultar_politica", {"question": "x", "top_k": 99}, context)
    with pytest.raises(ToolExecutionError):
        execute("consultar_politica", {"probability_of_default": 0.2}, context)
    assert context.scorer.seen == []


def test_the_corpus_retriever_refuses_a_document_outside_the_corpus():
    chunks = chunk_corpus(load_corpus(settings.corpus_dir))
    retriever = CorpusRetriever(None, {chunk.chunk_id: chunk for chunk in chunks})

    assert "politica-interna-credito" in retriever.document_ids
    with pytest.raises(ToolExecutionError):
        retriever.retrieve("cualquier cosa", top_k=3, document_ids=["no-existe"])
    assert retriever.fragment_by_chunk_id("no::existe") is None
