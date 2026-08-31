"""The four tools of the copilot: strict contracts in, strict contracts out.

**The separation this module is built on.** The language model *proposes* a call; this code
*validates and executes* it. Every tool has a Pydantic model for what it accepts and a
Pydantic model for what it returns, and nothing reaches the production artefact or the index
without passing through the first one. That boundary is the whole safety argument: a model
that hallucinates an argument produces a validation error here, not a probability.

**Why the applicant is never an argument the model may fill.** The tools that score, explain
and simulate all need the applicant's 23 raw attributes, and those attributes are held in the
graph state, put there by the caller. The schemas the model sees do not contain them, and
`execute` binds them from the state. A language model asked to re-emit twenty-three integers
will eventually get one wrong, and a wrong `PAY_STATUS_1` produces a perfectly well-formed
probability about a client who does not exist. The rule *"ninguna herramienta inventa un
número"* is therefore enforced by the shape of the schema and not by the prompt.

The one place the model does supply numbers is `simular_escenario`, and that is correct - a
scenario *is* the analyst's hypothesis, arriving through the model as text. Those values go
through the same data contract the applicant does, so a scenario cannot enter through a door
the applicant could not.

**Why `consultar_politica` is not retrieval alone.** This is the first finding of
`docs/analysis/retrieval-evidence.md` turned into design. A question that carries a
probability - *"el puntaje me dio 0,19, ¿qué hago?"* - fails in **all four** chunking
strategies compared, outside the top ten, and the reason is not the chunking: a dense
retriever matches surfaces and does not evaluate whether 0,19 falls inside [0,160 ; 0,300).
No way of cutting the document repairs that. So the band is resolved here, by comparing
numbers, and retrieval is asked for the normative context around it instead of for a
deduction it cannot perform. The fragment that backs the band is fetched **by identifier from
the corpus**, not by similarity, so the citation is not contingent on a search succeeding.

**Why every fragment carries its citation as a required field.** An affirmation about a norm
without its source is the failure this whole phase exists to avoid, and a citation that can
be empty is one that will be. `RetrievedFragment.citation` has a minimum length, so a
fragment without one cannot be constructed at all.

**Where the applicant contract and the decision vocabulary live now.** `ApplicantRecord`,
the operating threshold and the sentences that travel with a probability - `COST_ASSUMPTION`,
`DECISION_CAVEAT`, `CAUSAL_NOTE`, `DIRECTION_NOTE` - are defined in `models/applicant.py` and
`models/decision.py`, and imported here. They used to be defined in this module, and moving
them changed no text and no behaviour: they are re-exported under the same names, so every
importer of `agent.tools` still reads them from here. The reason is the two-service
deployment - the service that serves the model must validate an applicant and report a
threshold without importing `anthropic`, `chromadb` or the embedding stack, all three of
which this module pulls in.

**Why the heavy objects are injected.** `ToolContext` holds the scorer, the explainer and the
retriever behind protocols. In production `build_tool_context` fills it with the pinned
registry artefact, a SHAP explainer over its forest, and the persistent Chroma index. In
`tests/test_tools.py` it is filled with hand-built doubles, which is what lets each tool be
tested in isolation, offline, without a model download and without a tracking server.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from anthropic.types import ToolParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from credit_copilot.agent.state import Citation
from credit_copilot.config import settings
from credit_copilot.explain.counterfactual import evaluate_scenario
from credit_copilot.explain.shap_service import DEFAULT_TOP_FEATURES, LocalExplanation
from credit_copilot.models.applicant import ApplicantRecord
from credit_copilot.models.decision import (
    CAUSAL_NOTE,
    COST_ASSUMPTION,
    COST_RATIO_FN_TO_FP,
    DECISION_CAVEAT,
    DIRECTION_NOTE,
    OPERATING_THRESHOLD,
    Decision,
    decide,
)
from credit_copilot.rag.chunking import Chunk
from credit_copilot.rag.vectorstore import SearchResult, VectorStore

__all__ = [
    "COST_RATIO_FN_TO_FP",
    "DEFAULT_TOP_K",
    "OPERATING_THRESHOLD",
    "POLICY_BANDS",
    "POLICY_BANDS_CHUNK_ID",
    "POLICY_DOCUMENT_ID",
    "TOOL_SPECS",
    "ApplicantRecord",
    "BandDefinition",
    "CorpusRetriever",
    "Decision",
    "ExplainArguments",
    "ExplainInput",
    "ExplainOutput",
    "FeatureContribution",
    "PolicyArguments",
    "PolicyInput",
    "PolicyOutput",
    "ResolvedBand",
    "RetrievedFragment",
    "ScoreArguments",
    "ScoreInput",
    "ScoreOutput",
    "SimulateArguments",
    "SimulateInput",
    "SimulateOutput",
    "ToolContext",
    "ToolExecutionError",
    "ToolSpec",
    "anthropic_tool_definitions",
    "build_tool_context",
    "consultar_politica",
    "execute",
    "explicar_decision",
    "resolve_band",
    "score_solicitante",
    "simular_escenario",
]

DEFAULT_TOP_K: Final[int] = 5
"""Fragments returned by a policy query by default.

Five because hit@5 is the figure the retrieval evaluation reports for the adopted strategy -
0.538 over 26 annotated questions - so the default is the number that measurement describes.
Asking for more would return fragments whose behaviour nothing here has measured.
"""

POLICY_DOCUMENT_ID: Final[str] = "politica-interna-credito"
"""Identifier of the synthetic internal policy in the corpus."""

POLICY_BANDS_CHUNK_ID: Final[str] = "politica-interna-credito::005-2-1-tabla-de-bandas"
"""The band table, addressed by identifier rather than found by similarity.

This is the fragment the three numeric questions of the retrieval evaluation needed and none
of the four strategies retrieved. Fetching it by identifier makes the citation that backs a
band decision independent of whether a search happens to succeed.
"""

RETRIEVAL_CAVEAT: Final[str] = (
    "La recuperación encuentra el artículo correcto entre los cinco primeros en algo más "
    "de la mitad de las preguntas (hit@5 = 0,538 sobre 26 preguntas anotadas a mano). Que "
    "un artículo NO aparezca aquí no prueba que la norma no lo cubra. Además, no existe un "
    "umbral de score que separe 'hay respuesta' de 'no hay respuesta': está medido que la "
    "peor pregunta sin respuesta puntúa por encima de 24 de las 26 con respuesta."
)
"""Read by the language model with every retrieval. Sections 11.3 and 11.4 of the model card."""


class ToolExecutionError(RuntimeError):
    """A tool call was refused, or failed while running.

    Carried back to the planner as the tool's result rather than raised out of the graph: a
    refusal is information the next planning cycle needs, and a stack trace is not.
    """


# ---------------------------------------------------------------------------
# Retrieved fragments and the band table
# ---------------------------------------------------------------------------


class RetrievedFragment(BaseModel):
    """One fragment of the corpus, with the citation that makes it usable.

    Attributes:
        chunk_id: Identifier of the fragment in the index.
        citation: What a reader would write to cite it. Required and non-empty: a fragment
            without a citation cannot be used to support a normative claim, so it cannot be
            constructed either.
        document_id: Source document.
        location: Heading path inside the document.
        text: The fragment as a person reads it, context header included.
        score: Cosine similarity to the query, when the fragment came from a search. `None`
            when it was fetched by identifier, where similarity is not defined.
        is_synthetic: Whether the source document was written for this project.
        integrity_notice: Warnings that change how the fragment must be read: that it is
            synthetic, that the chapter is derogated, or both.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    location: str = ""
    text: str = Field(min_length=1)
    score: float | None = None
    is_synthetic: bool = False
    integrity_notice: str = ""

    def to_citation(self) -> Citation:
        """Reduce the fragment to the record a credit file keeps.

        Returns:
            The citation, without the fragment's text.
        """
        return Citation(
            chunk_id=self.chunk_id,
            citation=self.citation,
            document_id=self.document_id,
            location=self.location,
            is_synthetic=self.is_synthetic,
            integrity_notice=self.integrity_notice,
        )


@dataclass(frozen=True)
class BandDefinition:
    """One decision band of section 2.1 of the internal credit policy.

    The interval is half-open, `[lower_inclusive, upper_exclusive)`, which is how the policy
    writes it: band D begins *at* 0,160, exactly where the operating threshold sits.

    Attributes:
        code: Band letter, A to E.
        label: The band's name in the policy.
        lower_inclusive: Lowest probability in the band. `None` for the lowest band.
        upper_exclusive: First probability outside the band. `None` for the highest band.
        decision: The primary decision the policy attaches to the band.
        authority: Who decides, per the policy's own column.
    """

    code: Literal["A", "B", "C", "D", "E"]
    label: str
    lower_inclusive: float | None
    upper_exclusive: float | None
    decision: str
    authority: str


POLICY_BANDS: Final[tuple[BandDefinition, ...]] = (
    BandDefinition(
        code="A",
        label="Riesgo bajo",
        lower_inclusive=None,
        upper_exclusive=0.060,
        decision="Aprobar con cupo y tasa estándar",
        authority="Automática, con revisión por muestreo",
    ),
    BandDefinition(
        code="B",
        label="Riesgo moderado",
        lower_inclusive=0.060,
        upper_exclusive=0.120,
        decision="Aprobar con condiciones estándar",
        authority="Analista de crédito",
    ),
    BandDefinition(
        code="C",
        label="Riesgo de vigilancia",
        lower_inclusive=0.120,
        upper_exclusive=0.160,
        decision="Aprobar con cupo reducido al 60% del solicitado y seguimiento mensual",
        authority="Analista de crédito, con concepto escrito",
    ),
    BandDefinition(
        code="D",
        label="Rechazo con excepción posible",
        lower_inclusive=0.160,
        upper_exclusive=0.300,
        decision="Rechazar, salvo excepción documentada conforme a la sección 3",
        authority="Analista de crédito para el rechazo; Comité de Crédito para la excepción",
    ),
    BandDefinition(
        code="E",
        label="Rechazo firme",
        lower_inclusive=0.300,
        upper_exclusive=None,
        decision="Rechazar",
        authority="Analista de crédito, sin facultad de excepción",
    ),
)
"""The band table, transcribed from the corpus and resolved by comparing numbers.

**Why it is in code and not only in the index.** Measured: the three questions that give a
probability and ask for the decision fail in all four chunking strategies, outside the top
ten, because a dense retriever matches surfaces and cannot evaluate an inequality. This is
the deduction the retriever cannot do.

**Why the duplication is bounded rather than accepted.** Two copies of one fact drift apart
unless something forbids it, so `tests/test_tools.py` asserts that every boundary here
appears in the text of the band-table fragment as the corpus writes it. The code answers
*which band*; the corpus fragment travels with the answer and is what gets cited.
"""


def resolve_band(probability: float) -> BandDefinition:
    """Find the band a probability of default falls into, by comparing numbers.

    Args:
        probability: Probability of default, in [0, 1].

    Returns:
        The band whose half-open interval contains it.

    Raises:
        ToolExecutionError: The probability is outside [0, 1], or - which would be a bug in
            the table rather than in the call - no band contains it.
    """
    if not 0.0 <= probability <= 1.0:
        raise ToolExecutionError(
            f"Una probabilidad de incumplimiento vale entre 0 y 1; se recibió {probability}. "
            "Un porcentaje como 19 se escribe 0,19."
        )
    for band in POLICY_BANDS:
        below_top = band.upper_exclusive is None or probability < band.upper_exclusive
        above_floor = band.lower_inclusive is None or probability >= band.lower_inclusive
        if above_floor and below_top:
            return band
    raise ToolExecutionError(
        f"Ninguna banda contiene {probability}. La tabla de bandas no cubre todo [0, 1]."
    )


class ResolvedBand(BaseModel):
    """The band a probability fell into, and where that probability came from.

    Attributes:
        code: Band letter.
        label: The band's name in the policy.
        lower_inclusive: Lowest probability in the band, or `None`.
        upper_exclusive: First probability outside the band, or `None`.
        decision: Primary decision the policy attaches to the band.
        authority: Who decides.
        probability_of_default: The probability that was resolved.
        probability_source: `model` when it came from a score this run computed, `query`
            when it came from the analyst's question. The distinction matters: only the
            first is a number this system produced.
    """

    model_config = ConfigDict(frozen=True)

    code: Literal["A", "B", "C", "D", "E"]
    label: str
    lower_inclusive: float | None
    upper_exclusive: float | None
    decision: str
    authority: str
    probability_of_default: float
    probability_source: Literal["model", "query"]


# ---------------------------------------------------------------------------
# What each tool accepts from the model, and what it accepts from the code
# ---------------------------------------------------------------------------


class ScoreArguments(BaseModel):
    """What the language model may supply to `score_solicitante`: nothing.

    The applicant comes from the graph state. A model allowed to fill in the attributes
    would eventually fill one in wrong, and the result would be a well-formed probability
    about a client who does not exist.
    """

    model_config = ConfigDict(extra="forbid")


class ExplainArguments(BaseModel):
    """What the language model may supply to `explicar_decision`.

    Attributes:
        top_n: How many features to report, ranked by absolute contribution.
    """

    model_config = ConfigDict(extra="forbid")

    top_n: int = Field(
        default=DEFAULT_TOP_FEATURES,
        ge=1,
        le=20,
        description=(
            "Cuántas variables reportar, ordenadas por contribución absoluta. La política "
            "interna exige cinco en el expediente."
        ),
    )


class SimulateArguments(BaseModel):
    """What the language model may supply to `simular_escenario`.

    Attributes:
        changes: Raw column name to the value it takes in the scenario. This is the one
            place the model legitimately supplies numbers: the scenario is the analyst's
            hypothesis, and it arrives as text.
    """

    model_config = ConfigDict(extra="forbid")

    changes: dict[str, int] = Field(
        min_length=1,
        description=(
            "Atributos crudos a modificar y su nuevo valor, por ejemplo "
            '{"LIMIT_BAL": 200000}. Solo se admiten las 23 columnas crudas del modelo; las '
            "variables derivadas las calcula el pipeline y no se pueden fijar a mano."
        ),
    )


class PolicyArguments(BaseModel):
    """What the language model may supply to `consultar_politica`.

    Attributes:
        question: The normative question, in the analyst's own words.
        probability_of_default: A probability mentioned in the query, if there is one. It is
            overridden by the code when this run has already scored the applicant.
        document_ids: Restrict the search to these corpus documents.
        top_k: How many fragments to return.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=3,
        description="La pregunta normativa, en las palabras del analista.",
    )
    probability_of_default: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Probabilidad de incumplimiento mencionada en la consulta, en [0, 1]. Solo se "
            "pasa si el usuario dio un número; si el score lo calculó una herramienta, el "
            "código lo sustituye por el valor del modelo."
        ),
    )
    document_ids: list[str] | None = Field(
        default=None,
        description=(
            "Restringir la búsqueda a estos documentos del corpus. Nulo busca en todo el corpus."
        ),
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=10,
        description="Cuántos fragmentos devolver.",
    )


class ScoreInput(BaseModel):
    """Everything `score_solicitante` needs, once the code has bound the applicant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicant: ApplicantRecord


class ExplainInput(BaseModel):
    """Everything `explicar_decision` needs, once the code has bound the applicant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicant: ApplicantRecord
    top_n: int = Field(default=DEFAULT_TOP_FEATURES, ge=1, le=20)


class SimulateInput(BaseModel):
    """Everything `simular_escenario` needs, once the code has bound the applicant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicant: ApplicantRecord
    changes: dict[str, int] = Field(min_length=1)


class PolicyInput(BaseModel):
    """Everything `consultar_politica` needs, once the code has resolved the score source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=3)
    probability_of_default: float | None = Field(default=None, ge=0.0, le=1.0)
    probability_source: Literal["model", "query"] = "query"
    document_ids: list[str] | None = None
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)


# ---------------------------------------------------------------------------
# What each tool returns
# ---------------------------------------------------------------------------


class ScoreOutput(BaseModel):
    """The probability, the decision it implies, and what makes that decision readable.

    Attributes:
        probability_of_default: The calibrated probability from the pinned artefact.
        decision: What the operating threshold recommends. A recommendation, not a verdict.
        threshold: The operating threshold applied.
        cost_assumption: The assumption the threshold is derived from. Travels with the
            number because the number alone is not interpretable.
        decision_caveat: What the policy says a band does not authorise.
        model_name: Registered model name.
        model_version: Registry version. Pinned.
    """

    model_config = ConfigDict(frozen=True)

    probability_of_default: float
    decision: Decision
    threshold: float
    cost_assumption: str
    decision_caveat: str
    model_name: str
    model_version: str


class FeatureContribution(BaseModel):
    """One feature's signed contribution to one applicant's score.

    Attributes:
        feature: Name of the transformed feature, as the pipeline declares it.
        shap_value: Signed contribution for **this** applicant. Positive pushes to default.
        magnitude: Absolute value, which is what ranks the features.
        direction: Sign of `shap_value` for this applicant, never a population sign.
        feature_value: What the feature is worth for this applicant, after preprocessing.
    """

    model_config = ConfigDict(frozen=True)

    feature: str
    shap_value: float
    magnitude: float
    direction: Literal["raises_risk", "lowers_risk"]
    feature_value: float


class ExplainOutput(BaseModel):
    """What pushed this applicant's score, and the two warnings that come with it.

    Attributes:
        probability_of_default: The calibrated probability the decision is taken on.
        forest_score: The uncalibrated forest score the contributions decompose.
        base_value: The forest's expected output. Base plus all contributions gives
            `forest_score`, not `probability_of_default`.
        top_features: The strongest contributions, descending by magnitude.
        features_considered: How many features the attribution ran over.
        direction_note: Why the direction reported is individual and not a population sign.
        causal_note: Why an attribution is not an effect.
        model_name: Registered model name.
        model_version: Registry version.
    """

    model_config = ConfigDict(frozen=True)

    probability_of_default: float
    forest_score: float
    base_value: float
    top_features: tuple[FeatureContribution, ...]
    features_considered: int
    direction_note: str
    causal_note: str
    model_name: str
    model_version: str


class SimulateOutput(BaseModel):
    """What the model would say about a modified applicant, and what that does not mean.

    Attributes:
        changes: The attributes changed and the values they were changed to.
        baseline_probability: The model's probability for the applicant as given.
        scenario_probability: The model's probability for the modified applicant.
        delta: Difference between the two model outputs. Not an estimated effect.
        baseline_decision: What the threshold recommends for the applicant as given.
        scenario_decision: What it recommends for the modified applicant.
        threshold: The operating threshold applied.
        causal_note: The sentence this output is not allowed to support.
        model_name: Registered model name.
        model_version: Registry version.
    """

    model_config = ConfigDict(frozen=True)

    changes: dict[str, int]
    baseline_probability: float
    scenario_probability: float
    delta: float
    baseline_decision: Decision
    scenario_decision: Decision
    threshold: float
    causal_note: str
    model_name: str
    model_version: str


class PolicyOutput(BaseModel):
    """The band resolved in code, the fragment that backs it, and what retrieval found.

    Attributes:
        question: The question that was searched, verbatim.
        band: The band the probability fell into, when a probability was given.
        band_fragment: The policy fragment describing the band table, fetched by identifier.
        fragments: What retrieval returned, most similar first.
        documents_searched: The documents the search was restricted to. `None` is the whole
            corpus.
        retrieval_caveat: What the measured performance of this retriever does and does not
            allow a reader to conclude from an absent article.
    """

    model_config = ConfigDict(frozen=True)

    question: str
    band: ResolvedBand | None
    band_fragment: RetrievedFragment | None
    fragments: tuple[RetrievedFragment, ...]
    documents_searched: tuple[str, ...] | None
    retrieval_caveat: str

    def citations(self) -> tuple[Citation, ...]:
        """Every source this answer rests on.

        Returns:
            The band fragment's citation, when there is one, followed by the retrieved ones.
        """
        fragments = ([self.band_fragment] if self.band_fragment else []) + list(self.fragments)
        return tuple(fragment.to_citation() for fragment in fragments)


# ---------------------------------------------------------------------------
# What the tools depend on, behind protocols so tests can supply doubles
# ---------------------------------------------------------------------------


class Scorer(Protocol):
    """Anything that can turn raw applicant rows into probabilities of default."""

    @property
    def name(self) -> str:
        """Registered model name."""

    @property
    def version(self) -> str:
        """Registry version."""

    def probability_of_default(
        self, applicants: pd.DataFrame
    ) -> npt.NDArray[np.float64]:  # pragma: no cover - structural declaration
        """Score raw applicant rows."""


class LocalExplainer(Protocol):
    """Anything that can attribute one applicant's score to its features."""

    def explain(
        self, applicant: pd.DataFrame, top_n: int
    ) -> LocalExplanation:  # pragma: no cover - structural declaration
        """Attribute the score of a single applicant."""


class Retriever(Protocol):
    """Anything that can reach the corpus, by similarity and by identifier."""

    def retrieve(
        self, query: str, top_k: int, document_ids: Sequence[str] | None
    ) -> tuple[RetrievedFragment, ...]:  # pragma: no cover - structural declaration
        """Search the corpus."""

    def fragment_by_chunk_id(
        self, chunk_id: str
    ) -> RetrievedFragment | None:  # pragma: no cover - structural declaration
        """Fetch one fragment by identifier, without searching."""


@dataclass(frozen=True)
class ToolContext:
    """The three collaborators every tool call runs against.

    Injected rather than constructed inside the tools, so a test can supply hand-built
    doubles and exercise each contract offline. Production builds it with
    `build_tool_context`.

    Attributes:
        scorer: The pinned registry artefact.
        explainer: A SHAP explainer over that artefact's forest.
        retriever: The corpus, by similarity and by identifier.
    """

    scorer: Scorer
    explainer: LocalExplainer
    retriever: Retriever


class CorpusRetriever:
    """The production retriever: Chroma for similarity, the parsed corpus for identifiers.

    **Why two access paths and not one.** Similarity is what a question needs and it is
    measurably imperfect; an identifier is what the band table needs and it is exact. Making
    the band's citation depend on a search succeeding would reintroduce, at the citation
    layer, the failure the band table was moved into code to avoid.
    """

    def __init__(self, store: VectorStore, chunks: Mapping[str, Chunk]) -> None:
        """Bind a retriever to a built index and the chunks that produced it.

        Args:
            store: A `rag.vectorstore.VectorStore` over the corpus.
            chunks: Every chunk of the corpus, keyed by identifier.
        """
        self._store = store
        self._chunks = dict(chunks)
        self._document_ids = frozenset(
            chunk.metadata.document_id for chunk in self._chunks.values()
        )

    @property
    def document_ids(self) -> frozenset[str]:
        """The documents this retriever can be restricted to.

        Returns:
            Every `document_id` present in the corpus.
        """
        return self._document_ids

    def retrieve(
        self, query: str, top_k: int, document_ids: Sequence[str] | None
    ) -> tuple[RetrievedFragment, ...]:
        """Search the corpus by similarity.

        Args:
            query: The question, as a user would write it.
            top_k: Maximum number of fragments to return.
            document_ids: Restrict to these documents. `None` searches everything.

        Returns:
            Fragments, most similar first.

        Raises:
            ToolExecutionError: A named document is not in the corpus - which Chroma would
                otherwise answer with an empty result, indistinguishable from "the corpus
                says nothing" - or the index has not been built.
        """
        if document_ids is not None:
            unknown = sorted(set(document_ids) - self._document_ids)
            if unknown:
                raise ToolExecutionError(
                    f"El corpus no contiene {unknown}. Documentos disponibles: "
                    f"{sorted(self._document_ids)}."
                )
        try:
            results = self._store.search(query=query, top_k=top_k, document_ids=document_ids)
        except LookupError as error:
            raise ToolExecutionError(f"El índice vectorial no está construido: {error}") from error
        return tuple(_fragment_from_search(result) for result in results)

    def fragment_by_chunk_id(self, chunk_id: str) -> RetrievedFragment | None:
        """Fetch one fragment by identifier, straight from the parsed corpus.

        Args:
            chunk_id: Identifier of the chunk.

        Returns:
            The fragment, or `None` when the corpus has no such chunk.
        """
        chunk = self._chunks.get(chunk_id)
        return None if chunk is None else _fragment_from_chunk(chunk)


def _fragment_from_search(result: SearchResult) -> RetrievedFragment:
    """Convert a vector-store hit into a fragment, refusing one without a citation."""
    citation = str(result.metadata.get("citation", "")).strip()
    if not citation:
        raise ToolExecutionError(
            f"El fragmento `{result.chunk_id}` llegó sin cita. Un fragmento sin cita no "
            "puede sostener una afirmación normativa, así que no se usa. Reconstruye el "
            "índice con `scripts/build_rag_index.py`."
        )
    return RetrievedFragment(
        chunk_id=result.chunk_id,
        citation=citation,
        document_id=str(result.metadata.get("document_id", "")),
        location=str(result.metadata.get("location", "")),
        text=result.text,
        score=result.score,
        is_synthetic=bool(result.metadata.get("is_synthetic", False)),
        integrity_notice=_notice_from_metadata(result.metadata),
    )


def _fragment_from_chunk(chunk: Chunk) -> RetrievedFragment:
    """Convert a corpus chunk into a fragment. No similarity is defined for it."""
    return RetrievedFragment(
        chunk_id=chunk.chunk_id,
        citation=chunk.metadata.citation,
        document_id=chunk.metadata.document_id,
        location=chunk.metadata.location,
        text=chunk.display_text,
        score=None,
        is_synthetic=chunk.metadata.is_synthetic,
        integrity_notice=chunk.integrity_notice,
    )


def _notice_from_metadata(metadata: Mapping[str, Any]) -> str:
    """Join the integrity warnings the index stores for a chunk."""
    warnings = [
        str(metadata[key]).strip()
        for key in ("synthetic_notice", "integrity_notice")
        if str(metadata.get(key, "")).strip()
    ]
    return "\n".join(warnings)


# ---------------------------------------------------------------------------
# The four tools
# ---------------------------------------------------------------------------


def score_solicitante(request: ScoreInput, context: ToolContext) -> ScoreOutput:
    """Score an applicant with the pinned production artefact.

    The threshold and the cost assumption behind it travel in the output on purpose: a
    probability of 0,19 is not interpretable without knowing that the entity refuses at
    0,160 and that 0,160 comes from a declared 5:1 cost ratio rather than from the model.

    Args:
        request: The applicant, already validated against the data contract.
        context: The collaborators; only the scorer is used.

    Returns:
        The probability, the recommendation at the operating threshold, and both caveats.

    Raises:
        ToolExecutionError: The model could not score the row.
    """
    try:
        probability = float(context.scorer.probability_of_default(request.applicant.to_frame())[0])
    except (ValueError, RuntimeError) as error:
        raise ToolExecutionError(f"No se pudo puntuar al solicitante: {error}") from error
    return ScoreOutput(
        probability_of_default=probability,
        decision=decide(probability),
        threshold=OPERATING_THRESHOLD,
        cost_assumption=COST_ASSUMPTION,
        decision_caveat=DECISION_CAVEAT,
        model_name=context.scorer.name,
        model_version=context.scorer.version,
    )


def explicar_decision(request: ExplainInput, context: ToolContext) -> ExplainOutput:
    """Attribute one applicant's score to the features that produced it.

    The direction of each feature is the sign of that feature's SHAP value **for this
    applicant**. It is not the sign of its population mean, and `DIRECTION_NOTE` travels in
    the output so the model reading it cannot substitute one for the other.

    Args:
        request: The applicant and how many features to report.
        context: The collaborators; the explainer and the scorer are used.

    Returns:
        The strongest contributions with their signs, and the two notes that bound how they
        may be read.

    Raises:
        ToolExecutionError: The explanation could not be produced in a trustworthy form.
    """
    try:
        explanation = context.explainer.explain(request.applicant.to_frame(), request.top_n)
    except (ValueError, RuntimeError) as error:
        raise ToolExecutionError(f"No se pudo explicar la decisión: {error}") from error
    return ExplainOutput(
        probability_of_default=explanation.probability_of_default,
        forest_score=explanation.forest_score,
        base_value=explanation.base_value,
        top_features=tuple(
            FeatureContribution(
                feature=effect.feature,
                shap_value=effect.shap_value,
                magnitude=effect.magnitude,
                direction=effect.direction,
                feature_value=effect.feature_value,
            )
            for effect in explanation.effects
        ),
        features_considered=explanation.features_considered,
        direction_note=DIRECTION_NOTE,
        causal_note=CAUSAL_NOTE,
        model_name=context.scorer.name,
        model_version=context.scorer.version,
    )


def simular_escenario(request: SimulateInput, context: ToolContext) -> SimulateOutput:
    """Score the applicant as given and as modified, and report the difference.

    **This is a statement about the model, not a causal claim.** It says how the model would
    evaluate an applicant carrying those attributes; it does not say what would happen if
    this client changed them. The first is verifiable by rerunning this function; the second
    is not supported by observational data that never intervened on anything. Section 4.3 of
    the internal credit policy draws the same line, and `CAUSAL_NOTE` travels in the output.

    The applicant passed in is never modified: `explain.counterfactual.apply_scenario` works
    on a copy, so the baseline stays the baseline.

    Args:
        request: The applicant and the attributes to change.
        context: The collaborators; only the scorer is used.

    Returns:
        Both probabilities, their difference, what each implies at the threshold, and the
        note bounding how the difference may be read.

    Raises:
        ToolExecutionError: The scenario names something the model does not read, proposes a
            value the data contract rejects, or could not be scored.
    """
    try:
        outcome = evaluate_scenario(context.scorer, request.applicant.to_frame(), request.changes)
    except (ValueError, RuntimeError) as error:
        raise ToolExecutionError(f"No se pudo simular el escenario: {error}") from error
    return SimulateOutput(
        changes=dict(outcome.changes),
        baseline_probability=outcome.baseline_probability,
        scenario_probability=outcome.scenario_probability,
        delta=outcome.delta,
        baseline_decision=decide(outcome.baseline_probability),
        scenario_decision=decide(outcome.scenario_probability),
        threshold=OPERATING_THRESHOLD,
        causal_note=CAUSAL_NOTE,
        model_name=context.scorer.name,
        model_version=context.scorer.version,
    )


def consultar_politica(request: PolicyInput, context: ToolContext) -> PolicyOutput:
    """Answer a normative question with retrieved fragments, and resolve a band in code.

    Two different mechanisms, kept apart because they fail differently. **The band is a
    comparison of numbers**, resolved against the transcribed table and always accompanied by
    the policy fragment that describes it, fetched by identifier. **The normative context is
    retrieval**, which is measurably imperfect and says so in `retrieval_caveat`.

    Args:
        request: The question, an optional probability with its provenance, an optional
            document restriction, and how many fragments to return.
        context: The collaborators; only the retriever is used.

    Returns:
        The band when a probability was given, the fragment that backs it, and what
        retrieval found for the question.

    Raises:
        ToolExecutionError: The probability is not in [0, 1], the band table is missing from
            the corpus, a named document does not exist, or the index is not built.
    """
    band: ResolvedBand | None = None
    band_fragment: RetrievedFragment | None = None
    if request.probability_of_default is not None:
        definition = resolve_band(request.probability_of_default)
        band = ResolvedBand(
            code=definition.code,
            label=definition.label,
            lower_inclusive=definition.lower_inclusive,
            upper_exclusive=definition.upper_exclusive,
            decision=definition.decision,
            authority=definition.authority,
            probability_of_default=request.probability_of_default,
            probability_source=request.probability_source,
        )
        band_fragment = context.retriever.fragment_by_chunk_id(POLICY_BANDS_CHUNK_ID)
        if band_fragment is None:
            raise ToolExecutionError(
                f"El corpus no contiene el fragmento `{POLICY_BANDS_CHUNK_ID}`, que es el "
                "que respalda la tabla de bandas. La banda se resolvió en código pero no "
                "hay con qué citarla, así que no se devuelve."
            )
    fragments = context.retriever.retrieve(
        query=request.question, top_k=request.top_k, document_ids=request.document_ids
    )
    return PolicyOutput(
        question=request.question,
        band=band,
        band_fragment=band_fragment,
        fragments=fragments,
        documents_searched=tuple(request.document_ids) if request.document_ids else None,
        retrieval_caveat=RETRIEVAL_CAVEAT,
    )


# ---------------------------------------------------------------------------
# The registry the graph dispatches through
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """One tool, as the language model sees it and as the code dispatches it.

    Attributes:
        name: The tool's name, which is what the model emits.
        description: What it does, in the language of the analyst using it.
        arguments_model: What the model is allowed to supply. Never the applicant.
        requires_applicant: Whether the code must bind an applicant from the state. When
            there is none, the call is refused rather than run on invented attributes.
    """

    name: str
    description: str
    arguments_model: type[BaseModel]
    requires_applicant: bool


TOOL_SPECS: Final[tuple[ToolSpec, ...]] = (
    ToolSpec(
        name="score_solicitante",
        description=(
            "Calcula la probabilidad de incumplimiento del solicitante con el modelo "
            "productivo registrado, y devuelve la decisión al umbral operativo junto con el "
            "supuesto de costos que lo sustenta. No recibe argumentos: los atributos del "
            "solicitante los aporta el sistema, nunca el modelo de lenguaje. Falla si no hay "
            "un solicitante cargado."
        ),
        arguments_model=ScoreArguments,
        requires_applicant=True,
    ),
    ToolSpec(
        name="explicar_decision",
        description=(
            "Devuelve las variables que más empujan el score DE ESTE solicitante, con su "
            "magnitud y su dirección individual (SHAP local sobre el bosque del artefacto "
            "registrado). Úsala cuando haya que justificar una decisión o llenar el "
            "expediente. Requiere un solicitante cargado."
        ),
        arguments_model=ExplainArguments,
        requires_applicant=True,
    ),
    ToolSpec(
        name="simular_escenario",
        description=(
            "Recalcula el score del solicitante cambiando atributos crudos y devuelve el "
            "delta. Responde '¿cómo evaluaría el modelo a alguien con estos atributos?', "
            "NO '¿qué pasaría si el cliente los cambiara?'. Requiere un solicitante cargado."
        ),
        arguments_model=SimulateArguments,
        requires_applicant=True,
    ),
    ToolSpec(
        name="consultar_politica",
        description=(
            "Recupera fragmentos del corpus normativo con su cita: política interna "
            "(sintética), Circular Básica Contable (capítulo derogado), Ley 1266 de 2008 y "
            "Principios de Basilea. Si la consulta involucra una probabilidad de "
            "incumplimiento, resuelve además la banda de decisión comparando rangos en "
            "código y devuelve el fragmento de la política que la describe."
        ),
        arguments_model=PolicyArguments,
        requires_applicant=False,
    ),
)
"""Every tool the planner may propose. The order is the order the model sees them in."""

_SPECS_BY_NAME: Final[Mapping[str, ToolSpec]] = {spec.name: spec for spec in TOOL_SPECS}


def anthropic_tool_definitions() -> list[ToolParam]:
    """Render the tool schemas in the shape the Messages API expects.

    The schema comes from the Pydantic model rather than being written twice: what the model
    is told it may send and what the code will accept are then the same object, and they
    cannot drift.

    Returns:
        One definition per tool, in `TOOL_SPECS` order.
    """
    return [
        ToolParam(
            name=spec.name,
            description=spec.description,
            input_schema=spec.arguments_model.model_json_schema(),
        )
        for spec in TOOL_SPECS
    ]


def execute(
    name: str,
    arguments: Mapping[str, Any],
    context: ToolContext,
    applicant: Mapping[str, int] | None = None,
    scored_probability: float | None = None,
) -> BaseModel:
    """Validate a proposed tool call and run it.

    This is the boundary between what a language model said and what this system does. The
    arguments arrive unvalidated; they are parsed into the tool's own contract, the fields
    the model was not allowed to supply are bound from the state here, and only then does
    anything reach the model artefact or the index.

    Args:
        name: Name of the tool the planner proposed.
        arguments: Arguments as the planner produced them.
        context: The collaborators the tool runs against.
        applicant: Raw attributes of the applicant under discussion, from the state.
        scored_probability: A probability this run already computed. When present it
            overrides whatever probability the planner proposed for `consultar_politica`,
            because a number the model produced is a number the model could have invented.

    Returns:
        The tool's validated output model.

    Raises:
        ToolExecutionError: The tool does not exist, the arguments do not satisfy its
            contract, the applicant is required and absent or does not satisfy the data
            contract, or the tool itself refused.
    """
    spec = _SPECS_BY_NAME.get(name)
    if spec is None:
        raise ToolExecutionError(
            f"`{name}` no es una herramienta de este copiloto. Las disponibles son: "
            f"{', '.join(_SPECS_BY_NAME)}."
        )
    try:
        parsed = spec.arguments_model.model_validate(dict(arguments))
    except ValidationError as error:
        raise ToolExecutionError(
            f"Los argumentos de `{name}` no cumplen su contrato: {error}"
        ) from error

    record = _bind_applicant(spec, applicant) if spec.requires_applicant else None

    if isinstance(parsed, ScoreArguments) and record is not None:
        return score_solicitante(ScoreInput(applicant=record), context)
    if isinstance(parsed, ExplainArguments) and record is not None:
        return explicar_decision(ExplainInput(applicant=record, top_n=parsed.top_n), context)
    if isinstance(parsed, SimulateArguments) and record is not None:
        return simular_escenario(SimulateInput(applicant=record, changes=parsed.changes), context)
    if isinstance(parsed, PolicyArguments):
        probability, source = _resolve_probability(
            parsed.probability_of_default, scored_probability
        )
        return consultar_politica(
            PolicyInput(
                question=parsed.question,
                probability_of_default=probability,
                probability_source=source,
                document_ids=parsed.document_ids,
                top_k=parsed.top_k,
            ),
            context,
        )
    raise ToolExecutionError(  # pragma: no cover - unreachable while TOOL_SPECS is complete
        f"`{name}` está declarada pero no despachada. Es un error de programación."
    )


def _bind_applicant(spec: ToolSpec, applicant: Mapping[str, int] | None) -> ApplicantRecord:
    """Turn the state's applicant into a validated record, or refuse the call."""
    if applicant is None:
        raise ToolExecutionError(
            f"`{spec.name}` necesita un solicitante y no hay ninguno cargado en esta "
            "consulta. No se inventa: cárgalo con --applicant-row o --applicant-file."
        )
    try:
        return ApplicantRecord.model_validate(dict(applicant))
    except ValidationError as error:
        raise ToolExecutionError(
            f"El solicitante no cumple el contrato de datos, así que no se puntúa: {error}"
        ) from error


def _resolve_probability(
    proposed: float | None, scored: float | None
) -> tuple[float | None, Literal["model", "query"]]:
    """Prefer a probability this system computed over one the language model produced."""
    if scored is not None:
        return scored, "model"
    if proposed is not None:
        return proposed, "query"
    return None, "query"


def build_tool_context() -> ToolContext:
    """Build the production context: the pinned artefact, its explainer, and the index.

    Every heavy object is constructed once here rather than per call - loading the registry
    artefact downloads it, and building a `TreeExplainer` walks 300 trees.

    Returns:
        A context wired to the real model and the real corpus.

    Raises:
        ModelUnavailableError: The registry version could not be loaded.
        ToolExecutionError: The corpus could not be read.
    """
    from credit_copilot.explain.shap_service import ShapLocalExplainer  # noqa: PLC0415
    from credit_copilot.models.registry import load_registered_model  # noqa: PLC0415
    from credit_copilot.rag.chunking import chunk_corpus  # noqa: PLC0415
    from credit_copilot.rag.documents import load_corpus  # noqa: PLC0415
    from credit_copilot.rag.embeddings import EmbeddingModel  # noqa: PLC0415
    from credit_copilot.rag.vectorstore import VectorStore  # noqa: PLC0415

    model = load_registered_model()
    chunks = chunk_corpus(load_corpus(settings.corpus_dir))
    store = VectorStore(settings.vector_store_dir, EmbeddingModel())
    return ToolContext(
        scorer=model,
        explainer=ShapLocalExplainer(model),
        retriever=CorpusRetriever(store, {chunk.chunk_id: chunk for chunk in chunks}),
    )
