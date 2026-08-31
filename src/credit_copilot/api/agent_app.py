"""The agent service: one query through the copilot's graph, with everything it did attached.

**Why the response is this large.** The prose is the part that reads well; the tool records
and the citations are the part that can be checked. Section 4.2 of the internal credit policy
requires a model-assisted credit file to record the model, its version, the threshold and the
top contributions - none of which live in the prose - and section 11.5 of
`docs/MODEL_CARD.md` establishes that a normative sentence counts as cited only when it
appears in the citation list, never because the surrounding answer sounds normative. A `/chat`
response that returned only `answer` would be unusable for the one job the copilot exists to
do, so the structure is the deliverable and the prose is one field of it.

**Why this module names no type from `agent/`.** Not for the import-weight reason - this
service ships the whole agent stack - but for a testing one. The endpoint converts whatever
the runner returned into the API's own contracts through a tolerant reader that accepts a
Pydantic object or a plain mapping, so `tests/test_api.py` drives `/chat` with a hand-built
run and no API key, no vector index and no model download. The conversion is also the place
where a citation without a fragment identifier is refused: `CitationOut.chunk_id` has a
minimum length, so such a citation cannot be constructed at all.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Annotated, Any, Final

from fastapi import Depends, FastAPI

from credit_copilot import __version__
from credit_copilot.api import dependencies
from credit_copilot.api.dependencies import AgentServiceState, AppState, get_agent_state
from credit_copilot.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationOut,
    DecisionContext,
    ErrorResponse,
    HealthResponse,
    TokenUsageOut,
    ToolInvocation,
)

__all__ = ["app", "create_app"]

AgentState = Annotated[AgentServiceState, Depends(get_agent_state)]
"""The loaded copilot, injected."""

SERVICE_NAME: Final[str] = "agent"
"""Name this application logs under and reports in `/health`."""

_OUTCOME_MEANINGS: Final[dict[str, str]] = {
    "answered": "El evaluador juzgó que la evidencia reunida alcanzaba para responder con citas.",
    "answered_without_tools": (
        "El planificador no pidió ninguna herramienta, así que la respuesta no se apoya en "
        "ningún resultado de herramienta ni en ningún fragmento del corpus."
    ),
    "answered_with_gaps": (
        "Se agotó el tope de ciclos de replanificación con el evaluador todavía insatisfecho. "
        "La respuesta debe decir qué no pudo establecer; `unresolved_gap` dice qué faltaba."
    ),
}
"""What each end state means, carried in the payload so the code is not the documentation."""

_READING_NOTES: Final[dict[str, str]] = {
    "que_es_una_cita": (
        "Solo lo que aparece en `citations` está respaldado por un fragmento del corpus. Una "
        "frase normativa dentro de `answer` no queda citada porque la respuesta cite en otra "
        "parte: está medido que el copiloto emite texto normativo procedente de constantes "
        "del propio código, sin fragmento detrás (sección 11.4 ter de docs/MODEL_CARD.md)."
    ),
    "ausencia_de_cita": (
        "Que no aparezca un artículo NO prueba que la norma no lo cubra. La recuperación "
        "encuentra el artículo correcto entre los cinco primeros en algo más de la mitad de "
        "las preguntas anotadas (hit@5 = 0,538 sobre 26)."
    ),
    "politica_sintetica": (
        "El documento `politica-interna-credito` del corpus es SINTÉTICO: se redactó para "
        "este proyecto y no representa la política de ninguna entidad real. Toda cita suya "
        "llega con `is_synthetic: true` y con su aviso en `integrity_notice`."
    ),
    "capitulo_derogado": (
        "El documento `circular-basica-contable-sfc-cap-ii` está DEROGADO desde el 1 de junio "
        "de 2023. Se conserva por su valor de referencia y no describe la norma vigente; sus "
        "fragmentos lo declaran en `integrity_notice`."
    ),
    "el_copiloto_no_decide": (
        "El copiloto aporta evidencia trazable para que decida una persona. Ninguna banda "
        "autoriza un rechazo automático, y en el umbral 0,160 aproximadamente 6 de cada 10 "
        "solicitantes rechazados habrían pagado."
    ),
    "herramientas_fallidas": (
        "Una entrada de `tools_invoked` con `ok: false` es información, no un error del "
        "servicio: la herramienta se negó a correr (por ejemplo, porque la consulta no traía "
        "solicitante) y esa negativa es parte de por qué la respuesta es la que es."
    ),
    "costo": (
        "`llm_calls` cuenta las llamadas al modelo de lenguaje de esta consulta. El techo es "
        "2·max_iterations + 1: planificar y evaluar en cada ciclo, más la síntesis."
    ),
}
"""What a reader needs to read this response correctly, carried inside the response itself.

The two documents are named by their `document_id` and not by their title. A citation carries
the corpus's `citation_prefix`, which for the synthetic policy is *Política Interna de
Crédito* while its title is *Política Interna de Otorgamiento de Crédito de Consumo
Rotativo* - so a note written with either name would fail to match the citations beside it.
The identifier is the one string that appears in both. `tests/test_api.py` checks that both
identifiers are documents the corpus actually holds.
"""


def _as_mapping(item: Any) -> Mapping[str, Any]:
    """Read one record of the graph's final state as a mapping, whatever shape it arrived in.

    Args:
        item: A Pydantic model from `agent.state`, or a plain mapping from a test double.

    Returns:
        The record's fields. Reading structurally rather than importing the agent's classes is
        what lets `/chat` be exercised without the agent stack.
    """
    if isinstance(item, Mapping):
        return dict(item)
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dict(dump(mode="json"))
    return dict(vars(item))


def _tool_invocations(records: Any) -> tuple[ToolInvocation, ...]:
    """Convert the run's tool history into the API contract, refusals included."""
    return tuple(
        ToolInvocation(
            call_id=str(fields.get("call_id", "")),
            name=str(fields.get("name", "")),
            arguments=dict(fields.get("arguments") or {}),
            ok=bool(fields.get("ok", False)),
            result=fields.get("result"),
            error=fields.get("error"),
        )
        for fields in (_as_mapping(record) for record in records or ())
    )


def _citations(citations: Any) -> tuple[CitationOut, ...]:
    """Convert the run's citations, dropping repeats and refusing one without an identifier.

    Deduplication happens here rather than in the graph, which accumulates every citation each
    tool call produced - the same fragment is legitimately retrieved twice. First-seen order is
    kept, so the reader meets the sources in the order the copilot found them.

    Args:
        citations: The citations the run accumulated.

    Returns:
        The distinct citations. A citation whose `chunk_id` is empty raises rather than being
        emitted: a source a reader cannot resolve back to a fragment is not a source.
    """
    seen: set[str] = set()
    unique: list[CitationOut] = []
    for fields in (_as_mapping(citation) for citation in citations or ()):
        chunk_id = str(fields.get("chunk_id", ""))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        notice = str(fields.get("integrity_notice", "") or "")
        unique.append(
            CitationOut(
                chunk_id=chunk_id,
                citation=str(fields.get("citation", "")),
                document_id=str(fields.get("document_id", "")),
                location=str(fields.get("location", "") or ""),
                is_synthetic=bool(fields.get("is_synthetic", False)),
                integrity_notice=notice,
                has_integrity_notice=bool(notice.strip()),
            )
        )
    return tuple(unique)


def _token_usage(usage: Any) -> tuple[TokenUsageOut, ...]:
    """Convert the per-call token counts, which are reported per call and never summed."""
    return tuple(
        TokenUsageOut(
            node=str(fields.get("node", "")),
            model=str(fields.get("model", "")),
            input_tokens=int(fields.get("input_tokens", 0) or 0),
            output_tokens=int(fields.get("output_tokens", 0) or 0),
            cache_read_tokens=int(fields.get("cache_read_tokens", 0) or 0),
        )
        for fields in (_as_mapping(item) for item in usage or ())
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the copilot's tool context and client once, and never crash the process.

    Args:
        app: The application starting up.

    Yields:
        Control back to the server once the load has been attempted. A missing API key, an
        unreachable registry or an unbuilt index leave the service running and degraded, with
        `/health` naming the reason and `/chat` answering 503. An application built with an
        explicit state loads nothing, which is how the tests run without an API key.
    """
    if app.state.autoload:
        app.state.services.agent = dependencies.load_agent_service()
    yield


def create_app(state: AppState | None = None) -> FastAPI:
    """Build the agent application.

    Args:
        state: A pre-built holder, which is how a test supplies a stub runner. Supplying
            one also turns the start-up load off.

    Returns:
        A configured FastAPI application. Nothing is loaded at import time.
    """
    autoload = state is None
    state = state if state is not None else AppState()
    app = FastAPI(
        lifespan=_lifespan,
        title="Credit Risk — servicio del copiloto",
        version=__version__,
        summary=(
            "Una consulta de analista a través del grafo del copiloto, con las herramientas "
            "que invocó, las citas en que se apoya y lo que costó."
        ),
        description=__doc__,
        responses={
            422: {"model": ErrorResponse, "description": "La petición no cumple el contrato."},
            503: {"model": ErrorResponse, "description": "El copiloto no está cargado."},
        },
    )
    app.state.services = state
    app.state.autoload = autoload
    dependencies.install_request_context(app, SERVICE_NAME)
    dependencies.install_error_handlers(app, SERVICE_NAME)
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    """Attach the two endpoints. Split out only to keep `create_app` readable."""

    @app.post("/chat", response_model=ChatResponse, tags=["copilot"])
    def chat(request: ChatRequest, state: AgentState) -> ChatResponse:
        """Run one analyst query through the copilot and return everything it did.

        The applicant, when supplied, goes through the same data contract `/predict` uses, so
        a query cannot reach the tools with an applicant the scoring endpoint would refuse.
        The language model never receives those 23 attributes as arguments it may fill: the
        code binds them from this request, which is why a hallucinated attribute cannot become
        a well-formed probability about a client who does not exist.

        Args:
            request: The question, optionally the applicant, optionally the cycle cap.
            state: The loaded copilot.

        Returns:
            The answer, the tools invoked with their arguments and results, the citations with
            their fragment identifiers and integrity marks, the call count and the cycles used.
        """
        runner = state.require()
        applicant = request.applicant.model_dump() if request.applicant is not None else None
        max_iterations = request.max_iterations or state.default_max_iterations
        final = runner.run(request.query, applicant, max_iterations)
        outcome = str(final.get("outcome", "answered"))
        return ChatResponse(
            request_id=dependencies.current_request_id(),
            query=request.query,
            answer=str(final.get("answer", "")),
            outcome=outcome,
            outcome_meaning=_OUTCOME_MEANINGS.get(
                outcome, "Estado final no declarado por el grafo."
            ),
            unresolved_gap=str(final.get("gap", "") or ""),
            tools_invoked=_tool_invocations(final.get("tool_records")),
            citations=_citations(final.get("citations")),
            llm_calls=int(final.get("llm_calls", 0) or 0),
            iterations=int(final.get("iterations", 0) or 0),
            max_iterations=max_iterations,
            token_usage=_token_usage(final.get("token_usage")),
            decision_context=DecisionContext.current(),
            reading_notes=dict(_READING_NOTES),
        )

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health(state: AgentState) -> HealthResponse:
        """Report whether the copilot is up and whether it can actually answer.

        Args:
            state: The agent service's state, loaded or not.

        Returns:
            The service's readiness, and the reason it is not ready when it is not. A copilot
            that could not build its tool context is running and cannot answer, and those are
            two different facts.
        """
        return HealthResponse(
            service="agent",
            status="ok" if state.is_ready else "degraded",
            version=__version__,
            model_loaded=state.model_loaded,
            detail=state.error,
            request_id=dependencies.current_request_id(),
        )


app = create_app()
"""The application `uvicorn credit_copilot.api.agent_app:app` serves."""
