"""Artefact lifecycle, correlation identifiers, structured logging and error translation.

**Why the artefact loads once, at start-up.** Loading a registry version downloads it, and
building a `TreeExplainer` walks 300 trees. Doing either per request would put a network
round trip and a few hundred milliseconds of tree walking in front of every probability, and
- worse - would make two requests capable of scoring against two different downloads. One
load, at start-up, held in `app.state`, is what makes the number this service returns the
number `docs/MODEL_CARD.md` describes.

**What happens when the registry is not available at start-up, and why the process does not
die.** `load_model_service` never raises. If MLflow is unreachable, unconfigured or has no
such version, it returns a state carrying the reason and no artefact. The application still
starts, `/health` answers `degraded` and says `model_loaded: false` with the reason, and every
endpoint that needs the artefact answers **503 Service Unavailable** naming the same reason.

Refusing to start would be the other defensible choice and it is worse here for two reasons.
A container that exits at boot reports its failure only in the orchestrator's logs, whereas a
process that stays up answers the question *"why is scoring down?"* to anyone who can reach
`/health` - including the load balancer and the next turn's `docker compose`. And the failure
is frequently transient: a tracking server that is slow to come up would turn into a crash
loop, while this shape recovers by itself the moment a restart succeeds. **What it must never
do is degrade quietly**, which is why no endpoint falls back to a rebuilt pipeline: a
refitted forest is a different object from the one phase 2 measured, and answering with it
would be answering with a number no document describes.

**Why the agent's imports happen inside a function.** `model_app` imports this module. If the
graph, the tools or the vector store were imported at module level, `anthropic`, `langgraph`,
`chromadb` and `torch` would be in the model service's process - the exact coupling the
two-service split exists to remove. `load_agent_service` therefore imports them locally, and
`AgentRunner` is a protocol declared here so that no annotation in this module names an agent
type either.

**What never reaches a log.** Credentials, and applicant attributes. The first because
`config.settings` holds them and nothing here reads it; any MLflow URI that leaves this module
goes through `tracking.redact_uri` first. The second because the 23 raw attributes are the
subject matter, not diagnostics: a log line is the wrong place for a client's payment history,
and a correlation identifier is enough to tie a request to its outcome without copying it.
"""

import json
import logging
import re
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final, Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from credit_copilot.api.schemas import ErrorBody, ErrorResponse
from credit_copilot.explain.counterfactual import ScenarioError
from credit_copilot.explain.shap_service import ExplanationError, ShapLocalExplainer
from credit_copilot.models.registry import (
    MissingColumnsError,
    ModelUnavailableError,
    RegisteredModel,
    UnknownValueError,
    load_registered_model,
)

__all__ = [
    "REQUEST_ID_HEADER",
    "AgentRunner",
    "AgentServiceState",
    "AppState",
    "ModelService",
    "ModelServiceState",
    "ValidationMetrics",
    "app_state",
    "configure_logging",
    "current_request_id",
    "get_agent_state",
    "get_model_state",
    "install_error_handlers",
    "install_request_context",
    "load_agent_service",
    "load_model_service",
]

REQUEST_ID_HEADER: Final[str] = "X-Request-ID"
"""Header the correlation identifier arrives in and leaves in."""

_REQUEST_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
"""What an incoming correlation identifier may contain.

A caller-supplied identifier is echoed into every log line this request produces, so it is
untrusted input that reaches a log. Anything outside this alphabet - a newline above all -
is discarded and replaced with a generated one, because a client that can write newlines into
a log file can write a whole fake log line into it.
"""

_request_id: ContextVar[str] = ContextVar("request_id", default="")
"""The identifier of the request being served, readable from anywhere in the call stack."""

_LOCATION_MARKERS: Final[frozenset[str]] = frozenset({"body", "query", "path", "header", "cookie"})
"""Leading elements of a Pydantic error location that name where in the request it was."""

_LOGGER_NAME: Final[str] = "credit_copilot.api"
"""Root of the logging tree both applications write to."""


def current_request_id() -> str:
    """Read the correlation identifier of the request being served.

    Returns:
        The identifier, or the empty string outside a request.
    """
    return _request_id.get()


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """One JSON object per line, with the correlation identifier on every one of them.

    JSON rather than a formatted string because these lines are read by a log collector
    before they are read by a person, and a regular expression over a human-readable line is
    a parser that breaks when somebody improves the wording.
    """

    def __init__(self, service: str) -> None:
        """Bind the formatter to the application that emits the lines.

        Args:
            service: `model` or `agent`. Both write to stdout, and in the next turn both
                write to the same collector, so every line has to say which one it came from.
        """
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        """Render one record as a JSON object.

        Args:
            record: The record to render.

        Returns:
            A single line of JSON. Extra fields travel in `record.context`, which keeps them
            out of the message text and therefore machine-readable.
        """
        payload: dict[str, Any] = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "level": record.levelname,
            "service": self._service,
            "logger": record.name,
            "request_id": _request_id.get(),
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if isinstance(context, Mapping):
            payload.update(context)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(service: str, level: int = logging.INFO) -> logging.Logger:
    """Point this application's logger at stdout with the JSON formatter.

    Idempotent: the handler is replaced rather than appended, so importing or creating an
    application twice - which the tests do - does not double every line.

    Args:
        service: `model` or `agent`.
        level: Lowest level to emit.

    Returns:
        The application's logger.
    """
    logger = logging.getLogger(_LOGGER_NAME).getChild(service)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(_JsonFormatter(service))
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False
    return logger


def install_request_context(app: FastAPI, service: str) -> None:
    """Give every request a correlation identifier, and log its start and its end.

    The identifier is taken from the request header when the caller supplied a well-formed
    one, so a trace that starts in another system keeps its name, and generated otherwise. It
    is bound to a context variable for the duration of the request, echoed back in the
    response header, and carried in every response body.

    **The applicant's attributes are deliberately not logged.** The log records what was
    asked of the service and how it ended; the subject matter of the request is not
    diagnostics.

    Args:
        app: The application to instrument.
        service: `model` or `agent`, used to name the logger.
    """
    logger = configure_logging(service)

    @app.middleware("http")
    async def _correlate(request: Request, call_next: Callable[[Request], Any]) -> Any:
        supplied = request.headers.get(REQUEST_ID_HEADER, "")
        identifier = supplied if _REQUEST_ID_PATTERN.match(supplied) else uuid.uuid4().hex
        token = _request_id.set(identifier)
        started = time.perf_counter()
        logger.info(
            "request.started",
            extra={"context": {"method": request.method, "path": request.url.path}},
        )
        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = identifier
            logger.info(
                "request.completed",
                extra={
                    "context": {
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    }
                },
            )
            return response
        finally:
            # Reset last: the closing log line is part of the request and has to carry its
            # identifier. Resetting before it would emit the one line an operator searches
            # for - the one with the status code - without anything to search it by.
            _request_id.reset(token)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _fields_of(error: RequestValidationError | ValidationError) -> tuple[str, ...]:
    """Name the request fields a validation failure is about.

    Args:
        error: The failure raised while parsing the request body.

    Returns:
        The offending field names, deduplicated, in the order they were reported. Only the
        **leading** location marker - `body`, `query`, `path`, `header`, `cookie` - is
        dropped, and only there: a field of the request may legitimately be called `query`,
        and filtering that word everywhere would report the copilot's own question field as
        `body`. What the caller reads is `PAY_AMT3`, not a path they have to interpret.
    """
    names: list[str] = []
    for item in error.errors():
        location = [str(part) for part in item.get("loc", ()) if isinstance(part, str)]
        if location and location[0] in _LOCATION_MARKERS:
            location = location[1:]
        name = location[-1] if location else "body"
        if name not in names:
            names.append(name)
    return tuple(names)


def _reasons_of(error: RequestValidationError | ValidationError) -> str:
    """Say what was wrong with each offending field, without echoing the request back.

    Args:
        error: The failure raised while parsing the request body.

    Returns:
        One `field: reason` clause per problem. Pydantic's own rendering carries an `input`
        block holding the whole payload, and repeating a client's 23 applicant attributes back
        into an error body - which is then quoted in tickets and pasted into chats - copies the
        subject matter of the request into places that only needed to know what was wrong with
        it. The field and the reason are what a caller can act on; the values they already
        have.
    """
    clauses: list[str] = []
    for item in error.errors():
        location = [str(part) for part in item.get("loc", ()) if isinstance(part, str)]
        if location and location[0] in _LOCATION_MARKERS:
            location = location[1:]
        name = ".".join(location) or "body"
        clause = f"{name}: {item.get('msg', 'inválido')}"
        if clause not in clauses:
            clauses.append(clause)
    return " | ".join(clauses)


def _error_response(
    status_code: int, error_type: str, message: str, fields: tuple[str, ...] = ()
) -> JSONResponse:
    """Render one failure in the single envelope both applications use."""
    body = ErrorResponse(
        error=ErrorBody(
            type=error_type,
            message=message,
            fields=fields,
            request_id=_request_id.get(),
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers={REQUEST_ID_HEADER: _request_id.get()},
    )


def install_error_handlers(app: FastAPI, service: str) -> None:
    """Translate every failure this API can produce into one envelope.

    The mapping is the contract, so it is written in one place rather than at each endpoint:

    - a request that does not satisfy a contract - a missing field, an explicit null, a
      value outside the plausible range, a category no ADR accepts - is **422**, and the
      response names the field. **Never a probability.**
    - a request the artefact cannot serve because it is not loaded is **503**, with the
      reason the load failed.
    - anything unforeseen is **500** with no internals in the body, and the full traceback in
      the log under the same correlation identifier.

    Args:
        app: The application to install the handlers on.
        service: `model` or `agent`, used to name the logger.
    """
    logger = configure_logging(service)

    @app.exception_handler(RequestValidationError)
    async def _on_request_validation(_: Request, error: RequestValidationError) -> JSONResponse:
        fields = _fields_of(error)
        logger.warning(
            "request.rejected",
            extra={"context": {"reason": "schema_violation", "fields": list(fields)}},
        )
        return _error_response(
            422,
            "invalid_request",
            "La petición no cumple el contrato de entrada. No se imputa ningún valor: un "
            "campo ausente o nulo significa 'no se sabe', y escribir un cero en él lo "
            "convertiría en un hecho de negocio falso. Campos con problema: "
            f"{', '.join(fields) if fields else 'cuerpo de la petición'}. "
            f"Detalle: {_reasons_of(error)}",
            fields,
        )

    @app.exception_handler(ValidationError)
    async def _on_validation(_: Request, error: ValidationError) -> JSONResponse:
        fields = _fields_of(error)
        logger.warning(
            "request.rejected",
            extra={"context": {"reason": "contract_violation", "fields": list(fields)}},
        )
        return _error_response(422, "invalid_request", _reasons_of(error), fields)

    @app.exception_handler(UnknownValueError)
    async def _on_unknown_value(_: Request, error: UnknownValueError) -> JSONResponse:
        logger.warning("request.rejected", extra={"context": {"reason": "unknown_value"}})
        return _error_response(422, "unknown_value", str(error))

    @app.exception_handler(MissingColumnsError)
    async def _on_missing_columns(_: Request, error: MissingColumnsError) -> JSONResponse:
        logger.warning("request.rejected", extra={"context": {"reason": "missing_columns"}})
        return _error_response(422, "missing_columns", str(error))

    @app.exception_handler(ScenarioError)
    async def _on_scenario(_: Request, error: ScenarioError) -> JSONResponse:
        logger.warning("request.rejected", extra={"context": {"reason": "invalid_scenario"}})
        return _error_response(422, "invalid_scenario", str(error), ("changes",))

    @app.exception_handler(ExplanationError)
    async def _on_explanation(_: Request, error: ExplanationError) -> JSONResponse:
        logger.error("request.failed", extra={"context": {"reason": "explanation_failed"}})
        return _error_response(422, "explanation_unavailable", str(error))

    @app.exception_handler(ModelUnavailableError)
    async def _on_model_unavailable(_: Request, error: ModelUnavailableError) -> JSONResponse:
        logger.error("request.failed", extra={"context": {"reason": "model_unavailable"}})
        return _error_response(503, "model_unavailable", str(error))

    @app.exception_handler(StarletteHTTPException)
    async def _on_http(_: Request, error: StarletteHTTPException) -> JSONResponse:
        return _error_response(error.status_code, "http_error", str(error.detail))

    @app.exception_handler(Exception)
    async def _on_unexpected(_: Request, error: Exception) -> JSONResponse:
        logger.exception("request.failed", extra={"context": {"reason": type(error).__name__}})
        return _error_response(
            500,
            "internal_error",
            "La petición falló por un error no previsto. El identificador de correlación de "
            "esta respuesta aparece en el log del servicio con el detalle completo.",
        )


# ---------------------------------------------------------------------------
# The model service
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationMetrics:
    """The cross-validated metrics of the registered artefact, read from its MLflow run.

    Read rather than restated. `scripts/register_production_model.py` attaches the figures of
    entry 006 of `docs/EVALUATION.md` to the run that produced the artefact, baselines
    included, precisely so that a consumer does not have to keep a second copy of them. A
    copy in this file would be a number that can disagree with the registry and never say so.

    Attributes:
        run_id: The run the metrics came from.
        metrics: Metric name to value, exactly as the run records them.
        error: Why the metrics could not be read. `None` when they were.
    """

    run_id: str | None
    metrics: Mapping[str, float]
    error: str | None = None


@dataclass(frozen=True)
class ModelService:
    """Everything the model application needs in memory, loaded once.

    Attributes:
        model: The pinned registry artefact: preprocessor plus calibrated forest.
        explainer: A SHAP explainer over that artefact's forest, built once.
        validation: The metrics attached to the run that registered the artefact.
        tracking_uri: The MLflow server it came from, already redacted.
    """

    model: RegisteredModel
    explainer: ShapLocalExplainer
    validation: ValidationMetrics
    tracking_uri: str | None = None


@dataclass(frozen=True)
class ModelServiceState:
    """The outcome of trying to load the model service. Never an exception.

    Attributes:
        service: The loaded artefacts, or `None` when the load failed.
        error: Why it failed. `None` when it did not.
    """

    service: ModelService | None = None
    error: str | None = None

    @property
    def is_ready(self) -> bool:
        """Whether the endpoints that need the artefact can be served.

        Returns:
            `True` when the artefact is in memory.
        """
        return self.service is not None

    def require(self) -> ModelService:
        """Return the loaded service, or refuse the request.

        Returns:
            The loaded artefacts.

        Raises:
            ModelUnavailableError: The artefact is not loaded. The message carries the
                reason the load failed, so a caller does not have to read the service's log
                to find out why scoring is down.
        """
        if self.service is None:
            raise ModelUnavailableError(
                "El modelo productivo no está cargado, así que este servicio no puede "
                f"devolver una probabilidad. Motivo del fallo al arrancar: {self.error}"
            )
        return self.service


def _read_validation_metrics(model: RegisteredModel) -> ValidationMetrics:
    """Read the metrics attached to the run that registered this artefact.

    Args:
        model: The loaded artefact, with its registry coordinates.

    Returns:
        The run's metrics, or an empty set with the reason it could not be read. A failure
        here is not a failure of the service: the artefact still scores, and `/model-info`
        reports that the figures are unavailable rather than inventing them.
    """
    import mlflow  # noqa: PLC0415 - kept local so a metrics failure cannot break the import

    try:
        client = mlflow.MlflowClient()
        version = client.get_model_version(model.name, model.version)
        run = client.get_run(str(version.run_id))
    except Exception as error:  # noqa: BLE001 - the registry raises many unrelated types
        return ValidationMetrics(
            run_id=None,
            metrics={},
            error=f"{type(error).__name__}: {error}",
        )
    return ValidationMetrics(
        run_id=str(version.run_id),
        metrics={str(key): float(value) for key, value in run.data.metrics.items()},
    )


def load_model_service() -> ModelServiceState:
    """Load the pinned artefact and its explainer. Never raises.

    Returns:
        A state carrying the loaded artefacts, or the reason they could not be loaded. See
        the module docstring for why a registry that is unreachable at start-up degrades the
        service instead of stopping the process.
    """
    logger = configure_logging("model")
    try:
        model = load_registered_model()
        explainer = ShapLocalExplainer(model)
    except ModelUnavailableError as error:
        logger.error("model.load_failed", extra={"context": {"reason": str(error)}})
        return ModelServiceState(service=None, error=str(error))
    except Exception as error:  # noqa: BLE001 - loading touches network, disk and pickles
        reason = f"{type(error).__name__}: {error}"
        logger.error("model.load_failed", extra={"context": {"reason": reason}})
        return ModelServiceState(service=None, error=reason)

    import mlflow  # noqa: PLC0415 - only needed to report where the artefact came from

    from credit_copilot.models.tracking import redact_uri  # noqa: PLC0415 - import kept local

    tracking_uri = redact_uri(str(mlflow.get_tracking_uri() or ""))
    validation = _read_validation_metrics(model)
    logger.info(
        "model.loaded",
        extra={
            "context": {
                "model_uri": model.uri,
                "run_id": validation.run_id,
                "metrics_read": len(validation.metrics),
            }
        },
    )
    return ModelServiceState(
        service=ModelService(
            model=model,
            explainer=explainer,
            validation=validation,
            tracking_uri=tracking_uri or None,
        )
    )


# ---------------------------------------------------------------------------
# The agent service
# ---------------------------------------------------------------------------


class AgentRunner(Protocol):
    """Anything that can run one analyst query through the copilot.

    Declared as a protocol so that this module - which `model_app` imports - never names a
    type from `agent/`, and so that `tests/test_api.py` can drive `/chat` with a hand-built
    runner and no API key, no index and no model download.
    """

    def run(
        self, query: str, applicant: Mapping[str, int] | None, max_iterations: int
    ) -> dict[str, Any]:  # pragma: no cover - structural declaration
        """Run one query and return the graph's final state."""


class _GraphRunner:
    """The production runner: the compiled graph, over the real tools and the real corpus.

    The graph is rebuilt per query, which is deliberate and cheap - `build_graph` wires
    nodes and compiles, and touches neither the network nor the artefact. What is expensive
    is the `ToolContext` and the client, and those are built once and held here.
    """

    def __init__(self, context: Any, client: Any) -> None:
        """Bind the runner to the collaborators built at start-up.

        Args:
            context: The `agent.tools.ToolContext` the tools run against.
            client: The Anthropic client the graph's three nodes call.
        """
        self._context = context
        self._client = client

    def run(
        self, query: str, applicant: Mapping[str, int] | None, max_iterations: int
    ) -> dict[str, Any]:
        """Run one query through the graph.

        Args:
            query: The analyst's question.
            applicant: The applicant under discussion, or `None`.
            max_iterations: Re-planning cycles allowed.

        Returns:
            The graph's final state: the answer, every tool record, every citation, the call
            count and the token usage.
        """
        from credit_copilot.agent.graph import run_query  # noqa: PLC0415 - see module docstring

        return dict(
            run_query(
                query,
                self._context,
                applicant=applicant,
                client=self._client,
                max_iterations=max_iterations,
            )
        )


@dataclass(frozen=True)
class AgentServiceState:
    """The outcome of trying to load the agent service. Never an exception.

    Attributes:
        runner: The loaded runner, or `None` when the load failed.
        error: Why it failed. `None` when it did not.
        default_max_iterations: The re-planning cap a request that does not name one runs
            under. Reported in every `/chat` response, because a run that ended with gaps is
            only interpretable next to the cap it ran under.
        model_loaded: Whether the pinned artefact behind the tools is in memory.
    """

    runner: AgentRunner | None = None
    error: str | None = None
    default_max_iterations: int = 3
    model_loaded: bool = False

    @property
    def is_ready(self) -> bool:
        """Whether `/chat` can be served.

        Returns:
            `True` when the runner is loaded.
        """
        return self.runner is not None

    def require(self) -> AgentRunner:
        """Return the loaded runner, or refuse the request.

        Returns:
            The runner.

        Raises:
            ModelUnavailableError: The copilot is not loaded. Translated to 503 by the
                handler installed in `install_error_handlers`.
        """
        if self.runner is None:
            raise ModelUnavailableError(
                "El copiloto no está cargado, así que este servicio no puede responder una "
                f"consulta. Motivo del fallo al arrancar: {self.error}"
            )
        return self.runner


def load_agent_service() -> AgentServiceState:
    """Build the copilot's tool context and language-model client. Never raises.

    Every import of the agent stack happens inside this function; see the module docstring
    for why. The failure modes it absorbs are the three that stop the copilot from existing:
    no API key, no registry version, and no vector index.

    Returns:
        A state carrying the runner, or the reason it could not be built.
    """
    logger = configure_logging("agent")
    from credit_copilot.agent.state import DEFAULT_MAX_ITERATIONS  # noqa: PLC0415

    try:
        from credit_copilot.agent.graph import build_client  # noqa: PLC0415
        from credit_copilot.agent.tools import build_tool_context  # noqa: PLC0415

        context = build_tool_context()
        client = build_client()
    except Exception as error:  # noqa: BLE001 - key, registry and index fail in unrelated ways
        reason = f"{type(error).__name__}: {error}"
        logger.error("agent.load_failed", extra={"context": {"reason": reason}})
        return AgentServiceState(
            runner=None, error=reason, default_max_iterations=DEFAULT_MAX_ITERATIONS
        )

    logger.info("agent.loaded", extra={"context": {"max_iterations": DEFAULT_MAX_ITERATIONS}})
    return AgentServiceState(
        runner=_GraphRunner(context, client),
        default_max_iterations=DEFAULT_MAX_ITERATIONS,
        model_loaded=True,
    )


# ---------------------------------------------------------------------------
# Reaching the loaded state from an endpoint
# ---------------------------------------------------------------------------


@dataclass
class AppState:
    """Mutable holder the lifespan writes into and the dependencies read from.

    Mutable, and the only mutable thing in this module: the application object is built
    before the artefacts are loaded, so something has to carry the result of a load that
    happens after construction. Both services are declared here rather than one per
    application, so that `get_model_state` works identically in both - the agent service
    reports whether the artefact behind its tools is loaded too.
    """

    model: ModelServiceState = field(default_factory=ModelServiceState)
    agent: AgentServiceState = field(default_factory=AgentServiceState)


def app_state(request: Request) -> AppState:
    """Reach the state the lifespan loaded.

    Args:
        request: The request being served.

    Returns:
        The holder attached to the application at start-up.
    """
    holder = getattr(request.app.state, "services", None)
    if not isinstance(holder, AppState):  # pragma: no cover - set by every create_app
        holder = AppState()
        request.app.state.services = holder
    return holder


def get_model_state(request: Request) -> ModelServiceState:
    """Read the model service's state without requiring it to be loaded.

    Used by `/health`, which has to be able to answer precisely when the artefact is absent.

    Args:
        request: The request being served.

    Returns:
        The state, loaded or not.
    """
    return app_state(request).model


def get_agent_state(request: Request) -> AgentServiceState:
    """Read the agent service's state without requiring it to be loaded.

    Args:
        request: The request being served.

    Returns:
        The state, loaded or not.
    """
    return app_state(request).agent
