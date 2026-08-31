"""The model service: probability, attribution, counterfactual, health and provenance.

**What this application is allowed to depend on.** `scikit-learn`, `mlflow`, `shap`, `pandas`
and `fastapi`. Not `anthropic`, not `chromadb`, not `langgraph`, not `sentence-transformers`
and therefore not `torch`. Returning `P(default)` must not cost two gigabytes of transformer,
and a change to the embedding model must not be a reason to redeploy the scorer. Nothing here
imports from `credit_copilot.agent`, and `tests/test_api.py` checks that in a fresh
interpreter rather than trusting the import list to stay clean.

**Why the endpoints re-use the same three services the copilot's tools use, and do not
reimplement them.** `models.registry`, `explain.shap_service` and `explain.counterfactual`
are the single implementation; the tools of `agent/tools.py` and these endpoints are two
consumers of it. That is section 6.3 of `docs/METHODOLOGY.md` applied one layer up: two
implementations of the same arithmetic diverge, and the symptom appears in production rather
than in the tests.

**Why no endpoint here imputes anything.** The contract refuses an incomplete applicant
before the artefact is touched, and the reason is written once, in `api/schemas.py`. What is
worth restating is the consequence: **there is no request to this service that returns a
probability without the 23 attributes being present, known and inside their plausible
ranges.** A refusal names the field. It never returns a number.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Annotated, Final

from fastapi import Depends, FastAPI

from credit_copilot import __version__
from credit_copilot.api import dependencies
from credit_copilot.api.dependencies import (
    AppState,
    LoadPhase,
    ModelServiceState,
    get_model_state,
)
from credit_copilot.api.schemas import (
    DecisionContext,
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    FeatureContributionOut,
    HealthResponse,
    HealthStatus,
    MetricWithBaseline,
    ModelIdentity,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ReadinessResponse,
    SimulateRequest,
    SimulateResponse,
    ValidationSummary,
)
from credit_copilot.explain.counterfactual import evaluate_scenario
from credit_copilot.models.decision import CAUSAL_NOTE, DIRECTION_NOTE, decide

__all__ = ["app", "create_app"]

ModelState = Annotated[ModelServiceState, Depends(get_model_state)]
"""The loaded artefact, injected. `Annotated` rather than a `Depends()` default so that the
dependency is part of the type and not a mutable argument default."""

SERVICE_NAME: Final[str] = "model"
"""Name this application logs under and reports in `/health`."""

_STATUS_BY_PHASE: Final[Mapping[LoadPhase, HealthStatus]] = {
    "loading": "starting",
    "ready": "ok",
    "degraded": "degraded",
}
"""Loader phase to the word `/health` reports.

A separate vocabulary because the two audiences differ: `phase` is the loader's own state and
`status` is what a person reads in a dashboard. They are kept in one map rather than in a
conditional so that adding a phase without deciding what it looks like from outside is a
`KeyError` and not a silent default.
"""

_VALIDATION_PROTOCOL: Final[str] = (
    "StratifiedKFold(5, shuffle=True, random_state=42), con el preprocesador ajustado dentro "
    "de cada fold. Ninguna de estas cifras se calculó sobre las filas con las que se ajustó "
    "el artefacto registrado."
)
"""How the metrics in `/model-info` were measured. Quoted from section 4 of the model card."""


class _MetricSpec:
    """One metric of the registry run, and what it has to be read against.

    Attributes:
        key: Metric name in the MLflow run.
        name: Name reported to the caller.
        std_key: Run metric holding the standard deviation across folds, when there is one.
        baseline_keys: Baseline label to the run metric that holds it.
        is_primary: Whether this is the decision metric of ADR-0002.
    """

    def __init__(
        self,
        key: str,
        name: str,
        std_key: str | None = None,
        baseline_keys: dict[str, str] | None = None,
        is_primary: bool = False,
    ) -> None:
        """Declare one metric row of `/model-info`."""
        self.key = key
        self.name = name
        self.std_key = std_key
        self.baseline_keys = baseline_keys or {}
        self.is_primary = is_primary


_METRIC_SPECS: Final[tuple[_MetricSpec, ...]] = (
    _MetricSpec(
        key="cv_pr_auc_mean",
        name="pr_auc",
        std_key="cv_pr_auc_std",
        baseline_keys={
            "trivial": "cv_baseline_trivial_pr_auc",
            "logistic_l2": "cv_baseline_logistic_pr_auc",
        },
        is_primary=True,
    ),
    _MetricSpec(key="cv_roc_auc_mean", name="roc_auc"),
    _MetricSpec(key="cv_ks_mean", name="ks"),
    _MetricSpec(key="cv_gini_mean", name="gini"),
    _MetricSpec(key="cv_brier_mean", name="brier"),
    _MetricSpec(key="cv_precision_at_top_10pct_mean", name="precision_at_top_10pct"),
    _MetricSpec(key="cv_precision_at_top_5pct_mean", name="precision_at_top_5pct"),
)
"""The metrics this service reports, in the order a reader should meet them.

The decision metric comes first and is the only one the registry run carries baselines for -
`scripts/register_production_model.py` attaches the trivial floor and the logistic baseline to
it precisely so that PR-AUC cannot be quoted alone. For the rest the run has no baseline, and
that is reported as a note rather than by leaving the field out: an absent baseline is a fact
about what was registered, and section 7.3 of `docs/METHODOLOGY.md` is explicit that a metric
without context is an opinion.
"""

_NO_BASELINE_NOTE: Final[str] = (
    "El run del registro no lleva baseline para esta métrica. Las cifras de la logística L2 y "
    "del baseline trivial para todas las métricas están en la sección 4 de docs/MODEL_CARD.md."
)
"""What a metric with no baseline in the run is reported with. Never an omission."""

_DOCUMENTATION: Final[dict[str, str]] = {
    "model_card": "docs/MODEL_CARD.md",
    "que_hace": "docs/MODEL_CARD.md, sección 1 · Qué hace",
    "datos_de_entrenamiento": "docs/MODEL_CARD.md, sección 2 · Con qué datos se entrenó",
    "metricas": "docs/MODEL_CARD.md, sección 4 · Métricas, con su baseline al lado",
    "umbral_y_supuesto_de_costos": (
        "docs/MODEL_CARD.md, sección 5 · Umbral operativo y el supuesto que lo sustenta"
    ),
    "equidad": "docs/MODEL_CARD.md, sección 6 bis · Equidad entre grupos demográficos",
    "limitaciones": "docs/MODEL_CARD.md, sección 7 · Limitaciones conocidas",
    "usos_prohibidos": "docs/MODEL_CARD.md, sección 8 · Para qué NO debe usarse",
    "diccionario_de_datos": "docs/DATA_DICTIONARY.md",
    "mediciones": "docs/EVALUATION.md",
    "decisiones": "docs/adr/",
}
"""Where the full account of this model lives. Sections are named, never numbered by line."""

_LIMITATIONS: Final[tuple[str, ...]] = (
    "No hay validación fuera de tiempo: el dataset no trae fecha de originación, así que "
    "ninguna cifra dice cómo se comportaría el modelo seis meses después. Es bloqueante para "
    "un despliegue real (sección 7.1 del Model Card).",
    "La población es Taiwán, tarjeta de crédito, abril-septiembre de 2005, antes de la crisis "
    "de 2008. No se afirma nada sobre otro país, otro producto ni otra década.",
    "El umbral no es una propiedad del modelo: sale de un supuesto de costos 5:1 declarado y "
    "no medido. Con 3:1 o 10:1 la recomendación cambia para el 48,5% del libro.",
    "La disparidad entre grupos demográficos está medida y NO mitigada: la razón de impacto "
    "dispar cae por debajo de 0,80 para EDUCATION (0,7364) y AGE (0,7796).",
    "No debe usarse como decisión automática sin revisión humana: en el umbral 0,160, "
    "aproximadamente 6 de cada 10 solicitantes rechazados habrían pagado.",
)
"""What a caller must carry with any number this service returns. Section 8 of the model card."""


def _identity(state: ModelServiceState) -> ModelIdentity:
    """Describe the artefact a response came from."""
    model = state.require().model
    return ModelIdentity(name=model.name, version=model.version, uri=model.uri)


def _validation_summary(state: ModelServiceState) -> ValidationSummary:
    """Turn the registry run's metrics into the reported table, or say why there is none."""
    validation = state.require().validation
    if validation.error is not None or not validation.metrics:
        return ValidationSummary(
            protocol=_VALIDATION_PROTOCOL,
            run_id=validation.run_id,
            metrics=(),
            note=None,
            detail=(
                "Las métricas de validación no se pudieron leer del run que registró el "
                f"artefacto: {validation.error or 'el run no lleva métricas'}. El artefacto "
                "sirve igual; lo que falta es su evidencia, y no se sustituye por cifras "
                "escritas en el código. Están en la sección 4 de docs/MODEL_CARD.md."
            ),
        )
    values = validation.metrics
    rows = [
        MetricWithBaseline(
            name=spec.name,
            value=values[spec.key],
            std=values.get(spec.std_key) if spec.std_key else None,
            baselines={
                label: values[key] for label, key in spec.baseline_keys.items() if key in values
            },
            baseline_note=None if spec.baseline_keys else _NO_BASELINE_NOTE,
            is_primary=spec.is_primary,
        )
        for spec in _METRIC_SPECS
        if spec.key in values
    ]
    return ValidationSummary(
        protocol=_VALIDATION_PROTOCOL,
        run_id=validation.run_id,
        metrics=tuple(rows),
        note=(
            "Leídas del run de MLflow que registró el artefacto, no transcritas en el "
            "código. PR-AUC es la métrica de decisión (ADR-0002) y es la única que el run "
            "lleva con baseline al lado."
        ),
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start loading the pinned artefact and return immediately.

    **The load does not happen here, and that is the point.** Measured: with the registry
    unreachable the load takes 263 seconds, and a lifespan that has not returned blocks
    uvicorn from accepting any connection - `/health` included. ADR-0010 decision 3 moves it
    to a daemon thread so the process serves from the first second and reports its loading
    state over HTTP instead of by silence.

    Args:
        app: The application starting up.

    Yields:
        Control back to the server as soon as the loader thread is running. An application
        built with an explicit state - which is how the tests build one - has already decided
        what it holds, so no load is started over it.
    """
    if app.state.autoload:
        dependencies.start_model_load(app.state.services)
    yield


def create_app(state: AppState | None = None) -> FastAPI:
    """Build the model application.

    Args:
        state: A pre-built holder, which is how a test supplies a stub artefact - or an
            explicitly failed load - without a registry, a network or a 300-tree forest.
            Supplying one also turns the start-up load off, so building an application in a
            test never reaches the registry.

    Returns:
        A configured FastAPI application. The artefact is loaded by the lifespan, so importing
        this module costs nothing and creating the application does not touch the network.
    """
    autoload = state is None
    state = state if state is not None else AppState()
    app = FastAPI(
        lifespan=_lifespan,
        title="Credit Risk — servicio del modelo",
        version=__version__,
        summary=(
            "Probabilidad de incumplimiento, atribución local y contrafactual, servidos "
            "desde una versión fija del MLflow Model Registry."
        ),
        description=__doc__,
        responses={
            422: {"model": ErrorResponse, "description": "La petición no cumple el contrato."},
            503: {"model": ErrorResponse, "description": "El artefacto no está cargado."},
        },
    )
    app.state.services = state
    app.state.autoload = autoload
    dependencies.install_request_context(app, SERVICE_NAME)
    dependencies.install_error_handlers(app, SERVICE_NAME)
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    """Attach the five endpoints. Split out only to keep `create_app` readable."""

    @app.post("/predict", response_model=PredictResponse, tags=["scoring"])
    def predict(request: PredictRequest, state: ModelState) -> PredictResponse:
        """Score one applicant with the pinned production artefact.

        The threshold and the cost assumption behind it travel with the number because the
        number alone is not interpretable: 0,19 is a verdict-looking quantity until it is
        read against the 0,160 the entity refuses at and the declared 5:1 ratio that 0,160
        comes from.

        Args:
            request: The applicant's 23 raw attributes.
            state: The loaded artefact.

        Returns:
            The probability, the decision the threshold recommends, and what makes both
            readable.
        """
        model = state.require().model
        probability = float(model.probability_of_default(request.applicant.to_frame())[0])
        return PredictResponse(
            request_id=dependencies.current_request_id(),
            probability_of_default=probability,
            decision=decide(probability),
            decision_context=DecisionContext.current(),
            model=_identity(state),
        )

    @app.post("/explain", response_model=ExplainResponse, tags=["scoring"])
    def explain(request: ExplainRequest, state: ModelState) -> ExplainResponse:
        """Attribute one applicant's score to the features that produced it.

        The feature names are the fitted pipeline's own, checked across the preprocessor, the
        forest and the explanation before anything is attributed. The direction reported for
        each feature is the sign of **this applicant's** SHAP value, never a population sign.

        Args:
            request: The applicant and how many contributions to report.
            state: The loaded artefact and its explainer.

        Returns:
            The probability, the forest score the contributions decompose, the strongest
            contributions, and the two notes that say what the attribution is not.
        """
        service = state.require()
        explanation = service.explainer.explain(request.applicant.to_frame(), request.top_n)
        return ExplainResponse(
            request_id=dependencies.current_request_id(),
            probability_of_default=explanation.probability_of_default,
            decision=decide(explanation.probability_of_default),
            forest_score=explanation.forest_score,
            base_value=explanation.base_value,
            top_features=tuple(
                FeatureContributionOut(
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
            decision_context=DecisionContext.current(),
            model=_identity(state),
        )

    @app.post("/simulate", response_model=SimulateResponse, tags=["scoring"])
    def simulate(request: SimulateRequest, state: ModelState) -> SimulateResponse:
        """Score the applicant as given and as modified, and report both.

        **The response declares what it is.** `claim_type` is `about_the_model` and
        `causal_note` says the rest: this answers *"how would the model evaluate an applicant
        with these attributes?"* and **not** *"what would happen if this client changed
        them?"*. The first is verifiable by re-running the endpoint; the second is a causal
        claim that observational data does not support.

        Args:
            request: The applicant and the attributes the scenario changes.
            state: The loaded artefact.

        Returns:
            Both probabilities, their difference, and both decisions.
        """
        model = state.require().model
        outcome = evaluate_scenario(model, request.applicant.to_frame(), request.changes)
        return SimulateResponse(
            request_id=dependencies.current_request_id(),
            changes=dict(outcome.changes),
            baseline_probability=outcome.baseline_probability,
            scenario_probability=outcome.scenario_probability,
            delta=outcome.delta,
            baseline_decision=decide(outcome.baseline_probability),
            scenario_decision=decide(outcome.scenario_probability),
            causal_note=CAUSAL_NOTE,
            decision_context=DecisionContext.current(),
            model=_identity(state),
        )

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health(state: ModelState) -> HealthResponse:
        """Report that the process is alive and where its artefact is in its lifecycle.

        **Always 200 while the process serves.** This is the container healthcheck's
        endpoint, and ADR-0010 decision 3 is that a service which is alive but still loading
        is not unhealthy: marking it so restarts it, which restarts the load, which is the
        crash loop the whole design exists to avoid. What the caller branches on is `phase`,
        not the status code. The gate is `/ready`.

        Args:
            state: The model service's state, whatever phase it is in.

        Returns:
            The phase, and the reason the load failed when it did.
        """
        return HealthResponse(
            service="model",
            status=_STATUS_BY_PHASE[state.phase],
            phase=state.phase,
            version=__version__,
            model_loaded=state.is_ready,
            detail=state.error,
            load_seconds=state.elapsed_seconds,
            request_id=dependencies.current_request_id(),
        )

    @app.get("/ready", response_model=ReadinessResponse, tags=["operations"])
    def ready(state: ModelState) -> ReadinessResponse:
        """Answer 200 only when this service can actually score.

        `state.require()` raises, and the two exceptions it can raise are what separates the
        two failures: `ServiceLoadingError` becomes 503 `model_loading` with a `Retry-After`,
        and `ModelUnavailableError` becomes 503 `model_unavailable` with the reason. Calling
        it rather than re-deriving the condition here is what keeps `/ready` and `/predict`
        from ever disagreeing about whether the service is up.

        Args:
            state: The model service's state.

        Returns:
            The readiness of the service, when it is ready.
        """
        state.require()
        return ReadinessResponse(
            service="model",
            ready=True,
            phase=state.phase,
            load_seconds=state.elapsed_seconds,
            request_id=dependencies.current_request_id(),
        )

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["operations"])
    def model_info(state: ModelState) -> ModelInfoResponse:
        """Report which artefact is serving, what it measured, and where to read the rest.

        The metrics are read from the MLflow run that registered the artefact, not restated
        here, so this endpoint cannot disagree with the registry. PR-AUC arrives with the
        trivial floor and the logistic baseline beside it, because the run carries them; the
        other metrics arrive with a note saying the run carries no baseline for them and
        where the baselines are.

        Args:
            state: The loaded artefact.

        Returns:
            The artefact's identity, its operating point, its validation evidence, and
            pointers into the model card.
        """
        service = state.require()
        return ModelInfoResponse(
            request_id=dependencies.current_request_id(),
            model=_identity(state),
            decision_context=DecisionContext.current(),
            validation=_validation_summary(state),
            tracking_uri=service.tracking_uri,
            documentation=dict(_DOCUMENTATION),
            limitations=_LIMITATIONS,
        )


app = create_app()
"""The application `uvicorn credit_copilot.api.model_app:app` serves."""
