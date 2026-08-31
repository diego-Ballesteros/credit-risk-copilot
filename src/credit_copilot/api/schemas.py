"""Request and response contracts for both applications. Nothing here computes anything.

**The risk this module was written to close, and how it closes it.** Section 7.5 of
`docs/MODEL_CARD.md` records an open problem: the signature registered in MLflow declares the
23 columns as integers, and *a Python integer cannot represent a missing value*. MLflow says
so itself when the signature is inferred. The project's validator guarantees the training
table has no nulls, but an API receives whatever a client sends.

**The resolution: a null is refused, by name, before the artefact is touched.** Every field of
`ApplicantAttributes` is required and typed `int`, so an absent field and an explicit
`null` both fail validation, and the error names the field. The model is never called.

**Why refusing is right and imputing is wrong here, stated once so it is not re-litigated.**
The three available behaviours are: invent a value, drop the applicant's row, or refuse. The
first is the failure mode section 7.1 of `docs/METHODOLOGY.md` names - `PAY_AMT3 = 0` does not
mean *"we do not know what they paid in July"*, it means *"they paid nothing in July"*, which
is a business fact, and a false one. It would also be invisible: the response would carry a
perfectly well-formed probability. The second is the first one wearing a different hat,
because a forest asked about an absent column routes the row down whichever branch the
comparison happens to take. The third is the only one that preserves the difference between
*unknown* and *zero*, and it is what section 2.3 of the internal credit policy already
prescribes: an application with incomplete information goes to full manual evaluation, not to
the model. **Refusing is not the API declining to help. It is the API declining to invent.**

**Why the applicant contract is stricter here than inside the process.**
`ApplicantAttributes` subclasses `models.applicant.ApplicantRecord` and adds exactly one
thing: `strict=True`. Inside the process the callers are the project's own code and Pydantic's
lax coercion is harmless. Over HTTP it is not: lax mode turns the JSON literal `true` into
`1`, and `PAY_STATUS_1 = 1` means *"one month of arrears"*. That is a business fact
manufactured out of a type error, which is the same failure as imputing, arriving by a
different door. Strict mode rejects `true`, `"1"` and `1.0` and names the field. The fields
themselves are **not** restated - they are inherited, so the contract cannot drift from
`schema.py`.

**Why every response that carries a probability also carries the threshold and the
assumption behind it.** `0,19` is not interpretable on its own; `0,19 against a threshold of
0,160, which comes from a declared 5:1 cost ratio that this dataset cannot measure` is. The
threshold is not a property of the model - moving the ratio between 3:1 and 10:1 moves 48.5%
of the book - so a response that omitted it would be inviting the reader to treat a business
assumption as a measurement. `DecisionContext` is therefore a required field of every
response that reports a probability, not an optional annotation.
"""

from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from credit_copilot.models.applicant import ApplicantRecord
from credit_copilot.models.decision import (
    CAUSAL_NOTE,
    COST_ASSUMPTION,
    COST_RATIO_FN_TO_FP,
    DECISION_CAVEAT,
    OPERATING_THRESHOLD,
    Decision,
)
from credit_copilot.models.registry import require_known_values

__all__ = [
    "MAX_TOP_FEATURES",
    "ApplicantAttributes",
    "ChatRequest",
    "ChatResponse",
    "CitationOut",
    "DecisionContext",
    "ErrorBody",
    "ErrorResponse",
    "ExplainRequest",
    "ExplainResponse",
    "FeatureContributionOut",
    "HealthResponse",
    "MetricWithBaseline",
    "ModelIdentity",
    "ModelInfoResponse",
    "PredictRequest",
    "PredictResponse",
    "SimulateRequest",
    "SimulateResponse",
    "TokenUsageOut",
    "ToolInvocation",
    "ValidationSummary",
]

MAX_TOP_FEATURES: Final[int] = 20
"""Ceiling on how many SHAP contributions `/explain` will report.

Section 4.2 of the internal credit policy requires five. The ceiling exists so that a client
cannot ask for an attribution over all 110 transformed features and read the tail - which is
noise around zero - as if it ranked anything.
"""


class _Strict(BaseModel):
    """Base for every contract here: unknown fields are refused, not ignored.

    An ignored field is a request the client believes it sent and the server never read.
    Over an API that difference is invisible until it matters, so it is refused up front.
    """

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# The applicant
# ---------------------------------------------------------------------------


class ApplicantAttributes(ApplicantRecord):
    """The 23 raw attributes, over HTTP. Inherits every field; adds strictness and blame.

    See the module docstring for why strict mode is the boundary's job and not the internal
    contract's, and for why a missing or null field is refused rather than filled in.

    **Why the value check is repeated per field here.** `ApplicantRecord` runs
    `require_known_values` over the whole record after it is built, which is right in process:
    the exception message names the column, and the caller is our own code reading a message.
    Over HTTP it is not enough. A whole-record validator makes Pydantic attribute the failure
    to `applicant`, so `error.fields` says `applicant` for an out-of-range `AGE` - and
    `fields` is the part a client can act on without parsing Spanish prose. Running the same
    check per field puts the blame where a caller can use it. It is the **same** function on
    the same contract, called with one column instead of twenty-three, so there is no second
    rule to keep in step - only a second place it is asked from.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("*")
    @classmethod
    def _value_must_be_known(cls, value: int, info: ValidationInfo) -> int:
        """Check one attribute against the data contract, so the error names that attribute.

        Args:
            value: The attribute's value.
            info: Carries the field name, which is what turns the check into a per-field one.

        Returns:
            The value unchanged. This validator never repairs anything.

        Raises:
            UnknownValueError: The value is a category the contract does not recognise or a
                magnitude outside its plausible range. Nothing is clipped and nothing is
                collapsed: a value outside the range is a data error, not a number to round.
        """
        if info.field_name:
            require_known_values({info.field_name: value})
        return value


# ---------------------------------------------------------------------------
# The pieces that travel with a probability
# ---------------------------------------------------------------------------


class DecisionContext(_Strict):
    """What makes a probability readable. Present in every response that carries one.

    Attributes:
        threshold: The operating threshold applied. A probability at or above it is a
            recommendation to refuse.
        cost_ratio_fn_to_fp: The cost ratio the threshold is derived from. **Declared, not
            measured**: this dataset carries no exposure, recovery or margin data.
        cost_assumption: The full statement of that assumption, including what moving the
            ratio would do.
        decision_caveat: What no band authorises. A probability reads like a verdict and is
            not one.
    """

    threshold: float
    cost_ratio_fn_to_fp: int
    cost_assumption: str
    decision_caveat: str

    @classmethod
    def current(cls) -> "DecisionContext":
        """Build the context from the project's single control point.

        Returns:
            The operating threshold and the sentences `models/decision.py` declares. Read
            from there rather than restated, so an API response and a tool result cannot
            disagree about what the threshold is or where it came from.
        """
        return cls(
            threshold=OPERATING_THRESHOLD,
            cost_ratio_fn_to_fp=COST_RATIO_FN_TO_FP,
            cost_assumption=COST_ASSUMPTION,
            decision_caveat=DECISION_CAVEAT,
        )


class ModelIdentity(_Strict):
    """Which artefact produced a number, in the form a credit file would record it.

    Attributes:
        name: Registered model name.
        version: Registry version. Pinned, never `latest`.
        uri: The `models:/name/version` URI the artefact was loaded from.
    """

    name: str
    version: str
    uri: str


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


class PredictRequest(_Strict):
    """One applicant to score.

    Attributes:
        applicant: The 23 raw attributes. All required; see the module docstring.
    """

    applicant: ApplicantAttributes


class PredictResponse(_Strict):
    """A probability, the decision it implies, and what makes both readable.

    Attributes:
        request_id: Correlation identifier of the request that produced this.
        probability_of_default: Calibrated probability from the pinned artefact.
        decision: What the operating threshold recommends. **A recommendation, not an
            authorisation**, and not an approval gate.
        decision_context: The threshold and the cost assumption behind it.
        model: The artefact that produced the number.
    """

    request_id: str
    probability_of_default: float
    decision: Decision
    decision_context: DecisionContext
    model: ModelIdentity


# ---------------------------------------------------------------------------
# /explain
# ---------------------------------------------------------------------------


class ExplainRequest(_Strict):
    """One applicant to attribute, and how many contributions to report.

    Attributes:
        applicant: The 23 raw attributes.
        top_n: How many features to report, ranked by absolute contribution. Defaults to the
            five that section 4.2 of the internal credit policy requires in a credit file.
    """

    applicant: ApplicantAttributes
    top_n: int = Field(default=5, ge=1, le=MAX_TOP_FEATURES)


class FeatureContributionOut(_Strict):
    """One feature's signed contribution to one applicant's score.

    Attributes:
        feature: Name of the transformed feature, as the fitted pipeline declares it.
        shap_value: Signed contribution **for this applicant**. Positive pushes to default.
        magnitude: Absolute value, which is what ranks the features.
        direction: Sign of `shap_value` for this applicant. Never a population sign.
        feature_value: What the feature is worth for this applicant, after preprocessing.
    """

    feature: str
    shap_value: float
    magnitude: float
    direction: Literal["raises_risk", "lowers_risk"]
    feature_value: float


class ExplainResponse(_Strict):
    """What moved this applicant's score, and the two warnings that come with it.

    Attributes:
        request_id: Correlation identifier.
        probability_of_default: The calibrated probability the decision is taken on.
        decision: What the threshold recommends for that probability.
        forest_score: The uncalibrated forest score the contributions decompose.
        base_value: The forest's expected output. `base_value` plus **all** contributions
            reconstructs `forest_score`, not `probability_of_default`.
        top_features: The strongest contributions, descending by magnitude.
        features_considered: How many transformed features the attribution ran over.
        direction_note: Why the direction reported is individual and not a population sign.
        causal_note: Why an attribution is not an effect.
        decision_context: The threshold and the cost assumption behind it.
        model: The artefact that produced the number.
    """

    request_id: str
    probability_of_default: float
    decision: Decision
    forest_score: float
    base_value: float
    top_features: tuple[FeatureContributionOut, ...]
    features_considered: int
    direction_note: str
    causal_note: str
    decision_context: DecisionContext
    model: ModelIdentity


# ---------------------------------------------------------------------------
# /simulate
# ---------------------------------------------------------------------------


class SimulateRequest(_Strict):
    """One applicant and the attributes a scenario changes.

    Attributes:
        applicant: The 23 raw attributes, as given.
        changes: Raw column name to the value it takes in the scenario. At least one. Only
            the 23 raw columns can be changed: the derived features are computed by the
            pipeline, and setting one directly would describe an applicant whose bills and
            limit do not add up to it.
    """

    applicant: ApplicantAttributes
    changes: dict[str, int] = Field(min_length=1)


class SimulateResponse(_Strict):
    """What the model would say about a modified applicant, and what that does not mean.

    Attributes:
        request_id: Correlation identifier.
        claim_type: Always `about_the_model`. The field exists so the limit is a value a
            client can branch on, not only a sentence a client may skip.
        changes: The attributes changed and the values they were changed to.
        baseline_probability: The model's probability for the applicant as given.
        scenario_probability: The model's probability for the modified applicant.
        delta: `scenario_probability - baseline_probability`. **A difference between two
            model outputs, never an estimated effect of making the change.**
        baseline_decision: What the threshold recommends for the applicant as given.
        scenario_decision: What it recommends for the modified applicant.
        causal_note: The sentence this response is not allowed to support.
        decision_context: The threshold and the cost assumption behind it.
        model: The artefact that produced both numbers.
    """

    request_id: str
    claim_type: Literal["about_the_model"] = "about_the_model"
    changes: dict[str, int]
    baseline_probability: float
    scenario_probability: float
    delta: float
    baseline_decision: Decision
    scenario_decision: Decision
    causal_note: str = CAUSAL_NOTE
    decision_context: DecisionContext
    model: ModelIdentity


# ---------------------------------------------------------------------------
# /health and /model-info
# ---------------------------------------------------------------------------


class HealthResponse(_Strict):
    """Whether the service is up, and whether it can actually do its job.

    The two are separate on purpose. A process that answers HTTP but could not load its
    artefact is *running* and not *ready*, and reporting a single boolean would hide which.

    Attributes:
        service: Which of the two applications answered.
        status: `ok` when the service can serve its endpoints, `degraded` when it is up but
            its artefacts are not loaded.
        version: Version of the `credit_copilot` package.
        model_loaded: Whether the pinned registry artefact is in memory. Always reported,
            including by the agent service, whose tools score with it.
        detail: Why the artefact is not loaded, when it is not. `None` when it is.
        request_id: Correlation identifier.
    """

    service: Literal["model", "agent"]
    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    detail: str | None = None
    request_id: str


class MetricWithBaseline(_Strict):
    """One validation metric with what it has to be read against.

    A metric without its baseline is an opinion: 78% accuracy is what a model that always
    predicts "no default" scores on this dataset. Where the registry run carries no baseline
    for a metric, `baselines` is empty and `baseline_note` says so rather than the field
    being quietly omitted.

    Attributes:
        name: Metric name as the registry run records it.
        value: The cross-validated value.
        std: Standard deviation across folds, when the run recorded one.
        baselines: Baseline name to value, for the baselines the run carries.
        baseline_note: What to read the metric against when `baselines` is empty.
        is_primary: Whether this is the decision metric of ADR-0002.
    """

    name: str
    value: float
    std: float | None = None
    baselines: dict[str, float] = Field(default_factory=dict)
    baseline_note: str | None = None
    is_primary: bool = False


class ValidationSummary(_Strict):
    """Where the numbers in `/model-info` come from, and what they do not cover.

    Attributes:
        protocol: The resampling protocol the metrics were measured with.
        run_id: MLflow run that produced the registered artefact and carries the metrics.
        metrics: The metrics, each with its baselines.
        note: What the metrics do and do not establish.
        detail: Why the metrics could not be read, when they could not.
    """

    protocol: str | None = None
    run_id: str | None = None
    metrics: tuple[MetricWithBaseline, ...] = ()
    note: str | None = None
    detail: str | None = None


class ModelInfoResponse(_Strict):
    """Everything a caller needs to quote a number from this service responsibly.

    Attributes:
        request_id: Correlation identifier.
        model: The pinned artefact.
        decision_context: The threshold and the cost assumption behind it.
        validation: The cross-validated metrics with their baselines, read from the
            registry run rather than restated here.
        tracking_uri: The MLflow server the artefact came from, with any embedded
            credentials removed.
        documentation: Where the full account of this model lives, and which section of it
            answers which question.
        limitations: The limitations a caller must carry with any number from this service.
    """

    request_id: str
    model: ModelIdentity
    decision_context: DecisionContext
    validation: ValidationSummary
    tracking_uri: str | None = None
    documentation: dict[str, str]
    limitations: tuple[str, ...]


# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------


class ChatRequest(_Strict):
    """A question for the copilot, and optionally the applicant it is about.

    Attributes:
        query: The analyst's question, verbatim.
        applicant: The 23 raw attributes of the applicant under discussion. `None` means
            there is no applicant, and the tools that need one refuse rather than inventing
            it - which is visible in `tools_invoked` when it happens.
        max_iterations: Re-planning cycles allowed. Capped; see `ChatResponse.max_iterations`
            for what the cap costs and buys.
    """

    query: str = Field(min_length=3, max_length=4000)
    applicant: ApplicantAttributes | None = None
    max_iterations: int | None = Field(default=None, ge=1, le=5)


class ToolInvocation(_Strict):
    """One tool call the copilot made, whether it succeeded or refused.

    A refused call is reported, not dropped: it is why the answer looks the way it does.

    Attributes:
        call_id: Identifier pairing the call with the model's proposal.
        name: Name of the tool.
        arguments: The arguments it was actually run with, after the code bound whatever the
            language model was not allowed to supply - the applicant's attributes are bound
            from the request and never proposed by the model.
        ok: Whether the tool produced a result.
        result: The tool's validated output. `None` when it refused.
        error: Why it refused. `None` when it succeeded.
    """

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None


class CitationOut(_Strict):
    """One source the answer rests on, in the form a credit file would record it.

    Attributes:
        chunk_id: Identifier of the fragment in the index. **Required and non-empty**: a
            citation a reader cannot resolve back to a fragment is not a citation, and the
            contract refuses to construct one.
        citation: What a reader would write to cite it, as the corpus declares it.
        document_id: Source document.
        location: Heading path inside the document.
        is_synthetic: Whether the source document was written for this project rather than
            issued by anyone. A synthetic fragment must never be quoted as regulation.
        integrity_notice: The warnings the fragment carries - that it is synthetic, that the
            chapter is derogated, or both - as the document declares them. Empty when none.
        has_integrity_notice: Whether `integrity_notice` carries anything, so a client can
            branch on it without parsing prose.
    """

    chunk_id: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    document_id: str
    location: str = ""
    is_synthetic: bool = False
    integrity_notice: str = ""
    has_integrity_notice: bool = False


class TokenUsageOut(_Strict):
    """What one call to a language model consumed.

    Reported per call rather than summed, because the graph uses two models at different
    prices and a single total would make the cost depend on a mix nobody could reconstruct.

    Attributes:
        node: Graph node that made the call.
        model: Model identifier the call was billed against.
        input_tokens: Prompt tokens, cached ones excluded.
        output_tokens: Generated tokens.
        cache_read_tokens: Prompt tokens served from cache.
    """

    node: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


class ChatResponse(_Strict):
    """The answer and everything needed to check it, for a reader who has no other context.

    **Why this shape.** The prose is the part that reads well; `tools_invoked` and
    `citations` are the part that can be checked. Section 4.2 of the internal credit policy
    requires a model-assisted credit file to record the model, its version, the threshold and
    the top contributions - all of which live in the tool records rather than in the answer -
    and section 11.5 of `docs/MODEL_CARD.md` establishes that a normative sentence counts as
    cited only when it appears in `citations`, never because the answer sounds normative.
    That is why `citations` is a field of its own and not something to be parsed out of text.

    Attributes:
        request_id: Correlation identifier.
        query: The question, verbatim.
        answer: The synthesised answer.
        outcome: How the run ended.
        outcome_meaning: What that outcome means, in one sentence, so the code does not have
            to be read to interpret it.
        unresolved_gap: What the assessor said was still missing when the run ended. Empty
            when nothing was.
        tools_invoked: Every tool call made, in completion order, refusals included.
        citations: Every distinct source retrieved, in first-seen order.
        llm_calls: How many calls to a language model the run made.
        iterations: How many times the planner ran.
        max_iterations: The cap it ran under.
        token_usage: What each language-model call consumed.
        decision_context: The threshold and the cost assumption. Always present, because any
            probability appearing in the answer or in a tool result is unreadable without it.
        reading_notes: What a reader must know to read this response correctly, keyed by
            topic. Carried in the payload rather than in documentation, so the response
            stands on its own when it is pasted into a credit file.
    """

    request_id: str
    query: str
    answer: str
    outcome: str
    outcome_meaning: str
    unresolved_gap: str = ""
    tools_invoked: tuple[ToolInvocation, ...] = ()
    citations: tuple[CitationOut, ...] = ()
    llm_calls: int = 0
    iterations: int = 0
    max_iterations: int = 0
    token_usage: tuple[TokenUsageOut, ...] = ()
    decision_context: DecisionContext
    reading_notes: dict[str, str]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorBody(_Strict):
    """What went wrong, in the terms the caller can act on.

    Attributes:
        type: Machine-readable class of failure.
        message: What was wrong, naming the field or the artefact involved.
        fields: The request fields the failure is about, when it is about fields. Empty
            otherwise. **This is what makes a refusal actionable**: an applicant rejected for
            a missing attribute is told which attribute, and nothing is filled in for them.
        request_id: Correlation identifier, quotable in a support request and present in
            every log line the request produced.
    """

    type: str
    message: str
    fields: tuple[str, ...] = ()
    request_id: str


class ErrorResponse(_Strict):
    """The single error envelope both applications answer with.

    Attributes:
        error: The failure.
    """

    error: ErrorBody
