"""The graph: plan, fan out to the tools, judge sufficiency, cycle once more or answer.

```
                     START
                       |
                       v
                +--------------+   <-------------------------+
                |     plan     |                             |
                +--------------+                             |
                       |  conditional: which tools, if any   |
       +---------+-----+------+-----------+                  |
       v         v            v           v                  |
 +----------+ +---------+ +----------+ +----------+          |
 | tool_    | | tool_   | | tool_    | | tool_    |          |
 | score    | | explain | | simulate | | policy   |          |
 +----------+ +---------+ +----------+ +----------+          |
       |         |            |           |                  |
       +---------+-----+------+-----------+                  |
                       v                                     |
                +--------------+  <--- un plan vacío entra aquí también
                |    assess    |  ¿alcanza la evidencia?     |
                +--------------+                             |
                       |  no, y quedan ciclos                |
                       +-----------------------------------> +
                       |  sí, o se agotó el límite
                       v
                +--------------+
                |  synthesize  |
                +--------------+
                       |
                       v
                      END
```

**Why an empty plan goes to the assessor and not to the answer.** It was routed straight to
synthesis at first, and running the agent showed why that is wrong: on a query that plainly
needed a simulation the planner called nothing, wrote prose, and the graph accepted it - no
tool ran, no citation existed, and the answer explained at length what it could not do. An
empty plan is not evidence that no tool was needed; it is one more thing the assessor is
there to judge, and judging it costs one extra call only in the case where it happened.

**Why the four tools are four nodes and not one dispatcher.** The routing decision *is* the
plan, and making it visible as edges means a reader can see which tool a query used without
reading a log. It also lets the independent calls run in one superstep instead of in
sequence.

**The cost of that shape, and how the one real dependency is paid.** Nodes in the same
superstep cannot see each other's writes: when the planner asks for a score *and* a policy
lookup in the same cycle, `tool_policy` cannot read the probability `tool_score` is computing
beside it. That dependency is not incidental - resolving the band is the whole point of the
first finding of the retrieval evaluation. It is paid by having the band lookup **score the
applicant itself** whenever one is loaded, against the same pinned artefact and the same
single row. Re-scoring one row is microseconds, and the alternative - trusting the planner to
relay the number it saw - is exactly the "the model supplies a number" failure the tool
contracts were built to prevent.

**Why the loop is capped, and where the cap is enforced twice.** `DEFAULT_MAX_ITERATIONS`
bounds the planner; on top of it, `run_query` passes LangGraph a `recursion_limit` derived
from the same constant, so a bug in the routing predicate cannot spin the graph even if the
counter stops being read. A cap that lives only in a condition is a cap that one wrong
comparison removes.

**Why the assessor is the way this agent says "I do not know".** ADR-0008, decision 4,
refused a similarity threshold on measured evidence: over the three unanswerable questions,
the best result scored above 24 of the 26 answerable ones, so the score cannot separate the
two. The replacement is a judgement over the retrieved *text* - does any of this actually
answer the question - which is a different question asked of a different faculty, and one a
distance between vectors cannot express. It is not claimed to work: it is the mechanism, and
measuring it is the next turn's job.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol, cast

import anthropic
from anthropic.types import Message
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from credit_copilot.agent.prompts import (
    SYSTEM_PROMPT,
    render_assessment_message,
    render_planner_message,
    render_synthesis_message,
)
from credit_copilot.agent.state import (
    DEFAULT_MAX_ITERATIONS,
    AgentState,
    Citation,
    PlannedCall,
    TokenUsage,
    ToolRecord,
    initial_state,
)
from credit_copilot.agent.tools import (
    TOOL_SPECS,
    ApplicantRecord,
    PolicyOutput,
    ToolContext,
    ToolExecutionError,
    anthropic_tool_definitions,
    execute,
)
from credit_copilot.config import settings

__all__ = [
    "ASSESSMENT_MODEL",
    "PLANNER_MODEL",
    "SYNTHESIS_MODEL",
    "Assessment",
    "CopilotConfigurationError",
    "build_client",
    "build_graph",
    "run_query",
]

PLANNER_MODEL: Final[str] = "claude-haiku-4-5"
"""Model that chooses the tools.

The stack table of `docs/ROADMAP.md` assigns the small model to tool calling and a larger one
to synthesis, and this is that decision. Choosing among four tools with declared schemas is a
routing problem, and the code validates every argument afterwards, so a planning mistake
costs a rejected call and not a wrong number.
"""

SYNTHESIS_MODEL: Final[str] = "claude-opus-5"
"""Model that writes the answer. The one place a wrong nuance reaches the analyst directly."""

ASSESSMENT_MODEL: Final[str] = SYNTHESIS_MODEL
"""Model that judges sufficiency.

Deliberately the larger one, and not the planner's. This node *is* the agent's ability to say
it does not know - the mechanism that replaces the similarity threshold ADR-0008 refused - so
its failure mode is the expensive one: a false "sufficient" produces a confident answer built
on fragments that do not support it.
"""

PLANNER_MAX_TOKENS: Final[int] = 4096
"""Output budget for the planning call. Tool calls are short; this is generous headroom."""

ASSESSMENT_MAX_TOKENS: Final[int] = 4096
"""Output budget for the sufficiency verdict."""

SYNTHESIS_MAX_TOKENS: Final[int] = 8192
"""Output budget for the final answer, which carries citations and caveats."""

NODE_PLAN: Final[str] = "plan"
NODE_ASSESS: Final[str] = "assess"
NODE_SYNTHESIZE: Final[str] = "synthesize"

TOOL_NODES: Final[Mapping[str, str]] = {spec.name: f"tool_{spec.name}" for spec in TOOL_SPECS}
"""Tool name to graph node name. Derived from `TOOL_SPECS`, so adding a tool adds a node."""

_SUPERSTEPS_PER_ITERATION: Final[int] = 3
"""Planner, tool fan-out, assessor. One iteration of the cycle costs three supersteps."""


class _GraphNode(Protocol):
    """A graph node: it reads the state and returns the fields it wants written.

    Declared as a protocol rather than as a `Callable` because LangGraph's own node type
    names its parameter `state`, and a bare `Callable[[AgentState], ...]` is positional-only
    and therefore not compatible with it.
    """

    def __call__(self, state: AgentState) -> dict[str, Any]:  # pragma: no cover
        """Run the node."""


class _Router(Protocol):
    """A conditional edge: it reads the state and names the next node or nodes."""

    def __call__(self, state: AgentState) -> str:  # pragma: no cover
        """Choose the next node."""


class CopilotConfigurationError(RuntimeError):
    """The copilot cannot reach the language model with the configuration available."""


class Assessment(BaseModel):
    """The sufficiency verdict, as structured output rather than as prose to be parsed.

    Attributes:
        sufficient: Whether the gathered evidence answers the question with citations.
        gap: One sentence naming what is missing and which call would get it. Empty when
            nothing is missing.
    """

    model_config = ConfigDict(extra="forbid")

    sufficient: bool = Field(
        description="¿La evidencia reunida alcanza para responder con citas?",
    )
    gap: str = Field(
        default="",
        description="Qué falta y qué llamada lo conseguiría. Vacío si no falta nada.",
    )


def build_client(api_key: str | None = None) -> anthropic.Anthropic:
    """Build the Anthropic client, or refuse to continue.

    The key is read from `config.settings`, which reads `.env`, and never from a literal.
    A missing key is a hard failure with a named variable, for the same reason
    `models/tracking.py` refuses a missing tracking URI: the alternative is a failure much
    later, in a node, phrased as an authentication error about something the caller was not
    thinking about.

    Args:
        api_key: Override for the key in `.env`. Used by tests, not by the scripts.

    Returns:
        A configured client.

    Raises:
        CopilotConfigurationError: No key is available.
    """
    key = (api_key or settings.anthropic_api_key or "").strip()
    if not key:
        raise CopilotConfigurationError(
            "ANTHROPIC_API_KEY missing or blank in the .env file, so the copilot cannot "
            "reach the language model. Copy .env.example to .env and fill it in."
        )
    return anthropic.Anthropic(api_key=key)


def build_graph(
    context: ToolContext,
    client: anthropic.Anthropic | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> CompiledStateGraph:
    """Assemble and compile the copilot's graph.

    Args:
        context: The collaborators the tools run against.
        client: Anthropic client. Built from `.env` when not given.
        max_iterations: How many times the planner may run before the graph answers with
            whatever it has. See `state.DEFAULT_MAX_ITERATIONS` for why the default is three.

    Returns:
        The compiled graph, ready to be invoked with a state from `state.initial_state`.

    Raises:
        CopilotConfigurationError: `max_iterations` is not positive, or no API key is
            available and none was passed.
    """
    if max_iterations < 1:
        raise CopilotConfigurationError(
            f"max_iterations={max_iterations} would leave the planner unable to run once."
        )
    llm = client if client is not None else build_client()

    builder = StateGraph(AgentState)
    builder.add_node(NODE_PLAN, _make_planner(llm))
    for tool_name, node_name in TOOL_NODES.items():
        builder.add_node(node_name, _make_tool_node(tool_name, context))
    builder.add_node(NODE_ASSESS, _make_assessor(llm))
    builder.add_node(NODE_SYNTHESIZE, _make_synthesizer(llm, max_iterations))

    builder.add_edge(START, NODE_PLAN)
    builder.add_conditional_edges(
        NODE_PLAN,
        _route_after_plan,
        [*TOOL_NODES.values(), NODE_ASSESS],
    )
    for node_name in TOOL_NODES.values():
        builder.add_edge(node_name, NODE_ASSESS)
    builder.add_conditional_edges(
        NODE_ASSESS,
        _make_assessment_router(max_iterations),
        [NODE_PLAN, NODE_SYNTHESIZE],
    )
    builder.add_edge(NODE_SYNTHESIZE, END)
    return builder.compile()


def run_query(
    query: str,
    context: ToolContext,
    applicant: Mapping[str, int] | None = None,
    client: anthropic.Anthropic | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> AgentState:
    """Run one query through the graph and return the final state.

    Args:
        query: The analyst's question.
        context: The collaborators the tools run against.
        applicant: Raw attributes of the applicant under discussion, if there is one.
        client: Anthropic client. Built from `.env` when not given.
        max_iterations: Planner cycles allowed.

    Returns:
        The final state: the answer, every tool call, every citation, and the call count.
    """
    graph = build_graph(context, client=client, max_iterations=max_iterations)
    final = graph.invoke(
        initial_state(query, applicant),
        config={"recursion_limit": max_iterations * _SUPERSTEPS_PER_ITERATION + 2},
    )
    return cast("AgentState", final)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _make_planner(client: anthropic.Anthropic) -> _GraphNode:
    """Build the planning node: ask the model which tools this query needs."""

    def plan(state: AgentState) -> dict[str, Any]:
        response = client.messages.create(
            model=PLANNER_MODEL,
            max_tokens=PLANNER_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=anthropic_tool_definitions(),
            messages=[
                {
                    "role": "user",
                    "content": render_planner_message(
                        state["query"], state["tool_records"], state["gap"]
                    ),
                }
            ],
        )
        proposed = _tool_calls(response)
        known = [call for call in proposed if call.name in TOOL_NODES]
        rejected = [
            ToolRecord(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                ok=False,
                error=f"`{call.name}` no es una herramienta de este copiloto.",
            )
            for call in proposed
            if call.name not in TOOL_NODES
        ]
        return {
            "plan": known,
            "iterations": state["iterations"] + 1,
            "llm_calls": 1,
            "token_usage": [_usage_of(response, NODE_PLAN, PLANNER_MODEL)],
            "tool_records": rejected,
        }

    return plan


def _make_tool_node(tool_name: str, context: ToolContext) -> _GraphNode:
    """Build the node that runs every call in the plan addressed to one tool."""

    def node(state: AgentState) -> dict[str, Any]:
        records: list[ToolRecord] = []
        citations: list[Citation] = []
        probability = _band_probability(state, context)
        for call in state["plan"]:
            if call.name != tool_name:
                continue
            try:
                output = execute(
                    call.name,
                    call.arguments,
                    context,
                    applicant=state["applicant"],
                    scored_probability=probability,
                )
            except ToolExecutionError as error:
                records.append(
                    ToolRecord(
                        call_id=call.call_id,
                        name=call.name,
                        arguments=call.arguments,
                        ok=False,
                        error=str(error),
                    )
                )
                continue
            records.append(
                ToolRecord(
                    call_id=call.call_id,
                    name=call.name,
                    arguments=call.arguments,
                    ok=True,
                    result=output.model_dump(mode="json"),
                )
            )
            if isinstance(output, PolicyOutput):
                citations.extend(output.citations())
        return {"tool_records": records, "citations": citations}

    return node


def _make_assessor(client: anthropic.Anthropic) -> _GraphNode:
    """Build the node that judges whether the evidence answers the question."""

    def assess(state: AgentState) -> dict[str, Any]:
        response = client.messages.parse(
            model=ASSESSMENT_MODEL,
            max_tokens=ASSESSMENT_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": render_assessment_message(state["query"], state["tool_records"]),
                }
            ],
            output_format=Assessment,
        )
        usage = [_usage_of(response, NODE_ASSESS, ASSESSMENT_MODEL)]
        verdict = response.parsed_output
        if verdict is None:
            return {"sufficient": True, "gap": "", "llm_calls": 1, "token_usage": usage}
        return {
            "sufficient": verdict.sufficient,
            "gap": verdict.gap.strip(),
            "llm_calls": 1,
            "token_usage": usage,
        }

    return assess


def _make_synthesizer(client: anthropic.Anthropic, max_iterations: int) -> _GraphNode:
    """Build the node that writes the answer, and records how the run ended."""

    def synthesize(state: AgentState) -> dict[str, Any]:
        exhausted = not state["sufficient"] and state["iterations"] >= max_iterations
        response = client.messages.create(
            model=SYNTHESIS_MODEL,
            max_tokens=SYNTHESIS_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": render_synthesis_message(
                        state["query"],
                        state["tool_records"],
                        state["citations"],
                        state["gap"],
                        exhausted,
                    ),
                }
            ],
        )
        return {
            "answer": _text_of(response),
            "llm_calls": 1,
            "token_usage": [_usage_of(response, NODE_SYNTHESIZE, SYNTHESIS_MODEL)],
            "outcome": _outcome(state, exhausted),
        }

    return synthesize


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_after_plan(state: AgentState) -> list[str] | str:
    """Send the cycle to the tools the plan named, or to the assessor when it named none.

    An empty plan is **not** routed to the answer. A query that genuinely needs no tool and a
    planner that simply failed to call one look identical here, and only the assessor - which
    reads the question against the evidence - can tell them apart. Sending both to the
    assessor costs one call in the case where the plan was empty and buys the re-planning
    cycle its most useful job.

    Args:
        state: The state after planning.

    Returns:
        The node names of every tool the plan uses, or `assess` when it uses none.
    """
    named = {call.name for call in state["plan"]}
    targets = [node for tool_name, node in TOOL_NODES.items() if tool_name in named]
    return targets if targets else NODE_ASSESS


def _make_assessment_router(max_iterations: int) -> _Router:
    """Build the predicate that decides between another planning cycle and the answer."""

    def route(state: AgentState) -> str:
        if state["sufficient"]:
            return NODE_SYNTHESIZE
        if state["iterations"] >= max_iterations:
            return NODE_SYNTHESIZE
        return NODE_PLAN

    return route


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_calls(response: Message) -> list[PlannedCall]:
    """Read the tool calls out of a planning response."""
    calls: list[PlannedCall] = []
    for block in response.content:
        if block.type != "tool_use":
            continue
        arguments = block.input if isinstance(block.input, dict) else {}
        calls.append(PlannedCall(call_id=block.id, name=block.name, arguments=dict(arguments)))
    return calls


def _usage_of(response: object, node: str, model: str) -> TokenUsage:
    """Read the token usage off a response, tolerating a shape that does not carry it.

    The usage block is read defensively rather than indexed: an SDK that stops reporting a
    field would otherwise turn a cost measurement into a crash in the middle of a run, and
    a missing count is a gap in the estimate, not a failure of the answer.

    Args:
        response: Any Messages API response.
        node: Graph node that made the call.
        model: Model the call was billed against.

    Returns:
        The usage of that call, with zeros where the response reported nothing.
    """
    usage = getattr(response, "usage", None)
    return TokenUsage(
        node=node,
        model=model,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
    )


def _text_of(response: Message) -> str:
    """Join the text blocks of a response, ignoring everything else."""
    return "\n".join(block.text for block in response.content if block.type == "text").strip()


def _band_probability(state: AgentState, context: ToolContext) -> float | None:
    """Find the probability the band lookup should use, preferring one the model produced.

    A probability already computed in an earlier cycle wins. Failing that, and whenever an
    applicant is loaded, the applicant is scored here - see the module docstring on why the
    fan-out cannot read a sibling branch's score. When there is no applicant this returns
    `None`, and the number in the analyst's question is used instead, tagged as such.

    Args:
        state: The current state.
        context: The collaborators; the scorer is used.

    Returns:
        The probability to resolve the band with, or `None` when this system has not
        produced one.
    """
    prior = _probability_from_records(state["tool_records"])
    if prior is not None:
        return prior
    applicant = state["applicant"]
    if applicant is None:
        return None
    try:
        record = ApplicantRecord.model_validate(dict(applicant))
        return float(context.scorer.probability_of_default(record.to_frame())[0])
    except (ValidationError, ValueError, RuntimeError):
        # The applicant is unusable. `score_solicitante` reports that properly if it was
        # planned; the band falls back to whatever number the question carried.
        return None


def _probability_from_records(records: Sequence[ToolRecord]) -> float | None:
    """Read the most recent successful score out of the tool history."""
    for record in reversed(records):
        if record.name == "score_solicitante" and record.ok and record.result:
            value = record.result.get("probability_of_default")
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _outcome(state: AgentState, exhausted: bool) -> str:
    """Classify how the run ended, for the report and for the caller."""
    if exhausted:
        return "answered_with_gaps"
    if not state["tool_records"]:
        return "answered_without_tools"
    return "answered"
