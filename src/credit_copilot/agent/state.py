"""The typed state the graph carries, and the records that make a run auditable afterwards.

**Why the state is typed and not a bag of keys.** Every node reads it and four of them write
to it concurrently. An untyped dictionary makes a misspelt key a silent no-op: the node
returns, the graph advances, and the information simply is not there - which surfaces as an
answer that omitted a tool result rather than as an error.

**Which fields accumulate and which are replaced, and why the difference matters.** The four
tool nodes run in parallel on the same superstep, so two of them writing `tool_records` at
once would race. LangGraph resolves that with a reducer, and the reducer is part of the type:
`tool_records`, `citations` and `llm_calls` are declared as accumulating, everything else is
replaced by whichever node wrote it last. Getting this backwards does not raise - a replaced
`tool_records` would quietly keep one branch's result and drop the other three.

**Why the plan is replaced and the records accumulate.** A re-planning cycle produces a new
plan, which supersedes the old one; it does not produce a new history. What every tool
returned across every cycle is the evidence the synthesis is built from and the audit trail
the credit file needs, so it only ever grows.

**Why a failed tool call is recorded rather than dropped.** A tool that refused - a missing
column, an unrecognised code, an applicant that was never supplied - is telling the planner
something it needs on the next cycle, and it is telling the reader why the answer looks the
way it does. A dropped failure turns both into a mystery.
"""

import operator
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Final, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "AgentState",
    "Citation",
    "Outcome",
    "PlannedCall",
    "ToolRecord",
    "format_tool_records",
    "initial_state",
    "unique_citations",
]

DEFAULT_MAX_ITERATIONS: Final[int] = 3
"""How many times the planner may run before the graph stops and answers with what it has.

**Why a cap exists at all.** The cycle back to the planner is the part of this graph that can
run forever, and an agent that cycles without converging is an agent that spends money
without producing anything. The cap is what turns "it re-plans when the evidence is thin"
into a bounded promise.

**Why three and not ten.** Each iteration costs two calls to a language model - one to plan
and one to judge sufficiency - so the ceiling is seven calls per query. Three is what the
four tools can actually use: the first plan gathers evidence, the second reacts to what the
first found (a failed tool, a query that has to be reworded in the vocabulary the corpus
uses - measured in `docs/analysis/retrieval-evidence.md`, where rephrasing moved the band
table from outside the top eight to third), and the third is the last chance before the
answer has to be honest about what is missing. Beyond that the planner is not converging: it
is asking the same index the same question, and the corpus does not change between attempts.

**Why one and not two would be wrong.** Without a second attempt the agent cannot use what
the first one taught it, and the retrieval evidence says the first phrasing of a question is
frequently the one that misses.
"""

Outcome = Literal[
    "answered",
    "answered_without_tools",
    "answered_with_gaps",
]
"""How a run ended.

`answered` - the assessor judged the gathered evidence sufficient.
`answered_without_tools` - the planner asked for no tool, so the question needed none.
`answered_with_gaps` - the iteration cap was reached with the assessor still unsatisfied.
The third is not a failure to report: it is the state in which the agent has to say what it
could not establish, which is the behaviour section 11.5 of `docs/MODEL_CARD.md` requires.
"""


class PlannedCall(BaseModel):
    """One tool call the language model proposed, before any validation.

    The arguments are held as a raw mapping on purpose: at this point they are a *proposal*,
    not a contract. Validating them against the tool's Pydantic model is what the execution
    step does, and keeping the unvalidated form visible is what lets a rejected call be
    reported with the arguments that were actually proposed.

    Attributes:
        call_id: Identifier the model assigned to this call, used to pair it with its result.
        name: Name of the tool proposed.
        arguments: Arguments as the model produced them. Not yet validated.
    """

    model_config = ConfigDict(frozen=True)

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolRecord(BaseModel):
    """What one tool call did, whether it succeeded or refused.

    Attributes:
        call_id: Identifier of the call this record answers.
        name: Name of the tool.
        arguments: The arguments the tool was actually run with, after the code bound
            whatever the model was not allowed to supply.
        ok: Whether the tool produced a result.
        result: The tool's validated output, flattened for transport. `None` on failure.
        error: Why the tool refused. `None` on success.
    """

    model_config = ConfigDict(frozen=True)

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None


class Citation(BaseModel):
    """The source of one retrieved fragment, in the form a credit file would record it.

    Attributes:
        chunk_id: Identifier of the fragment in the index.
        citation: The citation a reader would write, as the corpus declares it.
        document_id: Source document.
        location: Heading path inside the document.
        is_synthetic: Whether the source document was written for this project.
        integrity_notice: The warnings the fragment carries: synthetic, derogated, or both.
            Empty when the document declares none.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    citation: str = Field(min_length=1)
    document_id: str
    location: str
    is_synthetic: bool = False
    integrity_notice: str = ""


class AgentState(TypedDict):
    """Everything the graph carries from one node to the next.

    Attributes:
        query: The analyst's question, verbatim.
        applicant: The raw attributes of the applicant under discussion, when the caller
            supplied one. `None` means there is no applicant, and the tools that need one
            refuse rather than inventing it.
        plan: The tool calls the last planning step proposed. Replaced each cycle.
        tool_records: Every tool call made in this run, in completion order. Accumulates.
        citations: Every source retrieved in this run. Accumulates, and may repeat; read it
            through `unique_citations`.
        iterations: How many times the planner has run.
        sufficient: The assessor's last verdict on whether the evidence answers the question.
        gap: What the assessor said was missing. Fed back into the next planning step and,
            when the cap is reached, into the answer.
        llm_calls: Calls to the language model made in this run. Accumulates.
        answer: The synthesised answer. Empty until the synthesis node runs.
        outcome: How the run ended; see `Outcome`.
    """

    query: str
    applicant: dict[str, int] | None
    plan: list[PlannedCall]
    tool_records: Annotated[list[ToolRecord], operator.add]
    citations: Annotated[list[Citation], operator.add]
    iterations: int
    sufficient: bool
    gap: str
    llm_calls: Annotated[int, operator.add]
    answer: str
    outcome: Outcome


def initial_state(query: str, applicant: Mapping[str, int] | None = None) -> AgentState:
    """Build a complete starting state.

    Every key is set here rather than left to the first node that happens to write it.
    A `TypedDict` does not enforce presence at runtime, so a partially built state fails
    later, inside a node, as a `KeyError` about a field nobody was thinking about.

    Args:
        query: The analyst's question.
        applicant: Raw attributes of the applicant under discussion, if there is one.

    Returns:
        A state ready to be passed to the compiled graph.
    """
    return AgentState(
        query=query,
        applicant=dict(applicant) if applicant is not None else None,
        plan=[],
        tool_records=[],
        citations=[],
        iterations=0,
        sufficient=False,
        gap="",
        llm_calls=0,
        answer="",
        outcome="answered",
    )


def unique_citations(citations: Iterable[Citation]) -> tuple[Citation, ...]:
    """Drop repeats while keeping the order the sources were first retrieved in.

    The same fragment is legitimately retrieved by two different tool calls - the band
    lookup and a policy question reach the same article - and the accumulating reducer keeps
    both. Deduplication happens on reading rather than on writing, so the record of what
    each call returned stays intact.

    Args:
        citations: Citations in the order they were gathered.

    Returns:
        The distinct citations, by chunk identifier, in first-seen order.
    """
    seen: set[str] = set()
    unique: list[Citation] = []
    for citation in citations:
        if citation.chunk_id in seen:
            continue
        seen.add(citation.chunk_id)
        unique.append(citation)
    return tuple(unique)


def format_tool_records(records: Sequence[ToolRecord]) -> str:
    """Render the tool history as the text a language model reads as evidence.

    Args:
        records: Every tool call made so far.

    Returns:
        One block per call, naming the tool, its arguments and either its result or the
        reason it refused. Empty input renders an explicit statement that no tool ran,
        because an empty string reads as an omission rather than as a fact.
    """
    if not records:
        return "(no se ejecutó ninguna herramienta todavía)"
    blocks: list[str] = []
    for record in records:
        header = f"### {record.name}  argumentos={record.arguments}"
        body = record.error if record.error else str(record.result)
        status = "OK" if record.ok else "FALLÓ"
        blocks.append(f"{header}\n[{status}] {body}")
    return "\n\n".join(blocks)
