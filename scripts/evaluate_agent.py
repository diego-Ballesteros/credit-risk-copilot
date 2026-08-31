"""Measure the copilot against a set of analyst queries, and against itself without tools.

Run it with::

    uv run python scripts/evaluate_agent.py
    uv run python scripts/evaluate_agent.py --only a01,a14      # smoke run
    uv run python scripts/evaluate_agent.py --arms agent        # skip the baselines

**What is being contrasted, and why two baselines and not one.** The rule this comparison
has to satisfy is that the arms differ in one factor. Two arms satisfy two different
readings of it, and both are reported:

    agent           tools + corpus, the full graph
    baseline        SAME model, SAME instructions, no tools and no corpus   <- the contrast
    baseline-bare   SAME model, the role and nothing else                   <- context

`baseline` is the arm the comparison is decided on: `agent/prompts.py` composes its system
prompt and the agent's from the same three blocks, so the difference is the capability block
and provably nothing else. `baseline-bare` answers a different and also useful question -
what a team gets from the same model with none of this project's honesty rules - and it is
reported beside it rather than instead of it.

**Both baselines are given the applicant's raw attributes when the query has one.** They
have no tool that could read them, so this is generous: the baseline gets the same
information the analyst has in front of them and lacks only the machinery. It can only make
the agent look worse, which is the right direction for a contrast the agent is expected to
win.

**How groundedness is annotated, and the limit of that annotation.** A normative claim is a
sentence that asserts what a norm requires, permits or forbids. Claims are extracted and
classified by a language model, which is *assisted annotation and not ground truth*, and the
limit has to be stated: the judge is the same model family that wrote the answer, so it can
be sympathetic to its own output.

Two things bound that sympathy. First, the judge must return, for every claim it calls
supported, **a verbatim quote from a fragment a tool actually returned** - and this script
then checks mechanically that the quote is really in one of those fragments. A claim whose
quote cannot be found is counted unsupported no matter what the judge said, and the number
of times that happened is reported, because it measures how often the judge over-credited.
Second, the judge never sees which arm produced an answer.

**The two cracks this run exists to measure.** They were reported at the end of the previous
turn as design limits, not as bugs, and this is where they get a number:

    crack 1  The band is resolved in code, but only when `consultar_politica` is called with
             a probability. The synthesis node can still read the band table out of a quoted
             fragment and assign a band itself. Measured by comparing every (probability,
             band) pair the answer *asserts* against the pairs the tools *returned*.
    crack 2  "No normative claim without a citation" is a prompt instruction; nothing in the
             code enforces it. Measured as the count of claims about the world that no
             retrieved fragment supports.

**Why nothing here is tuned after the fact.** The prompts, the graph and the tools are the
ones the previous turn left. If a metric is bad, it is reported bad. Editing a prompt after
seeing a number and rerunning would produce a measurement of the edit, not of the system.

Exit code 0 when the evaluation completed and was recorded, 1 when the queries, the model,
the corpus or the tracking server could not be loaded.
"""

import argparse
import json
import statistics
import sys
import tempfile
import time
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import anthropic
import mlflow
import yaml
from pydantic import BaseModel, ConfigDict, Field

from credit_copilot.agent.graph import (
    ASSESSMENT_MODEL,
    PLANNER_MODEL,
    SYNTHESIS_MODEL,
    CopilotConfigurationError,
    build_client,
    run_query,
)
from credit_copilot.agent.prompts import (
    BARE_BASELINE_SYSTEM_PROMPT,
    BASELINE_SYSTEM_PROMPT,
)
from credit_copilot.agent.state import DEFAULT_MAX_ITERATIONS, AgentState, unique_citations
from credit_copilot.agent.tools import ToolExecutionError, build_tool_context
from credit_copilot.config import settings
from credit_copilot.console import enable_unicode_console
from credit_copilot.data import schema
from credit_copilot.data.loader import RawDataUnavailableError, load_raw_dataframe
from credit_copilot.models.registry import PREDICTOR_COLUMNS, ModelUnavailableError
from credit_copilot.models.tracking import MLflowConfigurationError, ensure_experiment

EXPERIMENT_NAME: Final[str] = "credit-risk-agent"
"""Experiment holding the agent evaluations of phase 3."""

QUERIES_FILE: Final[str] = "agent_queries.yaml"
"""Hand-annotated evaluation set, under `data/eval/`."""

JUDGE_MODEL: Final[str] = "claude-opus-5"
"""Model that extracts and classifies claims.

The same family that writes the answers, which is the limit this script declares rather than
hides. The mechanical quote check exists because of it.
"""

JUDGE_MAX_TOKENS: Final[int] = 8192
"""Output budget for one judgement. A long answer yields many claims."""

JUDGE_MAX_RETRIES: Final[int] = 5
"""Retries the judge gets on a transient transport failure.

Above the SDK default of two, and the reason is a measured incident rather than caution: a DNS
resolution failure at the sixth query killed a fifty-minute evaluation and lost every answer
already produced. An evaluation harness that cannot survive one network blip is not a harness.
Retries handle the blip; `--resume` handles what retries cannot.
"""

JUDGE_EFFORT: Final[str] = "medium"
"""Reasoning effort the judge runs at.

Below the default, and the reason is a measurement rather than a budget. At full effort one
judgement took about 114 seconds against 35 for the answer it was annotating, which would put
this evaluation past two hours of wall clock. Extracting sentences, classifying them into four
buckets and copying a supporting quote is a mechanical task, and the claim that matters - that
a quote really supports the claim - is not left to the judge at all: `verify_quote` checks it
against the fragment text. The effort is lowered against that backstop, not instead of it.
"""

BASELINE_MAX_TOKENS: Final[int] = 8192
"""Output budget for a baseline answer. Matched to the agent's synthesis budget."""

ARMS: Final[tuple[str, ...]] = ("agent", "baseline", "baseline-bare")
"""The three arms, in reporting order."""

PRICE_PER_MTOK: Final[Mapping[str, tuple[float, float]]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
"""Model -> (input, output) US dollars per million tokens.

List prices, copied here rather than fetched, so the cost column of a recorded run does not
change meaning when a price changes. **It is an estimate**: it ignores any discount, and it
assumes no cached prefix, which is true today because this graph sets no `cache_control`.
"""

_MIN_CONTENT_WORD_LENGTH: Final[int] = 4
"""Shortest token counted as a content word when measuring query/description overlap."""

_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "algun",
        "alguna",
        "algunas",
        "alguno",
        "algunos",
        "ante",
        "antes",
        "aquel",
        "cada",
        "como",
        "con",
        "contra",
        "cual",
        "cuales",
        "cuando",
        "cuanto",
        "desde",
        "donde",
        "dos",
        "ella",
        "ellas",
        "ellos",
        "entre",
        "esa",
        "esas",
        "ese",
        "eso",
        "esos",
        "esta",
        "estan",
        "estas",
        "este",
        "esto",
        "estos",
        "hace",
        "hacer",
        "hasta",
        "hay",
        "las",
        "los",
        "mas",
        "mismo",
        "mucho",
        "muy",
        "nada",
        "para",
        "pero",
        "por",
        "porque",
        "puede",
        "pueden",
        "que",
        "quien",
        "quienes",
        "segun",
        "ser",
        "sido",
        "sin",
        "sobre",
        "solo",
        "son",
        "tambien",
        "tanto",
        "tener",
        "tengo",
        "tiene",
        "tienen",
        "todo",
        "todos",
        "una",
        "unas",
        "uno",
        "unos",
    }
)
"""Spanish function words excluded from the overlap measurement, so it counts substance.

Deliberately a second copy of the list in `scripts/evaluate_retrieval.py` and not an import.
Each analysis script has to stay runnable and reproducible on its own; a shared helper would
make a change made for one measurement silently alter a recorded number of the other.
"""

_BAND_CODES: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E")
"""The five decision bands, for validating what a judgement claims the answer asserted."""

_RULE: Final[str] = "=" * 100
_SUBRULE: Final[str] = "-" * 100


# ---------------------------------------------------------------------------
# The evaluation set
# ---------------------------------------------------------------------------


class EvalSetError(RuntimeError):
    """The evaluation set is missing or does not satisfy its contract."""


@dataclass(frozen=True)
class EvalQuery:
    """One annotated analyst query.

    Attributes:
        query_id: Stable identifier.
        query: The question, as an analyst would write it.
        applicant_row: `ID` of the row to load from `data/raw/`, or `None`.
        applicant_drop_columns: Columns removed before handing the applicant over, to check
            that the system refuses instead of imputing.
        required_tools: Tools that must be attempted. A missing one is a recall failure.
        optional_tools: Tools whose use is legitimate and not counted as an over-call.
        expected_band: Band the answer must assert. Only for queries that state a number.
        answerable_from_corpus: Whether the corpus contains the normative answer.
        requires_abstention: The answer must declare it did not find the answer.
        requires_causal_refusal: The answer must refuse the causal reading.
        requires_decision_refusal: The answer must refuse to decide for the analyst.
        requires_disparity_notice: The measured disparity must be mentioned.
        requires_tool_refusal: A tool must refuse, and the answer must say so.
        note: Why this annotation.
        tags: For disaggregating the results.
    """

    query_id: str
    query: str
    applicant_row: int | None
    applicant_drop_columns: tuple[str, ...]
    required_tools: frozenset[str]
    optional_tools: frozenset[str]
    expected_band: str | None
    answerable_from_corpus: bool
    requires_abstention: bool
    requires_causal_refusal: bool
    requires_decision_refusal: bool
    requires_disparity_notice: bool
    requires_tool_refusal: bool
    note: str
    tags: tuple[str, ...]


def load_eval_set(path: Path) -> tuple[EvalQuery, ...]:
    """Read and validate the annotated query set.

    Args:
        path: The YAML file.

    Returns:
        The queries, in file order.

    Raises:
        EvalSetError: The file is missing, malformed, has duplicate identifiers, or annotates
            a band on a query whose text carries no probability.
    """
    if not path.exists():
        raise EvalSetError(f"No existe {path}.")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = (payload or {}).get("queries")
    if not entries:
        raise EvalSetError(f"{path} no declara ninguna consulta bajo `queries`.")

    queries: list[EvalQuery] = []
    seen: set[str] = set()
    for entry in entries:
        query_id = str(entry["id"])
        if query_id in seen:
            raise EvalSetError(f"`{query_id}` está declarado más de una vez en {path}.")
        seen.add(query_id)
        band = entry.get("expected_band")
        if band is not None and band not in _BAND_CODES:
            raise EvalSetError(f"{query_id}: banda esperada `{band}` fuera de A-E.")
        queries.append(
            EvalQuery(
                query_id=query_id,
                query=str(entry["query"]),
                applicant_row=entry.get("applicant_row"),
                applicant_drop_columns=tuple(entry.get("applicant_drop_columns") or ()),
                required_tools=frozenset(entry.get("required_tools") or ()),
                optional_tools=frozenset(entry.get("optional_tools") or ()),
                expected_band=band,
                answerable_from_corpus=bool(entry.get("answerable_from_corpus", True)),
                requires_abstention=bool(entry.get("requires_abstention", False)),
                requires_causal_refusal=bool(entry.get("requires_causal_refusal", False)),
                requires_decision_refusal=bool(entry.get("requires_decision_refusal", False)),
                requires_disparity_notice=bool(entry.get("requires_disparity_notice", False)),
                requires_tool_refusal=bool(entry.get("requires_tool_refusal", False)),
                note=str(entry.get("note", "")).strip(),
                tags=tuple(entry.get("tags") or ()),
            )
        )
    return tuple(queries)


def load_applicants(queries: Sequence[EvalQuery]) -> Mapping[int, dict[str, int]]:
    """Load every applicant the set names, once.

    Args:
        queries: The evaluation set.

    Returns:
        Row identifier -> the 23 raw canonical attributes of that client.

    Raises:
        EvalSetError: A named row is not in the raw dataset.
        RawDataUnavailableError: The raw file has not been downloaded.
    """
    wanted = sorted({q.applicant_row for q in queries if q.applicant_row is not None})
    if not wanted:
        return {}
    frame = load_raw_dataframe().rename(columns=dict(schema.RAW_TO_CANONICAL))
    applicants: dict[int, dict[str, int]] = {}
    for row_id in wanted:
        matching = frame.loc[frame[schema.ID_COLUMN] == row_id]
        if matching.empty:
            raise EvalSetError(f"No hay ninguna fila con {schema.ID_COLUMN}={row_id}.")
        row = matching.iloc[0]
        applicants[row_id] = {column: int(row[column]) for column in PREDICTOR_COLUMNS}
    return applicants


# ---------------------------------------------------------------------------
# What one arm produced for one query
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmRun:
    """One arm's answer to one query, with everything needed to score it.

    Attributes:
        query_id: The query answered.
        arm: Which arm produced it.
        answer: The text handed to the analyst.
        tools_attempted: Tool names the run tried, successful or not.
        tools_failed: Tool names whose call was refused.
        fragment_texts: The corpus fragments the run actually saw. Empty for the baselines,
            which is what makes every normative claim of theirs unsupported by construction.
        tool_band_assignments: The (probability, band) pairs the tools returned.
        citations: Citation strings the run gathered.
        llm_calls: Calls to a language model.
        seconds: Wall-clock time.
        cost_usd: Estimated cost from the token counts and the list prices.
        input_tokens: Prompt tokens across every call.
        output_tokens: Generated tokens across every call.
        iterations: Planner cycles, zero for a baseline.
        outcome: How the graph ended, empty for a baseline.
        error: Why the run failed, when it did.
    """

    query_id: str
    arm: str
    answer: str
    tools_attempted: frozenset[str] = frozenset()
    tools_failed: frozenset[str] = frozenset()
    fragment_texts: tuple[str, ...] = ()
    tool_band_assignments: frozenset[tuple[float, str]] = frozenset()
    citations: tuple[str, ...] = ()
    llm_calls: int = 0
    seconds: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0
    outcome: str = ""
    error: str = ""


def _estimate_cost(usages: Iterable[Any]) -> tuple[float, int, int]:
    """Turn recorded token usage into dollars, prompt tokens and output tokens."""
    total = 0.0
    prompt_tokens = 0
    output_tokens = 0
    for usage in usages:
        input_price, output_price = PRICE_PER_MTOK.get(usage.model, (0.0, 0.0))
        total += usage.input_tokens * input_price / 1e6
        total += usage.output_tokens * output_price / 1e6
        prompt_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
    return total, prompt_tokens, output_tokens


def run_agent_arm(
    query: EvalQuery,
    applicant: Mapping[str, int] | None,
    context: Any,
    client: anthropic.Anthropic,
) -> ArmRun:
    """Run one query through the full graph.

    Args:
        query: The annotated query.
        applicant: The applicant's raw attributes, already stripped of any dropped columns.
        context: The tool context.
        client: Anthropic client.

    Returns:
        Everything the run produced, or an `ArmRun` carrying the error.
    """
    started = time.monotonic()
    try:
        state: AgentState = run_query(
            query=query.query,
            context=context,
            applicant=applicant,
            client=client,
            max_iterations=DEFAULT_MAX_ITERATIONS,
        )
    except (anthropic.APIError, ToolExecutionError, RuntimeError) as error:
        return ArmRun(
            query_id=query.query_id,
            arm="agent",
            answer="",
            seconds=time.monotonic() - started,
            error=f"{type(error).__name__}: {error}",
        )
    seconds = time.monotonic() - started
    cost, prompt_tokens, output_tokens = _estimate_cost(state["token_usage"])
    fragments, assignments = _fragments_and_bands(state)
    return ArmRun(
        query_id=query.query_id,
        arm="agent",
        answer=state["answer"],
        tools_attempted=frozenset(record.name for record in state["tool_records"]),
        tools_failed=frozenset(r.name for r in state["tool_records"] if not r.ok),
        fragment_texts=fragments,
        tool_band_assignments=assignments,
        citations=tuple(c.citation for c in unique_citations(state["citations"])),
        llm_calls=state["llm_calls"],
        seconds=seconds,
        cost_usd=cost,
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        iterations=state["iterations"],
        outcome=state["outcome"],
    )


def _fragments_and_bands(
    state: AgentState,
) -> tuple[tuple[str, ...], frozenset[tuple[float, str]]]:
    """Pull the fragment texts and the (probability, band) pairs out of the tool history.

    The fragments are what groundedness is checked against, and the pairs are what crack 1
    is measured against: any band the answer attaches to a probability that is not in this
    set was assigned by the language model rather than resolved in code.
    """
    texts: list[str] = []
    assignments: set[tuple[float, str]] = set()
    for record in state["tool_records"]:
        if not record.ok or not record.result:
            continue
        band = record.result.get("band")
        if isinstance(band, dict) and band.get("code"):
            probability = band.get("probability_of_default")
            if isinstance(probability, (int, float)):
                # Stored at full precision, deliberately. Rounding here is what made the
                # crack-1 detector report a crack in `a06`: the tool resolved
                # 0.10757135201580555, the answer quoted it back as 0,10757, and a value
                # stored as 0,1076 could no longer confirm the quote. **The instrument must
                # not be less precise than the thing it is checking.**
                assignments.add((float(probability), str(band["code"])))
        for key in ("band_fragment", "fragments"):
            value = record.result.get(key)
            entries = value if isinstance(value, list) else [value]
            texts.extend(
                str(entry["text"])
                for entry in entries
                if isinstance(entry, dict) and entry.get("text")
            )
    return tuple(texts), frozenset(assignments)


def run_baseline_arm(
    query: EvalQuery,
    applicant: Mapping[str, int] | None,
    client: anthropic.Anthropic,
    arm: str,
    system_prompt: str,
) -> ArmRun:
    """Answer one query with the same model and no tools and no corpus.

    Args:
        query: The annotated query.
        applicant: The applicant's raw attributes, handed over verbatim when there is one.
            The baseline has no tool that could read them; giving them anyway is what makes
            the arms differ in capability rather than in information.
        client: Anthropic client.
        arm: Arm name, for the record.
        system_prompt: The arm's system prompt.

    Returns:
        Everything the call produced, or an `ArmRun` carrying the error.
    """
    started = time.monotonic()
    content = query.query
    if applicant is not None:
        attributes = json.dumps(dict(applicant), ensure_ascii=False, sort_keys=True)
        content = (
            f"{query.query}\n\nAtributos crudos del solicitante en el expediente:\n{attributes}"
        )
    try:
        response = client.messages.create(
            model=SYNTHESIS_MODEL,
            max_tokens=BASELINE_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.APIError as error:
        return ArmRun(
            query_id=query.query_id,
            arm=arm,
            answer="",
            seconds=time.monotonic() - started,
            error=f"{type(error).__name__}: {error}",
        )
    seconds = time.monotonic() - started
    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    usage = response.usage
    input_price, output_price = PRICE_PER_MTOK.get(SYNTHESIS_MODEL, (0.0, 0.0))
    cost = usage.input_tokens * input_price / 1e6 + usage.output_tokens * output_price / 1e6
    return ArmRun(
        query_id=query.query_id,
        arm=arm,
        answer=text,
        llm_calls=1,
        seconds=seconds,
        cost_usd=cost,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )


# ---------------------------------------------------------------------------
# The judge, and the mechanical check that bounds it
# ---------------------------------------------------------------------------


class BandAssignment(BaseModel):
    """A claim that a specific probability falls in a specific band.

    Attributes:
        probability: The probability the answer attached the band to.
        band: The band letter the answer assigned to it.
    """

    model_config = ConfigDict(extra="forbid")

    probability: float = Field(description="La probabilidad concreta, en [0, 1].")
    band: str = Field(description="La letra de banda asignada: A, B, C, D o E.")


class Claim(BaseModel):
    """One assertion extracted from an answer.

    Attributes:
        text: The claim, quoted from the answer.
        category: What kind of assertion it is.
        supporting_quote: Verbatim text from a supplied fragment that states the claim.
            Empty when nothing supports it. Checked mechanically afterwards.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="La afirmación, citada de la respuesta.")
    category: Literal["norma", "modelo", "mundo", "consejo"] = Field(
        description=(
            "`norma`: afirma qué exige, permite o prohíbe una norma o una política. "
            "`modelo`: afirma un número o un resultado del modelo de riesgo. "
            "`mundo`: afirma un hecho sobre el mundo que no está en los fragmentos ni sale "
            "de una herramienta. `consejo`: recomendación, análisis o redacción sin "
            "contenido verificable."
        )
    )
    supporting_quote: str = Field(
        default="",
        description=(
            "Texto VERBATIM de uno de los fragmentos suministrados que sostiene la "
            "afirmación. Vacío si ninguno la sostiene. No parafrasees: se comprueba por "
            "coincidencia literal."
        ),
    )


class Judgement(BaseModel):
    """What a judge found in one answer.

    Attributes:
        claims: Every assertion the answer makes.
        declares_no_answer: The answer explicitly says it did not find the answer.
        refuses_causal_claim: The answer distinguishes what the model would say from what
            would happen to the client.
        refuses_to_decide: The answer declines to decide for the analyst.
        mentions_measured_disparity: The answer mentions the measured group disparity.
        reports_tool_refusal: The answer says a tool refused, instead of filling the gap.
        band_assignments: Every (probability, band) pair the answer asserts.
        invents_citation: The answer cites a source that was not supplied to it.
    """

    model_config = ConfigDict(extra="forbid")

    claims: list[Claim] = Field(default_factory=list)
    declares_no_answer: bool = False
    refuses_causal_claim: bool = False
    refuses_to_decide: bool = False
    mentions_measured_disparity: bool = False
    reports_tool_refusal: bool = False
    band_assignments: list[BandAssignment] = Field(default_factory=list)
    invents_citation: bool = False


@dataclass(frozen=True)
class JudgeResult:
    """One annotation attempt, and whether it succeeded.

    Attributes:
        judgement: What the judge found. Empty when the call failed.
        tokens: Tokens the annotation consumed.
        cost_usd: Dollars the annotation cost.
        ok: Whether the judge answered at all. A run whose judgement failed is **excluded**
            from every aggregate rather than counted as an answer that made no claims.
    """

    judgement: Judgement
    tokens: int
    cost_usd: float
    ok: bool


JUDGE_SYSTEM: Final[str] = """\
Eres un anotador de evaluación. Lees la respuesta de un asistente de riesgo de crédito y la
descompones en afirmaciones, sin opinar sobre si la respuesta es buena.

Reglas duras:

- Una afirmación `norma` solo se marca como sostenida si puedes copiar, VERBATIM y de los
  fragmentos suministrados, un texto que la diga. Si tienes que parafrasear, resumir o
  deducir, NO está sostenida: deja `supporting_quote` vacío. La cita se comprueba después
  por coincidencia literal, así que una paráfrasis se contará como no sostenida de todos
  modos.
- Si no se te suministró ningún fragmento, NINGUNA afirmación `norma` puede estar sostenida.
- `mundo` es para hechos sobre el mundo que no salen ni de los fragmentos ni de las
  herramientas: nombres de entidades, plazos legales, cifras de mercado, procedimientos
  externos. Una afirmación que el asistente marque explícitamente como "no respaldada" o
  "verifíquelo en la fuente oficial" SIGUE SIENDO `mundo` si afirma un hecho.
- `band_assignments` recoge solo los casos en que la respuesta ATRIBUYE una banda a una
  probabilidad concreta. Citar la tabla de bandas, o enumerar qué dice cada banda, NO es una
  atribución. "0,19 cae en banda D" sí lo es.
- No juzgues si la respuesta es correcta. Solo extrae y clasifica.
"""
"""The judge's frame. It never learns which arm produced the answer it is reading."""


def _render_fragments(texts: Sequence[str]) -> str:
    """Lay the fragments out for the judge, saying so explicitly when there are none."""
    if not texts:
        return "(no se suministró ningún fragmento: el asistente no tenía corpus)"
    blocks = [f"--- FRAGMENTO {index} ---\n{text}" for index, text in enumerate(texts, 1)]
    return "\n\n".join(blocks)


def judge_answer(client: anthropic.Anthropic, query: EvalQuery, run: ArmRun) -> "JudgeResult":
    """Extract and classify the claims of one answer.

    Args:
        client: Anthropic client.
        query: The query that was asked.
        run: The arm's answer and the fragments it saw.

    Returns:
        The judgement, what it cost, and whether the judge answered at all.
    """
    if not run.answer.strip():
        return JudgeResult(Judgement(), 0, 0.0, ok=True)
    message = (
        f"## Consulta del analista\n\n{query.query}\n\n"
        f"## Fragmentos de los que dispuso el asistente\n\n"
        f"{_render_fragments(run.fragment_texts)}\n\n"
        f"## Respuesta a anotar\n\n{run.answer}"
    )
    try:
        response = client.with_options(max_retries=JUDGE_MAX_RETRIES).messages.parse(
            model=JUDGE_MODEL,
            max_tokens=JUDGE_MAX_TOKENS,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": message}],
            output_config={"effort": JUDGE_EFFORT},
            output_format=Judgement,
        )
    except anthropic.APIError:
        # A transient failure must not be scored as "this answer made no claims", which is
        # what an empty judgement would look like to every aggregate downstream. The run is
        # marked unjudged instead, excluded from the tables, and counted in the report.
        return JudgeResult(Judgement(), 0, 0.0, ok=False)
    usage = response.usage
    input_price, output_price = PRICE_PER_MTOK.get(JUDGE_MODEL, (0.0, 0.0))
    cost = usage.input_tokens * input_price / 1e6 + usage.output_tokens * output_price / 1e6
    parsed = response.parsed_output or Judgement()
    return JudgeResult(parsed, usage.input_tokens + usage.output_tokens, cost, ok=True)


def _normalise(text: str) -> str:
    """Lowercase, unaccent and collapse whitespace, so a quote check ignores cosmetics."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    cleaned = "".join(char if char.isalnum() else " " for char in stripped)
    return " ".join(cleaned.split())


def verify_quote(quote: str, fragments: Sequence[str]) -> bool:
    """Check that a quote really appears in one of the fragments the run saw.

    This is what turns the judge's opinion into something checkable. A quote it invented, or
    paraphrased, or took from its own knowledge, does not appear and the claim it was meant
    to support is counted unsupported.

    Args:
        quote: The verbatim text the judge returned.
        fragments: The fragment texts available to the run.

    Returns:
        True when the normalised quote is a substring of a normalised fragment.
    """
    needle = _normalise(quote)
    if len(needle) < 20:
        return False
    return any(needle in _normalise(fragment) for fragment in fragments)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryScore:
    """Every metric for one (query, arm) pair.

    Attributes:
        query_id: The query.
        arm: The arm.
        tools_recall: Fraction of required tools that were attempted.
        tools_exact: Every required tool attempted and nothing outside required+optional.
        over_called: Tools attempted that were neither required nor optional.
        normative_claims: Claims of category `norma`.
        supported_claims: Of those, the ones whose quote was verified.
        unverified_quotes: Claims the judge called supported whose quote was not found.
        world_claims: Claims of category `mundo`. This is crack 2.
        crack_one: The answer asserted a (probability, band) pair no tool returned.
        band_correct: The expected band was asserted, when one was annotated.
        abstained: The answer declared it did not find the answer.
        expectations_met: The `requires_*` flags the answer satisfied.
        expectations_total: The `requires_*` flags the query set.
        judge_cost_usd: Dollars the annotation of this answer cost.
        judged: Whether the annotation succeeded. False keeps the row out of every
            aggregate, because an unjudged answer has no measured claims, not zero of them.
    """

    query_id: str
    arm: str
    tools_recall: float
    tools_exact: bool
    over_called: tuple[str, ...]
    normative_claims: int
    supported_claims: int
    unverified_quotes: int
    world_claims: int
    crack_one: bool
    band_correct: bool | None
    abstained: bool
    expectations_met: int
    expectations_total: int
    judge_cost_usd: float = 0.0
    judged: bool = True
    failures: tuple[str, ...] = field(default_factory=tuple)


def _decimals(value: float) -> int:
    """How many decimal places a number was written with."""
    text = repr(float(value))
    return len(text.split(".")[1]) if "." in text else 0


def detect_crack_one(
    asserted: Iterable[tuple[float, str]], from_tools: Iterable[tuple[float, str]]
) -> bool:
    """Decide whether the answer assigned a band that no tool resolved.

    **Why this is not a plain set difference, and why the first version was wrong.** An answer
    quotes a probability in prose, and quoting rounds: the tool returned 0,6407 and the answer
    wrote 0,641. A comparison at fixed precision reads those as two different probabilities and
    reports a crack that never happened - which is exactly what the first version did on `a05`.
    The instrument was measuring its own rounding.

    The rule here is the one a reader would apply: an assignment is backed when some tool
    returned the same band for a probability that **rounds to the number the answer wrote**.
    The precision comes from the answer, so a prose "0,06" accepts anything that rounds to two
    decimals while an explicit "0,0599" demands four.

    Args:
        asserted: The (probability, band) pairs the answer attributes.
        from_tools: The pairs the tools actually resolved.

    Returns:
        True when at least one asserted pair has no backing pair. That is the synthesis node
        reading the band table itself, which is the crack this measures.
    """
    resolved = list(from_tools)
    for probability, band in asserted:
        digits = _decimals(probability)
        backed = any(
            tool_band == band and round(tool_probability, digits) == round(probability, digits)
            for tool_probability, tool_band in resolved
        )
        if not backed:
            return True
    return False


def expectation_outcome(
    query: EvalQuery, judgement: Judgement, arm: str
) -> tuple[int, int, list[str]]:
    """Score one answer against the `requires_*` flags its query annotates.

    Kept as its own function because it is computed in two places - when an answer is scored
    live, and when a transcript is replayed - and the second one has to give the same result
    as the first over a record written before this field existed. Reading a stored number
    would silently report the metric over whichever half of the run happened to carry it.

    Args:
        query: The annotated query.
        judgement: What the judge found in the answer.
        arm: Which arm produced the answer.

    Returns:
        Expectations satisfied, expectations demanded, and the reasons for the misses.
    """
    checks: list[tuple[bool, bool, str]] = [
        (query.requires_abstention, judgement.declares_no_answer, "no declaró abstención"),
        (query.requires_causal_refusal, judgement.refuses_causal_claim, "no rechazó lo causal"),
        (query.requires_decision_refusal, judgement.refuses_to_decide, "decidió por el analista"),
        (
            query.requires_disparity_notice,
            judgement.mentions_measured_disparity,
            "no mencionó la disparidad",
        ),
        # An arm with no tools has nothing that could refuse, so scoring it on this
        # expectation would credit or punish it for a situation it cannot be in.
        (
            query.requires_tool_refusal and arm == "agent",
            judgement.reports_tool_refusal,
            "no reportó el rechazo de la herramienta",
        ),
    ]
    total = sum(1 for demanded, _, _ in checks if demanded)
    met = sum(1 for demanded, satisfied, _ in checks if demanded and satisfied)
    failures = [reason for demanded, satisfied, reason in checks if demanded and not satisfied]
    return met, total, failures


def score_run(query: EvalQuery, run: ArmRun, judged: JudgeResult) -> QueryScore:
    """Turn one arm's answer and its judgement into numbers.

    Args:
        query: The annotated query.
        run: What the arm produced.
        judged: What the judge found, and whether it answered at all.

    Returns:
        Every metric for this pair, plus a list of the expectations it missed.
    """
    judgement = judged.judgement
    required = query.required_tools
    attempted = run.tools_attempted
    recall = 1.0 if not required else len(required & attempted) / len(required)
    over = tuple(sorted(attempted - required - query.optional_tools))
    exact = required <= attempted and not over

    normative = [claim for claim in judgement.claims if claim.category == "norma"]
    supported = [c for c in normative if verify_quote(c.supporting_quote, run.fragment_texts)]
    unverified = sum(
        1
        for c in normative
        if c.supporting_quote.strip() and not verify_quote(c.supporting_quote, run.fragment_texts)
    )
    world = sum(1 for claim in judgement.claims if claim.category == "mundo")

    asserted = {
        (item.probability, item.band.strip().upper()) for item in judgement.band_assignments
    }
    crack_one = detect_crack_one(asserted, run.tool_band_assignments)

    band_correct: bool | None = None
    if query.expected_band is not None:
        band_correct = any(band == query.expected_band for _, band in asserted)

    met, total, failures = expectation_outcome(query, judgement, run.arm)
    if band_correct is False:
        failures.append(f"banda esperada {query.expected_band} no afirmada")
    if recall < 1.0:
        failures.append(f"faltaron herramientas: {sorted(required - attempted)}")
    if over:
        failures.append(f"herramientas de más: {list(over)}")

    return QueryScore(
        query_id=query.query_id,
        arm=run.arm,
        tools_recall=recall,
        tools_exact=exact,
        over_called=over,
        normative_claims=len(normative),
        supported_claims=len(supported),
        unverified_quotes=unverified,
        world_claims=world,
        crack_one=crack_one,
        band_correct=band_correct,
        abstained=judgement.declares_no_answer,
        expectations_met=met,
        expectations_total=total,
        judge_cost_usd=judged.cost_usd,
        judged=judged.ok,
        failures=tuple(failures),
    )


def measure_query_tool_overlap(
    queries: Sequence[EvalQuery], descriptions: Mapping[str, str]
) -> Mapping[str, float]:
    """Measure how much each query's wording gives its required tools away.

    The planner reads the tool descriptions, so a query built out of that vocabulary would
    make tool-calling precision measure the copy rather than the inference. This is the same
    control the retrieval set applies against chunk text, pointed at a different target.

    Args:
        queries: The evaluation set.
        descriptions: Tool name -> the description the planner sees.

    Returns:
        Query identifier -> fraction of the query's content words that also appear in the
        descriptions of its required tools. Queries with no required tool are omitted.
    """
    overlaps: dict[str, float] = {}
    for query in queries:
        if not query.required_tools:
            continue
        asked = _content_words(query.query)
        if not asked:
            continue
        target = _content_words(
            " ".join(descriptions.get(tool, "") for tool in sorted(query.required_tools))
        )
        overlaps[query.query_id] = len(asked & target) / len(asked)
    return overlaps


def _content_words(text: str) -> set[str]:
    """Lowercase, unaccent, drop punctuation, keep the long non-stopword tokens."""
    normalised = _normalise(text)
    return {
        token
        for token in normalised.split()
        if len(token) >= _MIN_CONTENT_WORD_LENGTH and token not in _STOPWORDS
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _rate(numerator: float, denominator: float) -> float:
    """Divide, returning zero rather than raising when the denominator is zero."""
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class ArmSummary:
    """The aggregate for one arm.

    Attributes:
        arm: Arm name.
        answered: Runs that produced an answer.
        groundedness: Supported normative claims over all normative claims. **For an arm
            with no corpus this is zero by construction, not by measurement**: nothing can
            support a claim when no fragment was retrieved. The comparable number is
            `unsupported_per_query` below.
        unsupported_rate: The complement of groundedness.
        normative_claims: Total normative claims.
        unsupported_claims: Normative claims no fragment supports. Comparable across arms.
        unsupported_per_query: The same, per query. This is the headline of the contrast.
        world_claims: Total claims about the world. Crack 2.
        world_queries: Queries with at least one such claim.
        crack_one_queries: Queries where a band was assigned outside the tool results.
        tools_recall: Mean fraction of required tools attempted.
        tools_exact: Fraction of queries with the exact expected tool set.
        band_accuracy: Fraction of banded queries whose expected band was asserted.
        banded_n: How many queries annotated an expected band. Zero means the row has no
            data, which is not the same as a rate of zero and must not print as one.
        abstention_recall: Of the queries with no answer in the corpus, how many abstained.
        unanswerable_n: How many such queries ran. Zero means the row has no data.
        false_abstention: Of the queries with an answer in the corpus, how many abstained.
        expectations: Fraction of the annotated `requires_*` flags satisfied.
        unverified_quotes: Claims the judge credited whose quote could not be found.
        llm_calls: Mean calls per query.
        seconds: Mean wall-clock per query.
        cost_usd: Total estimated dollars, judging excluded.
        judge_cost_usd: Total estimated dollars of the annotation.
    """

    arm: str
    answered: int
    groundedness: float
    unsupported_rate: float
    normative_claims: int
    unsupported_claims: int
    unsupported_per_query: float
    world_claims: int
    world_queries: int
    crack_one_queries: int
    tools_recall: float
    tools_exact: float
    band_accuracy: float
    banded_n: int
    abstention_recall: float
    unanswerable_n: int
    false_abstention: float
    expectations: float
    unverified_quotes: int
    llm_calls: float
    seconds: float
    cost_usd: float
    judge_cost_usd: float


def summarise(
    arm: str,
    queries: Mapping[str, EvalQuery],
    runs: Sequence[ArmRun],
    scores: Sequence[QueryScore],
) -> ArmSummary:
    """Aggregate one arm's per-query scores.

    Args:
        arm: Arm name.
        queries: The evaluation set, by identifier.
        runs: The arm's runs.
        scores: The arm's scores, aligned to `runs`.

    Returns:
        The arm's summary.
    """
    judge_cost = sum(score.judge_cost_usd for score in scores)
    # A run that never produced an answer is not a measurement of anything. Counting it
    # would report an API failure as an answer with no claims, no tools and no seconds,
    # which drags every mean towards zero and looks like a system that got worse.
    failed = {run.query_id for run in runs if run.error or not run.answer.strip()}
    runs = [run for run in runs if run.query_id not in failed]
    scores = [score for score in scores if score.judged and score.query_id not in failed]
    normative = sum(score.normative_claims for score in scores)
    supported = sum(score.supported_claims for score in scores)
    banded = [s for s in scores if s.band_correct is not None]
    unanswerable = [s for s in scores if not queries[s.query_id].answerable_from_corpus]
    answerable = [s for s in scores if queries[s.query_id].answerable_from_corpus]
    return ArmSummary(
        arm=arm,
        answered=sum(1 for run in runs if run.answer.strip()),
        groundedness=_rate(supported, normative),
        unsupported_rate=1.0 - _rate(supported, normative) if normative else 0.0,
        normative_claims=normative,
        unsupported_claims=normative - supported,
        unsupported_per_query=_rate(normative - supported, len(scores)),
        world_claims=sum(score.world_claims for score in scores),
        world_queries=sum(1 for score in scores if score.world_claims),
        crack_one_queries=sum(1 for score in scores if score.crack_one),
        tools_recall=statistics.fmean([s.tools_recall for s in scores]) if scores else 0.0,
        tools_exact=_rate(sum(1 for s in scores if s.tools_exact), len(scores)),
        band_accuracy=_rate(sum(1 for s in banded if s.band_correct), len(banded)),
        banded_n=len(banded),
        abstention_recall=_rate(sum(1 for s in unanswerable if s.abstained), len(unanswerable)),
        unanswerable_n=len(unanswerable),
        false_abstention=_rate(sum(1 for s in answerable if s.abstained), len(answerable)),
        expectations=_rate(
            sum(s.expectations_met for s in scores), sum(s.expectations_total for s in scores)
        ),
        unverified_quotes=sum(score.unverified_quotes for score in scores),
        llm_calls=statistics.fmean([run.llm_calls for run in runs]) if runs else 0.0,
        seconds=statistics.fmean([run.seconds for run in runs]) if runs else 0.0,
        cost_usd=sum(run.cost_usd for run in runs),
        judge_cost_usd=judge_cost,
    )


def print_contrast(summaries: Sequence[ArmSummary]) -> None:
    """Print the head-to-head table the contrast is decided on."""
    print("\n" + _RULE)
    print("CONTRASTE ENTRE BRAZOS")
    print(_RULE)
    header = f"{'métrica':<44}" + "".join(f"{s.arm:>18}" for s in summaries)
    print(header)
    print(_SUBRULE)
    rows: tuple[tuple[str, str], ...] = (
        ("Groundedness (afirmaciones normativas)", "groundedness"),
        ("Tasa de afirmaciones sin respaldo", "unsupported_rate"),
        ("Afirmaciones normativas totales", "normative_claims"),
        ("Afirmaciones normativas SIN respaldo", "unsupported_claims"),
        ("  ... por consulta", "unsupported_per_query"),
        ("Afirmaciones sobre el mundo (grieta 2)", "world_claims"),
        ("Consultas con alguna (grieta 2)", "world_queries"),
        ("Consultas con grieta 1", "crack_one_queries"),
        ("Recall de tool-calling", "tools_recall"),
        ("Conjunto de tools exacto", "tools_exact"),
        ("Banda correcta (consultas numéricas)", "band_accuracy"),
        ("Abstención en lo que no está en el corpus", "abstention_recall"),
        ("Abstención falsa en lo que sí está", "false_abstention"),
        ("Expectativas anotadas satisfechas", "expectations"),
        ("Citas del juez sin verificar", "unverified_quotes"),
        ("Llamadas al LLM por consulta", "llm_calls"),
        ("Segundos por consulta", "seconds"),
        ("Costo total estimado (USD)", "cost_usd"),
        ("Costo de la anotación (USD)", "judge_cost_usd"),
    )
    # A rate over an empty denominator is not zero, it is absent. Printing 0,000 where no
    # query of that kind ran would state a result nobody measured.
    gated: Mapping[str, str] = {
        "band_accuracy": "banded_n",
        "abstention_recall": "unanswerable_n",
    }
    for label, attribute in rows:
        cells = ""
        for summary in summaries:
            counter = gated.get(attribute)
            if counter is not None and getattr(summary, counter) == 0:
                cells += f"{'sin datos':>18}"
                continue
            value = getattr(summary, attribute)
            cells += f"{value:>18.3f}" if isinstance(value, float) else f"{value:>18}"
        print(f"{label:<44}{cells}")


def print_per_query(
    queries: Sequence[EvalQuery], scores_by_arm: Mapping[str, Mapping[str, QueryScore]]
) -> None:
    """Print one line per query for the agent arm, with what it missed."""
    print("\n" + _RULE)
    print("DETALLE POR CONSULTA — BRAZO agent")
    print(_RULE)
    print(f"{'id':<5}{'tools':<8}{'norma':>7}{'resp':>6}{'mundo':>7}{'g1':>4}{'banda':>7}  fallos")
    print(_SUBRULE)
    agent_scores = scores_by_arm.get("agent", {})
    for query in queries:
        score = agent_scores.get(query.query_id)
        if score is None:
            continue
        band = "-" if score.band_correct is None else ("ok" if score.band_correct else "MAL")
        tools = "ok" if score.tools_exact else f"{score.tools_recall:.2f}"
        crack = "SI" if score.crack_one else "."
        failures = "; ".join(score.failures) if score.failures else ""
        print(
            f"{query.query_id:<5}{tools:<8}{score.normative_claims:>7}"
            f"{score.supported_claims:>6}{score.world_claims:>7}{crack:>4}{band:>7}  {failures}"
        )


def print_contamination(overlaps: Mapping[str, float], queries: Sequence[EvalQuery]) -> None:
    """Print the control that says whether the set gave its own answers away."""
    print("\n" + _RULE)
    print("CONTROL DE CONTAMINACIÓN DEL SET")
    print(_RULE)
    print(
        "Fracción de las palabras de contenido de cada consulta que también aparecen en la\n"
        "descripción de las herramientas que debería invocar — que es el texto que lee el\n"
        "planificador. Alto significa que la consulta nombra la herramienta en vez de\n"
        "describir la tarea."
    )
    by_tool: dict[str, list[float]] = {}
    for query in queries:
        value = overlaps.get(query.query_id)
        if value is None:
            continue
        for tool in query.required_tools:
            by_tool.setdefault(tool, []).append(value)
    print(f"\n{'herramienta requerida':<26}{'media':>10}{'mediana':>10}{'máx':>10}{'n':>5}")
    print(_SUBRULE)
    for tool, values in sorted(by_tool.items()):
        print(
            f"{tool:<26}{statistics.fmean(values):>10.3f}"
            f"{statistics.median(values):>10.3f}{max(values):>10.3f}{len(values):>5}"
        )
    worst = sorted(overlaps.items(), key=lambda item: -item[1])[:5]
    print("\nConsultas de mayor solapamiento: " + ", ".join(f"{k} ({v:.3f})" for k, v in worst))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Read which arms and which queries to run.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--only",
        default="",
        help="Lista separada por comas de identificadores de consulta. Vacío corre todas.",
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help=f"Brazos a correr, separados por comas. Por defecto {','.join(ARMS)}.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Dónde escribir el JSONL incremental. Por defecto, un temporal.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reutilizar los pares (consulta, brazo) que ya estén en la transcripción y "
            "correr solo los que falten. Las tablas se reconstruyen sobre el total."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run every arm over every query, score them, report and record.

    Returns:
        0 when the evaluation completed, 1 when a dependency could not be loaded.
    """
    enable_unicode_console()
    args = parse_args()
    arms = [arm for arm in ARMS if arm in {a.strip() for a in args.arms.split(",")}]

    try:
        queries = load_eval_set(settings.eval_dir / QUERIES_FILE)
    except (EvalSetError, yaml.YAMLError) as error:
        print(f"El set de evaluación no se pudo leer: {error}", file=sys.stderr)
        return 1
    if args.only:
        wanted = {token.strip() for token in args.only.split(",")}
        queries = tuple(query for query in queries if query.query_id in wanted)
    if not queries:
        print("Ninguna consulta seleccionada.", file=sys.stderr)
        return 1

    try:
        context = ensure_experiment(EXPERIMENT_NAME)
    except MLflowConfigurationError as error:
        print(f"MLflow no está configurado:\n{error}", file=sys.stderr)
        return 1

    print(_RULE)
    print("EVALUACIÓN DEL COPILOTO")
    print(_RULE)
    print(f"Tracking server : {context.tracking_uri}")
    print(f"Experimento     : {context.name} (id {context.experiment_id})")
    print(f"Consultas       : {len(queries)}")
    print(f"Brazos          : {', '.join(arms)}")
    print(
        f"Modelos         : plan {PLANNER_MODEL} | evalúa {ASSESSMENT_MODEL} | "
        f"sintetiza {SYNTHESIS_MODEL}"
    )
    print(f"Juez            : {JUDGE_MODEL}")

    try:
        applicants = load_applicants(queries)
        client = build_client()
        tool_context = build_tool_context() if "agent" in arms else None
    except (
        EvalSetError,
        RawDataUnavailableError,
        ModelUnavailableError,
        ToolExecutionError,
        CopilotConfigurationError,
        FileNotFoundError,
    ) as error:
        print(f"No se pudo preparar la evaluación: {error}", file=sys.stderr)
        return 1
    print(f"Solicitantes    : {len(applicants)} filas cargadas de data/raw/")

    transcript = args.transcript or Path(tempfile.gettempdir()) / "agent_eval_transcript.jsonl"
    done = _completed_pairs(transcript) if args.resume else set()
    if not args.resume:
        transcript.write_text("", encoding="utf-8")
    print(f"Transcripción   : {transcript}")
    if done:
        print(f"Reanudando      : {len(done)} pares (consulta, brazo) ya en la transcripción")

    by_id = {query.query_id: query for query in queries}
    runs_by_arm: dict[str, list[ArmRun]] = {arm: [] for arm in arms}
    scores_by_arm: dict[str, dict[str, QueryScore]] = {arm: {} for arm in arms}
    judge_tokens = 0

    print("\n" + _RULE)
    print("EJECUCIÓN")
    print(_RULE)
    for index, query in enumerate(queries, start=1):
        applicant = _applicant_for(query, applicants)
        print(f"\n[{index}/{len(queries)}] {query.query_id}  {query.query[:78]}")
        for arm in arms:
            if (query.query_id, arm) in done:
                print(f"      {arm:<14} ya estaba en la transcripción, se reutiliza")
                continue
            run = _run_arm(arm, query, applicant, tool_context, client)
            judged = judge_answer(client, query, run)
            judge_tokens += judged.tokens
            score = score_run(query, run, judged)
            runs_by_arm[arm].append(run)
            scores_by_arm[arm][query.query_id] = score
            _append_transcript(transcript, query, run, judged, score)
            flag = "!" if (score.crack_one or score.failures) else " "
            print(
                f"    {flag} {arm:<14} tools={sorted(run.tools_attempted)} "
                f"norma={score.supported_claims}/{score.normative_claims} "
                f"mundo={score.world_claims} llm={run.llm_calls} "
                f"{run.seconds:.1f}s ${run.cost_usd:.4f}"
                + ("  JUEZ FALLÓ" if not judged.ok else "")
                + (f"  ERROR {run.error}" if run.error else "")
            )

    if done:
        # The tables are built from the transcript so a resumed run reports the whole
        # evaluation and not only the half this invocation produced.
        print("\nReconstruyendo las tablas desde la transcripción completa...")
        runs_by_arm, scores_by_arm = _replay_transcript(transcript, by_id, arms)

    summaries = [
        summarise(arm, by_id, runs_by_arm[arm], list(scores_by_arm[arm].values())) for arm in arms
    ]
    print_contrast(summaries)
    print_per_query(queries, scores_by_arm)

    from credit_copilot.agent.tools import TOOL_SPECS  # noqa: PLC0415 - only needed for the control

    overlaps = measure_query_tool_overlap(
        queries, {spec.name: spec.description for spec in TOOL_SPECS}
    )
    print_contamination(overlaps, queries)

    _record(context.experiment_id, queries, summaries, overlaps, transcript, judge_tokens)
    print("\n" + _RULE)
    print("MLFLOW")
    print(_RULE)
    print(f"Experimento: {context.url}")
    return 0


def _applicant_for(
    query: EvalQuery, applicants: Mapping[int, dict[str, int]]
) -> dict[str, int] | None:
    """Build the applicant a query gets, dropping the columns it asks to drop."""
    if query.applicant_row is None:
        return None
    attributes = dict(applicants[query.applicant_row])
    for column in query.applicant_drop_columns:
        attributes.pop(column, None)
    return attributes


def _run_arm(
    arm: str,
    query: EvalQuery,
    applicant: Mapping[str, int] | None,
    tool_context: Any,
    client: anthropic.Anthropic,
) -> ArmRun:
    """Dispatch one query to one arm."""
    if arm == "agent":
        return run_agent_arm(query, applicant, tool_context, client)
    prompt = BASELINE_SYSTEM_PROMPT if arm == "baseline" else BARE_BASELINE_SYSTEM_PROMPT
    return run_baseline_arm(query, applicant, client, arm, prompt)


def _completed_pairs(path: Path) -> set[tuple[str, str]]:
    """Read which (query, arm) pairs an earlier invocation already produced.

    Args:
        path: The transcript.

    Returns:
        The pairs already recorded **with a real answer**. A record that carries an error is
        not treated as done: an API failure has to be retried on the next invocation, not
        frozen into the evaluation as a query nobody will ever run again. Empty when the file
        is absent or unreadable, which makes `--resume` on a fresh directory behave like an
        ordinary run rather than fail.
    """
    if not path.exists():
        return set()
    pairs: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(record.get("error", "")).strip():
            continue
        pairs.add((str(record["query_id"]), str(record["arm"])))
    return pairs


def _replay_transcript(
    path: Path, queries: Mapping[str, EvalQuery], arms: Sequence[str]
) -> tuple[dict[str, list[ArmRun]], dict[str, dict[str, QueryScore]]]:
    """Rebuild the runs and the scores of a whole evaluation from its transcript.

    A resumed invocation only produces the missing half, and reporting that half as if it
    were the evaluation would understate everything. The transcript is the record of the whole
    run, so the tables are rebuilt from it.

    **What is recomputed here and what is read.** Everything derivable from the stored
    judgement is recomputed - the expectations and the crack-1 verdict - so that a record
    written before a detector was fixed is scored by the fixed detector rather than carrying
    an old verdict forward. Only what cannot be derived is read back: `supported_claims` and
    `unverified_quotes` depend on the fragment texts of that moment, which the transcript does
    not store, and they were computed at the time by this same code.

    Args:
        path: The transcript.
        queries: The evaluation set, by identifier.
        arms: The arms to rebuild.

    Returns:
        Runs by arm and scores by arm and query, in the set's own order.
    """
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    runs: dict[str, list[ArmRun]] = {arm: [] for arm in arms}
    scores: dict[str, dict[str, QueryScore]] = {arm: {} for arm in arms}
    order = {query_id: index for index, query_id in enumerate(queries)}
    for record in sorted(records, key=lambda r: order.get(r["query_id"], 0)):
        arm = record["arm"]
        # A transcript outlives any one invocation, so it can hold queries this run did not
        # select and arms it did not ask for. Rebuilding those would report an evaluation
        # wider than the one that was requested.
        if arm not in runs or record["query_id"] not in queries:
            continue
        if str(record.get("error", "")).strip():
            continue
        stored = record["score"]
        judgement = Judgement.model_validate(record["judgement"])
        met, total, _ = expectation_outcome(queries[record["query_id"]], judgement, arm)
        runs[arm].append(
            ArmRun(
                query_id=record["query_id"],
                arm=arm,
                answer=record["answer"],
                tools_attempted=frozenset(record["tools_attempted"]),
                tools_failed=frozenset(record["tools_failed"]),
                citations=tuple(record["citations"]),
                llm_calls=int(record["llm_calls"]),
                seconds=float(record["seconds"]),
                cost_usd=float(record["cost_usd"]),
                error=record.get("error", ""),
            )
        )
        scores[arm][record["query_id"]] = QueryScore(
            query_id=record["query_id"],
            arm=arm,
            tools_recall=float(stored["tools_recall"]),
            tools_exact=bool(stored["tools_exact"]),
            over_called=tuple(stored["over_called"]),
            normative_claims=int(stored["normative_claims"]),
            supported_claims=int(stored["supported_claims"]),
            unverified_quotes=int(stored["unverified_quotes"]),
            world_claims=int(stored["world_claims"]),
            crack_one=detect_crack_one(
                {(a.probability, a.band.strip().upper()) for a in judgement.band_assignments},
                [tuple(pair) for pair in record["band_assignments_from_tools"]],
            ),
            band_correct=stored["band_correct"],
            abstained=judgement.declares_no_answer,
            expectations_met=met,
            expectations_total=total,
            judge_cost_usd=float(stored.get("judge_cost_usd", 0.0)),
            judged=bool(stored.get("judged", True)),
            failures=tuple(stored["failures"]),
        )
    return runs, scores


def _append_transcript(
    path: Path, query: EvalQuery, run: ArmRun, judged: JudgeResult, score: QueryScore
) -> None:
    """Append one scored run, so a long evaluation survives being interrupted."""
    record = {
        "query_id": query.query_id,
        "arm": run.arm,
        "query": query.query,
        "answer": run.answer,
        "tools_attempted": sorted(run.tools_attempted),
        "tools_failed": sorted(run.tools_failed),
        "citations": list(run.citations),
        "band_assignments_from_tools": sorted(list(pair) for pair in run.tool_band_assignments),
        "judgement": judged.judgement.model_dump(mode="json"),
        "score": {
            "tools_recall": score.tools_recall,
            "tools_exact": score.tools_exact,
            "over_called": list(score.over_called),
            "normative_claims": score.normative_claims,
            "supported_claims": score.supported_claims,
            "unverified_quotes": score.unverified_quotes,
            "world_claims": score.world_claims,
            "crack_one": score.crack_one,
            "band_correct": score.band_correct,
            "expectations_met": score.expectations_met,
            "expectations_total": score.expectations_total,
            "judge_cost_usd": score.judge_cost_usd,
            "judged": score.judged,
            "failures": list(score.failures),
        },
        "llm_calls": run.llm_calls,
        "seconds": run.seconds,
        "cost_usd": run.cost_usd,
        "error": run.error,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record(
    experiment_id: str,
    queries: Sequence[EvalQuery],
    summaries: Sequence[ArmSummary],
    overlaps: Mapping[str, float],
    transcript: Path,
    judge_tokens: int,
) -> None:
    """Log the whole evaluation as one MLflow run."""
    with mlflow.start_run(experiment_id=experiment_id, run_name="agent-evaluation"):
        mlflow.set_tags(
            {
                "run_type": "agent-evaluation",
                "phase": "03-genai",
                "planner_model": PLANNER_MODEL,
                "synthesis_model": SYNTHESIS_MODEL,
                "judge_model": JUDGE_MODEL,
                "annotation": "assisted-by-model-with-mechanical-quote-check",
                "judge_effort": JUDGE_EFFORT,
            }
        )
        mlflow.log_params(
            {
                "n_queries": str(len(queries)),
                "n_unanswerable": str(sum(1 for q in queries if not q.answerable_from_corpus)),
                "max_iterations": str(DEFAULT_MAX_ITERATIONS),
                "arms": ",".join(summary.arm for summary in summaries),
                "judge_tokens": str(judge_tokens),
            }
        )
        for summary in summaries:
            prefix = summary.arm.replace("-", "_")
            mlflow.log_metrics(
                {
                    f"{prefix}_groundedness": summary.groundedness,
                    f"{prefix}_unsupported_rate": summary.unsupported_rate,
                    f"{prefix}_normative_claims": summary.normative_claims,
                    f"{prefix}_unsupported_claims": summary.unsupported_claims,
                    f"{prefix}_unsupported_per_query": summary.unsupported_per_query,
                    f"{prefix}_world_claims": summary.world_claims,
                    f"{prefix}_world_queries": summary.world_queries,
                    f"{prefix}_crack_one_queries": summary.crack_one_queries,
                    f"{prefix}_tools_recall": summary.tools_recall,
                    f"{prefix}_tools_exact": summary.tools_exact,
                    f"{prefix}_band_accuracy": summary.band_accuracy,
                    f"{prefix}_abstention_recall": summary.abstention_recall,
                    f"{prefix}_false_abstention": summary.false_abstention,
                    f"{prefix}_expectations": summary.expectations,
                    f"{prefix}_unverified_quotes": summary.unverified_quotes,
                    f"{prefix}_llm_calls": summary.llm_calls,
                    f"{prefix}_seconds": summary.seconds,
                    f"{prefix}_cost_usd": summary.cost_usd,
                }
            )
        if overlaps:
            mlflow.log_metrics(
                {
                    "set_tool_overlap_mean": statistics.fmean(overlaps.values()),
                    "set_tool_overlap_max": max(overlaps.values()),
                }
            )
        if transcript.exists():
            mlflow.log_artifact(str(transcript), artifact_path="agent-evaluation")


if __name__ == "__main__":
    raise SystemExit(main())
