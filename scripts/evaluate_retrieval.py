"""Measure the retrieval, and compare the current chunking strategy against two others.

Run it with::

    uv run python scripts/evaluate_retrieval.py

**What is being compared, and why these four.** The strategies are not an arbitrary menu:
they differ by one factor at a time, so each pair isolates one claim.

    A  structural unit  +  full context header encoded   <- the strategy before ADR-0008
    B  structural unit  +  nothing added                 <- A minus B isolates the HEADER
    C  fixed-length cut +  nothing added                 <- B minus C isolates the STRUCTURE
    D  structural unit  +  integrity warnings encoded    <- the strategy ADR-0008 adopted

A, B and C are kept exactly as they were measured before the change. That is the point of
keeping them: **D is only interpretable next to the arm it is meant to reproduce**, and a
comparison rerun with the losing arms deleted cannot be checked by anybody.

D sits between A and B by construction. It carries the part of the header that ADR-0008
decision 3 refuses to remove - the notice that a document is synthetic, the notice that a
chapter is derogated - and drops the rest. **B minus D is therefore the price of those
warnings**, which is a number the project had committed to paying before knowing it.

**Why the baseline is sized from a measurement instead of chosen.** Strategy A produces 89
chunks with a mean body of 591 characters. A 700-character window with 15% overlap over the
same text produces roughly the same number of vectors, so the comparison isolates *where*
the cuts fall rather than *how many* there are. A baseline with half as many chunks would
lose on coverage and tell us nothing about boundaries.

**Why the baseline is given the headings.** Strategy C slides its window over the document
text *including its heading lines*, which is exactly what a naive chunker pointed at the
markdown file would produce. That is generous to the baseline - some windows get a heading
for free - and it is the honest comparison, because it is the thing a team would actually
build rather than a straw man built to lose.

**Why the search here is exact and not the production HNSW index.** What is being measured
is the chunking strategy, so the retrieval has to be free of the index's own approximation.
Every vector is unit length, so cosine similarity is a dot product and an exact ranking is
one matrix multiply. The consequence is worth stating: these numbers describe the
strategies, and the production store adds HNSW's approximation on top of strategy A.

**How a hit is decided across strategies with different chunk shapes.** Strategies A and B
produce one chunk per structural unit, but strategy C produces windows that straddle units,
so chunk identifiers are not comparable. The headline metric is therefore resolved at
**unit** level: a gold annotation names chunks, those chunks resolve to structural units,
each retrieved passage resolves to the set of units its text overlaps, and a hit is a
non-empty intersection. For A and B the exact-fragment metric is also reported, because it
is stricter and it is the one that predicts what an agent actually gets to quote.

Exit code 0 when the comparison completed and was recorded, 1 when the questions, the
corpus, the model or the tracking server could not be loaded.
"""

import csv
import statistics
import sys
import tempfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import mlflow
import numpy as np
import numpy.typing as npt
import yaml

from credit_copilot.config import settings
from credit_copilot.console import enable_unicode_console
from credit_copilot.models.tracking import MLflowConfigurationError, ensure_experiment
from credit_copilot.rag.chunking import (
    DEFAULT_MAX_BODY_CHARS,
    DEFAULT_OVERLAP_CHARS,
    Chunk,
    chunk_corpus,
)
from credit_copilot.rag.documents import CorpusFormatError, SourceDocument, load_corpus
from credit_copilot.rag.embeddings import EMBEDDING_MODEL_NAME, EmbeddingModel

EXPERIMENT_NAME: Final[str] = "credit-risk-retrieval"
"""Experiment holding the retrieval measurements of phase 3."""

QUESTIONS_FILE: Final[str] = "retrieval_questions.yaml"
"""Hand-annotated evaluation set, under `data/eval/`."""

CUTOFFS: Final[tuple[int, ...]] = (1, 3, 5)
"""The k values reported for hit@k."""

RANKING_DEPTH: Final[int] = 10
"""How deep the ranking is kept. The reciprocal rank is zero beyond this."""

FIXED_WINDOW_CHARS: Final[int] = 700
"""Window of the fixed-length baseline.

Not a taste: strategy A produces 89 chunks whose bodies average 591 characters, and a
700-character window with the overlap below yields a comparable number of vectors over the
same text. Matching the vector count is what makes the comparison about boundaries.
"""

FIXED_OVERLAP_CHARS: Final[int] = 105
"""Overlap of the fixed-length baseline: 15% of the window.

A fixed-length cut lands mid-sentence by construction, and overlap is the standard defence:
whatever one cut separated, the next window repeats. Giving the baseline that defence is
the point - a baseline without it would lose for a reason nobody disputes.
"""

_UNIT_OVERLAP_FLOOR: Final[int] = 120
"""Characters a window must share with a unit before it counts as retrieving that unit.

Roughly one sentence. Below it, a window that clipped the last line of an article cannot
plausibly answer from that article, and counting it would inflate the baseline's recall
with fragments that carry no answer.
"""

_MIN_CONTENT_WORD_LENGTH: Final[int] = 4
"""Shortest token counted as a content word when measuring question/chunk overlap."""

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
"""Spanish function words excluded from the overlap measurement, so it counts substance."""

_RULE: Final[str] = "=" * 88


# ---------------------------------------------------------------------------
# The evaluation set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalQuestion:
    """One annotated question of the evaluation set.

    Attributes:
        question_id: Stable identifier.
        question: The question as an analyst would write it.
        expected_chunk_ids: Chunks that answer it. Empty means the corpus has no answer.
        note: Why those chunks and not others.
        tags: Labels used to break the results down.
    """

    question_id: str
    question: str
    expected_chunk_ids: tuple[str, ...]
    note: str
    tags: tuple[str, ...]

    @property
    def answerable(self) -> bool:
        """Whether the corpus contains an answer.

        Returns:
            True when at least one chunk was annotated. There is no separate flag on
            purpose: two copies of the same fact drift apart.
        """
        return bool(self.expected_chunk_ids)


def load_questions(path: Path) -> tuple[EvalQuestion, ...]:
    """Read the annotated evaluation set.

    Args:
        path: Path to the YAML question file.

    Returns:
        The questions, in file order.

    Raises:
        ValueError: The file has no `questions` list, or a question is missing a field.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = payload.get("questions") if isinstance(payload, Mapping) else None
    if not raw:
        raise ValueError(f"{path.name}: no `questions` list found.")
    questions = []
    for entry in raw:
        missing = {"id", "question", "expected_chunk_ids", "note"} - set(entry)
        if missing:
            raise ValueError(f"{path.name}: question {entry.get('id')} misses {sorted(missing)}")
        questions.append(
            EvalQuestion(
                question_id=str(entry["id"]),
                question=str(entry["question"]),
                expected_chunk_ids=tuple(entry["expected_chunk_ids"] or ()),
                note=str(entry["note"]),
                tags=tuple(entry.get("tags") or ()),
            )
        )
    return tuple(questions)


def validate_annotations(questions: Sequence[EvalQuestion], chunks: Sequence[Chunk]) -> None:
    """Refuse an evaluation set that names a chunk the corpus does not contain.

    A gold identifier that no longer resolves is the quiet way an evaluation set rots: the
    question simply never scores a hit, and the strategy takes the blame for an annotation
    that went stale when a document was edited.

    Args:
        questions: The annotated questions.
        chunks: Every chunk of the corpus under strategy A.

    Raises:
        ValueError: A question names a chunk identifier that does not exist.
    """
    known = {chunk.chunk_id for chunk in chunks}
    unknown = {
        f"{question.question_id} -> {chunk_id}"
        for question in questions
        for chunk_id in question.expected_chunk_ids
        if chunk_id not in known
    }
    if unknown:
        raise ValueError(f"The evaluation set names unknown chunks: {sorted(unknown)}")


# ---------------------------------------------------------------------------
# The three strategies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Passage:
    """One indexable unit of a strategy, whatever shape that strategy gives it.

    Attributes:
        passage_id: Identifier inside the strategy.
        text: The text that gets embedded.
        unit_ids: Structural units this passage covers. One for A and B, one or more
            for the fixed-length baseline.
        document_id: Source document.
    """

    passage_id: str
    text: str
    unit_ids: frozenset[str]
    document_id: str


@dataclass(frozen=True)
class Strategy:
    """One chunking strategy, with the passages it produces.

    Attributes:
        key: Short label used in tables and in MLflow metric names.
        name: Human-readable name.
        description: What it does, for the report.
        passages: The passages it produces over the corpus.
        exact_ids_comparable: Whether its passage ids match the gold chunk ids, which is
            true for the structural strategies and false for the fixed-length baseline.
    """

    key: str
    name: str
    description: str
    passages: tuple[Passage, ...]
    exact_ids_comparable: bool


def _unit_passages(chunks: Sequence[Chunk], text_of: Callable[[Chunk], str]) -> tuple[Passage, ...]:
    """Build one passage per structural chunk, taking its text from `text_of`."""
    return tuple(
        Passage(
            passage_id=chunk.chunk_id,
            text=text_of(chunk),
            unit_ids=frozenset({chunk.metadata.unit_id}),
            document_id=chunk.metadata.document_id,
        )
        for chunk in chunks
    )


def build_strategy_a(chunks: Sequence[Chunk]) -> Strategy:
    """Structural units with the whole context header encoded: the strategy before ADR-0008."""
    return Strategy(
        key="A",
        name="Unidad estructural + header completo incrustado",
        description=(
            f"La estrategia anterior a la ADR-0008. max_body_chars={DEFAULT_MAX_BODY_CHARS}, "
            f"overlap_chars={DEFAULT_OVERLAP_CHARS}."
        ),
        passages=_unit_passages(chunks, lambda chunk: f"{chunk.context_header}\n\n{chunk.body}"),
        exact_ids_comparable=True,
    )


def build_strategy_b(chunks: Sequence[Chunk]) -> Strategy:
    """The same units, encoding only the body. Isolates what the header buys."""
    return Strategy(
        key="B",
        name="Unidad estructural, sin nada añadido",
        description="Los mismos cortes que A; se codifica solo el cuerpo del fragmento.",
        passages=_unit_passages(chunks, lambda chunk: chunk.body),
        exact_ids_comparable=True,
    )


def build_strategy_d(chunks: Sequence[Chunk]) -> Strategy:
    """The adopted strategy: only the integrity warnings ride inside the vector."""
    warned = sum(1 for chunk in chunks if chunk.has_integrity_notice)
    return Strategy(
        key="D",
        name="Unidad estructural + solo avisos de integridad",
        description=(
            f"La estrategia adoptada en la ADR-0008. El header va a la metadata; se "
            f"codifican los avisos de documento sintético y de capítulo derogado. "
            f"{warned} de {len(chunks)} chunks llevan aviso."
        ),
        passages=_unit_passages(chunks, lambda chunk: chunk.embed_text),
        exact_ids_comparable=True,
    )


def build_strategy_c(documents: Sequence[SourceDocument]) -> Strategy:
    """Fixed-length windows with overlap, ignoring the structure. The baseline."""
    passages: list[Passage] = []
    for document in documents:
        text, spans = _flatten_document(document)
        for index, (start, end) in enumerate(_windows(len(text))):
            window = text[start:end].strip()
            if not window:
                continue
            passages.append(
                Passage(
                    passage_id=f"{document.document_id}::win{index:03d}",
                    text=window,
                    unit_ids=_units_in_span(spans, start, end),
                    document_id=document.document_id,
                )
            )
    return Strategy(
        key="C",
        name="Corte por longitud fija con solape",
        description=(
            f"Ventana de {FIXED_WINDOW_CHARS} caracteres con {FIXED_OVERLAP_CHARS} de "
            f"solape sobre el texto del documento, encabezados incluidos, ignorando "
            f"dónde empieza y termina cada unidad."
        ),
        passages=tuple(passages),
        exact_ids_comparable=False,
    )


def _flatten_document(document: SourceDocument) -> tuple[str, dict[str, tuple[int, int]]]:
    """Concatenate a document into one string, recording where each unit lives in it.

    The heading line of each unit is kept inline, because that is what a chunker pointed
    at the markdown file would read. The spans are what lets a window be resolved back to
    the units it covers.
    """
    parts: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for unit in document.units:
        segment = f"{unit.heading_path[-1]}\n\n{unit.body}"
        spans[unit.unit_id] = (cursor, cursor + len(segment))
        parts.append(segment)
        cursor += len(segment) + 2
    return "\n\n".join(parts), spans


def _windows(length: int) -> list[tuple[int, int]]:
    """Slide the fixed window across a document, stepping by window minus overlap."""
    step = FIXED_WINDOW_CHARS - FIXED_OVERLAP_CHARS
    starts = range(0, max(length - FIXED_OVERLAP_CHARS, 1), step)
    return [(start, min(start + FIXED_WINDOW_CHARS, length)) for start in starts]


def _units_in_span(spans: Mapping[str, tuple[int, int]], start: int, end: int) -> frozenset[str]:
    """Which units a window covers, requiring enough shared text to be able to answer."""
    covered = set()
    for unit_id, (unit_start, unit_end) in spans.items():
        overlap = min(end, unit_end) - max(start, unit_start)
        if overlap >= min(_UNIT_OVERLAP_FLOOR, unit_end - unit_start):
            covered.add(unit_id)
    return frozenset(covered)


# ---------------------------------------------------------------------------
# Scoring one strategy against the question set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionOutcome:
    """What one strategy did on one question.

    Attributes:
        question_id: Which question.
        answerable: Whether the corpus has an answer.
        unit_rank: 1-based rank of the first passage covering a gold unit, or 0 if none.
        exact_rank: 1-based rank of the first passage whose id is a gold chunk, or 0.
        document_rank: 1-based rank of the first passage from a gold document, or 0. Kept
            apart from `unit_rank` because the two answer different questions: whether the
            strategy found the right body of rules, and whether it found the right article
            inside it. The context header turns out to affect them in opposite directions.
        top_scores: Similarity of the ranked passages, best first.
        top_locations: Where the ranked passages come from, for the report.
    """

    question_id: str
    answerable: bool
    unit_rank: int
    exact_rank: int
    document_rank: int
    top_scores: tuple[float, ...]
    top_locations: tuple[str, ...]

    @property
    def top_score(self) -> float:
        """Similarity of the best-ranked passage.

        Returns:
            The top-1 score, or 0.0 if nothing was retrieved.
        """
        return self.top_scores[0] if self.top_scores else 0.0

    @property
    def margin(self) -> float:
        """How far the best passage stands above the fifth.

        Returns:
            `top-1 minus top-5`, which is the discrimination a cut-off would need.
        """
        return self.top_scores[0] - self.top_scores[4] if len(self.top_scores) >= 5 else 0.0


def evaluate_strategy(
    strategy: Strategy,
    questions: Sequence[EvalQuestion],
    gold_units: Mapping[str, frozenset[str]],
    gold_documents: Mapping[str, frozenset[str]],
    model: EmbeddingModel,
) -> tuple[QuestionOutcome, ...]:
    """Rank every passage of a strategy against every question and record the outcomes.

    Args:
        strategy: The strategy to score.
        questions: The annotated questions.
        gold_units: Structural units annotated as correct, per question identifier.
        gold_documents: Source documents those units belong to, per question identifier.
        model: The embedding model, shared by every strategy so only the text differs.

    Returns:
        One outcome per question, in question order.
    """
    passage_vectors = model.embed_passages([passage.text for passage in strategy.passages])
    query_vectors = model.embed_queries([question.question for question in questions])
    similarity = _cosine(query_vectors, passage_vectors)

    outcomes: list[QuestionOutcome] = []
    for row, question in enumerate(questions):
        order = np.argsort(-similarity[row])[:RANKING_DEPTH]
        ranked = [strategy.passages[index] for index in order]
        gold = gold_units[question.question_id]
        documents = gold_documents[question.question_id]
        outcomes.append(
            QuestionOutcome(
                question_id=question.question_id,
                answerable=question.answerable,
                unit_rank=_first_rank(passage.unit_ids & gold for passage in ranked),
                exact_rank=_first_rank(
                    {passage.passage_id} & set(question.expected_chunk_ids)
                    if strategy.exact_ids_comparable
                    else set()
                    for passage in ranked
                ),
                document_rank=_first_rank(passage.document_id in documents for passage in ranked),
                top_scores=tuple(float(similarity[row][index]) for index in order),
                top_locations=tuple(_describe(passage) for passage in ranked),
            )
        )
    return tuple(outcomes)


def measure_homogenisation(strategy: Strategy, model: EmbeddingModel) -> tuple[float, float]:
    """How alike a strategy makes two passages of the same document, against two of different ones.

    This is the mechanism behind the header's cost, measured rather than argued. The context
    header is a block of near-identical text repeated on every chunk of a document, so it
    pushes all of that document's vectors together. Routing to the right document gets
    easier and telling two articles of that document apart gets harder.

    Args:
        strategy: The strategy whose passages are compared.
        model: The embedding model.

    Returns:
        Mean cosine similarity within a document, and across documents.
    """
    vectors = model.embed_passages([passage.text for passage in strategy.passages])
    similarity = vectors @ vectors.T
    documents = [passage.document_id for passage in strategy.passages]
    within: list[float] = []
    across: list[float] = []
    for left in range(len(documents)):
        for right in range(left + 1, len(documents)):
            bucket = within if documents[left] == documents[right] else across
            bucket.append(float(similarity[left][right]))
    return statistics.fmean(within), statistics.fmean(across)


def _cosine(
    queries: npt.NDArray[np.float32], passages: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Cosine similarity of every query against every passage.

    Both sides come out of the model unit length, so the dot product is the cosine and the
    ranking is exact rather than approximate.
    """
    return queries @ passages.T


def _first_rank(hits: Any) -> int:
    """1-based position of the first truthy element, or 0 when there is none."""
    for rank, hit in enumerate(hits, start=1):
        if hit:
            return rank
    return 0


def _describe(passage: Passage) -> str:
    """Short label of a passage for the per-question table."""
    return passage.passage_id


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyMetrics:
    """The headline numbers of one strategy.

    Attributes:
        key: Strategy label.
        answerable: How many questions had an answer in the corpus.
        hit_at: Unit-level hit rate at each cut-off.
        exact_hit_at: Fragment-level hit rate, or `None` for the baseline.
        mrr: Mean reciprocal rank at unit level, over the kept ranking depth.
        score_summary: Distribution of top-1 similarity over answerable questions.
        hit_scores: Top-1 similarity when the top-1 was a hit.
        miss_scores: Top-1 similarity when it was not.
        unanswerable_scores: Top-1 similarity on the questions with no answer.
        margins: `top-1 minus top-5` per answerable question.
    """

    key: str
    answerable: int
    hit_at: Mapping[int, float]
    document_hit_at: Mapping[int, float]
    exact_hit_at: Mapping[int, float] | None
    mrr: float
    score_summary: Mapping[str, float]
    hit_scores: tuple[float, ...]
    miss_scores: tuple[float, ...]
    unanswerable_scores: tuple[float, ...]
    margins: tuple[float, ...]
    units_per_passage: float
    chars_per_passage: float


def compute_metrics(strategy: Strategy, outcomes: Sequence[QuestionOutcome]) -> StrategyMetrics:
    """Reduce the per-question outcomes of one strategy to its headline numbers.

    Args:
        strategy: The strategy scored.
        outcomes: Its per-question outcomes.

    Returns:
        The metrics for the comparison table.
    """
    answerable = [outcome for outcome in outcomes if outcome.answerable]
    unanswerable = [outcome for outcome in outcomes if not outcome.answerable]
    total = len(answerable)
    reciprocal = [1.0 / o.unit_rank if o.unit_rank else 0.0 for o in answerable]
    return StrategyMetrics(
        key=strategy.key,
        answerable=total,
        hit_at={k: _rate(answerable, "unit_rank", k) for k in CUTOFFS},
        document_hit_at={k: _rate(answerable, "document_rank", k) for k in CUTOFFS},
        exact_hit_at=(
            {k: _rate(answerable, "exact_rank", k) for k in CUTOFFS}
            if strategy.exact_ids_comparable
            else None
        ),
        mrr=sum(reciprocal) / total if total else 0.0,
        score_summary=_summarise([o.top_score for o in answerable]),
        hit_scores=tuple(o.top_score for o in answerable if o.unit_rank == 1),
        miss_scores=tuple(o.top_score for o in answerable if o.unit_rank != 1),
        unanswerable_scores=tuple(o.top_score for o in unanswerable),
        margins=tuple(o.margin for o in answerable),
        units_per_passage=statistics.fmean(len(passage.unit_ids) for passage in strategy.passages),
        chars_per_passage=statistics.fmean(len(passage.text) for passage in strategy.passages),
    )


def _rate(outcomes: Sequence[QuestionOutcome], field: str, cutoff: int) -> float:
    """Fraction of questions whose first hit on `field` lands at or above the cut-off."""
    if not outcomes:
        return 0.0
    hits = sum(1 for o in outcomes if 0 < getattr(o, field) <= cutoff)
    return hits / len(outcomes)


def _summarise(values: Sequence[float]) -> dict[str, float]:
    """Minimum, quartiles, maximum, mean and standard deviation of a sample."""
    if not values:
        return {"count": 0.0}
    ordered = sorted(values)
    return {
        "count": float(len(ordered)),
        "min": ordered[0],
        "p25": ordered[max(0, round(0.25 * (len(ordered) - 1)))],
        "median": statistics.median(ordered),
        "p75": ordered[max(0, round(0.75 * (len(ordered) - 1)))],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "stdev": statistics.pstdev(ordered),
    }


# ---------------------------------------------------------------------------
# Contamination control: how much vocabulary the questions share with their gold
# ---------------------------------------------------------------------------


def measure_lexical_overlap(
    questions: Sequence[EvalQuestion], chunks: Sequence[Chunk]
) -> dict[str, float]:
    """Fraction of each question's content words that also appear in its gold text.

    This is what turns "the questions were written without copying the chunks" from a
    claim into a measurement. A set written by lifting vocabulary out of the corpus would
    score high here, and its hit@k would be measuring the copy rather than the retrieval.

    Args:
        questions: The annotated questions.
        chunks: Every chunk of the corpus, to look the gold text up.

    Returns:
        Overlap per question identifier. Unanswerable questions are absent.
    """
    bodies = {chunk.chunk_id: chunk.body for chunk in chunks}
    overlaps: dict[str, float] = {}
    for question in questions:
        if not question.answerable:
            continue
        asked = _content_words(question.question)
        if not asked:
            continue
        gold = set().union(
            *(_content_words(bodies[chunk_id]) for chunk_id in question.expected_chunk_ids)
        )
        overlaps[question.question_id] = len(asked & gold) / len(asked)
    return overlaps


def _content_words(text: str) -> set[str]:
    """Lowercase, unaccent, drop punctuation, keep the long non-stopword tokens."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    cleaned = "".join(char if char.isalnum() else " " for char in stripped)
    return {
        token
        for token in cleaned.split()
        if len(token) >= _MIN_CONTENT_WORD_LENGTH and token not in _STOPWORDS
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_comparison(strategies: Sequence[Strategy], metrics: Sequence[StrategyMetrics]) -> None:
    """Print the headline comparison table."""
    print()
    print(_RULE)
    print("COMPARACIÓN DE ESTRATEGIAS — hit@k y MRR a nivel de unidad estructural")
    print(_RULE)
    print(f"{'':<3} {'pasajes':>8} {'hit@1':>8} {'hit@3':>8} {'hit@5':>8} {'MRR':>8}   estrategia")
    print("-" * 88)
    for strategy, metric in zip(strategies, metrics, strict=True):
        print(
            f"{strategy.key:<3} {len(strategy.passages):>8} "
            f"{metric.hit_at[1]:>8.3f} {metric.hit_at[3]:>8.3f} {metric.hit_at[5]:>8.3f} "
            f"{metric.mrr:>8.3f}   {strategy.name}"
        )
    print()
    print("Fragmento exacto (solo A y B: la línea base no tiene identificadores comparables)")
    print("-" * 88)
    for strategy, metric in zip(strategies, metrics, strict=True):
        if metric.exact_hit_at is None:
            print(f"{strategy.key:<3} {'—':>8} {'—':>8} {'—':>8}   {strategy.name}")
            continue
        print(
            f"{strategy.key:<3} {metric.exact_hit_at[1]:>8.3f} {metric.exact_hit_at[3]:>8.3f} "
            f"{metric.exact_hit_at[5]:>8.3f}   {strategy.name}"
        )
    print()
    print("Acertar el DOCUMENTO frente a acertar la UNIDAD dentro de él")
    print("-" * 88)
    print(f"{'':<3} {'doc@1':>8} {'doc@3':>8} {'doc@5':>8} {'uni@5':>8} {'brecha':>8}   estrategia")
    for strategy, metric in zip(strategies, metrics, strict=True):
        gap = metric.document_hit_at[5] - metric.hit_at[5]
        print(
            f"{strategy.key:<3} {metric.document_hit_at[1]:>8.3f} "
            f"{metric.document_hit_at[3]:>8.3f} {metric.document_hit_at[5]:>8.3f} "
            f"{metric.hit_at[5]:>8.3f} {gap:>8.3f}   {strategy.name}"
        )
    print()
    print("Cuánto texto y cuántas unidades entrega cada pasaje (confusor de la línea base)")
    print("-" * 88)
    for strategy, metric in zip(strategies, metrics, strict=True):
        print(
            f"{strategy.key:<3} unidades/pasaje {metric.units_per_passage:>5.2f} | "
            f"caracteres/pasaje {metric.chars_per_passage:>7.1f} | "
            f"unidades alcanzables en top-5 ≈ {5 * metric.units_per_passage:>5.2f}   "
            f"{strategy.name}"
        )


def print_paired_comparison(
    strategies: Sequence[Strategy],
    questions: Sequence[EvalQuestion],
    outcomes: Mapping[str, Sequence[QuestionOutcome]],
    cutoff: int = 5,
) -> None:
    """Compare strategies question by question, which is the only honest test at this size.

    With 26 answerable questions one question is 3.8 percentage points, so two hit rates
    that differ by a couple of points differ by one question. A paired count says how many
    questions each strategy won outright, and that is what a reader needs to judge whether
    a gap is a result or a coin flip.
    """
    answerable = [question.question_id for question in questions if question.answerable]
    print()
    print(_RULE)
    print(f"COMPARACIÓN PAREADA A NIVEL DE PREGUNTA (hit@{cutoff}, n={len(answerable)})")
    print(_RULE)
    ranks = {
        key: {outcome.question_id: outcome.unit_rank for outcome in strategy_outcomes}
        for key, strategy_outcomes in outcomes.items()
    }

    def hit(key: str, question_id: str) -> bool:
        rank = ranks[key][question_id]
        return 0 < rank <= cutoff

    for left in range(len(strategies)):
        for right in range(left + 1, len(strategies)):
            first, second = strategies[left].key, strategies[right].key
            only_first = [q for q in answerable if hit(first, q) and not hit(second, q)]
            only_second = [q for q in answerable if hit(second, q) and not hit(first, q)]
            both = sum(1 for q in answerable if hit(first, q) and hit(second, q))
            neither = sum(1 for q in answerable if not hit(first, q) and not hit(second, q))
            print(
                f"  {first} vs {second}: ambas {both} | ninguna {neither} | "
                f"solo {first} {len(only_first)} | solo {second} {len(only_second)}"
            )
            if only_first:
                print(f"      gana {first} en: {', '.join(only_first)}")
            if only_second:
                print(f"      gana {second} en: {', '.join(only_second)}")


def print_common_failures(
    strategies: Sequence[Strategy],
    questions: Sequence[EvalQuestion],
    outcomes: Mapping[str, Sequence[QuestionOutcome]],
    cutoff: int = 5,
) -> None:
    """List the questions no strategy answers, which is where the corpus is really weak.

    A question that all three miss is not evidence about chunking - every strategy had the
    same text available and none of them surfaced it. It is evidence about the retriever,
    or about the corpus, and separating those two piles is the point of printing it.
    """
    ranks = {
        key: {outcome.question_id: outcome.unit_rank for outcome in strategy_outcomes}
        for key, strategy_outcomes in outcomes.items()
    }
    missed = [
        question
        for question in questions
        if question.answerable
        and all(
            not 0 < ranks[strategy.key][question.question_id] <= cutoff for strategy in strategies
        )
    ]
    print()
    print(_RULE)
    print(f"PREGUNTAS QUE FALLAN EN LAS TRES ESTRATEGIAS (hit@{cutoff})")
    print(_RULE)
    print("Fallar en las tres no dice nada sobre el chunking: las tres tenían el mismo texto.")
    print("Dice algo sobre el recuperador o sobre el corpus.")
    print()
    for question in missed:
        best = min((ranks[strategy.key][question.question_id] or 99) for strategy in strategies)
        depth = (
            f"mejor puesto {best}" if best <= RANKING_DEPTH else f"fuera del top-{RANKING_DEPTH}"
        )
        print(f"  {question.question_id} [{','.join(question.tags)}] — {depth}")
        print(f"      {question.question}")
    print()
    print(f"  {len(missed)} de {sum(1 for q in questions if q.answerable)} preguntas.")


def print_homogenisation(homogenisation: Mapping[str, tuple[float, float]]) -> None:
    """Print how alike each strategy makes the passages of one same document."""
    print()
    print(_RULE)
    print("HOMOGENEIZACIÓN: ¿cuánto se parecen entre sí los pasajes de un mismo documento?")
    print(_RULE)
    print("Es el mecanismo detrás del costo del header: un bloque casi idéntico repetido en")
    print("cada chunk de un documento acerca sus vectores entre sí. Enrutar al documento")
    print("correcto se vuelve más fácil, y distinguir dos artículos dentro de él, más difícil.")
    print()
    print(f"{'':<3} {'intra-doc':>10} {'inter-doc':>10} {'separación':>11}")
    print("-" * 88)
    for key, (within, across) in homogenisation.items():
        print(f"{key:<3} {within:>10.4f} {across:>10.4f} {within - across:>11.4f}")


def print_scores(strategies: Sequence[Strategy], metrics: Sequence[StrategyMetrics]) -> None:
    """Print the similarity distributions, which is where the header's cost shows up."""
    print()
    print(_RULE)
    print("DISTRIBUCIÓN DEL SCORE DE SIMILITUD DEL PRIMER RESULTADO")
    print(_RULE)
    print(
        f"{'':<3} {'min':>8} {'p25':>8} {'mediana':>8} {'p75':>8} {'max':>8} {'rango':>8} "
        f"{'desv':>8}   estrategia"
    )
    print("-" * 88)
    for strategy, metric in zip(strategies, metrics, strict=True):
        summary = metric.score_summary
        span = summary["max"] - summary["min"]
        print(
            f"{strategy.key:<3} {summary['min']:>8.4f} {summary['p25']:>8.4f} "
            f"{summary['median']:>8.4f} {summary['p75']:>8.4f} {summary['max']:>8.4f} "
            f"{span:>8.4f} {summary['stdev']:>8.4f}   {strategy.name}"
        )
    print()
    print("Margen entre el primer y el quinto resultado (capacidad de discriminar)")
    print("-" * 88)
    for strategy, metric in zip(strategies, metrics, strict=True):
        margins = metric.margins
        print(
            f"{strategy.key:<3} media {statistics.fmean(margins):>7.4f} | "
            f"mediana {statistics.median(margins):>7.4f} | "
            f"min {min(margins):>7.4f} | max {max(margins):>7.4f}   {strategy.name}"
        )


def print_threshold_analysis(
    strategies: Sequence[Strategy], metrics: Sequence[StrategyMetrics]
) -> None:
    """Print what the unanswerable questions scored, which is where a cut-off would live."""
    print()
    print(_RULE)
    print("¿HAY UMBRAL DE CORTE PARA 'NO SÉ'?")
    print(_RULE)
    print("Compara el score del primer resultado en preguntas CON respuesta contra el de las")
    print("preguntas SIN respuesta. Un umbral es viable solo si las dos nubes se separan.")
    print()
    for strategy, metric in zip(strategies, metrics, strict=True):
        answerable = sorted(metric.hit_scores + metric.miss_scores)
        unanswerable = sorted(metric.unanswerable_scores)
        overlap = sum(1 for score in answerable if score <= max(unanswerable))
        print(f"  {strategy.key} — {strategy.name}")
        print(
            f"      con respuesta   : min {answerable[0]:.4f} | mediana "
            f"{statistics.median(answerable):.4f} | max {answerable[-1]:.4f}  (n={len(answerable)})"
        )
        print(
            f"      sin respuesta   : min {unanswerable[0]:.4f} | mediana "
            f"{statistics.median(unanswerable):.4f} | max {unanswerable[-1]:.4f}  "
            f"(n={len(unanswerable)})"
        )
        print(
            f"      separación      : el peor 'sin respuesta' llega a {max(unanswerable):.4f}, "
            f"por encima de {overlap} de las {len(answerable)} preguntas con respuesta"
        )
        print()


def print_contamination(
    overlaps: Mapping[str, float], questions: Sequence[EvalQuestion], chunks: Sequence[Chunk]
) -> None:
    """Print the lexical overlap between each question and its annotated answer."""
    document_of = {chunk.chunk_id: chunk.metadata.document_id for chunk in chunks}
    by_document: dict[str, list[float]] = {}
    for question in questions:
        if question.question_id not in overlaps:
            continue
        for chunk_id in question.expected_chunk_ids:
            by_document.setdefault(document_of[chunk_id], []).append(overlaps[question.question_id])
    print()
    print(_RULE)
    print("CONTROL DE CONTAMINACIÓN — solapamiento léxico pregunta / fragmento anotado")
    print(_RULE)
    print("Fracción de las palabras de contenido de la pregunta que aparecen en su fragmento")
    print("anotado. Alto = la pregunta se redactó copiando el vocabulario del chunk.")
    print()
    for document_id, values in sorted(by_document.items()):
        print(
            f"  {document_id:<40} media {statistics.fmean(values):.3f} | "
            f"mediana {statistics.median(values):.3f} | max {max(values):.3f}  (n={len(values)})"
        )
    print()
    worst = sorted(overlaps.items(), key=lambda item: -item[1])[:5]
    print("  Las cinco preguntas de mayor solapamiento:")
    for question_id, value in worst:
        print(f"    {question_id}  {value:.3f}")


def print_numeric_questions(
    strategies: Sequence[Strategy],
    questions: Sequence[EvalQuestion],
    outcomes: Mapping[str, Sequence[QuestionOutcome]],
) -> None:
    """Print the rank of the gold unit for the numeric questions, in every strategy."""
    numeric = [question for question in questions if "numerica" in question.tags]
    print()
    print(_RULE)
    print("PREGUNTAS NUMÉRICAS — el caso que falló en el turno anterior")
    print(_RULE)
    columns = "".join(f"{strategy.key:>5}" for strategy in strategies)
    print(f"{'':<5}{columns}   pregunta")
    print("-" * 88)
    for question in numeric:
        ranks = ""
        for strategy in strategies:
            outcome = next(
                o for o in outcomes[strategy.key] if o.question_id == question.question_id
            )
            ranks += f"{outcome.unit_rank if outcome.unit_rank else '—':>5}"
        print(f"{question.question_id:<5}{ranks}   {question.question[:55]}")
    print()
    print("  Rango del primer acierto en el top-10; '—' significa que no apareció.")


def write_per_question_csv(
    path: Path,
    strategies: Sequence[Strategy],
    questions: Sequence[EvalQuestion],
    outcomes: Mapping[str, Sequence[QuestionOutcome]],
    overlaps: Mapping[str, float],
) -> None:
    """Write the per-question detail that the console table cannot hold."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["question_id", "question", "answerable", "tags", "lexical_overlap"]
        for strategy in strategies:
            header += [f"{strategy.key}_unit_rank", f"{strategy.key}_top_score"]
        writer.writerow(header)
        for question in questions:
            row: list[Any] = [
                question.question_id,
                question.question,
                question.answerable,
                "|".join(question.tags),
                f"{overlaps.get(question.question_id, float('nan')):.4f}",
            ]
            for strategy in strategies:
                outcome = next(
                    o for o in outcomes[strategy.key] if o.question_id == question.question_id
                )
                row += [outcome.unit_rank, f"{outcome.top_score:.4f}"]
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Load the set, score the three strategies, print the tables and record the run.

    Returns:
        0 when the comparison completed and was recorded; 1 otherwise.
    """
    enable_unicode_console()
    print(_RULE)
    print("EVALUACIÓN DEL RETRIEVAL SOBRE EL CORPUS NORMATIVO")
    print(_RULE)

    try:
        documents = load_corpus(settings.corpus_dir)
        chunks = chunk_corpus(documents)
        questions = load_questions(settings.eval_dir / QUESTIONS_FILE)
        validate_annotations(questions, chunks)
    except (CorpusFormatError, FileNotFoundError, ValueError) as error:
        print(f"\nNo se pudo preparar la evaluación: {error}", file=sys.stderr)
        return 1

    gold_units = {
        question.question_id: frozenset(
            chunk_id.split("#", 1)[0] for chunk_id in question.expected_chunk_ids
        )
        for question in questions
    }
    gold_documents = {
        question.question_id: frozenset(
            chunk_id.split("::", 1)[0] for chunk_id in question.expected_chunk_ids
        )
        for question in questions
    }
    answerable = sum(1 for question in questions if question.answerable)
    print(f"Documentos            : {len(documents)}")
    print(
        f"Preguntas             : {len(questions)}  "
        f"({answerable} con respuesta, {len(questions) - answerable} sin respuesta)"
    )
    print(f"Modelo de embeddings  : {EMBEDDING_MODEL_NAME}")
    print("Búsqueda              : exacta (producto punto), no el índice HNSW de producción")

    try:
        model = EmbeddingModel()
    except Exception as error:  # noqa: BLE001 - the loader raises many unrelated types
        print(f"\nNo se pudo cargar el modelo de embeddings: {error}", file=sys.stderr)
        return 1

    strategies = (
        build_strategy_a(chunks),
        build_strategy_b(chunks),
        build_strategy_c(documents),
        build_strategy_d(chunks),
    )
    print()
    for strategy in strategies:
        print(f"  {strategy.key}: {strategy.name} — {len(strategy.passages)} pasajes")
        print(f"     {strategy.description}")

    outcomes = {
        strategy.key: evaluate_strategy(strategy, questions, gold_units, gold_documents, model)
        for strategy in strategies
    }
    metrics = [compute_metrics(strategy, outcomes[strategy.key]) for strategy in strategies]
    overlaps = measure_lexical_overlap(questions, chunks)
    homogenisation = {
        strategy.key: measure_homogenisation(strategy, model) for strategy in strategies
    }

    print_comparison(strategies, metrics)
    print_paired_comparison(strategies, questions, outcomes)
    print_homogenisation(homogenisation)
    print_scores(strategies, metrics)
    print_threshold_analysis(strategies, metrics)
    print_numeric_questions(strategies, questions, outcomes)
    print_common_failures(strategies, questions, outcomes)
    print_contamination(overlaps, questions, chunks)

    try:
        context = ensure_experiment(EXPERIMENT_NAME)
    except MLflowConfigurationError as error:
        print(f"\nNo se pudo registrar la comparación: {error}", file=sys.stderr)
        return 1

    _record(
        context.experiment_id, strategies, questions, outcomes, metrics, overlaps, homogenisation
    )
    print()
    print(_RULE)
    print("REGISTRADO EN MLFLOW")
    print(_RULE)
    print(f"Experimento : {context.name}")
    print(f"URL         : {context.url}")
    return 0


def _record(
    experiment_id: str,
    strategies: Sequence[Strategy],
    questions: Sequence[EvalQuestion],
    outcomes: Mapping[str, Sequence[QuestionOutcome]],
    metrics: Sequence[StrategyMetrics],
    overlaps: Mapping[str, float],
    homogenisation: Mapping[str, tuple[float, float]],
) -> None:
    """Log one MLflow run per strategy, plus the per-question detail as an artifact."""
    for strategy, metric in zip(strategies, metrics, strict=True):
        with mlflow.start_run(
            experiment_id=experiment_id, run_name=f"retrieval-strategy-{strategy.key}"
        ):
            mlflow.set_tags(
                {
                    "run_type": "retrieval-evaluation",
                    "strategy": strategy.key,
                    "strategy_name": strategy.name,
                }
            )
            mlflow.log_params(
                {
                    "embedding_model": EMBEDDING_MODEL_NAME,
                    "passages": len(strategy.passages),
                    "questions": len(questions),
                    "answerable_questions": metric.answerable,
                    "ranking_depth": RANKING_DEPTH,
                    "max_body_chars": DEFAULT_MAX_BODY_CHARS,
                    "overlap_chars": DEFAULT_OVERLAP_CHARS,
                    "fixed_window_chars": FIXED_WINDOW_CHARS,
                    "fixed_overlap_chars": FIXED_OVERLAP_CHARS,
                    "search": "exact-dot-product",
                }
            )
            recorded = {f"hit_at_{k}": metric.hit_at[k] for k in CUTOFFS}
            recorded["mrr"] = metric.mrr
            recorded["top1_score_median"] = metric.score_summary["median"]
            recorded["top1_score_stdev"] = metric.score_summary["stdev"]
            recorded["top1_score_span"] = metric.score_summary["max"] - metric.score_summary["min"]
            recorded["margin_top1_top5_mean"] = statistics.fmean(metric.margins)
            recorded["unanswerable_top1_max"] = max(metric.unanswerable_scores)
            recorded["lexical_overlap_mean"] = statistics.fmean(overlaps.values())
            recorded.update({f"document_hit_at_{k}": metric.document_hit_at[k] for k in CUTOFFS})
            within, across = homogenisation[strategy.key]
            recorded["similarity_within_document"] = within
            recorded["similarity_across_documents"] = across
            recorded["units_per_passage"] = metric.units_per_passage
            recorded["chars_per_passage"] = metric.chars_per_passage
            if metric.exact_hit_at is not None:
                recorded.update({f"exact_hit_at_{k}": metric.exact_hit_at[k] for k in CUTOFFS})
            mlflow.log_metrics(recorded)

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "per_question_results.csv"
                write_per_question_csv(path, strategies, questions, outcomes, overlaps)
                mlflow.log_artifact(str(path))


if __name__ == "__main__":
    raise SystemExit(main())
