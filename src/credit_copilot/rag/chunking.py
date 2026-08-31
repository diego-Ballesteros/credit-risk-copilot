"""The chunking strategy: structural units, context embedded in the text, safe subdivision.

Three principles, in this order. They are not independent: the second only pays off because
the first holds, and the third exists so the first can be broken without losing the second.

**1. The boundary of a chunk is a structural unit of the document, never a character count.**
Regulation arrives already segmented into chapters, articles, numerals and literals. That
segmentation is not decoration: it is the unit of meaning its drafters chose, and it is the
unit an analyst cites. Cutting every N characters destroys a structure the document already
provides and produces fragments that begin mid-sentence and belong to nothing. The one thing
a fixed-size cut buys - uniform chunks - is worth less than the thing it costs.

**2. The metadata is embedded in the text of the chunk, not only in the index.** A fragment
reading *"el porcentaje mínimo será del 20%"* produces an embedding that does not encode what
it is about; nothing in those six words says credit, or provisioning, or which rule. The same
fragment preceded by its document, its chapter and its article produces a vector that encodes
the context, so a question phrased in the vocabulary of the domain can reach it. The second
payoff is free: the citation is already inside the text the model reads, so an agent that
quotes the fragment quotes its source with it, rather than reconstructing the reference from
a metadata field it may or may not consult.

**3. When a unit exceeds the maximum size it is subdivided, and every subchunk keeps the
header of its parent unit.** Subdivision is the exception, not the rule, and it is a
concession to the encoder's context window rather than a strategy in its own right. A
subchunk that lost its parent heading would be exactly the orphan fragment principle 1 was
written to avoid, so the header is repeated verbatim on each part and the part is numbered
inside its unit.

---

WHAT IS EMBEDDED IN THE TEXT AND WHAT IS ONLY IN THE INDEX
----------------------------------------------------------

The separating question is: **does a reader who sees only this fragment need it to understand
what the fragment says?** If yes, it goes in the text and is embedded with it. If it is a
handle for filtering, tracing or displaying, it goes in the index metadata, where it can be
read without polluting the vector.

Embedded in the text: the document title and issuer, the validity note, the synthetic
warning, the full structural path, and the part number when the unit was subdivided. Each one
changes how the body is to be read. The synthetic warning is the clearest case - a policy
fragment that does not say it is synthetic is a fragment that will be quoted as if it were
real, and a note in the README does not travel with it.

Index metadata only: the chunk and unit identifiers, the source URL, the retrieval date, the
scope note, and the character counts. An identifier and a URL tokenise into noise and carry
no meaning a question could match; the scope note describes **our extraction**, which is a
fact about this repository rather than about the norm.

Everything embedded is also stored in the index, because a consumer needs to filter on the
document and display the citation without parsing the text back. That duplication is the
reason `tests/test_chunking.py` asserts the two cannot contradict each other: two copies of a
fact diverge unless something forbids it.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from credit_copilot.rag.documents import DocumentUnit, SourceDocument

__all__ = [
    "DEFAULT_MAX_BODY_CHARS",
    "DEFAULT_OVERLAP_CHARS",
    "Chunk",
    "ChunkMetadata",
    "chunk_corpus",
    "chunk_document",
    "group_by_document",
    "summarise_lengths",
]

DEFAULT_MAX_BODY_CHARS: Final[int] = 1200
"""Longest body a chunk may carry before the unit is subdivided.

Chosen against the encoder, not by taste. The embedding model of `embeddings.py` has a
512-token window, and a token of Spanish prose costs roughly three and a half characters
under its tokeniser, so the window holds on the order of 1,700 characters. The context header
of principle 2 spends a few hundred of those on every chunk, and text beyond the window is
**silently truncated** rather than rejected - the worst kind of failure, because the index
builds cleanly and the tail of the article is simply not in the vector. 1,200 characters of
body leaves the header its room with margin to spare.

It bounds the **body**, not the finished chunk: the header is a fixed overhead the strategy
adds, and charging the body for it would make the usable size depend on how long the
document's title happens to be. `scripts/build_rag_index.py` reports the token length of
every finished chunk, so the margin is measured and not assumed.
"""

DEFAULT_OVERLAP_CHARS: Final[int] = 0
"""Characters repeated from the end of one subchunk at the start of the next.

Zero by default, which is a claim and not a default chosen by omission. Overlap exists to
rescue a fact that a cut separated from the context that makes it readable; here subdivision
only ever cuts at a paragraph boundary, and every part carries the full header of its parent
unit, so the context overlap normally rescues is already present in each part. Paying for it
would mean indexing the same sentences twice and letting one unit occupy several of the top-k
slots with near-duplicate text.

It is a parameter rather than a constant because that claim is measurable and has not been
measured. The retrieval evaluation of the next turn can raise it and report what it buys.
"""

_MIN_MAX_BODY_CHARS: Final[int] = 200
"""Floor on `max_body_chars`. Below this, subdivision cuts inside ordinary sentences."""

_SENTENCE_TERMINATORS: Final[tuple[str, ...]] = (". ", "? ", "! ", ".\n", "?\n", "!\n")
"""Sentence ends used as the second-choice cut, when a single paragraph is itself too long."""


@dataclass(frozen=True)
class ChunkMetadata:
    """Everything the index stores about a chunk, embedded in the text or not.

    Attributes:
        chunk_id: Identifier unique in the corpus; the primary key of the vector store.
        unit_id: Identifier of the structural unit this chunk came from. Shared by the
            parts of a subdivided unit, which is how they are recognised as siblings.
        document_id: Identifier of the source document, and the key search filters on.
        document_title: Full title of the document.
        issuer: Body that issued the document.
        citation: Citation a reader would write for this fragment.
        location: Heading path inside the document, joined with ` > `.
        language: ISO 639-1 code of the language of the body.
        is_synthetic: Whether the document was written for this project rather than issued.
        part_index: Position of this chunk inside its unit, starting at 1.
        part_count: Number of parts the unit was split into. 1 when it was not split.
        body_chars: Length of the body, without the context header.
        text_chars: Length of the finished chunk, header included.
        source_url: Where the document was taken from, when it has a source.
        retrieved_at: Date the document was taken from its source.
        status_note: Short statement of validity. Embedded in the text when present.
        scope_note: What was transcribed and what was left out. Index metadata only.
        synthetic_notice: Warning carried in the text of every chunk of a synthetic document.
    """

    chunk_id: str
    unit_id: str
    document_id: str
    document_title: str
    issuer: str
    citation: str
    location: str
    language: str
    is_synthetic: bool
    part_index: int
    part_count: int
    body_chars: int
    text_chars: int
    source_url: str | None = None
    retrieved_at: str | None = None
    status_note: str | None = None
    scope_note: str | None = None
    synthetic_notice: str | None = None

    def to_index_metadata(self) -> dict[str, str | int | bool]:
        """Flatten to the scalar types a vector store accepts.

        Chroma stores strings, numbers and booleans, so `None` is dropped rather than
        written as the string `"None"` - a filter comparing against a literal `"None"` is a
        bug waiting to be written.

        Returns:
            Every set field, as a scalar.
        """
        flat: dict[str, str | int | bool] = {}
        for key, value in vars(self).items():
            if value is not None:
                flat[key] = value
        return flat


@dataclass(frozen=True)
class Chunk:
    """One indexable fragment: its embedded text, its body alone, and its metadata.

    `text` is what gets embedded and what a retriever returns; `header` and `body` are kept
    apart so a test, a report or a user interface can look at either half without splitting
    the string back on a separator.

    Attributes:
        header: Context block prepended to the body, per principle 2.
        body: The document text of the fragment, with no context added.
        text: Header and body joined. The string that is embedded.
        metadata: Everything the index stores about this chunk.
    """

    header: str
    body: str
    text: str
    metadata: ChunkMetadata

    @property
    def chunk_id(self) -> str:
        """Identifier of the chunk.

        Returns:
            The `chunk_id` of the metadata.
        """
        return self.metadata.chunk_id


def chunk_document(
    document: SourceDocument,
    max_body_chars: int = DEFAULT_MAX_BODY_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> tuple[Chunk, ...]:
    """Turn one document into chunks, one per structural unit unless a unit is too long.

    Args:
        document: A parsed corpus document.
        max_body_chars: Longest body a chunk may carry before its unit is subdivided.
        overlap_chars: Characters repeated between consecutive parts of a subdivided unit.

    Returns:
        The chunks of the document, in document order.

    Raises:
        ValueError: `max_body_chars` is below the floor, or `overlap_chars` is negative or
            not smaller than `max_body_chars`, which would make subdivision fail to advance.
    """
    _validate_parameters(max_body_chars, overlap_chars)
    chunks: list[Chunk] = []
    for unit in document.units:
        parts = _split_body(unit.body, max_body_chars, overlap_chars)
        chunks.extend(
            _build_chunk(document, unit, part, index, len(parts))
            for index, part in enumerate(parts, start=1)
        )
    return tuple(chunks)


def chunk_corpus(
    documents: Iterable[SourceDocument],
    max_body_chars: int = DEFAULT_MAX_BODY_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> tuple[Chunk, ...]:
    """Chunk every document of a corpus with one set of parameters.

    Args:
        documents: Parsed corpus documents.
        max_body_chars: Longest body a chunk may carry before its unit is subdivided.
        overlap_chars: Characters repeated between consecutive parts of a subdivided unit.

    Returns:
        The chunks of every document, in corpus order.
    """
    return tuple(
        chunk
        for document in documents
        for chunk in chunk_document(document, max_body_chars, overlap_chars)
    )


def group_by_document(chunks: Iterable[Chunk]) -> dict[str, list[Chunk]]:
    """Group chunks by the document they came from, preserving order within each group.

    Args:
        chunks: Chunks to group.

    Returns:
        A mapping from document identifier to its chunks.
    """
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.metadata.document_id, []).append(chunk)
    return grouped


# ---------------------------------------------------------------------------
# Building one chunk
# ---------------------------------------------------------------------------


def _build_chunk(
    document: SourceDocument,
    unit: DocumentUnit,
    body: str,
    part_index: int,
    part_count: int,
) -> Chunk:
    """Assemble the header, the body and the metadata of a single chunk."""
    metadata = document.metadata
    header = _build_header(document, unit, part_index, part_count)
    text = f"{header}\n\n{body}"
    suffix = f"#{part_index:02d}" if part_count > 1 else ""
    return Chunk(
        header=header,
        body=body,
        text=text,
        metadata=ChunkMetadata(
            chunk_id=f"{unit.unit_id}{suffix}",
            unit_id=unit.unit_id,
            document_id=metadata.document_id,
            document_title=metadata.title,
            issuer=metadata.issuer,
            citation=document.citation_for(unit),
            location=unit.location,
            language=metadata.language,
            is_synthetic=metadata.is_synthetic,
            part_index=part_index,
            part_count=part_count,
            body_chars=len(body),
            text_chars=len(text),
            source_url=metadata.source_url,
            retrieved_at=metadata.retrieved_at,
            status_note=metadata.status_note,
            scope_note=metadata.scope_note,
            synthetic_notice=metadata.synthetic_notice,
        ),
    )


def _build_header(
    document: SourceDocument,
    unit: DocumentUnit,
    part_index: int,
    part_count: int,
) -> str:
    """Write the context block that principle 2 puts in front of every body.

    The order is deliberate. What the document *is* comes first, because it conditions
    everything after it; the synthetic warning comes second so it cannot be read as a
    footnote to the location; the location comes last, closest to the text it locates.
    """
    metadata = document.metadata
    lines = [f"Documento: {metadata.title} — {metadata.issuer}."]
    if metadata.is_synthetic and metadata.synthetic_notice:
        lines.append(f"Aviso: {metadata.synthetic_notice}")
    if metadata.status_note:
        lines.append(f"Estado: {metadata.status_note}")
    lines.append(f"Ubicación: {unit.location}.")
    if part_count > 1:
        lines.append(f"Fragmento {part_index} de {part_count} de esta unidad.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subdivision, when a unit does not fit
# ---------------------------------------------------------------------------


def _validate_parameters(max_body_chars: int, overlap_chars: int) -> None:
    """Refuse parameter combinations that would silently produce a bad index."""
    if max_body_chars < _MIN_MAX_BODY_CHARS:
        raise ValueError(
            f"max_body_chars={max_body_chars} is below {_MIN_MAX_BODY_CHARS}; below that, "
            f"subdivision has to cut inside ordinary sentences."
        )
    if overlap_chars < 0:
        raise ValueError(f"overlap_chars={overlap_chars} is negative.")
    if overlap_chars >= max_body_chars:
        raise ValueError(
            f"overlap_chars={overlap_chars} is not smaller than max_body_chars="
            f"{max_body_chars}; subdivision would not advance."
        )


def _split_body(body: str, max_body_chars: int, overlap_chars: int) -> tuple[str, ...]:
    """Split a unit body into parts that fit, cutting at the most natural boundary available.

    Preference order: keep the whole unit; cut between paragraphs; cut between sentences;
    cut on a character. The last case only happens for a single sentence longer than the
    limit, which no document of this corpus contains but which must not raise if one ever
    does.
    """
    if len(body) <= max_body_chars:
        return (body,)

    parts: list[str] = []
    current = ""
    for paragraph in _paragraphs(body):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_body_chars:
            current = candidate
            continue
        if current:
            parts.append(current)
            current = ""
        if len(paragraph) <= max_body_chars:
            current = _with_overlap(paragraph, parts, overlap_chars, max_body_chars)
            continue
        pieces = _split_long_paragraph(paragraph, max_body_chars, overlap_chars)
        parts.extend(pieces[:-1])
        current = pieces[-1]
    if current:
        parts.append(current)
    return tuple(parts)


def _paragraphs(body: str) -> list[str]:
    """Split a body on blank lines, dropping the empties."""
    return [block.strip() for block in body.split("\n\n") if block.strip()]


def _split_long_paragraph(
    paragraph: str, max_body_chars: int, overlap_chars: int
) -> list[str]:
    """Split one over-long paragraph at sentence ends, falling back to a character cut."""
    pieces: list[str] = []
    remaining = paragraph
    while len(remaining) > max_body_chars:
        cut = _last_sentence_end(remaining, max_body_chars)
        pieces.append(remaining[:cut].strip())
        remaining = _prefix_overlap(pieces[-1], overlap_chars) + remaining[cut:].lstrip()
    pieces.append(remaining.strip())
    return pieces


def _last_sentence_end(text: str, limit: int) -> int:
    """Find the latest sentence end at or before `limit`, or `limit` itself if there is none."""
    window = text[:limit]
    best = max((window.rfind(terminator) for terminator in _SENTENCE_TERMINATORS), default=-1)
    return best + 2 if best > 0 else limit


def _with_overlap(
    paragraph: str, parts: Sequence[str], overlap_chars: int, max_body_chars: int
) -> str:
    """Open a new part with the tail of the previous one, when overlap is enabled."""
    tail = _prefix_overlap(parts[-1], overlap_chars) if parts else ""
    candidate = tail + paragraph
    return candidate if len(candidate) <= max_body_chars else paragraph


def _prefix_overlap(previous: str, overlap_chars: int) -> str:
    """Take the tail of the previous part that opens the next one."""
    if overlap_chars <= 0:
        return ""
    return previous[-overlap_chars:] + "\n\n"


def summarise_lengths(chunks: Iterable[Chunk]) -> Mapping[str, float]:
    """Describe the distribution of finished chunk lengths, in characters.

    Args:
        chunks: Chunks to describe.

    Returns:
        Count, minimum, maximum, mean and the quartiles of `text_chars`. An empty input
        gives a count of zero and nothing else, rather than a division by zero.
    """
    lengths = sorted(chunk.metadata.text_chars for chunk in chunks)
    if not lengths:
        return {"count": 0}
    return {
        "count": len(lengths),
        "min": lengths[0],
        "p25": _percentile(lengths, 0.25),
        "median": _percentile(lengths, 0.50),
        "p75": _percentile(lengths, 0.75),
        "max": lengths[-1],
        "mean": sum(lengths) / len(lengths),
    }


def _percentile(sorted_values: Sequence[int], fraction: float) -> float:
    """Nearest-rank percentile of an already sorted sequence."""
    index = max(0, min(len(sorted_values) - 1, round(fraction * (len(sorted_values) - 1))))
    return float(sorted_values[index])
