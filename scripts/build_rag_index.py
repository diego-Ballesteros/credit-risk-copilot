"""Build the vector index over the normative corpus, from scratch, and report what it built.

Run it with::

    uv run python scripts/build_rag_index.py

**Idempotent by construction, not by care.** The collection is dropped before it is written,
so the index is a pure function of `data/corpus/` and of the two chunking parameters. Running
this twice leaves the same store; running it after a document changed leaves a store with no
trace of the old shape. Nothing here appends.

**Why it reports the token length and not only the character length.** Chunk sizes are chosen
against the encoder's 512-token window, and text past that window is truncated **silently** -
the index builds, the run reports success, and the tail of an article is simply not in the
vector. The only way that failure ever surfaces is if somebody measures it, so this script
measures it and says out loud how much headroom is left.

Exit code 0 when the index was built, 1 when the corpus is malformed or the embedding model
cannot be loaded.
"""

import sys
from collections.abc import Sequence
from typing import Final

from credit_copilot.config import settings
from credit_copilot.console import enable_unicode_console
from credit_copilot.rag.chunking import (
    DEFAULT_MAX_BODY_CHARS,
    DEFAULT_OVERLAP_CHARS,
    Chunk,
    chunk_corpus,
    group_by_document,
    summarise_lengths,
)
from credit_copilot.rag.documents import CorpusFormatError, SourceDocument, load_corpus
from credit_copilot.rag.embeddings import EmbeddingModel
from credit_copilot.rag.vectorstore import VectorStore

_HISTOGRAM_BUCKETS: Final[tuple[int, ...]] = (200, 400, 600, 800, 1000, 1200, 1400)
"""Upper edges, in characters, of the chunk-length histogram. The last bucket is open."""

_HISTOGRAM_WIDTH: Final[int] = 40
"""Longest bar drawn, in characters."""

_RULE: Final[str] = "=" * 78


def main() -> int:
    """Read the corpus, chunk it, embed it, index it, and report the result.

    Returns:
        0 when the index was built; 1 when the corpus or the model could not be loaded.
    """
    enable_unicode_console()
    corpus_dir = settings.corpus_dir
    print(_RULE)
    print("BUILDING THE RAG INDEX OVER THE NORMATIVE CORPUS")
    print(_RULE)
    print(f"Corpus directory : {corpus_dir}")
    print(f"Vector store     : {settings.vector_store_dir}")
    print(f"max_body_chars   : {DEFAULT_MAX_BODY_CHARS}")
    print(f"overlap_chars    : {DEFAULT_OVERLAP_CHARS}")

    try:
        documents = load_corpus(corpus_dir)
    except (CorpusFormatError, FileNotFoundError) as error:
        print(f"\nThe corpus could not be read: {error}", file=sys.stderr)
        return 1

    chunks = chunk_corpus(documents, DEFAULT_MAX_BODY_CHARS, DEFAULT_OVERLAP_CHARS)
    _report_documents(documents, chunks)
    _report_length_distribution(chunks)

    try:
        model = EmbeddingModel()
    except Exception as error:  # noqa: BLE001 - the loader raises many unrelated types
        print(f"\nThe embedding model could not be loaded: {error}", file=sys.stderr)
        return 1

    _report_model(model, chunks)

    store = VectorStore(settings.vector_store_dir, model)
    written = store.build(chunks)

    print()
    print(_RULE)
    print("INDEX BUILT")
    print(_RULE)
    print(f"Collection       : {store.collection_name}")
    print(f"Chunks written   : {written}")
    print(f"Chunks in store  : {store.count()}")
    return 0


def _report_documents(documents: Sequence[SourceDocument], chunks: Sequence[Chunk]) -> None:
    """Print one line per document: its units, its chunks and what kind of source it is."""
    grouped = group_by_document(chunks)
    print()
    print(f"Documents        : {len(documents)}")
    print(f"Chunks           : {len(chunks)}")
    print()
    print(f"{'document_id':<40} {'lang':>5} {'units':>7} {'chunks':>7}  source")
    print("-" * 78)
    for document in documents:
        document_chunks = grouped.get(document.document_id, [])
        kind = "SYNTHETIC" if document.metadata.is_synthetic else "issued"
        print(
            f"{document.document_id:<40} {document.metadata.language:>5} "
            f"{len(document.units):>7} {len(document_chunks):>7}  {kind}"
        )
    subdivided = sorted(
        {chunk.metadata.unit_id for chunk in chunks if chunk.metadata.part_count > 1}
    )
    print()
    print(f"Units subdivided : {len(subdivided)}")
    for unit_id in subdivided:
        parts = [chunk for chunk in chunks if chunk.metadata.unit_id == unit_id]
        print(f"  {unit_id} -> {len(parts)} parts | {parts[0].metadata.location}")


def _report_length_distribution(chunks: Sequence[Chunk]) -> None:
    """Print the summary statistics and a histogram of finished chunk lengths."""
    summary = summarise_lengths(chunks)
    print()
    print("Encoded length in characters (integrity warnings + body; header NOT encoded)")
    print("-" * 78)
    print(
        f"  min {summary['min']:>6.0f} | p25 {summary['p25']:>6.0f} | "
        f"median {summary['median']:>6.0f} | p75 {summary['p75']:>6.0f} | "
        f"max {summary['max']:>6.0f} | mean {summary['mean']:>7.1f}"
    )
    print()
    warned = sum(1 for chunk in chunks if chunk.has_integrity_notice)
    print(f"  Chunks con aviso de integridad incrustado: {warned} de {len(chunks)}")
    print()
    counts = _histogram([chunk.metadata.embed_chars for chunk in chunks])
    widest = max(counts.values()) or 1
    edges = [*_HISTOGRAM_BUCKETS, None]
    for edge, count in zip(edges, counts.values(), strict=True):
        label = f"<{edge:>5}" if edge is not None else f">={_HISTOGRAM_BUCKETS[-1]:>4}"
        bar = "#" * round(_HISTOGRAM_WIDTH * count / widest)
        print(f"  {label} chars | {count:>3} {bar}")


def _histogram(values: Sequence[int]) -> dict[str, int]:
    """Count values into the fixed buckets, with an open bucket at the top."""
    counts = {f"<{edge}": 0 for edge in _HISTOGRAM_BUCKETS}
    counts[f">={_HISTOGRAM_BUCKETS[-1]}"] = 0
    for value in values:
        for edge in _HISTOGRAM_BUCKETS:
            if value < edge:
                counts[f"<{edge}"] += 1
                break
        else:
            counts[f">={_HISTOGRAM_BUCKETS[-1]}"] += 1
    return counts


def _report_model(model: EmbeddingModel, chunks: Sequence[Chunk]) -> None:
    """Print the model, its dimension, and how close the chunks come to its window."""
    tokens = model.count_tokens([chunk.embed_text for chunk in chunks])
    window = model.max_sequence_length
    over = [(chunk, count) for chunk, count in zip(chunks, tokens, strict=True) if count > window]
    print()
    print("Embedding model")
    print("-" * 78)
    print(f"  name           : {model.name}")
    print(f"  dimension      : {model.dimension}")
    print(f"  window         : {window} tokens")
    print(
        f"  chunk tokens   : min {min(tokens)} | median {sorted(tokens)[len(tokens) // 2]} "
        f"| max {max(tokens)}"
    )
    print(f"  headroom       : {window - max(tokens)} tokens below the window at the worst chunk")
    if over:
        print(f"  TRUNCATED      : {len(over)} chunk(s) exceed the window and lose their tail:")
        for chunk, count in over:
            print(f"    {chunk.chunk_id} ({count} tokens) | {chunk.metadata.location}")
    else:
        print("  TRUNCATED      : none. Every chunk is embedded whole.")


if __name__ == "__main__":
    raise SystemExit(main())
