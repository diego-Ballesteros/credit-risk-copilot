"""The persistent vector store: indexing, similarity search, and filtering by document.

**Why the embeddings are computed here and handed to Chroma, rather than configured into it.**
Chroma will happily own an embedding function and call it on every write and every query. That
is convenient and it puts the choice of model inside the database, where it becomes invisible:
the model would be a connection parameter instead of a project decision, and the `query:` /
`passage:` asymmetry that `embeddings.py` exists to enforce would have nowhere to live. So
this module treats Chroma as what it is - an index over vectors somebody else produced - and
`EmbeddingModel` stays the single place that knows how text becomes a vector.

**Why the collection is dropped and rebuilt instead of upserted.** `build` is meant to be
idempotent in the strong sense: running it twice over the same corpus leaves the same index,
and running it after a chunk was deleted from a document leaves an index without that chunk.
An upsert gives the first property and not the second - stale chunks from a previous shape of
the corpus survive, and they survive invisibly, because nothing in the store says which run
wrote them. Dropping the collection makes the index a pure function of the corpus.

**Why search can filter by source document.** The agent's two questions are not the same
question. *"¿Qué exige la norma sobre la evaluación de la capacidad de pago?"* must not be
answered out of the synthetic internal policy, and *"¿en qué banda cae un score de 0,19?"* has
no answer in the public regulation at all. A retriever that cannot be told where to look forces
the caller to filter after the fact, on a top-k that may already be full of the wrong document.

**Why what is encoded is not what is stored.** ADR-0008 took the context header out of the
vector and left it beside it, so a chunk has two texts: `embed_text`, which becomes the
vector, and `display_text`, which is what a person reads. This module encodes the first and
stores the second, so a retrieved result arrives already carrying its document, its location
and its citation while none of that was ever charged to the encoder. Storing `embed_text`
instead would show the reader a fragment stripped of its source; encoding `display_text`
would silently undo the ADR and nothing would fail.

**On the score.** The collection is created with cosine distance, so Chroma returns
`1 - cosine_similarity`. `SearchResult.score` reports the similarity, because a number that
grows as the match improves is the one a report can be read against without a footnote.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import chromadb
from chromadb.api.models.Collection import Collection

from credit_copilot.rag.chunking import Chunk
from credit_copilot.rag.embeddings import EmbeddingModel

__all__ = [
    "DEFAULT_COLLECTION_NAME",
    "SearchResult",
    "VectorStore",
]

DEFAULT_COLLECTION_NAME: Final[str] = "credit_risk_corpus"
"""Name of the collection holding the normative corpus."""

_DISTANCE_SPACE: Final[str] = "cosine"
"""Distance the collection is built with. E5 vectors are normalised, so cosine is the match."""

_ADD_BATCH_SIZE: Final[int] = 128
"""Chunks written per call. Keeps a single request well inside Chroma's payload limits."""


@dataclass(frozen=True)
class SearchResult:
    """One hit of a similarity search.

    Attributes:
        chunk_id: Identifier of the retrieved chunk.
        text: The chunk exactly as indexed, context header included.
        metadata: Everything the index stores about the chunk.
        score: Cosine similarity to the query, in [-1, 1]; higher is closer.
        distance: The cosine distance Chroma returned, kept so the conversion is auditable.
    """

    chunk_id: str
    text: str
    metadata: Mapping[str, Any]
    score: float
    distance: float

    @property
    def citation(self) -> str:
        """Citation of the retrieved fragment.

        Returns:
            The stored citation, or the chunk identifier if the index predates the field.
        """
        return str(self.metadata.get("citation", self.chunk_id))

    @property
    def is_synthetic(self) -> bool:
        """Whether the fragment comes from the synthetic internal policy.

        Returns:
            True when the source document is synthetic.
        """
        return bool(self.metadata.get("is_synthetic", False))


class VectorStore:
    """A persistent Chroma collection over the chunked corpus."""

    def __init__(
        self,
        persist_dir: Path,
        embedding_model: EmbeddingModel,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Open, creating it if needed, the on-disk store.

        Args:
            persist_dir: Directory Chroma writes its database into.
            embedding_model: Model used for both indexing and querying. Indexing with one
                model and querying with another compares vectors from different spaces, so
                the store holds exactly one.
            collection_name: Name of the collection inside the database.
        """
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._client = chromadb.PersistentClient(path=str(persist_dir))

    @property
    def collection_name(self) -> str:
        """Name of the collection this store reads and writes.

        Returns:
            The collection name.
        """
        return self._collection_name

    @property
    def persist_dir(self) -> Path:
        """Directory the database lives in.

        Returns:
            The persistence directory.
        """
        return self._persist_dir

    def build(self, chunks: Sequence[Chunk]) -> int:
        """Rebuild the collection from scratch over the given chunks.

        Any previous collection of the same name is dropped first, so the resulting index
        depends only on the chunks passed in and never on what a previous run left behind.

        Args:
            chunks: Chunks to index.

        Returns:
            The number of chunks written.

        Raises:
            ValueError: No chunks were given. An empty index is never what a caller meant,
                and it fails later, at query time, far from its cause.
        """
        if not chunks:
            raise ValueError("Refusing to build an empty index: no chunks were given.")
        self._drop_collection()
        collection = self._create_collection()
        for start in range(0, len(chunks), _ADD_BATCH_SIZE):
            batch = chunks[start : start + _ADD_BATCH_SIZE]
            collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.display_text for chunk in batch],
                embeddings=self._embedding_model.embed_passages(
                    [chunk.embed_text for chunk in batch]
                ),
                metadatas=[chunk.metadata.to_index_metadata() for chunk in batch],
            )
        return len(chunks)

    def count(self) -> int:
        """Count the chunks currently indexed.

        Returns:
            The number of records in the collection, or 0 if it does not exist yet.
        """
        collection = self._get_collection()
        return 0 if collection is None else collection.count()

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Iterable[str] | None = None,
    ) -> tuple[SearchResult, ...]:
        """Retrieve the chunks most similar to a question.

        Args:
            query: The question, written as a user would write it. It is embedded on the
                query side of the model's asymmetry.
            top_k: Maximum number of results to return.
            document_ids: Restrict the search to these source documents. `None` searches the
                whole corpus; a single identifier and a list behave the same way.

        Returns:
            Hits ordered from most to least similar, at most `top_k` of them.

        Raises:
            LookupError: The collection does not exist. The index has not been built.
            ValueError: `top_k` is not positive.
        """
        if top_k <= 0:
            raise ValueError(f"top_k={top_k} is not positive.")
        collection = self._get_collection()
        if collection is None:
            raise LookupError(
                f"Collection `{self._collection_name}` does not exist in {self._persist_dir}. "
                f"Run `scripts/build_rag_index.py` first."
            )
        response = collection.query(
            query_embeddings=self._embedding_model.embed_queries([query]),
            n_results=top_k,
            where=_document_filter(document_ids),
            include=["documents", "metadatas", "distances"],
        )
        return _to_results(response)

    # -----------------------------------------------------------------------
    # Collection handling
    # -----------------------------------------------------------------------

    def _create_collection(self) -> Collection:
        """Create the collection with the project's distance metric."""
        return self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": _DISTANCE_SPACE},
        )

    def _get_collection(self) -> Collection | None:
        """Return the collection, or `None` when it has not been created yet."""
        try:
            return self._client.get_collection(name=self._collection_name)
        except Exception:  # noqa: BLE001 - Chroma raises different types across versions
            return None

    def _drop_collection(self) -> None:
        """Delete the collection if it exists, so a rebuild starts from nothing."""
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:  # noqa: BLE001 - absence is the normal case on a first build
            return


def _document_filter(document_ids: Iterable[str] | None) -> dict[str, Any] | None:
    """Translate a document restriction into a Chroma `where` clause."""
    if document_ids is None:
        return None
    identifiers = list(document_ids)
    if not identifiers:
        raise ValueError(
            "document_ids is empty. Pass None to search the whole corpus; an empty list "
            "would silently match nothing."
        )
    if len(identifiers) == 1:
        return {"document_id": identifiers[0]}
    return {"document_id": {"$in": identifiers}}


def _to_results(response: Mapping[str, Any]) -> tuple[SearchResult, ...]:
    """Turn Chroma's column-oriented answer into ordered results."""
    ids = _first_row(response, "ids")
    documents = _first_row(response, "documents")
    metadatas = _first_row(response, "metadatas")
    distances = _first_row(response, "distances")
    return tuple(
        SearchResult(
            chunk_id=str(chunk_id),
            text=str(document),
            metadata=dict(metadata or {}),
            score=1.0 - float(distance),
            distance=float(distance),
        )
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        )
    )


def _first_row(response: Mapping[str, Any], key: str) -> list[Any]:
    """Read one column of a single-query Chroma response."""
    rows = response.get(key)
    if not rows:
        return []
    return list(rows[0])
