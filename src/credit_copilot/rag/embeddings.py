"""The embedding model, and the two reasons this particular one was chosen.

**Why multilingual is not optional here.** The corpus is mostly Spanish - a chapter of the
Colombian banking regulator's accounting circular, a Colombian statute, and an internal credit
policy - and one document is not. The Basel Committee's *Principles for the Management of
Credit Risk* has no official Spanish version on bis.org, so translating it would have produced
a text that reads well and cannot be cited. It is therefore indexed in English, and an analyst
asking *"¿qué exige Basilea sobre el sistema de calificación interna?"* has to reach an English
paragraph. That is a cross-lingual retrieval problem, and only a model with a shared
multilingual vector space can solve it.

**Why the 512-token window is the deciding property, not the accuracy leaderboard.** The
obvious multilingual sentence-transformers baselines - the `paraphrase-multilingual-*` family -
have a **128-token** window. Under a 128-token encoder the size of a chunk stops being a
decision of the chunking strategy and becomes a property of the tokeniser: any article longer
than roughly four hundred characters would be truncated, silently, at index time. That is
precisely the failure `chunking.py` was written to prevent, and no retrieval score compensates
for indexing the first third of an article and believing it indexed the whole. `multilingual-e5`
carries 512 tokens, which is what lets a structural unit be embedded whole.

`intfloat/multilingual-e5-base` is the chosen size: 768 dimensions, trained for retrieval on
a hundred languages. The `-small` variant would halve the download and the dimension; the
`-large` would roughly triple the download. Neither has been measured here, and the retrieval
evaluation of the next turn is where that comparison belongs.

**Why queries and passages get different prefixes.** E5 is trained asymmetrically: a document
is encoded as `passage: ...` and a question as `query: ...`. Dropping the prefixes, or using
the same one on both sides, degrades retrieval on a model that was trained to expect them.
The prefixes are applied here, inside the one object that owns the model, so no caller can
forget them - a caller that embedded a query as a passage would get quietly worse results and
no error.
"""

from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

__all__ = [
    "EMBEDDING_MODEL_NAME",
    "PASSAGE_PREFIX",
    "QUERY_PREFIX",
    "EmbeddingModel",
]

EMBEDDING_MODEL_NAME: Final[str] = "intfloat/multilingual-e5-base"
"""Identifier of the embedding model on the Hugging Face hub."""

QUERY_PREFIX: Final[str] = "query: "
"""Prefix E5 expects on the question side of an asymmetric search."""

PASSAGE_PREFIX: Final[str] = "passage: "
"""Prefix E5 expects on the document side of an asymmetric search."""

_DEFAULT_BATCH_SIZE: Final[int] = 16
"""Passages encoded per forward pass. Small enough to run on a laptop CPU."""


class EmbeddingModel:
    """The project's embedding model, with the query/passage asymmetry built in.

    Loading the model downloads it on first use and keeps it in memory afterwards, so a
    caller should build one instance and reuse it rather than constructing one per call.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        """Load the sentence-transformers model.

        Args:
            model_name: Identifier on the Hugging Face hub. Defaults to the project's model;
                overriding it is how the next turn compares alternatives.
        """
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def name(self) -> str:
        """Identifier of the loaded model.

        Returns:
            The model name this instance was built with.
        """
        return self._model_name

    @property
    def dimension(self) -> int:
        """Length of the vectors the model produces.

        Returns:
            The embedding dimension.

        Raises:
            RuntimeError: The loaded model does not declare one, so nothing downstream can
                assume a vector width.
        """
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"{self._model_name} does not declare an embedding dimension.")
        return int(dimension)

    @property
    def max_sequence_length(self) -> int:
        """Longest input the model reads before truncating.

        Returns:
            The maximum sequence length, in tokens.

        Raises:
            RuntimeError: The loaded model does not declare one. Chunk sizes are chosen
                against this number, so guessing it would defeat the point of measuring.
        """
        window = self._model.max_seq_length
        if window is None:
            raise RuntimeError(f"{self._model_name} does not declare a maximum sequence length.")
        return int(window)

    def embed_passages(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Embed corpus chunks, on the passage side of the asymmetry.

        Args:
            texts: Chunk texts, each already carrying its context header.

        Returns:
            One unit-length vector per input, as rows in the same order.
        """
        return self._encode([f"{PASSAGE_PREFIX}{text}" for text in texts])

    def embed_queries(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Embed questions, on the query side of the asymmetry.

        Args:
            texts: Questions as a user would write them.

        Returns:
            One unit-length vector per input, as rows in the same order.
        """
        return self._encode([f"{QUERY_PREFIX}{text}" for text in texts])

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Count the tokens each passage costs, prefix included.

        This is what turns "the chunks fit in the window" from an assumption into a
        measurement. `scripts/build_rag_index.py` reports it and flags anything truncated.

        Args:
            texts: Chunk texts, without the passage prefix.

        Returns:
            Token counts, in the same order as the inputs.
        """
        tokenizer = self._model.tokenizer
        encoded = tokenizer(
            [f"{PASSAGE_PREFIX}{text}" for text in texts],
            add_special_tokens=True,
            truncation=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def _encode(self, prefixed: Sequence[str]) -> npt.NDArray[np.float32]:
        """Run the model and return unit-length vectors as one float32 matrix."""
        vectors = self._model.encode(
            list(prefixed),
            batch_size=_DEFAULT_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
