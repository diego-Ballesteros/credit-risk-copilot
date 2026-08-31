"""Tests of the chunking strategy, against its three principles and against the real corpus.

Two families of test live here and they check different things.

The **hand-built** tests use small documents written inside the test, where the expected
result is worked out from the definition rather than read off the implementation. They can
disagree with the code, which is the only reason a test is worth writing. In particular the
subdivision tests build a unit that is deliberately too long, because no document of the real
corpus may be long enough to exercise that path today - and a guarantee that is never
exercised is a guarantee nobody is checking.

The **corpus** tests run over `data/corpus/` as it actually is. They are what catch a document
that was edited into a shape the strategy cannot handle, and the synthetic-policy test is one
of them on purpose: the claim being defended is about the file on disk, not about a fixture
that resembles it.

None of these tests loads an embedding model. `chunking` depends only on `documents`, so the
whole strategy is testable without a download, and that layering is deliberate.
"""

import re

import pytest

from credit_copilot.config import settings
from credit_copilot.rag.chunking import (
    DEFAULT_MAX_BODY_CHARS,
    Chunk,
    chunk_corpus,
    chunk_document,
)
from credit_copilot.rag.documents import CorpusFormatError, load_corpus, parse_document

SYNTHETIC_DOCUMENT_ID = "politica-interna-credito"
DEROGATED_DOCUMENT_ID = "circular-basica-contable-sfc-cap-ii"

FRONT_MATTER = """---
document_id: doc-de-prueba
title: Documento de prueba
issuer: Emisor de prueba
citation_prefix: Documento de prueba
language: es
is_synthetic: false
source_url: https://example.invalid/fuente
retrieved_at: 2026-08-30
---
"""


def write_document(tmp_path, body: str, front_matter: str = FRONT_MATTER):
    """Write a corpus file into a temporary directory and parse it back."""
    path = tmp_path / "doc-de-prueba.md"
    path.write_text(front_matter + body, encoding="utf-8")
    return parse_document(path)


@pytest.fixture(scope="module")
def corpus_chunks() -> tuple[Chunk, ...]:
    """Every chunk of the real corpus, built with the default parameters."""
    return chunk_corpus(load_corpus(settings.corpus_dir))


# ---------------------------------------------------------------------------
# Principle 1: a chunk never cuts a structural unit in half
# ---------------------------------------------------------------------------


def test_every_short_unit_becomes_exactly_one_chunk(tmp_path):
    document = write_document(
        tmp_path,
        """
## Capítulo I

### Artículo 1

Texto del primer artículo.

### Artículo 2

Texto del segundo artículo.
""",
    )
    chunks = chunk_document(document)

    assert len(chunks) == 2
    assert [chunk.body for chunk in chunks] == [
        "Texto del primer artículo.",
        "Texto del segundo artículo.",
    ]
    assert all(chunk.metadata.part_count == 1 for chunk in chunks)


def test_no_corpus_chunk_mixes_two_structural_units(corpus_chunks):
    """A chunk belongs to exactly one unit, and the unit's own text is never merged."""
    for chunk in corpus_chunks:
        assert chunk.metadata.location, chunk.chunk_id
        assert chunk.metadata.unit_id.startswith(chunk.metadata.document_id)


def test_the_units_of_a_document_partition_its_text(tmp_path):
    """Concatenating the bodies of the chunks returns every unit's text exactly once."""
    document = write_document(
        tmp_path,
        """
## Título

### Artículo 1

Primero.

#### Literal a

Segundo.

### Artículo 2

Tercero.
""",
    )
    chunks = chunk_document(document)

    assert [chunk.body for chunk in chunks] == ["Primero.", "Segundo.", "Tercero."]


def test_a_heading_without_text_produces_no_chunk_but_stays_in_the_path(tmp_path):
    """A pure container contributes ancestry and nothing else."""
    document = write_document(
        tmp_path,
        """
## Título I

### Artículo 1

Texto.
""",
    )
    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].metadata.location == "Título I > Artículo 1"


def test_body_above_the_first_heading_is_refused(tmp_path):
    """Text belonging to no unit is a malformed document, not a chunk without a location."""
    with pytest.raises(CorpusFormatError, match="above the first heading"):
        write_document(tmp_path, "\nTexto huérfano.\n\n## Título\n\nTexto.\n")


# ---------------------------------------------------------------------------
# Principle 2, as ADR-0008 rewrote it: the header is stored and shown, never encoded;
# the integrity warnings are encoded
# ---------------------------------------------------------------------------


def test_the_context_header_never_reaches_the_encoded_text(corpus_chunks):
    """The measured reason for the whole change: the header cost hit@k and bought nothing."""
    for chunk in corpus_chunks:
        assert chunk.metadata.document_title not in chunk.embed_text, chunk.chunk_id
        assert chunk.metadata.location not in chunk.embed_text, chunk.chunk_id
        assert "Documento:" not in chunk.embed_text, chunk.chunk_id
        assert "Ubicación:" not in chunk.embed_text, chunk.chunk_id


def test_the_context_header_is_kept_in_the_metadata_and_in_the_shown_text(corpus_chunks):
    """Taking it out of the vector must not lose it: a result still has to name its source."""
    for chunk in corpus_chunks:
        assert chunk.metadata.document_title in chunk.context_header, chunk.chunk_id
        assert chunk.metadata.issuer in chunk.context_header, chunk.chunk_id
        assert chunk.metadata.location in chunk.context_header, chunk.chunk_id
        assert chunk.metadata.context_header == chunk.context_header, chunk.chunk_id
        assert chunk.display_text == f"{chunk.context_header}\n\n{chunk.body}", chunk.chunk_id
        assert chunk.display_text.startswith("Documento: "), chunk.chunk_id


def test_the_encoded_text_is_the_warnings_plus_the_body_and_nothing_else(corpus_chunks):
    for chunk in corpus_chunks:
        expected = (
            f"{chunk.integrity_notice}\n\n{chunk.body}"
            if chunk.integrity_notice
            else chunk.body
        )
        assert chunk.embed_text == expected, chunk.chunk_id
        assert chunk.body in chunk.embed_text, chunk.chunk_id


def test_a_status_note_is_context_and_stays_out_of_the_vector(corpus_chunks):
    """A note saying a law is in force is context; only warnings are encoded."""
    for chunk in corpus_chunks:
        note = chunk.metadata.status_note
        if note:
            assert note in chunk.context_header, chunk.chunk_id
            assert note not in chunk.embed_text, chunk.chunk_id


def test_the_body_alone_carries_neither_header_nor_warning(tmp_path):
    document = write_document(tmp_path, "\n## Título\n\n### Artículo 1\n\nTexto.\n")
    chunk = chunk_document(document)[0]

    assert chunk.body == "Texto."
    assert "Documento:" not in chunk.body
    assert "AVISO:" not in chunk.body


# ---------------------------------------------------------------------------
# Principle 3: a subchunk keeps the header of its parent unit
# ---------------------------------------------------------------------------


def test_an_over_long_unit_is_subdivided_at_a_paragraph_boundary(tmp_path):
    paragraph = "Palabra " * 40
    document = write_document(
        tmp_path,
        f"\n## Título\n\n### Artículo largo\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}\n",
    )
    chunks = chunk_document(document, max_body_chars=700)

    assert len(chunks) > 1
    assert all(len(chunk.body) <= 700 for chunk in chunks)
    assert all(chunk.body == chunk.body.strip() for chunk in chunks)
    assert all(chunk.body.startswith("Palabra") for chunk in chunks)


def test_every_subchunk_repeats_the_header_of_its_parent_unit(tmp_path):
    paragraph = "Palabra " * 40
    document = write_document(
        tmp_path,
        f"\n## Título\n\n### Artículo largo\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}\n",
    )
    chunks = chunk_document(document, max_body_chars=700)

    assert len(chunks) > 1
    for index, chunk in enumerate(chunks, start=1):
        assert chunk.metadata.location == "Título > Artículo largo"
        assert "Título > Artículo largo" in chunk.context_header
        assert chunk.metadata.document_title in chunk.context_header
        assert f"Fragmento {index} de {len(chunks)}" in chunk.context_header
        assert chunk.metadata.part_index == index
        assert chunk.metadata.part_count == len(chunks)


def test_every_subchunk_of_a_warned_document_repeats_the_warning_in_its_vector(tmp_path):
    """A part that lost the warning would be the orphan the warning exists to prevent."""
    front_matter = FRONT_MATTER.replace(
        "retrieved_at: 2026-08-30",
        "retrieved_at: 2026-08-30\nintegrity_notice: Capítulo DEROGADO; no es norma vigente.",
    )
    paragraph = "Palabra " * 40
    document = write_document(
        tmp_path,
        f"\n## Título\n\n### Artículo largo\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}\n",
        front_matter=front_matter,
    )
    chunks = chunk_document(document, max_body_chars=700)

    assert len(chunks) > 1
    for chunk in chunks:
        assert "AVISO: Capítulo DEROGADO; no es norma vigente." in chunk.embed_text
        assert chunk.embed_text.startswith("AVISO: ")


def test_the_parts_of_a_subdivided_unit_share_its_unit_id_and_differ_in_chunk_id(tmp_path):
    paragraph = "Palabra " * 40
    document = write_document(
        tmp_path,
        f"\n## Título\n\n### Artículo largo\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}\n",
    )
    chunks = chunk_document(document, max_body_chars=700)

    assert len({chunk.metadata.unit_id for chunk in chunks}) == 1
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_a_single_over_long_paragraph_is_cut_at_a_sentence_end(tmp_path):
    sentences = " ".join(f"Esta es la oración número {n} del párrafo." for n in range(40))
    document = write_document(tmp_path, f"\n## Título\n\n### Artículo\n\n{sentences}\n")
    chunks = chunk_document(document, max_body_chars=600)

    assert len(chunks) > 1
    assert all(len(chunk.body) <= 600 for chunk in chunks)
    assert all(chunk.body.endswith(".") for chunk in chunks)


def test_parameters_that_would_break_subdivision_are_refused(tmp_path):
    document = write_document(tmp_path, "\n## Título\n\n### Artículo\n\nTexto.\n")

    with pytest.raises(ValueError, match="below"):
        chunk_document(document, max_body_chars=10)
    with pytest.raises(ValueError, match="negative"):
        chunk_document(document, overlap_chars=-1)
    with pytest.raises(ValueError, match="would not advance"):
        chunk_document(document, max_body_chars=500, overlap_chars=500)


# ---------------------------------------------------------------------------
# The index metadata and the embedded text must not contradict each other
# ---------------------------------------------------------------------------


def test_the_citation_stored_in_the_index_is_derivable_from_the_shown_text(corpus_chunks):
    """Two copies of a fact diverge unless something forbids it. This forbids it."""
    for chunk in corpus_chunks:
        citation = chunk.metadata.citation
        assert citation.endswith(chunk.metadata.location), chunk.chunk_id
        assert chunk.metadata.location in chunk.display_text, chunk.chunk_id


def test_the_synthetic_flag_in_the_index_agrees_with_the_warning_in_the_vector(corpus_chunks):
    for chunk in corpus_chunks:
        if chunk.metadata.is_synthetic:
            assert chunk.metadata.synthetic_notice in chunk.embed_text, chunk.chunk_id
        else:
            notice = chunk.metadata.synthetic_notice
            assert notice is None or notice not in chunk.embed_text, chunk.chunk_id


def test_the_character_counts_stored_in_the_index_are_the_real_ones(corpus_chunks):
    for chunk in corpus_chunks:
        assert chunk.metadata.body_chars == len(chunk.body), chunk.chunk_id
        assert chunk.metadata.embed_chars == len(chunk.embed_text), chunk.chunk_id
        assert chunk.metadata.display_chars == len(chunk.display_text), chunk.chunk_id


def test_index_metadata_holds_only_scalars_and_drops_the_absent_fields(corpus_chunks):
    for chunk in corpus_chunks:
        flat = chunk.metadata.to_index_metadata()
        assert all(isinstance(value, str | int | bool) for value in flat.values())
        assert "None" not in flat.values()
        assert flat["document_id"] == chunk.metadata.document_id
        assert flat["citation"] == chunk.metadata.citation


def test_chunk_ids_are_unique_across_the_whole_corpus(corpus_chunks):
    identifiers = [chunk.chunk_id for chunk in corpus_chunks]

    assert len(set(identifiers)) == len(identifiers)


# ---------------------------------------------------------------------------
# The synthetic policy says so in its own text, in every fragment
# ---------------------------------------------------------------------------


def test_every_chunk_of_the_synthetic_policy_declares_itself_synthetic(corpus_chunks):
    """A README nobody reads cannot warn anybody, and neither can a metadata field.

    ADR-0008 decision 3: this is the warning that survives the header's removal, and it has
    to be inside the text a retriever encodes and returns, not beside it.
    """
    policy_chunks = [
        chunk for chunk in corpus_chunks if chunk.metadata.document_id == SYNTHETIC_DOCUMENT_ID
    ]

    assert policy_chunks, "the synthetic policy produced no chunk"
    for chunk in policy_chunks:
        assert chunk.metadata.is_synthetic, chunk.chunk_id
        for text in (chunk.embed_text, chunk.display_text):
            assert re.search(r"SINT[ÉE]TICO", text, flags=re.IGNORECASE), chunk.chunk_id
            assert "no representa la política de ninguna entidad" in text.lower()


def test_every_chunk_of_the_derogated_chapter_says_so_inside_its_vector(corpus_chunks):
    """The other integrity warning: a derogated chapter quoted as law in force."""
    chapter_chunks = [
        chunk
        for chunk in corpus_chunks
        if chunk.metadata.document_id == DEROGATED_DOCUMENT_ID
    ]

    assert chapter_chunks, "the derogated chapter produced no chunk"
    for chunk in chapter_chunks:
        for text in (chunk.embed_text, chunk.display_text):
            assert "DEROGADO" in text, chunk.chunk_id


def test_no_chunk_of_a_real_document_claims_to_be_synthetic(corpus_chunks):
    """The negative half of the previous test, which is the half that can rot silently."""
    for chunk in corpus_chunks:
        if chunk.metadata.document_id == SYNTHETIC_DOCUMENT_ID:
            continue
        assert not chunk.metadata.is_synthetic, chunk.chunk_id
        assert "SINTÉTICO" not in chunk.embed_text, chunk.chunk_id


def test_a_document_without_warnings_encodes_its_body_and_nothing_more(corpus_chunks):
    """No warning must mean no overhead: the vector of a clean document is its text."""
    clean = [
        chunk
        for chunk in corpus_chunks
        if chunk.metadata.document_id not in {SYNTHETIC_DOCUMENT_ID, DEROGATED_DOCUMENT_ID}
    ]

    assert clean, "no document without warnings in the corpus"
    for chunk in clean:
        assert chunk.embed_text == chunk.body, chunk.chunk_id
        assert not chunk.has_integrity_notice, chunk.chunk_id


def test_a_synthetic_document_without_its_notice_is_refused(tmp_path):
    front_matter = FRONT_MATTER.replace("is_synthetic: false", "is_synthetic: true")

    with pytest.raises(CorpusFormatError, match="synthetic_notice"):
        write_document(tmp_path, "\n## Título\n\nTexto.\n", front_matter=front_matter)


# ---------------------------------------------------------------------------
# The real corpus, as a whole
# ---------------------------------------------------------------------------


def test_the_corpus_holds_the_four_expected_documents():
    documents = load_corpus(settings.corpus_dir)

    assert {document.document_id for document in documents} == {
        "basilea-principios-riesgo-credito",
        "circular-basica-contable-sfc-cap-ii",
        "ley-1266-2008-habeas-data",
        SYNTHETIC_DOCUMENT_ID,
    }
    assert sum(document.metadata.is_synthetic for document in documents) == 1


def test_every_corpus_chunk_fits_the_configured_body_limit(corpus_chunks):
    for chunk in corpus_chunks:
        assert chunk.metadata.body_chars <= DEFAULT_MAX_BODY_CHARS, chunk.chunk_id


def test_every_non_synthetic_document_declares_where_its_text_came_from():
    for document in load_corpus(settings.corpus_dir):
        if document.metadata.is_synthetic:
            continue
        assert document.metadata.source_url, document.document_id
        assert document.metadata.retrieved_at, document.document_id
