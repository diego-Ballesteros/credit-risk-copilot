"""What a corpus document is, and how its structural units are read off the file.

**Why the corpus files carry their own structure instead of being plain text.** The chunking
strategy of this project cuts on the structural unit of the document - the article, the
numeral, the literal, the principle - and never on a character count. That is only possible
if the structure survives the trip from the source into the repository. Storing the corpus as
undifferentiated prose would throw away the one thing the source already provides for free
and force the chunker to guess it back with a regular expression, which is exactly the
failure the strategy exists to avoid.

So the corpus files are markdown, and **a heading is a structural unit**. `##` is the top
unit of the document (a chapter, a title, a principle), `###` the article or numeral inside
it, `####` the literal or paragraph inside that. A unit owns the text that follows its
heading up to the next heading of any level; the headings above it are its ancestry, and that
ancestry is what a citation is made of.

**Why the metadata lives in front matter and not in a side table.** A document that does not
carry its own provenance stops being citable the moment it is copied. Every file states what
it is, who issued it, where it came from, when it was retrieved, and - the field that matters
most here - whether it is **synthetic**. The internal credit policy of this corpus was
written for the project and represents no real institution; `is_synthetic` is what carries
that fact into every fragment the retriever ever returns, so nobody has to have read a README
for the warning to arrive.

**Why a malformed file raises instead of being tolerated.** An unknown front-matter key, a
missing required field, or body text sitting above the first heading are all cases where the
file means something the reader intended and the parser cannot see. Silently ignoring them
produces a corpus that indexes fine and answers wrong. The project rule is that an
undocumented category fails loudly, and a corpus document is a category of data like any
other.
"""

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "CORPUS_FILE_SUFFIX",
    "CorpusFormatError",
    "DocumentMetadata",
    "DocumentUnit",
    "SourceDocument",
    "load_corpus",
    "parse_document",
]

CORPUS_FILE_SUFFIX: Final[str] = ".md"
"""Extension of a corpus document. Anything else under the corpus directory is ignored."""

_FRONT_MATTER_FENCE: Final[str] = "---"
"""Line that opens and closes the front-matter block, as in the usual markdown convention."""

_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "document_id",
        "title",
        "issuer",
        "citation_prefix",
        "language",
        "is_synthetic",
        "retrieved_at",
    }
)
"""Front-matter keys every corpus document must declare."""

_OPTIONAL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "source_url",
        "status_note",
        "scope_note",
        "synthetic_notice",
    }
)
"""Front-matter keys a document may declare. Any other key is an error, not a comment."""

_MIN_HEADING_LEVEL: Final[int] = 2
"""Body headings start at `##`. Level 1 is reserved for the title, which lives in front matter."""

_SLUG_MAX_LENGTH: Final[int] = 40
"""Cap on the readable part of a unit identifier. Identifiers are read by humans debugging."""


class CorpusFormatError(ValueError):
    """A corpus file does not satisfy the format this module requires."""


@dataclass(frozen=True)
class DocumentMetadata:
    """Provenance of one corpus document, as the file itself declares it.

    Attributes:
        document_id: Stable identifier, also the filter key of the vector store.
        title: Full title of the document.
        issuer: Body that issued it, or the statement that it is synthetic.
        citation_prefix: Leading text of any citation to this document.
        language: ISO 639-1 code of the language the body is written in.
        is_synthetic: Whether the document was written for this project rather than issued.
        retrieved_at: Date the text was taken from its source, as `YYYY-MM-DD`.
        source_url: Where the text was taken from. Required unless the document is synthetic.
        status_note: Short statement of validity, embedded into every chunk of the document.
        scope_note: What was transcribed and what was left out. Index metadata only.
        synthetic_notice: The warning carried into every chunk. Required when synthetic.
    """

    document_id: str
    title: str
    issuer: str
    citation_prefix: str
    language: str
    is_synthetic: bool
    retrieved_at: str
    source_url: str | None = None
    status_note: str | None = None
    scope_note: str | None = None
    synthetic_notice: str | None = None


@dataclass(frozen=True)
class DocumentUnit:
    """One structural unit of a document: a heading and the text it owns.

    Attributes:
        unit_id: Identifier unique within the corpus.
        heading_path: Headings from the outermost down to this unit's own, in order.
        body: Text between this heading and the next one, with surrounding blanks stripped.
    """

    unit_id: str
    heading_path: tuple[str, ...]
    body: str

    @property
    def location(self) -> str:
        """Human-readable position of the unit inside its document.

        Returns:
            The heading path joined with ` > `, which is what a citation names.
        """
        return " > ".join(self.heading_path)


@dataclass(frozen=True)
class SourceDocument:
    """A corpus document: its provenance and the structural units it contains.

    Attributes:
        metadata: Provenance declared in the file's front matter.
        units: Units carrying text, in document order. A heading with no text of its own
            is not a unit; it only contributes to the ancestry of the units below it.
    """

    metadata: DocumentMetadata
    units: tuple[DocumentUnit, ...]

    @property
    def document_id(self) -> str:
        """Identifier of the document.

        Returns:
            The `document_id` declared in front matter.
        """
        return self.metadata.document_id

    def citation_for(self, unit: DocumentUnit) -> str:
        """Build the citation a reader would write for one unit.

        Args:
            unit: A unit of this document.

        Returns:
            The document's citation prefix followed by the unit's position in it.
        """
        return f"{self.metadata.citation_prefix}, {unit.location}"


def parse_document(path: Path) -> SourceDocument:
    """Read one corpus file into its metadata and its structural units.

    Args:
        path: Path to a markdown corpus file with front matter.

    Returns:
        The parsed document.

    Raises:
        CorpusFormatError: The front matter is missing, incomplete, carries an unknown key,
            or the body holds text above its first heading.
    """
    raw = path.read_text(encoding="utf-8")
    front_matter, body = _split_front_matter(raw, path)
    metadata = _build_metadata(front_matter, path)
    units = _parse_units(body, metadata.document_id, path)
    if not units:
        raise CorpusFormatError(f"{path.name}: the document declares no unit with text.")
    return SourceDocument(metadata=metadata, units=units)


def load_corpus(corpus_dir: Path) -> tuple[SourceDocument, ...]:
    """Read every corpus document under a directory.

    Files are read in sorted order so that two runs over the same directory produce the same
    identifiers, which is what makes the index rebuild reproducible.

    Args:
        corpus_dir: Directory holding the corpus files.

    Returns:
        The parsed documents, ordered by file name.

    Raises:
        FileNotFoundError: The directory does not exist.
        CorpusFormatError: The directory holds no corpus file, or one of them is malformed.
    """
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    paths = sorted(corpus_dir.glob(f"*{CORPUS_FILE_SUFFIX}"))
    if not paths:
        raise CorpusFormatError(f"No `*{CORPUS_FILE_SUFFIX}` document under {corpus_dir}")
    documents = tuple(parse_document(path) for path in paths)
    _reject_duplicate_ids(documents)
    return documents


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------


def _split_front_matter(raw: str, path: Path) -> tuple[Mapping[str, str], str]:
    """Separate the front-matter block from the body of a corpus file."""
    lines = raw.splitlines()
    if not lines or lines[0].strip() != _FRONT_MATTER_FENCE:
        raise CorpusFormatError(f"{path.name}: the file must open with a `---` front-matter fence.")
    try:
        closing = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == _FRONT_MATTER_FENCE
        )
    except StopIteration:
        raise CorpusFormatError(f"{path.name}: the front-matter block is never closed.") from None
    return _parse_key_values(lines[1:closing], path), "\n".join(lines[closing + 1 :])


def _parse_key_values(lines: Sequence[str], path: Path) -> Mapping[str, str]:
    """Read `key: value` lines. One key per line; a value may itself contain colons."""
    values: dict[str, str] = {}
    for number, line in enumerate(lines, start=2):
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise CorpusFormatError(f"{path.name}: front-matter line {number} is not `key: value`.")
        key = key.strip()
        if key in values:
            raise CorpusFormatError(f"{path.name}: front-matter key `{key}` is declared twice.")
        values[key] = value.strip()
    return values


def _build_metadata(front_matter: Mapping[str, str], path: Path) -> DocumentMetadata:
    """Validate the front matter and turn it into a `DocumentMetadata`."""
    keys = set(front_matter)
    unknown = keys - _REQUIRED_KEYS - _OPTIONAL_KEYS
    if unknown:
        raise CorpusFormatError(f"{path.name}: unknown front-matter key(s): {sorted(unknown)}")
    missing = _REQUIRED_KEYS - keys
    if missing:
        raise CorpusFormatError(f"{path.name}: missing front-matter key(s): {sorted(missing)}")

    is_synthetic = _parse_bool(front_matter["is_synthetic"], path)
    optional = {key: front_matter.get(key) or None for key in _OPTIONAL_KEYS}

    if is_synthetic and not optional["synthetic_notice"]:
        raise CorpusFormatError(
            f"{path.name}: a synthetic document must declare `synthetic_notice`, because that "
            f"is the text every retrieved fragment carries."
        )
    if not is_synthetic and not optional["source_url"]:
        raise CorpusFormatError(
            f"{path.name}: a non-synthetic document must declare `source_url`, or its text "
            f"cannot be traced back to the body that issued it."
        )

    return DocumentMetadata(
        document_id=front_matter["document_id"],
        title=front_matter["title"],
        issuer=front_matter["issuer"],
        citation_prefix=front_matter["citation_prefix"],
        language=front_matter["language"],
        is_synthetic=is_synthetic,
        retrieved_at=front_matter["retrieved_at"],
        **optional,
    )


def _parse_bool(value: str, path: Path) -> bool:
    """Read a boolean front-matter value, refusing anything ambiguous."""
    normalised = value.strip().lower()
    if normalised == "true":
        return True
    if normalised == "false":
        return False
    raise CorpusFormatError(f"{path.name}: `{value}` is not a boolean; write `true` or `false`.")


# ---------------------------------------------------------------------------
# Structural units
# ---------------------------------------------------------------------------


def _parse_units(body: str, document_id: str, path: Path) -> tuple[DocumentUnit, ...]:
    """Walk the headings of a document body and collect the units that carry text."""
    units: list[DocumentUnit] = []
    ancestry: list[tuple[int, str]] = []
    pending: list[str] = []
    index = 0

    def flush() -> None:
        nonlocal index
        text = "\n".join(pending).strip()
        pending.clear()
        if not text or not ancestry:
            return
        heading_path = tuple(heading for _, heading in ancestry)
        slug = _slugify(heading_path[-1])
        units.append(
            DocumentUnit(
                unit_id=f"{document_id}::{index:03d}-{slug}",
                heading_path=heading_path,
                body=text,
            )
        )
        index += 1

    for line in body.splitlines():
        level, heading = _read_heading(line)
        if level is None:
            if not ancestry and line.strip():
                raise CorpusFormatError(
                    f"{path.name}: text appears above the first heading. Every line of a corpus "
                    f"document must belong to a structural unit."
                )
            pending.append(line)
            continue
        if level < _MIN_HEADING_LEVEL:
            raise CorpusFormatError(
                f"{path.name}: heading `{heading}` uses level {level}. The document title lives "
                f"in front matter, so body headings start at level {_MIN_HEADING_LEVEL}."
            )
        flush()
        while ancestry and ancestry[-1][0] >= level:
            ancestry.pop()
        ancestry.append((level, heading))
    flush()
    return tuple(units)


def _read_heading(line: str) -> tuple[int | None, str]:
    """Read an ATX markdown heading, returning its level and text."""
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None, ""
    level = len(stripped) - len(stripped.lstrip("#"))
    remainder = stripped[level:]
    if remainder and not remainder.startswith(" "):
        return None, ""
    return level, remainder.strip()


def _slugify(text: str) -> str:
    """Reduce a heading to an ascii fragment usable inside an identifier."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(char for char in decomposed if not unicodedata.combining(char))
    kept = [char.lower() if char.isalnum() else "-" for char in ascii_only]
    slug = "".join(kept).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:_SLUG_MAX_LENGTH].strip("-") or "unit"


def _reject_duplicate_ids(documents: Iterable[SourceDocument]) -> None:
    """Refuse a corpus where two files claim the same `document_id`."""
    seen: set[str] = set()
    for document in documents:
        if document.document_id in seen:
            raise CorpusFormatError(
                f"`{document.document_id}` is declared by more than one corpus file. The "
                f"identifier is the filter key of the vector store and must be unique."
            )
        seen.add(document.document_id)
