"""Tests of the two things the agent evaluation rests on: the contrast, and the set.

**The contrast.** `scripts/evaluate_agent.py` claims that its baseline differs from the agent
in exactly one factor. That claim is what makes the comparison mean anything, and it is a
property of two strings, so it is checked here rather than promised in a docstring. The
prompts are composed from three shared blocks precisely so this test can be written.

**The set.** Every annotation in `data/eval/agent_queries.yaml` is a claim about the world
that can rot: a band that does not match the arithmetic, a tool name that no longer exists, a
question annotated as unanswerable whose answer somebody later added to the corpus. Each of
those is checked against the thing it describes instead of being trusted. The band annotations
in particular are re-derived from the number written in the query text, so the set cannot
disagree with `resolve_band` without a test failing.

Nothing here calls a language model or loads the production artefact. The evaluation itself
costs money and takes an hour; these are the checks that have to pass before spending either.
"""

import re
from pathlib import Path

import pytest
import yaml

from credit_copilot.agent.prompts import (
    _ROLE,
    _RULES,
    _WITH_TOOLS,
    _WITHOUT_TOOLS,
    BARE_BASELINE_SYSTEM_PROMPT,
    BASELINE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from credit_copilot.agent.tools import TOOL_SPECS, resolve_band
from credit_copilot.config import settings
from credit_copilot.models.registry import PREDICTOR_COLUMNS
from credit_copilot.rag.chunking import chunk_corpus
from credit_copilot.rag.documents import load_corpus

TOOL_NAMES = {spec.name for spec in TOOL_SPECS}

# Terms whose absence from the corpus is what makes a question unanswerable. Annotating
# "no answer" is a claim about the corpus, so it is checked against the corpus.
ABSENT_TERMS = {
    "a14": ("usura", "interes bancario corriente"),
    # a16 is a harder case than it looks, and this list records why. The corpus DOES talk
    # about provisions in general - Basel paragraph 33, and the circular's opening - so the
    # absence that makes the question unanswerable is narrower: no percentage tied to a
    # rating category, which is what the analyst is actually asking for.
    "a16": ("porcentaje de provision", "provision individual", "provision general"),
}


@pytest.fixture(scope="module")
def eval_set() -> list[dict]:
    """The annotated agent queries, parsed straight from the YAML."""
    path = settings.eval_dir / "agent_queries.yaml"
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return list(payload["queries"])


@pytest.fixture(scope="module")
def corpus_text() -> str:
    """Every chunk of the real corpus, lowercased and unaccented, as one string."""
    chunks = chunk_corpus(load_corpus(settings.corpus_dir))
    joined = " ".join(chunk.display_text for chunk in chunks).lower()
    for accented, plain in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u")):
        joined = joined.replace(accented, plain)
    return joined


# ---------------------------------------------------------------------------
# The contrast differs in one factor, and it is this one
# ---------------------------------------------------------------------------


def test_the_baseline_differs_from_the_agent_only_in_the_capability_block():
    # This is the whole validity of the comparison. If the two prompts ever drift anywhere
    # else, the contrast stops measuring the capability and starts measuring the drift.
    assert SYSTEM_PROMPT.replace(_WITH_TOOLS, "") == BASELINE_SYSTEM_PROMPT.replace(
        _WITHOUT_TOOLS, ""
    )


def test_both_arms_carry_the_role_and_the_rules_verbatim():
    for prompt in (SYSTEM_PROMPT, BASELINE_SYSTEM_PROMPT):
        assert _ROLE in prompt
        assert _RULES in prompt


def test_the_baseline_is_told_it_has_no_tools_and_names_none():
    assert "Ninguna herramienta y ningún corpus" in BASELINE_SYSTEM_PROMPT
    for name in TOOL_NAMES:
        assert name not in _WITHOUT_TOOLS
        assert name in _WITH_TOOLS


def test_the_bare_arm_carries_the_role_and_none_of_the_rules():
    # The second arm answers a different question - what the model does without this
    # project's honesty rules - so it must not carry them.
    assert _ROLE in BARE_BASELINE_SYSTEM_PROMPT
    assert _RULES not in BARE_BASELINE_SYSTEM_PROMPT
    assert "cita" not in BARE_BASELINE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# The set says what it claims to say
# ---------------------------------------------------------------------------


def test_the_set_has_the_coverage_the_turn_asked_for(eval_set):
    assert len(eval_set) >= 15
    unanswerable = [q for q in eval_set if not q.get("answerable_from_corpus", True)]
    assert len(unanswerable) >= 2
    assert any(q.get("expected_band") for q in eval_set)
    assert any(q.get("requires_causal_refusal") for q in eval_set)
    assert any(q.get("requires_decision_refusal") for q in eval_set)
    assert any(q.get("requires_tool_refusal") for q in eval_set)
    assert any(q.get("applicant_row") for q in eval_set)


def test_every_query_identifier_is_unique(eval_set):
    identifiers = [q["id"] for q in eval_set]
    assert len(identifiers) == len(set(identifiers))


def test_every_annotated_tool_is_a_tool_that_exists(eval_set):
    for query in eval_set:
        named = set(query.get("required_tools") or ()) | set(query.get("optional_tools") or ())
        assert named <= TOOL_NAMES, query["id"]


def test_required_and_optional_tools_do_not_overlap(eval_set):
    # A tool that is both required and optional makes the over-call metric meaningless.
    for query in eval_set:
        required = set(query.get("required_tools") or ())
        optional = set(query.get("optional_tools") or ())
        assert not (required & optional), query["id"]


def test_every_dropped_column_is_a_column_the_model_actually_reads(eval_set):
    # Dropping a column the model never reads would not test the refusal at all.
    for query in eval_set:
        for column in query.get("applicant_drop_columns") or ():
            assert column in PREDICTOR_COLUMNS, query["id"]


def test_the_annotated_band_is_the_band_the_code_resolves_for_the_number_in_the_query(eval_set):
    # The band annotation is re-derived from the probability written in the query text, so
    # a wrong annotation cannot survive. This is the check that makes a01-a04 evidence
    # rather than opinion.
    banded = [q for q in eval_set if q.get("expected_band")]
    assert banded
    for query in banded:
        numbers = re.findall(r"\b0,(\d+)\b", query["query"])
        assert len(numbers) == 1, f"{query['id']} debe declarar exactamente una probabilidad"
        probability = float(f"0.{numbers[0]}")
        assert resolve_band(probability).code == query["expected_band"], query["id"]


def test_a_query_annotated_as_unanswerable_names_nothing_the_corpus_contains(eval_set, corpus_text):
    # "No answer in the corpus" is a claim about the corpus, and the corpus can change.
    annotated = {q["id"] for q in eval_set if not q.get("answerable_from_corpus", True)}
    assert set(ABSENT_TERMS) <= annotated
    for query_id, terms in ABSENT_TERMS.items():
        for term in terms:
            assert term not in corpus_text, f"{query_id}: el corpus ya contiene «{term}»"


def test_the_unanswerable_question_about_deadlines_has_no_article_16_in_the_corpus(corpus_text):
    # a15 asks for the term to answer a complaint, which Ley 1266 fixes in article 16. The
    # corpus transcribes articles 4, 6, 13 and 15 only; if 16 ever gets added, a15 stops
    # being an abstention case and this test says so.
    assert "articulo 16" not in corpus_text
    assert "artículo 16".replace("í", "i") not in corpus_text


def test_every_query_carries_a_note_explaining_its_annotation(eval_set):
    for query in eval_set:
        assert query.get("note", "").strip(), query["id"]


# ---------------------------------------------------------------------------
# The crack-1 detector, which produced a wrong number once
# ---------------------------------------------------------------------------


def _evaluate_agent_module():
    """Load `scripts/evaluate_agent.py` by path.

    The evaluation script is not importable as a package, and normally that is fine because
    a script's job is to be run. This one is different: `detect_crack_one` is the instrument
    the headline finding of the agent evaluation is measured with, and it **reported a crack
    that had not happened** in its first version, because it compared a probability the answer
    had rounded in prose against the unrounded number the tool returned. A wrong instrument
    produces a wrong finding silently, so the cases that exposed it are locked here.
    """
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_agent.py"
    spec = importlib.util.spec_from_file_location("evaluate_agent", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_agent"] = module
    spec.loader.exec_module(module)
    return module


def test_a_probability_the_answer_rounded_in_prose_is_not_a_crack():
    # The real case: the tool resolved 0,6407 to band E and the answer wrote 0,641. Same
    # assignment, quoted to three decimals. Reading that as a second band measured the
    # instrument's own rounding.
    detect = _evaluate_agent_module().detect_crack_one
    assert detect({(0.641, "E")}, [(0.6407, "E")]) is False
    assert detect({(0.06, "B")}, [(0.0604, "B")]) is False


def test_a_band_the_tools_never_resolved_is_a_crack():
    # The other real case: the answer assigned 0,0599 to band A to illustrate the boundary.
    # Arithmetically right, and still the synthesis node doing the comparison itself.
    detect = _evaluate_agent_module().detect_crack_one
    assert detect({(0.06, "B"), (0.0599, "A")}, [(0.06, "B")]) is True


def test_the_same_number_in_a_different_band_is_a_crack():
    detect = _evaluate_agent_module().detect_crack_one
    assert detect({(0.19, "C")}, [(0.19, "D")]) is True


def test_a_band_asserted_with_no_tool_result_at_all_is_a_crack():
    # This is what a baseline does: it has no tools, so any band it assigns is its own.
    detect = _evaluate_agent_module().detect_crack_one
    assert detect({(0.19, "D")}, []) is True


def test_an_answer_that_assigns_no_band_is_not_a_crack():
    detect = _evaluate_agent_module().detect_crack_one
    assert detect(set(), [(0.19, "D")]) is False


def test_a_quote_more_precise_than_the_stored_value_is_not_a_crack():
    # The third false positive from the same root cause, and the reason the transcript now
    # stores probabilities at full precision. The tool resolved 0.10757135201580555 and the
    # answer quoted it as both 0,108 and 0,10757. Against a value stored rounded to four
    # decimals the five-decimal quote could not be confirmed, and the detector reported a
    # crack that was the agent quoting the tool faithfully.
    detect = _evaluate_agent_module().detect_crack_one
    true_value = 0.10757135201580555
    assert detect({(0.10757, "B"), (0.108, "B")}, [(true_value, "B")]) is False
    # Against the rounded value it cannot be confirmed, which is what went wrong.
    assert detect({(0.10757, "B")}, [(0.1076, "B")]) is True
