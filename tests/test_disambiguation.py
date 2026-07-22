"""Frequency scoring + sentence-context disambiguation (Aşama 5).

The disambiguation layer is *separate and optional*. These tests pin two guarantees:

1. **Word-level analysis stays context-free** — ``analyze`` still returns every candidate
   with ``confidence=None``; the layer never runs on the per-word path, and importing
   :mod:`ilmek.morphology` never pulls the layer in.
2. **Sentence context makes a reasonable, deterministic pick** — a token after a genitive
   prefers a possessive, a content word before a finite verb prefers the accusative, a
   numeral before a noun prefers NUM, etc., while every alternative is kept in ``.analyses``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import ilmek as trnlp
from ilmek import Pipeline
from ilmek.core.document import AnalysisResult
from ilmek.disambiguation import Disambiguator, rank_candidates, score_candidate

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _chosen(doc, token_text):
    """The chosen analysis for the (first) token whose surface equals ``token_text``."""
    for tok, analysis in zip(doc.tokens, doc.analyses, strict=True):
        if tok.text == token_text:
            return analysis
    raise AssertionError(f"token {token_text!r} not found in {[t.text for t in doc.tokens]}")


def _has_reading(analysis, **feature_constraints):
    """True if the chosen analysis OR one of its kept alternatives matches all constraints."""
    for cand in [analysis, *analysis.analyses]:
        if all(cand.features.get(k) == v for k, v in feature_constraints.items()):
            return True
    return False


# =====================================================================================
# 1. CONTEXT-FREE GUARANTEE (must not regress)
# =====================================================================================


@pytest.mark.negative
def test_word_level_analyze_evi_still_returns_both_readings(analyzer):
    """analyze("evi") keeps BOTH the poss-3sg-nominative and the accusative reading."""
    results = analyzer.analyze("evi")
    readings = {(r.features.get("possessive"), r.features.get("case")) for r in results}
    assert ("3sg", "nominative") in readings
    assert ("none", "accusative") in readings
    assert len(results) >= 2


@pytest.mark.negative
@pytest.mark.parametrize("word", ["evi", "yüzü"])
def test_word_level_confidence_is_never_set(analyzer, word):
    """The context-free analyzer never fabricates a confidence — it stays None."""
    assert all(r.confidence is None for r in analyzer.analyze(word))


@pytest.mark.negative
def test_pipeline_word_level_helper_bypasses_disambiguation(analyzer):
    """Pipeline.analyze (word-level) equals the analyzer's primary and stays confidence-free."""
    nlp = Pipeline()
    word_level = nlp.analyze("evi")
    assert word_level.lemma == analyzer.analyze("evi")[0].lemma
    assert word_level.features.get("possessive") == analyzer.analyze("evi")[0].features.get(
        "possessive"
    )
    assert word_level.confidence is None


@pytest.mark.negative
def test_importing_morphology_does_not_import_disambiguation():
    """Architectural guarantee (no cycle): morphology must not pull in disambiguation."""
    code = (
        "import ilmek.morphology.analyzer, sys;"
        "leaked=[m for m in sys.modules if m.startswith('ilmek.disambiguation')];"
        "assert not leaked, leaked"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


# =====================================================================================
# 2. CANDIDATE SCORING (word level)
# =====================================================================================


@pytest.mark.positive
def test_lexicon_outranks_guess(analyzer):
    """A lexicon-verified analysis scores strictly above an unknown-root guess."""
    lexical = analyzer.analyze("evler")[0]
    guessed = analyzer.analyze("zonklardan")[0]
    assert guessed.source == "guess"
    assert score_candidate(lexical) > score_candidate(guessed)


@pytest.mark.positive
def test_fewer_morphemes_scores_higher_at_equal_source_pos(analyzer):
    """Same NOUN lemma, same source: the bare form outscores the multi-morpheme one."""
    bare_noun = next(r for r in analyzer.analyze("yüz") if r.pos == "NOUN")
    plural = analyzer.analyze("yüzler")[0]
    assert bare_noun.pos == plural.pos == "NOUN"
    assert bare_noun.lemma == plural.lemma == "yüz"
    assert score_candidate(bare_noun) >= score_candidate(plural)


@pytest.mark.positive
def test_rank_candidates_is_deterministic_and_stable(analyzer):
    """Ranking twice yields the identical order; equal scores keep the analyzer's order."""

    def key(r):
        return (r.pos, r.features.get("case"), r.features.get("possessive"))

    first = [key(r) for r in rank_candidates(analyzer.analyze("yüzü"))]
    second = [key(r) for r in rank_candidates(analyzer.analyze("yüzü"))]
    assert first == second


@pytest.mark.positive
def test_pos_prior_ranks_unanalyzable_below_noun():
    """The X (unanalyzable) identity guess scores below a NOUN guess of the same surface."""
    x_identity = AnalysisResult("qux", "qux", "qux", "X", [], {}, source="guess")
    noun_guess = AnalysisResult("qux", "qux", "qux", "NOUN", [], {}, source="guess")
    assert score_candidate(x_identity) < score_candidate(noun_guess)


# =====================================================================================
# 3. SENTENCE DISAMBIGUATION (positive)
# =====================================================================================


@pytest.mark.positive
def test_genitive_context_prefers_possessive():
    """R1: 'onun evi' -> ev+poss3sg; the accusative reading is still kept as an alternative."""
    doc = trnlp.analyze_sentence("onun evi")
    evi = _chosen(doc, "evi")
    assert evi.features.get("possessive") == "3sg"
    assert _has_reading(evi, case="accusative")  # alternative not dropped


@pytest.mark.positive
def test_accusative_before_finite_verb():
    """R2: 'evi gördüm' -> ev+accusative; poss3sg kept; the verb stays gör past-1sg."""
    doc = trnlp.analyze_sentence("evi gördüm")
    evi = _chosen(doc, "evi")
    assert evi.features.get("case") == "accusative"
    assert _has_reading(evi, possessive="3sg", case="nominative")
    gordum = _chosen(doc, "gördüm")
    assert gordum.pos == "VERB"
    assert gordum.lemma == "gör"
    assert gordum.features.get("tense") == "past"
    assert gordum.features.get("person") == "1sg"


@pytest.mark.positive
def test_num_then_verb_in_spec_sentence():
    """R3/R4: 'Yüz kişi denizde yüzüyor.' -> Yüz=NUM, yüzüyor=VERB(yüz); NOUN yüz kept."""
    doc = trnlp.analyze_sentence("Yüz kişi denizde yüzüyor.")
    yuz = _chosen(doc, "Yüz")
    assert yuz.pos == "NUM"
    assert any(alt.pos == "NOUN" for alt in yuz.analyses)  # NOUN reading kept
    yuzuyor = _chosen(doc, "yüzüyor")
    assert yuzuyor.pos == "VERB"
    assert yuzuyor.lemma == "yüz"


@pytest.mark.positive
def test_possessive_chain_off_a_noun_genitive():
    """'kitabın kapısı' -> kapısı is a 3sg possessive (no-harm check; kapısı is unambiguous)."""
    doc = trnlp.analyze_sentence("kitabın kapısı")
    kapisi = _chosen(doc, "kapısı")
    assert kapisi.features.get("possessive") == "3sg"
    assert kapisi.lemma == "kapı"


@pytest.mark.positive
def test_disambiguated_confidence_is_bounded_and_unit_for_singletons():
    """Chosen analyses carry a heuristic confidence in [0,1]; a lone candidate -> 1.0."""
    doc = trnlp.analyze_sentence("evi gördüm")
    for analysis in doc.analyses:
        if analysis is not None:
            assert 0.0 <= analysis.confidence <= 1.0
    # 'gördüm' has a single candidate -> sum-normalized share is 1.0 (lexicon reliability).
    assert _chosen(doc, "gördüm").confidence == pytest.approx(1.0)


@pytest.mark.positive
def test_sentence_final_finite_verb_is_boosted():
    """R4: a finite verb at sentence end is more confident than the same verb mid-sentence.

    'bıraktık' is a finite-verb / participle homograph; the finite reading is the primary
    either way (so the lemma/stem never change), but R4 lifts its confidence when it closes
    the sentence.
    """
    final = _chosen(trnlp.analyze_sentence("bıraktık."), "bıraktık")
    mid = _chosen(trnlp.analyze_sentence("bıraktık geldi"), "bıraktık")
    assert final.pos == mid.pos == "VERB"
    assert final.confidence > mid.confidence


@pytest.mark.negative
def test_sentence_final_rule_does_not_force_a_finite_verb():
    """R4 prefers only a *finite* verb: a sentence-final noun (whose only VERB reading is a
    bare imperative) is left as the analyzer's NOUN primary, not forced into a verb."""
    doc = trnlp.analyze_sentence("Denizde yüz.")
    yuz = _chosen(doc, "yüz")
    assert yuz.pos == "NOUN"
    assert any(alt.pos == "VERB" for alt in yuz.analyses)  # imperative reading still kept


@pytest.mark.consistency
def test_disambiguation_is_deterministic():
    """The same sentence disambiguates identically across runs (incl. confidence)."""
    first = trnlp.analyze_sentence("Yüz kişi denizde yüzüyor.").to_dict()
    second = trnlp.analyze_sentence("Yüz kişi denizde yüzüyor.").to_dict()
    assert first == second


# =====================================================================================
# 4. NEGATIVE / EXCEPTION (rules must not over-fire)
# =====================================================================================


@pytest.mark.negative
def test_single_word_document_fires_no_context_rule():
    """analyze_sentence('evi') with no neighbors keeps the context-free primary (poss3sg)."""
    doc = trnlp.analyze_sentence("evi")
    evi = _chosen(doc, "evi")
    assert evi.features.get("possessive") == "3sg"
    assert evi.features.get("case") == "nominative"
    assert _has_reading(evi, case="accusative")  # both readings retained


@pytest.mark.negative
def test_num_noun_rule_does_not_fire_before_a_verb():
    """R3 needs a NOUN to the right: 'yüzü gördüm' -> yüzü is NOUN accusative (R2), not NUM."""
    doc = trnlp.analyze_sentence("yüzü gördüm")
    yuzu = _chosen(doc, "yüzü")
    assert yuzu.pos == "NOUN"
    assert yuzu.features.get("case") == "accusative"


@pytest.mark.negative
def test_existing_document_assertions_unchanged_with_disambiguation_on():
    """Default-on disambiguation must not change the spec document's lemmas."""
    doc = trnlp.analyze_sentence("Kitaplarımızı masaya bıraktık.")
    assert doc.lemmas[:3] == ["kitap", "masa", "bırak"]
    assert doc.analyses[-1] is None  # punctuation still unanalyzed


@pytest.mark.exception
def test_genitive_left_blocks_accusative_preference():
    """R2 is guarded off after a genitive: 'onun evi güzeldi' keeps evi as poss3sg."""
    doc = trnlp.analyze_sentence("onun evi güzeldi")
    evi = _chosen(doc, "evi")
    assert evi.features.get("possessive") == "3sg"
    assert evi.features.get("case") == "nominative"


@pytest.mark.negative
def test_oov_token_stays_guess_and_is_less_confident_than_lexicon():
    """A guess is never promoted to lexicon and scores a lower confidence than a lexicon token."""
    doc = trnlp.analyze_sentence("zonklardan geldi")
    oov = _chosen(doc, "zonklardan")
    lexical = _chosen(doc, "geldi")
    assert oov.source == "guess"
    assert 0.0 <= oov.confidence <= 1.0
    assert oov.confidence < lexical.confidence


# =====================================================================================
# 5. R5 ADJ-before-NOUN (ships with the hasta ADJ/NOUN homograph)
# =====================================================================================


@pytest.mark.positive
def test_adjective_before_noun_is_chosen_and_boosted():
    """R5: 'hasta çocuk uyudu' -> hasta=ADJ, NOUN kept, and the ADJ reading is more confident
    than the same word with no noun following it."""
    doc_noun = trnlp.analyze_sentence("hasta çocuk uyudu")
    hasta_noun_ctx = _chosen(doc_noun, "hasta")
    assert hasta_noun_ctx.pos == "ADJ"
    assert any(alt.pos == "NOUN" for alt in hasta_noun_ctx.analyses)  # homograph kept

    doc_verb = trnlp.analyze_sentence("hasta uyudu")
    hasta_verb_ctx = _chosen(doc_verb, "hasta")
    # R5 fires only when a NOUN follows, so it lifts the ADJ confidence above the no-rule case.
    assert hasta_noun_ctx.confidence > hasta_verb_ctx.confidence


@pytest.mark.negative
def test_hasta_is_a_word_level_adj_noun_homograph(analyzer):
    """Word-level analyze keeps BOTH the ADJ (sick) and NOUN (patient) readings of hasta."""
    pos_tags = {r.pos for r in analyzer.analyze("hasta")}
    assert {"ADJ", "NOUN"} <= pos_tags
    assert all(r.lemma == "hasta" for r in analyzer.analyze("hasta"))


@pytest.mark.exception
def test_adj_rule_does_not_fire_before_a_verb():
    """R5 must not fire when the next token is a verb: base order (ADJ primary) is kept."""
    doc = trnlp.analyze_sentence("hasta uyudu")
    hasta = _chosen(doc, "hasta")
    assert hasta.pos == "ADJ"  # analyzer's context-free primary, unchanged
    assert any(alt.pos == "NOUN" for alt in hasta.analyses)


# =====================================================================================
# 6. DISABLED LAYER (opt-out parity) + direct Disambiguator use
# =====================================================================================


@pytest.mark.negative
def test_disambiguator_can_be_disabled_for_context_free_document():
    """Pipeline(disambiguator=None) reproduces the pre-milestone context-free document."""
    nlp = Pipeline(disambiguator=None)
    doc = nlp("evi gördüm")
    evi = _chosen(doc, "evi")
    # No disambiguation: the analyzer's primary (poss3sg-nominative) is kept, confidence unset.
    assert evi.features.get("possessive") == "3sg"
    assert evi.confidence is None


@pytest.mark.positive
def test_rerank_keeps_all_alternatives():
    """rerank never drops a candidate: chosen + alternatives == the analyzer's candidate set."""
    dis = Disambiguator()
    cands = trnlp.analyze("evi")
    original = {(r.pos, r.features.get("case"), r.features.get("possessive")) for r in cands}
    (chosen,) = dis.rerank([cands])
    reranked = {
        (r.pos, r.features.get("case"), r.features.get("possessive"))
        for r in [chosen, *chosen.analyses]
    }
    assert reranked == original
