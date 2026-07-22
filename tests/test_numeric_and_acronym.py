"""Numeric tokens as NUM (digits / percent / dates / times) and the consonant-final
acronym + apostrophe case fix (TBMM'den, TÜİK'in).

Two independent milestone fixes are exercised here:

* the analyzer resolves a bare numeric token as ``pos=NUM`` by rule (``source="rule"``),
  never routing digits to the morphological guesser; and
* a consonant-final acronym written with an apostrophe suffix keeps its case, and a
  case reading is preferred over a homographic possessive one on the apostrophe path.
"""

from __future__ import annotations

import pytest

import ilmek as trnlp

# --- Numeric tokens as NUM -----------------------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize("surface", ["3", "25", "2020"])
def test_bare_cardinal_is_num_by_rule(analyzer, surface):
    r = analyzer.analyze(surface)[0]
    assert r.pos == "NUM"
    assert r.lemma == surface
    assert r.morphemes == []
    # Deterministic rule — NOT the unknown-root guesser, NOT the lexicon.
    assert r.source == "rule"


@pytest.mark.positive
def test_ordinal_dot_is_num_ordinal(analyzer):
    r = analyzer.analyze("3.")[0]
    assert r.pos == "NUM"
    assert r.features.get("num_type") == "ordinal"
    # The dot is orthography, not a morpheme, so the lemma is the bare figure.
    assert r.lemma == "3"
    assert r.source == "rule"


@pytest.mark.positive
def test_percent_is_num(analyzer):
    r = analyzer.analyze("%25")[0]
    assert r.pos == "NUM"
    assert r.features.get("num_type") == "percent"
    assert r.source == "rule"


@pytest.mark.positive
@pytest.mark.parametrize("surface", ["1.000,50", "01.01.2020", "14:30"])
def test_grouped_date_time_are_num(analyzer, surface):
    r = analyzer.analyze(surface)[0]
    assert r.pos == "NUM"
    assert r.source == "rule"


@pytest.mark.negative
@pytest.mark.parametrize("surface", ["3x4", "v2"])
def test_mixed_alphanumeric_is_not_num(analyzer, surface):
    # A token that is not a WHOLE numeric string must not become NUM: it stays on the
    # normal word/guesser path (identity X here), so the numeric fast path is exact.
    r = analyzer.analyze(surface)[0]
    assert r.pos != "NUM"


@pytest.mark.negative
def test_guesser_is_untouched_by_the_numeric_path(analyzer):
    # A non-numeric OOV word must still fall to the honest identity guess (regression pin
    # for test_guesser: hastadı is NOT a confident verb strip).
    r = analyzer.analyze("hastadı")[0]
    assert r.pos == "X"
    assert r.source == "guess"


@pytest.mark.positive
def test_number_token_carries_analysis_in_document():
    # The pipeline now analyzes number tokens (kind="number") instead of skipping them.
    doc = trnlp.load()("3 kitap aldım")
    first = doc.analyses[0]
    assert first is not None
    assert first.pos == "NUM"


# --- Consonant-final acronym + apostrophe + suffix -----------------------------------


@pytest.mark.positive
def test_consonant_acronym_keeps_ablative(analyzer):
    # Regression for the bug where a consonant-final acronym dropped its suffix and case.
    r = analyzer.analyze("TBMM'den")[0]
    assert r.pos == "PROPN"
    assert r.lemma == "tbmm"
    assert r.features.get("case") == "ablative"
    assert r.morphemes  # non-empty: the suffix after the apostrophe was analyzed
    assert r.source == "rule"


@pytest.mark.positive
def test_consonant_acronym_keeps_locative(analyzer):
    r = analyzer.analyze("TBMM'de")[0]
    assert r.pos == "PROPN"
    assert r.lemma == "tbmm"
    assert r.features.get("case") == "locative"


@pytest.mark.positive
@pytest.mark.parametrize("surface,lemma", [("KDV'den", "kdv"), ("THY'den", "thy")])
def test_other_consonant_acronyms_keep_case(analyzer, surface, lemma):
    r = analyzer.analyze(surface)[0]
    assert r.pos == "PROPN"
    assert r.lemma == lemma
    assert r.features.get("case") == "ablative"


@pytest.mark.positive
def test_acronym_genitive_preferred_over_possessive(analyzer):
    # "TÜİK'in" is genitive on the apostrophe convention, not the homographic possessive-2sg.
    r = analyzer.analyze("TÜİK'in")[0]
    assert r.pos == "PROPN"
    assert r.lemma == "tüik"
    assert r.features.get("case") == "genitive"
    assert r.features.get("possessive") in (None, "none")


@pytest.mark.exception
def test_acronym_possessive_reading_is_kept_as_alternative(analyzer):
    # The possessive reading is never silently dropped — only ranked below the genitive.
    results = analyzer.analyze("TÜİK'in")
    assert any(r.features.get("possessive") == "2sg" for r in results[1:])


@pytest.mark.positive
def test_vowel_final_acronym_dative_regression(analyzer):
    # The vowel-final path already worked; it must stay correct after the re-rank/fallback.
    r = analyzer.analyze("NATO'ya")[0]
    assert r.pos == "PROPN"
    assert r.lemma == "nato"
    assert r.features.get("case") == "dative"


@pytest.mark.exception
def test_possessive_only_apostrophe_still_wins(analyzer):
    # When the ONLY parse is possessive (no case reading of "ankaram"), it stays primary —
    # the case-preferring re-rank must not demote a possessive that has no rival.
    r = analyzer.analyze("Ankara'm")[0]
    assert r.pos == "PROPN"
    assert r.lemma == "ankara"
    assert r.features.get("possessive") == "1sg"


@pytest.mark.positive
def test_ankara_da_apostrophe_unchanged(analyzer):
    # The existing apostrophe behavior (test_api::test_proper_noun_apostrophe) is preserved.
    r = analyzer.analyze("Ankara'da")[0]
    assert r.lemma == "ankara"
    assert r.pos == "PROPN"
    assert r.features.get("case") == "locative"


# --- Deferred: buffer-consonant suffixes on a vowelless acronym ----------------------
# TBMM'ye / TBMM'nin need the acronym's spoken form ("te-be-me-me", vowel-final) to trigger
# the (y)/(n) buffer; realize() only sees the letter string "tbmm" (consonant-final), so the
# buffer is not inserted and these do not yet parse. They fail identically to before this
# milestone (bare-PROPN fallback), so there is no regression — deferred, not a wrong rule.


@pytest.mark.xfail(
    reason="acronym pronunciation is vowel-final; the (y) buffer needs the spoken letter-name "
    "context, not the consonant-final letter string TBMM",
    strict=True,
)
def test_buffer_consonant_acronym_dative_deferred(analyzer):
    assert analyzer.analyze("TBMM'ye")[0].features.get("case") == "dative"


@pytest.mark.xfail(
    reason="acronym pronunciation is vowel-final; the (n) buffer needs the spoken letter-name "
    "context, not the consonant-final letter string TBMM",
    strict=True,
)
def test_buffer_consonant_acronym_genitive_deferred(analyzer):
    assert analyzer.analyze("TBMM'nin")[0].features.get("case") == "genitive"
