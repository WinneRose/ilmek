"""Closed classes: personal & demonstrative pronouns, existentials, interrogatives.

These decline suppletively (bana/sana, pronominal -n- of onu/bunda, genitive-based
instrumentals benimle/seninle) and are enumerated whole in ``data/lexicon/pronouns.json``,
matched by the analyzer BEFORE the regular FSM. They are dictionary-verified, so their
``source`` is ``lexicon`` — never a guess. The closed-class reading is ranked first but
never erases a genuine open-class alternative (onu = o-PRON-acc vs on-NUM-acc "ten").
"""

from __future__ import annotations

import pytest
from conftest import has_analysis


def _primary(analyzer, word):
    return analyzer.analyze(word)[0]


# --- Personal pronouns: irregular case (positive) ------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,case,person",
    [
        ("benim", "ben", "genitive", "1sg"),
        ("bana", "ben", "dative", "1sg"),
        ("beni", "ben", "accusative", "1sg"),
        ("bende", "ben", "locative", "1sg"),
        ("benden", "ben", "ablative", "1sg"),
        ("senin", "sen", "genitive", "2sg"),
        ("sana", "sen", "dative", "2sg"),
        ("sende", "sen", "locative", "2sg"),
        ("onun", "o", "genitive", "3sg"),
        ("ona", "o", "dative", "3sg"),
        ("onu", "o", "accusative", "3sg"),
        ("onda", "o", "locative", "3sg"),
        ("ondan", "o", "ablative", "3sg"),
        ("bizim", "biz", "genitive", "1pl"),
        ("bize", "biz", "dative", "1pl"),
        ("sizde", "siz", "locative", "2pl"),
        ("size", "siz", "dative", "2pl"),
    ],
)
def test_personal_pronoun_irregular_case(analyzer, word, lemma, case, person):
    assert has_analysis(
        analyzer, word, lemma=lemma, pos="PRON", features={"case": case, "person": person}
    )
    assert _primary(analyzer, word).source == "lexicon"


@pytest.mark.positive
def test_benim_segmentation_and_source(analyzer):
    # The headline shape: morphemes read off the enumerated row, source is lexicon.
    assert has_analysis(analyzer, "benim", lemma="ben", pos="PRON", morphemes=["im"])
    assert _primary(analyzer, "benim").features["case"] == "genitive"


# --- Instrumental (genitive-based, -(y)lA on the genitive stem) -----------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("benimle", "ben"),
        ("seninle", "sen"),
        ("onunla", "o"),
        ("bizimle", "biz"),
        ("sizinle", "siz"),
        ("onlarla", "o"),
        ("bununla", "bu"),
    ],
)
def test_instrumental_is_lexicon_pronoun_not_guess(analyzer, word, lemma):
    # The milestone's headline requirement: seninle MUST resolve to sen (PRON), not a guess.
    best = _primary(analyzer, word)
    assert best.lemma == lemma
    assert best.pos == "PRON"
    assert best.source == "lexicon"
    assert best.features["case"] == "instrumental"


# --- onlar paradigm: lemmatizes to "o" (UD convention onlar = o + lar) ----------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,case",
    [
        ("onlar", "nominative"),
        ("onların", "genitive"),
        ("onlara", "dative"),
        ("onları", "accusative"),
        ("onlarda", "locative"),
        ("onlardan", "ablative"),
    ],
)
def test_onlar_paradigm_lemmatizes_to_o(analyzer, word, case):
    assert has_analysis(
        analyzer,
        word,
        lemma="o",
        pos="PRON",
        features={"case": case, "number": "plural", "person": "3pl"},
    )


# --- Demonstratives bu / şu: pronominal -n-, pron_type=demonstrative ------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,case",
    [
        ("bunu", "bu", "accusative"),
        ("buna", "bu", "dative"),
        ("bunun", "bu", "genitive"),
        ("bunda", "bu", "locative"),
        ("bundan", "bu", "ablative"),
        ("şunu", "şu", "accusative"),
        ("şuna", "şu", "dative"),
        ("şunda", "şu", "locative"),
    ],
)
def test_demonstrative_pronominal_n(analyzer, word, lemma, case):
    assert has_analysis(
        analyzer,
        word,
        lemma=lemma,
        pos="PRON",
        features={"case": case, "pron_type": "demonstrative"},
    )


@pytest.mark.negative
def test_o_paradigm_carries_no_pron_type(analyzer):
    # The o/onlar surface is genuinely ambiguous personal/demonstrative: we do NOT fabricate
    # a pron_type. bu/şu, which are unambiguously demonstrative, DO carry it.
    onu = next(a for a in analyzer.analyze("onu") if a.lemma == "o")
    assert "pron_type" not in onu.features
    bunu = next(a for a in analyzer.analyze("bunu") if a.lemma == "bu")
    assert bunu.features.get("pron_type") == "demonstrative"


# --- Existentials var / yok ----------------------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize("word", ["var", "yok"])
def test_existential_is_lexicon_not_guess(analyzer, word):
    best = _primary(analyzer, word)
    assert best.lemma == word
    assert best.pos == "ADJ"
    assert best.features.get("existential") is True
    assert best.source == "lexicon"


# --- Turkish-aware casing: sentence-initial capital folds correctly ------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,case", [("Sana", "sen", "dative"), ("Bunu", "bu", "accusative")]
)
def test_capitalized_form_folds_to_pronoun(analyzer, word, lemma, case):
    best = _primary(analyzer, word)
    assert best.lemma == lemma
    assert best.pos == "PRON"
    assert best.features["case"] == case
    assert best.surface == word  # original surface (casing) preserved on the result


# --- The three views stay consistent -------------------------------------------------


@pytest.mark.consistency
def test_stem_lemma_analyze_agree_for_pronouns():
    import ilmek

    assert ilmek.lemmatize("sana") == "sen"
    assert ilmek.stem("seninle") == "sen"
    assert ilmek.lemmatize("onları") == "o"
    best = ilmek.analyze("seninle")[0]
    assert best.stem == best.lemma == "sen"


# --- Ambiguity retained: closed class outranks but never erases (exception) -----------


@pytest.mark.exception
def test_onu_keeps_numeral_ten_alternative(analyzer):
    # onu = o(PRON, accusative) is primary, but on(NUM "ten") + accusative survives as an alt.
    results = analyzer.analyze("onu")
    assert results[0].lemma == "o" and results[0].pos == "PRON"
    assert any(
        a.lemma == "on" and a.pos == "NUM" and a.features.get("case") == "accusative"
        for a in results
    )


@pytest.mark.exception
def test_onda_keeps_numeral_ten_alternative(analyzer):
    results = analyzer.analyze("onda")
    assert results[0].lemma == "o" and results[0].pos == "PRON"
    assert any(
        a.lemma == "on" and a.pos == "NUM" and a.features.get("case") == "locative" for a in results
    )


# --- Negatives: wrong/colloquial forms are NOT lexicon pronouns ----------------------


@pytest.mark.negative
@pytest.mark.parametrize("word", ["bene", "benin"])
def test_wrong_dative_genitive_are_not_lexicon_pronouns(analyzer, word):
    # Real forms are bana / benim. The made-up regular shapes must never claim lemma=ben PRON.
    for a in analyzer.analyze(word):
        assert not (a.pos == "PRON" and a.lemma == "ben")
        assert not (a.source == "lexicon" and a.lemma == "ben")


@pytest.mark.negative
@pytest.mark.parametrize("word", ["benle", "senle", "bunla"])
def test_colloquial_instrumental_is_excluded(analyzer, word):
    # Standard forms only (benimle/seninle/bununla). The colloquial variants get no
    # source=lexicon pronoun analysis — documents the deliberate standard-only stance.
    assert not any(a.source == "lexicon" and a.pos == "PRON" for a in analyzer.analyze(word))


@pytest.mark.negative
def test_regular_s_initial_word_untouched_by_su_pronoun(analyzer):
    # şu rows are ş-initial and must not shadow s-initial regular words: su (water) still
    # analyzes via the regular lexicon path.
    assert has_analysis(analyzer, "sular", lemma="su", pos="NOUN", features={"number": "plural"})
    assert _primary(analyzer, "su").source == "lexicon"


# --- Interrogatives: regular decliners under `entries` + irregular kiminle -------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,case",
    [
        ("kime", "kim", "dative"),
        ("kimi", "kim", "accusative"),
        ("kimde", "kim", "locative"),
        ("kimden", "kim", "ablative"),
        ("nerede", "nere", "locative"),
        ("nereden", "nere", "ablative"),
        ("nereye", "nere", "dative"),
    ],
)
def test_interrogatives_decline_via_regular_path(analyzer, word, lemma, case):
    assert has_analysis(analyzer, word, lemma=lemma, pos="PRON", features={"case": case})


@pytest.mark.positive
def test_kiminle_irregular_instrumental(analyzer):
    best = _primary(analyzer, "kiminle")
    assert best.lemma == "kim"
    assert best.pos == "PRON"
    assert best.features["case"] == "instrumental"
    assert best.features.get("pron_type") == "interrogative"
    assert best.source == "lexicon"
