"""Verb morphology: negation, tense/aspect, person, copular stacking, voicing."""

from __future__ import annotations

import pytest
from conftest import has_analysis


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("geldi", "gel", {"tense": "past", "person": "3sg"}),
        ("geldim", "gel", {"tense": "past", "person": "1sg"}),
        ("geldiniz", "gel", {"tense": "past", "person": "2pl"}),
        ("geliyor", "gel", {"aspect": "progressive", "person": "3sg"}),
        ("geliyorum", "gel", {"aspect": "progressive", "person": "1sg"}),
        ("gelecek", "gel", {"tense": "future", "person": "3sg"}),
        ("gelmiş", "gel", {"evidential": True, "person": "3sg"}),
        ("gelmişsiniz", "gel", {"evidential": True, "person": "2pl"}),
    ],
)
def test_basic_verb_inflection(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, features=features)


@pytest.mark.positive
def test_negation(analyzer):
    assert has_analysis(
        analyzer, "gelmedi", lemma="gel", features={"polarity": "negative", "tense": "past"}
    )
    assert has_analysis(
        analyzer,
        "gelmiyor",
        lemma="gel",
        features={"polarity": "negative", "aspect": "progressive"},
    )


@pytest.mark.positive
def test_showcase_verb_full_chain(analyzer):
    # gelmeyecekmişsiniz = gel + Neg + Future + Evidential + 2Pl
    assert has_analysis(
        analyzer,
        "gelmeyecekmişsiniz",
        lemma="gel",
        morphemes=["me", "yecek", "miş", "siniz"],
        features={
            "polarity": "negative",
            "tense": "future",
            "evidential": True,
            "person": "2pl",
        },
    )


@pytest.mark.positive
def test_future_person_softens_final_k(analyzer):
    # gelecek + Im -> geleceğim (final k of -AcAk softens before the vowel).
    assert has_analysis(
        analyzer, "geleceğim", lemma="gel", features={"tense": "future", "person": "1sg"}
    )


@pytest.mark.exception
def test_verb_voicing_before_vowel_only(analyzer):
    # git: gidiyor / gidecek (t->d before vowel) but gitti (t stays before -DI).
    assert has_analysis(analyzer, "gidiyor", lemma="git", features={"aspect": "progressive"})
    assert has_analysis(analyzer, "gidecek", lemma="git", features={"tense": "future"})
    assert has_analysis(analyzer, "gitti", lemma="git", features={"tense": "past"})


@pytest.mark.negative
def test_non_voicing_verb_stays(analyzer):
    # yap does not voice: yapıyor (p stays), never yabıyor.
    assert has_analysis(analyzer, "yapıyor", lemma="yap", features={"aspect": "progressive"})
    assert not has_analysis(analyzer, "yabıyor", lemma="yap")


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [("okuyor", "oku"), ("başlıyor", "başla"), ("yürüyor", "yürü"), ("arıyor", "ara")],
)
def test_progressive_vowel_drop(analyzer, word, lemma):
    # -Iyor deletes a preceding stem-final vowel: oku -> okuyor, başla -> başlıyor.
    assert has_analysis(analyzer, word, lemma=lemma, features={"aspect": "progressive"})


@pytest.mark.positive
def test_bare_verb_root_is_imperative(analyzer):
    assert has_analysis(
        analyzer, "gel", lemma="gel", pos="VERB", features={"mood": "imperative", "person": "2sg"}
    )


@pytest.mark.exception
@pytest.mark.xfail(
    reason="Irregular de-/ye- glide raising (diyor/yiyor) is a v0.3 target.", strict=True
)
@pytest.mark.parametrize("word,lemma", [("diyor", "de"), ("yiyor", "ye")])
def test_irregular_de_ye_progressive_known_limitation(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, features={"aspect": "progressive"})


@pytest.mark.exception
def test_regular_de_ye_forms_work(analyzer):
    # The regular (non-glide) forms are handled: dedi, demiş.
    assert has_analysis(analyzer, "dedi", lemma="de", features={"tense": "past"})
    assert has_analysis(analyzer, "demiş", lemma="de", features={"evidential": True})


# --- Negative imperative -------------------------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize("word,lemma", [("gelme", "gel"), ("yapma", "yap"), ("gitme", "git")])
def test_negative_imperative(analyzer, word, lemma):
    # A bare negated stem is a 2sg negative imperative ("gelme!" = don't come).
    assert has_analysis(
        analyzer, word, lemma=lemma, features={"polarity": "negative", "person": "2sg"}
    )


# --- Copular (ek-fiil) stacking with the -y- buffer ----------------------------------


@pytest.mark.positive
def test_copula_takes_y_buffer_after_vowel(analyzer):
    # geldi + (y)di -> geldiydi ; geldi + (y)miş -> geldiymiş
    assert has_analysis(
        analyzer, "geldiydi", lemma="gel", features={"tense": "past", "copula": "past"}
    )
    assert has_analysis(
        analyzer, "geldiymiş", lemma="gel", features={"tense": "past", "evidential": True}
    )


@pytest.mark.positive
def test_copula_after_consonant_has_no_buffer_and_keeps_tense(analyzer):
    # gelecek + ti -> gelecekti; the primary future feature must NOT be overwritten.
    assert has_analysis(
        analyzer, "gelecekti", lemma="gel", features={"tense": "future", "copula": "past"}
    )


@pytest.mark.negative
@pytest.mark.parametrize("nonword", ["geldidi", "geldimiş"])
def test_copula_without_buffer_is_not_a_word(analyzer, nonword):
    # The engine must NOT accept the buffer-less non-words as lexicon analyses.
    results = analyzer.analyze(nonword)
    assert results[0].source == "guess"
    assert not has_analysis(analyzer, nonword, lemma="gel")
