"""Noun morphology: plural, possessive, case, voicing, vowel-drop, pronominal buffer."""

from __future__ import annotations

import pytest
from conftest import has_analysis

# --- Plural / case basics (positive) -------------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("evler", "ev", {"number": "plural"}),
        ("evde", "ev", {"case": "locative"}),
        ("evden", "ev", {"case": "ablative"}),
        ("eve", "ev", {"case": "dative"}),
        ("evim", "ev", {"possessive": "1sg"}),
        ("kapım", "kapı", {"possessive": "1sg"}),
        ("kapısı", "kapı", {"possessive": "3sg"}),
        ("kapıya", "kapı", {"case": "dative"}),
        ("kapıda", "kapı", {"case": "locative"}),
    ],
)
def test_basic_noun_inflection(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, features=features)


# --- The showcase example ------------------------------------------------------------


@pytest.mark.positive
def test_showcase_noun_full_chain(analyzer):
    assert has_analysis(
        analyzer,
        "kitaplarımızdan",
        lemma="kitap",
        morphemes=["lar", "ımız", "dan"],
        features={"number": "plural", "possessive": "1pl", "case": "ablative"},
    )


@pytest.mark.positive
def test_evlerimizden(analyzer):
    assert has_analysis(
        analyzer,
        "evlerimizden",
        lemma="ev",
        morphemes=["ler", "imiz", "den"],
        features={"number": "plural", "possessive": "1pl", "case": "ablative"},
    )


# --- Consonant voicing (positive + negative) -----------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("kitabı", "kitap"),  # p -> b
        ("ağacı", "ağaç"),  # ç -> c
        ("kanadı", "kanat"),  # t -> d
        ("ayağı", "ayak"),  # k -> ğ
        ("rengi", "renk"),  # nk -> ng
        ("çocuğu", "çocuk"),
        ("bardağı", "bardak"),
    ],
)
def test_voicing_before_vowel_suffix(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, features={"case": "accusative"})


@pytest.mark.negative
def test_voicing_not_applied_before_consonant_suffix(analyzer):
    # -DA is consonant-initial: kitap stays kitap (kitapta, not kitabda).
    assert has_analysis(analyzer, "kitapta", lemma="kitap", features={"case": "locative"})
    assert not has_analysis(analyzer, "kitabda", lemma="kitap")


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("atı", "at"), ("topu", "top"), ("sütü", "süt")])
def test_non_voicing_roots_do_not_voice(analyzer, word, lemma):
    # These stop-final roots are lexically non-voicing: at->atı (t stays), top->topu.
    assert has_analysis(analyzer, word, lemma=lemma)


# --- Vowel drop (exception data) -----------------------------------------------------


@pytest.mark.exception
@pytest.mark.parametrize(
    "word,lemma",
    [("burnu", "burun"), ("şehri", "şehir"), ("aklı", "akıl"), ("resmi", "resim")],
)
def test_vowel_drop_before_vowel_suffix(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma)


@pytest.mark.negative
def test_vowel_drop_not_applied_before_consonant_suffix(analyzer):
    # Plural is consonant-initial: the medial vowel is retained (burunlar, not burnlar).
    assert has_analysis(analyzer, "burunlar", lemma="burun", features={"number": "plural"})
    assert not has_analysis(analyzer, "burnlar", lemma="burun")


# --- Pronominal -n- buffer after 3rd-person possessive -------------------------------


@pytest.mark.positive
def test_pronominal_buffer_after_third_possessive(analyzer):
    # evi (3sg poss) + locative -> evinde (with pronominal n).
    assert has_analysis(
        analyzer,
        "evinde",
        lemma="ev",
        features={"possessive": "3sg", "case": "locative"},
    )


@pytest.mark.negative
def test_no_pronominal_buffer_after_plain_stem(analyzer):
    # Plain locative on a vowel stem takes no n: kapıda, never kapında.
    assert has_analysis(analyzer, "kapıda", lemma="kapı", features={"case": "locative"})
    assert not has_analysis(
        analyzer, "kapında", lemma="kapı", features={"possessive": "none", "case": "locative"}
    )


# --- Ambiguity: multiple valid analyses must all be returned -------------------------


@pytest.mark.positive
def test_evi_is_ambiguous_accusative_and_possessive(analyzer):
    # "evi" = the house (acc) OR his/her house (3sg poss). Both must survive.
    assert has_analysis(analyzer, "evi", lemma="ev", features={"case": "accusative"})
    assert has_analysis(analyzer, "evi", lemma="ev", features={"possessive": "3sg"})


@pytest.mark.positive
def test_evin_is_ambiguous_possessive_and_genitive(analyzer):
    assert has_analysis(analyzer, "evin", lemma="ev", features={"possessive": "2sg"})
    assert has_analysis(analyzer, "evin", lemma="ev", features={"case": "genitive"})


# --- Instrumental case (-(y)lA) ------------------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [("kalemle", "kalem"), ("arabayla", "araba"), ("gözle", "göz")],
)
def test_instrumental_case(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, features={"case": "instrumental"})


@pytest.mark.negative
def test_instrumental_buffer_only_after_vowel(analyzer):
    # Consonant stem takes no -y- (kalemle), vowel stem takes it (arabayla).
    assert has_analysis(analyzer, "kalemle", lemma="kalem", features={"case": "instrumental"})
    assert not has_analysis(analyzer, "kalemyle", lemma="kalem")


# --- Genitive: the -n- linking buffer appears only after a vowel stem ----------------


@pytest.mark.positive
def test_genitive_on_vowel_stem_takes_n_buffer(analyzer):
    # kapı -> kapının (with -n-), whereas ev -> evin (no -n-).
    assert has_analysis(analyzer, "kapının", lemma="kapı", features={"case": "genitive"})


@pytest.mark.negative
def test_genitive_no_n_buffer_after_consonant_stem(analyzer):
    assert has_analysis(analyzer, "evin", lemma="ev", features={"case": "genitive"})
    assert not has_analysis(analyzer, "evnin", lemma="ev")
