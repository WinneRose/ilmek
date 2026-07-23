"""Regression tests for the alpha-review findings."""

from __future__ import annotations

import pytest

import ilmek
from ilmek.core.tokenization import tokenize


@pytest.mark.positive
def test_circumflex_vowels_stay_inside_a_word_token():
    tokens = tokenize("kâğıt")
    assert [(token.text, token.kind) for token in tokens] == [("kâğıt", "word")]


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,pos",
    [
        ("posta", "posta", "NOUN"),
        ("antiye", "antiye", "NOUN"),
        ("gönderilerinde", "gönderi", "NOUN"),
        ("makinesinden", "makine", "NOUN"),
        ("pul", "pul", "NOUN"),
        ("zarf", "zarf", "NOUN"),
        ("kart", "kart", "NOUN"),
        ("etiket", "etiket", "NOUN"),
        ("damga", "damga", "NOUN"),
        ("yazı", "yazı", "NOUN"),
        ("ek", "ek", "NOUN"),
        ("diğer", "diğer", "ADJ"),
        ("birleşik", "birleşik", "ADJ"),
        ("yazılı", "yazılı", "ADJ"),
        ("yapışkanlı", "yapışkanlı", "ADJ"),
        ("en", "en", "ADV"),
        ("çoğunlukla", "çoğunlukla", "ADV"),
    ],
)
def test_review_vocabulary_is_lexicon_verified(analyzer, word, lemma, pos):
    best = analyzer.analyze(word)[0]
    assert (best.lemma, best.pos, best.source) == (lemma, pos, "lexicon")


@pytest.mark.positive
def test_lexicon_noun_chains_prefer_possessive_readings(analyzer):
    expected = {
        "gönderilerinde": ("gönderi", "plural", "3sg", "locative"),
        "makinesinden": ("makine", "singular", "3sg", "ablative"),
        "tarihinde": ("tarih", "singular", "3sg", "locative"),
        "pulları": ("pul", "plural", "3sg", "nominative"),
        "idareleri": ("idare", "plural", "3sg", "nominative"),
        "malzemeleri": ("malzeme", "plural", "3sg", "nominative"),
    }
    for word, (lemma, number, possessive, case) in expected.items():
        best = analyzer.analyze(word)[0]
        assert best.lemma == lemma
        assert best.features.get("number") == number
        assert best.features.get("possessive") == possessive
        assert best.features.get("case") == case


@pytest.mark.positive
def test_context_resolves_determiner_passive_and_infinitive():
    doc = ilmek.analyze_sentence("bir posta hazırlanan pul göstermek")
    chosen = {
        token.text: analysis
        for token, analysis in zip(doc.tokens, doc.analyses, strict=True)
        if analysis
    }

    assert chosen["bir"].pos == "DET"
    assert chosen["hazırlanan"].pos == "ADJ"
    assert chosen["hazırlanan"].features.get("voice") == ("passive",)
    assert chosen["göstermek"].pos == "VERB"
    assert chosen["göstermek"].features.get("verbform") == "infinitive"


@pytest.mark.positive
def test_finite_evidential_assertive_chain_is_a_verb(analyzer):
    best = analyzer.analyze("başlanmıştır")[0]
    assert best.lemma == "başla"
    assert best.pos == "VERB"
    assert best.features.get("voice") == ("passive",)
    assert best.features.get("evidential") is True
    assert best.features.get("copula") == "assertive"
    assert best.features.get("person") == "3sg"


@pytest.mark.positive
def test_past_as_a_temporal_noun_beats_participle_split(analyzer):
    best = analyzer.analyze("geçmişte")[0]
    assert best.lemma == "geçmiş"
    assert best.pos == "NOUN"
    assert best.features.get("case") == "locative"


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,pos",
    [
        ("meşruiyet", "meşruiyet", "NOUN"),
        ("meşrutiyet", "meşrutiyet", "NOUN"),
        ("hakim", "hakim", "ADJ"),
        ("hâkim", "hâkim", "NOUN"),
        ("irtica", "irtica", "NOUN"),
        ("iltica", "iltica", "NOUN"),
        ("muhasebe", "muhasebe", "NOUN"),
        ("musahabe", "musahabe", "NOUN"),
        ("kâr", "kâr", "NOUN"),
        ("mütevazi", "mütevazi", "ADJ"),
        ("mütevazı", "mütevazı", "ADJ"),
        ("aciz", "aciz", "NOUN"),
        ("âciz", "âciz", "ADJ"),
        ("hâk", "hâk", "NOUN"),
        ("etken", "etken", "ADJ"),
        ("mürteci", "mürteci", "ADJ"),
        ("mülteci", "mülteci", "NOUN"),
        ("karşın", "karşın", "ADP"),
        ("karşılık", "karşılık", "NOUN"),
        ("âmâ", "âmâ", "ADJ"),
        ("lam", "lam", "NOUN"),
        ("lâm", "lâm", "NOUN"),
        ("berat", "berat", "NOUN"),
        ("beraat", "beraat", "NOUN"),
        ("mütehassis", "mütehassis", "ADJ"),
        ("mütehassıs", "mütehassıs", "ADJ"),
        ("muhabere", "muhabere", "NOUN"),
        ("muharebe", "muharebe", "NOUN"),
        ("kabil", "kabil", "ADJ"),
        ("kabîl", "kabîl", "NOUN"),
        ("kânun", "kânun", "NOUN"),
        ("mahsur", "mahsur", "ADJ"),
        ("mahzur", "mahzur", "NOUN"),
        ("vakıa", "vakıa", "NOUN"),
        ("vâkıâ", "vâkıâ", "ADV"),
    ],
)
def test_tdk_frequently_confused_words_are_lexicon_verified(analyzer, word, lemma, pos):
    assert any(
        a.lemma == lemma and a.pos == pos and a.source == "lexicon" for a in analyzer.analyze(word)
    )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("kâr", "kâr"),
        ("hâk", "hâk"),
        ("hâkim", "hâkim"),
        ("âmâ", "âmâ"),
        ("lâm", "lâm"),
        ("kabîl", "kabîl"),
        ("kânun", "kânun"),
        ("vâkıâ", "vâkıâ"),
    ],
)
def test_tdk_circumflex_surface_prefers_matching_lemma(analyzer, word, lemma):
    assert analyzer.analyze(word)[0].lemma == lemma
