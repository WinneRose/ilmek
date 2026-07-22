"""Relative / pronominal -ki (evdeki, benimki) and the temporal -ki/-kü (dünkü, yarınki).

The relative -ki turns a locative/genitive nominal (or a temporal noun/adverb) into an
ADJ/pronominal "the one that is (at/of) X". Design pins:

* -ki does NOT harmonize: the template is the literal lowercase "ki", so evdeki/masadaki never
  become *evdekı by construction. The sole rounded allomorph -kü (dünkü, bugünkü) is a per-word
  LEXICAL fact (temporal_rounded attribute), NOT a vowel-harmony rule — so *dünki and *yarınkü
  are both impossible.
* -ki attaches only after a LOCATIVE or GENITIVE case (evde-ki, evin-ki, kimin-ki), never after
  the ablative/dative/instrumental (*evdenki, *ondanki): the loc/gen case edges land in the
  dedicated N_CASE_LG state, the only one carrying the -ki edge.
* The genitive/locative PRONOUNS host -ki through curated ``ki_host`` roots (benim->benimki),
  which license NOTHING but -ki (no *benimler / *benimde).
* The -ki result declines like a pronoun (buffer-n before case: evdeki->evdekini) and hosts the
  ek-fiil (evdekiydi).
"""

from __future__ import annotations

import pytest
from conftest import has_analysis

import ilmek


def _lemma_sourced(analyzer, word, lemma, *, source="lexicon"):
    return any(a.lemma == lemma and a.source == source for a in analyzer.analyze(word))


def _has_ki_lexicon(analyzer, word, lemma=None):
    return any(
        a.source == "lexicon"
        and a.features.get("derivation") == ("ki",)
        and (lemma is None or a.lemma == lemma)
        for a in analyzer.analyze(word)
    )


# =====================================================================================
# 1. RELATIVE -ki after a locative / genitive (evdeki, evindeki, kiminki)
# =====================================================================================


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("evdeki", "ev"),  # locative + ki
        ("masadaki", "masa"),  # vowel-final stem, still literal "ki" (not *masadakı)
        ("evindeki", "ev"),  # poss3sg + pronominal locative + ki
        ("kiminki", "kim"),  # regular PRON root through the genitive
    ],
)
def test_relative_ki_positive(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, pos="ADJ", features={"derivation": ("ki",)})
    assert _lemma_sourced(analyzer, word, lemma)


@pytest.mark.positive
def test_relative_ki_stem_is_whole_surface(analyzer):
    # -ki is derivational, so the stem is the surface at the -ki boundary (evdeki), not the root.
    a = next(r for r in analyzer.analyze("evdeki") if r.features.get("derivation") == ("ki",))
    assert a.stem == "evdeki"
    assert a.morphemes == ["de", "ki"]


@pytest.mark.positive
def test_relative_ki_further_inflection(analyzer):
    # The -ki form declines like a pronoun (buffer-n) and pluralizes.
    assert has_analysis(
        analyzer, "evdekini", lemma="ev", pos="ADJ", features={"case": "accusative"}
    )
    assert has_analysis(analyzer, "evdekini", morphemes=["de", "ki", "ni"])
    assert has_analysis(analyzer, "evdekiler", lemma="ev", pos="ADJ", features={"number": "plural"})


@pytest.mark.positive
def test_relative_ki_hosts_ekfiil(analyzer):
    # evdekiydi = evde + ki + (y)DI: the -ki adjective takes the past copula.
    assert has_analysis(analyzer, "evdekiydi", lemma="ev", pos="ADJ", features={"copula": "past"})


# =====================================================================================
# 2. PRONOMINAL -ki on a genitive/locative pronoun host (benimki, ondaki)
# =====================================================================================


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("benimki", "ben"),
        ("seninki", "sen"),
        ("onunki", "o"),
        ("bizimki", "biz"),
        ("sizinki", "siz"),
        ("ondaki", "o"),  # locative pronoun host
    ],
)
def test_pronominal_ki_positive(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, pos="PRON", features={"derivation": ("ki",)})
    assert _lemma_sourced(analyzer, word, lemma)


@pytest.mark.positive
def test_pronominal_ki_further_inflection(analyzer):
    # benimkini = benim + ki + (n)I: the pronominal -ki takes the buffer-n accusative.
    assert has_analysis(
        analyzer, "benimkini", lemma="ben", pos="PRON", features={"case": "accusative"}
    )


# =====================================================================================
# 3. TEMPORAL -ki / -kü (dünkü, yarınki) — the rounded allomorph is a per-word lexical fact
# =====================================================================================


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("dünkü", "dün"),  # rounded -kü (temporal_rounded)
        ("bugünkü", "bugün"),  # rounded -kü
        ("yarınki", "yarın"),  # plain -ki (temporal)
        ("sabahki", "sabah"),  # NOUN temporal
        ("akşamki", "akşam"),  # NOUN temporal
        ("önceki", "önce"),  # ADV temporal
        ("sonraki", "sonra"),
    ],
)
def test_temporal_ki_positive(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, pos="ADJ", features={"derivation": ("ki",)})
    assert _lemma_sourced(analyzer, word, lemma)


# =====================================================================================
# 4. NEGATIVE: non-harmonization, the -kü/-ki allomorph split, the loc/gen-only gate
# =====================================================================================


@pytest.mark.negative
@pytest.mark.parametrize(
    "word",
    [
        "evdekı",  # -ki never harmonizes (pins the literal "ki")
        "dünki",  # dün takes -kü, not -ki
        "bugünki",
        "yarınkü",  # yarın takes -ki, not -kü
        "sabahkü",
        "evki",  # no case and no temporal attribute -> no -ki
        "evdenki",  # ablative never hosts -ki (loc/gen-only split)
        "ondanki",
        "gelki",  # a verb can never reach the -ki edge
    ],
)
def test_ki_does_not_overgenerate(analyzer, word):
    assert not _has_ki_lexicon(analyzer, word)


@pytest.mark.negative
@pytest.mark.parametrize("word", ["benimler", "benimde"])
def test_ki_host_licenses_nothing_but_ki(analyzer, word):
    # The ki_host pronoun root (benim) has exactly one edge (the pronominal -ki), so it can never
    # produce a plural/case form. No lexicon analysis exists (only a source=guess backoff may).
    assert not any(a.source == "lexicon" for a in analyzer.analyze(word))


# =====================================================================================
# 5. EXCEPTION / consistency
# =====================================================================================


@pytest.mark.exception
def test_kiminki_via_regular_genitive(analyzer):
    # kim is a regular PRON root: kim -> genitive kimin -> +ki = kiminki (lemma kim), a bonus
    # positive showing the relative -ki works off any genitive, not just the curated pronoun set.
    assert _has_ki_lexicon(analyzer, "kiminki", lemma="kim")


@pytest.mark.consistency
@pytest.mark.parametrize("word,lemma", [("evdeki", "ev"), ("benimki", "ben"), ("dünkü", "dün")])
def test_ki_stem_lemma_analyze_agree(word, lemma):
    assert ilmek.lemmatize(word) == lemma
    assert ilmek.analyze(word)[0].lemma == lemma
