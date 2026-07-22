"""Agentive ``-mAn`` nouns: a closed, lexicalized set enumerated whole (not a suffix rule).

The ``-mAn`` agentive (öğretmen "teacher", sayman "accountant", danışman "advisor") is only
semi-productive in modern Turkish and collides head-on with the fully-productive verbal-noun
chain -mA + -n (2sg possessive): ``sayman`` is *both* the noun "accountant" *and* say-+-mA+-n
"your counting". Modelling ``-mAn`` as a productive suffix would overgenerate onto non-agentive
look-alikes (zaman, orman, roman, duman, yaman) and demand a blacklist — a wrong rule by the
correctness-over-coverage contract. So the members are enumerated as plain NOUN roots in
``data/lexicon/nouns.json``; each whole-word entry outranks its verb+-mA+-n split via the
analyzer's ``-len(lemma)`` tie-break (the same mechanism that already keeps öğretmen a noun).

The verb+-mA+-n reading is never erased — it survives as a ranked alternative — and a
NON-enumerated coincidental ``-mAn`` string (koşman, gezmen) still resolves to that split, which
proves no productive ``-mAn`` suffix was introduced.
"""

from __future__ import annotations

import pytest
from conftest import has_analysis


def _primary(analyzer, word):
    return analyzer.analyze(word)[0]


#: The enumerated agentive nouns. lemma == surface (all n-final, so no voicing).
MAN_NOUNS = [
    "sayman",
    "danışman",
    "uzman",
    "eleman",
    "yönetmen",
    "çevirmen",
    "eğitmen",
    "seçmen",
    "göçmen",
]


# --- Positive: each enumerated -mAn noun is its own lexicon NOUN ----------------------


@pytest.mark.positive
@pytest.mark.parametrize("word", MAN_NOUNS)
def test_man_noun_is_whole_word_lexicon_noun(analyzer, word):
    best = _primary(analyzer, word)
    assert best.lemma == word
    assert best.pos == "NOUN"
    assert best.source == "lexicon"
    assert best.morphemes == []
    # The headline fix: the primary is NOT a false 2sg possessive (say+ma+n "your counting").
    assert best.features.get("possessive") != "2sg"


@pytest.mark.positive
def test_sayman_inflects_as_a_noun(analyzer):
    # saymanı -> sayman + ı (a real noun inflection), lemma stays "sayman".
    assert has_analysis(analyzer, "saymanı", lemma="sayman", pos="NOUN")


# --- Exception: the verb+-mA+-n split is demoted, not erased --------------------------


@pytest.mark.exception
def test_sayman_keeps_verbal_noun_possessive_alternative(analyzer):
    # sayman = NOUN "accountant" is primary, but say-+-mA(verbal noun)+-n(2sg poss) "your
    # counting" survives as a lower-ranked, grammatically-real alternative.
    results = analyzer.analyze("sayman")
    assert results[0].lemma == "sayman" and results[0].pos == "NOUN"
    assert any(a.lemma == "say" and a.features.get("possessive") == "2sg" for a in results)


# --- Regression: the model form öğretmen stays a noun --------------------------------


@pytest.mark.positive
def test_ogretmen_still_noun_primary(analyzer):
    best = _primary(analyzer, "öğretmen")
    assert best.lemma == "öğretmen"
    assert best.pos == "NOUN"
    assert best.features.get("possessive") != "2sg"


# --- Negative: no productive -mAn suffix (a non-enumerated string stays the verb split) ---


@pytest.mark.negative
@pytest.mark.parametrize("word,verb", [("koşman", "koş"), ("gezmen", "gez")])
def test_no_productive_man_suffix(analyzer, word, verb):
    # A coincidental verb+-mA+-n string that is NOT an enumerated agentive must NOT gain a
    # whole-word noun entry: it stays the verbal-noun+2sg-possessive split. Proves enumeration,
    # not a productive (over-generating) -mAn rule.
    results = analyzer.analyze(word)
    assert not any(a.lemma == word for a in results)
    assert results[0].lemma == verb and results[0].features.get("possessive") == "2sg"
