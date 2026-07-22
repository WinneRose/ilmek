"""Negative copula değil + ek-fiil, and the standalone substantive verb i- (idi/imiş/ise/iken).

Two declining function words share the ek-fiil (copula/person) machinery already used by the
interrogative particle mi, routed to a dedicated FSM start state by a lexicon attribute:

* ``değil`` (attribute ``negative_copula`` -> NEG_COP_ROOT) is the NEGATIVE copula of nominal
  predicates — the negative mirror of güzeldim. Its polarity is INHERENT: değildi/değildim/
  değilim/değilsin/değildir/değilse/değilmiş/değiller all resolve to lemma ``değil`` (PART) with
  ``polarity=negative``, which is seeded as a base feature and never overwritten by any
  copula/person suffix. Fixes the old bug where değildim fell to the guesser and was stamped
  ``polarity=positive`` by finalize_verbal_features.
* ``i-`` (attribute ``substantive_verb`` -> I_ROOT) is the ek-fiil written as a SEPARATE word:
  idi/idim/imiş/ise/isem/iken, lemma ``i``, pos AUX. It is BUFFERLESS (idi, never *iydi) and
  I_ROOT is non-final (no bare ``i``). It is polarity-NEUTRAL — nothing fabricates a polarity.

Both take NO nominal number/possessive/case (they are not nouns), enforced structurally by the
absence of those edges on their dedicated start states.
"""

from __future__ import annotations

import pytest
from conftest import has_analysis

import ilmek
from ilmek.core.tokenization import tokenize


def _primary(analyzer, word):
    return analyzer.analyze(word)[0]


def _degil_reading(analyzer, word):
    """The lexicon analysis of ``word`` whose lemma is ``değil`` (exactly one), or None."""
    for a in analyzer.analyze(word):
        if a.lemma == "değil" and a.source == "lexicon":
            return a
    return None


def _i_reading(analyzer, word):
    """The lexicon analysis of ``word`` whose lemma is ``i`` (the substantive verb), or None."""
    for a in analyzer.analyze(word):
        if a.lemma == "i" and a.source == "lexicon":
            return a
    return None


# --- Positive: bare değil ------------------------------------------------------------


@pytest.mark.positive
def test_bare_degil_is_negative_copula_particle(analyzer):
    best = _primary(analyzer, "değil")
    assert best.lemma == "değil"
    assert best.stem == "değil"
    assert best.pos == "PART"
    assert best.source == "lexicon"
    assert best.morphemes == []  # bare copula: no suffixes
    assert best.features == {"polarity": "negative"}  # negation inherent, nothing else


# --- Positive: değil + copular / person inflection -----------------------------------


@pytest.mark.positive
def test_degildi_past_copula(analyzer):
    a = _degil_reading(analyzer, "değildi")
    assert a is not None
    assert a.pos == "PART" and a.stem == "değil"
    assert a.morphemes == ["di"]
    assert a.features.get("polarity") == "negative"
    assert a.features.get("copula") == "past"
    assert a.features.get("person") == "3sg"


@pytest.mark.positive
def test_degildim_is_negative_not_positive(analyzer):
    # THE bug-fix pin: değildim was previously a guess stamped polarity=positive. It must now
    # be a lexicon-verified negative copula, and NO analysis of it may carry polarity=positive.
    a = _degil_reading(analyzer, "değildim")
    assert a is not None
    assert a.morphemes == ["di", "m"]
    assert a.features.get("polarity") == "negative"
    assert a.features.get("copula") == "past"
    assert a.features.get("person") == "1sg"
    assert _primary(analyzer, "değildim").features.get("polarity") == "negative"
    for cand in analyzer.analyze("değildim"):
        assert cand.features.get("polarity") != "positive"


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,person,morph",
    [
        ("değilim", "1sg", "im"),
        ("değilsin", "2sg", "sin"),
        ("değiliz", "1pl", "iz"),
        ("değilsiniz", "2pl", "siniz"),
    ],
)
def test_degil_zero_copula_present_persons(analyzer, word, person, morph):
    a = _degil_reading(analyzer, word)
    assert a is not None
    assert a.morphemes == [morph]
    assert a.features.get("polarity") == "negative"
    assert a.features.get("person") == person
    assert "copula" not in a.features  # present zero-copula: person only


@pytest.mark.positive
def test_degiller_is_third_plural_person_not_number(analyzer):
    # değiller comes via the present-3pl person -lAr, NOT a nominal plural edge (NEG_COP_ROOT
    # exposes none), so it is person=3pl and carries NO number key. Downstream consumers
    # distinguish the two, so this is pinned explicitly.
    a = _degil_reading(analyzer, "değiller")
    assert a is not None
    assert a.morphemes == ["ler"]
    assert a.features.get("person") == "3pl"
    assert a.features.get("polarity") == "negative"
    assert "number" not in a.features


@pytest.mark.positive
def test_degildir_assertive_copula(analyzer):
    a = _degil_reading(analyzer, "değildir")
    assert a is not None
    assert a.morphemes == ["dir"]
    assert a.features.get("polarity") == "negative"
    assert a.features.get("copula") == "assertive"
    assert a.features.get("person") == "3sg"


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,person,morphs",
    [("değilse", "3sg", ["se"]), ("değilsem", "1sg", ["se", "m"])],
)
def test_degil_conditional_copula(analyzer, word, person, morphs):
    a = _degil_reading(analyzer, word)
    assert a is not None
    assert a.morphemes == morphs
    assert a.features.get("polarity") == "negative"
    assert a.features.get("mood") == "conditional"
    assert a.features.get("person") == person


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,person,morphs", [("değilmiş", "3sg", ["miş"]), ("değilmişim", "1sg", ["miş", "im"])]
)
def test_degil_evidential_copula(analyzer, word, person, morphs):
    a = _degil_reading(analyzer, word)
    assert a is not None
    assert a.morphemes == morphs
    assert a.features.get("polarity") == "negative"
    assert a.features.get("evidential") is True
    assert a.features.get("person") == person


@pytest.mark.positive
def test_degildi_casing_folds_sentence_initial(analyzer):
    # "Değildi" (sentence-initial, Turkish İ/i-aware fold) analyzes identically to "değildi".
    best = _primary(analyzer, "Değildi")
    assert best.lemma == "değil" and best.pos == "PART"
    assert best.source == "lexicon"
    assert best.features.get("polarity") == "negative"
    assert best.features.get("copula") == "past"
    assert best.surface == "Değildi"  # original casing preserved


# --- Negative: değil takes no nominal inflection -------------------------------------


@pytest.mark.negative
@pytest.mark.parametrize(
    "word", ["değile", "değilde", "değilin", "değili", "değilden", "değilleri"]
)
def test_degil_takes_no_case_possessive_or_plural(analyzer, word):
    # NEG_COP_ROOT exposes no case/possessive/plural edges (değil is a particle, not a noun), so
    # none of these yields a lemma "değil" analysis — they stay honest guesses.
    results = analyzer.analyze(word)
    assert not any(a.source == "lexicon" and a.lemma == "değil" for a in results)


# --- Positive: substantive verb i- (idi / imiş / ise / iken) -------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,person,morphs",
    [
        ("idi", "3sg", ["di"]),
        ("idim", "1sg", ["di", "m"]),
        ("idin", "2sg", ["di", "n"]),
        ("idik", "1pl", ["di", "k"]),
        ("idiniz", "2pl", ["di", "niz"]),
        ("idiler", "3pl", ["di", "ler"]),
    ],
)
def test_i_past_copula(analyzer, word, person, morphs):
    a = _i_reading(analyzer, word)
    assert a is not None
    assert a.pos == "AUX" and a.stem == "i"
    assert a.morphemes == morphs
    assert a.features.get("copula") == "past"
    assert a.features.get("person") == person


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,person", [("imiş", "3sg"), ("imişim", "1sg"), ("imişsin", "2sg"), ("imişiz", "1pl")]
)
def test_i_evidential_copula(analyzer, word, person):
    a = _i_reading(analyzer, word)
    assert a is not None
    assert a.features.get("evidential") is True
    assert a.features.get("person") == person


@pytest.mark.positive
@pytest.mark.parametrize("word,person", [("isem", "1sg"), ("iseniz", "2pl")])
def test_i_conditional_copula(analyzer, word, person):
    # isem/iseniz are unambiguous (no frozen homograph), so the i+sA reading is primary.
    best = _primary(analyzer, word)
    assert best.lemma == "i" and best.pos == "AUX" and best.source == "lexicon"
    assert best.features.get("mood") == "conditional"
    assert best.features.get("person") == person


@pytest.mark.positive
def test_iken_is_converb(analyzer):
    # iken mirrors the CVB_KEN converb closure: verb->adverb, verbform=converb, and NOTHING
    # fabricated (no person/case). Bufferless ("iken", not *iyken).
    a = _i_reading(analyzer, "iken")
    assert a is not None
    assert a.pos == "ADV"
    assert a.morphemes == ["ken"]
    assert a.features.get("verbform") == "converb"
    assert "person" not in a.features
    assert "case" not in a.features


@pytest.mark.positive
def test_i_is_polarity_neutral(analyzer):
    # i- fabricates NO polarity (unlike değil, whose negation is inherent).
    assert "polarity" not in _i_reading(analyzer, "idi").features


# --- Negative: substantive verb i- is bufferless, non-final, closed edge set ----------


@pytest.mark.negative
@pytest.mark.parametrize("word", ["iydi", "iymiş", "iyse"])
def test_i_takes_no_y_buffer(analyzer, word):
    # The standalone forms take NO (y) buffer: only the bufferless suffixes are wired to I_ROOT,
    # so *iydi/*iymiş/*iyse produce no lemma "i" analysis (pins the bufferless-suffix choice).
    assert not any(a.source == "lexicon" and a.lemma == "i" for a in analyzer.analyze(word))


@pytest.mark.negative
def test_bare_i_is_not_a_word(analyzer):
    # I_ROOT is non-final, so a bare "i" never accepts as lemma "i"; it falls to the guesser.
    results = analyzer.analyze("i")
    assert not any(a.lemma == "i" and a.source == "lexicon" for a in results)
    assert results[0].source == "guess"


@pytest.mark.negative
@pytest.mark.parametrize("word", ["iyor", "iecek", "ilerledi", "ikinci"])
def test_i_edge_set_is_closed(analyzer, word):
    # The i- edge set is exactly four (past/evid/cond/-ken). No other tail matches, so i-initial
    # words (other tenses, ordinary vocabulary) gain no spurious lemma "i" parse.
    assert not any(a.source == "lexicon" and a.lemma == "i" for a in analyzer.analyze(word))


# --- Exception / homograph: "ise" keeps the frozen topic-particle primary -------------


@pytest.mark.exception
def test_ise_topic_particle_stays_primary_conditional_survives(analyzer):
    # ise: the frozen topic-particle irregular (lemma "ise", "as for") stays PRIMARY (irregulars
    # are prepended), while the i+se conditional reading coexists as a lexicon alternative.
    results = analyzer.analyze("ise")
    assert results[0].lemma == "ise" and results[0].pos == "PART" and results[0].source == "lexicon"
    assert has_analysis(analyzer, "ise", lemma="i", pos="AUX", features={"mood": "conditional"})
    assert ilmek.lemmatize("ise") == "ise"  # primary lemma unchanged for downstream consumers


# --- Integration: separate-word ek-fiil, both tokens lexicon-verified -----------------


@pytest.mark.positive
def test_geliyor_idi_two_lexicon_tokens(analyzer):
    toks = [t.text for t in tokenize("geliyor idi") if t.kind == "word"]
    assert toks == ["geliyor", "idi"]
    assert has_analysis(analyzer, "geliyor", lemma="gel", pos="VERB")
    assert has_analysis(analyzer, "idi", lemma="i", pos="AUX", features={"copula": "past"})


@pytest.mark.positive
def test_guzel_ise_two_lexicon_tokens(analyzer):
    toks = [t.text for t in tokenize("güzel ise") if t.kind == "word"]
    assert toks == ["güzel", "ise"]
    assert has_analysis(analyzer, "güzel", lemma="güzel", pos="ADJ")
    assert has_analysis(analyzer, "ise", lemma="i", pos="AUX", features={"mood": "conditional"})


# --- Guesser regression: neither new state is reachable for an OOV word ----------------


@pytest.mark.negative
def test_guesser_gets_no_negative_copula_or_substantive_strip(analyzer):
    # NEG_COP_ROOT / I_ROOT are reachable only via the negative_copula / substantive_verb root
    # attributes, which synthetic (guessed) roots never carry. An OOV word is never stripped of
    # a değil/i- ending nor assigned their lemmas.
    for word in ("zorgalar", "kitaplı"):
        for r in analyzer.analyze(word):
            assert r.lemma not in ("değil", "i")


# --- Consistency: lemma / stem / lemmatize agree -------------------------------------


@pytest.mark.consistency
def test_lemma_stem_agree(analyzer):
    assert ilmek.lemmatize("değildi") == "değil"
    assert ilmek.lemmatize("değilim") == "değil"
    best = ilmek.analyze("değil")[0]
    assert best.stem == best.lemma == "değil"
    idi = _i_reading(analyzer, "idi")
    assert idi.stem == idi.lemma == "i"
