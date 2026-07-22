"""Interrogative particle mi/mı/mu/mü (soru eki) and its copular/personal inflections.

The Turkish yes/no question particle is written as a SEPARATE token but hosts the ek-fiil
copula and person suffixes: midir/mıdır/mudur (assertive -DIr), misin/mısın/musun/müsün (2sg),
miyim/mıyım/muyum (1sg -(y)Im), miyiz (1pl), misiniz (2pl), miydi (past -(y)DI), miymiş
(evidential -(y)mIş), and bare mi/mı/mu/mü. It is implemented generatively: the lexicon entry
carries attribute ``interrogative`` which routes it to the dedicated Q_ROOT state whose only
edges are the (filtered) nominal-copula ones, so every inflected form falls out for free and
the copula/person harmonize to the particle's own vowel (mu+sun -> musun, mü+sün -> müsün).

Every form is lexicon-verified, lemma/stem ``"mi"`` (all four surfaces share it), and carries
``features["question"] is True``. Q_ROOT takes NO plural/possessive/case: mi is a particle, not
a full noun (no *miler, no *mide, no *mim). The copular conditional -(y)sA is deliberately not
wired (correctness over coverage: *miyse/mıysa is marginal); a negative test pins that.
"""

from __future__ import annotations

import pytest
from conftest import has_analysis

import ilmek
from ilmek.core.tokenization import tokenize


def _primary(analyzer, word):
    return analyzer.analyze(word)[0]


def _mi_reading(analyzer, word):
    """The lexicon analysis whose lemma is ``"mi"`` (there is exactly one per form), or None."""
    for a in analyzer.analyze(word):
        if a.lemma == "mi" and a.source == "lexicon":
            return a
    return None


# --- Positive: bare particle, all four harmonic surfaces -----------------------------


@pytest.mark.positive
@pytest.mark.parametrize("surface", ["mi", "mı", "mu", "mü"])
def test_bare_particle_is_interrogative(analyzer, surface):
    best = _primary(analyzer, surface)
    assert best.lemma == "mi"  # all four harmonic surfaces share one lemma
    assert best.stem == "mi"
    assert best.pos == "PART"
    assert best.source == "lexicon"
    assert best.morphemes == []  # bare particle: no suffixes
    assert best.features.get("question") is True


@pytest.mark.positive
@pytest.mark.parametrize("surface", ["mi", "mı", "mu", "mü"])
def test_bare_particle_has_no_nominal_or_verbal_features(analyzer, surface):
    # A bare particle is question=True and NOTHING else: no fabricated person/copula/case/
    # number/possessive (it is not a noun) and no verbal mood=imperative (it is not a verb).
    feats = _primary(analyzer, surface).features
    for key in ("person", "copula", "evidential", "case", "number", "possessive", "mood"):
        assert key not in feats


# --- Positive: assertive -DIr (midir) ------------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize("word,morph", [("midir", "dir"), ("mıdır", "dır"), ("mudur", "dur")])
def test_assertive_copula(analyzer, word, morph):
    a = _mi_reading(analyzer, word)
    assert a is not None
    assert a.lemma == "mi" and a.stem == "mi" and a.pos == "PART"
    assert a.source == "lexicon"
    assert a.morphemes == [morph]
    assert a.features.get("question") is True
    assert a.features.get("copula") == "assertive"
    assert a.features.get("person") == "3sg"


# --- Positive: 2sg person -sIn (misin), harmony follows the particle vowel ------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,morph", [("misin", "sin"), ("mısın", "sın"), ("musun", "sun"), ("müsün", "sün")]
)
def test_second_person_singular(analyzer, word, morph):
    a = _mi_reading(analyzer, word)
    assert a is not None
    assert a.lemma == "mi" and a.pos == "PART" and a.source == "lexicon"
    assert a.morphemes == [morph]
    assert a.features.get("question") is True
    assert a.features.get("person") == "2sg"
    assert "copula" not in a.features  # present zero-copula: person only


# --- Positive: 1sg / 1pl / 2pl persons -----------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize("word,morph", [("miyim", "yim"), ("mıyım", "yım"), ("muyum", "yum")])
def test_first_person_singular(analyzer, word, morph):
    a = _mi_reading(analyzer, word)
    assert a is not None
    assert a.morphemes == [morph]
    assert a.features.get("question") is True
    assert a.features.get("person") == "1sg"


@pytest.mark.positive
@pytest.mark.parametrize("word,person", [("miyiz", "1pl"), ("muyuz", "1pl"), ("misiniz", "2pl")])
def test_first_person_plural_and_second_plural(analyzer, word, person):
    a = _mi_reading(analyzer, word)
    assert a is not None
    assert a.features.get("question") is True
    assert a.features.get("person") == person


# --- Positive: past copula -(y)DI (miydi) and its persons ----------------------------


@pytest.mark.positive
def test_past_copula_defaults_third_person(analyzer):
    a = _mi_reading(analyzer, "miydi")
    assert a is not None
    assert a.morphemes == ["ydi"]
    assert a.features.get("question") is True
    assert a.features.get("copula") == "past"
    assert a.features.get("person") == "3sg"


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,morphs,person", [("miydim", ["ydi", "m"], "1sg"), ("miydin", ["ydi", "n"], "2sg")]
)
def test_past_copula_with_person(analyzer, word, morphs, person):
    a = _mi_reading(analyzer, word)
    assert a is not None
    assert a.morphemes == morphs
    assert a.features.get("copula") == "past"
    assert a.features.get("person") == person


@pytest.mark.positive
def test_past_copula_third_plural_after_copula_is_licensed(analyzer):
    # mıydılar: 3pl AFTER the past copula IS grammatical (only the BARE present *miler is
    # blocked). The 3pl -lAr comes from V_COP2's own person set, not the filtered Q_ROOT edge.
    a = _mi_reading(analyzer, "mıydılar")
    assert a is not None
    assert a.morphemes == ["ydı", "lar"]
    assert a.features.get("copula") == "past"
    assert a.features.get("person") == "3pl"


# --- Positive: evidential copula -(y)mIş (miymiş) ------------------------------------


@pytest.mark.positive
def test_evidential_copula(analyzer):
    a = _mi_reading(analyzer, "miymiş")
    assert a is not None
    assert a.morphemes == ["ymiş"]
    assert a.features.get("question") is True
    assert a.features.get("evidential") is True
    assert a.features.get("person") == "3sg"


@pytest.mark.positive
@pytest.mark.parametrize("word,person", [("miymişim", "1sg"), ("mıymış", "3sg")])
def test_evidential_copula_persons_and_back_harmony(analyzer, word, person):
    a = _mi_reading(analyzer, word)
    assert a is not None
    assert a.features.get("evidential") is True
    assert a.features.get("person") == person


# --- Positive: lemma / stem / lemmatize agree ----------------------------------------


@pytest.mark.consistency
@pytest.mark.parametrize("word", ["mi", "mı", "mu", "mü", "midir", "misin", "miydi", "miymiş"])
def test_lemma_and_stem_are_mi(word):
    best = ilmek.analyze(word)[0] if word == "mi" else None
    # For the bare particle the primary IS the mi reading; for inflected forms the mi reading
    # is unambiguous (no homograph), so lemmatize returns "mi" and stem is "mi".
    assert ilmek.lemmatize(word) == "mi"
    if best is not None:
        assert best.stem == best.lemma == "mi"


# --- Positive: tokenizer keeps the particle a separate token -------------------------


@pytest.mark.positive
def test_particle_is_a_separate_token(analyzer):
    # "Geliyor mu?" -> the particle is its own token, analyzed on its own to lemma "mi".
    toks = [t for t in tokenize("Geliyor mu?") if t.kind == "word"]
    assert [t.text for t in toks] == ["Geliyor", "mu"]
    assert analyzer.analyze("mu")[0].lemma == "mi"


# --- Positive: Turkish-aware casing folds to the particle ----------------------------


@pytest.mark.positive
@pytest.mark.parametrize("surface", ["Mü", "MI", "Mi", "Mu"])
def test_casing_folds_to_particle(analyzer, surface):
    best = _primary(analyzer, surface)
    assert best.lemma == "mi"
    assert best.pos == "PART"
    assert best.source == "lexicon"
    assert best.features.get("question") is True


# --- Exception: müdür homograph — the noun stays primary, the particle survives -------


@pytest.mark.exception
def test_mudur_noun_stays_primary_particle_is_alternative(analyzer):
    # müdür (director, NOUN) collides with mü+DIr (assertive question). The real noun stays
    # primary (ranked by -len(lemma)); the mi reading survives only as a ranked alternative.
    results = analyzer.analyze("müdür")
    assert results[0].lemma == "müdür"
    assert results[0].pos == "NOUN"
    assert has_analysis(analyzer, "müdür", lemma="mi", pos="PART", features={"copula": "assertive"})


# --- Negative: mi does NOT inflect as a noun (no case / plural / possessive) ----------


@pytest.mark.negative
@pytest.mark.parametrize("word", ["miler", "mide", "miye", "miyi", "miden", "minin", "mim", "misi"])
def test_particle_takes_no_nominal_inflection(analyzer, word):
    # Q_ROOT has no plural/case/possessive edges, so none of these yields a lemma "mi"
    # analysis: no *miler (plural / present-3pl), no *mide/miye/miyi/miden/minin (case),
    # no *mim/misi (possessive). They stay honest guesses.
    results = analyzer.analyze(word)
    assert not any(a.lemma == "mi" for a in results)
    assert not any(a.source == "lexicon" and a.lemma == "mi" for a in results)


@pytest.mark.negative
@pytest.mark.parametrize("word", ["miyse", "mıysa"])
def test_conditional_copula_is_not_wired(analyzer, word):
    # The copular conditional -(y)sA is deliberately filtered off Q_ROOT (correctness over
    # coverage: *miyse/mıysa is marginal), so these produce no lemma "mi" analysis.
    results = analyzer.analyze(word)
    assert not any(a.lemma == "mi" for a in results)


@pytest.mark.negative
def test_bare_particle_primary_has_no_number_case_or_mood(analyzer):
    # Guards both leaks at once: the bare particle must not gain nominal defaults
    # (number/case/possessive) nor a verbal mood=imperative from a wrong closure branch.
    feats = _primary(analyzer, "mi").features
    assert feats == {"question": True}


@pytest.mark.negative
def test_deger_degildi_stays_a_guess(analyzer):
    # değil stays frozen/indeclinable this milestone: its copular inflection değildi must NOT
    # become lexicon-verified (this milestone widens only the interrogative particle).
    assert not any(a.source == "lexicon" for a in analyzer.analyze("değildi"))


@pytest.mark.negative
def test_guesser_gets_no_question_or_copula_strip(analyzer):
    # An OOV word must never be stripped of a question/copula ending: Q_ROOT is reachable only
    # via the "interrogative" attribute, which synthetic (guessed) roots never carry.
    for r in analyzer.analyze("zorgalar"):
        assert r.features.get("question") is None
        assert "copula" not in r.features
        assert r.lemma != "mi"
