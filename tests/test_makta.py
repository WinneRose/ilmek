"""Formal / written present-continuous -mAktA (gelmekte) and -mAktAdIr / -mAktAydI.

-mAktA is the same present + progressive as -Iyor but a distinct morpheme (the name "makta"
keeps the two apart, no fabricated feature). It lands in its own V_MAKTA state, whose edges are:

* the copula layer — gelmekteydi (past), gelmekteymiş (evidential);
* the type-1 persons — gelmekteyim (1sg), gelmekteler (3pl);
* the assertive -DIr — gelmektedir (accepted at N_COP_DIR, but with cur_pos==VERB it takes
  verbal finalization, so lemma gel + tense present + copula assertive).

-DIr sits on V_MAKTA ONLY, never on V_T1, so the generalizing *gelirdir (finite aorist + DIr)
stays deferred; N_COP_DIR is terminal, so *gelmektedirler is likewise deferred. -mAktA is
reachable from the bare root, every voiced stem (yazılmakta), negation (gelmemekte), the
impossibilitive, and ability (gelebilmekte).
"""

from __future__ import annotations

import pytest
from conftest import has_analysis

import ilmek


def _verb_makta(analyzer, word):
    """The lexicon VERB -mAktA analyses of ``word`` (present + progressive)."""
    return [
        a
        for a in analyzer.analyze(word)
        if a.source == "lexicon"
        and a.pos == "VERB"
        and a.features.get("tense") == "present"
        and a.features.get("aspect") == "progressive"
    ]


# =====================================================================================
# 1. POSITIVE: the bare progressive and its copula / person / -DIr stack
# =====================================================================================


@pytest.mark.positive
@pytest.mark.parametrize("word,lemma", [("gelmekte", "gel"), ("yazmakta", "yaz")])
def test_makta_bare(analyzer, word, lemma):
    assert has_analysis(
        analyzer,
        word,
        lemma=lemma,
        pos="VERB",
        features={"tense": "present", "aspect": "progressive", "person": "3sg"},
    )


@pytest.mark.positive
def test_makta_bare_morphemes(analyzer):
    assert has_analysis(analyzer, "gelmekte", lemma="gel", morphemes=["mekte"])


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,feature",
    [
        ("gelmektedir", {"copula": "assertive"}),
        ("gelmekteydi", {"copula": "past"}),
        ("gelmekteymiş", {"evidential": True}),
        ("gelmekteyim", {"person": "1sg"}),
        ("gelmekteler", {"person": "3pl"}),
    ],
)
def test_makta_copula_and_persons(analyzer, word, feature):
    assert has_analysis(analyzer, word, lemma="gel", pos="VERB", features=feature)


@pytest.mark.positive
def test_makta_on_voice_negation_ability(analyzer):
    # -mAktA is reachable after a voiced stem, negation, and ability.
    assert has_analysis(
        analyzer, "yazılmaktadır", lemma="yaz", pos="VERB", features={"voice": ("passive",)}
    )
    assert has_analysis(
        analyzer, "gelmemektedir", lemma="gel", pos="VERB", features={"polarity": "negative"}
    )
    assert has_analysis(
        analyzer, "gelebilmektedir", lemma="gel", pos="VERB", features={"ability": True}
    )


# =====================================================================================
# 2. NEGATIVE: harmony, nominal roots, and the deferred -DIr placements
# =====================================================================================


@pytest.mark.negative
def test_makta_harmony(analyzer):
    # gel is front-voweled, so the suffix is -mekte, never -makta: *gelmekta must not analyze.
    assert not _verb_makta(analyzer, "gelmekta")


@pytest.mark.negative
def test_makta_never_on_nominal_root(analyzer):
    # ev is a noun and never reaches the verbal graph: *evmekte has no lexicon VERB reading.
    assert not _verb_makta(analyzer, "evmekte")


@pytest.mark.negative
def test_dir_absent_from_finite_aorist(analyzer):
    # Regression pin: -DIr lives on V_MAKTA only, never on V_T1, so the finite aorist gelir + DIr
    # (*gelirdir) is not a finite VERB. (The pre-existing aorist *participle* gelir + DIr, an ADJ
    # predicate, is a separate, older reading and is left untouched.)
    assert not any(a.pos == "VERB" for a in analyzer.analyze("gelirdir"))


@pytest.mark.negative
def test_makta_dir_is_terminal(analyzer):
    # N_COP_DIR is terminal, so the 3pl stack *gelmektedirler is deferred: no lexicon analysis.
    assert not any(a.source == "lexicon" for a in analyzer.analyze("gelmektedirler"))


# =====================================================================================
# 3. GUESSER: -mAktA is a plain inflection, so an OOV verb is strippable through it
# =====================================================================================


@pytest.mark.negative
def test_makta_guesser_strips_oov_verb(analyzer):
    # florla is OOV; the guesser reaches -mAktA + -DIr like any tense, recovering a VERB stem
    # with present/progressive (source=guess, never masquerading as lexicon).
    best = analyzer.analyze("florlamaktadır")[0]
    assert best.source == "guess"
    assert best.pos == "VERB"
    assert best.lemma == "florla"
    assert best.features.get("tense") == "present"
    assert best.features.get("aspect") == "progressive"


# =====================================================================================
# 4. Consistency
# =====================================================================================


@pytest.mark.consistency
@pytest.mark.parametrize("word", ["gelmekte", "gelmektedir", "gelmekteydi"])
def test_makta_lemma_analyze_agree(word):
    assert ilmek.lemmatize(word) == "gel"
    assert ilmek.stem(word) == "gel"
    assert ilmek.analyze(word)[0].lemma == "gel"
