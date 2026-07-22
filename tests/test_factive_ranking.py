"""Ranking of the unmarked factive / verbal-noun readings over their marked homographs.

Both parses of each surface here are *already produced* by the analyzer — this milestone only
fixes their ORDER, via two data-driven tie-breaks (see analyzer._sort_key and the scoring
weights ``archaic_reading_penalty`` / ``possessive_markedness_penalty``):

* a ``-DIk`` / ``-(y)AcAk`` factive clause with a possessive+case (geldiğini "that he/she
  came") ranks the unmarked **3sg** possessive above the 2sg reading;
* ``gidenlerin`` ranks the **genitive** headless-relative (poss=none) above the 2sg-possessive;
* ``gelmeye`` ranks the **-mA verbal-noun + dative** above the archaic negative-optative;
* the rule is general (not participle-specific): ``kitabını`` ranks 3sg above 2sg too.

Every demoted reading is KEPT in the candidate list — the tweaks are re-ranks, never filters —
and the living 1sg/1pl optative is never demoted.
"""

from __future__ import annotations

import dataclasses

import pytest
from conftest import has_analysis

from ilmek.disambiguation.scoring import rank_candidates, score_candidate

# --- Positive: the unmarked 3sg factive is primary -----------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("geldiğini", "gel"),
        ("yaptığını", "yap"),
        ("okuduğunu", "oku"),
        ("olduğunu", "ol"),
        ("söylediğini", "söyle"),
    ],
)
def test_dik_factive_primary_is_3sg(analyzer, word, lemma):
    top = analyzer.analyze(word)[0]
    assert top.lemma == lemma
    assert top.features.get("derivation") == ("dik",)
    assert top.features.get("possessive") == "3sg"
    assert top.features.get("case") == "accusative"


@pytest.mark.positive
def test_acak_factive_primary_is_3sg(analyzer):
    top = analyzer.analyze("geleceğini")[0]
    assert top.lemma == "gel"
    assert top.features.get("derivation") == ("acak",)
    assert top.features.get("possessive") == "3sg"
    assert top.features.get("case") == "accusative"


@pytest.mark.positive
def test_verbal_noun_dative_beats_archaic_optative(analyzer):
    # gelmeye: the -mA verbal-noun + dative ("to coming") is the live reading; the negative
    # optative is archaic and must be demoted below it.
    top = analyzer.analyze("gelmeye")[0]
    assert top.pos == "NOUN"
    assert top.features.get("derivation") == ("ma",)
    assert top.features.get("case") == "dative"


@pytest.mark.positive
def test_headless_relative_genitive_beats_2sg(analyzer):
    # gidenlerin: the genitive headless-relative ("of those who go") outranks the 2sg-possessive.
    top = analyzer.analyze("gidenlerin")[0]
    assert top.features.get("case") == "genitive"
    assert top.features.get("possessive") == "none"


@pytest.mark.positive
def test_markedness_rule_is_general_not_participle_specific(analyzer):
    # A plain noun with the -In/-InI ambiguity: 3sg (his/her book, accusative) over 2sg.
    top = analyzer.analyze("kitabını")[0]
    assert top.lemma == "kitap"
    assert top.features.get("possessive") == "3sg"
    assert top.features.get("case") == "accusative"


# --- Exception / non-drop: the demoted readings stay in the list ---------------------


@pytest.mark.negative
def test_2sg_factive_reading_still_present(analyzer):
    results = analyzer.analyze("geldiğini")
    assert any(
        r.features.get("possessive") == "2sg" and r.features.get("case") == "accusative"
        for r in results[1:]
    )


@pytest.mark.negative
def test_archaic_optative_still_present(analyzer):
    # Demoted, not erased: gelmeye still carries an optative reading among its candidates.
    assert has_analysis(analyzer, "gelmeye", features={"mood": "optative"})


@pytest.mark.negative
def test_headless_relative_2sg_still_present(analyzer):
    assert has_analysis(analyzer, "gidenlerin", features={"possessive": "2sg"})


# --- Negative gate: living optatives and single-candidate surfaces untouched ----------


@pytest.mark.negative
@pytest.mark.parametrize("word,person", [("geleyim", "1sg"), ("gelelim", "1pl")])
def test_living_optative_not_demoted(analyzer, word, person):
    # 1sg/1pl optatives are fully alive and excluded from the archaic set; they stay primary.
    top = analyzer.analyze(word)[0]
    assert top.features.get("mood") == "optative"
    assert top.features.get("person") == person


@pytest.mark.negative
def test_archaic_optative_is_reranked_not_filtered(analyzer):
    # gele has a single parse (the 3sg optative); demotion is a rank, never a removal, so it
    # remains the primary of its own candidate list.
    results = analyzer.analyze("gele")
    assert results[0].features.get("mood") == "optative"
    assert results[0].features.get("person") == "3sg"


@pytest.mark.negative
@pytest.mark.parametrize("word,verb", [("koşman", "koş"), ("gezmen", "gez")])
def test_single_candidate_2sg_unaffected(analyzer, word, verb):
    # A lone verbal-noun+2sg-possessive split (no competing reading) keeps its 2sg primary —
    # the markedness tie-break only settles a genuine tie. (Guards test_agentive_man_nouns.)
    top = analyzer.analyze(word)[0]
    assert top.lemma == verb
    assert top.features.get("possessive") == "2sg"


@pytest.mark.negative
def test_evi_primary_unchanged(analyzer):
    # none and 3sg are EQUALLY unmarked, so evi keeps its prior stable order (3sg first),
    # protecting the disambiguator coupling in test_disambiguation.py.
    top = analyzer.analyze("evi")[0]
    assert top.features.get("possessive") == "3sg"
    assert top.features.get("case") == "nominative"


# --- Scoring layer agrees with the analyzer primary ----------------------------------


@pytest.mark.positive
def test_scoring_prefers_3sg_over_2sg(analyzer):
    candidates = analyzer.analyze("geldiğini")
    s3 = next(r for r in candidates if r.features.get("possessive") == "3sg")
    s2 = next(r for r in candidates if r.features.get("possessive") == "2sg")
    assert score_candidate(s3) > score_candidate(s2)
    assert rank_candidates(candidates)[0].features.get("possessive") == "3sg"


@pytest.mark.positive
def test_scoring_prefers_verbal_noun_over_optative(analyzer):
    candidates = analyzer.analyze("gelmeye")
    vn = next(r for r in candidates if r.features.get("derivation") == ("ma",))
    op = next(r for r in candidates if r.features.get("mood") == "optative")
    assert score_candidate(vn) > score_candidate(op)
    assert rank_candidates(candidates)[0].features.get("derivation") == ("ma",)


@pytest.mark.negative
def test_archaic_penalty_never_inverts_source_tiers(analyzer):
    # The archaic penalty is a re-rank within a source tier, not across tiers: a lexicon
    # (archaic-penalized) reading must still outscore the SAME reading were it a mere guess.
    op = next(r for r in analyzer.analyze("gelmeye") if r.features.get("mood") == "optative")
    assert op.source == "lexicon"
    as_guess = dataclasses.replace(op, source="guess")
    assert score_candidate(op) > score_candidate(as_guess)
