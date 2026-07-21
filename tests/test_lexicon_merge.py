"""Expanded-lexicon merge: sampled new roots inflect correctly (v0.3).

These words were merged from ``docs/lexicon-candidates`` into ``ilmek/data/lexicon``.
The sample proves each flag class behaves in real Turkish: ``voicing`` softens the final
stop only before a vowel-initial suffix (kulak->kulağı, but kitapta-style consonant
suffixes keep the stop), ``vowel_drop`` reduces the stem only before a vowel suffix
(ömür->ömrü, but ömürler keeps the vowel), and stop-final roots left *unflagged* must
NOT voice (bilet->bileti, never biledi). It also pins the milestone headline: teminat and
ömür now resolve via the lexicon, not the guesser.
"""

from __future__ import annotations

import pytest
from conftest import has_analysis


def _sourced_from_lexicon(analyzer, word, lemma):
    return any(a.lemma == lemma and a.source == "lexicon" for a in analyzer.analyze(word))


# --- Voicing: final stop softens before a vowel suffix (positive) --------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("kulağı", "kulak"),  # k -> ğ
        ("borcu", "borç"),  # ç -> c
        ("mektubu", "mektup"),  # p -> b
        ("kurdu", "kurt"),  # t -> d
    ],
)
def test_new_voicing_roots_soften_before_vowel(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, features={"case": "accusative"})


@pytest.mark.positive
def test_new_adjective_root_voices(analyzer):
    # A voicing flag on an ADJ root that inflects nominally: genç -> genci (ç -> c).
    assert has_analysis(analyzer, "genci", lemma="genç", pos="ADJ", features={"case": "accusative"})


# --- Voicing exceptions: monosyllables that genuinely voice --------------------------


@pytest.mark.exception
def test_monosyllabic_voicer(analyzer):
    # uç -> ucu really voices, unlike the packaged non-voicing monosyllables at/kat/top.
    assert has_analysis(analyzer, "ucu", lemma="uç", features={"case": "accusative"})


@pytest.mark.exception
def test_vowel_drop_and_voicing_combined(analyzer):
    # kutup -> kutb: the medial vowel drops AND p voices to b in one bound form.
    assert has_analysis(analyzer, "kutbu", lemma="kutup", features={"case": "accusative"})


# --- Vowel drop: medial vowel drops before a vowel suffix (positive) -----------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("ömrü", "ömür", {"case": "accusative"}),
        ("göğsüm", "göğüs", {"possessive": "1sg"}),
    ],
)
def test_new_vowel_drop_roots(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, features=features)


# --- Plain stop-final roots must NOT voice (positive) --------------------------------


@pytest.mark.positive
@pytest.mark.parametrize("word,lemma", [("bileti", "bilet"), ("hayatı", "hayat")])
def test_unflagged_stop_final_roots_keep_stop(analyzer, word, lemma):
    assert has_analysis(analyzer, word, lemma=lemma, features={"case": "accusative"})


# --- New verb roots inflect (positive) -----------------------------------------------


@pytest.mark.positive
def test_new_verb_roots_inflect(analyzer):
    # unut is stop-final but a verb root never voices: unutuyorum, never unuduyorum.
    assert has_analysis(
        analyzer,
        "unutuyorum",
        lemma="unut",
        pos="VERB",
        features={"aspect": "progressive", "person": "1sg"},
    )
    # gez is a plain new verb root.
    assert has_analysis(
        analyzer, "gezdik", lemma="gez", pos="VERB", features={"tense": "past", "person": "1pl"}
    )


# --- Milestone headline: these words now come from the lexicon, not the guesser ------


@pytest.mark.positive
@pytest.mark.parametrize("word,lemma", [("teminatı", "teminat"), ("ömrü", "ömür")])
def test_milestone_words_resolve_via_lexicon(analyzer, word, lemma):
    assert _sourced_from_lexicon(analyzer, word, lemma)


# --- Negatives: a bound form must not appear before a consonant-initial suffix -------


@pytest.mark.negative
def test_voiced_bound_form_not_before_consonant_suffix(analyzer):
    # -DA is consonant-initial: the stop stays (mektupta), the voiced form is invalid.
    assert has_analysis(analyzer, "mektupta", lemma="mektup", features={"case": "locative"})
    assert not has_analysis(analyzer, "mektupda", lemma="mektup")


@pytest.mark.negative
def test_dropped_form_not_before_consonant_suffix(analyzer):
    # Plural is consonant-initial: the medial vowel is retained (ömürler, not ömrler).
    assert has_analysis(analyzer, "ömürler", lemma="ömür", features={"number": "plural"})
    assert not has_analysis(analyzer, "ömrler", lemma="ömür")


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("biledi", "bilet"), ("hayadı", "hayat")])
def test_unflagged_roots_do_not_false_voice(analyzer, word, lemma):
    # A spurious voiced surface must never parse back to an unflagged root.
    assert not has_analysis(analyzer, word, lemma=lemma)


@pytest.mark.negative
def test_verb_root_does_not_false_voice(analyzer):
    # No fabricated verb voicing: unut stays unut before a vowel suffix.
    assert not has_analysis(analyzer, "unuduyorum", lemma="unut")
