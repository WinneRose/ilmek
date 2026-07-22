"""Closed-class function words: conjunctions, postpositions, particles, adverbs.

Most of these high-frequency words are INDECLINABLE — they do not inflect — so each is
enumerated whole in ``data/lexicon/function_words.json`` and matched by the analyzer BEFORE
the FSM. They are dictionary-verified: ``source == "lexicon"``, ``lemma == surface`` (except
the harmony variants and the circumflex spelling), and ``morphemes == []``. (The one declining
particle, the interrogative mi/mı/mu/mü, is a regular entry that inflects via the dedicated
Q_ROOT state — its copular/personal forms are covered in ``test_interrogative.py``; here only
its BARE surface is checked, which still resolves lexicon-verified with ``morphemes == []``.)

The closed-class reading is ranked first but never erases a genuine open-class alternative
(de = PART vs the verb de- imperative; göre = ADP vs gör-+optative; birden = ADV vs
bir(NUM)+ablative). Forced inflection and run-on misspellings deliberately stay guesses —
they never become lexicon-verified (negative tests below lock this in).
"""

from __future__ import annotations

import pytest
from conftest import has_analysis

import ilmek

#: (surface, lemma, pos) for every enumerated function word. The harmony variants of the
#: question particle (mi/mı/mu/mü) and the clitic (de/da) share one lemma; the circumflex
#: spelling ``hâlâ`` lemmatizes to ``hala``.
FUNCTION_WORDS = [
    # CONJ
    ("ve", "ve", "CONJ"),
    ("veya", "veya", "CONJ"),
    ("ya", "ya", "CONJ"),
    ("ama", "ama", "CONJ"),
    ("fakat", "fakat", "CONJ"),
    ("ancak", "ancak", "CONJ"),
    ("çünkü", "çünkü", "CONJ"),
    ("ki", "ki", "CONJ"),
    ("hem", "hem", "CONJ"),
    ("oysa", "oysa", "CONJ"),
    ("yani", "yani", "CONJ"),
    ("lakin", "lakin", "CONJ"),
    ("zira", "zira", "CONJ"),
    # ADP (postpositions)
    ("ile", "ile", "ADP"),
    ("için", "için", "ADP"),
    ("gibi", "gibi", "ADP"),
    ("kadar", "kadar", "ADP"),
    ("göre", "göre", "ADP"),
    ("karşı", "karşı", "ADP"),
    ("rağmen", "rağmen", "ADP"),
    ("dolayı", "dolayı", "ADP"),
    ("beri", "beri", "ADP"),
    # PART (particles)
    ("değil", "değil", "PART"),
    ("mi", "mi", "PART"),
    ("mı", "mi", "PART"),
    ("mu", "mi", "PART"),
    ("mü", "mi", "PART"),
    ("de", "de", "PART"),
    ("da", "de", "PART"),
    ("bile", "bile", "PART"),
    ("dahi", "dahi", "PART"),
    ("sadece", "sadece", "PART"),
    ("ise", "ise", "PART"),
    ("hatta", "hatta", "PART"),
    # ADV
    ("henüz", "henüz", "ADV"),
    ("hala", "hala", "ADV"),
    ("hâlâ", "hala", "ADV"),
    ("artık", "artık", "ADV"),
    ("belki", "belki", "ADV"),
    ("galiba", "galiba", "ADV"),
    ("yine", "yine", "ADV"),
    ("hemen", "hemen", "ADV"),
    ("birden", "birden", "ADV"),
    ("tekrar", "tekrar", "ADV"),
]


def _primary(analyzer, word):
    return analyzer.analyze(word)[0]


# --- Positive: every function word resolves as itself, lexicon-verified --------------


@pytest.mark.positive
@pytest.mark.parametrize("surface,lemma,pos", FUNCTION_WORDS)
def test_function_word_is_indeclinable_lexicon_entry(analyzer, surface, lemma, pos):
    best = _primary(analyzer, surface)
    assert best.lemma == lemma
    assert best.pos == pos  # explicit POS (guards the IrregularForm PRON default)
    assert best.source == "lexicon"
    assert best.morphemes == []  # indeclinable: no suffixes
    assert best.stem == lemma  # inflection-free: stem == lemma


@pytest.mark.positive
@pytest.mark.parametrize("surface,lemma,pos", FUNCTION_WORDS)
def test_function_word_primary_is_never_a_guess(analyzer, surface, lemma, pos):
    # The headline requirement: these words stop hitting the guesser and become lexicon.
    assert _primary(analyzer, surface).source == "lexicon"


@pytest.mark.positive
@pytest.mark.parametrize(
    "surface,lemma,pos",
    [
        ("ve", "ve", "CONJ"),
        ("için", "için", "ADP"),
        ("ama", "ama", "CONJ"),
        ("değil", "değil", "PART"),
        ("hala", "hala", "ADV"),
    ],
)
def test_milestone_headline_examples(analyzer, surface, lemma, pos):
    # The four milestone examples plus the ADV headline.
    assert has_analysis(analyzer, surface, lemma=lemma, pos=pos)
    assert _primary(analyzer, surface).pos == pos


# --- Positive: harmony variants share one lemma --------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize("surface", ["mi", "mı", "mu", "mü"])
def test_question_particle_variants_share_lemma_mi(analyzer, surface):
    best = _primary(analyzer, surface)
    assert best.lemma == "mi"
    assert best.pos == "PART"
    assert best.source == "lexicon"


@pytest.mark.positive
@pytest.mark.parametrize("surface", ["de", "da"])
def test_clitic_particle_variants_share_lemma_de(analyzer, surface):
    # Both harmony variants of the clitic lemmatize to "de"; the PART reading is primary.
    best = _primary(analyzer, surface)
    assert best.lemma == "de"
    assert best.pos == "PART"
    assert best.source == "lexicon"


@pytest.mark.positive
def test_circumflex_hala_folds_to_lemma_hala(analyzer):
    # normalize() does NOT fold â->a, so the 'still' spelling hâlâ is listed separately.
    best = _primary(analyzer, "hâlâ")
    assert best.lemma == "hala"
    assert best.pos == "ADV"
    assert best.source == "lexicon"


# --- Positive: Turkish-aware casing folds sentence-initial capitals -------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "surface,lemma,pos",
    [("Ve", "ve", "CONJ"), ("İçin", "için", "ADP"), ("Ancak", "ancak", "CONJ")],
)
def test_capitalized_function_word_folds(analyzer, surface, lemma, pos):
    # "İçin" exercises the İ->i Turkish fold; the original surface casing is preserved.
    best = _primary(analyzer, surface)
    assert best.lemma == lemma
    assert best.pos == pos
    assert best.source == "lexicon"
    assert best.surface == surface


# --- Exception: closed class outranks but never erases the homograph -------------------


@pytest.mark.exception
def test_de_keeps_verb_imperative_alternative(analyzer):
    # de -> PART is primary, but the verb de- imperative reading survives as an alternative
    # (milestone requirement: keep the verb reading of "de").
    results = analyzer.analyze("de")
    assert results[0].pos == "PART" and results[0].lemma == "de"
    assert any(a.pos == "VERB" and a.lemma == "de" for a in results)


@pytest.mark.exception
def test_gore_keeps_verb_optative_alternative(analyzer):
    results = analyzer.analyze("göre")
    assert results[0].pos == "ADP" and results[0].lemma == "göre"
    assert any(
        a.pos == "VERB" and a.lemma == "gör" and a.features.get("mood") == "optative"
        for a in results
    )


@pytest.mark.exception
def test_bile_keeps_verb_optative_alternative(analyzer):
    results = analyzer.analyze("bile")
    assert results[0].pos == "PART" and results[0].lemma == "bile"
    assert any(
        a.pos == "VERB" and a.lemma == "bil" and a.features.get("mood") == "optative"
        for a in results
    )


@pytest.mark.exception
def test_birden_keeps_numeral_ablative_alternative(analyzer):
    results = analyzer.analyze("birden")
    assert results[0].pos == "ADV" and results[0].lemma == "birden"
    assert any(
        a.pos == "NUM" and a.lemma == "bir" and a.features.get("case") == "ablative"
        for a in results
    )


@pytest.mark.exception
@pytest.mark.parametrize("word", ["dedi", "der", "demiş"])
def test_verb_de_paradigm_not_shadowed_by_particle(analyzer, word):
    # The exact-surface particle row cannot shadow the inflected verb paradigm: dedi/der/demiş
    # still resolve to lemma "de" (VERB). Locks in that irregulars are exact-surface only.
    assert has_analysis(analyzer, word, lemma="de", pos="VERB")


# --- Negative: misspellings and forced inflection never become lexicon ----------------


@pytest.mark.negative
@pytest.mark.parametrize("word", ["belkide", "yinede", "vede"])
def test_runon_misspellings_stay_guesses(analyzer, word):
    # Run-on misspellings of "belki de" / "yine de" / "ve de" must NOT become confident
    # lexicon-verified locatives — they stay honest guesses.
    results = analyzer.analyze(word)
    assert not any(a.source == "lexicon" for a in results)
    assert results[0].source == "guess"


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,lemma", [("amalar", "ama"), ("gibiye", "gibi"), ("fakattan", "fakat")]
)
def test_forced_inflection_on_indeclinables_not_lexicon(analyzer, word, lemma):
    # Indeclinables do not inflect: *amalar/*gibiye/*fakattan get no lexicon analysis with the
    # base lemma. (A guesser strip may coincidentally reach "fakat", but only as source=guess.)
    assert not any(a.source == "lexicon" and a.lemma == lemma for a in analyzer.analyze(word))


@pytest.mark.negative
@pytest.mark.parametrize("word", ["değildi"])
def test_deferred_copular_inflection_stays_guess(analyzer, word):
    # Copular inflection of değil (değildi) is deferred: it falls to the guesser rather than
    # being wrongly parsed. Documents the known limitation honestly. (The interrogative
    # particle mi is NO LONGER deferred — see test_interrogative.py: miyim/misin/midir now
    # parse as lexicon-verified lemma "mi" via the dedicated Q_ROOT state.)
    assert not any(a.source == "lexicon" for a in analyzer.analyze(word))


@pytest.mark.negative
def test_ve_does_not_prefix_shadow_verdi(analyzer):
    # The "ve" row is an exact-surface match, so it cannot prefix-shadow "verdi" (ver + di).
    best = _primary(analyzer, "verdi")
    assert best.lemma == "ver" and best.pos == "VERB"


@pytest.mark.negative
def test_su_water_not_shadowed_by_function_words(analyzer):
    # Regression guard: adding s-/ş-initial function words must not disturb regular words.
    assert has_analysis(analyzer, "sular", lemma="su", pos="NOUN", features={"number": "plural"})


# --- Consistency: stem / lemma / analyze agree ---------------------------------------


@pytest.mark.consistency
def test_stem_lemma_analyze_agree_for_function_words():
    assert ilmek.lemmatize("ve") == "ve"
    assert ilmek.stem("için") == "için"
    assert ilmek.lemmatize("mı") == "mi"
    assert ilmek.lemmatize("da") == "de"
    best = ilmek.analyze("değil")[0]
    assert best.stem == best.lemma == "değil"


@pytest.mark.consistency
def test_hala_is_adverb_consistent():
    best = ilmek.analyze("hala")[0]
    assert best.lemma == "hala"
    assert best.pos == "ADV"
    assert best.stem == "hala"
