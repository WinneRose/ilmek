"""Ek-fiil: the copula ("to be") as a suffix on NOUN/ADJ/PRON/NUM predicates.

Turkish nominal/adjectival predicates take the copula *directly* as a suffix (no separate
verb): güzeldi "was beautiful", öğretmenim "I am a teacher". From a nominal base — the bare
root, or after plural/possessive/a non-accusative case/a derivation — the engine allows the
past -(y)DI, the evidential -(y)mIş, the conditional -(y)sA, the assertive -DIr, and the
zero-copula present personal endings. It reuses the verbal copula states, so -(y)DI/-(y)sA
take the type-2 persons (güzeldim/güzelsen) and -(y)mIş / the present take the type-1 persons
(güzelmişim/güzelim).

Coverage per the testing contract: positive (rule applies), negative (rule must NOT apply),
exception/deferral (strict xfail), a long suffix chain (evlerimizdeydi), and a consistency
check across stem/lemma/analyze. The file also pins the overgeneration guards: adding the
ek-fiil must NOT flip the primary analysis of a bare nominal (ev, güzel), an inflected noun
(evler, öğretmenim), or a finite verb / noun-verb homograph (geldi, yüzdü).
"""

from __future__ import annotations

import pytest
from conftest import has_analysis

import ilmek


def _find(analyzer, word, *, lemma=None, pos=None, features=None):
    """Return the first analysis of ``word`` matching the given constraints, or None."""
    for a in analyzer.analyze(word):
        if lemma is not None and a.lemma != lemma:
            continue
        if pos is not None and a.pos != pos:
            continue
        if features is not None and not all(a.features.get(k) == v for k, v in features.items()):
            continue
        return a
    return None


# --- Headline target: güzeldi -> güzel (ADJ) via the past ek-fiil (positive) ----------


@pytest.mark.positive
def test_headline_guzeldi_is_past_copula(analyzer):
    results = analyzer.analyze("güzeldi")
    best = results[0]  # the only reading, hence primary
    assert best.lemma == "güzel"
    assert best.pos == "ADJ"
    assert best.morphemes == ["di"]
    assert best.features.get("copula") == "past"
    assert best.features.get("person") == "3sg"
    assert best.source == "lexicon"
    assert best.stem == best.lemma == "güzel"  # ek-fiil is inflection, not derivation


@pytest.mark.consistency
def test_headline_views_agree():
    # stem / lemma / analyze are three views of one analysis and must agree.
    assert ilmek.lemmatize("güzeldi") == "güzel"
    assert ilmek.stem("güzeldi") == "güzel"
    best = ilmek.analyze("güzeldi")[0]
    assert best.lemma == best.stem == "güzel"
    assert best.source == "lexicon"


# --- The (y) buffer after a vowel-final base (positive / exception) -------------------


@pytest.mark.exception
@pytest.mark.parametrize(
    "word,lemma,morph,features",
    [
        ("hastaydı", "hasta", "ydı", {"copula": "past"}),  # hasta + (y)DI
        ("arabaymış", "araba", "ymış", {"evidential": True, "person": "3sg"}),  # araba + (y)mIş
    ],
)
def test_copula_takes_y_buffer_after_vowel(analyzer, word, lemma, morph, features):
    a = _find(analyzer, word, lemma=lemma, features=features)
    assert a is not None
    assert a.morphemes == [morph]
    assert a.source == "lexicon"


# --- Evidential -(y)mIş takes the type-1 persons (positive) --------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,person",
    [("güzelmiş", "3sg"), ("güzelmişim", "1sg"), ("güzelmişsin", "2sg"), ("güzelmişsiniz", "2pl")],
)
def test_evidential_copula_type1_persons(analyzer, word, person):
    assert has_analysis(
        analyzer, word, lemma="güzel", pos="ADJ", features={"evidential": True, "person": person}
    )


# --- Past -(y)DI takes the type-2 persons (positive) --------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,morphemes,person",
    [
        ("güzeldim", ["di", "m"], "1sg"),
        ("güzeldin", ["di", "n"], "2sg"),
        ("güzeldik", ["di", "k"], "1pl"),
    ],
)
def test_past_copula_type2_persons(analyzer, word, morphemes, person):
    a = _find(analyzer, word, lemma="güzel", features={"copula": "past", "person": person})
    assert a is not None
    assert a.morphemes == morphemes


# --- Conditional -(y)sA (type-2 persons) (positive) ---------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,person",
    [("güzelse", "güzel", "3sg"), ("güzelsen", "güzel", "2sg"), ("evse", "ev", "3sg")],
)
def test_conditional_copula(analyzer, word, lemma, person):
    assert has_analysis(
        analyzer, word, lemma=lemma, features={"mood": "conditional", "person": person}
    )


# --- Assertive/generalizing -DIr (positive; D->t after voiceless) --------------------


@pytest.mark.positive
def test_assertive_dir_keeps_case(analyzer):
    # evdedir: the assertive -DIr sits on the locative and does NOT overwrite case=locative.
    a = _find(analyzer, "evdedir", lemma="ev", features={"copula": "assertive"})
    assert a is not None
    assert a.morphemes == ["de", "dir"]
    assert a.features.get("case") == "locative"
    assert a.features.get("person") == "3sg"


@pytest.mark.exception
def test_assertive_dir_hardens_after_voiceless(analyzer):
    # kitap ends in voiceless p, so D -> t: kitaptır (not *kitapdır).
    a = _find(analyzer, "kitaptır", lemma="kitap", features={"copula": "assertive"})
    assert a is not None
    assert a.morphemes == ["tır"]


@pytest.mark.positive
def test_bare_adjective_assertive(analyzer):
    assert has_analysis(
        analyzer, "güzeldir", lemma="güzel", pos="ADJ", features={"copula": "assertive"}
    )


# --- Zero-copula present personal endings (positive) --------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,person",
    [("güzelim", "1sg"), ("güzelsin", "2sg"), ("güzeliz", "1pl"), ("güzelsiniz", "2pl")],
)
def test_present_copula_persons(analyzer, word, person):
    assert has_analysis(analyzer, word, lemma="güzel", pos="ADJ", features={"person": person})
    # The zero copula is marked by person alone — no fabricated tense/polarity/copula key.
    a = _find(analyzer, word, lemma="güzel", features={"person": person})
    assert "tense" not in a.features
    assert "polarity" not in a.features
    assert "copula" not in a.features
    assert a.source == "lexicon"


@pytest.mark.positive
def test_present_copula_on_noun(analyzer):
    # öğretmenim "I am a teacher": the copular reading EXISTS (person 1sg)...
    assert has_analysis(
        analyzer, "öğretmenim", lemma="öğretmen", pos="NOUN", features={"person": "1sg"}
    )
    # ...but the possessive "my teacher" stays the PRIMARY analysis (guard: no primary flip).
    best = analyzer.analyze("öğretmenim")[0]
    assert best.features.get("possessive") == "1sg"
    assert "person" not in best.features


# --- Case + copula: the case is never overwritten (positive; long chain) ------------


@pytest.mark.positive
def test_case_plus_past_copula(analyzer):
    a = _find(analyzer, "evdeydim", lemma="ev", features={"copula": "past"})
    assert a is not None
    assert a.morphemes == ["de", "ydi", "m"]
    assert a.features.get("case") == "locative"
    assert a.features.get("person") == "1sg"


@pytest.mark.positive
def test_long_chain_evlerimizdeydi(analyzer):
    a = _find(analyzer, "evlerimizdeydi", lemma="ev", features={"copula": "past"})
    assert a is not None
    assert a.lemma == "ev"
    assert a.morphemes == ["ler", "imiz", "de", "ydi"]
    assert a.features.get("number") == "plural"
    assert a.features.get("possessive") == "1pl"
    assert a.features.get("case") == "locative"
    assert a.features.get("copula") == "past"
    assert a.features.get("person") == "3sg"
    assert a.source == "lexicon"


@pytest.mark.positive
def test_possessive_plus_copula_uses_plain_y_buffer(analyzer):
    # eviydi = evi (3sg poss) + (y)DI — the plain (y) buffer, NOT the pronominal -n-.
    a = _find(analyzer, "eviydi", lemma="ev", features={"copula": "past"})
    assert a is not None
    assert a.morphemes == ["i", "ydi"]
    assert a.features.get("possessive") == "3sg"


# --- PRON predicate through the regular interrogative root (positive) ----------------


@pytest.mark.positive
def test_pronoun_predicate_via_regular_kim(analyzer):
    a = _find(analyzer, "kimdi", lemma="kim", pos="PRON", features={"copula": "past"})
    assert a is not None
    assert a.morphemes == ["di"]
    assert a.source == "lexicon"


# --- Genuine ambiguity is preserved (positive) --------------------------------------


@pytest.mark.positive
def test_plural_stays_primary_over_3pl_person(analyzer):
    # öğretmenler: plural "teachers" stays primary; the copular "they are teachers" (3pl)
    # survives as a ranked alternative.
    results = analyzer.analyze("öğretmenler")
    assert results[0].features.get("number") == "plural"
    assert "person" not in results[0].features
    assert any(r.features.get("person") == "3pl" for r in results)


@pytest.mark.positive
def test_homograph_yuzdu_verb_stays_primary(analyzer):
    # yüzdü is legitimately ambiguous: the finite verb (yüz- "swim") past, AND the noun/num
    # yüz + past copula. The finite verb must stay primary; the copula readings survive.
    results = analyzer.analyze("yüzdü")
    assert results[0].pos == "VERB"
    assert results[0].features.get("tense") == "past"
    assert has_analysis(analyzer, "yüzdü", pos="NOUN", features={"copula": "past"})
    assert has_analysis(analyzer, "yüzdü", pos="NUM", features={"copula": "past"})


# --- Negatives: harmony, the obligatory buffer, the person paradigms -----------------


@pytest.mark.negative
def test_vowel_harmony_violation_rejected(analyzer):
    # güzel + (y)DI harmonizes to -di (front), so the back *güzeldı is not a word.
    assert has_analysis(analyzer, "güzeldi", lemma="güzel")
    assert not has_analysis(analyzer, "güzeldı", lemma="güzel")


@pytest.mark.negative
def test_y_buffer_is_obligatory_after_vowel(analyzer):
    # The (y) buffer after a vowel-final base is obligatory: hastaydı only, never *hastadı.
    assert has_analysis(analyzer, "hastaydı", lemma="hasta")
    assert not has_analysis(analyzer, "hastadı", lemma="hasta")


@pytest.mark.negative
@pytest.mark.parametrize("word", ["güzeldiyim", "güzeldisin"])
def test_past_copula_rejects_type1_persons(analyzer, word):
    # Past -(y)DI takes type-2 persons (güzeldim/güzeldin); the type-1 shapes are not words.
    assert not has_analysis(analyzer, word, lemma="güzel")


@pytest.mark.negative
def test_conditional_copula_rejects_type1_person(analyzer):
    # Conditional -(y)sA takes type-2 (güzelsem), never the type-1 *güzelseyim.
    assert has_analysis(analyzer, "güzelsem", lemma="güzel", features={"mood": "conditional"})
    assert not has_analysis(analyzer, "güzelseyim", lemma="güzel")


@pytest.mark.negative
def test_evidential_copula_rejects_type2_person(analyzer):
    # Evidential -(y)mIş takes type-1 (güzelmişsin), never the *güzelmişin shape.
    assert has_analysis(analyzer, "güzelmişsin", lemma="güzel", features={"evidential": True})
    assert not has_analysis(analyzer, "güzelmişin", lemma="güzel")


@pytest.mark.negative
def test_copula_never_follows_the_accusative(analyzer):
    # *eviydi read as accusative+copula is ungrammatical: the accusative (N_ACC) is terminal.
    # The 3sg-possessive+copula reading of eviydi is fine and IS produced (tested above); only
    # the accusative+copula combination must be absent.
    assert not has_analysis(
        analyzer, "eviydi", lemma="ev", features={"case": "accusative", "copula": "past"}
    )


@pytest.mark.negative
def test_assertive_dir_is_terminal(analyzer):
    # -DIr is terminal this milestone: person/plural cannot stack on it (*güzeldirim).
    assert not has_analysis(analyzer, "güzeldirim", lemma="güzel")


# --- Overgeneration guards: primaries of common words must NOT flip -------------------


@pytest.mark.negative
def test_bare_nominal_primaries_unchanged(analyzer):
    ev = analyzer.analyze("ev")[0]
    assert ev.features.get("number") == "singular"
    assert "person" not in ev.features and "copula" not in ev.features
    guzel = analyzer.analyze("güzel")[0]
    assert guzel.pos == "ADJ"
    assert "person" not in guzel.features and "copula" not in guzel.features


@pytest.mark.negative
def test_inflected_and_verb_primaries_unchanged(analyzer):
    # evler stays plural; geldi stays a finite past verb; none acquire a copular primary.
    evler = analyzer.analyze("evler")[0]
    assert evler.features.get("number") == "plural" and "person" not in evler.features
    geldi = analyzer.analyze("geldi")[0]
    assert geldi.pos == "VERB" and geldi.features.get("tense") == "past"


@pytest.mark.negative
def test_guesser_does_not_strip_copula(analyzer):
    # The guesser forbids the ek-fiil, so an unknown word is never stripped of a copular
    # ending: 'zonktu' stays a conservative identity guess, never lemma 'zonk' + copula.
    for r in analyzer.analyze("zonktu"):
        assert not (r.source == "guess" and r.features.get("copula"))
    assert analyzer.analyze("zonktu")[0].source == "guess"


# --- Documented deferrals (strict xfail): honest known limitations -------------------


@pytest.mark.exception
@pytest.mark.xfail(
    reason="Suppletive personal-pronoun predicates (oydu, bendim) need enumerated "
    "IrregularForm surfaces; a regular o/ben root would overgenerate (*bene, *benler).",
    strict=True,
)
def test_pronoun_predicate_oydu_deferred(analyzer):
    assert has_analysis(analyzer, "oydu", lemma="o", pos="PRON", features={"copula": "past"})


@pytest.mark.exception
@pytest.mark.xfail(
    reason="-DIr stacking on the evidential copula (güzelmiştir) is a later milestone.",
    strict=True,
)
def test_dir_on_evidential_deferred(analyzer):
    assert has_analysis(
        analyzer, "güzelmiştir", lemma="güzel", features={"evidential": True, "copula": "assertive"}
    )
