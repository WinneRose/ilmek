"""Longtail milestone: associative -DAş, attenuative -(I)msI, circumflex lookup-folding,
and fraction/quantity numerals.

Four independent additions, all modelled on existing precedents (no new engine mechanism):

* Associative -DAş (NOUN -> NOUN): meslek->meslektaş, vatan->vatandaş, ses->sesteş. It does NOT
  vowel-harmonize in the common set (meslektaş, NOT *meslekteş), so it is TWO literal-vowel edges
  (-DAş / -DEş), each attribute-gated exactly like the -ki/-kü temporal split.
* Attenuative -(I)msI (ADJ -> ADJ): mavi->mavimsi, beyaz->beyazımsı ("-ish"), attribute-gated.
* Circumflex folding for LOOKUP: kâğıt matches kağıt, hâlâ matches hala, âlim matches alim; the
  original circumflex surface is preserved.
* Fraction numerals as NUM: yarım, çeyrek, buçuk (the ADJ homograph of yarım is kept).

Per the testing contract each rule carries positive + negative (+ exception) cases, and the
overgeneration guards are pinned: -DAş/-(I)msI fire ONLY on curated roots, the wrong-harmony
allomorph is rejected, and adding fractions leaves the lexicalized roots (arkadaş, kardeş) and
the yarımşar xfail untouched.
"""

from __future__ import annotations

import pytest
from conftest import has_analysis

import ilmek


def _find(analyzer, word, *, lemma=None, pos=None, derivation=None):
    """Return the first analysis of ``word`` matching the given constraints, or None."""
    for a in analyzer.analyze(word):
        if lemma is not None and a.lemma != lemma:
            continue
        if pos is not None and a.pos != pos:
            continue
        if derivation is not None and a.features.get("derivation") != derivation:
            continue
        return a
    return None


# --- Associative -DAş (positive) -----------------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes",
    [
        ("meslektaş", "meslek", ["taş"]),  # D hardens to t after voiceless k, NO harmony (-taş)
        ("yoldaş", "yol", ["daş"]),  # d stays voiced after l
        ("dindaş", "din", ["daş"]),  # d stays voiced after n; NOT *dindeş
        ("çağdaş", "çağ", ["daş"]),  # d after ğ (çağdaş also a lexicon ADJ; derived NOUN alt)
        ("sesteş", "ses", ["teş"]),  # the front -DEş allomorph (t after voiceless s)
    ],
)
def test_associative_das_positive(analyzer, word, lemma, morphemes):
    a = _find(analyzer, word, lemma=lemma, pos="NOUN", derivation=("das",))
    assert a is not None, f"no {lemma}+das analysis for {word}"
    assert a.morphemes == morphemes
    assert a.stem == word  # stem is the derived surface
    assert a.lemma == lemma
    assert a.source == "lexicon"


@pytest.mark.positive
def test_associative_das_inflects_and_hosts_ekfiil(analyzer):
    # A -DAş stem inflects (possessive/plural) and hosts the ek-fiil, all keeping derivation=das.
    a = _find(analyzer, "meslektaşım", lemma="meslek", derivation=("das",))
    assert a is not None and a.morphemes == ["taş", "ım"]
    b = _find(analyzer, "dindaştı", lemma="din", derivation=("das",))
    assert b is not None and b.morphemes == ["daş", "tı"]  # ek-fiil past copula


@pytest.mark.positive
def test_vatandas_root_primary_with_derived_alternative(analyzer):
    # vatandaş is a lexicalized whole-word root (kept), so its root reading stays PRIMARY;
    # vatan+daş is a ranked ALTERNATIVE, not erased (kumsal precedent).
    results = analyzer.analyze("vatandaş")
    assert results[0].lemma == "vatandaş" and results[0].features.get("derivation") is None
    alt = _find(analyzer, "vatandaş", lemma="vatan", derivation=("das",))
    assert alt is not None and alt.morphemes == ["daş"]
    # The derived alternative also inflects (vatandaşlar -> vatan+daş+lar as an alternative).
    plural_alt = _find(analyzer, "vatandaşlar", lemma="vatan", derivation=("das",))
    assert plural_alt is not None and plural_alt.morphemes == ["daş", "lar"]


# --- Associative -DAş (negative) -----------------------------------------------------


@pytest.mark.negative
@pytest.mark.parametrize("word", ["evdaş", "kitaptaş", "sudaş"])
def test_associative_das_needs_attribute(analyzer, word):
    # -DAş fires ONLY on a curated root (assoc_das): a plain noun gets no das-derivation.
    assert not any(a.features.get("derivation") == ("das",) for a in analyzer.analyze(word))


@pytest.mark.negative
@pytest.mark.parametrize("word", ["meslekteş", "dindeş", "sestaş"])
def test_associative_das_harmony_is_lexical_not_a_rule(analyzer, word):
    # The wrong-harmony allomorph is NOT analyzable: the two-edge literal-vowel split is what
    # pins this (a single harmonizing "DAş" template would wrongly accept *meslekteş/*dindeş).
    assert not any(a.features.get("derivation") == ("das",) for a in analyzer.analyze(word))


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("arkadaş", "arkadaş"), ("kardeş", "kardeş")])
def test_lexicalized_das_words_stay_roots(analyzer, word, lemma):
    # arkadaş/kardeş are whole-word roots; arka carries no assoc_das, so no arka+daş split.
    assert analyzer.analyze(word)[0].lemma == lemma
    assert not any(a.features.get("derivation") == ("das",) for a in analyzer.analyze(word))


# --- Attenuative -(I)msI (positive) --------------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes",
    [
        ("mavimsi", "mavi", ["msi"]),  # no linking vowel after the vowel-final mavi
        ("ekşimsi", "ekşi", ["msi"]),
        ("beyazımsı", "beyaz", ["ımsı"]),  # (I) linking vowel after the consonant z
        ("sarımsı", "sarı", ["msı"]),
        ("acımsı", "acı", ["msı"]),
    ],
)
def test_attenuative_imsi_positive(analyzer, word, lemma, morphemes):
    a = _find(analyzer, word, lemma=lemma, pos="ADJ", derivation=("imsi",))
    assert a is not None, f"no {lemma}+imsi analysis for {word}"
    assert a.morphemes == morphemes
    assert a.stem == word
    assert a.lemma == lemma
    assert a.source == "lexicon"


@pytest.mark.positive
def test_attenuative_imsi_inflects_and_hosts_ekfiil(analyzer):
    a = _find(analyzer, "mavimsiydi", lemma="mavi", derivation=("imsi",))
    assert a is not None and a.morphemes == ["msi", "ydi"]
    b = _find(analyzer, "beyazımsılar", lemma="beyaz", derivation=("imsi",))
    assert b is not None and b.morphemes == ["ımsı", "lar"]


# --- Attenuative -(I)msI (negative) --------------------------------------------------


@pytest.mark.negative
@pytest.mark.parametrize("word", ["güzelimsi", "büyükümsü"])
def test_attenuative_imsi_needs_attribute(analyzer, word):
    # -(I)msI fires ONLY on a curated ADJ (attenuative): a plain adjective gets no imsi.
    assert not any(a.features.get("derivation") == ("imsi",) for a in analyzer.analyze(word))


@pytest.mark.negative
def test_attenuative_imsi_blocked_on_noun(analyzer):
    # applies_to={ADJ}: even a NOUN never takes -(I)msI (no *evimsi).
    assert not any(a.features.get("derivation") == ("imsi",) for a in analyzer.analyze("evimsi"))


@pytest.mark.negative
def test_sarimsak_is_not_an_attenuative(analyzer):
    # sarımsak (garlic) must NOT gain a spurious sarı+imsi attenuative parse (surface differs:
    # msak != msI). (An unrelated ek-fiil conditional sarı+m+sa+k coexists — that is not this
    # rule; we pin only that no attenuative derivation is produced.)
    assert not any(a.features.get("derivation") == ("imsi",) for a in analyzer.analyze("sarımsak"))


# --- Circumflex folding for lookup ---------------------------------------------------


@pytest.mark.positive
def test_circumflex_word_matches_plain_root(analyzer):
    # kâğıt folds to the plain root kağıt for lookup, but the original surface is preserved.
    a = _find(analyzer, "kâğıt", lemma="kağıt")
    assert a is not None and a.pos == "NOUN"
    assert a.surface == "kâğıt"  # circumflex preserved in the surface


@pytest.mark.positive
def test_circumflex_voicing_through_folding(analyzer):
    # kâğıdı folds to kağıdı and parses as kağıt + accusative (root voicing works through folding).
    assert has_analysis(analyzer, "kâğıdı", lemma="kağıt", features={"case": "accusative"})


@pytest.mark.positive
def test_circumflex_irregular_is_single_row(analyzer):
    # hâlâ folds to the 'hala' irregular key: exactly ONE ADV analysis (pins the duplicate-row
    # cleanup — a leftover hâlâ row would double it, since irregulars are prepended un-deduped).
    results = analyzer.analyze("hâlâ")
    assert len(results) == 1
    assert results[0].lemma == "hala" and results[0].pos == "ADV"
    assert results[0].surface == "hâlâ"


@pytest.mark.positive
def test_circumflex_alim(analyzer):
    a = analyzer.analyze("âlim")[0]
    assert a.lemma == "alim" and a.pos == "NOUN"


@pytest.mark.negative
def test_plain_hala_still_primary_adv(analyzer):
    # Regression: the plain 'hala' spelling still resolves to ADV hala primary.
    results = analyzer.analyze("hala")
    assert results[0].lemma == "hala" and results[0].pos == "ADV"


# --- Fraction / quantity numerals ----------------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize("word", ["yarım", "çeyrek", "buçuk"])
def test_fraction_numeral_has_num_reading(analyzer, word):
    assert has_analysis(analyzer, word, lemma=word, pos="NUM")


@pytest.mark.positive
def test_fraction_yarim_keeps_adj_homograph(analyzer):
    # yarım is BOTH a NUM (fraction "half") and the pre-existing ADJ; homographs are preserved.
    results = analyzer.analyze("yarım")
    assert any(a.pos == "NUM" and a.lemma == "yarım" for a in results)
    assert any(a.pos == "ADJ" and a.lemma == "yarım" for a in results)


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes",
    [
        ("çeyreği", "çeyrek", ["i"]),  # k -> ğ voicing before the accusative vowel
        ("buçuğu", "buçuk", ["u"]),  # k -> ğ voicing
        ("yarıma", "yarım", ["a"]),  # dative; m does NOT voice
    ],
)
def test_fraction_numeral_case_inflection(analyzer, word, lemma, morphemes):
    a = _find(analyzer, word, lemma=lemma, pos="NUM")
    assert a is not None and a.morphemes == morphemes


@pytest.mark.negative
def test_fraction_voicing_is_not_spurious(analyzer):
    # yarım ends in m (no voicing): yarığı has no yarım parse.
    assert not has_analysis(analyzer, "yarığı", lemma="yarım")
    # buçuk keeps its k before a consonant suffix (buçukta, not *buçuğta).
    a = _find(analyzer, "buçukta", lemma="buçuk", pos="NUM")
    assert a is not None and a.morphemes == ["ta"] and a.features.get("case") == "locative"


@pytest.mark.negative
@pytest.mark.parametrize(
    "word",
    ["yarımıncı", "yarımar", "çeyreğinci", "çeyreğer", "buçuğuncu", "buçuğar"],
)
def test_fraction_numeral_blocks_ordinal_and_distributive(analyzer, word):
    # yarım/çeyrek/buçuk carry "fraction": Turkish has no ordinal or distributive reading of a
    # fraction (no "*yarımıncı", no "*çeyreğer") — the productive NUM-wide -(I)ncI/-(ş)Ar edges
    # must NOT fire on them just because they share pos=NUM with the true cardinals (bir, iki).
    # (analyze() still returns the unmarked guesser fallback for any string; what must be absent
    # is a lexicon-sourced NUM analysis.)
    assert not any(a.pos == "NUM" and a.source == "lexicon" for a in analyzer.analyze(word))


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes",
    [("birinci", "bir", ["inci"]), ("ikişer", "iki", ["şer"]), ("onuncu", "on", ["uncu"])],
)
def test_fraction_attribute_does_not_affect_true_cardinals(analyzer, word, lemma, morphemes):
    # The excludes_attribute="fraction" guard on ORD/DIST is scoped to roots carrying the
    # attribute; true cardinals (unflagged) keep taking the ordinal/distributive as before.
    a = _find(analyzer, word, lemma=lemma, pos="NUM")
    assert a is not None and a.morphemes == morphemes


# --- Cross-cutting consistency -------------------------------------------------------


@pytest.mark.consistency
@pytest.mark.parametrize(
    "word,lemma",
    [("meslektaş", "meslek"), ("mavimsi", "mavi"), ("kâğıt", "kağıt")],
)
def test_stem_lemma_analyze_agree(word, lemma):
    assert ilmek.lemmatize(word) == lemma
    assert ilmek.analyze(word)[0].lemma == lemma
