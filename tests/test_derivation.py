"""Derivational morphology: a single derivation slot between root and inflection.

The engine grows a *derived* stem (evli, yaşadık, gelme) from a root, marks the derivation
boundary in ``features['derivation']`` (an ordered tuple of suffix names), and then lets the
derived stem inflect normally (evlilerden, yolcular, yaşadıklarımızın). The lemma stays the
base lexeme; the stem is the surface at the last derivation boundary.

Coverage per the testing contract: positive (rule applies), negative (rule must not apply),
exception (phonology twists: C-hardening, k->ğ softening, (y) buffers), a long suffix chain,
and a consistency check across stem/lemma/analyze. Crucially, the file also pins the
*overgeneration guards*: adding derivation must NOT flip the primary analysis of common
words (geldik stays finite past, gelme stays negative imperative, gözlük/yemek/elli stay
their lexicon reading).
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


# --- Nominal derivation: -lI, -sIz, -lIk, -CI (positive) -----------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,pos,morphemes,derivation",
    [
        ("evli", "ev", "ADJ", ["li"], ("li",)),  # -lI  (having a house)
        ("evsiz", "ev", "ADJ", ["siz"], ("siz",)),  # -sIz (houseless)
        ("kitaplık", "kitap", "NOUN", ["lık"], ("lik",)),  # -lIk (bookcase)
        ("güzellik", "güzel", "NOUN", ["lik"], ("lik",)),  # -lIk on an ADJ base (beauty)
        ("yolcu", "yol", "NOUN", ["cu"], ("ci",)),  # -CI  (traveller)
    ],
)
def test_nominal_derivation(analyzer, word, lemma, pos, morphemes, derivation):
    a = _find(analyzer, word, lemma=lemma, pos=pos, derivation=derivation)
    assert a is not None, f"no {lemma}+{derivation} analysis for {word}"
    assert a.morphemes == morphemes
    assert a.stem == word  # stem is the derived surface, lemma is the base lexeme
    assert a.lemma == lemma
    assert a.source == "lexicon"


@pytest.mark.positive
def test_derivation_boundary_is_visible(analyzer):
    # The derivation tuple makes the derivational-vs-inflectional boundary explicit.
    a = _find(analyzer, "evli", lemma="ev", derivation=("li",))
    assert a.features["derivation"] == ("li",)
    # A purely inflected word carries no derivation key and stem == lemma.
    b = analyzer.analyze("evler")[0]
    assert "derivation" not in b.features
    assert b.stem == b.lemma == "ev"


# --- -CI hardening: C -> ç after a voiceless consonant (exception) -------------------


@pytest.mark.exception
@pytest.mark.parametrize(
    "word,lemma,morph",
    [
        ("kitapçı", "kitap", "çı"),  # C hardens after voiceless p
        ("işçi", "iş", "çi"),  # C hardens after voiceless ş
    ],
)
def test_ci_hardens_after_voiceless(analyzer, word, lemma, morph):
    a = _find(analyzer, word, lemma=lemma, derivation=("ci",))
    assert a is not None and a.morphemes == [morph]


@pytest.mark.negative
def test_ci_stays_voiced_after_voiced(analyzer):
    # After a voiced consonant (l) the affricate stays c: yolcu, never *yolçu.
    assert has_analysis(analyzer, "yolcu", lemma="yol", morphemes=["cu"])
    assert not has_analysis(analyzer, "yolçu", lemma="yol")


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("kitapcı", "kitap"), ("işci", "iş")])
def test_unhardened_ci_is_rejected(analyzer, word, lemma):
    # The un-hardened spellings are not valid Turkish and must not parse to the root.
    assert not has_analysis(analyzer, word, lemma=lemma)


# --- Negatives: harmony, applies_to gating, no stacking ------------------------------


@pytest.mark.negative
def test_derivation_respects_vowel_harmony(analyzer):
    # -lI harmonizes to -li after e; the rounded *evlü violates I-harmony.
    assert has_analysis(analyzer, "evli", lemma="ev")
    assert not has_analysis(analyzer, "evlü", lemma="ev")


@pytest.mark.negative
def test_ci_does_not_apply_to_adjective(analyzer):
    # -CI is applies_to={NOUN}; on the ADJ 'güzel' it must not fire (no *güzelci).
    assert not has_analysis(analyzer, "güzelci", lemma="güzel")


@pytest.mark.negative
def test_li_does_not_apply_to_adjective(analyzer):
    # -lI is applies_to={NOUN}: it must not stack onto an adjective either.
    assert not has_analysis(analyzer, "güzelli", lemma="güzel")


@pytest.mark.negative
def test_no_derivation_stacking_single_slot(analyzer):
    # Single derivation slot: evli (ev+lI) cannot take a second derivation -sIz.
    assert not has_analysis(analyzer, "evlisiz", lemma="ev")


# --- Verb -> noun: -mA, -(y)Iş, infinitive -mAk (positive) --------------------------


@pytest.mark.positive
def test_verbal_noun_ma(analyzer):
    a = _find(analyzer, "gelme", lemma="gel", pos="NOUN", derivation=("ma",))
    assert a is not None and a.morphemes == ["me"] and a.stem == "gelme"


@pytest.mark.positive
def test_verbal_noun_is(analyzer):
    a = _find(analyzer, "geliş", lemma="gel", pos="NOUN", derivation=("is",))
    assert a is not None and a.morphemes == ["iş"]


@pytest.mark.exception
def test_verbal_noun_is_takes_y_buffer_after_vowel(analyzer):
    # -(y)Iş: the (y) buffer appears after a vowel-final verb root (yürü -> yürüyüş).
    a = _find(analyzer, "yürüyüş", lemma="yürü", derivation=("is",))
    assert a is not None and a.morphemes == ["yüş"]


@pytest.mark.positive
def test_infinitive_mak(analyzer):
    a = _find(analyzer, "gelmek", lemma="gel", pos="NOUN", derivation=("mak",))
    assert a is not None and a.morphemes == ["mek"]


@pytest.mark.negative
def test_infinitive_is_terminal_no_case_yet(analyzer):
    # V_INF is terminal this milestone: -mAk does not inflect for case (gelmeği rejected).
    assert not has_analysis(analyzer, "gelmeği", lemma="gel")


# --- Verb -> adjective participles: -(y)An, -DIk, -(y)AcAk --------------------------


@pytest.mark.positive
def test_participle_an(analyzer):
    a = _find(analyzer, "gelen", lemma="gel", pos="ADJ", derivation=("an",))
    assert a is not None and a.morphemes == ["en"]


@pytest.mark.exception
def test_participle_an_after_negation_takes_y_buffer(analyzer):
    # NEG feeds the participle: gelme + (y)An -> gelmeyen, keeping polarity=negative.
    a = _find(analyzer, "gelmeyen", lemma="gel", derivation=("an",))
    assert a is not None
    assert a.morphemes == ["me", "yen"]
    assert a.features.get("polarity") == "negative"
    # A verb-derived nominal must NOT get a fabricated verbal person/mood.
    assert "mood" not in a.features


@pytest.mark.positive
def test_participle_dik(analyzer):
    a = _find(analyzer, "yaşadık", lemma="yaşa", pos="ADJ", derivation=("dik",))
    assert a is not None and a.morphemes == ["dık"]


@pytest.mark.exception
def test_participle_dik_softens_k_before_vowel(analyzer):
    # -DIk final k -> ğ before a vowel-initial suffix: bil+DIk+I -> bildiği.
    a = _find(analyzer, "bildiği", lemma="bil", derivation=("dik",))
    assert a is not None
    assert a.morphemes == ["dik", "i"]
    assert a.features.get("possessive") == "3sg"


@pytest.mark.exception
def test_lik_softens_k_before_vowel(analyzer):
    # -lIk final k -> ğ before a vowel: kitap+lIk+I -> kitaplığı.
    a = _find(analyzer, "kitaplığı", lemma="kitap", derivation=("lik",))
    assert a is not None and a.morphemes == ["lık", "ı"]


@pytest.mark.positive
def test_participle_acak_inflects(analyzer):
    a = _find(analyzer, "geleceği", lemma="gel", derivation=("acak",))
    assert a is not None
    assert a.morphemes == ["ecek", "i"]
    assert a.features.get("possessive") == "3sg"


# --- Derived stems inflect normally (positive; long chain) --------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,stem,morphemes,features,derivation",
    [
        (
            "evlilerden",
            "ev",
            "evli",
            ["li", "ler", "den"],
            {"number": "plural", "case": "ablative"},
            ("li",),
        ),
        ("yolcular", "yol", "yolcu", ["cu", "lar"], {"number": "plural"}, ("ci",)),
        ("gelmesi", "gel", "gelme", ["me", "si"], {"possessive": "3sg"}, ("ma",)),
        ("gelenler", "gel", "gelen", ["en", "ler"], {"number": "plural"}, ("an",)),
    ],
)
def test_derived_stem_inflects(analyzer, word, lemma, stem, morphemes, features, derivation):
    a = _find(analyzer, word, lemma=lemma, derivation=derivation)
    assert a is not None
    assert a.morphemes == morphemes
    assert a.stem == stem
    assert all(a.features.get(k) == v for k, v in features.items())


# --- Milestone target: yaşadıklarımızın -> lemma yaşa (via -DIk) --------------------


@pytest.mark.positive
def test_milestone_target_yasadiklarimizin(analyzer):
    a = _find(analyzer, "yaşadıklarımızın", lemma="yaşa", derivation=("dik",))
    assert a is not None
    assert a.lemma == "yaşa"
    assert a.stem == "yaşadık"
    assert a.morphemes == ["dık", "lar", "ımız", "ın"]
    assert a.features.get("number") == "plural"
    assert a.features.get("possessive") == "1pl"
    assert a.features.get("case") == "genitive"
    assert a.pos == "ADJ"


@pytest.mark.consistency
def test_milestone_target_views_agree():
    # stem/lemma/analyze are three views of one analysis and must agree.
    assert ilmek.lemmatize("yaşadıklarımızın") == "yaşa"
    assert ilmek.stem("yaşadıklarımızın") == "yaşadık"
    best = ilmek.analyze("yaşadıklarımızın")[0]
    assert best.lemma == "yaşa" and best.stem == "yaşadık"


# --- Overgeneration guards: the primary analysis of common words must NOT flip -------


@pytest.mark.negative
def test_gelme_primary_is_negative_imperative(analyzer):
    # gelme is ambiguous: the finite negative imperative must stay primary; the verbal
    # noun (derivation) is present only as a ranked alternative.
    results = analyzer.analyze("gelme")
    assert results[0].pos == "VERB"
    assert results[0].features.get("polarity") == "negative"
    assert "derivation" not in results[0].features
    assert any(r.features.get("derivation") == ("ma",) for r in results)


@pytest.mark.negative
def test_geldik_primary_is_finite_past(analyzer):
    # -DIk collides with past+1pl: the finite reading must win, participle stays an alt.
    results = analyzer.analyze("geldik")
    assert results[0].pos == "VERB"
    assert results[0].features.get("tense") == "past"
    assert results[0].features.get("person") == "1pl"
    assert any(r.features.get("derivation") == ("dik",) for r in results)


@pytest.mark.negative
def test_gelecek_primary_is_finite_future(analyzer):
    # -(y)AcAk collides with the finite future: finite wins, participle stays an alt.
    results = analyzer.analyze("gelecek")
    assert results[0].pos == "VERB"
    assert results[0].features.get("tense") == "future"
    assert "derivation" not in results[0].features
    assert any(r.features.get("derivation") == ("acak",) for r in results)


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,lemma,pos",
    [
        ("gözlük", "gözlük", "NOUN"),  # lexicon whole word beats göz+lük
        ("yemek", "yemek", "NOUN"),  # NOUN entry beats ye+mek infinitive
        ("elli", "elli", "NUM"),  # NUM entry beats el+li
    ],
)
def test_lexicon_whole_word_beats_derived_split(analyzer, word, lemma, pos):
    best = analyzer.analyze(word)[0]
    assert best.lemma == lemma
    assert best.pos == pos
    assert "derivation" not in best.features


@pytest.mark.negative
def test_derived_readings_kept_as_alternatives(analyzer):
    # The split reading is never erased — genuine ambiguity is preserved.
    assert any(
        r.lemma == "göz" and r.features.get("derivation") == ("lik",)
        for r in analyzer.analyze("gözlük")
    )
    assert any(
        r.lemma == "el" and r.features.get("derivation") == ("li",)
        for r in analyzer.analyze("elli")
    )


# --- Regression: plain inflection is byte-identical (consistency) --------------------


@pytest.mark.consistency
@pytest.mark.parametrize(
    "word,lemma", [("evler", "ev"), ("kitaplarımızdan", "kitap"), ("geldim", "gel")]
)
def test_plain_inflection_unchanged(analyzer, word, lemma):
    best = analyzer.analyze(word)[0]
    assert best.lemma == lemma
    assert best.source == "lexicon"
    assert "derivation" not in best.features
    assert best.stem == best.lemma  # no derivation -> stem equals lemma


@pytest.mark.negative
def test_guesser_does_not_strip_derivation(analyzer):
    # The guesser forbids derivation, so an unknown word is never split on -lIk/-sIz etc.
    # 'malik' must not become ma+lik; it stays a conservative identity/guess, never lemma 'ma'.
    for r in analyzer.analyze("malik"):
        assert r.lemma != "ma"
        assert r.source != "lexicon"


# --- Documented deferrals (strict xfail): honest known limitations -------------------


@pytest.mark.exception
@pytest.mark.xfail(reason="Infinitive (-mAk) case inflection is a later milestone.", strict=True)
def test_infinitive_case_deferred(analyzer):
    assert has_analysis(analyzer, "gelmekten", lemma="gel", features={"case": "ablative"})


@pytest.mark.exception
@pytest.mark.xfail(reason="Derivation stacking (-lI + -lIk) is a later milestone.", strict=True)
def test_derivation_stacking_deferred(analyzer):
    assert has_analysis(analyzer, "evlilik", lemma="ev", features={"derivation": ("li", "lik")})
