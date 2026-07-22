"""Verb morphology: negation, tense/aspect, person, copular stacking, voicing."""

from __future__ import annotations

import pytest
from conftest import has_analysis


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("geldi", "gel", {"tense": "past", "person": "3sg"}),
        ("geldim", "gel", {"tense": "past", "person": "1sg"}),
        ("geldiniz", "gel", {"tense": "past", "person": "2pl"}),
        ("geliyor", "gel", {"aspect": "progressive", "person": "3sg"}),
        ("geliyorum", "gel", {"aspect": "progressive", "person": "1sg"}),
        ("gelecek", "gel", {"tense": "future", "person": "3sg"}),
        ("gelmiş", "gel", {"evidential": True, "person": "3sg"}),
        ("gelmişsiniz", "gel", {"evidential": True, "person": "2pl"}),
    ],
)
def test_basic_verb_inflection(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, features=features)


@pytest.mark.positive
def test_negation(analyzer):
    assert has_analysis(
        analyzer, "gelmedi", lemma="gel", features={"polarity": "negative", "tense": "past"}
    )
    assert has_analysis(
        analyzer,
        "gelmiyor",
        lemma="gel",
        features={"polarity": "negative", "aspect": "progressive"},
    )


@pytest.mark.positive
def test_showcase_verb_full_chain(analyzer):
    # gelmeyecekmişsiniz = gel + Neg + Future + Evidential + 2Pl
    assert has_analysis(
        analyzer,
        "gelmeyecekmişsiniz",
        lemma="gel",
        morphemes=["me", "yecek", "miş", "siniz"],
        features={
            "polarity": "negative",
            "tense": "future",
            "evidential": True,
            "person": "2pl",
        },
    )


@pytest.mark.positive
def test_future_person_softens_final_k(analyzer):
    # gelecek + Im -> geleceğim (final k of -AcAk softens before the vowel).
    assert has_analysis(
        analyzer, "geleceğim", lemma="gel", features={"tense": "future", "person": "1sg"}
    )


@pytest.mark.exception
def test_verb_voicing_before_vowel_only(analyzer):
    # git: gidiyor / gidecek (t->d before vowel) but gitti (t stays before -DI).
    assert has_analysis(analyzer, "gidiyor", lemma="git", features={"aspect": "progressive"})
    assert has_analysis(analyzer, "gidecek", lemma="git", features={"tense": "future"})
    assert has_analysis(analyzer, "gitti", lemma="git", features={"tense": "past"})


@pytest.mark.negative
def test_non_voicing_verb_stays(analyzer):
    # yap does not voice: yapıyor (p stays), never yabıyor.
    assert has_analysis(analyzer, "yapıyor", lemma="yap", features={"aspect": "progressive"})
    assert not has_analysis(analyzer, "yabıyor", lemma="yap")


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma",
    [("okuyor", "oku"), ("başlıyor", "başla"), ("yürüyor", "yürü"), ("arıyor", "ara")],
)
def test_progressive_vowel_drop(analyzer, word, lemma):
    # -Iyor deletes a preceding stem-final vowel: oku -> okuyor, başla -> başlıyor.
    assert has_analysis(analyzer, word, lemma=lemma, features={"aspect": "progressive"})


@pytest.mark.positive
def test_bare_verb_root_is_imperative(analyzer):
    assert has_analysis(
        analyzer, "gel", lemma="gel", pos="VERB", features={"mood": "imperative", "person": "2sg"}
    )


@pytest.mark.exception
@pytest.mark.parametrize("word,lemma", [("diyor", "de"), ("yiyor", "ye")])
def test_irregular_de_ye_progressive(analyzer, word, lemma):
    # de-/ye- raise their /e/ to /i/ before the vowel-initial -Iyor (de -> diyor, ye -> yiyor).
    # The drop of the CV root's only vowel would otherwise misharmonize to *dıyor; the engine
    # realizes -Iyor against the pre-drop stem so the high vowel raises correctly.
    assert has_analysis(analyzer, word, lemma=lemma, features={"aspect": "progressive"})


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("dıyor", "de"), ("yıyor", "ye")])
def test_de_ye_progressive_without_raising_is_rejected(analyzer, word, lemma):
    # The un-raised back-harmony surface (*dıyor / *yıyor) must NOT be a lexicon analysis: the
    # glide raising is obligatory, so only diyor/yiyor parse (this was silently accepted before).
    assert not has_analysis(analyzer, word, lemma=lemma)
    assert analyzer.analyze(word)[0].source == "guess"


@pytest.mark.exception
def test_regular_de_ye_forms_work(analyzer):
    # The raising is confined to vowel-initial suffixes; the regular (non-glide) forms are
    # unchanged: dedi, demiş (consonant-initial suffix, no raising).
    assert has_analysis(analyzer, "dedi", lemma="de", features={"tense": "past"})
    assert has_analysis(analyzer, "demiş", lemma="de", features={"evidential": True})


@pytest.mark.exception
@pytest.mark.parametrize(
    "word,lemma,morphemes,features",
    [
        ("diyecek", "de", ["yecek"], {"tense": "future"}),  # -(y)AcAk raises de -> di
        ("yiyecek", "ye", ["yecek"], {"tense": "future"}),
        ("diye", "de", ["ye"], {"mood": "optative"}),  # -(y)A raises de -> di
        ("yiye", "ye", ["ye"], {"mood": "optative"}),
        ("diyebilir", "de", ["yebil", "ir"], {"ability": True}),  # -(y)Abil raises too
    ],
)
def test_irregular_de_ye_raising_before_vowel_suffixes(analyzer, word, lemma, morphemes, features):
    assert has_analysis(analyzer, word, lemma=lemma, morphemes=morphemes, features=features)


@pytest.mark.exception
def test_de_ye_raising_softens_future_k(analyzer):
    # diyeceğim = de -> di + (y)AcAk + (y)Im: raising AND the -AcAk final k softening compose.
    assert has_analysis(
        analyzer, "diyeceğim", lemma="de", features={"tense": "future", "person": "1sg"}
    )


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("deyecek", "de"), ("yeyecek", "ye")])
def test_de_ye_future_without_raising_is_rejected(analyzer, word, lemma):
    # Once the raised base replaces the free base, the un-raised *deyecek/*yeyecek no longer
    # parse (they were accepted before the raising rule landed).
    assert not has_analysis(analyzer, word, lemma=lemma)


@pytest.mark.exception
def test_de_ye_verbal_noun_stays_regular(analyzer):
    # The -(y)Iş verbal noun is deliberately NOT a raising context: deyiş (a real lexicalized
    # word "manner of saying"), never *diyiş. This is the exception that guards over-raising.
    assert has_analysis(analyzer, "deyiş", lemma="de", features={"derivation": ("is",)})
    assert not has_analysis(analyzer, "diyiş", lemma="de")


# --- Negative imperative -------------------------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize("word,lemma", [("gelme", "gel"), ("yapma", "yap"), ("gitme", "git")])
def test_negative_imperative(analyzer, word, lemma):
    # A bare negated stem is a 2sg negative imperative ("gelme!" = don't come).
    assert has_analysis(
        analyzer, word, lemma=lemma, features={"polarity": "negative", "person": "2sg"}
    )


# --- Copular (ek-fiil) stacking with the -y- buffer ----------------------------------


@pytest.mark.positive
def test_copula_takes_y_buffer_after_vowel(analyzer):
    # geldi + (y)di -> geldiydi ; geldi + (y)miş -> geldiymiş
    assert has_analysis(
        analyzer, "geldiydi", lemma="gel", features={"tense": "past", "copula": "past"}
    )
    assert has_analysis(
        analyzer, "geldiymiş", lemma="gel", features={"tense": "past", "evidential": True}
    )


@pytest.mark.positive
def test_copula_after_consonant_has_no_buffer_and_keeps_tense(analyzer):
    # gelecek + ti -> gelecekti; the primary future feature must NOT be overwritten.
    assert has_analysis(
        analyzer, "gelecekti", lemma="gel", features={"tense": "future", "copula": "past"}
    )


@pytest.mark.negative
@pytest.mark.parametrize("nonword", ["geldidi", "geldimiş"])
def test_copula_without_buffer_is_not_a_word(analyzer, nonword):
    # The engine must NOT accept the buffer-less non-words as lexicon analyses.
    results = analyzer.analyze(nonword)
    assert results[0].source == "guess"
    assert not has_analysis(analyzer, nonword, lemma="gel")


# --- The verb "ol" (olmak, the most common Turkish verb) -----------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,morphemes,features",
    [
        ("oldu", ["du"], {"tense": "past", "person": "3sg"}),
        ("olur", ["ur"], {"tense": "aorist", "person": "3sg"}),  # -Ir class, not *olar
        ("olmaz", ["maz"], {"polarity": "negative", "tense": "aorist"}),
    ],
)
def test_ol_finite_inflection(analyzer, word, morphemes, features):
    assert has_analysis(
        analyzer, word, lemma="ol", pos="VERB", morphemes=morphemes, features=features
    )


@pytest.mark.positive
def test_olmadiktan_neg_participle_ablative(analyzer):
    # The milestone headline: olmadıktan = ol + Neg(-mA) + Participle(-DIk) + Ablative(-tAn).
    # The -DIk participle derives a nominal, so the accepting pos is ADJ, lemma stays ol.
    best = analyzer.analyze("olmadıktan")[0]
    assert best.lemma == "ol"
    assert best.source == "lexicon"
    assert best.morphemes == ["ma", "dık", "tan"]
    assert best.features.get("polarity") == "negative"
    assert best.features.get("case") == "ablative"


@pytest.mark.positive
def test_olmak_infinitive_is_lexicon_ol(analyzer):
    # -mAk infinitive derives a verbal noun; lemma is still ol.
    assert has_analysis(analyzer, "olmak", lemma="ol", morphemes=["mak"])
    assert analyzer.analyze("olmak")[0].source == "lexicon"


@pytest.mark.negative
def test_ol_aorist_is_ir_not_ar(analyzer):
    # ol is in the ~13 -Ir monosyllable class: the -Ar aorist edge is guarded off, so *olar
    # is never a lexicon analysis of ol (it falls back to an honest OOV guess instead).
    assert not has_analysis(analyzer, "olar", lemma="ol")
    assert analyzer.analyze("olar")[0].source == "guess"


@pytest.mark.positive
def test_ol_ability_chain(analyzer):
    # Ability + aorist: olabilir = ol + (y)Abil + Ir.
    assert has_analysis(
        analyzer, "olabilir", lemma="ol", morphemes=["abil", "ir"], features={"ability": True}
    )


@pytest.mark.consistency
def test_ol_long_chain_and_views_agree(analyzer):
    import ilmek

    # Long chain: negation + participle + plural + possessive + case still lemmatizes to ol.
    assert ilmek.lemmatize("olmadıklarından") == "ol"
    assert has_analysis(analyzer, "olmadıklarından", lemma="ol", features={"case": "ablative"})
    # Three views of the participle chain agree: lemma is the root, stem is the surface at the
    # last derivation boundary (-DIk), per the project's stem contract.
    best = analyzer.analyze("olmadıktan")[0]
    assert ilmek.lemmatize("olmadıktan") == best.lemma == "ol"
    assert ilmek.stem("olmadıktan") == best.stem == "olmadık"
