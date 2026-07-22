"""Verb voice (çatı): causative, passive, reflexive, reciprocal.

Milestone scope (correctness over coverage): a bounded voice layer between the root and the
negation/tense chain, ordered reflexive/reciprocal < causative(<=2) < passive, marked in
``features['voice']`` as an ordered tuple (stacked voices are preserved).

* causative -DIr/-t (yaptır, okut, oturt) plus the lexically-limited -Ir/-Ar (içir, çıkar),
  with two-deep stacking (yaptırt, okuttur);
* passive -Il/-In/-n (yazıl, alın, okun), fully productive, chosen by the stem's final
  segment; the -In allomorph is genuinely also the reflexive, so both readings are emitted;
* reflexive -In (yıkan, giyin) and reciprocal -Iş (görüş, dövüş, anlaş), each semi-productive
  and gated by a curated root attribute; -Iş collides with the verbal-noun -(y)Iş, both kept.

Per the testing contract each rule carries positive, negative, and (where one exists)
exception cases, a long suffix chain (yazdırılmıştı), and a stem/lemma/analysis consistency
check. Deliberately-deferred forms (the bare voiced imperative, the double/impersonal passive)
are strict-xfail so the roadmap stays visible and no wrong Turkish is emitted meanwhile.
"""

from __future__ import annotations

import pytest
from conftest import has_analysis

import ilmek


def _find(analyzer, word, *, lemma=None, pos=None, voice=None):
    """First analysis of ``word`` matching the constraints (readings may coexist), or None."""
    for a in analyzer.analyze(word):
        if lemma is not None and a.lemma != lemma:
            continue
        if pos is not None and a.pos != pos:
            continue
        if voice is not None and a.features.get("voice") != voice:
            continue
        return a
    return None


# --- Causative: -DIr / -t / -Ir / -Ar, and stacking ---------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes,voice",
    [
        ("yaptırdı", "yap", ["tır", "dı"], ("causative",)),  # -DIr default
        ("güldürdü", "gül", ["dür", "dü"], ("causative",)),  # -DIr, D stays d after l
        ("okuttu", "oku", ["t", "tu"], ("causative",)),  # -t after a vowel
        ("oturttu", "otur", ["t", "tu"], ("causative",)),  # -t after r (polysyllable)
        ("içirdi", "iç", ["ir", "di"], ("causative",)),  # lexical -Ir
        ("yaptırttı", "yap", ["tır", "t", "tı"], ("causative", "causative")),  # stacked
        ("okutturdu", "oku", ["t", "tur", "du"], ("causative", "causative")),  # okut+tur
    ],
)
def test_causative(analyzer, word, lemma, morphemes, voice):
    a = _find(analyzer, word, lemma=lemma, voice=voice)
    assert a is not None, f"no {lemma}+{voice} analysis for {word}"
    assert a.morphemes == morphemes
    assert a.pos == "VERB"
    assert a.features.get("tense") == "past"
    assert a.source == "lexicon"


@pytest.mark.positive
def test_causative_ar_and_aorist_ambiguity_both_kept(analyzer):
    # çık takes the lexical -Ar causative (çıkar); the SAME surface çıkardı is also the
    # aorist+past-copula. Genuine ambiguity: both readings must survive.
    caus = _find(analyzer, "çıkardı", lemma="çık", voice=("causative",))
    assert caus is not None and caus.morphemes == ["ar", "dı"]
    assert caus.features.get("tense") == "past"
    # The pre-existing aorist + copular-past reading is untouched.
    assert has_analysis(
        analyzer, "çıkardı", lemma="çık", features={"tense": "aorist", "copula": "past"}
    )


# --- Passive: -Il / -In / -n, chosen by the stem's final segment --------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes",
    [
        ("yazıldı", "yaz", ["ıl", "dı"]),  # -Il after a plain consonant
        ("okundu", "oku", ["n", "du"]),  # -n after a vowel
        ("alındı", "al", ["ın", "dı"]),  # -In after l
        ("bulundu", "bul", ["un", "du"]),  # -In after l
    ],
)
def test_passive(analyzer, word, lemma, morphemes):
    a = _find(analyzer, word, lemma=lemma, voice=("passive",))
    assert a is not None, f"no {lemma}+passive analysis for {word}"
    assert a.morphemes == morphemes
    assert a.pos == "VERB"
    assert a.source == "lexicon"


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes",
    [
        ("okunur", "oku", ["n", "ur"]),  # passive + deterministic -Ir aorist
        ("yapılır", "yap", ["ıl", "ır"]),
        ("görüşürüz", "gör", ["üş", "ür", "üz"]),  # reciprocal + -Ir aorist + 1pl
    ],
)
def test_post_voice_aorist_is_deterministic_ir(analyzer, word, lemma, morphemes):
    # A voiced stem is always consonant-final, so its aorist is -Ir regardless of the root's
    # lexical class (proving AOR_VOICE ignores root.aorist).
    a = _find(analyzer, word, lemma=lemma)
    assert a is not None and a.morphemes == morphemes
    assert a.features.get("tense") == "aorist"


@pytest.mark.exception
def test_passive_on_bound_root_voices_before_linking_vowel(analyzer):
    # gidildi = git + (I)l + di: the (I) linking vowel is vowel-initial, so the root voices to
    # the bound form gid- before it.
    a = _find(analyzer, "gidildi", lemma="git", voice=("passive",))
    assert a is not None and a.morphemes == ["il", "di"]


@pytest.mark.exception
def test_passive_denir_proves_deterministic_aorist(analyzer):
    # denir = de + n(passive) + ir: the post-voice aorist is -Ir (denir), never -Ar (*dener).
    assert has_analysis(analyzer, "denir", lemma="de", features={"tense": "aorist"})
    assert _find(analyzer, "denir", lemma="de", voice=("passive",)) is not None


@pytest.mark.positive
def test_passive_before_negation(analyzer):
    # Voice is strictly before negation: yapılmadı = yap + ıl(passive) + ma(neg) + dı.
    a = _find(analyzer, "yapılmadı", lemma="yap", voice=("passive",))
    assert a is not None
    assert a.morphemes == ["ıl", "ma", "dı"]
    assert a.features.get("polarity") == "negative"


@pytest.mark.positive
def test_passive_negative_aorist(analyzer):
    # yapılmaz = yap + ıl(passive) + maz(neg-aorist): -mAz attaches to the voiced stem.
    a = _find(analyzer, "yapılmaz", lemma="yap", voice=("passive",))
    assert a is not None
    assert a.morphemes == ["ıl", "maz"]
    assert a.features.get("polarity") == "negative"
    assert a.features.get("tense") == "aorist"


# --- Reflexive -In, and the mandated passive/reflexive -In ambiguity -----------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes",
    [
        ("giyindi", "giy", ["in", "di"]),  # -In after a consonant (linking vowel)
        ("tarandı", "tara", ["n", "dı"]),  # -In after a vowel (new root)
        ("sevindi", "sev", ["in", "di"]),  # sev -> sevin- "rejoice"
    ],
)
def test_reflexive(analyzer, word, lemma, morphemes):
    a = _find(analyzer, word, lemma=lemma, voice=("reflexive",))
    assert a is not None, f"no {lemma}+reflexive analysis for {word}"
    assert a.morphemes == morphemes
    assert a.pos == "VERB"


@pytest.mark.exception
def test_yikandi_has_both_reflexive_and_passive(analyzer):
    # The milestone-mandated -In ambiguity: on a vowel/l-final attributed verb, -In is BOTH
    # reflexive and passive, so yıkandı carries both readings (lemma yıka on each).
    refl = _find(analyzer, "yıkandı", lemma="yıka", voice=("reflexive",))
    pas = _find(analyzer, "yıkandı", lemma="yıka", voice=("passive",))
    assert refl is not None and refl.morphemes == ["n", "dı"]
    assert pas is not None and pas.morphemes == ["n", "dı"]


# --- Reciprocal / collective -Iş ----------------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes,person",
    [
        ("görüştü", "gör", ["üş", "tü"], "3sg"),
        ("görüştüler", "gör", ["üş", "tü", "ler"], "3pl"),  # milestone example
        ("dövüştü", "döv", ["üş", "tü"], "3sg"),  # new root
        ("bakıştılar", "bak", ["ış", "tı", "lar"], "3pl"),
        ("gülüştü", "gül", ["üş", "tü"], "3sg"),  # gül gained the reciprocal attribute
    ],
)
def test_reciprocal(analyzer, word, lemma, morphemes, person):
    a = _find(analyzer, word, lemma=lemma, voice=("reciprocal",))
    assert a is not None, f"no {lemma}+reciprocal analysis for {word}"
    assert a.morphemes == morphemes
    assert a.features.get("tense") == "past"
    assert a.features.get("person") == person


@pytest.mark.positive
def test_gulustu_reciprocal_is_primary_over_noun_copula(analyzer):
    # gülüştü: the verbal reciprocal reading (0 derivations) outranks the gülüş+copula noun
    # reading (1 derivation), so it is primary; the -Iş verbal-noun reading survives alongside.
    best = analyzer.analyze("gülüştü")[0]
    assert best.lemma == "gül" and best.pos == "VERB"
    assert best.features.get("voice") == ("reciprocal",)
    # The bare -Iş verbal noun gülüş stays a NOUN (V_RECIP is non-final: no bare reciprocal verb).
    bare = analyzer.analyze("gülüş")[0]
    assert bare.lemma == "gül" and bare.pos == "NOUN"
    assert bare.features.get("derivation") == ("is",)


@pytest.mark.exception
def test_reciprocal_vowel_final_takes_bare_s(analyzer):
    # A vowel-final root takes the reciprocal -ş with NO (y) buffer (anlaş), distinct from the
    # verbal-noun anla-(y)Iş = anlayış.
    a = _find(analyzer, "anlaştık", lemma="anla", voice=("reciprocal",))
    assert a is not None
    assert a.morphemes == ["ş", "tı", "k"]
    assert a.features.get("person") == "1pl"


# --- Long chains and voice-with-derivation interaction ------------------------------


@pytest.mark.positive
def test_causative_passive_long_chain(analyzer):
    # Required long chain: yazdırılmıştı = yaz + dır(caus) + ıl(pass) + mış(evid) + tı(cop).
    a = _find(analyzer, "yazdırılmıştı", lemma="yaz", voice=("causative", "passive"))
    assert a is not None
    assert a.morphemes == ["dır", "ıl", "mış", "tı"]
    assert a.features.get("evidential") is True
    assert a.features.get("copula") == "past"


@pytest.mark.positive
def test_causative_passive_negative_future_copula(analyzer):
    # yaptırılmayacaktı = yap + tır + ıl + ma + yacak + tı.
    a = _find(analyzer, "yaptırılmayacaktı", lemma="yap", voice=("causative", "passive"))
    assert a is not None
    assert a.features.get("polarity") == "negative"
    assert a.features.get("tense") == "future"
    assert a.features.get("copula") == "past"


@pytest.mark.positive
def test_triple_voice_reciprocal_causative_passive(analyzer):
    # görüştürüldü = gör + üş(recip) + tür(caus) + ül(pass) + dü.
    a = _find(analyzer, "görüştürüldü", lemma="gör", voice=("reciprocal", "causative", "passive"))
    assert a is not None
    assert a.morphemes == ["üş", "tür", "ül", "dü"]


@pytest.mark.positive
def test_causative_feeds_ability(analyzer):
    # yaptırabilir = yap + tır(caus) + abil(ability) + ir(aorist).
    a = _find(analyzer, "yaptırabilir", lemma="yap", voice=("causative",))
    assert a is not None
    assert a.morphemes == ["tır", "abil", "ir"]
    assert a.features.get("ability") is True
    assert a.features.get("tense") == "aorist"


@pytest.mark.positive
def test_voice_feeds_verbal_noun(analyzer):
    # A voiced stem derives a verbal noun (-mA): the noun reading keeps voice AND derivation.
    a = _find(analyzer, "görüşme", lemma="gör", pos="NOUN", voice=("reciprocal",))
    assert a is not None
    assert a.features.get("derivation") == ("ma",)
    assert a.morphemes == ["üş", "me"]
    b = _find(analyzer, "yıkanma", lemma="yıka", pos="NOUN", voice=("reflexive",))
    assert b is not None and b.features.get("derivation") == ("ma",)


# --- Exceptions: lexically-irregular vowel-final causative (-DIr) --------------------


@pytest.mark.exception
def test_vowel_final_dir_causative_exceptions(analyzer):
    # ye/de are vowel-final but take -DIr (yedir, dedir), never the -t default (*yet, *det).
    a = _find(analyzer, "yedirdi", lemma="ye", voice=("causative",))
    assert a is not None and a.morphemes == ["dir", "di"]
    # ...and they still stack: dedirtti = de + dir(caus) + t(caus) + ti.
    b = _find(analyzer, "dedirtti", lemma="de", voice=("causative", "causative"))
    assert b is not None and b.morphemes == ["dir", "t", "ti"]


# --- Negatives: the guards must block overgeneration ---------------------------------


@pytest.mark.negative
def test_passive_il_guard_blocks_fake_passive(analyzer):
    # The -Il passive guard is consonant-not-l: it must NOT fire after the vowel of oku, so
    # 'okuldu' is only okul(NOUN)+copula-past, never a fake oku+passive.
    assert not has_analysis(analyzer, "okuldu", lemma="oku")
    best = analyzer.analyze("okuldu")[0]
    assert best.lemma == "okul" and best.pos == "NOUN"


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("içtirdi", "iç"),  # iç's causative class is Ir, so the -DIr edge is blocked
        ("çıkırdı", "çık"),  # çık is Ar, not Ir
    ],
)
def test_wrong_causative_allomorph_is_rejected(analyzer, word, lemma):
    assert not has_analysis(analyzer, word, lemma=lemma)
    assert analyzer.analyze(word)[0].source == "guess"


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("geldirdi", "gel"), ("gittirdi", "git")])
def test_suppletive_causative_never_generated(analyzer, word, lemma):
    # gel/git are causative="none" (suppletive getir/götür): *geldir / *gittir must not parse.
    assert not has_analysis(analyzer, word, lemma=lemma)
    assert analyzer.analyze(word)[0].source == "guess"


@pytest.mark.negative
def test_reciprocal_requires_attribute(analyzer):
    # gel lacks the 'reciprocal' attribute, so 'gelişti' gains NO reciprocal reading and the
    # existing geliş verbal-noun reading is untouched.
    assert not has_analysis(analyzer, "gelişti", lemma="gel", features={"voice": ("reciprocal",)})
    assert has_analysis(analyzer, "geliş", lemma="gel", features={"derivation": ("is",)})


@pytest.mark.negative
def test_passive_is_not_also_reflexive_without_attribute(analyzer):
    # yap is not reflexive-gated: yapıldı has ONLY a passive reading, no reflexive one.
    assert _find(analyzer, "yapıldı", lemma="yap", voice=("passive",)) is not None
    assert _find(analyzer, "yapıldı", lemma="yap", voice=("reflexive",)) is None


@pytest.mark.negative
def test_primaries_unchanged_by_voice(analyzer):
    # High-frequency forms keep their pre-voice primary analysis.
    for word, lemma, pos in [
        ("geldi", "gel", "VERB"),
        ("yaptı", "yap", "VERB"),
        ("gördü", "gör", "VERB"),
        ("görüş", "gör", "NOUN"),  # the -Iş verbal noun, not a reciprocal imperative
        ("sorun", "sorun", "NOUN"),  # lexicon noun, not sor+un
        ("alın", "alın", "NOUN"),  # lexicon noun, not al+ın
    ]:
        best = analyzer.analyze(word)[0]
        assert best.lemma == lemma and best.pos == pos
        assert "voice" not in best.features


@pytest.mark.negative
def test_guesser_never_voice_splits(analyzer):
    # Voice is off for the guesser: an OOV word is never split on a (productive) passive, and
    # no guess result carries a voice feature.
    for r in analyzer.analyze("zorlatıldı"):
        assert r.source != "lexicon"
        assert "voice" not in r.features


# --- Consistency: stem / lemma / analyze agree (voice is not derivational) -----------


@pytest.mark.consistency
@pytest.mark.parametrize(
    "word,root",
    [("yaptırdı", "yap"), ("yazıldı", "yaz"), ("yıkandı", "yıka"), ("görüştürüldü", "gör")],
)
def test_voice_is_not_derivational_stem_equals_lemma(word, root):
    # Voice keeps stem and lemma at the root (the milestone goal): yaptırdı -> stem/lemma yap.
    assert ilmek.lemmatize(word) == root
    assert ilmek.stem(word) == root
    # The best analysis need not be the voiced one (real ambiguity), but every reading of a
    # voiced form shares the root as both stem and lemma, with no derivation boundary.
    for a in ilmek.analyze(word):
        if "voice" in a.features:
            assert a.lemma == a.stem == root
            assert "derivation" not in a.features


# --- Documented deferrals (strict xfail): honest known limitations -------------------


@pytest.mark.exception
@pytest.mark.xfail(
    reason="Bare voiced 2sg imperative (yıkan!, görüş!) is deferred: the voice states are "
    "deliberately non-final to protect the noun primaries (sorun, alın, görüş).",
    strict=True,
)
@pytest.mark.parametrize(
    "word,lemma,voice", [("yıkan", "yıka", ("reflexive",)), ("görüş", "gör", ("reciprocal",))]
)
def test_bare_voiced_imperative_deferred(analyzer, word, lemma, voice):
    assert has_analysis(
        analyzer, word, lemma=lemma, pos="VERB", features={"voice": voice, "mood": "imperative"}
    )


@pytest.mark.exception
@pytest.mark.xfail(
    reason="Double/lexicalized passive (de-n-il = denil) is deferred: V_PASS takes no second "
    "passive by design.",
    strict=True,
)
def test_double_passive_denildi_deferred(analyzer):
    assert has_analysis(analyzer, "denildi", lemma="de", features={"voice": ("passive", "passive")})


@pytest.mark.exception
@pytest.mark.xfail(reason="Impersonal double passive (ara-n-ıl = aranıl) is deferred.", strict=True)
def test_impersonal_double_passive_deferred(analyzer):
    assert has_analysis(analyzer, "aranıldı", lemma="ara")
