"""Completing the sıfat-fiil (participle) inventory: -mIş, aorist -(A/I)r, negative -mAz.

The repo already had the -(y)An / -DIk / -(y)AcAk participles (verb->ADJ derivations). This
module covers the three added ones, modelled on those:

* the -mIş participle (completed-action adjective): pişmiş, geçmiş, okunmuş, kırılmış —
  DISTINCT from the finite evidential -mIş, so gelmiş is genuinely ambiguous (both are kept);
* the aorist participle -(A/I)r (akar, güler, gelir), reusing the SAME lexically-irregular
  aorist allomorph mechanism as the finite aorist (the ``aorist_class`` guard);
* the negative-aorist participle -mAz (bitmez, çıkmaz, tükenmez).

All three keep lemma/stem = the verb root, are marked verbform=participle, and inflect
nominally afterwards (geçmişi, akarlar, çıkmazda). Per the testing contract each rule carries
positive, negative/ambiguity, and (where one exists) exception cases, plus the overgeneration
guards: the finite reading of every homograph surface stays primary, and the deliberately-
defective finite negative-aorist persons (*gelmezim / *gelmeziz) are NOT resurrected.
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


# --- The -mIş participle (verb -> ADJ), distinct from the finite evidential -----------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes",
    [
        ("pişmiş", "piş", ["miş"]),  # cooked (new lexicon root piş)
        ("geçmiş", "geç", ["miş"]),  # past / bygone
    ],
)
def test_mis_participle(analyzer, word, lemma, morphemes):
    a = _find(analyzer, word, lemma=lemma, pos="ADJ", derivation=("mis",))
    assert a is not None, f"no {lemma}+mis participle for {word}"
    assert a.morphemes == morphemes
    assert a.stem == word  # stem is the derived surface
    assert a.features.get("verbform") == "participle"
    assert a.source == "lexicon"


@pytest.mark.positive
def test_mis_participle_over_a_voiced_stem(analyzer):
    # The -mIş participle fires from a voice state too (passive), keeping the voice feature.
    a = _find(analyzer, "okunmuş", lemma="oku", pos="ADJ", derivation=("mis",))
    assert a is not None
    assert a.morphemes == ["n", "muş"]
    assert a.features.get("voice") == ("passive",)
    assert a.features.get("verbform") == "participle"

    b = _find(analyzer, "kırılmış", lemma="kır", pos="ADJ", derivation=("mis",))
    assert b is not None
    assert b.morphemes == ["ıl", "mış"]
    assert b.features.get("voice") == ("passive",)


@pytest.mark.positive
def test_mis_participle_after_negation(analyzer):
    # NEG feeds the participle: gelme + mIş -> gelmemiş, keeping polarity=negative.
    a = _find(analyzer, "gelmemiş", lemma="gel", pos="ADJ", derivation=("mis",))
    assert a is not None
    assert a.morphemes == ["me", "miş"]
    assert a.features.get("polarity") == "negative"
    # A verb-derived nominal must NOT get a fabricated verbal person/mood.
    assert "mood" not in a.features and "person" not in a.features


@pytest.mark.negative
def test_gelmis_is_ambiguous_evidential_and_participle(analyzer):
    # gelmiş = finite evidential (primary) AND the -mIş participle. Both are returned; lemma
    # stays gel. (Exact clone of the -(y)AcAk finite-primary test.)
    results = analyzer.analyze("gelmiş")
    assert results[0].pos == "VERB"
    assert results[0].features.get("evidential") is True
    assert "derivation" not in results[0].features
    assert any(
        r.pos == "ADJ"
        and r.features.get("derivation") == ("mis",)
        and r.features.get("verbform") == "participle"
        and r.lemma == "gel"
        for r in results
    )


@pytest.mark.negative
@pytest.mark.parametrize("word", ["gelmiş", "gelmişim", "gelmişti", "gelmişsiniz"])
def test_finite_evidential_stays_primary(analyzer, word):
    # The finite evidential reading of every -mIş surface is unchanged and ranked first.
    best = analyzer.analyze(word)[0]
    assert best.pos == "VERB"
    assert best.features.get("evidential") is True
    assert "derivation" not in best.features


# --- The aorist participle -(A/I)r (reusing the lexical aorist_class guard) -----------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes",
    [
        ("akar", "ak", ["ar"]),  # class Ar (default monosyllable); new root ak
        ("güler", "gül", ["er"]),  # class Ar
        ("gelir", "gel", ["ir"]),  # class Ir (lexical exception) -> proves aorist_class reuse
        ("okur", "oku", ["r"]),  # class r (vowel-final)
        ("yapar", "yap", ["ar"]),  # class Ar
        ("oturur", "otur", ["ur"]),  # class Ir (polysyllable)
    ],
)
def test_aorist_participle(analyzer, word, lemma, morphemes):
    a = _find(analyzer, word, lemma=lemma, pos="ADJ", derivation=("ar",))
    assert a is not None, f"no {lemma}+aorist participle for {word}"
    assert a.morphemes == morphemes
    assert a.stem == word
    assert a.features.get("verbform") == "participle"
    # The participle is NOT finite: it carries no tense.
    assert "tense" not in a.features


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("gelir", "gel"), ("yapar", "yap"), ("okur", "oku")])
def test_finite_aorist_stays_primary(analyzer, word, lemma):
    # "Do not break the finite aorist": the finite aorist reading is ranked first, the
    # participle survives only as an alternative.
    best = analyzer.analyze(word)[0]
    assert best.pos == "VERB"
    assert best.features.get("tense") == "aorist"
    assert "derivation" not in best.features


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("geler", "gel"), ("yapır", "yap")])
def test_wrong_allomorph_aorist_participle_is_rejected(analyzer, word, lemma):
    # The aorist participle reuses the SAME aorist_class guard as the finite aorist: gel is
    # class Ir, so the -Ar edge never fires (no *geler participle either), and yap is class Ar,
    # so the -Ir edge never fires (no *yapır). Both stay honest OOV guesses.
    assert not has_analysis(analyzer, word, lemma=lemma)
    assert analyzer.analyze(word)[0].source == "guess"


# --- The negative-aorist participle -mAz ---------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes",
    [
        ("bitmez", "bit", ["mez"]),  # front harmony
        ("çıkmaz", "çık", ["maz"]),  # back harmony
        ("tükenmez", "tüken", ["mez"]),  # new root tüken
        ("geçmez", "geç", ["mez"]),
    ],
)
def test_negative_aorist_participle(analyzer, word, lemma, morphemes):
    a = _find(analyzer, word, lemma=lemma, pos="ADJ", derivation=("maz",))
    assert a is not None, f"no {lemma}+maz participle for {word}"
    assert a.morphemes == morphemes
    assert a.stem == word
    assert a.features.get("polarity") == "negative"
    assert a.features.get("verbform") == "participle"
    assert "tense" not in a.features  # a participle is not finite


@pytest.mark.positive
def test_negative_aorist_participle_over_voiced_stem(analyzer):
    # From a passive stem: yapılmaz "undoable" (participle), keeping the passive voice.
    a = _find(analyzer, "yapılmaz", lemma="yap", pos="ADJ", derivation=("maz",))
    assert a is not None
    assert a.features.get("voice") == ("passive",)
    assert a.features.get("polarity") == "negative"


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("bitmez", "bit"), ("çıkmaz", "çık")])
def test_finite_negative_aorist_stays_primary(analyzer, word, lemma):
    # The finite negative aorist (3sg) stays the primary reading; participle is an alternative.
    best = analyzer.analyze(word)[0]
    assert best.pos == "VERB"
    assert best.features.get("tense") == "aorist"
    assert best.features.get("polarity") == "negative"
    assert "derivation" not in best.features


@pytest.mark.negative
def test_maz_participle_not_reachable_from_negation(analyzer):
    # -mAz is wired on the bare root / voiced stems only, never after -mA negation: there is no
    # NEG + -mAz participle path, mirroring the finite *gelmemez gap.
    assert not has_analysis(analyzer, "gelmemez", lemma="gel")
    assert not any(r.features.get("derivation") == ("maz",) for r in analyzer.analyze("gelmemez"))


@pytest.mark.negative
@pytest.mark.parametrize("word", ["gelmezim", "gelmeziz"])
def test_maz_participle_does_not_revive_defective_persons(analyzer, word):
    # The crux: the -mAz participle lands in a case-only state (N_PART_NEG), taking NEITHER the
    # possessive NOR the ek-fiil copula. Both would add -Im/-Iz and resurrect the deliberately-
    # defective finite negative-aorist persons (gelmem/gelmeyiz, never *gelmezim/*gelmeziz).
    assert not has_analysis(analyzer, word, lemma="gel")
    assert analyzer.analyze(word)[0].source == "guess"


# --- The derived participle inflects nominally (case / number / possessive) -----------


@pytest.mark.positive
def test_mis_participle_inflects_possessive(analyzer):
    # geçmişi = geç + mIş + possessive-3sg (or accusative); both are participle readings.
    a = _find(analyzer, "geçmişi", lemma="geç", derivation=("mis",))
    assert a is not None
    assert a.lemma == "geç"
    assert a.stem == "geçmiş"
    assert a.morphemes == ["miş", "i"]
    assert a.features.get("possessive") == "3sg" or a.features.get("case") == "accusative"


@pytest.mark.positive
def test_aorist_participle_inflects_plural(analyzer):
    a = _find(analyzer, "akarlar", lemma="ak", derivation=("ar",))
    assert a is not None
    assert a.stem == "akar"
    assert a.morphemes == ["ar", "lar"]
    assert a.features.get("number") == "plural"


@pytest.mark.positive
def test_negative_aorist_participle_inflects_case(analyzer):
    # çıkmazda = çık + mAz + locative: the -mAz participle DOES take case (just not poss/copula).
    a = _find(analyzer, "çıkmazda", lemma="çık", derivation=("maz",))
    assert a is not None
    assert a.stem == "çıkmaz"
    assert a.morphemes == ["maz", "da"]
    assert a.features.get("case") == "locative"
    assert a.features.get("polarity") == "negative"


# --- No re-derivation, guesser untouched, and the retrofit -----------------------------


@pytest.mark.negative
@pytest.mark.parametrize("word", ["geçmişli", "akarcı"])
def test_participle_does_not_re_derive(analyzer, word):
    # A derived stem cannot derive again (single derivation slot): no analysis stacks a second
    # derivation onto a participle.
    for r in analyzer.analyze(word):
        assert len(r.features.get("derivation", ())) < 2


@pytest.mark.negative
@pytest.mark.parametrize("word", ["zortmuş", "flomar", "zonkmaz"])
def test_guesser_never_emits_a_participle(analyzer, word):
    # The guesser forbids derivation (and the aorist edges are class-guarded, aorist=None on a
    # synthetic root): an OOV word is never split into a participle.
    for r in analyzer.analyze(word):
        assert "derivation" not in r.features
        assert r.features.get("verbform") is None
        assert r.source != "lexicon"


@pytest.mark.consistency
def test_existing_participles_now_carry_verbform(analyzer):
    # Retrofit: the pre-existing -(y)An / -DIk / -(y)AcAk participles are now uniformly marked
    # verbform=participle too (subset-matching keeps their older tests green).
    assert _find(analyzer, "gelen", derivation=("an",)).features.get("verbform") == "participle"
    assert _find(analyzer, "yaşadık", derivation=("dik",)).features.get("verbform") == "participle"
    assert (
        _find(analyzer, "geleceği", derivation=("acak",)).features.get("verbform") == "participle"
    )


@pytest.mark.consistency
@pytest.mark.parametrize(
    "word,lemma,stem,derivation",
    [
        ("geçmiş", "geç", "geçmiş", ("mis",)),
        ("akar", "ak", "akar", ("ar",)),
        ("çıkmaz", "çık", "çıkmaz", ("maz",)),
    ],
)
def test_participle_views_agree(word, lemma, stem, derivation):
    # stem / lemma / analyze are three views of one analysis: the lemma is the verb root, the
    # stem is the surface at the participle boundary. (The finite homograph is the primary, so
    # lemmatize/stem read it; both still resolve to the verb root, which is what matters here.)
    assert ilmek.lemmatize(word) == lemma
    assert any(
        r.features.get("derivation") == derivation and r.stem == stem and r.lemma == lemma
        for r in ilmek.analyze(word)
    )
