"""Denominal verb-formers -lA / -lAn / -lAş (NOUN/ADJ -> VERB).

Milestone scope (correctness over coverage): the three productive denominal verbalizers turn a
noun or adjective into a VERB stem that then takes the FULL verbal inflection — taş->taşla-,
temiz->temizle-, ev->evlen-, süs->süslen-, güzel->güzelleş-, selam->selamlaş-, kucak->kucaklaş-.
Each is DERIVATIONAL (the lemma stays the base; the morpheme name -- la/lan/las -- is recorded
in order under features['derivation']) and lands in the NON-final V_DENOM state, so the result
inflects fully (past, progressive, negation, future, aorist, mood, the verb->nominal
derivations, converbs) but a bare denominal stem is not accepted (taşla stays taş+instrumental;
the bare imperative temizle!/selamlaş! is deferred, an xfail below).

Overgeneration is double-guarded, exactly like the reciprocal/temporal-ki precedent:
``applies_to={NOUN, ADJ}`` blocks NUM/PRON bases structurally and ``requires_attribute`` gates
each verbalizer to a CURATED per-base list in the lexicon, so -lAş/-lAn fire only on plausible
bases and the (derivation-free) guesser never walks them. Per the testing contract every rule
carries positive, negative and (where one exists) exception/xfail cases.
"""

from __future__ import annotations

import pytest
from conftest import has_analysis

import ilmek  # noqa: F401  (import parity with the sibling suites)


def _primary(analyzer, word):
    return analyzer.analyze(word)[0]


# --- Positive: the denominal verb is the primary reading -----------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes,derivation,features",
    [
        # Milestone examples: lemma = the base noun/adjective, pos = VERB.
        ("selamlaştı", "selam", ["laş", "tı"], ("las",), {"tense": "past", "person": "3sg"}),
        ("güzelleşiyor", "güzel", ["leş", "iyor"], ("las",), {"aspect": "progressive"}),
        ("evlendi", "ev", ["len", "di"], ("lan",), {"tense": "past", "person": "3sg"}),
        # süs is front-rounded (ü): -lAn harmonizes to -len (süslen), not *süslan.
        ("süslendi", "süs", ["len", "di"], ("lan",), {"tense": "past", "person": "3sg"}),
        ("temizledi", "temiz", ["le", "di"], ("la",), {"tense": "past", "person": "3sg"}),
        ("temizledim", "temiz", ["le", "di", "m"], ("la",), {"person": "1sg"}),
        # -Iyor deletes the -lA stem vowel (temizle -> temizl+iyor), so the surface is temizliyor.
        ("temizliyor", "temiz", ["le", "iyor"], ("la",), {"aspect": "progressive"}),
        # kucaklaştı is ONE -lAş morpheme on the noun kucak, NOT a reciprocal voice (see below).
        ("kucaklaştı", "kucak", ["laş", "tı"], ("las",), {"tense": "past", "person": "3sg"}),
        ("taşladı", "taş", ["la", "dı"], ("la",), {"tense": "past"}),
        ("avladı", "av", ["la", "dı"], ("la",), {"tense": "past"}),
        # hasta is an ADJ base (adjective -> verb).
        ("hastalandı", "hasta", ["lan", "dı"], ("lan",), {"tense": "past"}),
        # taş carries BOTH -lA and -lAş; -lAş gives taşlaştı (to petrify).
        ("taşlaştı", "taş", ["laş", "tı"], ("las",), {"tense": "past"}),
    ],
)
def test_denominal_primary(analyzer, word, lemma, morphemes, derivation, features):
    best = _primary(analyzer, word)
    assert best.lemma == lemma, f"{word}: lemma {best.lemma!r} != {lemma!r}"
    assert best.pos == "VERB"
    assert best.morphemes == morphemes
    assert best.features.get("derivation") == derivation
    assert best.source == "lexicon"
    for key, value in features.items():
        assert best.features.get(key) == value
    # A finite denominal verb is walked from a NOMINAL root, but must NOT keep the nominal
    # default number/possessive/case (regression guard for the seeded-defaults leak).
    for key in ("number", "possessive", "case"):
        assert key not in best.features, f"{word}: finite verb leaked {key}"


@pytest.mark.positive
def test_kucaklas_is_denominal_not_reciprocal(analyzer):
    # kucaklaştı = kucak(NOUN) + laş + tı, a single denominal morpheme — NOT kucak + reciprocal.
    best = _primary(analyzer, "kucaklaştı")
    assert best.lemma == "kucak" and best.features.get("derivation") == ("las",)
    assert "voice" not in best.features
    assert not has_analysis(analyzer, "kucaklaştı", features={"voice": ("reciprocal",)})


# --- Positive: the denominal stem takes the full verbal continuation -----------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,pos,morphemes,features",
    [
        # Negation.
        (
            "evlenmedi",
            "ev",
            "VERB",
            ["len", "me", "di"],
            {"polarity": "negative", "tense": "past", "derivation": ("lan",)},
        ),
        # Future.
        (
            "güzelleşecek",
            "güzel",
            "VERB",
            ["leş", "ecek"],
            {"tense": "future", "derivation": ("las",)},
        ),
        # Denominal aorist: -Ir after a consonant-final stem (evlen, güzelleş).
        ("evlenir", "ev", "VERB", ["len", "ir"], {"tense": "aorist", "derivation": ("lan",)}),
        ("güzelleşir", "güzel", "VERB", ["leş", "ir"], {"tense": "aorist", "derivation": ("las",)}),
        # The infinitive -mAk derives back to a NOUN: lemma stays the base, order preserved.
        ("evlenmek", "ev", "NOUN", ["len", "mek"], {"derivation": ("lan", "mak")}),
        # Participle -(y)An derives to an ADJ.
        (
            "güzelleşen",
            "güzel",
            "ADJ",
            ["leş", "en"],
            {"verbform": "participle", "derivation": ("las", "an")},
        ),
        # Converbs (zarf-fiil) derive to an ADV.
        (
            "selamlaşarak",
            "selam",
            "ADV",
            ["laş", "arak"],
            {"verbform": "converb", "derivation": ("las", "arak")},
        ),
        (
            "selamlaşınca",
            "selam",
            "ADV",
            ["laş", "ınca"],
            {"verbform": "converb", "derivation": ("las", "inca")},
        ),
    ],
)
def test_denominal_full_inflection(analyzer, word, lemma, pos, morphemes, features):
    assert has_analysis(
        analyzer, word, lemma=lemma, pos=pos, morphemes=morphemes, features=features
    )
    assert any(a.source == "lexicon" for a in analyzer.analyze(word))


@pytest.mark.positive
def test_denominal_vowel_final_aorist_is_r_alternative(analyzer):
    # taşlar keeps taş+plural as the primary (0 derivations); the denominal aorist taş+la+r
    # (vowel-final stem -> -r) survives as a ranked ALTERNATIVE.
    best = _primary(analyzer, "taşlar")
    assert best.lemma == "taş" and best.pos == "NOUN" and best.features.get("number") == "plural"
    assert has_analysis(
        analyzer,
        "taşlar",
        lemma="taş",
        pos="VERB",
        morphemes=["la", "r"],
        features={"tense": "aorist", "derivation": ("la",)},
    )


# --- Reciprocal reading availability -------------------------------------------------


@pytest.mark.positive
def test_gulustu_reciprocal_available(analyzer):
    # gül gained the "reciprocal" attribute: gülüştü resolves to the verbal reciprocal (primary),
    # while the -Iş verbal-noun gülüş stays a NOUN alongside (V_RECIP is non-final).
    best = _primary(analyzer, "gülüştü")
    assert best.lemma == "gül" and best.pos == "VERB"
    assert best.features.get("voice") == ("reciprocal",)
    assert best.morphemes == ["üş", "tü"]
    assert has_analysis(
        analyzer, "gülüş", lemma="gül", pos="NOUN", features={"derivation": ("is",)}
    )


# --- Negative: the overgeneration guards must block the wrong bases ------------------


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,base",
    [
        ("evleşti", "ev"),  # ev is a -lAn base only, not -lAş
        ("evledi", "ev"),  # ...and not -lA
        ("süsleşti", "süs"),  # süs is -lAn only
        ("taşlandı", "taş"),  # taş is -lA/-lAş only (no -lAn), and no passive is wired off V_DENOM
        ("hastalaştı", "hasta"),  # hasta is -lAn only, not -lAş
        ("birledi", "bir"),  # bir is a NUM: applies_to={NOUN,ADJ} blocks it (doubly, no attribute)
    ],
)
def test_denominal_wrong_base_is_blocked(analyzer, word, base):
    # No lexicon analysis at all with the base as lemma: the attribute/POS guard blocked it, so
    # the word falls to the honest guesser.
    assert not has_analysis(analyzer, word, lemma=base)
    assert _primary(analyzer, word).source == "guess"


@pytest.mark.negative
def test_basla_stays_root_verb_no_denominal_split(analyzer):
    # başla is a ROOT verb: başladı stays lemma başla, never a baş(NOUN)+la denominal split
    # (baş carries no verbalizer attribute).
    best = _primary(analyzer, "başladı")
    assert best.lemma == "başla" and best.pos == "VERB"
    assert best.features.get("derivation") is None
    assert not has_analysis(analyzer, "başladı", lemma="baş")


@pytest.mark.negative
def test_no_re_derivation_off_a_derived_stem(analyzer):
    # The verbalizers hang only off N_ROOT, never the derived N_DERIV, so a -lIk-derived stem
    # cannot re-verbalize: güzellikleşti has no lexicon analysis (structural, not a filter).
    assert not has_analysis(analyzer, "güzellikleşti", lemma="güzel")
    assert _primary(analyzer, "güzellikleşti").source == "guess"


@pytest.mark.negative
def test_guesser_never_splits_a_denominal(analyzer):
    # An OOV word must never be split on a (derivational) verbalizer: parselledi stays an honest
    # guess with no derivation, exactly as before this milestone.
    assert _primary(analyzer, "parselledi").source == "guess"
    assert not has_analysis(analyzer, "parselledi", features={"derivation": ("la",)})


# --- Voicing-flag guards on the new lexicon entries ---------------------------------


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,lemma,morpheme",
    [
        ("selamı", "selam", "ı"),  # selam must NOT voice (no *selağı)
        ("süsü", "süs", "ü"),  # süs must NOT voice (no *süzü)
        ("avı", "av", "ı"),  # av must NOT voice (no *abı)
    ],
)
def test_new_nouns_do_not_false_voice(analyzer, word, lemma, morpheme):
    # The ordinary noun inflection of the newly-added bases is intact and carries no wrong
    # voicing flag: the accusative is just root + the case vowel.
    assert has_analysis(
        analyzer,
        word,
        lemma=lemma,
        pos="NOUN",
        morphemes=[morpheme],
        features={"case": "accusative"},
    )


@pytest.mark.positive
def test_kucak_voices_but_denominal_keeps_free_form(analyzer):
    # kucak DOES voice before a vowel-initial suffix (kucağı), but the consonant-initial -lAş
    # attaches to the FREE form, so kucaklaştı keeps the k (no *kucağlaş).
    assert has_analysis(
        analyzer, "kucağı", lemma="kucak", pos="NOUN", features={"case": "accusative"}
    )
    best = _primary(analyzer, "kucaklaştı")
    assert best.morphemes == ["laş", "tı"] and best.lemma == "kucak"


@pytest.mark.positive
def test_kotu_denominal_reading_exists_alongside_lexicalized_verb(analyzer):
    # kötüleş is a lexicalized ROOT verb (its past kötüleşti stays primary, pinned elsewhere);
    # adding verbalizer_las to kötü ALSO exposes the productive kötü+leş reading as an
    # alternative (lemma = the base kötü), proving -lAş fires on the kötü base.
    assert has_analysis(
        analyzer, "kötüleşti", lemma="kötü", pos="VERB", features={"derivation": ("las",)}
    )
    # The lexicalized verb still wins the primary slot (longer root, fewer derivations).
    assert _primary(analyzer, "kötüleşti").lemma == "kötüleş"


# --- Deferred (xfail): documented gaps kept visible ---------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="Bare denominal imperative (temizle!, selamlaş! as a VERB) is deferred: V_DENOM is "
    "deliberately non-final to protect the noun primaries (taşla = taş+instrumental), mirroring "
    "the non-final voice-state precedent.",
)
def test_bare_denominal_imperative_deferred(analyzer):
    assert has_analysis(analyzer, "selamlaş", lemma="selam", pos="VERB")


@pytest.mark.xfail(
    strict=True,
    reason="sosyalleşti is deferred: sosyal is a palatal-l loan whose -lAş harmonizes front "
    "(sosyalleş), which needs the front_harmony attribute; but adding it would flip the pinned "
    "sosyaldı->sosyaldi form (test_lexicon_batch2), so per correctness-over-coverage sosyal is "
    "left unattributed rather than shipping the wrong *sosyallaş.",
)
def test_sosyallas_deferred(analyzer):
    assert has_analysis(analyzer, "sosyalleşti", lemma="sosyal", pos="VERB")
