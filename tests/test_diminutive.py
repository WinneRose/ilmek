"""Diminutive / intensive -CIk: a regular derivation plus enumerated irregular intensives.

Two mechanisms, per the milestone:

* **Regular diminutive -CIk** (kitapçık, evcik, kuşçuk) is a data-declared derivation edge
  (``D_CIK``) with ``applies_to={NOUN}``. The ``C`` archiphoneme hardens to ç after a
  voiceless consonant for free (kitapçık vs. evcik), and ``voice_final`` softens the final k
  before a vowel (kitapçığı). The ``{NOUN}`` gate is the overgeneration guard: the rule never
  fires on an ADJ/ADV base, so *sıcakçık / *güzelcik / *azcık are impossible by rule.

* **Intensive-adjective diminutives** (sıcacık -> sıcak, küçücük -> küçük) reshape their
  stem unpredictably (sıcak drops its k, çabucak changes harmony, azıcık inserts ı), so they
  are enumerated as ``IrregularForm`` exceptions (surface -> base lemma), matched before the
  FSM. They are never produced by the regular edge — their bases are ADJ/ADV, which the
  ``{NOUN}`` gate blocks.

Coverage per the testing contract: positive (the rule applies), negative (it must not),
exception (the ten irregular intensives + C-hardening/k-softening phonology), a long suffix
chain, and a stem/lemma/analyze consistency check. The file also pins the milestone headline
(sıcacık -> sıcak) and the overgeneration guards (common words keep their correct primaries).
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


# --- Regular diminutive -CIk on nouns (positive) -------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morph",
    [
        ("evcik", "ev", "cik"),  # C stays voiced after voiced v; e-harmony
        ("kuşçuk", "kuş", "çuk"),  # C hardens after voiceless ş; u-harmony
        ("gözcük", "göz", "cük"),  # voiced z; ö-harmony (little eye)
    ],
)
def test_regular_diminutive(analyzer, word, lemma, morph):
    a = _find(analyzer, word, lemma=lemma, pos="NOUN", derivation=("cik",))
    assert a is not None, f"no {lemma}+cik analysis for {word}"
    assert a.morphemes == [morph]
    assert a.stem == word  # stem is the derived surface, lemma is the base lexeme
    assert a.source == "lexicon"


@pytest.mark.exception
def test_diminutive_c_hardens_after_voiceless(analyzer):
    # -CIk: the C hardens to ç after the voiceless p of kitap (kitapçık), stays c after v.
    a = _find(analyzer, "kitapçık", lemma="kitap", derivation=("cik",))
    assert a is not None
    assert a.morphemes == ["çık"]
    assert a.pos == "NOUN"
    assert a.stem == "kitapçık"


@pytest.mark.exception
def test_diminutive_softens_k_before_vowel(analyzer):
    # voice_final: kitapçık's final k -> ğ before a vowel-initial suffix (kitapçığı).
    a = _find(analyzer, "kitapçığı", lemma="kitap", derivation=("cik",))
    assert a is not None
    assert a.morphemes == ["çık", "ı"]
    # Both the accusative and the 3sg-possessive reading are valid for -ğı; at least one.
    assert has_analysis(
        analyzer, "kitapçığı", lemma="kitap", features={"case": "accusative"}
    ) or has_analysis(analyzer, "kitapçığı", lemma="kitap", features={"possessive": "3sg"})


@pytest.mark.positive
def test_diminutive_derived_stem_inflects_long_chain(analyzer):
    # A derived diminutive inflects normally: ev+cik+ler+den -> evciklerden.
    a = _find(analyzer, "evciklerden", lemma="ev", derivation=("cik",))
    assert a is not None
    assert a.morphemes == ["cik", "ler", "den"]
    assert a.stem == "evcik"
    assert a.features.get("number") == "plural"
    assert a.features.get("case") == "ablative"
    assert a.pos == "NOUN"


@pytest.mark.consistency
def test_regular_diminutive_views_agree():
    # stem/lemma/analyze are three views of one analysis: lemma is the base, stem the surface.
    assert ilmek.lemmatize("kitapçık") == "kitap"
    assert ilmek.stem("kitapçık") == "kitapçık"
    best = _first_with_derivation("kitapçık")
    assert best.lemma == "kitap" and best.stem == "kitapçık"


def _first_with_derivation(word):
    for a in ilmek.analyze(word):
        if a.features.get("derivation") == ("cik",):
            return a
    raise AssertionError(f"no cik-derivation analysis for {word}")


# --- Intensive-adjective diminutives: enumerated irregular exceptions -----------------


_INTENSIVES = [
    ("sıcacık", "sıcak", "ADJ", ["cık"]),
    ("küçücük", "küçük", "ADJ", ["cük"]),
    ("ufacık", "ufak", "ADJ", ["cık"]),
    ("çabucak", "çabuk", "ADV", ["cak"]),  # irregular harmony: -cak, NOT -cuk
    ("minicik", "minik", "ADJ", ["cik"]),
    ("yumuşacık", "yumuşak", "ADJ", ["cık"]),
    ("kısacık", "kısa", "ADJ", ["cık"]),
    ("incecik", "ince", "ADJ", ["cik"]),
    ("azıcık", "az", "ADV", ["ıcık"]),  # irregular linking ı: az+ıcık, not *azcık
    ("birazcık", "biraz", "ADV", ["cık"]),
]


@pytest.mark.exception
@pytest.mark.parametrize("surface,lemma,pos,morphemes", _INTENSIVES)
def test_intensive_diminutive_is_primary(analyzer, surface, lemma, pos, morphemes):
    # The intensive resolves to its base lemma as the PRIMARY analysis, lexicon-verified.
    best = analyzer.analyze(surface)[0]
    assert best.lemma == lemma
    assert best.pos == pos
    assert best.morphemes == morphemes
    assert best.features.get("derivation") == ("cik",)
    assert best.source == "lexicon"
    assert best.stem == surface  # derived irregular: stem is the whole surface


@pytest.mark.consistency
def test_milestone_target_sicacik():
    # Milestone headline: sıcacık MUST resolve to lemma sıcak, and the views agree.
    assert ilmek.lemmatize("sıcacık") == "sıcak"
    assert ilmek.stem("sıcacık") == "sıcacık"
    best = ilmek.analyze("sıcacık")[0]
    assert best.lemma == "sıcak" and best.stem == "sıcacık"


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,pos",
    [
        ("sıcak", "sıcak", "ADJ"),
        ("küçük", "küçük", "ADJ"),
        ("ufak", "ufak", "ADJ"),
        ("minik", "minik", "ADJ"),
        ("çabuk", "çabuk", "ADJ"),
        ("biraz", "biraz", "ADV"),
    ],
)
def test_intensive_bases_still_analyze_bare(analyzer, word, lemma, pos):
    # The bases (including the four newly added) stay their own lemma with no derivation.
    best = analyzer.analyze(word)[0]
    assert best.lemma == lemma
    assert best.pos == pos
    assert "derivation" not in best.features
    assert best.stem == best.lemma
    assert best.source == "lexicon"


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,features",
    [
        ("ufağı", "ufak", {"case": "accusative"}),  # voicing k -> ğ before a vowel
        ("miniği", "minik", {"case": "accusative"}),
    ],
)
def test_new_voicing_bases_soften_before_vowel(analyzer, word, lemma, features):
    assert has_analysis(analyzer, word, lemma=lemma, features=features)


# --- Negatives: the regular rule must not fire on ADJ/ADV bases ----------------------


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("sıcakçık", "sıcak"),  # regular -CIk must not fire on the ADJ 'sıcak'
        ("sıcakcık", "sıcak"),
        ("küçükçük", "küçük"),  # nor on 'küçük'
        ("güzelcik", "güzel"),  # applies_to={NOUN}: no *güzelcik on an ADJ
        ("azcık", "az"),  # nor on the ADV 'az' (the real form is the irregular azıcık)
    ],
)
def test_regular_cik_does_not_fire_on_non_noun(analyzer, word, lemma):
    assert not has_analysis(analyzer, word, lemma=lemma)


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("kitapcık", "kitap"),  # un-hardened after voiceless p is not Turkish
        ("evçik", "ev"),  # wrongly hardened after voiced v
        ("evcık", "ev"),  # I-harmony violation (must be evcik after e)
    ],
)
def test_regular_cik_phonology_violations_rejected(analyzer, word, lemma):
    assert not has_analysis(analyzer, word, lemma=lemma)


@pytest.mark.negative
def test_no_diminutive_after_inflection(analyzer):
    # Graph position: N_PL has no derivation edge, so *evlercik cannot form.
    assert not has_analysis(analyzer, "evlercik", lemma="ev")


@pytest.mark.negative
def test_no_ci_plus_cik_stacking(analyzer):
    # Single derivation slot: yolcu (yol+CI) is in N_DERIV and cannot take a second -CIk.
    assert not has_analysis(analyzer, "yolcucuk", lemma="yol")


@pytest.mark.negative
def test_guesser_does_not_strip_cik(analyzer):
    # The guesser forbids derivation: an OOV word is never split on -CIk (pancik !-> pan).
    for r in analyzer.analyze("pancik"):
        assert r.lemma != "pan" or r.source != "lexicon"


# --- Overgeneration guards: common words keep their correct primaries -----------------


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,lemma,pos",
    [
        ("çiçek", "çiçek", "NOUN"),  # -CIk's vowel is always high, never the wide e of -çek
        ("çocuk", "çocuk", "NOUN"),
        ("bıçak", "bıçak", "NOUN"),
        ("uçak", "uçak", "NOUN"),
        ("oyuncak", "oyuncak", "NOUN"),
        ("böcek", "böcek", "NOUN"),
        ("gerçek", "gerçek", "ADJ"),
        ("alçak", "alçak", "ADJ"),
    ],
)
def test_common_words_keep_primary_no_derivation(analyzer, word, lemma, pos):
    best = analyzer.analyze(word)[0]
    assert best.lemma == lemma
    assert best.pos == pos
    assert "derivation" not in best.features


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,lemma,pos",
    [
        ("gözlük", "gözlük", "NOUN"),  # re-assert the -lIk guard is unchanged
        ("yemek", "yemek", "NOUN"),
        ("elli", "elli", "NUM"),
    ],
)
def test_existing_derivation_guards_unchanged(analyzer, word, lemma, pos):
    best = analyzer.analyze(word)[0]
    assert best.lemma == lemma
    assert best.pos == pos
    assert "derivation" not in best.features


# --- Documented deferrals (strict xfail): honest known limitations -------------------


@pytest.mark.exception
@pytest.mark.xfail(
    reason="Intensive exception surfaces do not inflect yet; promoting them to Root "
    "variants (with voicing) is a later milestone.",
    strict=True,
)
def test_intensive_inflection_deferred_case(analyzer):
    # küçücüğü = küçücük + accusative/possessive: the intensive stem does not inflect yet.
    assert has_analysis(analyzer, "küçücüğü", lemma="küçük", features={"case": "accusative"})


@pytest.mark.exception
@pytest.mark.xfail(
    reason="Intensive exception surfaces do not inflect yet; the copular past on an "
    "intensive (sıcacıktı) is a later milestone.",
    strict=True,
)
def test_intensive_copular_past_deferred(analyzer):
    assert has_analysis(analyzer, "sıcacıktı", lemma="sıcak", features={"copula": "past"})
