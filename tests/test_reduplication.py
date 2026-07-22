"""Emphatic reduplication (pekiştirme): kapkara, yemyeşil, tertemiz, sımsıcak, ...

Turkish intensifies an adjective by prefixing a copy of its first (C)V closed by a
lexicalized linking consonant from {p, m, r, s}: kara->kapkara, temiz->tertemiz,
sıcak->sımsıcak. Both the linker and the exact truncation are IRREGULAR (kara takes -p,
temiz takes -r, sıcak takes -m), so the 16 attested forms are enumerated whole as
``IrregularForm`` surfaces (surface -> base ADJ lemma, feature ``intensity=emphatic``) in
``data/lexicon/reduplication.json`` and matched before the FSM — exactly the intensive-
diminutive precedent (``diminutives.json`` / ``test_diminutive.py``). The base adjective is
the lemma; the intensified surface is the stem.

A curated table, NOT a strip-and-check rule: stripping a leading ``(C)V + p/m/r/s`` would
accept unattested linkers (*kamkara) and misparse coincidental non-reduplicative words, so
"correctness over coverage" mandates enumeration. The tests pin every row exactly (the linker
is data, and a wrong linker would be a shipped wrong Turkish rule), the two new base entries
(kara, düz), the audit misparses that must be gone (tertemiz->terte+1pl, upuzun->upuz+gen),
the overgeneration guards, and the accepted inflection deferral (kapkaraydı, one xfail).
"""

from __future__ import annotations

import pytest
from conftest import has_analysis

import ilmek

# surface -> (base lemma, prefix morpheme). Every attested row is pinned (do NOT sample):
# an ı/i orthography slip or a wrong linker would silently no-op or ship a wrong rule.
_REDUPLICATIONS = [
    ("kapkara", "kara", ["kap"]),
    ("yemyeşil", "yeşil", ["yem"]),
    ("tertemiz", "temiz", ["ter"]),
    ("sımsıcak", "sıcak", ["sım"]),
    ("masmavi", "mavi", ["mas"]),
    ("bembeyaz", "beyaz", ["bem"]),
    ("upuzun", "uzun", ["up"]),
    ("sapsarı", "sarı", ["sap"]),
    ("bomboş", "boş", ["bom"]),
    ("kupkuru", "kuru", ["kup"]),
    ("dosdoğru", "doğru", ["dos"]),
    ("apaçık", "açık", ["ap"]),
    ("dapdar", "dar", ["dap"]),
    ("dümdüz", "düz", ["düm"]),
    ("yepyeni", "yeni", ["yep"]),
    ("ıpıslak", "ıslak", ["ıp"]),
]


# --- Positive: each reduplication resolves to its base ADJ, lexicon-verified ----------


@pytest.mark.positive
@pytest.mark.parametrize("surface,lemma,morphemes", _REDUPLICATIONS)
def test_reduplication_is_primary(analyzer, surface, lemma, morphemes):
    # The emphatic form resolves to its base lemma as the PRIMARY analysis, lexicon-verified.
    best = analyzer.analyze(surface)[0]
    assert best.lemma == lemma
    assert best.pos == "ADJ"
    assert best.morphemes == morphemes
    assert best.features.get("intensity") == "emphatic"
    assert best.source == "lexicon"
    assert best.stem == surface  # derived irregular: stem is the whole surface


@pytest.mark.consistency
@pytest.mark.parametrize("surface,lemma", [("tertemiz", "temiz"), ("kapkara", "kara")])
def test_reduplication_views_agree(surface, lemma):
    # Milestone headline: lemmatize is the base, stem is the intensified surface, and the
    # analyze primary agrees with both — three views of one analysis.
    assert ilmek.lemmatize(surface) == lemma
    assert ilmek.stem(surface) == surface
    best = ilmek.analyze(surface)[0]
    assert best.lemma == lemma and best.stem == surface


# --- Turkish-aware casing -------------------------------------------------------------


@pytest.mark.positive
def test_reduplication_casing_folds():
    # Input casing is folded before lookup; TERTEMİZ exercises the İ -> i Turkish fold.
    assert analyzer_lemma("Kapkara") == "kara"
    assert analyzer_lemma("TERTEMİZ") == "temiz"


def analyzer_lemma(word):
    return ilmek.analyze(word)[0].lemma


# --- The two new base entries parse plainly as ADJ, with no fabricated feature --------


@pytest.mark.positive
@pytest.mark.parametrize("word", ["kara", "düz"])
def test_new_base_adjectives_parse_bare(analyzer, word):
    best = analyzer.analyze(word)[0]
    assert best.lemma == word
    assert best.pos == "ADJ"
    assert best.source == "lexicon"
    assert best.stem == best.lemma
    assert "intensity" not in best.features


@pytest.mark.consistency
def test_kara_keeps_kar_dative_as_alternative():
    # Adding the ADJ "kara" makes it the primary, but the genuine kar(snow)+dative reading
    # is kept as a ranked alternative — ambiguity preserved, nothing erased.
    assert has_analysis(analyzer_(), "kara", lemma="kara", pos="ADJ")
    assert has_analysis(analyzer_(), "kara", lemma="kar", pos="NOUN", features={"case": "dative"})


def analyzer_():
    from ilmek.morphology.analyzer import default_analyzer

    return default_analyzer()


# --- Regression: the audit misparses must be gone -------------------------------------


@pytest.mark.negative
def test_tertemiz_no_terte_no_possessive(analyzer):
    # The audit found tertemiz -> "terte" + 1pl possessive. It must be gone entirely.
    for r in analyzer.analyze("tertemiz"):
        assert r.lemma != "terte"
        assert r.features.get("possessive") != "1pl"
    assert analyzer.analyze("tertemiz")[0].lemma == "temiz"


@pytest.mark.negative
def test_upuzun_no_upuz_genitive(analyzer):
    # Today's live misparse: upuzun -> "upuz" NOUN + genitive "un". Must be gone.
    for r in analyzer.analyze("upuzun"):
        assert r.lemma != "upuz"
        assert not (r.lemma == "upuz" and r.features.get("case") == "genitive")
    best = analyzer.analyze("upuzun")[0]
    assert best.lemma == "uzun"
    assert best.features.get("intensity") == "emphatic"


# --- Negative: overgeneration guards --------------------------------------------------


@pytest.mark.negative
@pytest.mark.parametrize("word", ["kamkara", "kaskara", "musmavi", "tepyeni"])
def test_wrong_linker_is_rejected(analyzer, word):
    # The linker is lexicalized: only the attested string matches. A wrong linker consonant
    # produces NO emphatic reading and NO lexicon-verified base at rank 0 — it falls to the
    # honest identity guess (pos X).
    results = analyzer.analyze(word)
    for r in results:
        assert r.features.get("intensity") != "emphatic"
    best = results[0]
    assert best.source != "lexicon"
    assert best.pos == "X"


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("karanlık", "karanlık"),  # begins like the reduplicant kap-/kar- but is its own word
        ("karar", "karar"),
    ],
)
def test_reduplicant_lookalikes_keep_own_parse(analyzer, word, lemma):
    best = analyzer.analyze(word)[0]
    assert best.lemma == lemma
    assert "intensity" not in best.features


@pytest.mark.negative
@pytest.mark.parametrize("word", ["kara", "temiz", "sıcak"])
def test_bare_base_never_gains_intensity(analyzer, word):
    # Nothing fabricated: the bare base adjective carries no intensity key on any analysis.
    for r in analyzer.analyze(word):
        assert "intensity" not in r.features


# --- Documented deferral (strict xfail): inflected reduplication ----------------------


@pytest.mark.exception
@pytest.mark.xfail(
    reason="Inflected reduplication (ek-fiil on kapkara) is a later milestone — "
    "IrregularForm matches whole surfaces only, mirroring the sıcacıktı diminutive.",
    strict=True,
)
def test_reduplication_copular_past_deferred(analyzer):
    assert has_analysis(analyzer, "kapkaraydı", lemma="kara", features={"copula": "past"})
