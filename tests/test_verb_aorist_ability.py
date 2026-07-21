"""Verb ability, conditional/optative, and aorist (positive + negative aorist).

Milestone scope (correctness over coverage):

* ability -(y)Abil (gelebilir, okuyabilir), feeding the tenses and the deterministic -Ir
  aorist;
* conditional -sA (gelse, gelseydi via the copula) with type-2 persons;
* optative -(y)A (gele) with its own person set (1pl is -lIm, never *(y)Iz);
* the lexically-irregular positive aorist, whose allomorph (-r / -Ar / -Ir) is a lexicon
  fact on the root, selected by the declarative ``aorist_class`` edge guard;
* the negative aorist -mAz (gelmez) with its *defective* person paradigm.

Per the testing contract each rule carries positive, negative, and (where one exists)
exception cases, a long suffix chain, and a stem/lemma/analysis consistency check.
Deliberately-deferred forms (the impossibilitive -(y)AmA, the copular conditional -(y)sA,
and the negative-aorist 1sg/1pl) are strict-xfail so the roadmap stays visible and no wrong
Turkish is emitted in the meantime.
"""

from __future__ import annotations

import pytest
from conftest import has_analysis

import ilmek

# --- Positive aorist: the lexical -r / -Ar / -Ir split -------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes,features",
    [
        # vowel-final stems take plain -r (the default), with type-1 persons.
        ("okur", "oku", ["r"], {"tense": "aorist", "person": "3sg"}),
        ("yürür", "yürü", ["r"], {"tense": "aorist", "person": "3sg"}),
        ("okurum", "oku", ["r", "um"], {"tense": "aorist", "person": "1sg"}),
        # gel is a lexically marked -Ir exception verb (one of the classic ~13).
        ("gelir", "gel", ["ir"], {"tense": "aorist", "person": "3sg"}),
        # consonant-final monosyllable default -Ar.
        ("yapar", "yap", ["ar"], {"tense": "aorist", "person": "3sg"}),
        # consonant-final polysyllable default -Ir.
        ("oturur", "otur", ["ur"], {"tense": "aorist", "person": "3sg"}),
    ],
)
def test_positive_aorist(analyzer, word, lemma, morphemes, features):
    assert has_analysis(analyzer, word, lemma=lemma, morphemes=morphemes, features=features)


@pytest.mark.positive
def test_aorist_habitual_past_via_copula(analyzer):
    # gelirdi = gel + aorist + past-copula (habitual past): the copula stacks for free.
    assert has_analysis(
        analyzer,
        "gelirdi",
        lemma="gel",
        morphemes=["ir", "di"],
        features={"tense": "aorist", "copula": "past"},
    )


# --- Ability -(y)Abil ----------------------------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes,features",
    [
        ("gelebilir", "gel", ["ebil", "ir"], {"ability": True, "tense": "aorist", "person": "3sg"}),
        # (y) buffer after the vowel-final root oku.
        ("okuyabilir", "oku", ["yabil", "ir"], {"ability": True, "tense": "aorist"}),
        ("gelebiliyor", "gel", ["ebil", "iyor"], {"ability": True, "aspect": "progressive"}),
    ],
)
def test_ability(analyzer, word, lemma, morphemes, features):
    assert has_analysis(analyzer, word, lemma=lemma, morphemes=morphemes, features=features)


@pytest.mark.positive
def test_ability_long_chain(analyzer):
    # gelebilecekmişsiniz = gel + Abil + Future + (copular) Evidential + 2Pl.
    assert has_analysis(
        analyzer,
        "gelebilecekmişsiniz",
        lemma="gel",
        morphemes=["ebil", "ecek", "miş", "siniz"],
        features={
            "ability": True,
            "tense": "future",
            "evidential": True,
            "person": "2pl",
        },
    )


@pytest.mark.positive
def test_negation_feeds_ability(analyzer):
    # gelmeyebilir "may not come": NEG feeds the ability slot (gel + me + yebil + ir).
    assert has_analysis(
        analyzer,
        "gelmeyebilir",
        lemma="gel",
        morphemes=["me", "yebil", "ir"],
        features={"polarity": "negative", "ability": True, "tense": "aorist"},
    )


# --- Conditional -sA (type-2 persons) and its copula --------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes,features",
    [
        ("gelse", "gel", ["se"], {"mood": "conditional", "person": "3sg"}),
        ("gelsek", "gel", ["se", "k"], {"mood": "conditional", "person": "1pl"}),  # type-2 -k
        ("gelmese", "gel", ["me", "se"], {"polarity": "negative", "mood": "conditional"}),
        ("gelebilse", "gel", ["ebil", "se"], {"ability": True, "mood": "conditional"}),
    ],
)
def test_conditional(analyzer, word, lemma, morphemes, features):
    assert has_analysis(analyzer, word, lemma=lemma, morphemes=morphemes, features=features)


@pytest.mark.positive
def test_conditional_copula(analyzer):
    # gelseydi = gel + sA + (y)DI (past copula, with the -y- buffer after the vowel).
    assert has_analysis(
        analyzer,
        "gelseydi",
        lemma="gel",
        morphemes=["se", "ydi"],
        features={"mood": "conditional", "copula": "past"},
    )


# --- Optative -(y)A: its own person paradigm (1pl is -lIm) --------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes,features",
    [
        ("gele", "gel", ["e"], {"mood": "optative", "person": "3sg"}),
        ("geleyim", "gel", ["e", "yim"], {"mood": "optative", "person": "1sg"}),
        ("gelelim", "gel", ["e", "lim"], {"mood": "optative", "person": "1pl"}),  # -lIm, not (y)Iz
    ],
)
def test_optative(analyzer, word, lemma, morphemes, features):
    assert has_analysis(analyzer, word, lemma=lemma, morphemes=morphemes, features=features)


# --- Negative aorist -mAz (defective persons + copula) ------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes,features",
    [
        ("gelmez", "gel", ["mez"], {"polarity": "negative", "tense": "aorist", "person": "3sg"}),
        ("yapmaz", "yap", ["maz"], {"polarity": "negative", "tense": "aorist"}),  # back harmony
        ("gelmezler", "gel", ["mez", "ler"], {"polarity": "negative", "person": "3pl"}),
    ],
)
def test_negative_aorist(analyzer, word, lemma, morphemes, features):
    assert has_analysis(analyzer, word, lemma=lemma, morphemes=morphemes, features=features)


@pytest.mark.positive
def test_negative_aorist_copula(analyzer):
    # gelmezdi = neg-aorist + past copula; gelmezdim (its regular 1sg past) is also correct.
    assert has_analysis(
        analyzer,
        "gelmezdi",
        lemma="gel",
        features={"polarity": "negative", "tense": "aorist", "copula": "past"},
    )
    assert has_analysis(
        analyzer,
        "gelmezdim",
        lemma="gel",
        features={"polarity": "negative", "tense": "aorist", "copula": "past", "person": "1sg"},
    )


# --- Exceptions: root voicing and two-letter vowel-final roots ----------------------


@pytest.mark.exception
@pytest.mark.parametrize(
    "word,lemma,morphemes",
    [
        ("gider", "git", ["er"]),  # git -> gid + er (-Ar with root voicing before the vowel)
        ("eder", "et", ["er"]),  # et -> ed + er
        ("gidebilir", "git", ["ebil", "ir"]),  # voicing before the vowel-initial -(y)Abil
    ],
)
def test_aorist_root_voicing(analyzer, word, lemma, morphemes):
    assert has_analysis(
        analyzer, word, lemma=lemma, morphemes=morphemes, features={"tense": "aorist"}
    )


@pytest.mark.exception
@pytest.mark.parametrize("word,lemma", [("der", "de"), ("yer", "ye")])
def test_two_letter_vowel_final_takes_plain_r(analyzer, word, lemma):
    # de/ye are vowel-final, so the aorist is plain -r (der, yer) — no glide raising.
    assert has_analysis(analyzer, word, lemma=lemma, morphemes=["r"], features={"tense": "aorist"})


# --- Negatives: the lexical guard, defective persons, paradigm consistency ----------


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,lemma",
    [
        ("geler", "gel"),  # gel is lexically -Ir, so the -Ar default must NOT fire
        ("yapır", "yap"),  # yap is a default -Ar verb, so it must NOT take -Ir
    ],
)
def test_wrong_aorist_allomorph_is_rejected(analyzer, word, lemma):
    assert not has_analysis(analyzer, word, lemma=lemma)
    assert analyzer.analyze(word)[0].source == "guess"


@pytest.mark.negative
@pytest.mark.parametrize("word", ["gelmezim", "gelmeziz"])
def test_negative_aorist_defective_persons_rejected(analyzer, word):
    # The negative-aorist paradigm is defective: 1sg is gelmem and 1pl is gelmeyiz, never
    # *gelmezim / *gelmeziz. The dedicated V_AOR_NEG state omits those endings.
    assert not has_analysis(analyzer, word, lemma="gel")
    assert analyzer.analyze(word)[0].source == "guess"


@pytest.mark.negative
def test_conditional_rejects_type1_person(analyzer):
    # The conditional takes type-2 persons only (gelsem), never type-1 (y)Im (*gelseyim).
    assert not has_analysis(analyzer, "gelseyim", lemma="gel")


@pytest.mark.negative
def test_neg_aorist_does_not_stack_on_negation(analyzer):
    # -mAz attaches to the bare root only; there is no NEG + -mAz path (*gelmemez).
    assert not has_analysis(analyzer, "gelmemez", lemma="gel")


@pytest.mark.negative
def test_impossibilitive_not_misparsed(analyzer):
    # gelemez (impossibilitive -(y)AmA) is deliberately unmodelled — but it must not be
    # wrongly parsed to a *lexicon* gel analysis by the new optative/conditional/aorist edges.
    for r in analyzer.analyze("gelemez"):
        assert not (r.lemma == "gel" and r.source == "lexicon")


# --- Consistency: stem / lemma / analyze agree (inflection, no derivation) -----------


@pytest.mark.consistency
@pytest.mark.parametrize("word,root", [("gelebilir", "gel"), ("okur", "oku"), ("gelmez", "gel")])
def test_stem_lemma_agree(word, root):
    # Ability / aorist / mood are inflectional: no derivation boundary, so stem == lemma.
    assert ilmek.lemmatize(word) == root
    assert ilmek.stem(word) == root
    best = ilmek.analyze(word)[0]
    assert best.lemma == best.stem == root
    assert "derivation" not in best.features


# --- Documented deferrals (strict xfail): honest known limitations -------------------


@pytest.mark.exception
@pytest.mark.xfail(
    reason="Impossibilitive -(y)AmA (gelemez) is a distinct morpheme, not NEG+mAz; deferred.",
    strict=True,
)
def test_impossibilitive_deferred(analyzer):
    assert has_analysis(
        analyzer, "gelemez", lemma="gel", features={"polarity": "negative", "ability": True}
    )


@pytest.mark.exception
@pytest.mark.xfail(
    reason="Negative-aorist 1sg gelmem deferred; only the gelme(VN)+poss-1sg reading exists.",
    strict=True,
)
def test_negative_aorist_1sg_deferred(analyzer):
    assert has_analysis(
        analyzer,
        "gelmem",
        lemma="gel",
        features={"polarity": "negative", "tense": "aorist", "person": "1sg"},
    )


@pytest.mark.exception
@pytest.mark.xfail(reason="Negative-aorist 1pl gelmeyiz deferred.", strict=True)
def test_negative_aorist_1pl_deferred(analyzer):
    assert has_analysis(
        analyzer,
        "gelmeyiz",
        lemma="gel",
        features={"polarity": "negative", "tense": "aorist", "person": "1pl"},
    )


@pytest.mark.exception
@pytest.mark.xfail(
    reason="Copular conditional -(y)sA after a finished tense (gelirse) is the next milestone.",
    strict=True,
)
def test_copular_conditional_deferred(analyzer):
    assert has_analysis(
        analyzer, "gelirse", lemma="gel", features={"tense": "aorist", "mood": "conditional"}
    )


# --- Companion (non-xfail): the deferrals must not break existing readings ------------


@pytest.mark.positive
def test_gelmem_keeps_verbal_noun_possessive_reading(analyzer):
    # Deferring the aorist 1sg must leave the existing gelme(VN)+poss-1sg reading intact.
    assert has_analysis(
        analyzer, "gelmem", lemma="gel", features={"possessive": "1sg", "derivation": ("ma",)}
    )
