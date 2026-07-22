"""Verb ability, conditional/optative, necessitative, impossibilitive, and aorist.

Milestone scope (correctness over coverage):

* ability -(y)Abil (gelebilir, okuyabilir), feeding the tenses and the deterministic -Ir
  aorist;
* conditional -sA (gelse, gelseydi via the copula) with type-2 persons, AND the copular
  conditional -(y)sA stacking on a finished tense (gelirse, geldiyse, gelmezse);
* optative -(y)A (gele) with its own person set (1pl is -lIm, never *(y)Iz);
* the lexically-irregular positive aorist, whose allomorph (-r / -Ar / -Ir) is a lexicon
  fact on the root, selected by the declarative ``aorist_class`` edge guard;
* the negative aorist -mAz (gelmez) with its *defective* person paradigm, plus the irregular
  1sg/1pl (gelmem, gelmeyiz) that attach to the -mA stem, not to -mAz;
* the impossibilitive -(y)AmA (gelemez = gel+eme+z), a distinct morpheme = the ability-
  negative (polarity=negative + ability), NOT NEG+aorist.

Per the testing contract each rule carries positive, negative, and (where one exists)
exception cases, a long suffix chain, and a stem/lemma/analysis consistency check.
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


# --- Impossibilitive -(y)AmA: the ability-negative (gelemez = gel+eme+z) -------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes,features",
    [
        # 3sg aorist is the bare -z; polarity+ability come from -(y)AmA, tense from -z.
        (
            "gelemez",
            "gel",
            ["eme", "z"],
            {"polarity": "negative", "ability": True, "tense": "aorist", "person": "3sg"},
        ),
        (
            "yapamaz",
            "yap",
            ["ama", "z"],
            {"polarity": "negative", "ability": True, "tense": "aorist"},
        ),
        # irregular 1sg/1pl (attach to the -(y)AmA stem, mirroring the negative aorist).
        ("gelemem", "gel", ["eme", "m"], {"ability": True, "tense": "aorist", "person": "1sg"}),
        ("gelemeyiz", "gel", ["eme", "yiz"], {"ability": True, "tense": "aorist", "person": "1pl"}),
        # defective persons reuse V_AOR_NEG (2sg/2pl/3pl).
        ("gelemezsin", "gel", ["eme", "z", "sin"], {"ability": True, "person": "2sg"}),
        ("gelemezsiniz", "gel", ["eme", "z", "siniz"], {"ability": True, "person": "2pl"}),
        ("gelemezler", "gel", ["eme", "z", "ler"], {"ability": True, "person": "3pl"}),
        # other tenses / aspect: past, progressive (via -Iyor's vowel drop of the -AmA -a).
        (
            "yapamadı",
            "yap",
            ["ama", "dı"],
            {"polarity": "negative", "ability": True, "tense": "past"},
        ),
        ("gelemiyor", "gel", ["eme", "iyor"], {"ability": True, "aspect": "progressive"}),
        # (y) buffer after a vowel-final root: oku -> okuyama.
        ("okuyamam", "oku", ["yama", "m"], {"ability": True, "tense": "aorist", "person": "1sg"}),
    ],
)
def test_impossibilitive(analyzer, word, lemma, morphemes, features):
    assert has_analysis(analyzer, word, lemma=lemma, morphemes=morphemes, features=features)


@pytest.mark.positive
def test_impossibilitive_long_chain(analyzer):
    # gelemeyecekti = gel + (y)AmA + (y)AcAk + (past copula): impossibilitive future in the past.
    assert has_analysis(
        analyzer,
        "gelemeyecekti",
        lemma="gel",
        features={"polarity": "negative", "ability": True, "tense": "future", "copula": "past"},
    )


@pytest.mark.exception
@pytest.mark.parametrize(
    "word,lemma,morphemes",
    [
        ("gidemez", "git", ["eme", "z"]),  # root voicing t->d before the vowel-initial -(y)AmA
        ("diyemem", "de", ["yeme", "m"]),  # de-/ye- glide raising feeds the impossibilitive
    ],
)
def test_impossibilitive_root_alternations(analyzer, word, lemma, morphemes):
    assert has_analysis(
        analyzer, word, lemma=lemma, morphemes=morphemes, features={"ability": True}
    )


@pytest.mark.negative
@pytest.mark.parametrize("word", ["gelemezim", "gelemeziz"])
def test_impossibilitive_defective_persons_rejected(analyzer, word):
    # Like the negative aorist, the -z aorist's person paradigm is defective: the 1sg/1pl are
    # gelemem/gelemeyiz, never *gelemezim / *gelemeziz.
    assert not has_analysis(analyzer, word, lemma="gel")
    assert analyzer.analyze(word)[0].source == "guess"


@pytest.mark.negative
def test_impossibilitive_not_reachable_from_negation(analyzer):
    # -(y)AmA attaches to the (voiced) root, never after the -mA negation: no *gelmeyeme path.
    assert not has_analysis(analyzer, "gelmeyemez", lemma="gel")


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


# --- Negative-aorist irregular 1sg / 1pl (gelmem, gelmeyiz) --------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes,features",
    [
        (
            "gelmem",
            "gel",
            ["me", "m"],
            {"polarity": "negative", "tense": "aorist", "person": "1sg"},
        ),
        (
            "gelmeyiz",
            "gel",
            ["me", "yiz"],
            {"polarity": "negative", "tense": "aorist", "person": "1pl"},
        ),
        (
            "yapmam",
            "yap",
            ["ma", "m"],
            {"polarity": "negative", "tense": "aorist", "person": "1sg"},
        ),
        (
            "yemem",
            "ye",
            ["me", "m"],
            {"polarity": "negative", "tense": "aorist", "person": "1sg"},
        ),  # CV root
    ],
)
def test_negative_aorist_irregular_persons(analyzer, word, lemma, morphemes, features):
    # The negative-aorist 1sg/1pl are irregular: they attach to the -mA stem (gel+me+m,
    # gel+me+yiz), not to -mAz. Polarity comes from the negation, tense/person from the ending.
    assert has_analysis(analyzer, word, lemma=lemma, morphemes=morphemes, features=features)


@pytest.mark.positive
def test_gelmem_keeps_verbal_noun_possessive_reading(analyzer):
    # gelmem is genuinely ambiguous: the finite neg-aorist-1sg AND the gelme(VN)+poss-1sg
    # noun reading both survive (the core keeps both; disambiguation is the separate layer).
    assert has_analysis(
        analyzer, "gelmem", lemma="gel", features={"possessive": "1sg", "derivation": ("ma",)}
    )


# --- Copular conditional -(y)sA stacking on a finished tense (gelirse) ----------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes,features",
    [
        ("gelirse", "gel", ["ir", "se"], {"tense": "aorist", "mood": "conditional"}),
        ("geldiyse", "gel", ["di", "yse"], {"tense": "past", "mood": "conditional"}),  # (y) buffer
        ("gelecekse", "gel", ["ecek", "se"], {"tense": "future", "mood": "conditional"}),
        ("geliyorsa", "gel", ["iyor", "sa"], {"aspect": "progressive", "mood": "conditional"}),
        ("gelmişse", "gel", ["miş", "se"], {"evidential": True, "mood": "conditional"}),
        (
            "gelmezse",
            "gel",
            ["mez", "se"],
            {"polarity": "negative", "tense": "aorist", "mood": "conditional"},
        ),
        # V_COP2's type-2 persons come free after the copular conditional.
        (
            "gelirsem",
            "gel",
            ["ir", "se", "m"],
            {"tense": "aorist", "mood": "conditional", "person": "1sg"},
        ),
        (
            "gelirsek",
            "gel",
            ["ir", "se", "k"],
            {"tense": "aorist", "mood": "conditional", "person": "1pl"},
        ),
    ],
)
def test_copular_conditional(analyzer, word, lemma, morphemes, features):
    assert has_analysis(analyzer, word, lemma=lemma, morphemes=morphemes, features=features)


@pytest.mark.negative
def test_copular_conditional_no_restack_on_bare_conditional(analyzer):
    # The bare conditional -sA lands in V_COND, whose copula omits -(y)sA: no conditional-on-
    # conditional restack (*gelseyse). (gelseydi / gelseymiş via -(y)DI / -(y)mIş still work.)
    assert not has_analysis(analyzer, "gelseyse", lemma="gel")
    assert has_analysis(
        analyzer, "gelseydi", lemma="gel", features={"mood": "conditional", "copula": "past"}
    )


@pytest.mark.exception
def test_nominal_conditional_unaffected(analyzer):
    # The nominal copular conditional (güzelse) is a separate edge set and stays unchanged.
    assert has_analysis(analyzer, "güzelse", lemma="güzel", features={"mood": "conditional"})


# --- Necessitative -mAlI (gelmeli "must come") --------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes,features",
    [
        ("gelmeli", "gel", ["meli"], {"mood": "necessitative", "person": "3sg"}),
        ("yapmalıyım", "yap", ["malı", "yım"], {"mood": "necessitative", "person": "1sg"}),
        (
            "gitmeliydik",
            "git",
            ["meli", "ydi", "k"],
            {"mood": "necessitative", "copula": "past", "person": "1pl"},
        ),
        ("gelmemeli", "gel", ["me", "meli"], {"polarity": "negative", "mood": "necessitative"}),
        ("gelmeliymiş", "gel", ["meli", "ymiş"], {"mood": "necessitative", "evidential": True}),
        ("yapılmalı", "yap", ["ıl", "malı"], {"mood": "necessitative"}),  # voiced (passive) stem
    ],
)
def test_necessitative(analyzer, word, lemma, morphemes, features):
    assert has_analysis(analyzer, word, lemma=lemma, morphemes=morphemes, features=features)


@pytest.mark.positive
def test_necessitative_passive_voice_recorded(analyzer):
    assert has_analysis(
        analyzer,
        "yapılmalı",
        lemma="yap",
        features={"voice": ("passive",), "mood": "necessitative"},
    )


@pytest.mark.negative
def test_necessitative_rejects_type2_1pl(analyzer):
    # -mAlI takes the type-1 persons (V_T1); the type-2 1pl -k is not licensed (*gelmelik).
    assert not has_analysis(analyzer, "gelmelik", lemma="gel")


@pytest.mark.exception
def test_necessitative_does_not_shadow_noun_li_derivation(analyzer):
    # elmalı is elma(NOUN)+lI (ADJ derivation), never a verb necessitative: -mAlI is verbal,
    # and elma is a noun, so no spurious *elma-necessitative reading appears.
    assert has_analysis(analyzer, "elmalı", lemma="elma", features={"derivation": ("li",)})
    assert not has_analysis(analyzer, "elmalı", features={"mood": "necessitative"})


@pytest.mark.consistency
def test_necessitative_is_inflectional(analyzer):
    # A mood is inflectional: stem == lemma == gel and no derivation key.
    assert ilmek.lemmatize("gelmeli") == "gel"
    assert ilmek.stem("gelmeli") == "gel"
    best = [r for r in analyzer.analyze("gelmeli") if r.features.get("mood") == "necessitative"][0]
    assert best.lemma == best.stem == "gel"
    assert "derivation" not in best.features
