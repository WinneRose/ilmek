"""The zarf-fiil (converb) inventory: verb -> ADVERB forms.

Converbs turn a verb into an adverb and are TERMINAL — they take no case/plural/possessive/
copula. This module covers, modelled on the participles (verb->ADJ) they sit beside:

* the root/voiced-stem/negation converbs -(y)ArAk (gelerek), -(y)Ip (gelip), -(y)IncA
  (gelince), -(y)AlI (geleli), -DIkçA (geldikçe);
* the privative converbs -mAdAn (gelmeden "without coming") and -mAksIzIn (gelmeksizin),
  each ONE negative-shaped morpheme on the bare root (never NEG + something);
* the temporal -ken (gelirken, geliyorken, gelecekken, gelmezken), which uniquely attaches to
  a FINISHED tense (the aorist/progressive/future/evidential/necessitative and negative aorist).

Every converb keeps lemma/stem = the verb root, is marked verbform=converb, and lands in the
terminal ADV_CVB state. Per the testing contract each rule carries positive, negative/guard,
and (where one exists) exception cases: the terminality guard (a converb takes no further
morpheme), the no-double-negative gap (*gelmemeden), -ken's finite-stem restriction (no
*geldiyken / *gelken), the wrong-allomorph rejection under -ken, and the untouched guesser.
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


# --- The root / voiced-stem converbs -------------------------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes,derivation",
    [
        ("gelerek", "gel", ["erek"], ("arak",)),  # -(y)ArAk
        ("okuyarak", "oku", ["yarak"], ("arak",)),  # (y) buffer after a vowel
        ("yaparak", "yap", ["arak"], ("arak",)),
        ("gelip", "gel", ["ip"], ("ip",)),  # -(y)Ip
        ("yapıp", "yap", ["ıp"], ("ip",)),
        ("okuyup", "oku", ["yup"], ("ip",)),
        ("gelince", "gel", ["ince"], ("inca",)),  # -(y)IncA
        ("okuyunca", "oku", ["yunca"], ("inca",)),
        ("geleli", "gel", ["eli"], ("ali",)),  # -(y)AlI
        ("yapalı", "yap", ["alı"], ("ali",)),
        ("geldikçe", "gel", ["dikçe"], ("dikce",)),  # -DIkçA
        ("yaptıkça", "yap", ["tıkça"], ("dikce",)),
        ("okudukça", "oku", ["dukça"], ("dikce",)),
    ],
)
def test_root_converb(analyzer, word, lemma, morphemes, derivation):
    a = _find(analyzer, word, lemma=lemma, pos="ADV", derivation=derivation)
    assert a is not None, f"no {lemma}+{derivation[0]} converb for {word}"
    assert a.morphemes == morphemes
    assert a.stem == word  # stem is the derived surface
    assert a.features.get("verbform") == "converb"
    assert a.source == "lexicon"
    # A converb is not finite and not a noun: it never fabricates a person/mood/case/number.
    assert "person" not in a.features and "mood" not in a.features
    assert "case" not in a.features and "number" not in a.features


@pytest.mark.positive
def test_converb_glide_raise_de(analyzer):
    # de- raises its /e/ to /i/ before the A-initial converbs (diyerek, diyeli), exactly as the
    # existing diyen participle. The (y)Ip converb likewise raises -> diyip (see the deyip xfail).
    a = _find(analyzer, "diyerek", lemma="de", pos="ADV", derivation=("arak",))
    assert a is not None and a.morphemes == ["yerek"]
    b = _find(analyzer, "diyeli", lemma="de", pos="ADV", derivation=("ali",))
    assert b is not None and b.morphemes == ["yeli"]


@pytest.mark.positive
def test_converb_no_glide_raise_deyince(analyzer):
    # -(y)IncA is deliberately NOT glide-raising: deyince is the uncontested standard form
    # (mirrors the repo's unflagged -(y)Iş -> deyiş precedent), so the raised *diyince is absent.
    a = _find(analyzer, "deyince", lemma="de", pos="ADV", derivation=("inca",))
    assert a is not None and a.morphemes == ["yince"]
    assert not has_analysis(analyzer, "diyince", lemma="de")


@pytest.mark.positive
def test_ip_converb_no_glide_raise_deyip(analyzer):
    # -(y)Ip is deliberately NOT glide-raising: deyip is the TDK-standard form (diyip is a
    # well-known common misspelling), mirroring the repo's unflagged -(y)Iş -> deyiş / -(y)IncA
    # -> deyince precedent, so the raised *diyip is absent (the ye- counterpart yiyip is
    # unreachable under the same unflagged suffix and xfailed below).
    a = _find(analyzer, "deyip", lemma="de", pos="ADV", derivation=("ip",))
    assert a is not None and a.morphemes == ["yip"]
    assert not has_analysis(analyzer, "diyip", lemma="de")


# --- The privative converbs -mAdAn / -mAksIzIn (negative, ONE morpheme on the root) ---


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes,derivation",
    [
        ("gelmeden", "gel", ["meden"], ("madan",)),
        ("yapmadan", "yap", ["madan"], ("madan",)),
        ("gelmeksizin", "gel", ["meksizin"], ("maksizin",)),
        ("yapmaksızın", "yap", ["maksızın"], ("maksizin",)),
    ],
)
def test_privative_converb(analyzer, word, lemma, morphemes, derivation):
    a = _find(analyzer, word, lemma=lemma, pos="ADV", derivation=derivation)
    assert a is not None, f"no {lemma}+{derivation[0]} converb for {word}"
    assert a.morphemes == morphemes  # ONE morpheme (not NEG + a separable ending)
    assert a.stem == word
    assert a.features.get("verbform") == "converb"
    assert a.features.get("polarity") == "negative"
    assert a.source == "lexicon"


@pytest.mark.negative
def test_gelmeden_converb_is_primary_but_vn_ablative_survives(analyzer):
    # gelmeden already parsed as gelme(-mA verbal noun) + ablative ("from the coming"); the new
    # 1-morpheme converb outranks that 2-morpheme reading and becomes primary ("without coming"),
    # while the noun reading is kept as a genuine alternative.
    results = analyzer.analyze("gelmeden")
    assert results[0].pos == "ADV"
    assert results[0].features.get("derivation") == ("madan",)
    assert results[0].features.get("verbform") == "converb"
    # The old verbal-noun + ablative parse is still present.
    assert any(
        r.features.get("derivation") == ("ma",)
        and r.features.get("case") == "ablative"
        and r.lemma == "gel"
        for r in results
    )


# --- Converbs over a voiced (passive) stem -------------------------------------------


@pytest.mark.positive
def test_converb_over_passive_stem(analyzer):
    # The converbs fire from a voice state for free (via _root_continuation), keeping the voice.
    a = _find(analyzer, "yapılarak", lemma="yap", pos="ADV", derivation=("arak",))
    assert a is not None
    assert a.morphemes == ["ıl", "arak"]
    assert a.features.get("voice") == ("passive",)
    assert a.features.get("verbform") == "converb"

    b = _find(analyzer, "yapılmadan", lemma="yap", pos="ADV", derivation=("madan",))
    assert b is not None
    assert b.features.get("voice") == ("passive",)
    assert b.features.get("polarity") == "negative"


# --- Converbs after negation (the reduced set: no privatives) -------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,morphemes,derivation",
    [
        ("gelmeyerek", ["me", "yerek"], ("arak",)),
        ("gelmeyip", ["me", "yip"], ("ip",)),
        ("gelmeyince", ["me", "yince"], ("inca",)),
        ("gelmeyeli", ["me", "yeli"], ("ali",)),
        ("gelmedikçe", ["me", "dikçe"], ("dikce",)),
    ],
)
def test_converb_after_negation(analyzer, word, morphemes, derivation):
    a = _find(analyzer, word, lemma="gel", pos="ADV", derivation=derivation)
    assert a is not None, f"no gel+neg+{derivation[0]} converb for {word}"
    assert a.morphemes == morphemes
    assert a.features.get("polarity") == "negative"
    assert a.features.get("verbform") == "converb"
    # No fabricated verbal closure (clone of the -mIş participle-after-negation guard).
    assert "mood" not in a.features and "person" not in a.features


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,derivation", [("gelmemeden", ("madan",)), ("gelmemeksizin", ("maksizin",))]
)
def test_no_double_negative_privative_converb(analyzer, word, derivation):
    # The privatives are excluded from V_NEG (mirroring -mAz off V_NEG): *gelmemeden as a converb
    # is a double negative. gelmemeden DOES still parse as gelmeme(VN) + ablative, so we assert on
    # the converb derivation tuple, not on has_analysis.
    assert not any(r.features.get("derivation") == derivation for r in analyzer.analyze(word))


# --- The temporal -ken (attaches to a FINISHED tense) --------------------------------


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,lemma,morphemes",
    [
        ("gelirken", "gel", ["ir", "ken"]),  # aorist class Ir -> proves aorist_class reuse
        ("yaparken", "yap", ["ar", "ken"]),  # class Ar
        ("koşarken", "koş", ["ar", "ken"]),  # class Ar
        ("okurken", "oku", ["r", "ken"]),  # class r (vowel-final)
        ("giderken", "git", ["er", "ken"]),  # bound_form gid -> aorist sits after normal edge
        ("geliyorken", "gel", ["iyor", "ken"]),  # progressive + ken
        ("gelecekken", "gel", ["ecek", "ken"]),  # future + ken (no k-softening: ken is a cons.)
        ("gelmişken", "gel", ["miş", "ken"]),  # evidential + ken
    ],
)
def test_ken_converb(analyzer, word, lemma, morphemes):
    a = _find(analyzer, word, lemma=lemma, pos="ADV", derivation=("ken",))
    assert a is not None, f"no {lemma}+ken converb for {word}"
    assert a.morphemes == morphemes
    assert a.stem == word
    assert a.features.get("verbform") == "converb"


@pytest.mark.positive
def test_ken_converb_over_negative_aorist(analyzer):
    # gelmezken = gel + mAz (negative aorist, V_AOR_NEG) + ken: keeps polarity=negative.
    a = _find(analyzer, "gelmezken", lemma="gel", pos="ADV", derivation=("ken",))
    assert a is not None
    assert a.morphemes == ["mez", "ken"]
    assert a.features.get("polarity") == "negative"
    assert a.features.get("verbform") == "converb"


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("gelerken", "gel"), ("yapırken", "yap")])
def test_ken_wrong_aorist_allomorph_is_rejected(analyzer, word, lemma):
    # -ken sits after the finite aorist, which reuses the aorist_class guard: gel is class Ir so
    # the -Ar edge never fires (*gelerken), and yap is class Ar so -Ir never fires (*yapırken).
    assert not has_analysis(analyzer, word, lemma=lemma)
    assert analyzer.analyze(word)[0].source == "guess"


@pytest.mark.negative
@pytest.mark.parametrize("word", ["geldiyken", "geldiken"])
def test_ken_not_on_the_past_tense(analyzer, word):
    # -ken attaches to V_T1 / V_AOR_NEG, never V_T2 (the past -DI): *geldiyken is not a word.
    assert not has_analysis(analyzer, word, lemma="gel")
    assert analyzer.analyze(word)[0].source == "guess"


@pytest.mark.negative
def test_ken_not_bare_on_the_root(analyzer):
    # -ken needs a finished tense under it — it is not wired onto the bare root (*gelken).
    assert not has_analysis(analyzer, "gelken", lemma="gel")


# --- Terminality: a converb takes NO further inflection (the milestone's required guard) ---


@pytest.mark.negative
@pytest.mark.parametrize("word", ["gelerekler", "gelerekte", "gelipler", "gelinceyi", "geldikçede"])
def test_converb_is_terminal_no_case_or_plural(analyzer, word):
    # ADV_CVB is a dead-end state: no converb reading can carry a trailing plural/case morpheme,
    # so none of these surfaces yields ANY verbform=converb analysis.
    assert not any(r.features.get("verbform") == "converb" for r in analyzer.analyze(word))


@pytest.mark.negative
@pytest.mark.parametrize("word", ["gelerekler", "gelipler"])
def test_converb_plus_plural_falls_to_guesser(analyzer, word):
    # A converb + plural is not a lexicon word: the primary is an honest guess, not a converb.
    assert analyzer.analyze(word)[0].source == "guess"


# --- Homograph / overgeneration guards -----------------------------------------------


@pytest.mark.negative
def test_lexicon_noun_outranks_ip_converb(analyzer):
    # kalıp is BOTH the lexicon noun "mold" AND kal + -(y)Ip ("having stayed"). The whole-word
    # noun (longer root) stays primary; the converb survives only as an alternative.
    best = analyzer.analyze("kalıp")[0]
    assert best.pos == "NOUN"
    assert best.lemma == "kalıp"
    assert "derivation" not in best.features


@pytest.mark.negative
@pytest.mark.parametrize("word,lemma", [("gelir", "gel"), ("gelmez", "gel")])
def test_finite_reading_stays_primary(analyzer, word, lemma):
    # The converbs never steal a finite primary: bare gelir/gelmez keep their finite aorist
    # reading first (they have no converb parse at all — only gelirken/gelmezken do).
    best = analyzer.analyze(word)[0]
    assert best.pos == "VERB"
    assert best.features.get("tense") == "aorist"
    assert "derivation" not in best.features


@pytest.mark.negative
@pytest.mark.parametrize("word", ["gelerekli", "gelipçi"])
def test_converb_does_not_re_derive(analyzer, word):
    # A single derivation slot: nothing stacks a second derivation onto a converb.
    for r in analyzer.analyze(word):
        assert len(r.features.get("derivation", ())) < 2


@pytest.mark.negative
@pytest.mark.parametrize("word", ["zortarak", "flomup", "zonkurken"])
def test_guesser_never_emits_a_converb(analyzer, word):
    # The guesser forbids derivation (and -ken is derivational, the aorist is class-guarded with
    # aorist=None on a synthetic root): an OOV word is never split into a converb.
    for r in analyzer.analyze(word):
        assert "derivation" not in r.features
        assert r.features.get("verbform") is None
        assert r.source != "lexicon"


# --- Consistency: stem / lemma / analyze are three views of one analysis --------------


@pytest.mark.consistency
def test_converb_views_agree(analyzer):
    assert ilmek.lemmatize("gelerek") == "gel"
    assert ilmek.lemmatize("gelirken") == "gel"
    assert ilmek.stem("gelerek") == "gelerek"
    for word, lemma, stem, derivation in [
        ("gelince", "gel", "gelince", ("inca",)),
        ("gelmeden", "gel", "gelmeden", ("madan",)),
        ("gelirken", "gel", "gelirken", ("ken",)),
    ]:
        assert any(
            r.features.get("derivation") == derivation and r.stem == stem and r.lemma == lemma
            for r in ilmek.analyze(word)
        )


# --- Documented deferrals (strict xfail): honest known limitations -------------------


@pytest.mark.xfail(
    reason="-(y)Ip is deliberately unflagged so the TDK-standard deyip is correct (see "
    "test_ip_converb_no_glide_raise_deyip); ye- raising to yiyip needs a per-root (not "
    "per-suffix) glide flag, which is a later milestone, so yiyip is deferred rather than "
    "shipping the wrong diyip.",
    strict=True,
)
def test_yiyip_deferred(analyzer):
    assert has_analysis(analyzer, "yiyip", lemma="ye")


@pytest.mark.xfail(
    reason="-(y)IncA takes a dative in the 'gelinceye kadar' construction; the converb state is "
    "terminal this milestone, so gelinceye is deferred.",
    strict=True,
)
def test_ince_converb_takes_dative_in_kadar_construction(analyzer):
    assert has_analysis(analyzer, "gelinceye", lemma="gel")


@pytest.mark.xfail(
    reason="The nominal ek-fiil -(y)ken (evdeyken, güzelken) is out of scope this milestone: "
    "-ken is wired onto the verbal finite stems only, not the nominal copula layer.",
    strict=True,
)
def test_nominal_ekfiil_ken_is_out_of_scope(analyzer):
    assert has_analysis(analyzer, "evdeyken", lemma="ev")
