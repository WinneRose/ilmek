"""Unknown-word (OOV) guesser: conservative, clearly-marked root proposal.

The guesser tries both a NOUN and a VERB reading of every prefix and strips ONLY the
phrase-final inflection ([plural -lAr]?[case]? for a noun, tense/person for a verb) — never
a standalone possessive (which used to fabricate a poss=2sg and eat a stem consonant via the
buffer -n-: enflasyonun -> *enflasyo, salgını -> *salg). Both POS stay in the race (VERB is
NOT dropped) so that -lAr, which is homographic between the nominal plural and the verbal
3rd-person-plural marker, does not get every OOV past-tense-3pl verb confidently mis-tagged
as a fabricated NOUN+plural (şaşırdılar must stay şaşır+dı+lar VERB, never *"şaşırdı"+lar
NOUN). A parse is confident only when the evidence is strong: the surviving root is >=3
chars with a vowel AND at least ``_MIN_STRONG_SUFFIX_LEN[pos]`` chars of inflection were
removed (2 for NOUN, 3 for VERB — a bare 2-char verbal suffix on a synthetic root is too weak
a signal, since Turkish nominal case suffixes are as short as 2 chars but verbal endings
without a copula rarely are). For a short root, a lone weak ending, or a garbage over-strip
it stays conservative and returns the surface as its own stem — never fabricating a root it
cannot justify. Every result is source='guess', never 'lexicon'. Words used here are not in
the seed lexicon so the guesser path is exercised.
"""

from __future__ import annotations

import pytest


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,root",
    [
        ("zonklardan", "zonk"),  # fake root + plural + ablative (2 morphemes)
        ("faktörlerden", "faktör"),  # real OOV noun + plural + ablative (rapor is now lexicon)
    ],
)
def test_guesser_strips_strong_multi_suffix(analyzer, word, root):
    result = analyzer.analyze(word)[0]
    assert result.source == "guess"
    assert result.lemma == root
    assert result.morphemes  # a non-trivial strip
    assert result.pos != "X"


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,root,case,morphemes",
    [
        # A lone 2-char case is now the inclusive minimum strip (threshold lowered from 3 to
        # 2): the possessive is banned, so these strip ONLY the phrase-final case and invent
        # no possessive. motivasyon/tabela pin the genitive-buffer disambiguation (the o/ö
        # tie-break keeps the true consonant-final / vowel-final root). These use still-OOV
        # analogues since enflasyon/kütüphane/başvuru are now lexicon (batch-2 milestone).
        ("motivasyonun", "motivasyon", "genitive", ["un"]),  # NOT motivasyo, NO poss=2sg
        ("sahneye", "sahne", "dative", ["ye"]),  # vowel-final root, strips only the dative
        ("tabelanın", "tabela", "genitive", ["nın"]),  # NOT tabelan (o/ö is irrelevant here)
        ("zonkta", "zonk", "locative", ["ta"]),  # boundary: 2-char strip, root exactly 3 chars
    ],
)
def test_guesser_strips_lone_case_no_possessive(analyzer, word, root, case, morphemes):
    result = analyzer.analyze(word)[0]
    assert result.source == "guess"
    assert result.lemma == root
    assert result.pos == "NOUN"
    assert result.features.get("case") == case
    assert result.features.get("possessive", "none") == "none"
    assert result.morphemes == morphemes


@pytest.mark.positive
def test_guesser_o_final_root_wins_when_unopposed(analyzer):
    # The o/ö final-letter penalty is a sort-key TIE-BREAK, never a filter: an o-final root
    # is still guessed when nothing competes with it (silodan -> silo, ablative).
    result = analyzer.analyze("silodan")[0]
    assert result.source == "guess"
    assert result.lemma == "silo"
    assert result.features.get("case") == "ablative"


@pytest.mark.exception
def test_guesser_strips_case_but_never_possessive(analyzer):
    # Policy (milestone: no fabricated possessive): a word that would parse as
    # plural+possessive+case is stripped of ONLY the phrase-final case — the possessive is
    # never removed and never invented. problemlerimizde -> "problemlerimiz" + locative, not
    # "problem" (which required eating the -imiz possessive).
    result = analyzer.analyze("problemlerimizde")[0]
    assert result.source == "guess"
    assert result.lemma == "problemlerimiz"
    assert result.pos == "NOUN"
    assert result.features.get("case") == "locative"
    assert result.features.get("possessive", "none") == "none"


@pytest.mark.negative
def test_guesser_never_eats_stem_via_buffer_n(analyzer):
    # salgını once over-stripped to "salg" (poss -ın- + acc -ı, eating the stem's -n). With
    # the possessive (and so the pronominal-case buffer -n-) banned, "salg" is unreachable:
    # the only case candidate salgın+ı is a lone 1-char strip (< 2), so the surface stays its
    # own identity and salgın survives as a plausible alternative — never "salg".
    result = analyzer.analyze("salgını")
    primary = result[0]
    assert primary.lemma in ("salgını", "salgın")
    for a in [primary, *getattr(primary, "analyses", [])]:
        assert a.lemma != "salg"


@pytest.mark.negative
@pytest.mark.parametrize("word", ["motivasyonun", "aksiyonu", "tabelanın", "problemlerimizde"])
def test_guesser_never_fabricates_a_possessive(analyzer, word):
    # The guesser strips only [plural]?[case]?, never a possessive: no result (primary or any
    # ranked alternative) may carry a possessive feature other than the "none" default.
    primary = analyzer.analyze(word)[0]
    for a in [primary, *getattr(primary, "analyses", [])]:
        assert a.features.get("possessive", "none") == "none"


@pytest.mark.negative
def test_guesser_conservative_on_lone_short_suffix(analyzer):
    # "zonku" could be zonk+ı(acc) OR a bare root; too ambiguous to strip confidently.
    result = analyzer.analyze("zonku")[0]
    assert result.source == "guess"
    assert result.lemma == "zonku"  # identity, not over-stripped to "zonk"
    # ...but the stripped candidate is preserved as an alternative, not lost.
    assert any(a.lemma == "zonk" for a in result.analyses)


@pytest.mark.negative
def test_guess_is_never_lexicon(analyzer):
    for word in ["zonklardan", "zonku", "qwxlmn", "flarpishmear"]:
        for a in analyzer.analyze(word):
            assert a.source != "lexicon"


@pytest.mark.positive
def test_guesser_does_not_crash_on_junk(analyzer):
    for word in ["", "x", "123", "!!!", "qğşü", "aeiou"]:
        results = analyzer.analyze(word)
        assert isinstance(results, list) and results


@pytest.mark.positive
def test_strong_single_suffix_is_stripped(analyzer):
    # -lardan / -ler etc. are distinctive (>=3 chars): strip even as a single suffix.
    result = analyzer.analyze("zonklar")[0]  # plural only
    assert result.source == "guess"
    assert result.lemma == "zonk"


@pytest.mark.positive
def test_boundary_root_and_suffix_len_are_inclusive(analyzer):
    # zoplar: root "zop" is exactly 3 chars (_MIN_GUESS_ROOT_LEN, inclusive), "lar" is 3
    # chars — well over the 2-char strip minimum. This pins the inclusive root-length
    # boundary; zonkta (above) pins the inclusive 2-char strip minimum.
    result = analyzer.analyze("zoplar")[0]
    assert result.source == "guess"
    assert result.lemma == "zop"


@pytest.mark.negative
@pytest.mark.parametrize(
    "word,over_strip",
    [
        # NB: senle / hazirana used to live here as guesser over-strips; they are now
        # lexicon words (sen PRON, haziran NOUN) and have moved to the closed-class / batch
        # tests. These remaining rows are still genuinely OOV.
        ("geler", "ge"),  # ge+ler: root "ge" has 2 chars -> reject
        ("sene", "se"),  # se+n+e: root "se" has 2 chars -> reject
        ("zolar", "zo"),  # zo+lar: "lar" is 3 chars but root "zo" is 2 -> reject
    ],
)
def test_guesser_does_not_over_strip_to_nonword_root(analyzer, word, over_strip):
    # The guesser must not reduce an unknown word to a 1-2 char non-word root; with weak
    # evidence it falls back to identity (correct until the true root enters the lexicon).
    result = analyzer.analyze(word)[0]
    assert result.source == "guess"
    assert result.lemma != over_strip
    assert result.lemma == word  # identity, not a fabricated root


@pytest.mark.exception
@pytest.mark.parametrize(
    "word,alt",
    [
        # Still-OOV analogues of the old senle/haziran rows (now lexicon): a lone one-char
        # accusative -(y)I is too weak to strip, so the surface stays its own identity, yet
        # the plausible root survives as a ranked alternative.
        ("motoru", "motor"),  # motor+u(acc): 1-char strip -> identity, motor kept as alt
        ("problemi", "problem"),  # problem+i(acc): same shape
    ],
)
def test_rejected_strip_survives_as_alternative(analyzer, word, alt):
    # Falling back to identity must not lose the stripped candidate: it stays in .analyses
    # so a later lexicon entry (rapor, problem) has a ready alternative to promote.
    result = analyzer.analyze(word)[0]
    assert result.lemma == word
    assert any(a.lemma == alt for a in result.analyses)


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,root,morphemes",
    [
        # Regression: an early version of the possessive fix dropped VERB from the guess pos
        # loop entirely (nominal-only), reasoning that the OOV tail grammar is purely
        # [plural]?[case]?. But -lAr is homographic between the nominal plural and the verbal
        # 3rd-person-plural person marker, so a NOUN-only guesser confidently mis-tagged any
        # OOV verb in the past 3pl as a fabricated NOUN+plural (şaşırdılar -> *NOUN "şaşırdı"
        # + lar, eating the past-tense marker "dı" into the lemma) — a new confidently-wrong
        # output, not an improvement. VERB must stay in the race so the correct, more-stripped
        # verbal parse (şaşır + dı + lar) outranks the spurious nominal one.
        ("şaşırdılar", "şaşır", ["dı", "lar"]),
        ("fırlattılar", "fırlat", ["tı", "lar"]),
    ],
)
def test_guesser_prefers_verbal_past_3pl_over_fabricated_noun_plural(
    analyzer, word, root, morphemes
):
    result = analyzer.analyze(word)[0]
    assert result.source == "guess"
    assert result.lemma == root
    assert result.pos == "VERB"
    assert result.features.get("tense") == "past"
    assert result.features.get("person") == "3pl"
    assert result.morphemes == morphemes


@pytest.mark.negative
def test_guesser_never_guesses_bare_verbal_past_without_y_buffer(analyzer):
    # hasta (a real lexicon adjective) takes the nominal ek-fiil only with the obligatory
    # (y) buffer after a vowel: hastaydı, never *hastadı (see
    # test_y_buffer_is_obligatory_after_vowel in test_ek_fiil.py). The OOV guesser must not
    # backdoor this by *confidently* treating "hasta" as a synthetic VERB root and "dı" as a
    # bare 2-char verbal past: VERB requires >=3 stripped chars, so hastadı stays a
    # conservative identity guess as the PRIMARY result (the weak hasta+VERB+dı reading may
    # still surface as a low-confidence alternative — every stripped candidate is kept, see
    # test_rejected_strip_survives_as_alternative — but it must never win).
    result = analyzer.analyze("hastadı")[0]
    assert result.source == "guess"
    assert result.lemma == "hastadı"
    assert result.pos == "X"


@pytest.mark.positive
def test_hazirandan_resolves_to_haziran_lexicon(analyzer):
    # Regression (was a strict xfail: the guesser over-stripped hazirandan -> *hazira+n+dan
    # while haziran was OOV). Now that "haziran" is a lexicon month, the ablative resolves
    # to it cleanly and outranks any guess — the documented fix, not a single-char penalty.
    result = analyzer.analyze("hazirandan")[0]
    assert result.lemma == "haziran"
    assert result.pos == "NOUN"
    assert result.source == "lexicon"
    assert result.features.get("case") == "ablative"
