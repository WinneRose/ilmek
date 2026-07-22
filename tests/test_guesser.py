"""Unknown-word (OOV) guesser: conservative, clearly-marked root proposal.

The guesser proposes a stripped root ONLY when the evidence is strong: the surviving root
is >=3 chars with a vowel AND >=3 chars of inflection were removed. For a short root, a
lone short/ambiguous ending, or a garbage over-strip it stays conservative and returns the
surface as its own stem — never fabricating a root it cannot justify. Every result is
source='guess', never 'lexicon'. Words used here are not in the seed lexicon so the guesser
path is exercised.
"""

from __future__ import annotations

import pytest


@pytest.mark.positive
@pytest.mark.parametrize(
    "word,root",
    [
        ("zonklardan", "zonk"),  # fake root + plural + ablative (2 morphemes)
        ("raporlardan", "rapor"),  # real OOV noun + plural + ablative
        ("problemlerimizde", "problem"),  # OOV + plural + poss + locative
    ],
)
def test_guesser_strips_strong_multi_suffix(analyzer, word, root):
    result = analyzer.analyze(word)[0]
    assert result.source == "guess"
    assert result.lemma == root
    assert result.morphemes  # a non-trivial strip
    assert result.pos != "X"


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
    # zoplar: root "zop" is exactly 3 chars, "lar" is exactly 3 chars — both thresholds
    # are inclusive, so this is the shortest strip the gate still trusts.
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
        ("raporu", "rapor"),  # rapor+ı(acc): 1-char strip -> identity, rapor kept as alt
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
def test_hazirandan_resolves_to_haziran_lexicon(analyzer):
    # Regression (was a strict xfail: the guesser over-stripped hazirandan -> *hazira+n+dan
    # while haziran was OOV). Now that "haziran" is a lexicon month, the ablative resolves
    # to it cleanly and outranks any guess — the documented fix, not a single-char penalty.
    result = analyzer.analyze("hazirandan")[0]
    assert result.lemma == "haziran"
    assert result.pos == "NOUN"
    assert result.source == "lexicon"
    assert result.features.get("case") == "ablative"
