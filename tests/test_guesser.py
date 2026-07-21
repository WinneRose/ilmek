"""Unknown-word (OOV) guesser: conservative, clearly-marked root proposal.

The guesser proposes a stripped root ONLY when the evidence is strong (a multi-suffix
chain, or a single distinctive suffix of >=3 chars). For a lone short/ambiguous ending it
stays conservative and returns the surface as its own stem — never fabricating a root it
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
