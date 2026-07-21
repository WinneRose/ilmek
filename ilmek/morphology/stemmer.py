"""Stemmer view over the shared analysis.

The stem is *not* produced by an independent suffix stripper — it is read off the same
morphological analysis the lemmatizer and analyzer use, so the three can never disagree.
For v0.1 (inflectional morphology only) the stem equals the lemma; once derivational
morphology lands, the stem may be a derived form while the lemma stays the base lexeme.
"""

from __future__ import annotations

from .analyzer import Analyzer, default_analyzer


def stem(word: str, *, analyzer: Analyzer | None = None) -> str:
    """Return the stem of ``word`` (best analysis)."""
    analyzer = analyzer or default_analyzer()
    return analyzer.analyze(word)[0].stem
