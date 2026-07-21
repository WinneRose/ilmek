"""Stemmer view over the shared analysis.

The stem is *not* produced by an independent suffix stripper — it is read off the same
morphological analysis the lemmatizer and analyzer use, so the three can never disagree.
For a purely inflected word the stem equals the lemma; when a derivation is present the
stem is the surface at the last derivation boundary (evlilerden -> stem ``evli``,
yaşadıklarımızın -> stem ``yaşadık``) while the lemma stays the base lexeme (``ev``,
``yaşa``). The visible boundary lives in ``features['derivation']``.
"""

from __future__ import annotations

from .analyzer import Analyzer, default_analyzer


def stem(word: str, *, analyzer: Analyzer | None = None) -> str:
    """Return the stem of ``word`` (best analysis)."""
    analyzer = analyzer or default_analyzer()
    return analyzer.analyze(word)[0].stem
