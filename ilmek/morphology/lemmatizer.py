"""Lemmatizer view over the shared analysis.

The lemma is the dictionary (citation) form of the word's root lexeme, taken from the
same analysis as :func:`~ilmek.morphology.stemmer.stem`. Contextual lemmatization
(picking among candidates by sentence context) is a later, separate disambiguation layer.
"""

from __future__ import annotations

from .analyzer import Analyzer, default_analyzer


def lemmatize(word: str, *, analyzer: Analyzer | None = None) -> str:
    """Return the lemma of ``word`` (best analysis)."""
    analyzer = analyzer or default_analyzer()
    return analyzer.analyze(word)[0].lemma
