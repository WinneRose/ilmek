"""Morphology layer: lexicon, morphotactics, phonology, and the native analyzer.

Exposes the native engine and the stemmer/lemmatizer views that read off its analyses.
Depends only on :mod:`ilmek.core`; never on backends or disambiguation.
"""

from __future__ import annotations

from .analyzer import Analyzer, default_analyzer
from .lemmatizer import lemmatize
from .lexicon import Lexicon, Root
from .stemmer import stem

__all__ = [
    "Analyzer",
    "default_analyzer",
    "Lexicon",
    "Root",
    "stem",
    "lemmatize",
]
