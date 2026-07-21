"""Core layer: normalization, tokenization, and the shared data model.

Everything here is context-free and depends on nothing above it (morphology, backends,
disambiguation). This is the bottom of the dependency graph.
"""

from __future__ import annotations

from . import alphabet, tags
from .document import AnalysisResult, Document, Token
from .normalization import (
    fold_for_lookup,
    normalize,
    standardize_apostrophes,
    turkish_lower,
    turkish_upper,
)
from .tokenization import tokenize

__all__ = [
    "alphabet",
    "tags",
    "AnalysisResult",
    "Document",
    "Token",
    "normalize",
    "standardize_apostrophes",
    "fold_for_lookup",
    "turkish_lower",
    "turkish_upper",
    "tokenize",
]
