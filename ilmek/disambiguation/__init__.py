"""Disambiguation: a separate, optional layer over the context-free analyzer.

Two halves:

* :mod:`.scoring` — deterministic per-candidate scoring (``score_candidate``,
  ``rank_candidates``) from coarse POS/source priors (rule + coarse "frequency", no ML);
* :mod:`.disambiguator` — :class:`Disambiguator`, a sentence-context re-ranker that picks a
  best analysis per token from left/right context and attaches a bounded **heuristic**
  confidence, keeping every alternative in ``.analyses``.

Import direction is one-way: this package depends on :mod:`ilmek.core` (and reads an
:class:`~ilmek.core.document.AnalysisResult`), never the other way. :mod:`ilmek.core` and
:mod:`ilmek.morphology` must not import this package; only the pipeline wires it in.
"""

from __future__ import annotations

from .disambiguator import Disambiguator
from .scoring import load_weights, rank_candidates, score_candidate

__all__ = ["Disambiguator", "score_candidate", "rank_candidates", "load_weights"]
