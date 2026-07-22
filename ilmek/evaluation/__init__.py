"""Evaluation & benchmark harness (a separate top layer, like disambiguation).

Two halves:

* :mod:`.metrics` — pure, analyzer-free, clock-free scoring functions
  (:func:`~.metrics.analysis_matches`, :func:`~.metrics.lemma_accuracy`,
  :func:`~.metrics.coverage`, :func:`~.metrics.disambiguation_accuracy`,
  :func:`~.metrics.unknown_word_rate`, :func:`~.metrics.throughput`);
* :mod:`.benchmark` — :func:`~.benchmark.load_gold` / :func:`~.benchmark.run_benchmark`
  over the packaged gold dataset, producing a :class:`~.benchmark.BenchmarkReport` with a
  readable per-category report and a JSON ``to_dict`` for regression comparison.

Import direction is one-way: this package depends on :mod:`ilmek.core`,
:mod:`ilmek.morphology` and :mod:`ilmek.pipeline`; none of them import it (the CLI wires it
in via a lazy import, mirroring the disambiguator).
"""

from __future__ import annotations

from .benchmark import BenchmarkReport, GoldError, load_gold, run_benchmark
from .metrics import (
    ItemRecord,
    Score,
    Throughput,
    analysis_matches,
    coverage,
    disambiguation_accuracy,
    lemma_accuracy,
    stem_accuracy,
    throughput,
    unknown_word_rate,
)

__all__ = [
    "BenchmarkReport",
    "GoldError",
    "load_gold",
    "run_benchmark",
    "ItemRecord",
    "Score",
    "Throughput",
    "analysis_matches",
    "coverage",
    "disambiguation_accuracy",
    "lemma_accuracy",
    "stem_accuracy",
    "throughput",
    "unknown_word_rate",
]
