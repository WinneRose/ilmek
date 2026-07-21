"""Native backend: the packaged, offline, deterministic morphology engine.

This is the project's core value — it needs no model download and no network. It simply
wraps :class:`~ilmek.morphology.analyzer.Analyzer` behind the backend contract.
"""

from __future__ import annotations

from ..core import tags
from ..core.document import AnalysisResult
from ..morphology.analyzer import Analyzer, default_analyzer
from .base import BaseBackend


class NativeBackend(BaseBackend):
    name = tags.BACKEND_NATIVE

    def __init__(self, analyzer: Analyzer | None = None):
        self.analyzer = analyzer or default_analyzer()

    def analyze(self, word: str) -> list[AnalysisResult]:
        return self.analyzer.analyze(word)
