"""Backend protocol: the contract every engine (native, Stanza, Zemberek) satisfies.

Consumers depend on this shape, never on a concrete engine. Every backend returns the
same :class:`AnalysisResult` schema and stamps ``backend`` so results stay comparable and
their provenance stays visible.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.document import AnalysisResult


@runtime_checkable
class Backend(Protocol):
    """Minimal analysis contract."""

    name: str

    def analyze(self, word: str) -> list[AnalysisResult]:
        """Return all analyses for ``word``, best candidate first."""
        ...


class BaseBackend:
    """Convenience base providing stem/lemmatize on top of :meth:`analyze`."""

    name: str = "base"

    def analyze(self, word: str) -> list[AnalysisResult]:  # pragma: no cover - abstract
        raise NotImplementedError

    def stem(self, word: str) -> str:
        return self.analyze(word)[0].stem

    def lemmatize(self, word: str) -> str:
        return self.analyze(word)[0].lemma
