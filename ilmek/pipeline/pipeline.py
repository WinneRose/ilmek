"""End-to-end pipeline: text -> normalization -> tokenization -> analysis -> Document.

``Pipeline`` is the object returned by :func:`ilmek.load`. Calling it on text
returns a :class:`Document`; the per-word helpers mirror the top-level functional API.
"""

from __future__ import annotations

from ..backends import BaseBackend, NativeBackend
from ..core.document import AnalysisResult, Document
from ..core.normalization import normalize
from ..core.tokenization import tokenize

#: Token kinds that carry morphology worth analyzing.
_ANALYZABLE_KINDS = frozenset({"word"})


class Pipeline:
    """A configured analysis pipeline over a chosen backend (native by default)."""

    def __init__(self, backend: BaseBackend | None = None):
        self.backend = backend or NativeBackend()

    def __call__(self, text: str) -> Document:
        return self.process(text)

    def process(self, text: str) -> Document:
        """Tokenize ``text`` and attach the best analysis (plus alternatives) per word.

        The text is normalized once here so that ``Document.text`` and the token character
        offsets share one coordinate system (normalization may change length, e.g. NFC
        composition or dropping a stray combining mark).
        """
        normalized = normalize(text)
        tokens = tokenize(normalized, normalize_text=False)
        analyses: list[AnalysisResult | None] = []
        for token in tokens:
            if token.kind in _ANALYZABLE_KINDS:
                candidates = self.backend.analyze(token.text)
                best = candidates[0]
                best.analyses = candidates[1:]
                analyses.append(best)
            else:
                analyses.append(None)
        return Document(text=normalized, tokens=tokens, analyses=analyses)

    # -- word-level helpers ------------------------------------------------------------

    def analyze(self, word: str) -> AnalysisResult:
        """Best analysis of a single word."""
        return self.backend.analyze(word)[0]

    def analyze_all(self, word: str) -> list[AnalysisResult]:
        """Every valid analysis of a single word, best first."""
        return self.backend.analyze(word)

    def stem(self, word: str) -> str:
        return self.backend.analyze(word)[0].stem

    def lemmatize(self, word: str) -> str:
        return self.backend.analyze(word)[0].lemma

    def analyze_sentence(self, text: str) -> Document:
        return self.process(text)
