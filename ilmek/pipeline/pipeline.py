"""End-to-end pipeline: text -> normalization -> tokenization -> analysis -> Document.

``Pipeline`` is the object returned by :func:`ilmek.load`. Calling it on text
returns a :class:`Document`; the per-word helpers mirror the top-level functional API.

Sentence-level processing runs the *optional* disambiguation layer
(:mod:`ilmek.disambiguation`) to pick a best analysis per token from context, keeping every
alternative in ``.analyses``. That layer is imported **lazily** (only when a disambiguating
pipeline is constructed), so ``import ilmek.morphology`` never pulls it in and the core stays
free of any dependency on it. The per-word helpers bypass disambiguation entirely, so
word-level analysis is context-free and its ``confidence`` stays ``None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..backends import BaseBackend, NativeBackend
from ..core.document import AnalysisResult, Document
from ..core.normalization import normalize
from ..core.tokenization import tokenize

if TYPE_CHECKING:  # avoid importing the disambiguation layer at module load time
    from ..disambiguation import Disambiguator

#: Token kinds that carry morphology worth analyzing. ``number``/``date``/``time`` resolve as
#: NUM by rule (the analyzer's numeric fast path), so their Document analyses are populated too;
#: ``abbr`` (Dr., T.C.) stays unanalyzed (its analysis is ``None``).
_ANALYZABLE_KINDS = frozenset({"word", "number", "date", "time"})


class Pipeline:
    """A configured analysis pipeline over a chosen backend (native by default).

    ``disambiguator`` selects the sentence-context layer: ``"auto"`` (default) lazily builds
    the packaged :class:`~ilmek.disambiguation.Disambiguator`; ``None`` disables it (a purely
    context-free document, the pre-milestone behavior); or pass a ready instance.
    """

    def __init__(
        self,
        backend: BaseBackend | None = None,
        disambiguator: Disambiguator | str | None = "auto",
    ):
        self.backend = backend or NativeBackend()
        if disambiguator == "auto":
            # Lazy import: keeps `import ilmek.morphology` free of the disambiguation layer.
            from ..disambiguation import Disambiguator

            self.disambiguator: Disambiguator | None = Disambiguator()
        elif disambiguator is None:
            self.disambiguator = None
        else:
            self.disambiguator = disambiguator

    def __call__(self, text: str) -> Document:
        return self.process(text)

    def process(self, text: str) -> Document:
        """Tokenize ``text`` and attach the best analysis (plus alternatives) per word.

        The text is normalized once here so that ``Document.text`` and the token character
        offsets share one coordinate system (normalization may change length, e.g. NFC
        composition or dropping a stray combining mark).

        With a disambiguator set (the default), the per-token candidate lists are re-ranked
        by sentence context — a best analysis is chosen per token and the rest are kept in
        ``.analyses`` — otherwise the analyzer's own primary is kept, exactly as before.
        """
        normalized = normalize(text)
        tokens = tokenize(normalized, normalize_text=False)
        candidate_lists: list[list[AnalysisResult] | None] = [
            self.backend.analyze(token.text) if token.kind in _ANALYZABLE_KINDS else None
            for token in tokens
        ]

        if self.disambiguator is not None:
            analyses = self.disambiguator.rerank(candidate_lists)
        else:
            analyses = []
            for candidates in candidate_lists:
                if candidates:
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
