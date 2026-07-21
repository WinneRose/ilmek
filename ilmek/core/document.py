"""Shared data model: :class:`Token`, :class:`AnalysisResult`, :class:`Document`.

This is the public contract. Every backend (native, and later Stanza/Zemberek) returns
:class:`AnalysisResult` objects with exactly these fields, so consumers never branch on
which engine produced the answer. The field set is taken verbatim from the project spec's
"Ortak Çıktı Şeması" (common output schema).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import tags


@dataclass(slots=True)
class Token:
    """A single token with its character span in the source text.

    ``kind`` is a coarse class from the tokenizer (``word``, ``number``, ``date``,
    ``abbr``, ``punct``, ``url``, ``mention``, ``hashtag``, ``emoticon``, ``space``).
    For proper nouns written with an apostrophe suffix (``Ankara'da``), ``stem_text``
    holds the part before the apostrophe (``Ankara``) and ``apostrophe_suffix`` the part
    after (``da``); this keeps the proper noun and its inflection separately analyzable,
    as the Turkish-language requirements demand.
    """

    text: str
    start: int
    end: int
    kind: str = "word"
    apostrophe_suffix: str | None = None

    @property
    def stem_text(self) -> str:
        """Surface up to an apostrophe suffix, else the whole token."""
        if self.apostrophe_suffix is not None and "'" in self.text:
            return self.text.split("'", 1)[0]
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "apostrophe_suffix": self.apostrophe_suffix,
        }


@dataclass(slots=True)
class AnalysisResult:
    """One morphological analysis of a surface form.

    ``stem`` and ``lemma`` are *derived from the same analysis* — never computed by
    independent suffix strippers. ``analyses`` carries the other valid candidates for the
    same surface form (never silently dropped). ``confidence`` stays ``None`` unless a
    scoring/disambiguation layer sets it — we never fabricate a score. ``source``
    separates lexicon-verified analyses from unknown-root guesses.
    """

    surface: str
    lemma: str
    stem: str
    pos: str
    morphemes: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    analyses: list[AnalysisResult] = field(default_factory=list)
    confidence: float | None = None
    backend: str = tags.BACKEND_NATIVE
    source: str = tags.SOURCE_LEXICON

    def to_dict(self, *, include_candidates: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "surface": self.surface,
            "lemma": self.lemma,
            "stem": self.stem,
            "pos": self.pos,
            "morphemes": list(self.morphemes),
            "features": dict(self.features),
            "confidence": self.confidence,
            "backend": self.backend,
            "source": self.source,
        }
        if include_candidates:
            # One level deep only — candidates don't recurse into their own candidates.
            data["analyses"] = [a.to_dict(include_candidates=False) for a in self.analyses]
        return data

    def feats_string(self) -> str:
        """Render ``features`` as a CoNLL-U-style ``Key=Value|...`` string (or ``_``)."""
        if not self.features:
            return "_"
        parts = []
        for key in sorted(self.features):
            value = self.features[key]
            if value is True:
                value = "Yes"
            parts.append(f"{key}={value}")
        return "|".join(parts)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        morphs = "+".join([self.lemma, *self.morphemes]) if self.morphemes else self.lemma
        return f"AnalysisResult({self.surface!r} -> {morphs} [{self.pos}] source={self.source})"


@dataclass(slots=True)
class Document:
    """A processed text: the original string plus its tokens (and their analyses)."""

    text: str
    tokens: list[Token] = field(default_factory=list)
    #: Parallel to ``tokens``: the chosen/first analysis per token (may be ``None`` for
    #: punctuation or unanalyzed tokens). Alternative candidates live in ``.analyses``.
    analyses: list[AnalysisResult | None] = field(default_factory=list)

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def lemmas(self) -> list[str | None]:
        return [a.lemma if a is not None else None for a in self.analyses]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tokens": [t.to_dict() for t in self.tokens],
            "analyses": [a.to_dict() if a is not None else None for a in self.analyses],
        }
