"""Pure evaluation metrics for the morphology engine.

This module is deliberately **I/O-free and analyzer-free**: every function here is a
pure computation over already-produced :class:`~ilmek.core.document.AnalysisResult` objects
and gold labels, so the math is unit-testable on hand-built synthetic records without ever
running the FSM or touching a clock. The benchmark harness (:mod:`.benchmark`) does the
analysis and timing, then hands the per-item records here for scoring.

Match semantics (:func:`analysis_matches`):

* **lemma** — compared with :func:`~ilmek.core.normalization.turkish_lower` on both sides,
  never a locale-naive ``str.lower`` (which mishandles ``I``/``İ``).
* **pos** — exact string equality against the tag vocabulary in :mod:`ilmek.core.tags`.
* **feats** — a **subset** match: every key the gold names must be present in
  ``analysis.features`` and equal; keys the gold does *not* mention are unconstrained, so a
  gold label stays valid as the engine grows new feature keys. Tuple-valued features
  (``voice``, ``derivation``) are normalized so a Python tuple ``("causative",)`` compares
  equal to a gold JSON list ``["causative"]``.

Every aggregator returns a :class:`Score` (``correct``/``total``, whose ``value`` is ``None``
when ``total == 0`` — rendered ``n/a``, never a fabricated ``0%``/``100%``). :func:`throughput`
takes elapsed seconds as an **argument** (the clock is injected by the caller) so tests never
measure wall-clock.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.document import AnalysisResult
from ..core.normalization import turkish_lower


def _normalize_feat(value: Any) -> Any:
    """Normalize a feature value so a tuple compares equal to its JSON-list form."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def analysis_matches(analysis: AnalysisResult, gold: Mapping[str, Any]) -> bool:
    """Whether ``analysis`` satisfies a gold reading (lemma + pos + feats-subset).

    ``gold`` is a mapping with ``"lemma"``, ``"pos"`` and an optional ``"feats"`` dict.
    Returns ``True`` only when the lemma (Turkish-lowercased) and pos are equal and every
    feature the gold names is present and equal in ``analysis.features``.
    """
    if turkish_lower(analysis.lemma) != turkish_lower(str(gold["lemma"])):
        return False
    if analysis.pos != gold["pos"]:
        return False
    for key, want in (gold.get("feats") or {}).items():
        if _normalize_feat(analysis.features.get(key)) != _normalize_feat(want):
            return False
    return True


@dataclass(frozen=True)
class ItemRecord:
    """One scored gold item: its gold label plus what the engine produced.

    ``candidates`` is the context-free analysis of the isolated surface (best first, possibly
    empty). ``chosen`` is the sentence-context disambiguated analysis when the item carries a
    context sentence (``has_context``), else ``None``. ``also_valid`` holds fully-resolved
    extra gold readings (each a ``{"lemma","pos","feats"}`` mapping) that must *also* be
    present among the candidates for the item to count as covered — this is how a genuinely
    ambiguous isolated surface is represented without fabricating one context-free "truth".
    """

    category: str
    gold_lemma: str
    gold_stem: str
    gold_pos: str
    gold_feats: Mapping[str, Any] = field(default_factory=dict)
    candidates: Sequence[AnalysisResult] = ()
    also_valid: Sequence[Mapping[str, Any]] = ()
    gold_source: str | None = None
    chosen: AnalysisResult | None = None
    has_context: bool = False
    known_gap: bool = False

    @property
    def best(self) -> AnalysisResult | None:
        """The primary (best) context-free candidate, or ``None`` if there is none."""
        return self.candidates[0] if self.candidates else None

    @property
    def gold(self) -> dict[str, Any]:
        return {"lemma": self.gold_lemma, "pos": self.gold_pos, "feats": self.gold_feats}


# --- Per-record predicates (each pure, each None-best safe) ---------------------------


def lemma_correct(rec: ItemRecord) -> bool:
    best = rec.best
    return best is not None and turkish_lower(best.lemma) == turkish_lower(rec.gold_lemma)


def stem_correct(rec: ItemRecord) -> bool:
    best = rec.best
    return best is not None and turkish_lower(best.stem) == turkish_lower(rec.gold_stem)


def is_covered(rec: ItemRecord) -> bool:
    """Whether the gold reading *and* every ``also_valid`` reading is among the candidates."""
    targets = [rec.gold, *rec.also_valid]
    return all(any(analysis_matches(c, t) for c in rec.candidates) for t in targets)


def disambiguation_correct(rec: ItemRecord) -> bool:
    return rec.chosen is not None and analysis_matches(rec.chosen, rec.gold)


def is_unknown(rec: ItemRecord) -> bool:
    """Whether the primary reading is an unknown-root guess (``source == "guess"``)."""
    best = rec.best
    return best is not None and best.source == "guess"


def source_correct(rec: ItemRecord) -> bool:
    """Whether the primary reading's ``source`` matches the gold's expected source.

    Vacuously ``True`` when the gold does not pin a source (most lexicon items don't).
    """
    if rec.gold_source is None:
        return True
    best = rec.best
    return best is not None and best.source == rec.gold_source


# --- Aggregates ----------------------------------------------------------------------


@dataclass(frozen=True)
class Score:
    """A ratio ``correct``/``total`` whose ``value`` is ``None`` (not ``0``) when empty."""

    correct: int
    total: int

    @property
    def value(self) -> float | None:
        return self.correct / self.total if self.total else None

    def to_dict(self) -> dict[str, Any]:
        return {"correct": self.correct, "total": self.total, "value": self.value}


@dataclass(frozen=True)
class Throughput:
    """Per-word latency and words/second from injected elapsed seconds (never measured here)."""

    total_seconds: float
    n_words: int

    @property
    def ms_per_word(self) -> float | None:
        return (self.total_seconds / self.n_words) * 1000.0 if self.n_words else None

    @property
    def words_per_sec(self) -> float | None:
        return self.n_words / self.total_seconds if self.total_seconds > 0 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_seconds": self.total_seconds,
            "n_words": self.n_words,
            "ms_per_word": self.ms_per_word,
            "words_per_sec": self.words_per_sec,
        }


def _score(records: Iterable[ItemRecord], predicate) -> Score:
    total = 0
    correct = 0
    for rec in records:
        total += 1
        if predicate(rec):
            correct += 1
    return Score(correct, total)


def lemma_accuracy(records: Iterable[ItemRecord]) -> Score:
    """Fraction of items whose primary analysis lemma equals the gold lemma."""
    return _score(records, lemma_correct)


def stem_accuracy(records: Iterable[ItemRecord]) -> Score:
    """Fraction of items whose primary analysis stem equals the gold stem."""
    return _score(records, stem_correct)


def coverage(records: Iterable[ItemRecord]) -> Score:
    """Fraction of items whose gold reading is present among the produced candidates."""
    return _score(records, is_covered)


def disambiguation_accuracy(records: Iterable[ItemRecord]) -> Score:
    """Fraction of *context* items whose sentence-chosen analysis matches the gold reading.

    Only items carrying a context sentence contribute to the denominator; an item with no
    context is not evidence about disambiguation and is skipped entirely.
    """
    total = 0
    correct = 0
    for rec in records:
        if not rec.has_context:
            continue
        total += 1
        if disambiguation_correct(rec):
            correct += 1
    return Score(correct, total)


def unknown_word_rate(records: Iterable[ItemRecord]) -> Score:
    """Fraction of analyzable items whose primary reading is an unknown-root guess.

    The denominator is items that produced at least one candidate; ``correct`` here counts
    the *unknown* (guessed) primaries, so ``value`` is the unknown-word rate in ``[0, 1]``.
    """
    total = 0
    unknown = 0
    for rec in records:
        if rec.best is None:
            continue
        total += 1
        if is_unknown(rec):
            unknown += 1
    return Score(unknown, total)


def throughput(total_seconds: float, n_words: int) -> Throughput:
    """Build a :class:`Throughput` from injected elapsed seconds and a word count."""
    return Throughput(total_seconds=total_seconds, n_words=n_words)
