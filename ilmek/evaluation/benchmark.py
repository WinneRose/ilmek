"""Benchmark harness: run the engine over the gold set and report regression metrics.

This is a **top layer** — like the disambiguator, it depends on the pipeline / morphology /
core and nothing depends back on it. It loads a hand-labeled gold dataset
(``ilmek/data/eval/gold.json``), analyzes each surface with the context-free
:class:`~ilmek.morphology.analyzer.Analyzer` (the same engine behind ``ilmek.analyze``),
disambiguates the context items through a :class:`~ilmek.pipeline.pipeline.Pipeline`, then
hands per-item records to the pure math in :mod:`.metrics`.

The report (per-category table + overall) is stable and plain-text (no box-drawing, so a
cp1254 Windows console degrades gracefully; run with ``PYTHONUTF8=1`` for the Turkish
surfaces). :meth:`BenchmarkReport.to_dict` emits the same numbers as JSON for
machine-readable regression comparison across runs.

Timing: the default analyzer is warmed once first, then :func:`time.perf_counter` brackets
only the per-word ``analyze`` calls (never lexicon load or disambiguation), so words/second
reflects steady-state core analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.normalization import turkish_lower
from . import metrics
from .metrics import ItemRecord, Score, Throughput

#: The packaged gold dataset (force-included in the wheel via pyproject, so an installed
#: ``ilmek benchmark`` finds it too).
DEFAULT_GOLD_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "gold.json"

#: The closed set of gold categories. A gold entry outside this set is a typo, caught
#: mechanically by the schema-validation test rather than silently mis-bucketed.
CATEGORIES: tuple[str, ...] = (
    "simple_noun",
    "simple_verb",
    "affix_chain",
    "voicing",
    "vowel_drop",
    "ek_fiil",
    "voice",
    "derivation",
    "ambiguity",
    "unknown",
)


class GoldError(ValueError):
    """Raised when the gold file is missing or structurally invalid (no silent fallback)."""


def load_gold(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the gold dataset (the packaged file by default) as a list of raw entry dicts."""
    gold_path = Path(path) if path is not None else DEFAULT_GOLD_PATH
    if not gold_path.exists():
        raise GoldError(f"gold dataset not found: {gold_path}")
    try:
        data = json.loads(gold_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise GoldError(f"gold dataset is not valid JSON: {gold_path}: {exc}") from exc
    entries = data["entries"] if isinstance(data, dict) else data
    if not isinstance(entries, list) or not entries:
        raise GoldError(f"gold dataset has no entries: {gold_path}")
    return entries


def _resolve_reading(entry: dict[str, Any], reading: dict[str, Any]) -> dict[str, Any]:
    """A fully-resolved ``{lemma,pos,feats}`` reading, inheriting lemma/pos from ``entry``."""
    return {
        "lemma": reading.get("lemma", entry["lemma"]),
        "pos": reading.get("pos", entry["pos"]),
        "feats": reading.get("feats", {}),
    }


def build_records(
    gold: list[dict[str, Any]],
    *,
    analyzer=None,
    pipeline=None,
) -> tuple[list[ItemRecord], Throughput]:
    """Analyze every gold entry into an :class:`ItemRecord`; return records + timing.

    Isolated context-free analysis (the metric backbone) is timed; the optional
    sentence-context disambiguation for context items is *not* counted toward throughput.
    """
    if analyzer is None:
        from ..morphology.analyzer import default_analyzer

        analyzer = default_analyzer()
    if pipeline is None:
        from ..pipeline.pipeline import Pipeline

        pipeline = Pipeline()

    # Warm the analyzer so first-word lexicon/JIT costs are not charged to word 1.
    analyzer.analyze("ev")

    records: list[ItemRecord] = []
    elapsed = 0.0
    n_words = 0
    for entry in gold:
        surface = entry["surface"]
        t0 = time.perf_counter()
        candidates = analyzer.analyze(surface)
        elapsed += time.perf_counter() - t0
        n_words += 1

        chosen = None
        has_context = "context" in entry and entry.get("token_index") is not None
        if has_context:
            doc = pipeline.process(entry["context"])
            idx = entry["token_index"]
            if 0 <= idx < len(doc.analyses):
                chosen = doc.analyses[idx]

        also_valid = tuple(_resolve_reading(entry, r) for r in entry.get("also_valid", ()))
        records.append(
            ItemRecord(
                category=entry["category"],
                gold_lemma=entry["lemma"],
                gold_stem=entry.get("stem", entry["lemma"]),
                gold_pos=entry["pos"],
                gold_feats=entry.get("feats", {}),
                candidates=tuple(candidates),
                also_valid=also_valid,
                gold_source=entry.get("source"),
                chosen=chosen,
                has_context=has_context,
                known_gap=bool(entry.get("known_gap", False)),
            )
        )
    return records, metrics.throughput(elapsed, n_words)


def _metric_block(records: list[ItemRecord]) -> dict[str, Score]:
    """Compute the headline metric bundle over a record list."""
    return {
        "lemma_accuracy": metrics.lemma_accuracy(records),
        "stem_accuracy": metrics.stem_accuracy(records),
        "coverage": metrics.coverage(records),
        "disambiguation_accuracy": metrics.disambiguation_accuracy(records),
        "unknown_word_rate": metrics.unknown_word_rate(records),
    }


@dataclass
class BenchmarkReport:
    """The computed benchmark: per-item records, per-category and overall metric bundles.

    Metric bundles are computed over the **non-known-gap** records, so the deliberately-hard
    ``known_gap`` items (reported separately in :attr:`known_gaps`) never inflate or deflate
    the headline numbers or the CI floors.
    """

    records: list[ItemRecord]
    overall: dict[str, Score]
    by_category: dict[str, dict[str, Score]]
    throughput: Throughput
    known_gaps: list[ItemRecord] = field(default_factory=list)

    @property
    def scored_records(self) -> list[ItemRecord]:
        """Records that count toward the headline metrics (everything but the known gaps)."""
        return [r for r in self.records if not r.known_gap]

    # -- rendering --------------------------------------------------------------------

    @staticmethod
    def _pct(score: Score) -> str:
        v = score.value
        return "n/a" if v is None else f"{v * 100:5.1f}%"

    def format_report(self) -> str:
        lines: list[str] = []
        lines.append(
            f"ilmek benchmark  (gold set: {len(self.records)} items, "
            f"{len(self.scored_records)} scored + {len(self.known_gaps)} known-gap)"
        )
        lines.append("")
        header = (
            f"{'category':<14} {'n':>4} {'lemma':>7} {'stem':>7} "
            f"{'cover':>7} {'disamb':>7} {'unk':>7}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for cat in CATEGORIES:
            block = self.by_category.get(cat)
            if not block:
                continue
            n = block["lemma_accuracy"].total
            lines.append(
                f"{cat:<14} {n:>4} "
                f"{self._pct(block['lemma_accuracy']):>7} "
                f"{self._pct(block['stem_accuracy']):>7} "
                f"{self._pct(block['coverage']):>7} "
                f"{self._pct(block['disambiguation_accuracy']):>7} "
                f"{self._pct(block['unknown_word_rate']):>7}"
            )
        lines.append("-" * len(header))
        n = self.overall["lemma_accuracy"].total
        lines.append(
            f"{'OVERALL':<14} {n:>4} "
            f"{self._pct(self.overall['lemma_accuracy']):>7} "
            f"{self._pct(self.overall['stem_accuracy']):>7} "
            f"{self._pct(self.overall['coverage']):>7} "
            f"{self._pct(self.overall['disambiguation_accuracy']):>7} "
            f"{self._pct(self.overall['unknown_word_rate']):>7}"
        )
        lines.append("")
        tp = self.throughput
        ms = "n/a" if tp.ms_per_word is None else f"{tp.ms_per_word:.3f} ms/word"
        wps = "n/a" if tp.words_per_sec is None else f"{tp.words_per_sec:,.0f} words/sec"
        lines.append(f"speed: {ms}   ({wps}, {tp.n_words} words)")

        if self.known_gaps:
            lines.append("")
            lines.append("known gaps (excluded from the metrics above; documented headroom):")
            for rec in self.known_gaps:
                best = rec.best
                got = "none" if best is None else f"{best.lemma}/{best.pos}[{best.source}]"
                lines.append(f"  - {rec.category}: gold {rec.gold_lemma}/{rec.gold_pos}  got {got}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": len(self.records),
            "n_scored": len(self.scored_records),
            "n_known_gap": len(self.known_gaps),
            "overall": {k: v.to_dict() for k, v in self.overall.items()},
            "categories": {
                cat: {k: v.to_dict() for k, v in block.items()}
                for cat, block in self.by_category.items()
            },
            "throughput": self.throughput.to_dict(),
            "known_gaps": [
                {"surface": turkish_lower(r.gold_lemma), "category": r.category}
                for r in self.known_gaps
            ],
        }


def run_benchmark(
    gold: list[dict[str, Any]] | None = None,
    *,
    path: str | Path | None = None,
    category: str | None = None,
    analyzer=None,
    pipeline=None,
) -> BenchmarkReport:
    """Run the full benchmark and return a :class:`BenchmarkReport`.

    ``category`` optionally restricts the run to one category. ``path`` loads a non-default
    gold file. ``analyzer`` / ``pipeline`` are injectable for tests.
    """
    if gold is None:
        gold = load_gold(path)
    if category is not None:
        gold = [e for e in gold if e.get("category") == category]
        if not gold:
            raise GoldError(f"no gold entries in category {category!r}")

    records, tp = build_records(gold, analyzer=analyzer, pipeline=pipeline)
    scored = [r for r in records if not r.known_gap]
    known_gaps = [r for r in records if r.known_gap]

    by_category: dict[str, dict[str, Score]] = {}
    for cat in CATEGORIES:
        cat_records = [r for r in scored if r.category == cat]
        if cat_records:
            by_category[cat] = _metric_block(cat_records)

    return BenchmarkReport(
        records=records,
        overall=_metric_block(scored),
        by_category=by_category,
        throughput=tp,
        known_gaps=known_gaps,
    )
