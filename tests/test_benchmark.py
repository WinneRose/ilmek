"""Benchmark harness tests: gold-schema validation, floors, determinism, consistency.

These exercise the real packaged gold set and the real engine (unlike
``test_evaluation_metrics``, which tests the pure math on synthetic records). The schema
test catches gold typos mechanically *before* a human reviews the labels; the floor test
guards against a regression in the engine; and the consistency test enforces the AGENTS.md
rule that stem / lemma / analyze are three views of one analysis.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import ilmek
from ilmek.core import tags
from ilmek.core.normalization import normalize, turkish_lower
from ilmek.core.tokenization import tokenize
from ilmek.evaluation import analysis_matches, load_gold
from ilmek.evaluation.benchmark import CATEGORIES, GoldError, run_benchmark

# --- Closed feature vocabularies (mirrors ilmek.core.tags) ----------------------------

_SCALAR_VOCAB = {
    "number": tags.NUMBERS,
    "num_type": tags.NUM_TYPES,
    "possessive": tags.PERSONS,
    "case": tags.CASES,
    "polarity": tags.POLARITIES,
    "tense": tags.TENSES,
    "aspect": frozenset({"progressive"}),
    "copula": tags.COPULAS,
    "mood": tags.MOODS,
    "person": tags.PERSONS,
    "pron_type": tags.PRON_TYPES,
    "register": tags.REGISTERS,
}
_BOOL_FEATS = frozenset({"evidential", "ability", "existential"})
_LIST_VOCAB = {"voice": tags.VOICES, "derivation": tags.DERIVATIONS}


@pytest.fixture(scope="module")
def gold():
    return load_gold()


@pytest.fixture(scope="module")
def report():
    return run_benchmark()


def _validate_feats(feats, where):
    for key, value in feats.items():
        if key in _SCALAR_VOCAB:
            assert value in _SCALAR_VOCAB[key], f"{where}: {key}={value!r} not in vocabulary"
        elif key in _BOOL_FEATS:
            assert isinstance(value, bool), f"{where}: {key} must be boolean, got {value!r}"
        elif key in _LIST_VOCAB:
            assert isinstance(value, list), f"{where}: {key} must be a list, got {value!r}"
            for item in value:
                assert item in _LIST_VOCAB[key], f"{where}: {key} item {item!r} invalid"
        else:
            raise AssertionError(f"{where}: unknown feature key {key!r}")


# =====================================================================================
# Schema validation (mechanical — catches gold typos before human review)
# =====================================================================================


@pytest.mark.positive
def test_gold_entries_have_required_fields_and_valid_pos(gold):
    for e in gold:
        where = f"entry {e.get('surface')!r}"
        assert set(e) >= {"surface", "category", "lemma", "pos"}, f"{where}: missing fields"
        assert e["pos"] in tags.POS_TAGS, f"{where}: pos {e['pos']!r} not a POS tag"
        assert e["category"] in CATEGORIES, f"{where}: category {e['category']!r} unknown"
        assert isinstance(e["lemma"], str) and e["lemma"], f"{where}: bad lemma"
        assert isinstance(e.get("stem", e["lemma"]), str)


@pytest.mark.positive
def test_gold_feature_keys_and_values_are_in_the_closed_vocabularies(gold):
    for e in gold:
        _validate_feats(e.get("feats", {}), f"entry {e['surface']!r}")
        for i, reading in enumerate(e.get("also_valid", ())):
            _validate_feats(reading.get("feats", {}), f"{e['surface']!r} also_valid[{i}]")


@pytest.mark.positive
def test_gold_context_entries_have_a_valid_token_index(gold):
    for e in gold:
        if "context" not in e:
            continue
        where = f"entry {e['surface']!r}"
        assert "token_index" in e, f"{where}: context without token_index"
        tokens = tokenize(normalize(e["context"]), normalize_text=False)
        idx = e["token_index"]
        assert 0 <= idx < len(tokens), f"{where}: token_index {idx} out of range"
        # The surface must actually be the token at that index (Turkish-cased comparison).
        assert turkish_lower(tokens[idx].text) == turkish_lower(e["surface"]), (
            f"{where}: token[{idx}]={tokens[idx].text!r} != surface"
        )


@pytest.mark.positive
def test_gold_unknown_entries_declare_guess_source(gold):
    for e in gold:
        if e["category"] == "unknown":
            assert e.get("source") == "guess", f"{e['surface']!r}: unknown must pin source=guess"


@pytest.mark.consistency
def test_gold_category_quotas_are_roughly_balanced(gold):
    from collections import Counter

    counts = Counter(e["category"] for e in gold)
    # Every documented category is represented; the set is not lopsided onto one bucket.
    for cat in CATEGORIES:
        assert counts[cat] >= 7, f"category {cat} underpopulated ({counts[cat]})"
    assert 120 <= len(gold) <= 220


# =====================================================================================
# End-to-end floors (conservative, so lexicon growth doesn't break CI)
# =====================================================================================


@pytest.mark.positive
def test_benchmark_meets_conservative_floors(report):
    scored = report.scored_records
    assert len(scored) >= 120
    assert report.overall["lemma_accuracy"].value >= 0.90
    assert report.overall["coverage"].value >= 0.90
    assert report.overall["stem_accuracy"].value >= 0.90
    assert report.overall["disambiguation_accuracy"].value >= 0.80


@pytest.mark.positive
def test_every_category_is_reported(report):
    for cat in CATEGORIES:
        assert cat in report.by_category, f"category {cat} missing from report"
        assert report.by_category[cat]["lemma_accuracy"].total > 0


@pytest.mark.positive
def test_report_text_names_metrics_and_categories(report):
    text = report.format_report()
    for token in ("lemma", "stem", "cover", "disamb", "cand", "words/sec", "OVERALL"):
        assert token in text
    for cat in CATEGORIES:
        assert cat in text


@pytest.mark.positive
def test_benchmark_reports_candidate_count_diagnostics(report):
    stats = report.candidate_counts
    assert stats.total == len(report.scored_records)
    assert stats.analyzable + stats.zero == stats.total
    assert stats.total_candidates >= stats.analyzable
    assert stats.max_candidates >= 1

    payload = report.to_dict()
    assert payload["candidate_counts"]["total"] == len(report.scored_records)
    assert set(payload["candidate_counts_by_category"]) == set(CATEGORIES)


# =====================================================================================
# known_gap: excluded from the floors, still counted + reported
# =====================================================================================


@pytest.mark.exception
def test_known_gaps_are_reported_and_genuinely_fail(report):
    # Positive half: the known-gap section exists.
    assert report.known_gaps, "expected documented known-gap entries"
    text = report.format_report()
    assert "known gaps" in text
    # ...and each gap is a REAL gap: the engine's primary does NOT match the gold reading
    # (otherwise it would be a cherry-pick masquerading as headroom).
    for rec in report.known_gaps:
        best = rec.best
        assert best is None or not analysis_matches(best, rec.gold), (
            f"{rec.gold_lemma!r} is labeled known_gap but the engine already matches it"
        )


@pytest.mark.negative
def test_floors_are_computed_over_scored_records_only(report):
    # Negative half: the known gaps are NOT in any metric denominator, so their failure can
    # never move the floors. Overall totals equal the scored (non-gap) count exactly.
    n_scored = len(report.scored_records)
    assert report.overall["lemma_accuracy"].total == n_scored
    assert report.overall["coverage"].total == n_scored
    assert n_scored + len(report.known_gaps) == len(report.records)


# =====================================================================================
# Determinism (deterministic core) + consistency (stem/lemma/analyze agree)
# =====================================================================================


@pytest.mark.consistency
def test_benchmark_accuracy_is_deterministic_across_runs():
    # Timing differs run to run, but every accuracy number is identical (deterministic core).
    a = run_benchmark().to_dict()
    b = run_benchmark().to_dict()
    assert a["overall"] == b["overall"]
    assert a["categories"] == b["categories"]


@pytest.mark.consistency
def test_stem_lemma_analyze_agree_for_every_non_gap_gold_entry(gold):
    for e in gold:
        if e.get("known_gap") or "context" in e:
            continue  # gaps are known-wrong; context entries are scored in-sentence
        surface = e["surface"]
        best = ilmek.analyze(surface)[0]
        # Three views of one analysis agree with each other...
        assert ilmek.stem(surface) == best.stem
        assert ilmek.lemmatize(surface) == best.lemma
        # ...and with the gold lemma/stem (invariant across the surface's ambiguous readings).
        assert turkish_lower(best.lemma) == turkish_lower(e["lemma"]), surface
        assert turkish_lower(best.stem) == turkish_lower(e.get("stem", e["lemma"])), surface


# =====================================================================================
# Loader errors (no silent fallback)
# =====================================================================================


@pytest.mark.negative
def test_load_gold_missing_file_raises():
    with pytest.raises(GoldError):
        load_gold("this-file-does-not-exist.json")


@pytest.mark.negative
def test_run_benchmark_unknown_category_raises():
    with pytest.raises(GoldError):
        run_benchmark(category="no-such-category")


# =====================================================================================
# Architecture: evaluation is a top layer (core/morphology must not import it)
# =====================================================================================


@pytest.mark.negative
def test_importing_morphology_does_not_import_evaluation():
    """No cycle: importing the analyzer must not pull in the evaluation layer."""
    repo_root = Path(__file__).resolve().parents[1]
    code = (
        "import ilmek.morphology.analyzer, sys;"
        "leaked=[m for m in sys.modules if m.startswith('ilmek.evaluation')];"
        "assert not leaked, leaked"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
