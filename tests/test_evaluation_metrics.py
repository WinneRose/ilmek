"""Unit tests for the pure evaluation metrics (no analyzer, no clock involved).

Every record here is hand-built from :class:`AnalysisResult` dataclasses, so these tests pin
the *math and match-semantics* independently of the morphology engine: the analyzer could
change and these assertions would still guard the metric definitions.
"""

from __future__ import annotations

import pytest

from ilmek.core.document import AnalysisResult
from ilmek.evaluation import metrics
from ilmek.evaluation.metrics import (
    ItemRecord,
    analysis_matches,
    candidate_count_stats,
    coverage,
    disambiguation_accuracy,
    lemma_accuracy,
    stem_accuracy,
    throughput,
    unknown_word_rate,
)


def mk(lemma, pos, feats=None, *, stem=None, source="lexicon", surface="x"):
    """A minimal AnalysisResult for metric tests."""
    return AnalysisResult(
        surface=surface,
        lemma=lemma,
        stem=stem if stem is not None else lemma,
        pos=pos,
        morphemes=[],
        features=feats or {},
        source=source,
    )


def rec(candidates, gold_lemma, gold_pos, gold_feats=None, **kw):
    return ItemRecord(
        category=kw.pop("category", "test"),
        gold_lemma=gold_lemma,
        gold_stem=kw.pop("gold_stem", gold_lemma),
        gold_pos=gold_pos,
        gold_feats=gold_feats or {},
        candidates=tuple(candidates),
        **kw,
    )


# =====================================================================================
# analysis_matches — positive / negative / exception
# =====================================================================================


@pytest.mark.positive
def test_analysis_matches_lemma_pos_and_feature_subset():
    a = mk("ev", "NOUN", {"case": "ablative", "number": "plural"})
    # gold names only a subset of features; unmentioned keys are unconstrained.
    assert analysis_matches(a, {"lemma": "ev", "pos": "NOUN", "feats": {"case": "ablative"}})
    assert analysis_matches(a, {"lemma": "ev", "pos": "NOUN"})  # no feats -> lemma+pos only


@pytest.mark.negative
def test_analysis_matches_wrong_lemma_and_wrong_value():
    a = mk("gel", "VERB", {"tense": "past"})
    assert not analysis_matches(a, {"lemma": "gül", "pos": "VERB"})  # lemma differs
    # right key, wrong value: dative vs ablative
    b = mk("ev", "NOUN", {"case": "dative"})
    assert not analysis_matches(b, {"lemma": "ev", "pos": "NOUN", "feats": {"case": "ablative"}})


@pytest.mark.negative
def test_analysis_matches_missing_feature_key_is_not_a_match():
    # gold requires possessive=3sg but the candidate lacks the key entirely (subset requires
    # presence, not just non-contradiction).
    a = mk("ev", "NOUN", {"case": "accusative"})
    assert not analysis_matches(a, {"lemma": "ev", "pos": "NOUN", "feats": {"possessive": "3sg"}})


@pytest.mark.exception
def test_analysis_matches_uses_turkish_casing_not_naive_lower():
    # Naive str.lower mishandles the dotted-I: "İstanbul".lower() keeps a combining dot.
    a = mk("İstanbul", "PROPN")
    assert analysis_matches(a, {"lemma": "istanbul", "pos": "PROPN"})
    assert "İstanbul".lower() != "istanbul"  # proof the naive path would MIScompare
    # ...and the back-vowel dotless-i pair too.
    b = mk("ISI", "NOUN")
    assert analysis_matches(b, {"lemma": "ısı", "pos": "NOUN"})
    assert "ISI".lower() != "ısı"


@pytest.mark.exception
def test_analysis_matches_tuple_feature_equals_json_list():
    # The engine emits voice as a Python tuple; gold arrives from JSON as a list. They match.
    a = mk("yaz", "VERB", {"voice": ("causative", "passive")})
    assert analysis_matches(
        a, {"lemma": "yaz", "pos": "VERB", "feats": {"voice": ["causative", "passive"]}}
    )
    # order matters: a reversed list must NOT match (voice is ordered).
    assert not analysis_matches(
        a, {"lemma": "yaz", "pos": "VERB", "feats": {"voice": ["passive", "causative"]}}
    )


@pytest.mark.exception
def test_analysis_matches_boolean_feature():
    a = mk("gel", "VERB", {"ability": True})
    assert analysis_matches(a, {"lemma": "gel", "pos": "VERB", "feats": {"ability": True}})
    assert not analysis_matches(a, {"lemma": "gel", "pos": "VERB", "feats": {"ability": False}})


# =====================================================================================
# lemma / stem accuracy
# =====================================================================================


@pytest.mark.positive
def test_lemma_and_stem_accuracy_use_the_primary_candidate():
    records = [
        rec([mk("kitap", "NOUN")], "kitap", "NOUN"),
        rec([mk("ev", "NOUN", stem="evli")], "ev", "ADJ", gold_stem="evli"),
    ]
    assert lemma_accuracy(records).value == 1.0
    assert stem_accuracy(records).value == 1.0


@pytest.mark.negative
def test_lemma_accuracy_counts_wrong_primary_lemma():
    records = [
        rec([mk("gel", "VERB")], "gül", "VERB"),  # wrong
        rec([mk("kitap", "NOUN")], "kitap", "NOUN"),  # right
    ]
    score = lemma_accuracy(records)
    assert score.correct == 1 and score.total == 2
    assert score.value == 0.5


@pytest.mark.negative
def test_stem_accuracy_distinguishes_stem_from_lemma():
    # A derived word: lemma is the base, stem is the derived surface. A record claiming the
    # stem equals the lemma is stem-incorrect even though the lemma matches.
    r = rec([mk("ev", "ADJ", stem="evli")], "ev", "ADJ", gold_stem="ev")
    assert lemma_accuracy([r]).value == 1.0
    assert stem_accuracy([r]).value == 0.0


# =====================================================================================
# coverage (+ also_valid)
# =====================================================================================


@pytest.mark.positive
def test_coverage_finds_gold_among_non_primary_candidates():
    # gold is the accusative reading, which is the SECOND candidate (primary is possessive).
    cands = [
        mk("ev", "NOUN", {"possessive": "3sg"}),
        mk("ev", "NOUN", {"case": "accusative"}),
    ]
    r = rec(cands, "ev", "NOUN", {"case": "accusative"})
    assert coverage([r]).value == 1.0


@pytest.mark.negative
def test_coverage_miss_when_gold_reading_absent():
    # Only an accusative candidate exists; gold wants poss3sg -> coverage miss.
    r = rec([mk("ev", "NOUN", {"case": "accusative"})], "ev", "NOUN", {"possessive": "3sg"})
    assert coverage([r]).value == 0.0


@pytest.mark.exception
def test_coverage_requires_every_also_valid_reading_present():
    cands = [mk("çiçek", "NOUN", {"case": "accusative"})]  # missing the poss3sg reading
    r = ItemRecord(
        category="voicing",
        gold_lemma="çiçek",
        gold_stem="çiçek",
        gold_pos="NOUN",
        gold_feats={"case": "accusative"},
        candidates=tuple(cands),
        also_valid=({"lemma": "çiçek", "pos": "NOUN", "feats": {"possessive": "3sg"}},),
    )
    assert coverage([r]).value == 0.0  # gold present but also_valid absent -> not covered
    # Add the missing reading and it becomes covered.
    r2 = ItemRecord(
        category="voicing",
        gold_lemma="çiçek",
        gold_stem="çiçek",
        gold_pos="NOUN",
        gold_feats={"case": "accusative"},
        candidates=(
            mk("çiçek", "NOUN", {"case": "accusative"}),
            mk("çiçek", "NOUN", {"possessive": "3sg"}),
        ),
        also_valid=({"lemma": "çiçek", "pos": "NOUN", "feats": {"possessive": "3sg"}},),
    )
    assert coverage([r2]).value == 1.0


# =====================================================================================
# disambiguation accuracy (context-only denominator)
# =====================================================================================


@pytest.mark.positive
def test_disambiguation_accuracy_only_counts_context_records():
    ctx_hit = ItemRecord(
        category="ambiguity",
        gold_lemma="ev",
        gold_stem="ev",
        gold_pos="NOUN",
        gold_feats={"case": "accusative"},
        candidates=(mk("ev", "NOUN", {"case": "accusative"}),),
        chosen=mk("ev", "NOUN", {"case": "accusative"}),
        has_context=True,
    )
    ctx_miss = ItemRecord(
        category="ambiguity",
        gold_lemma="ev",
        gold_stem="ev",
        gold_pos="NOUN",
        gold_feats={"possessive": "3sg"},
        candidates=(mk("ev", "NOUN", {"case": "accusative"}),),
        chosen=mk("ev", "NOUN", {"case": "accusative"}),  # picked the wrong reading
        has_context=True,
    )
    no_ctx = rec([mk("kitap", "NOUN")], "kitap", "NOUN")  # not counted at all
    score = disambiguation_accuracy([ctx_hit, ctx_miss, no_ctx])
    assert score.correct == 1 and score.total == 2  # the no-context record is excluded


# =====================================================================================
# unknown-word rate
# =====================================================================================


@pytest.mark.positive
def test_unknown_word_rate_fraction_of_guessed_primaries():
    records = [
        rec([mk("ev", "NOUN")], "ev", "NOUN"),
        rec([mk("zonk", "NOUN", source="guess")], "zonk", "NOUN"),
    ]
    score = unknown_word_rate(records)
    assert score.correct == 1 and score.total == 2
    assert score.value == 0.5


@pytest.mark.negative
def test_unknown_word_rate_is_zero_over_all_lexicon_records():
    records = [rec([mk("ev", "NOUN")], "ev", "NOUN"), rec([mk("kitap", "NOUN")], "kitap", "NOUN")]
    assert unknown_word_rate(records).value == 0.0


# =====================================================================================
# candidate-count diagnostics
# =====================================================================================


@pytest.mark.positive
def test_candidate_count_stats_reports_ambiguity_and_empty_results():
    records = [
        rec([mk("ev", "NOUN")], "ev", "NOUN"),
        rec(
            [mk("yüz", "NOUN", source="guess"), mk("yüz", "VERB")],
            "yüz",
            "NOUN",
        ),
        rec([], "bilinmeyen", "X"),
    ]

    stats = candidate_count_stats(records)

    assert stats.total == 3
    assert stats.analyzable == 2
    assert stats.zero == 1
    assert stats.single == 1
    assert stats.multiple == 1
    assert stats.total_candidates == 3
    assert stats.guessed_primary == 1
    assert stats.mean == pytest.approx(1.0)
    assert stats.max_candidates == 2


@pytest.mark.negative
def test_candidate_count_stats_empty_input_has_safe_sentinels():
    stats = candidate_count_stats([])

    assert stats.total == 0
    assert stats.mean is None
    assert stats.max_candidates == 0
    assert stats.guessed_primary == 0
    assert stats.to_dict()["mean"] is None


# =====================================================================================
# n == 0 sentinels (no ZeroDivisionError, value is None not a fake 0/100%)
# =====================================================================================


@pytest.mark.negative
@pytest.mark.parametrize(
    "agg", [lemma_accuracy, stem_accuracy, coverage, disambiguation_accuracy, unknown_word_rate]
)
def test_empty_record_list_returns_none_sentinel(agg):
    score = agg([])
    assert score.total == 0
    assert score.value is None  # never a fabricated 0% or 100%


@pytest.mark.negative
def test_disambiguation_accuracy_none_when_no_context_records():
    # Records exist but none carry context: denominator is 0 -> value None.
    records = [rec([mk("ev", "NOUN")], "ev", "NOUN")]
    assert disambiguation_accuracy(records).value is None


# =====================================================================================
# throughput — pure arithmetic on an injected clock
# =====================================================================================


@pytest.mark.positive
def test_throughput_arithmetic():
    tp = throughput(2.0, 100)
    assert tp.words_per_sec == pytest.approx(50.0)
    assert tp.ms_per_word == pytest.approx(20.0)


@pytest.mark.negative
def test_throughput_handles_zero_without_dividing():
    tp = throughput(0.0, 0)
    assert tp.ms_per_word is None
    assert tp.words_per_sec is None


@pytest.mark.consistency
def test_score_to_dict_shape():
    score = metrics.Score(3, 4)
    assert score.to_dict() == {"correct": 3, "total": 4, "value": 0.75}
