"""Deterministic rule + coarse-frequency scoring of analysis candidates.

This is the **candidate scoring** half of the disambiguation layer. It is a *separate,
optional* layer: :mod:`ilmek.core` and :mod:`ilmek.morphology` never import it, so the
context-free :class:`~ilmek.morphology.analyzer.Analyzer` keeps returning every candidate
with ``confidence=None`` exactly as before.

``score_candidate`` assigns each :class:`~ilmek.core.document.AnalysisResult` a **heuristic**
plausibility score built only from features the analysis already carries:

* ``source`` rank — a lexicon-verified analysis beats a rule/apostrophe one, which beats an
  unknown-root ``guess`` (mirrors the analyzer's own ``source`` tie-break);
* a coarse ``pos`` prior — content parts of speech (NOUN/VERB/ADJ/...) sit above closed
  function classes, and both sit above the ``X`` unanalyzable fallback. **Content POS share
  one prior**, so a same-length ADJ/NOUN homograph is a tie broken by the analyzer's own
  order — the score never silently contradicts the core ranking;
* penalties for more morphemes / more derivations / being a nominal *predicate* reading —
  the same signals the analyzer's ``_sort_key`` already prefers, so on a token with no
  context evidence the score agrees with the analyzer's primary;
* a tiny root-length bonus mirroring the analyzer's ``-len(lemma)`` tie-break.

The word "frequency" here means these **coarse, hand-set POS/source priors** — NOT corpus
counts (this repo ships no corpus). ``score_candidate`` reads an optional per-lemma
``frequency`` off a result's features if one is ever added (weight ``frequency_weight``,
``0.0`` today), a documented extension point that is a no-op until such data exists.

Every weight lives in ``data/disambiguation/weights.json``; this module holds only the
generic machinery. Scores are strictly positive so the sentence layer can sum-normalize
them into a bounded heuristic confidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import tags
from ..core.document import AnalysisResult

_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "disambiguation" / "weights.json"

#: Feature keys that mark a *nominal predicate* reading (a noun/adj wearing a copular or
#: person ending: "güzelim" = I am beautiful). Mirrors the analyzer's ``n_pred`` so a
#: homograph's finite-verb reading stays ranked above its noun+copula reading.
_PREDICATE_KEYS = (tags.PERSON, tags.COPULA, tags.EVIDENTIAL, tags.MOOD)

#: Floor so a score never reaches zero (keeps the confidence normalization well-defined).
_EPS = 1e-6

_WEIGHTS_CACHE: dict[str, Any] | None = None


def load_weights(path: Path | None = None) -> dict[str, Any]:
    """Load (and cache) the scoring/context weights from JSON.

    The packaged file is loaded once and cached. An explicit ``path`` bypasses the cache
    (used by tests), so callers can supply alternate weights without global state.
    """
    global _WEIGHTS_CACHE
    if path is not None:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    if _WEIGHTS_CACHE is None:
        _WEIGHTS_CACHE = json.loads(_WEIGHTS_PATH.read_text(encoding="utf-8"))
    return _WEIGHTS_CACHE


def _n_predicate(r: AnalysisResult) -> int:
    """1 if ``r`` is a nominal predicate reading (see :data:`_PREDICATE_KEYS`), else 0."""
    if r.pos == tags.VERB:
        return 0
    return 1 if any(k in r.features for k in _PREDICATE_KEYS) else 0


def score_candidate(r: AnalysisResult, weights: dict[str, Any] | None = None) -> float:
    """Return the **heuristic** plausibility score of one analysis (higher = more likely).

    Pure and deterministic: depends only on ``r``'s own fields, never on context. The value
    is a hand-tuned coarse prior, *not* a learned probability — it exists to rank candidates
    and to feed the sentence layer's bounded confidence, nothing more. Always ``> 0``.
    """
    w = weights if weights is not None else load_weights()
    src_rank: dict[str, float] = w["source_rank"]
    pos_prior: dict[str, float] = w["pos_prior"]

    score = src_rank.get(r.source, src_rank.get(tags.SOURCE_GUESS, 0.0))
    score += pos_prior.get(r.pos, w["default_pos_prior"])
    score -= w["morpheme_penalty"] * len(r.morphemes)
    score -= w["derivation_penalty"] * len(r.features.get(tags.DERIVATION, ()))
    score -= w["nominal_predicate_penalty"] * _n_predicate(r)
    score += w["root_length_bonus"] * len(r.lemma)
    # Extension point: coarse per-lemma frequency, if a lexicon ever supplies one. No entry
    # carries a "frequency" feature today, so this term is a documented no-op (weight 0.0).
    freq = r.features.get("frequency")
    if isinstance(freq, (int, float)):
        score += w.get("frequency_weight", 0.0) * freq

    return max(score, _EPS)


def rank_candidates(
    candidates: list[AnalysisResult], weights: dict[str, Any] | None = None
) -> list[AnalysisResult]:
    """Return ``candidates`` ordered by descending :func:`score_candidate` (best first).

    A **stable** sort: candidates with an equal score keep their incoming (analyzer) order,
    so this never contradicts the core ranking on a tie. Returns a new list; the inputs are
    not mutated. This is a word-level scoring utility — the sentence layer
    (:class:`~ilmek.disambiguation.disambiguator.Disambiguator`) adds context on top.
    """
    w = weights if weights is not None else load_weights()
    return sorted(candidates, key=lambda r: -score_candidate(r, w))
