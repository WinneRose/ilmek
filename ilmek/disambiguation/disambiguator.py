"""Sentence-context disambiguation: a separate, optional re-ranking layer.

After the context-free analyzer produces every candidate per token, :class:`Disambiguator`
makes a single deterministic left-to-right pass and re-ranks each token's candidates using
simple, declarative context rules (loaded from ``data/disambiguation/weights.json``):

* **R1 gen->poss** — a token after a genitive-marked word prefers a possessive reading
  (``onun evi`` -> ``ev+poss3sg``);
* **R2 acc-before-verb** — a content word before a finite verb, *not* itself preceded by a
  genitive, prefers the accusative direct-object reading (``evi gördüm`` -> ``ev+acc``);
* **R3 num+noun** — a numeral right before a noun prefers the bare-numeral reading
  (``Yüz kişi`` -> ``yüz=NUM``);
* **R4 final finite verb** — the last analyzable token prefers a finite-verb reading;
* **R5 adj-before-noun** — a word right before a noun prefers an adjective reading
  (``hasta çocuk`` -> ``hasta=ADJ``).

**Architecture.** This module depends only on :mod:`ilmek.core` and the sibling
:mod:`.scoring`; :mod:`ilmek.core` / :mod:`ilmek.morphology` never import it (no cycle).
Only :class:`~ilmek.pipeline.pipeline.Pipeline` wires it in, via a lazy import.

**How the pick stays safe.** The re-rank sorts each token's candidates by
``(-context_bonus, original_analyzer_index)``. When no rule fires every bonus is ``0`` and
the order is *identical to the analyzer's*, so turning this layer on by default never
silently flips a token that has no context evidence — a rule can only promote a candidate on
*explicit* evidence. Alternatives are never dropped: they stay in ``chosen.analyses`` in the
re-ranked order (the contract forbids forcing/removing candidates — we only re-order the ones
the analyzer already found).

**Confidence is HEURISTIC.** ``chosen.confidence`` is ``reliability(source) * share`` where
``share`` is the chosen candidate's sum-normalized :func:`~.scoring.score_candidate` (plus
its context bonus) among the token's candidates, and ``reliability`` discounts less-trusted
sources (a lexicon reading reaches ``1.0``, an unknown-root ``guess`` is capped below it).
It is a bounded rule-derived signal in ``[0, 1]``, **not** a learned probability, and is set
*only here* — the context-free ``analyze`` still returns ``confidence=None``.
"""

from __future__ import annotations

from typing import Any

from ..core.document import AnalysisResult
from .scoring import _EPS, load_weights, score_candidate


def _match_condition(cond: dict[str, Any], r: AnalysisResult) -> bool:
    """Whether analysis ``r`` satisfies a declarative feature-match ``cond`` (all clauses).

    Recognised clauses (each optional; all present ones must hold):
    ``pos`` (exact), ``pos_in`` (membership), ``feature_eq`` / ``feature_ne`` (per-key
    equality / inequality on ``r.features``), ``feature_present`` (keys that must exist),
    ``feature_any`` (at least one of the keys exists), ``feature_absent`` (keys that must
    not exist). ``feature_ne`` pairs with ``feature_present`` when the key must exist *and*
    differ from a value (so a verb, lacking the key, does not vacuously match).
    """
    if "pos" in cond and r.pos != cond["pos"]:
        return False
    if "pos_in" in cond and r.pos not in cond["pos_in"]:
        return False
    for key, value in cond.get("feature_eq", {}).items():
        if r.features.get(key) != value:
            return False
    for key, value in cond.get("feature_ne", {}).items():
        if r.features.get(key) == value:
            return False
    for key in cond.get("feature_present", ()):
        if key not in r.features:
            return False
    any_keys = cond.get("feature_any")
    if any_keys and not any(key in r.features for key in any_keys):
        return False
    for key in cond.get("feature_absent", ()):
        if key in r.features:
            return False
    return True


def _rule_fires(
    rule: dict[str, Any],
    left_chosen: AnalysisResult | None,
    right_best: AnalysisResult | None,
    is_final: bool,
) -> bool:
    """Whether ``rule``'s context trigger holds at the current token.

    Left context is the *already-chosen* analysis of the previous token; right context is
    the analyzer's primary reading of the next token (base-best). ``left``/``right`` require
    a matching neighbor (a missing neighbor fails the trigger); ``left_absent`` is a guard
    that fails the rule when the left neighbor *does* match; ``is_final`` is positional.
    """
    if "left" in rule:
        if left_chosen is None or not _match_condition(rule["left"], left_chosen):
            return False
    if "left_absent" in rule:
        if left_chosen is not None and _match_condition(rule["left_absent"], left_chosen):
            return False
    if "right" in rule:
        if right_best is None or not _match_condition(rule["right"], right_best):
            return False
    if "is_final" in rule and bool(rule["is_final"]) != is_final:
        return False
    return True


class Disambiguator:
    """Sentence-context candidate re-ranker (rule + coarse-frequency, no ML)."""

    def __init__(self, weights: dict[str, Any] | None = None):
        self.weights = weights if weights is not None else load_weights()
        self.rules: list[dict[str, Any]] = list(self.weights.get("context_rules", ()))
        self._reliability: dict[str, float] = self.weights.get("source_reliability", {})

    def rerank(
        self, candidate_lists: list[list[AnalysisResult] | None]
    ) -> list[AnalysisResult | None]:
        """Re-rank each token's candidates in one left-to-right context pass.

        ``candidate_lists`` is parallel to a document's tokens: a per-token candidate list
        (analyzer order, best first) or ``None`` for a non-analyzable token (punctuation).
        Returns the chosen analysis per token (or ``None``); the chosen carries the other
        candidates in ``.analyses`` (never dropped) and a heuristic ``.confidence``.

        The chosen objects are assumed to be *fresh* per call (the native backend builds new
        :class:`AnalysisResult` objects on every ``analyze``), so setting ``.confidence`` /
        reordering ``.analyses`` here never leaks into cached word-level output.
        """
        n = len(candidate_lists)
        chosen: list[AnalysisResult | None] = [None] * n
        #: analyzer primary per token, used as the (order-independent) right context.
        base_best = [cands[0] if cands else None for cands in candidate_lists]
        last_idx = max((i for i, cands in enumerate(candidate_lists) if cands), default=-1)

        for i, cands in enumerate(candidate_lists):
            if not cands:
                continue
            left_chosen = chosen[i - 1] if i > 0 else None
            right_best = base_best[i + 1] if i + 1 < n else None
            is_final = i == last_idx

            bonuses = [0.0] * len(cands)
            for rule in self.rules:
                if not _rule_fires(rule, left_chosen, right_best, is_final):
                    continue
                prefer, amount = rule["prefer"], rule["bonus"]
                for j, cand in enumerate(cands):
                    if _match_condition(prefer, cand):
                        bonuses[j] += amount

            # Stable argmax: highest bonus wins; ties keep the analyzer's original order, so
            # a token with no firing rule is left exactly as the analyzer ranked it.
            order = sorted(range(len(cands)), key=lambda j: (-bonuses[j], j))
            best_j = order[0]
            best = cands[best_j]
            best.analyses = [cands[j] for j in order[1:]]
            best.confidence = self._confidence(cands, bonuses, best_j)
            chosen[i] = best

        return chosen

    def _confidence(self, cands: list[AnalysisResult], bonuses: list[float], best_j: int) -> float:
        """Bounded HEURISTIC confidence for the chosen candidate (see the module docstring).

        ``reliability(source) * share``, where ``share`` is the chosen candidate's
        score-plus-bonus divided by the sum over all the token's candidates. A single
        candidate gives ``share = 1`` so the value collapses to the source's reliability
        (a lone lexicon reading -> ``1.0``; a lone unknown-root guess stays below it). Never
        a learned probability; clamped to ``[0, 1]``.
        """
        totals = [
            max(score_candidate(cand, self.weights) + bonuses[j], _EPS)
            for j, cand in enumerate(cands)
        ]
        total_sum = sum(totals)
        share = totals[best_j] / total_sum if total_sum > 0 else 1.0
        reliability = self._reliability.get(cands[best_j].source, 0.5)
        value = reliability * share
        return min(1.0, max(0.0, value))
