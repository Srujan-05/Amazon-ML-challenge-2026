"""Turning pair probabilities into the final match sets.

Three steps, each motivated by a property of the metric or the data:

1. **Threshold tuning against macro F_0.5 directly.** The usual 0.5 cutoff is
   wrong here. F_0.5 weights precision twice as heavily as recall, so the optimal
   operating point sits well above 0.5 -- measured at 0.90 on a validation split.
   The threshold is chosen by scanning the grid and evaluating the real metric,
   not a proxy.

2. **Exclusivity resolution.** Each S2/S3 record matches at most one S1 entity --
   verified exactly: 7,638,365 ground-truth match IDs, 7,638,365 mentions, every
   count 1. A per-pair classifier cannot enforce this, so when several entities
   claim one record, the strongest claim keeps it and the rest are dropped. Every
   claim removed is a false merge removed, which F_0.5 punishes at double weight.

3. **Singletons come free.** An entity with nothing above the threshold gets an
   empty prediction, which scores a full 1.0 when it is genuinely a singleton
   (5.58% of entities). No special casing is needed -- just the discipline of not
   forcing a match where the model does not want one.
"""

from __future__ import annotations

import collections
import logging

import numpy as np

from .eval import score_predictions

log = logging.getLogger(__name__)


def resolve_exclusivity(
    probs: np.ndarray, s1_ids: np.ndarray, cand_ids: np.ndarray, selected: np.ndarray
) -> np.ndarray:
    """Drop all but the strongest claim on each contested comparable record.

    ``selected`` is a boolean mask of pairs above the threshold; the returned mask
    is a subset of it. Ties fall to the first occurrence, which is stable given a
    stable input order.
    """
    chosen = np.flatnonzero(selected)
    if chosen.size == 0:
        return selected
    # Sort by descending probability, then claim greedily: the first claim on a
    # record wins, which is the highest-scoring one.
    order = chosen[np.argsort(-probs[chosen], kind="stable")]
    winners = np.zeros(len(selected), dtype=bool)
    taken: set[str] = set()
    for index in order:
        candidate = cand_ids[index]
        if candidate in taken:
            continue
        taken.add(candidate)
        winners[index] = True
    return winners


def apply_source_cap(
    probs: np.ndarray,
    s1_ids: np.ndarray,
    cand_ids: np.ndarray,
    selected: np.ndarray,
    max_per_source: int,
) -> np.ndarray:
    """Keep at most ``max_per_source`` matches per source, per entity.

    The ground truth never assigns more than 5 S2 records or 6 S3 records to one
    entity, so a longer prediction is almost certainly over-merging. Off by
    default: it is a blunt instrument, and the threshold usually handles it.
    """
    chosen = np.flatnonzero(selected)
    if chosen.size == 0:
        return selected
    order = chosen[np.argsort(-probs[chosen], kind="stable")]
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    keep = np.zeros(len(selected), dtype=bool)
    for index in order:
        key = (s1_ids[index], cand_ids[index][:3])
        if counts[key] >= max_per_source:
            continue
        counts[key] += 1
        keep[index] = True
    return keep


def decide(
    probs: np.ndarray,
    s1_ids: np.ndarray,
    cand_ids: np.ndarray,
    threshold: float,
    cfg,
) -> dict[str, set[str]]:
    """Final ``{s1_id: {matched ids}}`` for one set of scored pairs."""
    settings = cfg["decision"]
    selected = probs >= threshold
    if bool(settings.get("enforce_exclusivity", True)):
        selected = resolve_exclusivity(probs, s1_ids, cand_ids, selected)
    cap = settings.get("max_per_source")
    if cap:
        selected = apply_source_cap(probs, s1_ids, cand_ids, selected, int(cap))

    out: dict[str, set[str]] = collections.defaultdict(set)
    for index in np.flatnonzero(selected):
        out[s1_ids[index]].add(cand_ids[index])
    return dict(out)


def tune_threshold(
    probs: np.ndarray,
    s1_ids: np.ndarray,
    cand_ids: np.ndarray,
    truth: dict[str, set[str]],
    cfg,
    entities: list[str] | None = None,
) -> tuple[float, float]:
    """Scan the configured grid and return ``(threshold, macro F_beta)``.

    Evaluated on the real metric over the given entities -- including those with no
    candidates at all, which still count toward the macro average.
    """
    start, stop, steps = cfg["decision"]["threshold_grid"]
    beta = float(cfg["evaluation"]["beta"])
    grid = np.linspace(float(start), float(stop), int(steps))
    eval_entities = entities if entities is not None else sorted(truth)

    best_threshold, best_score = float(grid[0]), -1.0
    for threshold in grid:
        predictions = decide(probs, s1_ids, cand_ids, float(threshold), cfg)
        score = score_predictions(predictions, truth, beta, entities=eval_entities).macro_fbeta
        if score > best_score:
            best_threshold, best_score = float(threshold), score
    log.info("tuned threshold %.3f -> macro F_%.1f %.5f", best_threshold, beta, best_score)
    return best_threshold, best_score


def contention_report(s1_ids: np.ndarray, cand_ids: np.ndarray) -> dict[str, float]:
    """How much the exclusivity constraint has to do.

    Reports how many comparable records appear in more than one entity's candidate
    list. On a small subset this is near zero and exclusivity resolution barely
    moves the score; at full scale each record sits in ~8-9 candidate lists, so it
    matters considerably more. Worth printing before drawing conclusions from an
    ablation run on sampled data.
    """
    counts = collections.Counter(cand_ids.tolist())
    contested = sum(1 for v in counts.values() if v > 1)
    return {
        "distinct_candidates": len(counts),
        "contested_candidates": contested,
        "contested_fraction": round(contested / max(len(counts), 1), 4),
        "max_claims_on_one_record": max(counts.values(), default=0),
        "mean_claims_per_record": round(sum(counts.values()) / max(len(counts), 1), 3),
    }
