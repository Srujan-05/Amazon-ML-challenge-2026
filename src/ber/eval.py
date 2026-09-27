"""Macro F_beta scoring, exactly as the problem statement defines it.

    F_0.5 = (1.25 x P x R) / (0.25 x P + R)

computed per Source-1 entity and then averaged over *all* entities. Singletons
are included in that average: a correct empty prediction scores 1.0, and any
prediction on a true singleton scores 0.0. This is the primary model-selection
metric -- not AUC -- because it is what the leaderboard computes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

Prediction = Mapping[str, set[str]]


def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    """F_beta from precision and recall, 0.0 when both are 0."""
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 0.0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def entity_fbeta(predicted: set[str], truth: set[str], beta: float = 0.5) -> float:
    """Score one Source-1 entity.

    Both empty is a correct singleton and scores 1.0. Exactly one empty scores
    0.0 -- predicting matches for a true singleton is a false merge, and missing
    every true match earns no credit.
    """
    if not truth and not predicted:
        return 1.0
    if not truth or not predicted:
        return 0.0
    tp = len(predicted & truth)
    if tp == 0:
        return 0.0
    return fbeta(tp / len(predicted), tp / len(truth), beta)


@dataclass
class Score:
    """Aggregate report over an evaluation set."""

    macro_fbeta: float
    micro_precision: float
    micro_recall: float
    n_entities: int
    n_singletons: int
    singleton_accuracy: float
    mean_predicted: float
    mean_truth: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "macro_fbeta": round(self.macro_fbeta, 6),
            "micro_precision": round(self.micro_precision, 6),
            "micro_recall": round(self.micro_recall, 6),
            "n_entities": self.n_entities,
            "n_singletons": self.n_singletons,
            "singleton_accuracy": round(self.singleton_accuracy, 6),
            "mean_predicted": round(self.mean_predicted, 4),
            "mean_truth": round(self.mean_truth, 4),
        }


def score_predictions(
    predictions: Prediction,
    truth: Prediction,
    beta: float = 0.5,
    entities: Iterable[str] | None = None,
) -> Score:
    """Macro-average F_beta over ``entities`` (default: every entity in ``truth``).

    Every entity in the evaluation set counts, whether or not ``predictions``
    contains a row for it -- a missing row is an empty prediction, which is how
    the scorer treats it.
    """
    keys = list(entities) if entities is not None else list(truth.keys())
    if not keys:
        return Score(0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0)

    total = 0.0
    tp = fp = fn = 0
    singletons = singleton_hits = 0
    n_pred = n_truth = 0

    for key in keys:
        pred = predictions.get(key) or set()
        gold = truth.get(key) or set()
        total += entity_fbeta(pred, gold, beta)

        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        n_pred += len(pred)
        n_truth += len(gold)

        if not gold:
            singletons += 1
            if not pred:
                singleton_hits += 1

    n = len(keys)
    return Score(
        macro_fbeta=total / n,
        micro_precision=tp / (tp + fp) if (tp + fp) else 0.0,
        micro_recall=tp / (tp + fn) if (tp + fn) else 0.0,
        n_entities=n,
        n_singletons=singletons,
        singleton_accuracy=singleton_hits / singletons if singletons else 0.0,
        mean_predicted=n_pred / n,
        mean_truth=n_truth / n,
    )


def blocking_recall(candidates: Prediction, truth: Prediction) -> dict[str, float]:
    """Recall ceiling and reduction ratio for a candidate set.

    ``recall_ceiling`` is the best macro F_beta any downstream classifier could
    reach: it is the share of true pairs the blocking stage kept. Reported for
    the methodology write-up, and the number that bounds everything after it.
    """
    kept = total = 0
    per_entity = 0.0
    n_candidates = 0
    entities = list(truth.keys())
    for key in entities:
        gold = truth.get(key) or set()
        cand = candidates.get(key) or set()
        n_candidates += len(cand)
        kept += len(gold & cand)
        total += len(gold)
        per_entity += 1.0 if not gold else len(gold & cand) / len(gold)
    n = len(entities) or 1
    return {
        "pair_recall": kept / total if total else 0.0,
        "macro_entity_recall": per_entity / n,
        "candidates_per_entity": n_candidates / n,
        "true_pairs_kept": kept,
        "true_pairs_total": total,
    }
