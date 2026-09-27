"""Model interface, registry, and evaluation.

Every model is trained and persisted independently, as ``CLAUDE.md`` asks, so any
one of them can be swapped into inference without retraining the others. They all
consume the same 49-dimensional pair feature vector and emit a probability per
pair; the decision layer is shared.

Models are scored on **macro F_0.5 first** and AUC-ROC second. AUC measures pair
ranking, which is not the task: the task is choosing a *set* per entity, where
predicting nothing for a singleton is worth a full 1.0 and one false merge costs
twice what a miss does. A model can rank pairs beautifully and still lose on
F_0.5 if its probabilities do not threshold cleanly.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..dataset import Chunk
from ..features import FEATURE_NAMES

log = logging.getLogger(__name__)

#: Populated only for modules that have actually been imported. The authoritative
#: name -> class mapping lives in :mod:`ber.models.registry`, which resolves lazily
#: so torch and LightGBM never load into the same process.
REGISTRY: dict[str, type[PairModel]] = {}


def register(cls: type[PairModel]) -> type[PairModel]:
    """Record a model class under its ``name`` when its module is imported."""
    REGISTRY[cls.name] = cls
    return cls


@dataclass
class ModelStats:
    """Everything recorded about one trained model."""

    name: str
    macro_fbeta: float = 0.0
    threshold: float = 0.5
    precision: float = 0.0
    recall: float = 0.0
    auc_roc: float = 0.0
    average_precision: float = 0.0
    singleton_accuracy: float = 0.0
    train_seconds: float = 0.0
    predict_seconds: float = 0.0
    n_train_pairs: int = 0
    n_eval_entities: int = 0
    feature_importance: dict[str, float] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class PairModel(ABC):
    """A binary classifier over candidate-pair features."""

    name: str = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self.stats = ModelStats(name=self.name)

    @abstractmethod
    def fit(self, train: Chunk, val: Chunk | None = None) -> None:
        """Train on ``train``, optionally using ``val`` for early stopping."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Probabilities in [0, 1], one per row."""

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path, cfg) -> PairModel: ...

    def importance(self) -> dict[str, float]:
        """Feature importances, when the model exposes them."""
        return {}

    # -- shared helpers ---------------------------------------------------- #

    def artifact_dir(self) -> Path:
        path = self.cfg.paths.artifacts / "models" / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_stats(self) -> Path:
        path = self.artifact_dir() / "stats.json"
        path.write_text(json.dumps(self.stats.as_dict(), indent=2), encoding="utf-8")
        return path

    def fit_timed(self, train: Chunk, val: Chunk | None = None) -> None:
        started = time.perf_counter()
        self.fit(train, val)
        self.stats.train_seconds = round(time.perf_counter() - started, 2)
        self.stats.n_train_pairs = len(train)
        self.stats.feature_importance = self.importance()
        log.info("%s trained on %d pairs in %.1fs", self.name, len(train), self.stats.train_seconds)

    def predict_timed(self, X: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        out = self.predict(X)
        self.stats.predict_seconds = round(time.perf_counter() - started, 2)
        return out


def standardize_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and std, with zero-variance columns left untouched."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def evaluate(
    model: PairModel,
    chunk: Chunk,
    truth: dict[str, set[str]],
    cfg,
    entities: set[str] | None = None,
) -> ModelStats:
    """Score a trained model: tune the threshold, then record every metric."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    from ..decide import decide, tune_threshold

    probs = model.predict_timed(chunk.X)
    eval_entities = sorted(entities) if entities else sorted(set(chunk.s1_ids.tolist()))

    threshold, best = tune_threshold(
        probs, chunk.s1_ids, chunk.cand_ids, truth, cfg, entities=eval_entities
    )
    predictions = decide(probs, chunk.s1_ids, chunk.cand_ids, threshold, cfg)

    from ..eval import score_predictions

    score = score_predictions(
        predictions, truth, float(cfg["evaluation"]["beta"]), entities=eval_entities
    )
    stats = model.stats
    stats.macro_fbeta = round(best, 6)
    stats.threshold = round(float(threshold), 4)
    stats.precision = round(score.micro_precision, 6)
    stats.recall = round(score.micro_recall, 6)
    stats.singleton_accuracy = round(score.singleton_accuracy, 6)
    stats.n_eval_entities = score.n_entities
    if chunk.y is not None and len(set(chunk.y.tolist())) > 1:
        stats.auc_roc = round(float(roc_auc_score(chunk.y, probs)), 6)
        stats.average_precision = round(float(average_precision_score(chunk.y, probs)), 6)
    return stats


def feature_importance_dict(values: np.ndarray) -> dict[str, float]:
    """Pair importance values with feature names, largest first."""
    total = float(values.sum()) or 1.0
    pairs = sorted(
        ((name, float(v) / total) for name, v in zip(FEATURE_NAMES, values, strict=True)),
        key=lambda kv: -kv[1],
    )
    return {name: round(value, 6) for name, value in pairs}
