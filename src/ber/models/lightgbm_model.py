"""LightGBM pair classifier.

The workhorse. Gradient-boosted trees suit this feature set: the features are
heterogeneous (bounded similarities, unbounded IDF weights, integer ranks, binary
flags) and interact non-linearly -- a high name similarity means something quite
different when the address also matches than when the address is missing. Trees
capture that without any scaling.

``scale_pos_weight`` below 1 down-weights positives, biasing the model toward
precision to match F_0.5. The decision threshold is then tuned on top, so this
parameter shapes the probability distribution rather than making the final call.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from ..dataset import Chunk
from .base import PairModel, feature_importance_dict, register


@register
class LightGBMModel(PairModel):
    name = "lightgbm"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = None

    def fit(self, train: Chunk, val: Chunk | None = None) -> None:
        import lightgbm as lgb

        params = dict(self.cfg["models"]["lightgbm"])
        self.model = lgb.LGBMClassifier(n_jobs=-1, verbose=-1, **params)
        kwargs = {}
        if val is not None and len(val):
            kwargs["eval_metric"] = "average_precision"
            kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
            # LightGBM 4.7 deprecated `eval_set` in favour of bare `eval_X`/`eval_y`
            # and the legacy path segfaults on this data, so prefer the new API and
            # fall back only where it does not exist.
            if "eval_X" in inspect.signature(self.model.fit).parameters:
                kwargs["eval_X"] = val.X
                kwargs["eval_y"] = val.y
            else:  # pragma: no cover - older lightgbm
                kwargs["eval_set"] = [(val.X, val.y)]
        self.model.fit(train.X, train.y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    def importance(self) -> dict[str, float]:
        return feature_importance_dict(self.model.feature_importances_)

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump(self.model, path / "model.joblib")

    @classmethod
    def load(cls, path: Path, cfg) -> LightGBMModel:
        import joblib

        instance = cls(cfg)
        instance.model = joblib.load(path / "model.joblib")
        return instance
