"""XGBoost pair classifier.

The second tree model. It differs from LightGBM in how it grows trees --
depth-wise with level-wise splits rather than leaf-wise -- which tends to
regularize differently on skewed feature distributions. Included because
comparing two genuinely different tree implementations is more informative than
comparing two configurations of one, and because their disagreements are a cheap
signal about which pairs are genuinely ambiguous.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..dataset import Chunk
from .base import PairModel, feature_importance_dict, register


@register
class XGBoostModel(PairModel):
    name = "xgboost"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = None

    def fit(self, train: Chunk, val: Chunk | None = None) -> None:
        import xgboost as xgb

        params = dict(self.cfg["models"]["xgboost"])
        kwargs = {}
        if val is not None and len(val):
            params["early_stopping_rounds"] = 50
            kwargs["eval_set"] = [(val.X, val.y)]
            kwargs["verbose"] = False
        self.model = xgb.XGBClassifier(
            n_jobs=-1, tree_method="hist", eval_metric="aucpr", **params
        )
        self.model.fit(train.X, train.y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    def importance(self) -> dict[str, float]:
        return feature_importance_dict(self.model.feature_importances_)

    def save(self, path: Path) -> None:
        self.model.save_model(str(path / "model.ubj"))

    @classmethod
    def load(cls, path: Path, cfg) -> XGBoostModel:
        import xgboost as xgb

        instance = cls(cfg)
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(str(path / "model.ubj"))
        return instance
