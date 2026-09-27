"""On-disk cache of the featurized training splits.

Featurizing is expensive -- ~21,500 pairs/second, so ~8 minutes for 10M pairs -- and
every model needs the identical matrix. Writing it once means the four training
subprocesses share it via memory-mapped ``.npy`` files instead of each recomputing
it, and retraining a single model costs nothing extra.

Split membership is stored as boolean masks over the pair rows rather than as three
separate matrices, so the features are written exactly once. The masks come from an
entity-level split, which is what prevents an entity's pairs from straddling two
splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..dataset import Chunk


class FeatureCache:
    """Reads and writes the cached pair features under ``artifacts/features``."""

    SPLITS = ("train", "val", "test")

    def __init__(self, cfg):
        self.directory = cfg.paths.artifacts / "features"

    @property
    def meta_path(self) -> Path:
        return self.directory / "meta.json"

    def exists(self) -> bool:
        return self.meta_path.is_file() and all(
            (self.directory / name).is_file()
            for name in ("X.npy", "y.npy", "s1.npy", "cand.npy")
        )

    def write(self, data: Chunk, masks: dict[str, np.ndarray]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self.directory / "X.npy", data.X)
        np.save(self.directory / "y.npy", data.y)
        np.save(self.directory / "s1.npy", data.s1_ids, allow_pickle=True)
        np.save(self.directory / "cand.npy", data.cand_ids, allow_pickle=True)
        for name, mask in masks.items():
            np.save(self.directory / f"mask_{name}.npy", mask)
        self.meta_path.write_text(
            json.dumps(
                {
                    "pairs": int(data.X.shape[0]),
                    "features": int(data.X.shape[1]),
                    "positive_rate": round(float(data.y.mean()), 6),
                    "splits": {k: int(v.sum()) for k, v in masks.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def meta(self) -> dict:
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def chunk(self, split: str) -> Chunk:
        """Load one split, memory-mapping the feature matrix."""
        X = np.load(self.directory / "X.npy", mmap_mode="r")
        y = np.load(self.directory / "y.npy", mmap_mode="r")
        s1 = np.load(self.directory / "s1.npy", allow_pickle=True)
        cand = np.load(self.directory / "cand.npy", allow_pickle=True)
        mask = np.load(self.directory / f"mask_{split}.npy")
        # Fancy indexing a memmap materializes only the selected rows.
        return Chunk(
            X=np.ascontiguousarray(X[mask]),
            y=np.ascontiguousarray(y[mask]),
            s1_ids=s1[mask],
            cand_ids=cand[mask],
        )
