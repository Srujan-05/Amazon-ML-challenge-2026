"""Shared training machinery for the neural models.

The distinguishing requirement is **entity-grouped batching**. The soft-F_0.5
surrogate is computed per Source-1 entity and then averaged, so every candidate of
an entity must land in the same batch -- a batch of random pairs would make the
per-entity precision and recall terms meaningless. Batches are therefore measured
in entities, not pairs, and entity order is shuffled between epochs.

Device selection follows ``training.device`` (``auto`` -> MPS -> CUDA -> CPU), which
is what ``CLAUDE.md`` asks for. Note the practical scale: these networks see ~10M
rows of 49 features, which is minutes of work. The embedding pass, not this, is
where the project's compute actually goes.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path

import numpy as np

from ..dataset import Chunk
from ..device import pick_device
from ..loss import build_loss
from .base import PairModel, standardize_stats

log = logging.getLogger(__name__)


def group_slices(s1_ids: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Sort pairs by entity and return the sort order plus per-entity spans."""
    order = np.argsort(s1_ids, kind="stable")
    sorted_ids = s1_ids[order]
    boundaries = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [len(sorted_ids)]])
    return order, list(zip(starts.tolist(), stops.tolist(), strict=True))


class TorchPairModel(PairModel):
    """Base for the neural pair classifiers."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.net = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.device = pick_device(cfg["training"]["device"])

    @property
    def settings(self) -> dict:
        return self.cfg["models"][self.name]

    @abstractmethod
    def build_net(self, n_features: int):
        """Construct the network for ``n_features`` inputs."""

    # -- training ---------------------------------------------------------- #

    def fit(self, train: Chunk, val: Chunk | None = None) -> None:
        import torch

        settings = self.settings
        self.mean, self.std = standardize_stats(train.X)
        self.net = self.build_net(train.X.shape[1]).to(self.device)
        loss_fn = build_loss(self.cfg)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=float(settings["lr"]))

        order, spans = group_slices(train.s1_ids)
        X = torch.from_numpy(self._scale(train.X[order]))
        y = torch.from_numpy(train.y[order].astype(np.float32))
        batch_entities = int(settings["batch_entities"])
        rng = np.random.default_rng(int(self.cfg["training"]["random_seed"]))

        best_state, best_loss = None, float("inf")
        for epoch in range(int(settings["epochs"])):
            self.net.train()
            shuffled = rng.permutation(len(spans))
            total, steps = 0.0, 0
            for start in range(0, len(shuffled), batch_entities):
                picked = [spans[i] for i in shuffled[start : start + batch_entities]]
                rows, groups = [], []
                for group_id, (lo, hi) in enumerate(picked):
                    rows.append(np.arange(lo, hi))
                    groups.append(np.full(hi - lo, group_id))
                if not rows:
                    continue
                index = torch.from_numpy(np.concatenate(rows))
                group_index = torch.from_numpy(np.concatenate(groups)).to(self.device)
                xb = X[index].to(self.device)
                yb = y[index].to(self.device)

                logits = self.net(xb, group_index, len(picked))
                loss = loss_fn(logits, yb, group_index, len(picked))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                steps += 1

            train_loss = total / max(steps, 1)
            message = f"{self.name} epoch {epoch + 1}: train loss {train_loss:.5f}"
            if val is not None and len(val):
                val_loss = self._validation_loss(val, loss_fn)
                message += f", val loss {val_loss:.5f}"
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            log.info(message)

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.stats.extra["best_val_loss"] = round(best_loss, 6) if best_state else None

    def _validation_loss(self, val: Chunk, loss_fn) -> float:
        import torch

        order, spans = group_slices(val.s1_ids)
        X = torch.from_numpy(self._scale(val.X[order]))
        y = torch.from_numpy(val.y[order].astype(np.float32))
        batch_entities = int(self.settings["batch_entities"])
        self.net.eval()
        total, steps = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(spans), batch_entities):
                picked = spans[start : start + batch_entities]
                rows, groups = [], []
                for group_id, (lo, hi) in enumerate(picked):
                    rows.append(np.arange(lo, hi))
                    groups.append(np.full(hi - lo, group_id))
                index = torch.from_numpy(np.concatenate(rows))
                group_index = torch.from_numpy(np.concatenate(groups)).to(self.device)
                xb = X[index].to(self.device)
                yb = y[index].to(self.device)
                logits = self.net(xb, group_index, len(picked))
                total += float(loss_fn(logits, yb, group_index, len(picked)))
                steps += 1
        return total / max(steps, 1)

    # -- inference --------------------------------------------------------- #

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def predict(self, X: np.ndarray, batch_rows: int = 500_000) -> np.ndarray:
        """Probabilities for arbitrary rows.

        Entity context is unavailable here (rows may arrive in any grouping), so
        each row is treated as its own group. For the plain MLP that is exact; for
        the entity-attention model, :meth:`predict_grouped` is the accurate path.
        """
        import torch

        self.net.eval()
        out = np.empty(X.shape[0], dtype=np.float32)
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_rows):
                block = self._scale(X[start : start + batch_rows])
                xb = torch.from_numpy(block).to(self.device)
                groups = torch.arange(xb.shape[0], device=self.device)
                logits = self.net(xb, groups, xb.shape[0])
                out[start : start + block.shape[0]] = torch.sigmoid(logits).cpu().numpy()
        return out

    def predict_grouped(self, X: np.ndarray, s1_ids: np.ndarray) -> np.ndarray:
        """Probabilities with true entity grouping, matching training conditions."""
        import torch

        self.net.eval()
        order, spans = group_slices(s1_ids)
        scaled = torch.from_numpy(self._scale(X[order]))
        out = np.empty(X.shape[0], dtype=np.float32)
        batch_entities = int(self.settings["batch_entities"])
        with torch.no_grad():
            for start in range(0, len(spans), batch_entities):
                picked = spans[start : start + batch_entities]
                rows, groups = [], []
                for group_id, (lo, hi) in enumerate(picked):
                    rows.append(np.arange(lo, hi))
                    groups.append(np.full(hi - lo, group_id))
                index = np.concatenate(rows)
                group_index = torch.from_numpy(np.concatenate(groups)).to(self.device)
                xb = scaled[torch.from_numpy(index)].to(self.device)
                logits = self.net(xb, group_index, len(picked))
                out[index] = torch.sigmoid(logits).cpu().numpy()
        # Undo the sort so probabilities line up with the caller's row order.
        restored = np.empty_like(out)
        restored[order] = out
        return restored

    # -- persistence ------------------------------------------------------- #

    def save(self, path: Path) -> None:
        import torch

        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "mean": self.mean,
                "std": self.std,
                "n_features": int(self.mean.shape[0]),
            },
            path / "model.pt",
        )

    @classmethod
    def load(cls, path: Path, cfg):
        import torch

        instance = cls(cfg)
        blob = torch.load(path / "model.pt", map_location=instance.device, weights_only=False)
        instance.mean = blob["mean"]
        instance.std = blob["std"]
        instance.net = instance.build_net(blob["n_features"]).to(instance.device)
        instance.net.load_state_dict(blob["state_dict"])
        instance.net.eval()
        return instance
