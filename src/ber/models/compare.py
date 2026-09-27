"""Train, evaluate and compare the four models.

Emits one stats JSON per model plus a combined comparison table, ranked by **macro
F_0.5** -- the leaderboard metric -- with precision, recall and AUC-ROC alongside.

The ranking is deliberately not by AUC. AUC asks whether pairs are ordered well;
the task asks whether the right *set* is chosen per entity, where a correctly empty
prediction earns a full 1.0 and a false merge costs double a miss. Those come apart
in practice, so the table reports both and ranks on the one that is scored.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .base import ModelStats, PairModel
from .registry import DEFAULT_MODELS, build, get_model_class  # noqa: F401

log = logging.getLogger(__name__)


def load_model(name: str, cfg) -> PairModel:
    """Load a previously trained model from its artifact directory.

    Imports only the requested model's module, so a process that loads a tree
    model never pulls in torch (and vice versa) -- see :mod:`ber.models.registry`.
    """
    directory = cfg.paths.artifacts / "models" / name
    if not directory.is_dir():
        raise FileNotFoundError(f"no trained {name!r} model at {directory}")
    model = get_model_class(name).load(directory, cfg)
    stats_path = directory / "stats.json"
    if stats_path.is_file():
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        model.stats = ModelStats(**payload)
    return model


def collect_stats(cfg, names: list[str] | None = None) -> list[dict]:
    """Read every model's stats JSON, best macro F_0.5 first."""
    root = cfg.paths.artifacts / "models"
    wanted = names or DEFAULT_MODELS
    rows = []
    for name in wanted:
        path = root / name / "stats.json"
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: -row.get("macro_fbeta", 0.0))
    return rows


def write_comparison(cfg, names: list[str] | None = None) -> tuple[Path, list[dict]]:
    """Write ``artifacts/model_comparison.json`` and return it with the rows."""
    rows = collect_stats(cfg, names)
    if not rows:
        raise FileNotFoundError("no model stats found; run `ber train` first")
    payload = {
        "ranked_by": "macro_fbeta",
        "beta": float(cfg["evaluation"]["beta"]),
        "best_model": rows[0]["name"],
        "models": rows,
    }
    path = cfg.paths.artifacts / "model_comparison.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path, rows


def format_table(rows: list[dict]) -> str:
    """Human-readable comparison table."""
    header = (
        f"{'model':<14} {'macroF0.5':>10} {'thresh':>7} {'precision':>10} "
        f"{'recall':>8} {'AUC-ROC':>8} {'AP':>8} {'singl.':>7} {'train_s':>8}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['name']:<14} {row.get('macro_fbeta', 0):>10.5f} "
            f"{row.get('threshold', 0):>7.2f} {row.get('precision', 0):>10.5f} "
            f"{row.get('recall', 0):>8.5f} {row.get('auc_roc', 0):>8.5f} "
            f"{row.get('average_precision', 0):>8.5f} "
            f"{row.get('singleton_accuracy', 0):>7.4f} {row.get('train_seconds', 0):>8.1f}"
        )
    return "\n".join(lines)


def best_model_name(cfg) -> str:
    rows = collect_stats(cfg)
    if not rows:
        raise FileNotFoundError("no model stats found; run `ber train` first")
    return rows[0]["name"]
