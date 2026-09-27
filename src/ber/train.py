"""Training orchestration.

Two stages, deliberately separated:

1. **Featurize once, cache to disk.** Every model needs the identical matrix, and
   building it runs at ~21,500 pairs/second, so ~8 minutes for 10M pairs. Writing it
   once and memory-mapping it is strictly better than recomputing it four times.

2. **Train each model in its own subprocess.** Required, not tidiness: PyTorch,
   scikit-learn and FAISS each bundle a ``libomp.dylib`` and LightGBM loads another,
   and LightGBM segfaults outright if torch was imported first. One model per process
   keeps the OpenMP runtimes apart, caps peak memory at one model's worth, and lets a
   single model be retrained without touching the others.

Sampling and splitting are both **by entity**, because the metric is a per-entity
average -- an entity is the atomic unit of training signal and of evaluation alike.
Taking ~200k of the 2.2M training entities yields ~10M pairs at K=50: enough to learn
the nuances without featurizing all 110M.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys

import numpy as np

from .blocking.candidates import load_candidates
from .dataset import collect, entity_split, iter_chunks
from .decide import contention_report
from .io import read_id_lists
from .models.features_cache import FeatureCache
from .models.registry import DEFAULT_MODELS

log = logging.getLogger(__name__)


def sample_entities(cfg, split: str = "train") -> list[str]:
    """Entities to train on: those with candidates, sampled to the pair budget."""
    frame = load_candidates(cfg, split).select("s1_id").unique().collect()
    available = frame["s1_id"].to_list()

    settings = cfg["training"]
    wanted = int(settings["sample_entities"])
    budget = int(settings["target_pairs"])
    k = int(cfg["blocking"]["k"])
    # Whichever binds first: the entity count or the pair budget.
    wanted = min(wanted, max(budget // max(k, 1), 1))

    if len(available) <= wanted:
        return sorted(available)
    rng = np.random.default_rng(int(settings["random_seed"]))
    picked = rng.choice(np.asarray(available, dtype=object), size=wanted, replace=False)
    return sorted(picked.tolist())


def materialize(cfg, split: str = "train", force: bool = False) -> FeatureCache:
    """Featurize the sampled entities and cache the matrix plus split masks."""
    cache = FeatureCache(cfg)
    if cache.exists() and not force:
        log.info("using cached features: %s", cache.meta())
        return cache

    truth = read_id_lists(cfg.paths.ground_truth())
    entities = sample_entities(cfg, split)
    log.info("featurizing %d sampled entities", len(entities))

    data = collect(iter_chunks(cfg, split, entity_filter=set(entities), labels=truth))
    log.info("featurized %d pairs (%.2f%% positive)", len(data), 100.0 * float(data.y.mean()))
    log.info("candidate contention: %s", contention_report(data.s1_ids, data.cand_ids))

    settings = cfg["training"]
    train_ids, val_ids, test_ids = entity_split(
        entities,
        float(settings["val_fraction"]),
        float(settings["test_fraction"]),
        int(settings["random_seed"]),
    )
    log.info("entity split: train=%d val=%d test=%d", len(train_ids), len(val_ids), len(test_ids))

    def mask(keep: set[str]) -> np.ndarray:
        return np.fromiter(
            (e in keep for e in data.s1_ids), dtype=bool, count=len(data.s1_ids)
        )

    cache.write(data, {"train": mask(train_ids), "val": mask(val_ids), "test": mask(test_ids)})
    log.info("cached features to %s", cache.directory)
    return cache


def fit_models(cfg, models: list[str], overrides: list[str] | None = None) -> list[str]:
    """Run one training subprocess per model; returns the ones that succeeded."""
    succeeded = []
    for name in models:
        command = [sys.executable, "-m", "ber.models.fit_one", "--model", name]
        for item in overrides or []:
            command += ["--set", item]
        log.info("training %s in a separate process", name)
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            succeeded.append(name)
        else:
            log.error(
                "%s training failed with exit code %d (139 = segfault, usually an "
                "OpenMP clash: check that only one model is loaded per process)",
                name, result.returncode,
            )
    return succeeded


def run(
    cfg,
    models: list[str] | None = None,
    split: str = "train",
    overrides: list[str] | None = None,
    force_features: bool = False,
) -> list[dict]:
    """Featurize, train each model, and write the comparison."""
    cfg.paths.ensure()
    names = models or DEFAULT_MODELS
    materialize(cfg, split, force=force_features)

    trained = fit_models(cfg, names, overrides)
    if not trained:
        raise RuntimeError("no model trained successfully")

    from .models.compare import format_table, write_comparison

    path, rows = write_comparison(cfg, trained)
    print(format_table(rows))
    print(f"\nwrote {path}")

    best = rows[0]
    holdout = best.get("extra", {}).get("holdout", {})
    if holdout:
        print(f"\nheld-out test (best = {best['name']}): {json.dumps(holdout, indent=2)}")
    return rows
