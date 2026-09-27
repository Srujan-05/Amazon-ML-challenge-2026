"""Train exactly one model, in its own process.

Invoked as ``python -m ber.models.fit_one --model lightgbm``. Running one model per
process is what keeps LightGBM and PyTorch's competing OpenMP runtimes apart (see
:mod:`ber.models.registry`), and it has two further benefits: peak memory is one
model's worth rather than four, and a single model can be retrained without
re-featurizing or disturbing the others.

Features are read from the cache that ``ber.train`` materializes, memory-mapped so
a 10M-pair matrix is not copied into each subprocess.
"""

from __future__ import annotations

import argparse
import logging
import sys

from ..config import load_config
from ..dataset import Chunk
from ..io import read_id_lists
from .base import evaluate
from .features_cache import FeatureCache
from .registry import build

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    from ..cli import _parse_overrides

    cfg = load_config(args.config, _parse_overrides(args.overrides))
    cache = FeatureCache(cfg)
    if not cache.exists():
        print("no cached features; run `ber train` first", file=sys.stderr)
        return 1

    train_chunk = cache.chunk("train")
    val_chunk = cache.chunk("val")
    test_chunk = cache.chunk("test")
    truth = read_id_lists(cfg.paths.ground_truth())

    model = build(args.model, cfg)
    model.fit_timed(train_chunk, val_chunk)

    val_entities = set(val_chunk.s1_ids.tolist())
    evaluate(model, val_chunk, truth, cfg, entities=val_entities)

    # Score the untouched split at the tuned threshold, so the headline number
    # is not the one the threshold was fitted on.
    model.stats.extra["holdout"] = _holdout(model, test_chunk, truth, cfg)

    directory = model.artifact_dir()
    model.save(directory)
    path = model.write_stats()
    log.info(
        "%s: val macro F_0.5 %.5f at threshold %.2f | holdout %.5f -> %s",
        args.model, model.stats.macro_fbeta, model.stats.threshold,
        model.stats.extra["holdout"].get("macro_fbeta", 0.0), path,
    )
    return 0


def _holdout(model, chunk: Chunk, truth: dict[str, set[str]], cfg) -> dict:
    """Evaluate on the held-out split at the already-chosen threshold."""
    if not len(chunk):
        return {}
    from ..decide import decide
    from ..eval import score_predictions

    probs = model.predict(chunk.X)
    entities = sorted(set(chunk.s1_ids.tolist()))
    predictions = decide(probs, chunk.s1_ids, chunk.cand_ids, model.stats.threshold, cfg)
    score = score_predictions(
        predictions, truth, float(cfg["evaluation"]["beta"]), entities=entities
    )
    return score.as_dict()


if __name__ == "__main__":
    raise SystemExit(main())
