"""Inference: score the test candidates and write the submission files.

Streams chunks rather than materializing all 87M test pairs -- 1.73M entities at
K=50 would be ~17 GB of features alone, before any model overhead.

Exclusivity is resolved **globally**, after every chunk is scored, not per chunk: a
record contested by two entities may well have those entities in different chunks,
and resolving early would let both keep their claim.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from .dataset import iter_chunks
from .decide import contention_report, decide
from .io import read_records, write_id_lists
from .models.compare import load_model

log = logging.getLogger(__name__)


def run(cfg, model_name: str, split: str = "test", chunk_entities: int = 20_000) -> dict:
    """Score ``split`` with ``model_name`` and write both output TSVs."""
    cfg.paths.ensure()
    model = load_model(model_name, cfg)
    threshold = (
        float(cfg["decision"]["threshold"])
        if cfg["decision"]["threshold"] is not None
        else float(model.stats.threshold)
    )
    log.info("scoring %s with %s at threshold %.3f", split, model_name, threshold)

    all_probs: list[np.ndarray] = []
    all_s1: list[np.ndarray] = []
    all_cand: list[np.ndarray] = []
    scored = 0
    for chunk in iter_chunks(cfg, split, chunk_entities=chunk_entities):
        probs = (
            model.predict_grouped(chunk.X, chunk.s1_ids)
            if hasattr(model, "predict_grouped")
            else model.predict(chunk.X)
        )
        all_probs.append(probs.astype(np.float32))
        all_s1.append(chunk.s1_ids)
        all_cand.append(chunk.cand_ids)
        scored += len(chunk)
        log.info("scored %d pairs (%s)", scored, chunk.country)

    if not all_probs:
        raise RuntimeError(f"no candidate pairs scored for split {split!r}")

    probs = np.concatenate(all_probs)
    s1_ids = np.concatenate(all_s1)
    cand_ids = np.concatenate(all_cand)
    log.info("contention: %s", contention_report(s1_ids, cand_ids))

    predictions = decide(probs, s1_ids, cand_ids, threshold, cfg)

    # Every S1 entity needs exactly one row, whether or not it matched anything.
    all_ids = [
        r.entity_id for r in read_records(cfg.paths.source_file(split, cfg["sources"]["S1"]))
    ]
    matching_path = cfg.paths.output / "matching_results.tsv"
    rows_written = write_id_lists(
        matching_path,
        ((eid, predictions.get(eid, set())) for eid in all_ids),
        header="matched_entity_ids",
    )

    from .blocking.candidates import write_candidate_tsv

    candidate_rows = write_candidate_tsv(cfg, split)

    matched = sum(len(v) for v in predictions.values())
    summary = {
        "model": model_name,
        "threshold": round(threshold, 4),
        "split": split,
        "entities": rows_written,
        "candidate_rows": candidate_rows,
        "pairs_scored": int(probs.shape[0]),
        "matches_predicted": matched,
        "mean_matches_per_entity": round(matched / max(rows_written, 1), 4),
        "empty_predictions": rows_written - len(predictions),
        "matching_results": str(matching_path),
    }
    path = cfg.paths.artifacts / "predict_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote %s and candidate_pairs.tsv", matching_path)
    return summary
