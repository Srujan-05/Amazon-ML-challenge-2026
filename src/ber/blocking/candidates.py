"""Candidate generation: union the channels, cap at K, persist.

Produces ``candidate_pairs.tsv`` -- the exact set the matching model scores, which
is what the problem statement asks for -- plus a parquet of per-pair channel
evidence that feature extraction and training consume.

Channels are fused with **Reciprocal Rank Fusion** rather than by adding scores.
Lexical scores are IDF-weighted cosines over a hashed token space and dense scores
are embedding cosines; they are not on a comparable scale, and calibrating them
would mean fitting yet another model. RRF only needs the *ordering* within each
channel, which both provide reliably.

Work is done one country at a time. That bounds memory -- the India comparable
partition alone is 4.72M records -- and costs nothing, because a true match never
crosses a country boundary.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import Record, read_records, write_id_lists
from ..vectors.store import BASE, COMPARABLE, StagedVectors, build_store
from .dense import DenseChannel
from .lexical import LexicalIndex

log = logging.getLogger(__name__)

#: RRF damping. 60 is the value from the original formulation and is insensitive
#: over roughly 20-100; it keeps rank 1 from dominating rank 2 outright.
RRF_K0 = 60.0

PAIR_SCHEMA = ["s1_id", "cand_id", "lex_score", "lex_rank", "dense_score", "dense_rank", "rrf"]


@dataclass
class BlockingReport:
    """Per-run diagnostics; also what the methodology write-up needs."""

    entities: int = 0
    candidates: int = 0
    seconds: float = 0.0
    per_country: dict[str, int] = None

    def as_dict(self) -> dict:
        return {
            "entities": self.entities,
            "candidate_pairs": self.candidates,
            "candidates_per_entity": round(self.candidates / max(self.entities, 1), 2),
            "seconds": round(self.seconds, 1),
            "per_country": self.per_country or {},
        }


def load_records_by_country(
    path: Path, countries: set[str] | None = None
) -> dict[str, list[Record]]:
    """Group a source file's records by country, optionally restricted."""
    grouped: dict[str, list[Record]] = collections.defaultdict(list)
    for record in read_records(path):
        if countries is None or record.country in countries:
            grouped[record.country].append(record)
    return dict(grouped)


def fuse(
    lex_ids: np.ndarray,
    lex_scores: np.ndarray,
    dense_ids: np.ndarray,
    dense_scores: np.ndarray,
    k: int,
    rrf_k0: float = RRF_K0,
) -> list[tuple[str, float, int, float, int, float]]:
    """Union one entity's channel results and keep the top ``k`` by RRF.

    Returns ``(cand_id, lex_score, lex_rank, dense_score, dense_rank, rrf)`` with
    a rank of ``-1`` where a channel did not retrieve that candidate -- the model
    gets to see *which* channel found a pair, which is itself informative.
    """
    evidence: dict[str, list[float]] = {}
    for rank, (cid, score) in enumerate(zip(lex_ids, lex_scores, strict=True)):
        if not cid:
            continue
        evidence.setdefault(cid, [0.0, -1.0, 0.0, -1.0])
        evidence[cid][0] = float(score)
        evidence[cid][1] = float(rank)
    for rank, (cid, score) in enumerate(zip(dense_ids, dense_scores, strict=True)):
        if not cid:
            continue
        evidence.setdefault(cid, [0.0, -1.0, 0.0, -1.0])
        evidence[cid][2] = float(score)
        evidence[cid][3] = float(rank)

    rows = []
    for cid, (ls, lr, ds, dr) in evidence.items():
        rrf = 0.0
        if lr >= 0:
            rrf += 1.0 / (rrf_k0 + lr + 1.0)
        if dr >= 0:
            rrf += 1.0 / (rrf_k0 + dr + 1.0)
        rows.append((cid, ls, int(lr), ds, int(dr), rrf))
    rows.sort(key=lambda row: -row[5])
    return rows[:k]


def generate(
    cfg,
    split: str,
    query_block: int = 2000,
    limit_entities: int | None = None,
) -> tuple[Path, BlockingReport]:
    """Run candidate generation for a split; returns the parquet path and a report.

    Iterates in the order of the staged base vectors, so the dense channel can
    reuse the S1 embeddings computed during the embedding pass rather than
    re-encoding 1.7M records.
    """
    import polars as pl

    started = time.perf_counter()
    paths = cfg.paths
    paths.ensure()
    blocking = cfg["blocking"]
    k = int(blocking["k"])
    lexical_cfg = blocking["lexical"]
    dense_cfg = blocking["dense_ann"]
    dense_enabled = bool(dense_cfg["enabled"])
    lex_enabled = bool(lexical_cfg["enabled"])
    lex_top_n = int(lexical_cfg["top_n"])
    min_digits = int(lexical_cfg["min_digits"])
    dense_top_n = int(dense_cfg["top_n"])
    rrf_k0 = float(blocking.get("rrf_k0", RRF_K0))
    if not (lex_enabled or dense_enabled):
        raise ValueError("at least one blocking channel must be enabled")

    s1_lookup = {
        r.entity_id: r for r in read_records(paths.source_file(split, cfg["sources"]["S1"]))
    }

    # With the dense channel on, iterate in staged-vector order so the S1
    # embeddings from the embedding pass can be reused directly -- re-encoding
    # 1.7M queries would otherwise dominate this stage. With it off, the pipeline
    # runs purely lexically and needs no embeddings at all, which is also how the
    # blocking sweep and the ablations run.
    base: StagedVectors | None = None
    comparable_store = None
    if dense_enabled:
        base = StagedVectors(paths.vectors, f"{split}_{BASE}", cfg["embedding"]["dim"])
        if not base.is_staged():
            build_store(cfg, split, BASE).stage()
            base = StagedVectors(paths.vectors, f"{split}_{BASE}", cfg["embedding"]["dim"])
        countries = base.countries()
        if not countries:
            raise RuntimeError(
                f"no staged base vectors for split {split!r}; run `ber embed` first, "
                "or set blocking.channels.dense_ann.enabled=false for a lexical-only run"
            )
        comparable_store = build_store(cfg, split, COMPARABLE)
        comparable_store.stage()
    else:
        countries = sorted({r.country for r in s1_lookup.values()})
        log.info("dense channel disabled: lexical-only run over countries %s", countries)
    comp_by_country: dict[str, list[Record]] = collections.defaultdict(list)
    for key in ("S2", "S3"):
        for country, records in load_records_by_country(
            paths.source_file(split, cfg["sources"][key]), set(countries)
        ).items():
            comp_by_country[country].extend(records)

    out_dir = paths.artifacts / f"candidates_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = BlockingReport(per_country={})

    for country in countries:
        comp_records = comp_by_country.get(country, [])
        if not comp_records:
            log.warning("no comparable records for country %r; skipping", country)
            continue
        log.info("country %s: %d comparable records", country, len(comp_records))
        index = (
            LexicalIndex.build(
                comp_records,
                min_digits=min_digits,
                max_df_ratio=float(lexical_cfg["max_df_ratio"]),
                min_df=1,
                weights=dict(lexical_cfg["channel_weights"]),
            )
            if lex_enabled
            else None
        )
        dense = None
        if comparable_store is not None:
            dense = DenseChannel(comparable_store, dense_top_n)

        if base is not None:
            vectors, s1_ids = base.load(country)
        else:
            vectors = None
            s1_ids = np.asarray(
                [i for i, r in s1_lookup.items() if r.country == country], dtype=object
            )
        if limit_entities is not None:
            s1_ids = s1_ids[:limit_entities]
            if vectors is not None:
                vectors = vectors[:limit_entities]

        rows: list[tuple] = []
        for start in range(0, len(s1_ids), query_block):
            stop = min(start + query_block, len(s1_ids))
            block_ids = s1_ids[start:stop]
            block_records = [s1_lookup[i] for i in block_ids]

            if index is not None:
                lex_idx, lex_scores = index.query(
                    block_records, top_n=lex_top_n, min_digits=min_digits
                )
            else:
                lex_idx = np.full((stop - start, 0), -1, dtype=np.int64)
                lex_scores = np.zeros((stop - start, 0), dtype=np.float32)
            if dense is not None and vectors is not None:
                found = dense.search_vectors(
                    country, np.asarray(vectors[start:stop], dtype=np.float32)
                )
                dense_ids, dense_scores = found.ids, found.scores
            else:
                dense_ids = np.full((stop - start, 0), "", dtype=object)
                dense_scores = np.zeros((stop - start, 0), dtype=np.float32)

            for row, s1_id in enumerate(block_ids):
                mask = lex_idx[row] >= 0
                lids = (
                    index.ids[lex_idx[row][mask]]
                    if index is not None
                    else np.empty(0, dtype=object)
                )
                lsc = lex_scores[row][mask]
                for cid, ls, lr, ds, dr, rrf in fuse(
                    lids, lsc, dense_ids[row], dense_scores[row], k, rrf_k0
                ):
                    rows.append((str(s1_id), str(cid), ls, lr, ds, dr, rrf))

            if stop % (query_block * 25) == 0:
                log.info("  %s: %d/%d entities", country, stop, len(s1_ids))

        frame = pl.DataFrame(rows, schema=PAIR_SCHEMA, orient="row")
        frame.write_parquet(out_dir / f"{country.replace('/', '_')}.parquet")
        report.entities += len(s1_ids)
        report.candidates += frame.height
        report.per_country[country] = frame.height
        log.info("country %s: %d candidate pairs", country, frame.height)

    report.seconds = time.perf_counter() - started
    return out_dir, report


def load_candidates(cfg, split: str):
    """All candidate pairs for a split as one lazy frame."""
    import polars as pl

    out_dir = cfg.paths.artifacts / f"candidates_{split}"
    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no candidates for {split!r}; run `ber block` first")
    return pl.concat([pl.scan_parquet(f) for f in files])


def candidate_map(cfg, split: str) -> dict[str, set[str]]:
    """``{s1_id: {candidate ids}}`` -- for recall reporting and the TSV."""
    frame = load_candidates(cfg, split).select(["s1_id", "cand_id"]).collect()
    grouped: dict[str, set[str]] = collections.defaultdict(set)
    for s1_id, cand_id in zip(
        frame["s1_id"].to_list(), frame["cand_id"].to_list(), strict=True
    ):
        grouped[s1_id].add(cand_id)
    return dict(grouped)


def write_candidate_tsv(cfg, split: str) -> int:
    """Write ``output/candidate_pairs.tsv``, one row per S1 entity.

    Every S1 entity gets a row even when blocking found nothing -- the submission
    rules require it, and an empty list is a valid declaration of no candidates.
    """
    grouped = candidate_map(cfg, split)
    all_ids = [
        r.entity_id for r in read_records(cfg.paths.source_file(split, cfg["sources"]["S1"]))
    ]
    path = cfg.paths.output / "candidate_pairs.tsv"
    return write_id_lists(
        path,
        ((eid, grouped.get(eid, set())) for eid in all_ids),
        header="candidate_entity_ids",
    )
