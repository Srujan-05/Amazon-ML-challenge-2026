"""Turning candidate pairs into feature matrices.

Two consumers with opposite memory profiles:

* **Training** needs ~10M pairs, which fits comfortably (10M x 49 float32 = 2 GB).
* **Inference** needs ~87M pairs (1.73M test entities x K=50), which at 17 GB does
  not fit alongside everything else on a 36 GB machine.

So the unit of work is a *chunk of entities*, yielded as a generator. Training
concatenates the chunks; inference scores and discards each one. Chunks never
split an entity, because the decision layer scores an entity's candidates
together and several features are relative to the entity's best candidate.

IDF tables are country-scoped and cached to disk: computing them means tokenizing
a country's full comparable corpus, which is wasted work on every re-run.
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, FeatureExtractor, PairContext, build_idf
from .io import Record, read_records

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    """One batch of candidate pairs with everything needed to score or train on it."""

    X: np.ndarray                  # (n_pairs, n_features) float32
    s1_ids: np.ndarray             # (n_pairs,) object -- the entity each pair belongs to
    cand_ids: np.ndarray           # (n_pairs,) object
    y: np.ndarray | None = None    # (n_pairs,) float32, when labels are available
    country: str = ""

    def __len__(self) -> int:
        return self.X.shape[0]


def _idf_cache_path(cfg, split: str, country: str, kind: str) -> Path:
    safe = "".join(ch if ch.isalnum() else "_" for ch in country)
    return cfg.paths.artifacts / "idf" / f"{split}_{safe}_{kind}.parquet"


def country_idf(cfg, split: str, country: str, kind: str, force: bool = False) -> dict[str, float]:
    """IDF over one country's comparable corpus, cached to parquet.

    Rarity is local: ``chennai`` is unremarkable in the India partition and
    absent from the US one, so a global table would mis-weight both.
    """
    import polars as pl

    path = _idf_cache_path(cfg, split, country, kind)
    if path.is_file() and not force:
        frame = pl.read_parquet(path)
        return dict(zip(frame["token"].to_list(), frame["idf"].to_list(), strict=True))

    records = [
        record
        for key in ("S2", "S3")
        for record in read_records(cfg.paths.source_file(split, cfg["sources"][key]))
        if record.country == country
    ]
    table = build_idf(records, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"token": list(table.keys()), "idf": list(table.values())}
    ).write_parquet(path)
    log.info("built %s idf for %s/%s: %d tokens", kind, split, country, len(table))
    return table


def _load_needed_records(cfg, split: str, needed: set[str]) -> dict[str, Record]:
    """Stream the three source files, keeping only the records we will touch."""
    found: dict[str, Record] = {}
    for key in ("S1", "S2", "S3"):
        for record in read_records(cfg.paths.source_file(split, cfg["sources"][key])):
            if record.entity_id in needed:
                found[record.entity_id] = record
    return found


def iter_chunks(
    cfg,
    split: str,
    entity_filter: set[str] | None = None,
    labels: dict[str, set[str]] | None = None,
    chunk_entities: int = 20_000,
    countries: list[str] | None = None,
) -> Iterator[Chunk]:
    """Yield feature chunks for a split, one country at a time.

    ``labels`` supplies ground truth; when given, ``Chunk.y`` is populated. A
    candidate absent from an entity's label set is a negative -- which is correct,
    because Source 1 is deduplicated and its match list is exhaustive.
    """
    import polars as pl

    candidate_dir = cfg.paths.artifacts / f"candidates_{split}"
    files = sorted(candidate_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no candidates for {split!r}; run `ber block` first")

    for path in files:
        country = path.stem
        if countries and country not in countries:
            continue
        frame = pl.read_parquet(path)
        if entity_filter is not None:
            frame = frame.filter(pl.col("s1_id").is_in(list(entity_filter)))
        if frame.height == 0:
            continue

        # Contention is measured over the whole country partition, not the chunk:
        # it counts how many S1 entities compete for a record, and the exclusivity
        # constraint that makes it meaningful is global.
        contention = collections.Counter(frame["cand_id"].to_list())

        extractor = FeatureExtractor(
            name_idf=country_idf(cfg, split, country, "name"),
            addr_idf=country_idf(cfg, split, country, "address"),
        )

        frame = frame.sort("s1_id")
        entity_ids = frame["s1_id"].unique(maintain_order=True).to_list()
        log.info("country %s: %d entities, %d pairs", country, len(entity_ids), frame.height)

        for start in range(0, len(entity_ids), chunk_entities):
            batch_ids = entity_ids[start : start + chunk_entities]
            block = frame.filter(pl.col("s1_id").is_in(batch_ids))
            needed = set(block["s1_id"].to_list()) | set(block["cand_id"].to_list())
            records = _load_needed_records(cfg, split, needed)
            chunk = _build_chunk(block, records, extractor, contention, labels, country)
            if len(chunk):
                yield chunk
            extractor.clear()


def _build_chunk(block, records, extractor, contention, labels, country) -> Chunk:
    """Featurize one entity batch."""
    rows: list[np.ndarray] = []
    s1_out: list[str] = []
    cand_out: list[str] = []
    y_out: list[float] = []

    grouped = block.group_by("s1_id", maintain_order=True)
    for (s1_id,), group in grouped:
        s1_record = records.get(s1_id)
        if s1_record is None:
            continue
        lex = group["lex_score"].to_numpy()
        dense = group["dense_score"].to_numpy()
        rrf = group["rrf"].to_numpy()
        context = PairContext(
            n_candidates=group.height,
            best_lex=float(lex.max()) if lex.size else 0.0,
            best_dense=float(dense.max()) if dense.size else 0.0,
            best_rrf=float(rrf.max()) if rrf.size else 0.0,
            contention=contention,
        )
        truth = labels.get(s1_id, set()) if labels is not None else None
        for rank, row in enumerate(group.iter_rows(named=True)):
            cand = records.get(row["cand_id"])
            if cand is None:
                continue
            rows.append(
                extractor.pair(
                    s1_record, cand,
                    lex_score=row["lex_score"], lex_rank=row["lex_rank"],
                    dense_score=row["dense_score"], dense_rank=row["dense_rank"],
                    rrf=row["rrf"], rank_in_entity=rank, context=context,
                )
            )
            s1_out.append(s1_id)
            cand_out.append(row["cand_id"])
            if truth is not None:
                y_out.append(1.0 if row["cand_id"] in truth else 0.0)

    if not rows:
        return Chunk(
            X=np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
            s1_ids=np.empty(0, dtype=object),
            cand_ids=np.empty(0, dtype=object),
            y=None if labels is None else np.empty(0, dtype=np.float32),
            country=country,
        )
    return Chunk(
        X=np.vstack(rows),
        s1_ids=np.asarray(s1_out, dtype=object),
        cand_ids=np.asarray(cand_out, dtype=object),
        y=np.asarray(y_out, dtype=np.float32) if labels is not None else None,
        country=country,
    )


def collect(chunks: Iterator[Chunk]) -> Chunk:
    """Concatenate chunks into one in-memory dataset (training only)."""
    parts = [c for c in chunks if len(c)]
    if not parts:
        raise RuntimeError("no feature rows produced")
    return Chunk(
        X=np.vstack([c.X for c in parts]),
        s1_ids=np.concatenate([c.s1_ids for c in parts]),
        cand_ids=np.concatenate([c.cand_ids for c in parts]),
        y=(
            np.concatenate([c.y for c in parts])
            if all(c.y is not None for c in parts)
            else None
        ),
    )


def entity_split(
    entity_ids: list[str], val_fraction: float, test_fraction: float, seed: int
) -> tuple[set[str], set[str], set[str]]:
    """Split **by entity**, never by pair.

    Splitting pairs would put a positive and its sibling negatives from the same
    entity on both sides, leaking the answer and inflating validation scores. It
    would also make the macro F_0.5 average meaningless, since that metric is
    defined per entity.
    """
    rng = np.random.default_rng(seed)
    order = np.asarray(entity_ids, dtype=object)
    rng.shuffle(order)
    n = len(order)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    return (
        set(order[n_test + n_val :].tolist()),
        set(order[n_test : n_test + n_val].tolist()),
        set(order[:n_test].tolist()),
    )
