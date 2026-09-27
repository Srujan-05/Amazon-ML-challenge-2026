"""Vector store interface and country-sharded staging.

Two collections exist, as ``CLAUDE.md`` specifies:

``base_vectors``
    Source-1 records -- the side we search *from*.
``comparable_vectors``
    Source-2 and Source-3 records -- the side we search *into*.

Both are sharded by country. That is not an optimisation guess: ``country`` never
disagrees across a true match (0 of 13,836 sampled ground-truth pairs), so a
search that crosses a country boundary can only return false positives. Sharding
cuts the India search space from 10M vectors to 4.7M and keeps each index inside
a laptop's memory. Country is treated as an opaque label, so the test-only
``France`` partition appears automatically.

Staging exists because the shards are written in file order but the index needs
them grouped by country, and 4.7M x 512 float32 is ~9.7 GB -- too much to buffer
in RAM. So staging streams shards into one ``.npy`` per country on disk, which
the index then consumes in batches.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..embed.shards import ShardSet

log = logging.getLogger(__name__)

BASE = "base_vectors"
COMPARABLE = "comparable_vectors"


@dataclass(frozen=True)
class SearchResult:
    """Neighbours for a query block.

    ``ids`` holds entity IDs (``-1`` padding becomes an empty string) and
    ``scores`` the inner products, which equal cosine similarity because every
    vector is L2-normalized.
    """

    ids: np.ndarray       # (n_queries, k) dtype=object / <U
    scores: np.ndarray    # (n_queries, k) float32


@runtime_checkable
class VectorStore(Protocol):
    """What the pipeline needs from a vector backend."""

    def build(self, countries: list[str] | None = None) -> None: ...

    def search(self, country: str, queries: np.ndarray, k: int) -> SearchResult: ...

    def countries(self) -> list[str]: ...

    def size(self, country: str) -> int: ...


class StagedVectors:
    """Per-country ``.npy`` vector files plus aligned entity IDs.

    Written once from the embedding shards and reused by whichever index backend
    is configured, so switching FAISS <-> Chroma never re-reads 24.2M shards.
    """

    def __init__(self, root: Path, name: str, dim: int):
        self.directory = root / "staged" / name
        self.name = name
        self.dim = dim
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def _meta_path(self) -> Path:
        return self.directory / "staged.json"

    def vecs_path(self, country: str) -> Path:
        return self.directory / f"{_slug(country)}.vecs.npy"

    def ids_path(self, country: str) -> Path:
        return self.directory / f"{_slug(country)}.ids.parquet"

    def is_staged(self) -> bool:
        if not self._meta_path.is_file():
            return False
        meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
        return all(
            self.vecs_path(c).is_file() and self.ids_path(c).is_file()
            for c in meta.get("counts", {})
        )

    def counts(self) -> dict[str, int]:
        if not self._meta_path.is_file():
            return {}
        return json.loads(self._meta_path.read_text(encoding="utf-8")).get("counts", {})

    def countries(self) -> list[str]:
        return sorted(self.counts())

    def load(self, country: str) -> tuple[np.ndarray, np.ndarray]:
        """Memmapped vectors and the entity-ID array for one country."""
        import polars as pl

        vectors = np.load(self.vecs_path(country), mmap_mode="r")
        ids = pl.read_parquet(self.ids_path(country))["entity_id"].to_numpy()
        return vectors, ids

    def stage(self, shard_sets: list[ShardSet], force: bool = False) -> dict[str, int]:
        """Group embedding shards by country into one file per country.

        Two passes: count rows per country from the (small) parquet metadata, then
        fill pre-sized ``.npy`` files. Never holds more than one shard in memory.
        """
        if self.is_staged() and not force:
            log.info("%s already staged: %s", self.name, self.counts())
            return self.counts()

        counts: Counter[str] = Counter()
        for shard_set in shard_sets:
            for ref in shard_set.refs():
                meta = ref.load_meta()
                counts.update(meta["country"].to_list())
        if not counts:
            raise RuntimeError(
                f"no embedding shards found for {self.name}; run `ber embed` first"
            )
        log.info("staging %s: %s", self.name, dict(counts))

        writers: dict[str, np.ndarray] = {}
        cursors: dict[str, int] = {}
        id_buffers: dict[str, list[str]] = {}
        for country, total in counts.items():
            writers[country] = np.lib.format.open_memmap(
                self.vecs_path(country), mode="w+", dtype=np.float16, shape=(total, self.dim)
            )
            cursors[country] = 0
            id_buffers[country] = []

        for shard_set in shard_sets:
            for _ref, vectors, meta in shard_set.iter_shards():
                countries = meta["country"].to_numpy()
                ids = meta["entity_id"].to_numpy()
                block = np.asarray(vectors)
                for country in np.unique(countries):
                    mask = countries == country
                    rows = block[mask]
                    start = cursors[country]
                    writers[country][start : start + rows.shape[0]] = rows
                    cursors[country] = start + rows.shape[0]
                    id_buffers[country].extend(ids[mask].tolist())

        import polars as pl

        for country, writer in writers.items():
            writer.flush()
            del writer
            pl.DataFrame({"entity_id": id_buffers[country]}).write_parquet(
                self.ids_path(country)
            )
        self._meta_path.write_text(
            json.dumps({"dim": self.dim, "counts": dict(counts)}, indent=2), encoding="utf-8"
        )
        return dict(counts)


def _slug(country: str) -> str:
    """Filesystem-safe country label; countries are arbitrary strings.

    Shared by every backend so a collection keeps the same name whichever one
    wrote it.
    """
    return "".join(ch if ch.isalnum() else "_" for ch in country) or "unknown"


def collection_shard_sets(cfg, split: str, collection: str) -> list[ShardSet]:
    """The embedding shard directories that feed one collection."""
    embedding = cfg["embedding"]
    wanted = ["S1"] if collection == BASE else ["S2", "S3"]
    return [
        ShardSet(
            root=cfg.paths.vectors,
            split=split,
            source=cfg["sources"][key],
            dim=embedding["dim"],
            shard_size=embedding["shard_size"],
        )
        for key in wanted
    ]


def build_store(cfg, split: str, collection: str) -> VectorStore:
    """Construct the configured backend for one collection of one split."""
    backend = cfg["vector_store"]["backend"]
    if backend == "faiss":
        from .faiss_store import FaissStore

        return FaissStore(cfg, split, collection)
    if backend == "chroma":
        from .chroma_store import ChromaStore

        return ChromaStore(cfg, split, collection)
    raise ValueError(f"unknown vector_store backend {backend!r}")
