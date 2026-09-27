"""FAISS backend, one index per country.

Index choice is a memory decision at this scale. All vectors are L2-normalized,
so inner product equals cosine and ``METRIC_INNER_PRODUCT`` is the right metric.

``hnsw``
    Best recall, but stores full float32 vectors: the India comparable shard
    (4.72M x 512) needs ~9.7 GB plus ~1.2 GB of graph.
``ivfpq``
    Product-quantized to ``m`` bytes per vector -- the same shard drops to
    ~0.3 GB, at some recall cost.
``auto`` (default)
    Picks ``hnsw`` when the estimate fits ``max_index_bytes``, else ``ivfpq``.
    Recall loss here is cheap because the dense channel is only one of four
    candidate sources, and the lexical channels already reach a ~99.4% ceiling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .store import SearchResult, StagedVectors, collection_shard_sets

log = logging.getLogger(__name__)

# Leave headroom on a 36 GB machine for the lexical indices and feature buffers.
DEFAULT_MAX_INDEX_BYTES = 8 * 1024**3


class FaissStore:
    """Country-sharded FAISS indices persisted under ``vectors/faiss/<collection>``."""

    def __init__(self, cfg, split: str, collection: str):
        self.cfg = cfg
        self.split = split
        self.collection = collection
        self.dim = cfg["embedding"]["dim"]
        self.settings = cfg["vector_store"]
        self.directory = cfg.paths.vectors / "faiss" / f"{split}_{collection}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.staged = StagedVectors(cfg.paths.vectors, f"{split}_{collection}", self.dim)
        self._indices: dict[str, object] = {}
        self._ids: dict[str, np.ndarray] = {}

    # -- layout ------------------------------------------------------------ #

    def _index_path(self, country: str) -> Path:
        from .store import _slug

        return self.directory / f"{_slug(country)}.faiss"

    def countries(self) -> list[str]:
        return self.staged.countries()

    def size(self, country: str) -> int:
        return self.staged.counts().get(country, 0)

    # -- index construction ------------------------------------------------ #

    def _choose_index_type(self, n: int) -> str:
        requested = self.settings.get("index_type", "auto")
        if requested != "auto":
            return requested
        budget = int(self.settings.get("max_index_bytes", DEFAULT_MAX_INDEX_BYTES))
        m = int(self.settings["hnsw"]["M"])
        estimate = n * self.dim * 4 + n * m * 2 * 4
        if estimate <= budget:
            return "hnsw"
        log.info(
            "hnsw would need ~%.1f GB for %d vectors (budget %.1f GB); using ivfpq",
            estimate / 1024**3, n, budget / 1024**3,
        )
        return "ivfpq"

    def _new_index(self, kind: str, n: int):
        import faiss

        if kind == "hnsw":
            hnsw = self.settings["hnsw"]
            index = faiss.IndexHNSWFlat(self.dim, int(hnsw["M"]), faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = int(hnsw["ef_construction"])
            index.hnsw.efSearch = int(hnsw["ef_search"])
            return index
        if kind == "ivfpq":
            ivf = self.settings["ivfpq"]
            m = int(ivf["m"])
            if self.dim % m:
                raise ValueError(f"ivfpq m={m} must divide dim={self.dim}")
            # nlist above ~n/39 leaves too few training points per centroid.
            nlist = max(1, min(int(ivf["nlist"]), n // 39 or 1))
            quantizer = faiss.IndexFlatIP(self.dim)
            return faiss.IndexIVFPQ(
                quantizer, self.dim, nlist, m, int(ivf["nbits"]), faiss.METRIC_INNER_PRODUCT
            )
        raise ValueError(f"unknown index_type {kind!r}")

    def stage(self, force: bool = False) -> dict[str, int]:
        """Group the embedding shards by country (idempotent)."""
        shard_sets = collection_shard_sets(self.cfg, self.split, self.collection)
        return self.staged.stage(shard_sets, force=force)

    def build(self, countries: list[str] | None = None, force: bool = False) -> None:
        """Build and persist one index per country, skipping ones already on disk."""
        import faiss

        self.stage()
        targets = countries or self.countries()
        for country in targets:
            path = self._index_path(country)
            if path.is_file() and not force:
                log.info("%s/%s index present, skipping", self.collection, country)
                continue
            vectors, _ids = self.staged.load(country)
            n = vectors.shape[0]
            kind = self._choose_index_type(n)
            index = self._new_index(kind, n)
            log.info("building %s %s index for %s (%d vectors)", kind, self.collection, country, n)

            if not index.is_trained:
                sample_size = min(n, 200_000)
                step = max(1, n // sample_size)
                sample = np.ascontiguousarray(vectors[::step][:sample_size], dtype=np.float32)
                index.train(sample)

            # Add in blocks so the float32 upcast never materializes the whole shard.
            block = 100_000
            for start in range(0, n, block):
                chunk = np.ascontiguousarray(vectors[start : start + block], dtype=np.float32)
                index.add(chunk)
            faiss.write_index(index, str(path))
            (self.directory / f"{_slugged(country)}.meta.json").write_text(
                json.dumps({"index_type": kind, "vectors": n, "dim": self.dim}, indent=2),
                encoding="utf-8",
            )
            log.info("wrote %s (%d vectors)", path.name, index.ntotal)

    # -- search ------------------------------------------------------------ #

    def _load(self, country: str):
        if country not in self._indices:
            import faiss

            path = self._index_path(country)
            if not path.is_file():
                raise FileNotFoundError(f"no index for {country!r}; run `ber index` first")
            index = faiss.read_index(str(path))
            if hasattr(index, "hnsw"):
                index.hnsw.efSearch = int(self.settings["hnsw"]["ef_search"])
            if hasattr(index, "nprobe"):
                index.nprobe = int(self.settings["ivfpq"].get("nprobe", 32))
            self._indices[country] = index
            _vectors, ids = self.staged.load(country)
            self._ids[country] = ids
        return self._indices[country], self._ids[country]

    def search(self, country: str, queries: np.ndarray, k: int) -> SearchResult:
        """Top-``k`` neighbours for each query row, within one country partition."""
        index, ids = self._load(country)
        block = np.ascontiguousarray(queries, dtype=np.float32)
        k = min(k, index.ntotal)
        if k == 0:
            empty = np.empty((block.shape[0], 0), dtype=ids.dtype)
            return SearchResult(empty, np.empty((block.shape[0], 0), dtype=np.float32))
        scores, indices = index.search(block, k)
        # FAISS pads short result rows with -1; map those to an empty id.
        safe = np.where(indices >= 0, indices, 0)
        found = ids[safe]
        found = np.where(indices >= 0, found, "")
        return SearchResult(found, scores.astype(np.float32))


def _slugged(country: str) -> str:
    from .store import _slug

    return _slug(country)
