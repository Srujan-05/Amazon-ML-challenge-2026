"""ChromaDB backend, behind the same interface as FAISS.

Kept so the store is swappable -- notably for the AWS move, where a managed
vector service is easier to justify than a local FAISS file. It is **not** the
local default: Chroma keeps a SQLite metadata row per vector, and at ~4.7M
vectors in the India partition that layer dominates both ingest time and memory
on a 36 GB machine. FAISS reads the same staged vectors and builds in minutes.

Install with the optional extra:  ``uv sync --extra chroma``
"""

from __future__ import annotations

import logging

import numpy as np

from .store import SearchResult, StagedVectors, _slug, collection_shard_sets

log = logging.getLogger(__name__)

#: Chroma ingests far more slowly than FAISS; keep batches modest.
ADD_BATCH = 5_000


class ChromaStore:
    """One Chroma collection per country, persisted locally."""

    def __init__(self, cfg, split: str, collection: str):
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ChromaDB backend requires the optional extra: uv sync --extra chroma"
            ) from exc

        self.cfg = cfg
        self.split = split
        self.collection = collection
        self.dim = cfg["embedding"]["dim"]
        self.directory = cfg.paths.vectors / "chroma"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.directory))
        self.staged = StagedVectors(cfg.paths.vectors, f"{split}_{collection}", self.dim)

    def _name(self, country: str) -> str:
        return f"{self.split}_{self.collection}_{_slug(country)}"

    def countries(self) -> list[str]:
        return self.staged.countries()

    def size(self, country: str) -> int:
        return self.staged.counts().get(country, 0)

    def stage(self, force: bool = False) -> dict[str, int]:
        shard_sets = collection_shard_sets(self.cfg, self.split, self.collection)
        return self.staged.stage(shard_sets, force=force)

    def build(self, countries: list[str] | None = None, force: bool = False) -> None:
        """Create one collection per country and ingest the staged vectors."""
        self.stage()
        for country in countries or self.countries():
            name = self._name(country)
            existing = {c.name for c in self.client.list_collections()}
            if name in existing:
                if not force:
                    log.info("chroma collection %s exists, skipping", name)
                    continue
                self.client.delete_collection(name)
            # Cosine, to match the L2-normalized vectors the embedder produces.
            handle = self.client.create_collection(
                name=name, metadata={"hnsw:space": "cosine", "country": country}
            )
            vectors, ids = self.staged.load(country)
            total = vectors.shape[0]
            log.info("ingesting %d vectors into %s", total, name)
            for start in range(0, total, ADD_BATCH):
                stop = min(start + ADD_BATCH, total)
                handle.add(
                    ids=[str(v) for v in ids[start:stop]],
                    embeddings=np.asarray(vectors[start:stop], dtype=np.float32).tolist(),
                )

    def search(self, country: str, queries: np.ndarray, k: int) -> SearchResult:
        """Top-``k`` neighbours within one country collection."""
        handle = self.client.get_collection(self._name(country))
        result = handle.query(
            query_embeddings=np.ascontiguousarray(queries, dtype=np.float32).tolist(),
            n_results=k,
            include=["distances"],
        )
        rows = result["ids"]
        dists = result["distances"]
        width = max((len(r) for r in rows), default=0)
        ids = np.full((len(rows), width), "", dtype=object)
        scores = np.zeros((len(rows), width), dtype=np.float32)
        for i, (row_ids, row_dist) in enumerate(zip(rows, dists, strict=True)):
            ids[i, : len(row_ids)] = row_ids
            # Chroma returns cosine *distance*; the pipeline works in similarity.
            scores[i, : len(row_dist)] = 1.0 - np.asarray(row_dist, dtype=np.float32)
        return SearchResult(ids, scores)
