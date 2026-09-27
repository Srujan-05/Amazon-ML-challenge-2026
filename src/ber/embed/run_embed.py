"""The full embedding pass over a split.

24.2M records across both splits, so this stage dominates the project's compute.
Three properties matter:

* **Resumable.** Shard *i* always covers the same record range, so completed
  shards are skipped by fast-forwarding the reader. An interrupted 8-hour run
  resumes rather than restarts.
* **Pipelined.** Several batches are in flight at once, so tokenization and HTTP
  overlap with the server's GPU work.
* **Order-preserving.** Vectors are reassembled in record order, keeping them
  aligned with the entity IDs written alongside them.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..io import Record, read_records
from ..normalize import embedding_text
from .backends import EmbeddingBackend, build_backend
from .instruction import format_for_source
from .shards import ShardSet

log = logging.getLogger(__name__)


@dataclass
class EmbedStats:
    """Throughput accounting for one run, used by the benchmark gate too."""

    embedded: int = 0
    skipped: int = 0
    seconds: float = 0.0
    per_source: dict[str, int] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.embedded / self.seconds if self.seconds > 0 else 0.0

    def project(self, total_records: int) -> float:
        """Hours needed for ``total_records`` at the observed rate."""
        return total_records / self.rate / 3600 if self.rate > 0 else float("inf")

    def as_dict(self) -> dict:
        return {
            "embedded": self.embedded,
            "skipped": self.skipped,
            "seconds": round(self.seconds, 2),
            "records_per_second": round(self.rate, 1),
            "per_source": self.per_source,
        }


def _chunks(iterable: Iterator[Record], size: int) -> Iterator[list[Record]]:
    """Group a record stream into fixed-size lists."""
    iterator = iter(iterable)
    while True:
        block = list(itertools.islice(iterator, size))
        if not block:
            return
        yield block


async def embed_batches(
    backend: EmbeddingBackend,
    texts: list[str],
    batch_size: int,
    concurrency: int,
) -> np.ndarray:
    """Embed ``texts`` with bounded concurrency, preserving input order."""
    semaphore = asyncio.Semaphore(concurrency)

    async def run(offset: int, chunk: list[str]) -> tuple[int, np.ndarray]:
        async with semaphore:
            return offset, await backend.embed(chunk)

    tasks = [
        run(offset, texts[offset : offset + batch_size])
        for offset in range(0, len(texts), batch_size)
    ]
    out = np.empty((len(texts), backend.dim), dtype=np.float32)
    for offset, vectors in await asyncio.gather(*tasks):
        out[offset : offset + vectors.shape[0]] = vectors
    return out


def build_texts(records: list[Record], instruction: str) -> list[str]:
    """Normalized, instruction-wrapped text for each record.

    S1 records get the instruction (they are the query side); S2/S3 records are
    embedded as plain documents, matching how the model was trained.
    """
    return [
        format_for_source(
            embedding_text(record.name, record.address, record.country),
            record.source,
            instruction,
        )
        for record in records
    ]


async def embed_source(
    cfg,
    split: str,
    source: str,
    backend: EmbeddingBackend,
    stats: EmbedStats,
    limit: int | None = None,
) -> None:
    """Embed one source file into its shard directory."""
    embedding = cfg["embedding"]
    paths = cfg.paths
    shards = ShardSet(
        root=paths.vectors,
        split=split,
        source=source,
        dim=embedding["dim"],
        shard_size=embedding["shard_size"],
    )
    shards.validate_dim()

    path = paths.source_file(split, source)
    if not path.is_file():
        raise FileNotFoundError(f"missing source file {path}")

    instruction = cfg.instruction
    stream = read_records(path)
    if limit is not None:
        stream = itertools.islice(stream, limit)

    written = 0
    for index, block in enumerate(_chunks(stream, embedding["shard_size"])):
        if shards.is_complete(index):
            stats.skipped += len(block)
            continue
        started = time.perf_counter()
        texts = build_texts(block, instruction)
        vectors = await embed_batches(
            backend, texts, embedding["batch_size"], embedding["concurrency"]
        )
        shards.write(
            index,
            vectors,
            [record.entity_id for record in block],
            [record.country for record in block],
        )
        elapsed = time.perf_counter() - started
        stats.embedded += len(block)
        stats.seconds += elapsed
        written += len(block)
        log.info(
            "%s/%s shard %05d: %d records in %.1fs (%.0f rec/s)",
            split, source, index, len(block), elapsed, len(block) / max(elapsed, 1e-9),
        )
    stats.per_source[f"{split}/{source}"] = written


async def embed_split(
    cfg,
    split: str,
    sources: list[str] | None = None,
    limit: int | None = None,
) -> EmbedStats:
    """Embed every configured source of ``split``."""
    names = sources or list(cfg["sources"].values())
    backend = build_backend(cfg)
    stats = EmbedStats()
    try:
        log.info("backend: %s", await backend.health())
        for source in names:
            await embed_source(cfg, split, source, backend, stats, limit=limit)
    finally:
        await backend.aclose()
    return stats


def run(cfg, split: str, sources: list[str] | None = None, limit: int | None = None) -> EmbedStats:
    """Blocking entry point used by the CLI."""
    cfg.paths.ensure()
    return asyncio.run(embed_split(cfg, split, sources=sources, limit=limit))


def count_records(path: Path) -> int:
    """Data-row count of a source TSV, for progress and projection."""
    with open(path, "rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)
