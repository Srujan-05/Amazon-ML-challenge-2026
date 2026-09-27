"""On-disk layout for embedding shards.

One directory per ``(split, source)``, holding fixed-size shards plus a manifest:

    vectors/train_source1/
        manifest.json
        shard_00000.vecs.npy      float16 (rows, dim)
        shard_00000.meta.parquet  entity_id, country
        ...

Shard *i* always covers records ``[i*shard_size, (i+1)*shard_size)`` of the source
file, which is what makes the embedding pass resumable: a shard listed in the
manifest is complete and can be skipped without reading it. Vectors are written
atomically, so an interrupted run never leaves a partial shard that a later stage
would mistake for finished work.

float16 halves the footprint at negligible cost for unit-norm vectors: all 24.2M
records at 512d come to ~24.8 GB instead of ~49.6 GB.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MANIFEST = "manifest.json"


@dataclass(frozen=True)
class ShardRef:
    """Locator for one shard on disk."""

    index: int
    rows: int
    directory: Path

    @property
    def vecs_path(self) -> Path:
        return self.directory / f"shard_{self.index:05d}.vecs.npy"

    @property
    def meta_path(self) -> Path:
        return self.directory / f"shard_{self.index:05d}.meta.parquet"

    def load_vectors(self, mmap: bool = True) -> np.ndarray:
        """Load vectors, memory-mapped by default so shards need not fit in RAM."""
        return np.load(self.vecs_path, mmap_mode="r" if mmap else None)

    def load_meta(self):
        import polars as pl

        return pl.read_parquet(self.meta_path)


class ShardSet:
    """Reader/writer for one ``(split, source)`` shard directory."""

    def __init__(self, root: Path, split: str, source: str, dim: int, shard_size: int):
        self.directory = root / f"{split}_{source}"
        self.split = split
        self.source = source
        self.dim = dim
        self.shard_size = shard_size
        self.directory.mkdir(parents=True, exist_ok=True)
        self._manifest = self._read_manifest()

    # -- manifest ---------------------------------------------------------- #

    def _read_manifest(self) -> dict:
        path = self.directory / MANIFEST
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {"dim": self.dim, "shard_size": self.shard_size, "shards": {}}

    def _write_manifest(self) -> None:
        path = self.directory / MANIFEST
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".manifest.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(self._manifest, handle, indent=2)
        Path(tmp).replace(path)

    def is_complete(self, index: int) -> bool:
        """True when shard ``index`` is recorded *and* its files are still present."""
        entry = self._manifest["shards"].get(str(index))
        if entry is None:
            return False
        ref = ShardRef(index, entry["rows"], self.directory)
        return ref.vecs_path.is_file() and ref.meta_path.is_file()

    @property
    def completed_rows(self) -> int:
        return sum(entry["rows"] for entry in self._manifest["shards"].values())

    def validate_dim(self) -> None:
        """Guard against mixing dimensions in one directory.

        Re-running after changing ``embedding.dim`` would otherwise silently
        produce a directory of shards that cannot be concatenated.
        """
        recorded = self._manifest.get("dim")
        if recorded is not None and recorded != self.dim:
            raise ValueError(
                f"{self.directory} holds {recorded}-dim vectors but config asks for "
                f"{self.dim}. Delete the directory or restore the previous dim."
            )

    # -- writing ----------------------------------------------------------- #

    def write(
        self, index: int, vectors: np.ndarray, ids: list[str], countries: list[str]
    ) -> ShardRef:
        """Write one shard atomically and record it in the manifest."""
        import polars as pl

        if vectors.shape[0] != len(ids) or len(ids) != len(countries):
            raise ValueError("vectors, ids and countries must be the same length")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")

        ref = ShardRef(index, vectors.shape[0], self.directory)
        _atomic_npy(ref.vecs_path, vectors.astype(np.float16, copy=False))
        _atomic_parquet(
            ref.meta_path,
            pl.DataFrame({"entity_id": ids, "country": countries}),
        )
        self._manifest["shards"][str(index)] = {"rows": ref.rows}
        self._manifest["dim"] = self.dim
        self._write_manifest()
        return ref

    # -- reading ----------------------------------------------------------- #

    def refs(self) -> list[ShardRef]:
        """Completed shards in index order."""
        entries = sorted(self._manifest["shards"].items(), key=lambda kv: int(kv[0]))
        return [ShardRef(int(i), e["rows"], self.directory) for i, e in entries]

    def iter_shards(self) -> Iterator[tuple[ShardRef, np.ndarray, object]]:
        """Yield ``(ref, memmapped vectors, meta frame)`` per completed shard."""
        for ref in self.refs():
            yield ref, ref.load_vectors(), ref.load_meta()

    def total_rows(self) -> int:
        return sum(ref.rows for ref in self.refs())


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.npy")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        np.save(tmp_path, array, allow_pickle=False)
        # np.save appends .npy when the name lacks it; ours already ends in .npy
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_parquet(path: Path, frame) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        frame.write_parquet(tmp_path)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
