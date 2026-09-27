"""Streaming TSV readers and writers.

The source files are 200-500 MB each and the test set is 1.7M+ entities, so
everything here streams. Fields are split on tabs with no quoting, matching the
problem statement: addresses contain commas, which is exactly why the format is
tab-separated.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

DELIM = "\t"


@dataclass(slots=True)
class Record:
    """One business record from any of the three sources."""

    entity_id: str
    name: str
    address: str
    country: str

    @property
    def source(self) -> str:
        """``S1`` / ``S2`` / ``S3``, taken from the entity_id prefix."""
        return self.entity_id.split("-", 1)[0]


def read_records(path: Path) -> Iterator[Record]:
    """Stream records from a source TSV, skipping the header.

    Rows with fewer than 4 fields are padded rather than dropped: a missing
    address is meaningful (3.3% of S2/S3 records have none) and the entity still
    needs a row in the submission.
    """
    with open(path, encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(DELIM)
            if len(parts) < 4:
                parts = parts + [""] * (4 - len(parts))
            yield Record(parts[0], parts[1], parts[2], parts[3])


def read_id_lists(path: Path) -> dict[str, set[str]]:
    """Read a ``{id}\\t{comma,separated,ids}`` file (ground truth or submission)."""
    mapping: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            key, tab, rest = line.partition(DELIM)
            if not tab:
                continue
            rest = rest.rstrip("\n").strip()
            mapping[key] = set(rest.split(",")) if rest else set()
    return mapping


def iter_id_lists(path: Path) -> Iterator[tuple[str, set[str]]]:
    """Streaming variant of :func:`read_id_lists` for files too big to hold."""
    with open(path, encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            key, tab, rest = line.partition(DELIM)
            if not tab:
                continue
            rest = rest.rstrip("\n").strip()
            yield key, (set(rest.split(",")) if rest else set())


@contextmanager
def atomic_write(path: Path, encoding: str = "utf-8"):
    """Write to a temp file in the same directory, then rename into place.

    Keeps a crashed multi-hour run from leaving a half-written shard or output
    file that a later stage would silently treat as complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        with open(tmp_path, "w", encoding=encoding, newline="") as handle:
            yield handle
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_id_lists(path: Path, rows: Iterator[tuple[str, set[str]]], header: str) -> int:
    """Write a submission-style TSV. Returns the number of data rows written.

    IDs are sorted for determinism, and an empty set writes an empty second
    column -- which is how a singleton is declared.
    """
    count = 0
    with atomic_write(path) as handle:
        handle.write(f"source1_entity_id{DELIM}{header}\n")
        for key, ids in rows:
            handle.write(f"{key}{DELIM}{','.join(sorted(ids))}\n")
            count += 1
    return count
