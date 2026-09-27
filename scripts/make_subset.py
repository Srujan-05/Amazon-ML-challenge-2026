#!/usr/bin/env python3
"""Carve a smaller, self-consistent dataset out of the full one.

The full training data is 2.2M S1 entities against 10.3M comparable records, so a
single end-to-end iteration is measured in hours. This builds a miniature copy
with the same structure -- sampled S1 entities, *all* of their true matches, plus
unmatched distractors -- so the pipeline can be exercised and tuned in minutes.

Keeping every true match is what makes the subset honest: recall measured on it
is directly comparable to recall on the full set, just at lower selectivity.

    uv run python scripts/make_subset.py --entities 20000 --distractors 400000 \
        --out student_resource/dataset_subset
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.config import load_config  # noqa: E402
from ber.io import DELIM, iter_id_lists  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=int, default=20_000, help="S1 entities to sample")
    parser.add_argument(
        "--distractors", type=int, default=400_000,
        help="unmatched S2/S3 records to keep, split across the two sources",
    )
    parser.add_argument("--out", default="student_resource/dataset_subset")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--test-entities", type=int, default=0,
        help="also carve a test split of this many S1 entities (no ground truth)",
    )
    args = parser.parse_args()

    cfg = load_config()
    paths = cfg.paths
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = Path(__file__).resolve().parents[1] / out_root

    rng = random.Random(args.seed)

    # 1. Reservoir-sample entities so the subset is not just the head of the file.
    reservoir: list[tuple[str, set[str]]] = []
    for i, row in enumerate(iter_id_lists(paths.ground_truth())):
        if i < args.entities:
            reservoir.append(row)
        else:
            j = rng.randint(0, i)
            if j < args.entities:
                reservoir[j] = row
    gt = dict(reservoir)
    matched = set().union(*gt.values()) if gt else set()
    print(f"sampled {len(gt)} S1 entities with {len(matched)} true matches")

    for split in (["train", "test"] if args.test_entities else ["train"]):
        (out_root / split).mkdir(parents=True, exist_ok=True)

    # 2. Copy the S1 records for the sampled entities.
    _filter_file(
        paths.source_file("train", "source1"),
        out_root / "train" / "train_source1.tsv",
        keep=set(gt),
    )
    # 3. Copy every true match, plus distractors.
    per_source = args.distractors // 2
    for source in ("source2", "source3"):
        _filter_file(
            paths.source_file("train", source),
            out_root / "train" / f"train_{source}.tsv",
            keep=matched,
            extra=per_source,
            rng=rng,
        )
    # 4. Ground truth for exactly the sampled entities.
    gt_out = out_root / "train" / "train_ground_truth.tsv"
    with open(gt_out, "w", encoding="utf-8") as handle:
        handle.write(f"source1_entity_id{DELIM}matched_entity_ids\n")
        for entity, ids in gt.items():
            handle.write(f"{entity}{DELIM}{','.join(sorted(ids))}\n")
    print(f"wrote {gt_out.relative_to(out_root.parent)}")

    if args.test_entities:
        _head_file(
            paths.source_file("test", "source1"),
            out_root / "test" / "test_source1.tsv",
            args.test_entities,
        )
        for source in ("source2", "source3"):
            _head_file(
                paths.source_file("test", source),
                out_root / "test" / f"test_{source}.tsv",
                args.distractors // 2,
            )

    print(f"\nsubset ready at {out_root}")
    print("use it with:  --set paths.data_root=" + str(out_root))
    return 0


def _filter_file(
    source: Path, target: Path, keep: set[str], extra: int = 0, rng: random.Random | None = None
) -> None:
    """Copy rows whose ID is in ``keep``, plus up to ``extra`` other rows."""
    written = added = 0
    with open(source, encoding="utf-8") as src, open(target, "w", encoding="utf-8") as dst:
        dst.write(src.readline())
        for line in src:
            entity_id = line.split(DELIM, 1)[0]
            if entity_id in keep:
                dst.write(line)
                written += 1
            elif added < extra:
                dst.write(line)
                added += 1
    print(f"wrote {target.name}: {written} matched + {added} distractors")


def _head_file(source: Path, target: Path, limit: int) -> None:
    with open(source, encoding="utf-8") as src, open(target, "w", encoding="utf-8") as dst:
        dst.write(src.readline())
        for i, line in enumerate(src):
            if i >= limit:
                break
            dst.write(line)
    print(f"wrote {target.name}: {limit} rows")


if __name__ == "__main__":
    raise SystemExit(main())
