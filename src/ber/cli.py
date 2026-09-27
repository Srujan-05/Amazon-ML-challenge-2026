"""The ``ber`` command line.

One subcommand per pipeline stage, in the order they run:

    bench-embedder   measure throughput, project the full run  (gate)
    embed            embed a split's records into shards        (resumable)
    index            stage by country and build the vector indices
    block            candidate generation -> candidate_pairs.tsv
    train            features + train models + comparison
    compare-models   re-print the comparison table
    predict          score a split -> matching_results.tsv
    validate         run the official submission validator

Any config value can be overridden inline with ``--set a.b.c=value``, so
experiments do not require editing the YAML.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import load_config


def _parse_overrides(pairs: list[str]) -> dict:
    """Turn ``--set a.b=1`` arguments into a nested override dict."""
    import yaml

    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        # YAML-parse the value so numbers, booleans and null arrive typed.
        out[key.strip()] = yaml.safe_load(value)
    return out


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument(
        "--set", dest="overrides", action="append", metavar="KEY=VALUE",
        help="override a config value, e.g. --set blocking.k=75",
    )
    parser.add_argument("--quiet", action="store_true", help="warnings only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ber", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("bench-embedder", help="measure embedding throughput")
    bench.add_argument("--sample", type=int, default=20_000, help="records to encode")
    bench.add_argument("--split", default="train")
    _add_common(bench)

    embed = subparsers.add_parser("embed", help="embed a split into shards (resumable)")
    embed.add_argument("--split", default="train", choices=["train", "test"])
    embed.add_argument("--sources", nargs="*", default=None, help="e.g. source1 source2")
    embed.add_argument("--limit", type=int, default=None, help="cap records per source")
    _add_common(embed)

    index = subparsers.add_parser("index", help="stage by country and build vector indices")
    index.add_argument("--split", default="train", choices=["train", "test"])
    index.add_argument("--force", action="store_true", help="rebuild existing indices")
    _add_common(index)

    block = subparsers.add_parser("block", help="generate candidate pairs")
    block.add_argument("--split", default="train", choices=["train", "test"])
    block.add_argument("--report-recall", action="store_true",
                       help="measure the blocking recall ceiling (train only)")
    block.add_argument("--write-tsv", action="store_true", help="also write candidate_pairs.tsv")
    block.add_argument("--limit-entities", type=int, default=None)
    _add_common(block)

    train = subparsers.add_parser("train", help="train and compare models")
    train.add_argument("--models", nargs="*", default=None,
                       help="subset of: lightgbm xgboost mlp entity_attn")
    train.add_argument("--force-features", action="store_true",
                       help="re-featurize instead of using the cache")
    _add_common(train)

    compare = subparsers.add_parser("compare-models", help="print the comparison table")
    _add_common(compare)

    predict = subparsers.add_parser("predict", help="score a split and write the outputs")
    predict.add_argument("--split", default="test", choices=["train", "test"])
    predict.add_argument("--model", default="best", help="model name, or 'best'")
    predict.add_argument("--chunk-entities", type=int, default=20_000)
    _add_common(predict)

    validate = subparsers.add_parser("validate", help="run the official submission validator")
    validate.add_argument("--check-ids", action="store_true")
    _add_common(validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config, _parse_overrides(args.overrides))

    if args.command == "bench-embedder":
        from .bench import run_benchmark

        print(json.dumps(run_benchmark(cfg, args.split, args.sample), indent=2))
        return 0

    if args.command == "embed":
        from .embed.run_embed import run

        stats = run(cfg, args.split, sources=args.sources, limit=args.limit)
        print(json.dumps(stats.as_dict(), indent=2))
        return 0

    if args.command == "index":
        from .vectors.store import BASE, COMPARABLE, build_store

        for collection in (BASE, COMPARABLE):
            store = build_store(cfg, args.split, collection)
            counts = store.stage()
            print(f"{collection}: staged {counts}")
            store.build(force=args.force)
        return 0

    if args.command == "block":
        from .blocking.candidates import generate, write_candidate_tsv

        _out, report = generate(cfg, args.split, limit_entities=args.limit_entities)
        print(json.dumps(report.as_dict(), indent=2))
        if args.report_recall:
            from .blocking.candidates import candidate_map
            from .eval import blocking_recall
            from .io import read_id_lists

            truth = read_id_lists(cfg.paths.ground_truth())
            print(json.dumps(blocking_recall(candidate_map(cfg, args.split), truth), indent=2))
        if args.write_tsv:
            rows = write_candidate_tsv(cfg, args.split)
            print(f"wrote candidate_pairs.tsv ({rows} rows)")
        return 0

    if args.command == "train":
        from .models.registry import DEFAULT_MODELS
        from .train import run

        run(
            cfg,
            args.models or DEFAULT_MODELS,
            overrides=args.overrides,
            force_features=args.force_features,
        )
        return 0

    if args.command == "compare-models":
        from .models.compare import format_table, write_comparison

        path, rows = write_comparison(cfg)
        print(format_table(rows))
        print(f"\nwrote {path}")
        return 0

    if args.command == "predict":
        from .models.compare import best_model_name
        from .predict import run

        name = best_model_name(cfg) if args.model == "best" else args.model
        summary = run(cfg, name, args.split, chunk_entities=args.chunk_entities)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "validate":
        return _validate(cfg, args.check_ids)

    raise SystemExit(f"unhandled command {args.command!r}")


def _validate(cfg, check_ids: bool) -> int:
    """Shell out to the challenge's own validator, so we test what they test."""
    import subprocess

    root = cfg.paths.data_root.parent
    script = root / "utils" / "validate_submission.py"
    if not script.is_file():
        print(f"validator not found at {script}", file=sys.stderr)
        return 1
    command = [
        sys.executable, str(script),
        "--matching", str(cfg.paths.output / "matching_results.tsv"),
        "--candidate", str(cfg.paths.output / "candidate_pairs.tsv"),
        "--test-dir", str(cfg.paths.data_root / "test"),
    ]
    if check_ids:
        command.append("--check-ids")
    return subprocess.call(command)
