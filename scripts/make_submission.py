#!/usr/bin/env python3
"""Assemble the submission zip in the exact layout the challenge requires.

    <team_name>_submission.zip
    |- output/
    |   |- matching_results.tsv
    |   `- candidate_pairs.tsv
    |- code/business_entity_resolution/
    |   |- src/
    |   |- README.md
    |   `- requirements.txt
    `- Documentation_template.md

Runs the official validator first and refuses to build a package that would be
rejected -- a failed submission costs more than a failed script.

    uv run python scripts/make_submission.py --team my_team
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ber.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", required=True, help="team name, used for the zip filename")
    parser.add_argument("--out", default="submission", help="staging directory")
    parser.add_argument("--skip-validate", action="store_true",
                        help="build even if the validator fails (not recommended)")
    parser.add_argument("--check-ids", action="store_true",
                        help="also verify every matched ID exists in the test set")
    args = parser.parse_args()

    cfg = load_config()
    output = cfg.paths.output
    matching = output / "matching_results.tsv"
    candidates = output / "candidate_pairs.tsv"

    for path in (matching, candidates):
        if not path.is_file():
            print(f"missing {path}; run `ber predict --split test` first", file=sys.stderr)
            return 1

    if not args.skip_validate and not _validate(cfg, args.check_ids):
        print("\nvalidator failed -- fix the issues above, or pass --skip-validate.",
              file=sys.stderr)
        return 1

    stage = ROOT / args.out
    if stage.exists():
        shutil.rmtree(stage)
    code = stage / "code" / "business_entity_resolution"
    (stage / "output").mkdir(parents=True)
    code.mkdir(parents=True)

    shutil.copy2(matching, stage / "output" / matching.name)
    shutil.copy2(candidates, stage / "output" / candidates.name)

    shutil.copytree(ROOT / "src", code / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "configs", code / "configs")
    shutil.copytree(ROOT / "scripts", code / "scripts",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "tests", code / "tests",
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    shutil.copy2(ROOT / "README.md", code / "README.md")
    shutil.copy2(ROOT / "pyproject.toml", code / "pyproject.toml")
    shutil.copy2(ROOT / "plan.md", code / "plan.md")
    _write_requirements(code / "requirements.txt")

    doc = ROOT / "student_resource" / "Documentation_template.md"
    if doc.is_file():
        shutil.copy2(doc, stage / "Documentation_template.md")
    else:
        print(f"warning: {doc} not found; add your methodology document to the zip")

    # Model stats and the comparison table are useful evidence for the reviewers.
    comparison = cfg.paths.artifacts / "model_comparison.json"
    if comparison.is_file():
        shutil.copy2(comparison, code / "model_comparison.json")

    archive = ROOT / f"{args.team}_submission.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(stage))

    size_mb = archive.stat().st_size / 1024**2
    print(f"\nwrote {archive.name} ({size_mb:.1f} MB)")
    print(f"staged tree kept at {stage}")
    return 0


def _validate(cfg, check_ids: bool) -> bool:
    """Run the challenge's own validator against the outputs."""
    script = cfg.paths.data_root.parent / "utils" / "validate_submission.py"
    if not script.is_file():
        print(f"warning: validator not found at {script}; skipping")
        return True
    command = [
        sys.executable, str(script),
        "--matching", str(cfg.paths.output / "matching_results.tsv"),
        "--candidate", str(cfg.paths.output / "candidate_pairs.tsv"),
        "--test-dir", str(cfg.paths.data_root / "test"),
    ]
    if check_ids:
        command.append("--check-ids")
    print("running the official validator ...")
    return subprocess.call(command) == 0


def _write_requirements(path: Path) -> None:
    """Export pinned dependencies, so the package is reproducible."""
    try:
        result = subprocess.run(
            ["uv", "export", "--no-hashes", "--no-dev", "--format", "requirements-txt"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        path.write_text(result.stdout, encoding="utf-8")
        print(f"wrote {path.name} from uv export")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fall back to the declared ranges if uv is unavailable in this context.
        import tomllib

        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        deps = data["project"]["dependencies"]
        path.write_text("\n".join(deps) + "\n", encoding="utf-8")
        print(f"wrote {path.name} from pyproject (uv export unavailable)")


if __name__ == "__main__":
    raise SystemExit(main())
