"""Configuration loading.

Every tunable lives in ``configs/default.yaml``; this module turns it into a
dotted-access mapping so callers read ``cfg.embedding.dim`` rather than
threading dictionaries around. Paths are resolved against the repo root so the
pipeline behaves the same from any working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class Section(dict):
    """A dict whose keys are also attributes, nested recursively."""

    def __init__(self, data: dict[str, Any]):
        super().__init__(data)
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = Section(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - programmer error
            raise AttributeError(f"no config key {name!r}") from exc


@dataclass(frozen=True)
class Paths:
    data_root: Path
    vectors: Path
    artifacts: Path
    output: Path

    def ensure(self) -> None:
        for path in (self.vectors, self.artifacts, self.output):
            path.mkdir(parents=True, exist_ok=True)

    def split_dir(self, split: str) -> Path:
        """``train`` / ``test`` subdirectory of the dataset."""
        return self.data_root / split

    def source_file(self, split: str, source: str) -> Path:
        """e.g. ``(train, source1)`` -> ``dataset/train/train_source1.tsv``."""
        return self.split_dir(split) / f"{split}_{source}.tsv"

    def ground_truth(self) -> Path:
        return self.split_dir("train") / "train_ground_truth.tsv"


class Config(Section):
    """Top-level config with resolved paths."""

    @property
    def paths(self) -> Paths:
        raw = self["paths"]
        return Paths(
            data_root=_resolve(raw["data_root"]),
            vectors=_resolve(raw["vectors"]),
            artifacts=_resolve(raw["artifacts"]),
            output=_resolve(raw["output"]),
        )

    @property
    def instruction(self) -> str:
        """The editable embedding instruction from ``configs/instruction.txt``."""
        path = _resolve(self["embedding"]["instruction_file"])
        return path.read_text(encoding="utf-8").strip()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load the YAML config, optionally applying ``a.b.c=value`` overrides."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if overrides:
        for dotted, value in overrides.items():
            _set_dotted(data, dotted, value)
    return Config(data)


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cursor = data
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value
