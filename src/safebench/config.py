from __future__ import annotations

from pathlib import Path

import yaml

from safebench.schemas import BenchmarkConfig


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    config = BenchmarkConfig.model_validate(raw)
    base_dir = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    config.dataset_path = (base_dir / config.dataset_path).resolve() if not config.dataset_path.is_absolute() else config.dataset_path
    config.rubric_path = (base_dir / config.rubric_path).resolve() if not config.rubric_path.is_absolute() else config.rubric_path
    config.output_dir = (base_dir / config.output_dir).resolve() if not config.output_dir.is_absolute() else config.output_dir
    return config

