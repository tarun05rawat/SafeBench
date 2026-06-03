from __future__ import annotations

from pathlib import Path

import yaml

from safebench.schemas import DatasetBundle, PromptItem


def load_dataset(path: str | Path, category: str | None = None) -> list[PromptItem]:
    dataset_path = Path(path)
    raw = yaml.safe_load(dataset_path.read_text())
    bundle = DatasetBundle.model_validate(raw)
    if category:
        return [item for item in bundle.prompts if item.category == category]
    return bundle.prompts


def load_rubric(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())
