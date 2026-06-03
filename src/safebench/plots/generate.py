from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="safebench-mpl-"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from safebench.utils.io import ensure_dir


def generate_plots(records_csv: str | Path, output_dir: str | Path) -> list[Path]:
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(records_csv)
    out_dir = Path(output_dir)
    ensure_dir(out_dir)
    paths: list[Path] = []

    plt.figure(figsize=(8, 5))
    overall = df.groupby("model_id")["score"].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=overall, x="model_id", y="score", hue="model_id", dodge=False, legend=False)
    plt.ylim(0, 1)
    plt.title("Mean Safety Score by Model")
    plt.tight_layout()
    overall_path = out_dir / "overall_scores.png"
    plt.savefig(overall_path, dpi=180)
    plt.close()
    paths.append(overall_path)

    plt.figure(figsize=(10, 6))
    category_scores = df.pivot_table(index="category", columns="model_id", values="score", aggfunc="mean")
    sns.heatmap(category_scores, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Category-Level Score Heatmap")
    plt.tight_layout()
    heatmap_path = out_dir / "category_heatmap.png"
    plt.savefig(heatmap_path, dpi=180)
    plt.close()
    paths.append(heatmap_path)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="category", y="score", hue="model_id")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title("Score Distribution by Category")
    plt.tight_layout()
    boxplot_path = out_dir / "score_distribution.png"
    plt.savefig(boxplot_path, dpi=180)
    plt.close()
    paths.append(boxplot_path)

    return paths
