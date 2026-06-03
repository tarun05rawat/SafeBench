from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import typer

from safebench.config import load_config
from safebench.datasets.loader import load_rubric
from safebench.evaluation.scorers import rescore_record
from safebench.metrics.aggregation import build_run_summary
from safebench.reporting.report_builder import build_markdown_report
from safebench.schemas import EvaluationRecord
from safebench.utils.io import write_json, write_text
from safebench.utils.logging import configure_logging
from safebench.utils.tables import markdown_table
from safebench.evaluation.runner import run_benchmark

app = typer.Typer(help="SafeBench CLI for running safety evaluations and generating reports.")


@app.command("run-benchmark")
def run_benchmark_command(
    config_path: Path = typer.Option(Path("config/benchmark.demo.yaml"), exists=True),
    category: str | None = typer.Option(None, help="Optional single category filter."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging(verbose)
    config = load_config(config_path)
    artifacts = asyncio.run(run_benchmark(config, category=category))
    typer.echo("Benchmark completed.")
    for name, path in artifacts.items():
        typer.echo(f"{name}: {path}")


@app.command("evaluate-category")
def evaluate_category(
    category: str = typer.Argument(..., help="Category slug to evaluate."),
    config_path: Path = typer.Option(Path("config/benchmark.demo.yaml"), exists=True),
) -> None:
    config = load_config(config_path)
    artifacts = asyncio.run(run_benchmark(config, category=category))
    typer.echo(f"Category run complete for {category}.")
    for name, path in artifacts.items():
        typer.echo(f"{name}: {path}")


@app.command("summarize-results")
def summarize_results(
    records_json: Path = typer.Argument(..., exists=True, help="Path to *_records.json artifact."),
    rubric_path: Path = typer.Option(Path("data/rubrics/scoring_guide.yaml"), exists=True),
    rescore: bool = typer.Option(True, help="Recompute scores from saved raw responses."),
) -> None:
    raw = json.loads(records_json.read_text())
    records = [EvaluationRecord.model_validate(item) for item in raw]
    if rescore:
        rubric = load_rubric(rubric_path)
        records = [rescore_record(record, rubric) for record in records]
    summary = build_run_summary(records[0].benchmark_name, records[0].run_id, records)
    typer.echo(markdown_table(summary.overall))


@app.command("generate-plots")
def generate_plots_command(
    records_csv: Path = typer.Argument(..., exists=True),
    output_dir: Path = typer.Option(Path("results/plots")),
) -> None:
    from safebench.plots.generate import generate_plots

    paths = generate_plots(records_csv, output_dir)
    for path in paths:
        typer.echo(path)


@app.command("build-report")
def build_report(
    records_json: Path = typer.Argument(..., exists=True),
    output_path: Path = typer.Option(Path("reports/generated/manual_report.md")),
    rubric_path: Path = typer.Option(Path("data/rubrics/scoring_guide.yaml"), exists=True),
    rescore: bool = typer.Option(True, help="Recompute scores from saved raw responses."),
) -> None:
    raw = json.loads(records_json.read_text())
    records = [EvaluationRecord.model_validate(item) for item in raw]
    if rescore:
        rubric = load_rubric(rubric_path)
        records = [rescore_record(record, rubric) for record in records]
    summary = build_run_summary(records[0].benchmark_name, records[0].run_id, records)
    df = pd.DataFrame([record.model_dump() for record in records])
    report = build_markdown_report(summary, df)
    write_text(output_path, report)
    write_json(output_path.with_suffix(".summary.json"), summary.model_dump())
    typer.echo(output_path)


if __name__ == "__main__":
    app()
