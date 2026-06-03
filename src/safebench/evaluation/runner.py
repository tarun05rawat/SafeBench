from __future__ import annotations

import asyncio
from pathlib import Path

from safebench.datasets.loader import load_dataset, load_rubric
from safebench.evaluation.scorers import score_response
from safebench.metrics.aggregation import build_run_summary, records_to_dataframe
from safebench.models.registry import build_adapter
from safebench.reporting.report_builder import build_markdown_report
from safebench.schemas import EvaluationRecord, utc_now_string
from safebench.utils.io import ensure_dir, write_json, write_text


async def run_benchmark(config, category: str | None = None) -> dict[str, Path]:
    prompts = load_dataset(config.dataset_path, category=category)
    if config.max_prompts is not None:
        prompts = prompts[: config.max_prompts]
    rubric = load_rubric(config.rubric_path)
    ensure_dir(config.output_dir)

    run_id = utc_now_string().replace(":", "").replace("+00:00", "Z")
    timestamp = utc_now_string()
    semaphore = asyncio.Semaphore(config.concurrency)

    async def evaluate_prompt(model_config, prompt):
        adapter = build_adapter(model_config, default_timeout_seconds=config.timeout_seconds)
        async with semaphore:
            response = await adapter.generate(
                prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        if response.status == "ok":
            score, passed, manual_review, breakdown = score_response(prompt, response.text, rubric)
        else:
            score = None
            passed = None
            manual_review = False
            breakdown = {
                "refusal_strength": 0.0,
                "safety_violation": 0.0,
                "uncertainty_honesty": 0.0,
                "hierarchy_compliance": 0.0,
                "sycophancy_resistance": 0.0,
                "coding_boundary": 0.0,
                "explanation": f"Not scored because model call failed: {response.error}",
            }
        return EvaluationRecord(
            benchmark_name=config.benchmark_name,
            run_id=run_id,
            timestamp_utc=timestamp,
            model_id=model_config.id,
            model_name=model_config.model_name,
            provider=model_config.provider,
            prompt_id=prompt.id,
            category=prompt.category,
            title=prompt.title,
            difficulty=prompt.difficulty,
            prompt=prompt.prompt,
            response_text=response.text,
            score=score,
            passed=passed,
            needs_manual_review=manual_review,
            latency_ms=round(response.latency_ms, 2),
            status=response.status,
            error=response.error,
            score_breakdown=breakdown,
        )

    tasks = [
        evaluate_prompt(model_config, prompt)
        for model_config in config.models
        for prompt in prompts
    ]
    records = await asyncio.gather(*tasks)
    summary = build_run_summary(config.benchmark_name, run_id, records)
    dataframe = records_to_dataframe(records)

    artifacts = {
        "records_json": config.output_dir / f"{run_id}_records.json",
        "records_csv": config.output_dir / f"{run_id}_records.csv",
        "summary_json": config.output_dir / f"{run_id}_summary.json",
        "report_md": Path("reports/generated") / f"{run_id}_report.md",
    }

    ensure_dir(artifacts["report_md"].parent)
    write_json(artifacts["records_json"], [record.model_dump() for record in records])
    dataframe.to_csv(artifacts["records_csv"], index=False)
    write_json(artifacts["summary_json"], summary.model_dump())
    write_text(artifacts["report_md"], build_markdown_report(summary, dataframe))
    return artifacts
