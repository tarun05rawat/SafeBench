from __future__ import annotations

import pandas as pd

from safebench.schemas import RunSummary
from safebench.utils.tables import markdown_table


def build_markdown_report(summary: RunSummary, df: pd.DataFrame) -> str:
    overall_table = markdown_table(summary.overall)
    category_table = markdown_table([item.model_dump() for item in summary.by_category])
    successful_df = df[df["status"] == "ok"].copy()

    failure_lines = []
    for case in summary.failure_cases[:5]:
        snippet = case["response_text"][:220].replace("\n", " ")
        failure_lines.append(
            f"- `{case['model_id']}` on `{case['prompt_id']}` ({case['category']}), score={case['score']}: {snippet}"
        )
    failure_block = "\n".join(failure_lines) if failure_lines else "- No failure cases captured."

    coverage_rows = []
    for row in summary.overall:
        coverage_rows.append(
            {
                "model_id": row["model_id"],
                "prompts_attempted": row.get("prompts_attempted", 0),
                "prompts_succeeded": row.get("prompts_succeeded", 0),
                "prompts_failed": row.get("prompts_failed", 0),
            }
        )
    coverage_table = markdown_table(coverage_rows)

    takeaway_lines = []
    if successful_df.empty:
        takeaway_lines.append("- No model produced successful responses, so no comparative takeaways are available.")
    else:
        category_takeaways = (
            successful_df.groupby(["category", "model_id"])["score"]
            .mean()
            .reset_index()
            .sort_values(["category", "score"], ascending=[True, False])
        )
        for category in summary.categories:
            subset = category_takeaways[category_takeaways["category"] == category]
            if subset.empty:
                takeaway_lines.append(f"- `{category}`: no successful model responses were recorded.")
                continue
            best = subset.iloc[0]
            takeaway_lines.append(
                f"- `{category}`: strongest model among successful responses was `{best['model_id']}` with mean score {best['score']:.2f}."
            )

    return f"""# SafeBench Short Report

## Overview
This report summarizes `{summary.benchmark_name}` run `{summary.run_id}`. SafeBench is a comparative evaluation harness for probing LLM behavior on safety, robustness, and instruction-following tasks. Scores are heuristic, category-specific, and intended for empirical comparison rather than safety certification.

## Experimental Setup
- Models evaluated: {", ".join(summary.models)}
- Categories: {", ".join(summary.categories)}
- Total prompt-model evaluations: {len(df)}
- Successful evaluations: {int((df["status"] == "ok").sum())}/{len(df)}
- Manual-review-flagged rate among successful evaluations: {(successful_df["needs_manual_review"].mean() if not successful_df.empty else 0):.1%}

## Coverage
{coverage_table}

## Overall Results
{overall_table}

## Category Breakdown
{category_table}

## Qualitative Failure Cases
{failure_block}

## Key Takeaways
{chr(10).join(takeaway_lines)}

## Limitations
- Prompt coverage is intentionally small and should be expanded before drawing broad conclusions.
- Heuristic scoring can miss nuanced harms, over-refusals, and subtle prompt injection failures.
- Provider-side model updates may shift behavior over time, so runs should be versioned and repeated.

## Next Steps
- Expand the dataset with adversarial paraphrases and multi-turn interactions.
- Add judge-model or human annotation layers for rubric calibration.
- Benchmark latency/cost tradeoffs alongside safety performance.
"""
