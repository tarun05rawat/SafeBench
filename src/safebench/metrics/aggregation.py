from __future__ import annotations

import pandas as pd

from safebench.schemas import CategorySummary, EvaluationRecord, RunSummary, utc_now_string


def _records_with_none(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


def records_to_dataframe(records: list[EvaluationRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = record.model_dump()
        row["score_explanation"] = row["score_breakdown"]["explanation"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_run_summary(benchmark_name: str, run_id: str, records: list[EvaluationRecord]) -> RunSummary:
    df = records_to_dataframe(records)
    successful_df = df[df["status"] == "ok"].copy()
    coverage_df = (
        df.groupby("model_id")
        .agg(
            prompts_attempted=("prompt_id", "count"),
            prompts_succeeded=("status", lambda values: int((values == "ok").sum())),
            prompts_failed=("status", lambda values: int((values != "ok").sum())),
        )
        .reset_index()
    )
    baseline_overall_df = coverage_df.assign(
        mean_score=None,
        pass_rate=None,
        avg_latency_ms=None,
        manual_review_rate=None,
    )
    overall = _records_with_none(baseline_overall_df)

    if successful_df.empty:
        by_category = []
        failure_cases = []
    else:
        overall_df = (
            successful_df.groupby("model_id")
            .agg(
                mean_score=("score", "mean"),
                pass_rate=("passed", "mean"),
                avg_latency_ms=("latency_ms", "mean"),
                manual_review_rate=("needs_manual_review", "mean"),
            )
            .reset_index()
        )
        overall_df = coverage_df.merge(overall_df, on="model_id", how="left").sort_values(
            ["prompts_succeeded", "mean_score"], ascending=[False, False]
        )
        overall = _records_with_none(overall_df.round(3))

        by_category_df = (
            df.groupby(["model_id", "category"])
            .agg(
                prompts_evaluated=("prompt_id", "count"),
                prompts_succeeded=("status", lambda values: int((values == "ok").sum())),
                prompts_failed=("status", lambda values: int((values != "ok").sum())),
            )
            .reset_index()
        )
        success_category_df = (
            successful_df.groupby(["model_id", "category"])
            .agg(
                mean_score=("score", "mean"),
                pass_rate=("passed", "mean"),
                manual_review_rate=("needs_manual_review", "mean"),
                avg_latency_ms=("latency_ms", "mean"),
            )
            .reset_index()
        )
        by_category_df = by_category_df.merge(
            success_category_df,
            on=["model_id", "category"],
            how="left",
        ).sort_values(["category", "prompts_succeeded", "mean_score"], ascending=[True, False, False])
        by_category = [
            CategorySummary(**row)
            for row in _records_with_none(by_category_df.round(3))
        ]

        failure_cases = (
            successful_df.sort_values(["score", "latency_ms"], ascending=[True, False])
            .head(8)[["model_id", "category", "prompt_id", "title", "score", "response_text", "score_explanation"]]
            .to_dict(orient="records")
        )

    if not successful_df.empty:
        successful_models = set(successful_df["model_id"].unique().tolist())
        failed_models = [model for model in coverage_df["model_id"].tolist() if model not in successful_models]
    else:
        failed_models = coverage_df["model_id"].tolist()

    notes = [
        "Scores are heuristic and intended for comparative benchmarking, not definitive safety guarantees.",
        "Manual review is recommended for borderline or context-sensitive examples.",
        "Errored model calls are excluded from scoring and should be interpreted as missing coverage, not low safety performance.",
    ]
    if failed_models:
        notes.append(
            "Models with zero successful responses in this run: " + ", ".join(sorted(failed_models)) + "."
        )

    return RunSummary(
        benchmark_name=benchmark_name,
        run_id=run_id,
        created_at_utc=utc_now_string(),
        categories=sorted(df["category"].unique().tolist()),
        models=sorted(df["model_id"].unique().tolist()),
        overall=overall,
        by_category=by_category,
        failure_cases=failure_cases,
        notes=notes,
    )
