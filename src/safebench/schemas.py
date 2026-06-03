from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field


SUPPORTED_CATEGORIES = {
    "harmful_request_refusal",
    "prompt_injection_resistance",
    "honesty_under_uncertainty",
    "instruction_hierarchy",
    "sycophancy_pressure",
    "unsafe_coding_assistance",
}


class PromptItem(BaseModel):
    id: str
    category: Literal[
        "harmful_request_refusal",
        "prompt_injection_resistance",
        "honesty_under_uncertainty",
        "instruction_hierarchy",
        "sycophancy_pressure",
        "unsafe_coding_assistance",
    ]
    title: str
    prompt: str
    expected_behavior: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    tags: list[str] = Field(default_factory=list)


class DatasetBundle(BaseModel):
    prompts: list[PromptItem]


class ModelConfig(BaseModel):
    id: str
    provider: str
    model_name: str
    api_key_env: str | None = None
    base_url: str | None = None
    system_prompt: str | None = None
    profile: str | None = None
    timeout_seconds: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    benchmark_name: str
    dataset_path: Path
    rubric_path: Path
    output_dir: Path
    models: list[ModelConfig]
    concurrency: int = 4
    timeout_seconds: int = 60
    temperature: float = 0.0
    max_tokens: int = 300
    max_prompts: int | None = None


class ModelResponse(BaseModel):
    prompt_id: str
    model_id: str
    text: str
    latency_ms: float
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ScoreBreakdown(BaseModel):
    refusal_strength: float = 0.0
    safety_violation: float = 0.0
    uncertainty_honesty: float = 0.0
    hierarchy_compliance: float = 0.0
    sycophancy_resistance: float = 0.0
    coding_boundary: float = 0.0
    explanation: str


class EvaluationRecord(BaseModel):
    benchmark_name: str
    run_id: str
    timestamp_utc: str
    model_id: str
    model_name: str
    provider: str
    prompt_id: str
    category: str
    title: str
    difficulty: str
    prompt: str
    response_text: str
    score: float | None
    passed: bool | None
    needs_manual_review: bool
    latency_ms: float
    status: str
    error: str | None = None
    score_breakdown: ScoreBreakdown


class CategorySummary(BaseModel):
    model_id: str
    category: str
    mean_score: float | None
    pass_rate: float | None
    prompts_evaluated: int
    prompts_succeeded: int
    prompts_failed: int
    manual_review_rate: float | None
    avg_latency_ms: float | None


class RunSummary(BaseModel):
    benchmark_name: str
    run_id: str
    created_at_utc: str
    categories: list[str]
    models: list[str]
    overall: list[dict[str, Any]]
    by_category: list[CategorySummary]
    failure_cases: list[dict[str, Any]]
    notes: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def overall_ranking(self) -> list[str]:
        sorted_rows = sorted(
            self.overall,
            key=lambda row: (
                row.get("prompts_succeeded", 0),
                row.get("mean_score") if row.get("mean_score") is not None else -1.0,
            ),
            reverse=True,
        )
        return [row["model_id"] for row in sorted_rows]


def utc_now_string() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
