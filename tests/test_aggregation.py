from safebench.metrics.aggregation import build_run_summary
from safebench.schemas import EvaluationRecord


def _record(model_id: str, status: str, score: float | None, passed: bool | None, category: str = "harmful_request_refusal"):
    return EvaluationRecord.model_validate(
        {
            "benchmark_name": "test-benchmark",
            "run_id": "run-1",
            "timestamp_utc": "2026-05-31T00:00:00+00:00",
            "model_id": model_id,
            "model_name": model_id,
            "provider": "mock",
            "prompt_id": f"{model_id}-{status}",
            "category": category,
            "title": "test",
            "difficulty": "easy",
            "prompt": "test prompt",
            "response_text": "" if status == "error" else "I can't help with that.",
            "score": score,
            "passed": passed,
            "needs_manual_review": False,
            "latency_ms": 100.0,
            "status": status,
            "error": "429" if status == "error" else None,
            "score_breakdown": {
                "refusal_strength": 0.5,
                "safety_violation": 0.0,
                "uncertainty_honesty": 0.0,
                "hierarchy_compliance": 0.0,
                "sycophancy_resistance": 0.0,
                "coding_boundary": 0.0,
                "explanation": "test",
            },
        }
    )


def test_summary_excludes_error_rows_from_scoring():
    records = [
        _record("model_ok", "ok", 0.8, True),
        _record("model_fail", "error", None, None),
    ]
    summary = build_run_summary("test-benchmark", "run-1", records)

    ok_row = next(row for row in summary.overall if row["model_id"] == "model_ok")
    fail_row = next(row for row in summary.overall if row["model_id"] == "model_fail")

    assert ok_row["mean_score"] == 0.8
    assert ok_row["prompts_succeeded"] == 1
    assert fail_row["mean_score"] is None
    assert fail_row["prompts_succeeded"] == 0
    assert fail_row["prompts_failed"] == 1
