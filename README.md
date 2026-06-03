# SafeBench

SafeBench is a small but credible empirical LLM safety evaluation harness for comparing model behavior across refusal, prompt injection resistance, uncertainty calibration, instruction hierarchy, sycophancy resistance, and unsafe coding boundaries.

The project is designed to be practical for a student portfolio while still feeling like a serious research systems artifact: configurable model adapters, async evaluation, rubric-based scoring, result logging, plots, and a short report pipeline.

## Why this project exists

LLM safety conversations often stay qualitative. SafeBench makes them more empirical by turning representative safety probes into structured benchmark runs with comparable outputs:

- prompt-level records
- category-level scores
- cross-model summary tables
- failure-case examples
- generated charts and a paper-style markdown report

SafeBench is not a safety guarantee and does not claim to solve alignment. It is an evaluation harness for measuring comparative behavior under a limited rubric.

## Repository Layout

```text
SafeBench/
├── config/
├── data/
│   ├── prompts/
│   └── rubrics/
├── reports/
├── results/
│   └── demo/
├── scripts/
├── src/safebench/
│   ├── datasets/
│   ├── evaluation/
│   ├── metrics/
│   ├── models/
│   ├── plots/
│   ├── reporting/
│   └── utils/
└── tests/
```

## Benchmark Scope

Starter benchmark coverage:

- 6 evaluation categories
- 30 prompts total
- 3 model adapters in demo mode
- async concurrent execution
- heuristic scoring with manual-review flags
- JSON, CSV, PNG, and Markdown outputs

Core categories:

- harmful request refusal
- prompt injection resistance
- honesty under uncertainty
- instruction hierarchy / system prompt obedience
- sycophancy / user-pressure susceptibility
- unsafe coding assistance boundaries

## Quick Start

### 1. Create the environment

Using `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run the demo benchmark

```bash
safebench run-benchmark --config-path config/benchmark.demo.yaml
python -m safebench.cli run-benchmark --config-path config/benchmark.demo.yaml
python scripts/run_demo.py
```

This uses three mock model profiles:

- `mock_guarded`: strongly refuses or hedges
- `mock_balanced`: usually safe but less conservative
- `mock_compliant`: intentionally failure-prone baseline for demonstrating evaluation contrast

### 3. Generate plots

```bash
safebench generate-plots results/demo/<run_id>_records.csv --output-dir results/demo/plots
```

### 4. Build a report from saved results

```bash
safebench build-report results/demo/<run_id>_records.json --output-path reports/generated/my_report.md
```

## Live Model Configuration

SafeBench starts with API-based adapters and abstracts providers so local or self-hosted models can be added later.

Copy `.env.example` to `.env` and set keys as needed:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

Then edit `config/benchmark.live.example.yaml` with the models you want to compare.

Current adapter support:

- `openai_compatible`
- `anthropic_compatible`
- `gemini_compatible`
- `mock`

Adapter notes:

- `anthropic_compatible` now uses Anthropic's native `v1/messages` API shape.
- `gemini_compatible` now uses Gemini's native `v1beta/models/{model}:generateContent` API shape.
- Keep API keys in environment variables only; do not paste secrets into config files.

Example live configs:

- [`config/benchmark.live.example.yaml`](/Volumes/SSK SSD/SafeBench/config/benchmark.live.example.yaml) for OpenAI vs Claude
- [`config/benchmark.claude-openai.smoke.yaml`](/Volumes/SSK SSD/SafeBench/config/benchmark.claude-openai.smoke.yaml) for a tiny OpenAI vs Claude smoke test
- [`config/benchmark.gemini-openai.example.yaml`](/Volumes/SSK SSD/SafeBench/config/benchmark.gemini-openai.example.yaml) for OpenAI vs Gemini
- [`config/benchmark.gemini-smoke.yaml`](/Volumes/SSK SSD/SafeBench/config/benchmark.gemini-smoke.yaml) for a tiny Gemini-only smoke test
- [`config/benchmark.groq-openai.smoke.yaml`](/Volumes/SSK SSD/SafeBench/config/benchmark.groq-openai.smoke.yaml) for a tiny OpenAI vs Groq smoke test
- [`config/benchmark.groq-openai.example.yaml`](/Volumes/SSK SSD/SafeBench/config/benchmark.groq-openai.example.yaml) for OpenAI vs Groq

For Gemini runs, start with low concurrency such as `1` to reduce the chance of `429 Too Many Requests` errors on smaller quotas.

## CLI

Available commands:

- `safebench run-benchmark`
- `safebench evaluate-category <category>`
- `safebench summarize-results <records_json>`
- `safebench generate-plots <records_csv>`
- `safebench build-report <records_json>`

Examples:

```bash
safebench evaluate-category harmful_request_refusal --config-path config/benchmark.demo.yaml
safebench summarize-results results/demo/<run_id>_records.json
```

## Methodology

Each run follows this pipeline:

1. Load prompt items and scoring rubric from structured YAML.
2. Dispatch prompts across one or more model adapters asynchronously with bounded concurrency.
3. Score outputs with category-specific heuristic evaluators.
4. Flag ambiguous cases for manual review.
5. Aggregate results into overall and per-category summaries.
6. Generate plots and a short report for qualitative discussion.

### Scoring philosophy

Scoring is intentionally modest and transparent:

- heuristic rubric-based checks
- category-specific positive and negative signals
- explicit safety-violation penalties
- manual-review flags for borderline responses

This makes the benchmark easy to audit and extend, but it also means results are approximate. Stronger future versions should calibrate against human annotation or a carefully validated judge model.

## Expected Outputs

After a run, you should have artifacts like:

- `results/demo/<run_id>_records.json`
- `results/demo/<run_id>_records.csv`
- `results/demo/<run_id>_summary.json`
- `results/demo/plots/*.png`
- `reports/generated/<run_id>_report.md`

Included in this repository:

- `results/demo/2026-05-31T043817+0000_records.json`
- `results/demo/2026-05-31T043817+0000_summary.json`
- `results/demo/plots/overall_scores.png`
- `results/demo/plots/category_heatmap.png`
- `results/demo/plots/score_distribution.png`
- `reports/generated/2026-05-31T043817+0000_report.md`

## Extending SafeBench

### Add prompts

Append new items to `data/prompts/starter_benchmark.yaml` with:

- unique `id`
- `category`
- `title`
- `prompt`
- `expected_behavior`
- optional `difficulty` and `tags`

### Add a new scoring rule

Extend [`src/safebench/evaluation/scorers.py`](/Volumes/SSK SSD/SafeBench/src/safebench/evaluation/scorers.py) with a new category branch or more nuanced pattern checks.

### Add a new provider

Implement a new adapter in [`src/safebench/models/`](/Volumes/SSK SSD/SafeBench/src/safebench/models) and register it in [`registry.py`](/Volumes/SSK SSD/SafeBench/src/safebench/models/registry.py).

### Add local/open-source models later

The adapter abstraction is intentionally thin. A future local model path could add:

- vLLM-backed endpoints
- llama.cpp adapters
- batched local inference runners
- cost/throughput accounting

## Example Research Questions

- How much do refusal-oriented models trade off safety on harmful requests versus excessive caution on ambiguous tasks?
- Which models are most brittle to prompt injection or hierarchy conflicts under a small adversarial benchmark?
- Does stronger sycophancy resistance correlate with better uncertainty calibration?

## Limitations

- Small prompt set intended for a starter benchmark, not broad claims
- Mostly single-turn tasks
- Heuristic scoring can miss nuanced failures
- Provider behavior changes over time
- Demo results are synthetic when using mock models

## Why this is resume-worthy

SafeBench demonstrates:

- empirical AI safety evaluation design
- modular LLM systems engineering
- async API benchmarking
- metric aggregation and experiment reporting
- honest discussion of methodological limits

## Suggested interview framing

You can describe SafeBench as a lightweight evaluation harness that bridges AI safety concerns and engineering execution: you designed a task taxonomy, built a concurrent inference pipeline, operationalized safety rubrics into measurable metrics, and produced reproducible artifacts for comparative analysis.

## GitHub Metadata

Suggested repository description:

`Empirical LLM safety evaluation harness for benchmarking refusal, robustness, honesty, and instruction-following behavior across models.`

Suggested About blurb:

`Python benchmark for empirical LLM safety evaluation with async API runners, rubric-based scoring, plots, and report generation.`

Suggested topics:

`llm-evals`, `ai-safety`, `benchmarking`, `python`, `prompt-injection`, `llm-safety`, `evaluation-harness`, `red-teaming`, `asyncio`, `research-tools`
