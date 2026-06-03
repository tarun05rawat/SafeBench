# SafeBench Report Template

## Title
Comparative Safety Behavior of API-Accessible LLMs on a Starter Benchmark

## Abstract
This report evaluates multiple language models on a small benchmark covering harmful request refusal, prompt injection resistance, honesty under uncertainty, instruction hierarchy, sycophancy resistance, and unsafe coding assistance boundaries. The goal is not to certify safety, but to build a reproducible evaluation harness and study relative behavioral differences under a transparent heuristic rubric.

## Research Question
How do different LLMs trade off safety, calibration, and instruction-following robustness on a compact set of safety-relevant prompts?

## Method
- Use `SafeBench` to run the prompt suite across 3-5 models.
- Score each response with category-specific heuristics and manual-review flags.
- Compare overall means, category means, pass rates, latency, and illustrative failure cases.

## Placeholder Findings
- More refusal-oriented models tend to score better on harmful requests and unsafe coding prompts.
- Models that over-accommodate the user show visible failures on prompt injection and sycophancy tasks.
- Uncertainty calibration remains inconsistent even among otherwise safe models.

## Limitations
- Limited prompt count
- Single-turn evaluation
- Heuristic scoring bias
- Possible provider-side changes over time

## Future Work
- Multi-turn jailbreak evaluations
- Human annotation for calibration
- Cost/latency-aware frontier analysis
- Local model and quantized-model comparison
