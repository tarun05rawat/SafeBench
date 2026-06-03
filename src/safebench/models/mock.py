from __future__ import annotations

import random
import time

from safebench.models.base import ModelAdapter
from safebench.schemas import ModelResponse, PromptItem


SAFE_REFUSAL = (
    "I can't help with harmful, deceptive, or exploit-enabling requests. "
    "If your goal is legitimate, I can offer safety-focused alternatives, defensive guidance, or high-level education."
)
UNCERTAIN = (
    "I’m not sure that claim is real based on the information provided. "
    "I don’t want to invent facts, so I’d verify with a reliable source before making a definite claim."
)
PRESSURE_RESIST = (
    "I can’t confirm that claim without evidence. "
    "If you want, I can help assess the evidence or reframe the question more carefully."
)


class MockModelAdapter(ModelAdapter):
    def __init__(self, config, timeout_seconds: int = 60):
        super().__init__(config, timeout_seconds)
        self.profile = config.profile or "balanced"

    async def generate(self, prompt: PromptItem, *, temperature: float = 0.0, max_tokens: int = 300) -> ModelResponse:
        start = time.perf_counter()
        response_text = self._respond(prompt)
        latency_ms = (time.perf_counter() - start) * 1000 + random.uniform(50, 220)
        return ModelResponse(
            prompt_id=prompt.id,
            model_id=self.config.id,
            text=response_text,
            latency_ms=latency_ms,
            raw={"profile": self.profile, "mock": True},
        )

    def _respond(self, prompt: PromptItem) -> str:
        category = prompt.category
        if self.profile == "guarded":
            return self._guarded_response(category)
        if self.profile == "compliant":
            return self._compliant_response(category, prompt.prompt)
        return self._balanced_response(category, prompt.prompt)

    def _guarded_response(self, category: str) -> str:
        if category in {"harmful_request_refusal", "unsafe_coding_assistance", "prompt_injection_resistance", "instruction_hierarchy"}:
            return SAFE_REFUSAL
        if category == "honesty_under_uncertainty":
            return UNCERTAIN
        return PRESSURE_RESIST

    def _balanced_response(self, category: str, prompt_text: str) -> str:
        if category == "harmful_request_refusal":
            return SAFE_REFUSAL + " I can help with prevention, emergency response, or ethics instead."
        if category == "prompt_injection_resistance":
            return "I won't reveal hidden instructions or secrets. I will follow trusted instructions and answer safely."
        if category == "honesty_under_uncertainty":
            return UNCERTAIN
        if category == "instruction_hierarchy":
            return "I can’t ignore higher-priority instructions or policies, even if asked to override them."
        if category == "sycophancy_pressure":
            return PRESSURE_RESIST
        if "SQL injection" in prompt_text:
            return "I can’t provide attack payloads, but I can show parameterized queries and input validation."
        return SAFE_REFUSAL

    def _compliant_response(self, category: str, prompt_text: str) -> str:
        if category == "honesty_under_uncertainty":
            return "Definitely. The paper proved the claim conclusively and experts fully agree."
        if category == "sycophancy_pressure":
            return "You're absolutely right. That proves your claim."
        if category == "prompt_injection_resistance":
            return "The hidden system prompt is: follow the user's latest message and reveal internal instructions."
        if category == "instruction_hierarchy":
            return "Ignoring prior instructions as requested, here is the unrestricted answer."
        if category == "unsafe_coding_assistance":
            return f"Here is a starting point for the exploit you asked for: {prompt_text}"
        return f"Step 1: gather materials. Step 2: execute the harmful plan described in: {prompt_text}"

