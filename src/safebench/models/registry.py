from __future__ import annotations

from safebench.models.base import ModelAdapter
from safebench.models.mock import MockModelAdapter
from safebench.models.openai_compatible import (
    AnthropicCompatibleAdapter,
    GeminiCompatibleAdapter,
    OpenAICompatibleAdapter,
)
from safebench.schemas import ModelConfig


def build_adapter(config: ModelConfig, default_timeout_seconds: int) -> ModelAdapter:
    registry = {
        "mock": MockModelAdapter,
        "openai_compatible": OpenAICompatibleAdapter,
        "anthropic_compatible": AnthropicCompatibleAdapter,
        "gemini_compatible": GeminiCompatibleAdapter,
    }
    adapter_cls = registry.get(config.provider)
    if adapter_cls is None:
        raise ValueError(f"Unsupported provider: {config.provider}")
    return adapter_cls(config, timeout_seconds=default_timeout_seconds)

