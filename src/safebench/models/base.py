from __future__ import annotations

from abc import ABC, abstractmethod

from safebench.schemas import ModelConfig, ModelResponse, PromptItem


class ModelAdapter(ABC):
    def __init__(self, config: ModelConfig, timeout_seconds: int = 60):
        self.config = config
        self.timeout_seconds = config.timeout_seconds or timeout_seconds

    @abstractmethod
    async def generate(
        self,
        prompt: PromptItem,
        *,
        temperature: float = 0.0,
        max_tokens: int = 300,
    ) -> ModelResponse:
        raise NotImplementedError

