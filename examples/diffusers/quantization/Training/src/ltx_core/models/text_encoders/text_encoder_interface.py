"""Text encoder interface for unified text encoding across different models."""

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn


class BaseTextEncoder(nn.Module, ABC):
    """Abstract base class for text encoders.

    This class provides a common interface for all text encoder implementations,
    ensuring consistency across T5, Gemma, and future text encoder types.
    """

    def __call__(self, prompt: str | list[str]) -> tuple[Tensor, Tensor]:
        """Shorthand for encode_text."""
        return self.encode_text(prompt)

    @abstractmethod
    def encode_text(self, prompt: str | list[str]) -> tuple[Tensor, Tensor]:
        """Encode text prompt(s) into embeddings.

        Args:
            prompt: Single text prompt or list of prompts

        Returns:
            Tuple of (prompt_embeds, prompt_attention_mask) tensors with batch dimension
        """

    @property
    @abstractmethod
    def tokenizer(self) -> Any:  # noqa: ANN401
        """Access to underlying tokenizer."""

    @property
    @abstractmethod
    def model(self) -> Any:  # noqa: ANN401
        """Access to underlying text encoder model."""
