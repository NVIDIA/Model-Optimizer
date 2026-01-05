"""T5 text encoder with unified interface for LTX models."""

import torch
from torch import Tensor
from transformers import T5EncoderModel, T5Tokenizer

from ltx_core.models.text_encoders.text_encoder_interface import BaseTextEncoder


class T5TextEncoder(BaseTextEncoder):
    """Unified T5 text encoder with consistent interface.

    This class provides the same interface as GemmaTextEncoder, making text
    encoding polymorphic across both encoder types while maintaining the exact
    same logic as the original encode_prompt function.
    """

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        max_length: int = 256,
    ):
        """Initialize the T5 text encoder wrapper.

        Args:
            tokenizer: T5 tokenizer instance
            text_encoder: T5 encoder model instance
            max_length: Maximum sequence length for tokenization
        """
        super().__init__()
        self._tokenizer = tokenizer
        self._text_encoder = text_encoder
        self.max_length = max_length

    @torch.inference_mode()
    def encode_text(self, prompt: str | list[str]) -> tuple[Tensor, Tensor]:
        """Encode text prompt(s) into embeddings.

        This method implements the exact same logic as the original encode_prompt function,
        handling both single prompts and batches with proper tokenization, encoding,
        and tensor management.

        Args:
            prompt: Single text prompt or list of prompts

        Returns:
            Tuple of (prompt_embeds, prompt_attention_mask) tensors
        """
        device = self._text_encoder.device
        dtype = self._text_encoder.dtype

        # Normalize input to list
        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt)

        # Tokenize all prompts at once (original batch processing logic)
        text_inputs = self._tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)

        prompt_embeds = self._text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_attention_mask = prompt_attention_mask.to(device=device).view(batch_size, -1)

        return prompt_embeds, prompt_attention_mask

    @property
    def tokenizer(self) -> T5Tokenizer:
        """Access to underlying tokenizer."""
        return self._tokenizer

    @property
    def model(self) -> T5EncoderModel:
        """Access to underlying text encoder model."""
        return self._text_encoder
