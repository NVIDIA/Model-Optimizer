# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Amit Pintz.

from pathlib import Path
from typing import Self

import torch
from einops import rearrange
from transformers import Gemma3ForConditionalGeneration

from ltx_core.model.clip.gemma.feature_extractor import (
    GemmaFeaturesExtractorProjLinear,
)
from ltx_core.model.clip.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_core.model.model_protocol import ModelConfigurator


class GemmaTextEncoderModelBase(torch.nn.Module):
    """
    Gemma Text Encoder Model.

    This base class combines the tokenizer, Gemma model and feature extractor to provide a preprocessing
    for implementation classes for multimodal pipelines. It processes input text through tokenization,
    obtains hidden states from the base language model, applies a linear feature extractor.

    Args:
        tokenizer (LTXVGemmaTokenizer): The tokenizer used for text preprocessing.
        model (Gemma3ForConditionalGeneration): The base Gemma LLM.
        feature_extractor_linear (GemmaFeaturesExtractorProjLinear): Linear projection for hidden state aggregation.
        dtype (torch.dtype, optional): The data type for model parameters (default: torch.bfloat16).
    """

    def __init__(
        self,
        tokenizer: LTXVGemmaTokenizer,
        model: Gemma3ForConditionalGeneration,
        feature_extractor_linear: GemmaFeaturesExtractorProjLinear,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.feature_extractor_linear = feature_extractor_linear.to(dtype=dtype)

    def _run_feature_extractor(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, padding_side: str = "right"
    ) -> torch.Tensor:
        encoded_text_features = torch.stack(hidden_states, dim=-1)
        encoded_text_features_dtype = encoded_text_features.dtype

        sequence_lengths = attention_mask.sum(dim=-1)
        normed_concated_encoded_text_features = _norm_and_concat_padded_batch(
            encoded_text_features, sequence_lengths, padding_side=padding_side
        )

        return self.feature_extractor_linear(normed_concated_encoded_text_features.to(encoded_text_features_dtype))

    def _convert_to_additive_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (attention_mask - 1).to(dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(dtype).max

    def _preprocess_text(self, text: str, padding_side: str = "right") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Encode a given string into feature tensors suitable for downstream tasks.

        Args:
            text (str): Input string to encode.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: Encoded features and a dictionary with attention mask.
        """
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=self.model.device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=self.model.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        projected = self._run_feature_extractor(
            hidden_states=outputs.hidden_states, attention_mask=attention_mask, padding_side=padding_side
        )
        return projected, attention_mask

    def forward(self, text: str, padding_side: str = "left") -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This method is not implemented for the base class")


def _norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    """Normalize and flatten multi-layer hidden states, respecting padding.

    Performs per-batch, per-layer normalization using masked mean and range,
    then concatenates across the layer dimension.

    Args:
        encoded_text: Hidden states of shape [batch, seq_len, hidden_dim, num_layers].
        sequence_lengths: Number of valid (non-padded) tokens per batch item.
        padding_side: Whether padding is on "left" or "right".

    Returns:
        Normalized tensor of shape [batch, seq_len, hidden_dim * num_layers],
        with padded positions zeroed out.
    """
    b, t, d, l = encoded_text.shape  # noqa: E741
    device = encoded_text.device

    # Build mask: [B, T, 1, 1]
    token_indices = torch.arange(t, device=device)[None, :]  # [1, T]

    if padding_side == "right":
        # For right padding, valid tokens are from 0 to sequence_length-1
        mask = token_indices < sequence_lengths[:, None]  # [B, T]
    elif padding_side == "left":
        # For left padding, valid tokens are from (T - sequence_length) to T-1
        start_indices = t - sequence_lengths[:, None]  # [B, 1]
        mask = token_indices >= start_indices  # [B, T]
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    mask = rearrange(mask, "b t -> b t 1 1")

    eps = 1e-6

    # Compute masked mean: [B, 1, 1, L]
    masked = encoded_text.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * d).view(b, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)

    # Compute masked min/max: [B, 1, 1, L]
    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    range_ = x_max - x_min

    # Normalize only the valid tokens
    normed = 8 * (encoded_text - mean) / (range_ + eps)

    # concat to be [Batch, T,  D * L] - this preserves the original structure
    normed = normed.reshape(b, t, -1)  # [B, T, D * L]

    # Apply mask to preserve original padding (set padded positions to 0)
    mask_flattened = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, d * l)
    normed = normed.masked_fill(~mask_flattened, 0.0)

    return normed


class GemmaTextEncoderModelConfiguratorBase(ModelConfigurator[GemmaTextEncoderModelBase]):
    gemma: torch.nn.Module
    tokenizer: LTXVGemmaTokenizer

    @classmethod
    def with_gemma_model(
        cls: type[Self], gemma_model: Gemma3ForConditionalGeneration, tokenizer: LTXVGemmaTokenizer
    ) -> type[Self]:
        name = f"{cls.__name__}_With_Gemma"
        return type(
            name,
            (cls,),
            {
                "gemma": gemma_model,
                "tokenizer": tokenizer,
            },
        )

    @classmethod
    def with_gemma_root_path(
        cls: type[Self],
        gemma_root_path: str,
    ) -> type[Self]:
        gemma_path = str((Path(gemma_root_path) / "model").absolute())
        gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_path, local_files_only=True, dtype=torch.bfloat16
        )
        tokenizer_path = str((Path(gemma_root_path) / "model").absolute())
        tokenizer = LTXVGemmaTokenizer(tokenizer_path, 1024)
        return cls.with_gemma_model(gemma_model, tokenizer)

    @classmethod
    def from_config(cls: type[Self], config: dict) -> Self:
        raise NotImplementedError("This method is not implemented for the base class")
