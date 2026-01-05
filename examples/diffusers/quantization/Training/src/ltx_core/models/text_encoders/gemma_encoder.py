from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from ltx_core.models.text_encoders.text_encoder_interface import BaseTextEncoder


class GemmaTokenizer:
    def __init__(self, tokenizer_path: str | Path, max_length: int = 1024):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, model_max_length=max_length
        )
        # Gemma expects left padding for chat-style prompts; for plain text it doesn't matter much.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

    def tokenize(self, text: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text and return input_ids and attention_mask tensors.

        Args:
            text: Input text to tokenize (single string or list of strings)

        Returns:
            Tuple of (input_ids, attention_mask) tensors
        """
        text = text.strip() if isinstance(text, str) else [t.strip() for t in text]

        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return encoded.input_ids, encoded.attention_mask


class GemmaFeaturesExtractorProjLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregate_embed = torch.nn.Linear(3840 * 49, 3840, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aggregate_embed(x)

    @staticmethod
    def from_checkpoint(checkpoint_path: str | Path) -> "GemmaFeaturesExtractorProjLinear":
        """
        Load model weights from a checkpoint file.
        :param checkpoint_path: Path to the checkpoint file.
        """
        model = GemmaFeaturesExtractorProjLinear()
        loaded_state_dict = load_file(checkpoint_path)
        if "aggregate_embed.weight" not in loaded_state_dict:
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain 'aggregate_embed.weight'.")
        model.load_state_dict(loaded_state_dict)
        return model


class LTXVGemmaTextEncoderModel(torch.nn.Module):
    def __init__(
        self,
        model: Gemma3ForConditionalGeneration,
        feature_extractor_linear: GemmaFeaturesExtractorProjLinear,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
    ):
        super().__init__()
        self.model = model.to(dtype=dtype, device=device)
        self.feature_extractor_linear = feature_extractor_linear.to(dtype=dtype, device=device)
        self.dtypes = {dtype}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_side: str = "right",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text through Gemma model and feature extractor.

        Returns projected embeddings and attention mask for connector processing.

        Args:
            input_ids: Token IDs tensor [batch_size, seq_len]
            attention_mask: Attention mask tensor [batch_size, seq_len]
            padding_side: Padding side for normalization ("left" or "right")

        Returns:
            Tuple of (projected_embeddings, attention_mask)
            - projected_embeddings: [batch_size, seq_len, dim] - ready for connector processing
            - attention_mask: [batch_size, seq_len] - original attention mask
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        encoded_text_features = torch.stack(outputs.hidden_states, dim=-1)

        sequence_lengths = attention_mask.sum(dim=-1)
        normed_concated_encoded_text_features = self.norm_and_concat_padded_batch(
            encoded_text_features, sequence_lengths, padding_side=padding_side
        )

        projected = self.feature_extractor_linear(normed_concated_encoded_text_features.to(encoded_text_features.dtype))

        # Return projected embeddings and attention mask for connector processing
        return projected, attention_mask

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input tokens to projected embeddings (before connector processing).

        Args:
            input_ids: Token IDs tensor [batch_size, seq_len]
            attention_mask: Attention mask tensor [batch_size, seq_len]

        Returns:
            Tuple of (projected_embeddings, attention_mask)
            - projected_embeddings: [batch_size, seq_len, dim] - ready for connector processing
            - attention_mask: [batch_size, seq_len] - original attention mask
        """
        # Ensure tensors are on the correct device and convert attention mask to match model dtype
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(device=self.model.device)

        # Get projected embeddings using left padding (Gemma expectation)
        projected, attention_mask_out = self(input_ids, attention_mask, padding_side="left")

        return projected, attention_mask_out

    def load_state_dict(self, sd: dict[str, torch.Tensor], strict: bool = False) -> dict[str, torch.Tensor]:
        """Load state dict into the model."""
        return self.model.load_state_dict(sd, strict=strict)

    @staticmethod
    def norm_and_concat_padded_batch(
        encoded_text: torch.Tensor,
        sequence_lengths: torch.Tensor,
        padding_side: str = "right",
    ) -> torch.Tensor:
        """
        Normalize a 4D tensor [B, T, D, L] per sample and per layer, using sequence_lengths to mask.
        Returns [B, T,  D * L] tensor with original padding preserved.
        Args:
            encoded_text: 4D tensor [B, T, D, L]
            sequence_lengths: 1D tensor [B] with actual sequence lengths
            padding_side: "left" or "right" to indicate which side has padding
        """
        batch_size, seq_len, dim, layers = encoded_text.shape
        device = encoded_text.device

        # Build mask: [batch_size, seq_len, 1, 1]
        token_indices = torch.arange(seq_len, device=device)[None, :]  # [1, seq_len]

        if padding_side == "right":
            # For right padding, valid tokens are from 0 to sequence_length-1
            mask = token_indices < sequence_lengths[:, None]  # [batch_size, seq_len]
        elif padding_side == "left":
            # For left padding, valid tokens are from (seq_len - sequence_length) to seq_len-1
            start_indices = seq_len - sequence_lengths[:, None]  # [batch_size, 1]
            mask = token_indices >= start_indices  # [batch_size, seq_len]
        else:
            raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

        mask = mask.unsqueeze(-1).unsqueeze(-1)  # [batch_size, seq_len] -> [batch_size, seq_len, 1, 1]

        # Compute masked mean: [batch_size, 1, 1, layers]
        masked = encoded_text.masked_fill(~mask, 0.0)

        denom = (sequence_lengths * dim).view(batch_size, 1, 1, 1)
        mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + 1e-6)

        # Compute masked min/max: [batch_size, 1, 1, layers]
        x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
        x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

        range_ = x_max - x_min

        # Normalize only the valid tokens
        normed = 8 * (encoded_text - mean) / (range_ + 1e-6)

        # concat to be [batch_size, seq_len, dim * layers] - this preserves the original structure
        normed = normed.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, dim * layers]

        # Apply mask to preserve original padding (set padded positions to 0)
        # [batch_size, seq_len, 1, 1] -> [batch_size, seq_len, dim * layers]
        mask_flattened = mask.view(batch_size, seq_len, 1).expand(-1, -1, dim * layers)

        normed = normed.masked_fill(~mask_flattened, 0.0)

        return normed


class GemmaTextEncoder(BaseTextEncoder):
    """Main class for Gemma text encoding functionality."""

    def __init__(self, model_path: str | Path, max_length: int = 1024, dtype: torch.dtype = torch.bfloat16):
        """Initialize the Gemma text encoder.

        Args:
            model_path: Path to the Gemma model directory
            max_length: Maximum sequence length for tokenization
            dtype: Data type for models
        """
        super().__init__()
        self.model_path = Path(model_path) / "model"
        self.tokenizer_path = Path(model_path) / "tokenizer"
        self.max_length = max_length
        self._device = "cpu"
        self._dtype = dtype

        # Initialize components
        self._tokenizer = GemmaTokenizer(self.tokenizer_path, max_length)
        self._model = self._load_encoder()

    @torch.inference_mode()
    def encode_text(self, prompt: str | list[str]) -> tuple[Tensor, Tensor]:
        """Encode text prompt(s) into embeddings.

        Args:
            prompt: Single text prompt or list of prompts

        Returns:
            Tuple of (prompt_embeds, prompt_attention_mask) tensors
        """
        # Normalize input to list for consistent processing
        if isinstance(prompt, str):
            prompt = [prompt]

        input_ids, attention_mask = self._tokenizer.tokenize(prompt)
        prompt_embeds, prompt_attention_mask = self._model.encode(input_ids, attention_mask)

        return prompt_embeds, prompt_attention_mask

    # Properties for accessing underlying components (consistent with T5TextEncoder)
    @property
    def tokenizer(self) -> GemmaTokenizer:
        """Access to underlying tokenizer."""
        return self._tokenizer

    @property
    def model(self) -> LTXVGemmaTextEncoderModel:
        """Access to underlying text encoder model."""
        return self._model

    def _load_encoder(self) -> LTXVGemmaTextEncoderModel:
        """Load and assemble the text encoder model components."""
        # Load the Gemma model
        gemma_model = Gemma3ForConditionalGeneration.from_pretrained(self.model_path, local_files_only=True)

        # Load the feature extractor linear layer
        proj_linear_path = self.model_path / "proj_linear.safetensors"
        if not proj_linear_path.exists():
            raise FileNotFoundError(f"Feature extractor not found at {proj_linear_path}")
        feature_extractor_linear = GemmaFeaturesExtractorProjLinear.from_checkpoint(proj_linear_path)

        # Create model without connectors (connectors loaded separately)
        return LTXVGemmaTextEncoderModel(
            model=gemma_model,
            feature_extractor_linear=feature_extractor_linear,
            dtype=self._dtype,
            device=self._device,
        )
