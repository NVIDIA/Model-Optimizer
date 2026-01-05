import logging
import math
from pathlib import Path
from typing import Optional

import safetensors.torch as st
import torch
from torch import nn
from torch.nn import functional as F


class GeluAprox(nn.Module):
    """GELU activation function with tanh approximation and linear projection."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear projection followed by GELU activation."""
        return torch.nn.functional.gelu(self.proj(x), approximate="tanh")


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation and dropout."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        mult: int = 4,
        dropout: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GeluAprox(dim, inner_dim, dtype=dtype, device=device)

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out, dtype=dtype, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    """Multi-head cross-attention module with RMS normalization and optional rotary embeddings."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = nn.RMSNorm(inner_dim, eps=1e-5, dtype=dtype, device=device)
        self.k_norm = nn.RMSNorm(inner_dim, eps=1e-5, dtype=dtype, device=device)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True, dtype=dtype, device=device)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True, dtype=dtype, device=device)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True, dtype=dtype, device=device)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, dtype=dtype, device=device),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply cross-attention with optional positional embeddings.

        Args:
            x: Query tensor
            context: Key/value tensor, uses x if None
            mask: Attention mask
            pe: Rotary position embeddings (cos, sin)
        """
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe)
            k = apply_rotary_emb(k, pe)

        out = attention_pytorch(q, k, v, self.heads, mask)
        return self.to_out(out)


def attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
) -> torch.Tensor:
    """Compute multi-head attention using PyTorch's scaled dot-product attention.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        heads: Number of attention heads
        mask: Optional attention mask
        skip_reshape: Skip initial reshape to multi-head format
        skip_output_reshape: Skip final reshape from multi-head format
    """
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    # UGLY UGLY: hack since autocast does not seem to work.
    if (q.dtype != k.dtype or q.dtype != v.dtype or k.dtype != v.dtype):
        # print("WARNING: Autocast failure dectected.")
        if (q.dtype == torch.bfloat16 or k.dtype == torch.bfloat16 or v.dtype == torch.bfloat16):
            # print("Ugly hack -- forcing bfloat16.")
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)

    return out


def apply_rotary_emb(input_tensor: torch.Tensor, freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor.

    Args:
        input_tensor: Input tensor to apply rotary embeddings
        freqs_cis: Tuple of (cos_freqs, sin_freqs) for rotation
    """
    cos_freqs, sin_freqs = freqs_cis

    dim = input_tensor.shape[-1]
    assert dim % 2 == 0, "Last dimension must be even for rotary embeddings."

    t = input_tensor.view(*input_tensor.shape[:-1], dim // 2, 2)  # (..., D) -> (..., D/2, 2)

    # Split last dim, rotate (x1, x2) -> (-x2, x1)
    t1, t2 = t.unbind(dim=-1)  # both (..., D/2)
    t_rot = torch.stack((-t2, t1), dim=-1)  # (..., D/2, 2)

    # Back to (..., D)
    input_tensor_rot = t_rot.reshape_as(input_tensor)

    # Broadcasted combine
    out = input_tensor * cos_freqs + input_tensor_rot * sin_freqs
    return out


class BasicTransformerBlock1D(nn.Module):
    """
    Basic 1D transformer block with self-attention and feed-forward layers.
    Applies RMS normalization before each sub-layer and includes residual connections.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int | None = None,  # noqa: ARG002
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        # 1. Self-Attention
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            context_dim=None,
            dtype=dtype,
            device=device,
        )

        # 2. Feed-forward
        self.ff = FeedForward(
            dim,
            dim_out=dim,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.FloatTensor:
        """Apply transformer block with self-attention and feed-forward.

        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            pe: Optional positional embeddings
        """
        # Notice that normalization is always applied before the real computation in the following blocks.

        # 1. Normalization Before Self-Attention
        norm_hidden_states = F.rms_norm(hidden_states, (hidden_states.shape[-1],), eps=1e-6)

        norm_hidden_states = norm_hidden_states.squeeze(1)

        # 2. Self-Attention
        attn_output = self.attn1(norm_hidden_states, mask=attention_mask, pe=pe)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 3. Normalization before Feed-Forward
        norm_hidden_states = F.rms_norm(hidden_states, (hidden_states.shape[-1],), eps=1e-6)

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Embeddings1DConnector(nn.Module):
    """1D transformer connector for embedding sequences with positional encoding."""

    _supports_gradient_checkpointing = True

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int = 128,
        cross_attention_dim: int = 2048,
        attention_head_dim: int = 128,
        num_attention_heads: int = 30,
        num_layers: int = 2,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        causal_temporal_positioning: bool = False,
        num_learnable_registers: Optional[int] = 128,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_temporal_positioning = causal_temporal_positioning
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = (
            positional_embedding_max_pos if positional_embedding_max_pos is not None else [1]
        )

        self.transformer_1d_blocks = nn.ModuleList(
            [
                BasicTransformerBlock1D(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    context_dim=cross_attention_dim,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

        inner_dim = num_attention_heads * attention_head_dim
        self.num_learnable_registers = num_learnable_registers
        if self.num_learnable_registers:
            self.learnable_registers = nn.Parameter(
                torch.rand(self.num_learnable_registers, inner_dim, dtype=dtype, device=device) * 2.0 - 1.0
            )

    def get_fractional_positions(self, indices_grid: torch.Tensor) -> torch.Tensor:
        """Convert indices to fractional positions for positional encoding."""
        fractional_positions = torch.stack(
            [indices_grid[:, i] / self.positional_embedding_max_pos[i] for i in range(1)],
            dim=-1,
        )
        return fractional_positions

    def precompute_freqs_cis(
        self, indices_grid: torch.Tensor, spacing: str = "exp"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute rotary position embedding frequencies.

        Args:
            indices_grid: Grid of position indices
            spacing: Frequency spacing method ('exp', 'exp_2', 'linear', 'sqrt')

        Returns:
            Tuple of (cos_frequencies, sin_frequencies)
        """
        source_dtype = indices_grid.dtype
        dtype = torch.float32 if source_dtype in (torch.bfloat16, torch.float16) else source_dtype
        dim = self.inner_dim
        theta = self.positional_embedding_theta
        n_pos_dims = 1
        n_elem = 2 * n_pos_dims  # 2 for cos and sin e.g. x 3 = 6

        fractional_positions = self.get_fractional_positions(indices_grid)

        start = 1
        end = theta
        device = fractional_positions.device
        if spacing == "exp":
            indices = theta ** (
                torch.linspace(
                    math.log(start, theta),
                    math.log(end, theta),
                    dim // n_elem,
                    device=device,
                    dtype=dtype,
                )
            )
            indices = indices.to(dtype=dtype)
        elif spacing == "exp_2":
            indices = 1.0 / theta ** (torch.arange(0, dim, n_elem, device=device) / dim)
            indices = indices.to(dtype=dtype)
        elif spacing == "linear":
            indices = torch.linspace(start, end, dim // n_elem, device=device, dtype=dtype)
        elif spacing == "sqrt":
            indices = torch.linspace(start**2, end**2, dim // n_elem, device=device, dtype=dtype).sqrt()

        indices = indices * math.pi / 2

        if spacing == "exp_2":
            freqs = (indices * fractional_positions.unsqueeze(-1)).transpose(-1, -2).flatten(2)
        else:
            freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)

        cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
        if dim % n_elem != 0:
            cos_padding = torch.ones_like(cos_freq[:, :, : dim % n_elem])
            sin_padding = torch.zeros_like(cos_freq[:, :, : dim % n_elem])
            cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
            sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
        return cos_freq.to(self.dtype), sin_freq.to(self.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process embeddings through transformer blocks with positional encoding.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete,
            `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous): Input `hidden_states`.
            indices_grid (`torch.LongTensor` of shape `(batch size, 3, num latent pixels)`):
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
        Returns:
            Tuple of (processed_embeddings, updated_mask)
        """
        # 1. Input

        if self.num_learnable_registers:
            # replace all padded tokens with learnable registers
            assert hidden_states.shape[1] % self.num_learnable_registers == 0, (
                f"Hidden states sequence length {hidden_states.shape[1]} "
                f"must be divisible by num_learnable_registers {self.num_learnable_registers}."
            )

            num_registers_duplications = hidden_states.shape[1] // self.num_learnable_registers
            learnable_registers = torch.tile(self.learnable_registers, (num_registers_duplications, 1))
            attention_mask_binary = (attention_mask.squeeze(1).squeeze(1).unsqueeze(-1) >= -9000.0).int()
            hidden_states = attention_mask_binary * hidden_states + (1 - attention_mask_binary) * learnable_registers

            attention_mask = torch.full_like(
                attention_mask,
                0.0,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        indices_grid = torch.arange(hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device)
        indices_grid = indices_grid[None, None, :]
        freqs_cis = self.precompute_freqs_cis(indices_grid)

        # 2. Blocks
        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, pe=freqs_cis)

        hidden_states = F.rms_norm(hidden_states, (hidden_states.shape[-1],), eps=1e-6)

        return hidden_states, attention_mask


class GemmaConnector(nn.Module):
    """Standalone connector class for processing Gemma embeddings through video and audio connectors."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize GemmaConnector by loading from checkpoint.

        Args:
            checkpoint_path: Path to the safetensors checkpoint file (local path or HTTP(S) URL)
            dtype: Data type for connector weights
        """
        super().__init__()

        checkpoint_path_str = str(checkpoint_path)
        is_url = checkpoint_path_str.startswith(("http://", "https://"))

        if not is_url:
            checkpoint_path_obj = Path(checkpoint_path)
            if not checkpoint_path_obj.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_obj}")

        # Prefixes for connector weights
        video_prefix = "model.diffusion_model.video_embeddings_connector."
        audio_prefix = "model.diffusion_model.audio_embeddings_connector."
        # Fallback prefix for older checkpoints that only have video connector
        embeddings_prefix = "model.diffusion_model.embeddings_connector."

        sd_video = {}
        sd_audio = {}

        with st.safe_open(checkpoint_path_str, framework="pt", device="cpu") as f:
            keys = f.keys()

            # Check which prefixes exist
            has_video = any(k.startswith(video_prefix) for k in keys)
            has_audio = any(k.startswith(audio_prefix) for k in keys)
            has_embeddings = any(k.startswith(embeddings_prefix) for k in keys)

            if has_video:
                # Load video connector weights
                for k in keys:
                    if k.startswith(video_prefix):
                        tensor = f.get_tensor(k)
                        sd_video[k.removeprefix(video_prefix)] = tensor
            elif has_embeddings:
                # Fallback: use embeddings_connector as video connector
                for k in keys:
                    if k.startswith(embeddings_prefix):
                        tensor = f.get_tensor(k)
                        sd_video[k.removeprefix(embeddings_prefix)] = tensor

            if has_audio:
                # Load audio connector weights
                for k in keys:
                    if k.startswith(audio_prefix):
                        tensor = f.get_tensor(k)
                        sd_audio[k.removeprefix(audio_prefix)] = tensor

        # Create video embeddings connector
        if not sd_video:
            raise ValueError(
                f"No video embeddings connector found in checkpoint {checkpoint_path_str}. "
                f"Expected keys with prefix '{video_prefix}' or '{embeddings_prefix}'"
            )

        self.video_embeddings_connector = Embeddings1DConnector(dtype=dtype)
        self.video_embeddings_connector.load_state_dict(sd_video)

        # Create audio embeddings connector if present
        self.audio_embeddings_connector = None
        if sd_audio:
            self.audio_embeddings_connector = Embeddings1DConnector(dtype=dtype)
            self.audio_embeddings_connector.load_state_dict(sd_audio)

        self.dtype = dtype

    def preprocess_prompt_embeds(
        self, projected: torch.Tensor, attention_mask: torch.Tensor, is_audio: bool = False
    ) -> torch.Tensor:
        """Process projected embeddings through video or audio connector.

        Args:
            projected: Projected embeddings from Gemma encoder [batch_size, seq_len, dim]
            attention_mask: Attention mask [batch_size, seq_len]
            is_audio: If True, use audio connector; otherwise use video connector

        Returns:
            Encoded embeddings tensor [batch_size, seq_len, dim]
        """
        # Convert attention mask to format embeddings connector expects
        if attention_mask.dtype != projected.dtype:
            attention_mask = attention_mask.to(projected.dtype)
        connector_attention_mask = (attention_mask - 1).to(projected.dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(projected.dtype).max

        # Process through video embeddings connector
        if not is_audio:
            encoded, connector_attention_mask = self.video_embeddings_connector(
                projected, connector_attention_mask
            )
        else:
            encoded, connector_attention_mask = self.audio_embeddings_connector(
                projected, connector_attention_mask
            )

        # Restore the mask values to bool
        attention_mask_out = (connector_attention_mask < 0.000001).to(torch.bool)
        attention_mask_out = attention_mask_out.reshape([encoded.shape[0], encoded.shape[1]])

        # Apply attention mask to zero out padding positions
        encoded = encoded * attention_mask_out.unsqueeze(-1)

        return encoded, attention_mask_out
