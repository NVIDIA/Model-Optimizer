# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Video Sparse Attention (VSA) method for video diffusion models.

VSA implements a two-branch sparse attention architecture:
1. Compression Branch: Averages tokens within 3D video blocks and computes coarse attention
2. Sparse Branch: Selects top-K blocks based on importance and computes fine-grained attention

This method requires model modification to expose gate_compress for optimal quality.

Uses the optimized Triton kernel from fastvideo_kernel for 2-6x speedup.

The data flow mirrors FastVideo's VideoSparseAttentionImpl:
  tile(Q,K,V,gate) -> Triton kernel -> untile(output)
"""

import math
from typing import Any

import torch

from . import SparseAttentionMethod, register_sparse_method
from .vsa_utils import (
    construct_variable_block_sizes,
    get_non_pad_index,
    get_reverse_tile_partition_indices,
    get_tile_partition_indices,
)


@register_sparse_method("vsa")
class VSA(SparseAttentionMethod):
    """Video Sparse Attention with two-branch architecture.

    VSA combines a compression branch (coarse-grained block attention) with
    a sparse branch (fine-grained attention on top-K selected blocks).

    The final output is: output = out_compression * gate_compress + out_sparse

    where gate_compress is a learned parameter from the model layer that
    controls the balance between compression and sparse branches.

    Configuration Parameters:
        - block_size_3d: 3D tile dimensions (T, H, W), default (4, 4, 4)
        - top_k_ratio: Ratio of blocks to keep (0.0-1.0), default 0.5
        - video_shape: Video dimensions (T, H, W) after patchification

    Requirements:
        - Model must expose gate_compress parameter in attention layers
        - Input tensors must be 4D: [batch, heads, seq_len, dim]
    """

    def __init__(self, method_config: dict | None = None):
        """Initialize VSA method.

        Args:
            method_config: Configuration dict with VSA parameters.
        """
        super().__init__()
        config = method_config or {}

        # Block configuration
        block_size = config.get("block_size_3d", (4, 4, 4))
        if isinstance(block_size, list):
            block_size = tuple(block_size)
        self.block_size_3d = block_size
        self.block_elements = block_size[0] * block_size[1] * block_size[2]

        # Sparsity configuration
        self.top_k_ratio = config.get("top_k_ratio", 0.5)

        # Video shape (can be set dynamically)
        self.video_shape = config.get("video_shape", None)

        # Track last computed statistics
        self._last_stats: dict = {}

        # Metadata cache: avoids recomputing tile indices on every forward pass.
        # Matches FastVideo's @lru_cache on utility functions.
        self._cached_metadata: dict[str, Any] | None = None
        self._cached_metadata_key: tuple | None = None

    def set_video_shape(self, video_shape: tuple[int, int, int]):
        """Set video shape for current forward pass.

        Args:
            video_shape: Video dimensions (T, H, W) after patchification.
        """
        self.video_shape = video_shape

    def _compute_metadata(self, seq_len: int, device: torch.device) -> dict[str, Any]:
        """Compute block metadata from video shape.

        Results are cached and reused when called with the same (seq_len, video_shape)
        to avoid recomputing tile indices on every denoising step, matching FastVideo's
        ``@functools.lru_cache`` on the underlying utility functions.

        Args:
            seq_len: Sequence length (should equal T * H * W).
            device: Device for tensors.

        Returns:
            Metadata dict with tile indices, variable sizes, etc.
        """
        if self.video_shape is None:
            raise ValueError(
                f"video_shape must be provided for VSA but is None (seq_len={seq_len}). "
                f"Set it via the VSA config ('video_shape' key), call set_video_shape(), "
                f"or use a model-specific plugin (e.g., LTX-2 plugin) that computes it "
                f"from the model's patchifier."
            )

        # Return cached metadata if inputs haven't changed
        cache_key = (seq_len, self.video_shape)
        if self._cached_metadata is not None and self._cached_metadata_key == cache_key:
            return self._cached_metadata

        vid_t, vid_h, vid_w = self.video_shape
        ts_t, ts_h, ts_w = self.block_size_3d

        # Validate sequence length matches video shape
        expected_seq_len = vid_t * vid_h * vid_w
        if seq_len != expected_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} does not match video shape {self.video_shape} "
                f"(expected {expected_seq_len})"
            )

        # Calculate number of tiles
        num_tiles = (
            math.ceil(vid_t / ts_t),
            math.ceil(vid_h / ts_h),
            math.ceil(vid_w / ts_w),
        )
        total_tiles = num_tiles[0] * num_tiles[1] * num_tiles[2]

        # Get partitioning indices
        tile_indices = get_tile_partition_indices(self.video_shape, self.block_size_3d, device)
        reverse_indices = get_reverse_tile_partition_indices(
            self.video_shape, self.block_size_3d, device
        )
        variable_sizes = construct_variable_block_sizes(
            self.video_shape, num_tiles, self.block_size_3d, device
        )
        non_pad_index = get_non_pad_index(variable_sizes, self.block_elements)

        # Calculate padded sizes
        t_padded = num_tiles[0] * ts_t
        h_padded = num_tiles[1] * ts_h
        w_padded = num_tiles[2] * ts_w
        padded_seq_len = t_padded * h_padded * w_padded

        metadata = {
            "video_shape": self.video_shape,
            "tile_size": self.block_size_3d,
            "num_tiles": num_tiles,
            "total_tiles": total_tiles,
            "tile_indices": tile_indices,
            "reverse_indices": reverse_indices,
            "variable_sizes": variable_sizes,
            "non_pad_index": non_pad_index,
            "padded_seq_len": padded_seq_len,
        }

        # Cache for reuse across denoising steps
        self._cached_metadata = metadata
        self._cached_metadata_key = cache_key

        return metadata

    def _tile_tensor(self, tensor: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Rearrange tensor into tile layout with padding.

        Args:
            tensor: Input tensor [batch, heads, seq_len, dim].
            metadata: Metadata from _compute_metadata.

        Returns:
            Tiled tensor [batch, heads, padded_seq_len, dim].
        """
        batch, heads, seq_len, dim = tensor.shape
        device = tensor.device
        dtype = tensor.dtype

        tile_indices = metadata["tile_indices"]
        non_pad_index = metadata["non_pad_index"]
        padded_seq_len = metadata["padded_seq_len"]

        # Create padded tensor
        padded = torch.zeros((batch, heads, padded_seq_len, dim), device=device, dtype=dtype)

        # Rearrange to tile order and place in padded positions
        padded[:, :, non_pad_index] = tensor[:, :, tile_indices]

        return padded

    def _untile_tensor(self, tensor: torch.Tensor, metadata: dict, seq_len: int) -> torch.Tensor:
        """Reverse tile layout back to original order.

        Args:
            tensor: Tiled tensor [batch, heads, padded_seq_len, dim].
            metadata: Metadata from _compute_metadata.
            seq_len: Original sequence length.

        Returns:
            Output tensor [batch, heads, seq_len, dim].
        """
        non_pad_index = metadata["non_pad_index"]
        reverse_indices = metadata["reverse_indices"]

        # Extract non-padded tokens and reverse order
        return tensor[:, :, non_pad_index][:, :, reverse_indices]

    def forward_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor | None = None,
        video_shape: tuple[int, int, int] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """Compute VSA two-branch sparse attention.

        Data flow (mirrors FastVideo's VideoSparseAttentionImpl):
        1. Compute tile metadata from video_shape
        2. Tile Q, K, V, gate_compress into padded tile order
        3. Run Triton VSA kernel on tiled tensors
        4. Untile output back to original token order

        Args:
            query: Query tensor [batch, heads, seq_len, dim].
            key: Key tensor [batch, heads, seq_len, dim].
            value: Value tensor [batch, heads, seq_len, dim].
            gate_compress: Learned gating weights [batch, heads, seq_len, dim].
                          If None, uses equal weighting (0.5) for both branches.
            video_shape: Video dimensions (T, H, W). If None, uses self.video_shape.
            **kwargs: Additional arguments (ignored).

        Returns:
            Tuple of (attention_output, stats) where:
            - attention_output: [batch, heads, seq_len, dim]
            - stats: Dict with sparsity statistics
        """
        if video_shape is not None:
            self.video_shape = video_shape

        batch, heads, seq_len, dim = query.shape
        device = query.device

        # Compute block metadata (cached across denoising steps)
        metadata = self._compute_metadata(seq_len, device)
        total_tiles = metadata["total_tiles"]
        variable_sizes = metadata["variable_sizes"]

        # Calculate top-K based on ratio
        top_k = max(1, int(self.top_k_ratio * total_tiles))

        # ========== TILE: rearrange tokens into tile order ==========
        # Mirrors FastVideo's VideoSparseAttentionImpl.preprocess_qkv (tile)
        query_tiled = self._tile_tensor(query, metadata)
        key_tiled = self._tile_tensor(key, metadata)
        value_tiled = self._tile_tensor(value, metadata)
        gate_tiled = (
            self._tile_tensor(gate_compress, metadata) if gate_compress is not None else None
        )

        # ========== TRITON VSA KERNEL ==========
        # Kernel operates on tiled tensors in [batch, heads, padded_seq, dim] format
        try:
            from fastvideo_kernel import video_sparse_attn as triton_vsa_kernel
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "VSA requires the 'fastvideo_kernel' package for its Triton sparse attention "
                "kernel.  The VSA method registered successfully, but the kernel is needed at "
                "runtime.  Install it with:\n"
                "  git clone https://github.com/FastVideo/FastVideo.git\n"
                "  cd FastVideo/fastvideo-kernel && ./build.sh\n"
                "See https://github.com/hao-ai-lab/FastVideo/tree/main/fastvideo-kernel for details."
            ) from None
        output_tiled = triton_vsa_kernel(
            query_tiled,
            key_tiled,
            value_tiled,
            variable_sizes,  # q_variable_sizes
            variable_sizes,  # kv_variable_sizes
            top_k,
            block_size=self.block_size_3d,
            compress_attn_weight=gate_tiled,
        )

        # ========== UNTILE: restore original token order ==========
        # Mirrors FastVideo's VideoSparseAttentionImpl.postprocess_output (untile)
        output = self._untile_tensor(output_tiled, metadata, seq_len)

        # Compute statistics
        actual_sparsity = 1.0 - (top_k / total_tiles)
        stats = {
            "sparsity": actual_sparsity,
            "phase": "vsa_triton",
            "total_blocks": total_tiles,
            "sparse_blocks": total_tiles - top_k,
            "top_k": top_k,
            "video_shape": self.video_shape,
        }
        self._last_stats = stats

        return output, stats

    def calculate_sparsity(
        self,
        attention_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Not used by VSA. Required stub for the abstract base class.

        VSA replaces the entire attention computation via ``forward_attention()``,
        which is called directly by model-specific plugins (e.g., ``_LTX2SparseAttention``).
        The softmax-patching path that calls this method is never reached in the VSA flow.

        Raises:
            NotImplementedError: Always. Use ``forward_attention()`` instead.
        """
        raise NotImplementedError(
            "VSA does not use the softmax-patching path. "
            "Use forward_attention() via a model-specific plugin instead."
        )

    def apply_sparsity(
        self,
        attention_scores: torch.Tensor,
        sparse_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Not used by VSA. Required stub for the abstract base class.

        See ``calculate_sparsity`` for details.

        Raises:
            NotImplementedError: Always. Use ``forward_attention()`` instead.
        """
        raise NotImplementedError(
            "VSA does not use the softmax-patching path. "
            "Use forward_attention() via a model-specific plugin instead."
        )

    def get_threshold_info(self) -> dict[str, Any]:
        """Get VSA configuration info.

        Returns:
            Dictionary with VSA configuration.
        """
        return {
            "type": "vsa",
            "block_size_3d": self.block_size_3d,
            "top_k_ratio": self.top_k_ratio,
            "video_shape": self.video_shape,
        }

    @property
    def name(self) -> str:
        """Method identifier."""
        return "vsa"
