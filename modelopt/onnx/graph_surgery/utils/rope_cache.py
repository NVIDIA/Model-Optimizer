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

"""RoPE (Rotary Position Embedding) cache computation utilities.

This module provides functions for computing cosine and sine caches
for rotary position embeddings, matching the onnxruntime_genai builder format.
"""

from typing import Any

import numpy as np
import torch


def _get_rope_parameters(config: Any) -> dict[str, Any]:
    """Return the config's RoPE settings as one flat dict that includes ``rope_theta``."""
    # transformers 5 folds config.json's rope_theta and rope_scaling into rope_parameters and drops
    # the rope_theta attribute; transformers 4 keeps them as separate attributes.
    params = getattr(config, "rope_parameters", None)
    if not params:
        params = {**(getattr(config, "rope_scaling", None) or {})}
        params["rope_theta"] = getattr(config, "rope_theta", 10000.0)
    if "rope_theta" not in params:
        raise ValueError(f"Per-layer-type RoPE parameters are not supported: {sorted(params)}")
    return params


def _rope_length_limit(config: Any, rope: dict[str, Any], rope_type: str) -> int | None:
    """Sequence length past which transformers switches this RoPE type to other frequencies."""
    if "dynamic" in rope_type:
        return config.max_position_embeddings
    if rope_type != "longrope":
        return None
    if top_level := getattr(config, "original_max_position_embeddings", None):
        return top_level
    # Without the top-level value, transformers 5 switches at rope_scaling's value and
    # transformers 4 at max_position_embeddings; take the earlier of the two.
    nested = rope.get("original_max_position_embeddings") or config.max_position_embeddings
    return min(nested, config.max_position_embeddings)


def get_rope_caches(
    model_id: str,
    max_seq_len: int,
    io_dtype: str = "float16",
    trust_remote_code: bool = False,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Compute cos/sin caches matching onnxruntime_genai builder.

    This function computes the rotary position embedding caches required
    for GroupQueryAttention (GQA) nodes. The caches are computed based on
    the model's configuration from HuggingFace.

    Args:
        model_id: HuggingFace model ID or path to config.
        max_seq_len: Maximum sequence length for the caches.
        io_dtype: Data type for output ("float16", "float32", or "bfloat16").
        trust_remote_code: Whether to trust remote code in HuggingFace model config.

    Returns:
        Tuple of (cos_cache, sin_cache, config) where:
        - cos_cache: Cosine cache as numpy array with shape [max_seq_len, rotary_dim//2]
        - sin_cache: Sine cache as numpy array with shape [max_seq_len, rotary_dim//2]
        - config: HuggingFace model configuration

    Example:
        >>> cos_cache, sin_cache, config = get_rope_caches(
        ...     model_id="meta-llama/Llama-2-7b-hf",
        ...     max_seq_len=4096,
        ...     io_dtype="float16",
        ... )
        >>> print(f"Cache shape: {cos_cache.shape}")
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    rope = _get_rope_parameters(config)
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    head_dim = config.hidden_size // config.num_attention_heads
    partial_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_factor)

    attention_scaling = 1.0
    if rope_type == "default":
        # Match builder: int64 -> float
        inv_freq = 1.0 / (
            rope["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
    else:
        # Only scaled RoPE needs this; transformers < 4.43 lacks it.
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        if rope_type not in ROPE_INIT_FUNCTIONS:
            raise ValueError(f"Unsupported rope_type {rope_type!r}")
        limit = _rope_length_limit(config, rope, rope_type)
        if limit is not None and max_seq_len > limit:
            raise ValueError(
                f"rope_type {rope_type!r} changes its frequencies past {limit} positions, which one "
                f"static cache cannot follow; use max_seq_len <= {limit}"
            )
        # Scaled RoPE (llama3, yarn, ...): the function the exported rotary embedding was traced from.
        inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, "cpu")
    t = torch.arange(max_seq_len, dtype=torch.int64).float()

    # GQA takes one column per inverse frequency: [max_seq_len, rotary_dim // 2].
    freqs = torch.outer(t, inv_freq)
    cos_cache = freqs.cos() * attention_scaling
    sin_cache = freqs.sin() * attention_scaling

    # Cast to target dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    target_dtype = dtype_map.get(io_dtype, torch.float16)

    cos_cache = cos_cache.to(target_dtype)
    sin_cache = sin_cache.to(target_dtype)

    # Convert to numpy - bfloat16 needs special handling
    if io_dtype == "bfloat16":
        # bfloat16 can't be converted directly to numpy
        # Return as int16 view, add_initializer will handle proper ONNX storage
        cos_np = cos_cache.view(torch.int16).numpy()
        sin_np = sin_cache.view(torch.int16).numpy()
        return cos_np, sin_np, config
    else:
        return cos_cache.numpy(), sin_cache.numpy(), config
