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
        - cos_cache: Cosine cache as numpy array with shape [max_seq_len, head_dim//2]
        - sin_cache: Sine cache as numpy array with shape [max_seq_len, head_dim//2]
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

    theta = getattr(config, "rope_theta", 10000.0)
    head_dim = config.hidden_size // config.num_attention_heads
    partial_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_factor)

    # Match builder: int64 -> float
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.int64).float()

    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos_cache = emb.cos()
    sin_cache = emb.sin()

    # Cast to target dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    target_dtype = dtype_map.get(io_dtype, torch.float16)

    cos_cache = cos_cache.to(target_dtype)
    sin_cache = sin_cache.to(target_dtype)

    # Slice to half (GQA expects this)
    if cos_cache.shape[-1] == head_dim:
        cos_cache = cos_cache[:, : head_dim // 2]
        sin_cache = sin_cache[:, : head_dim // 2]

    # Convert to numpy - bfloat16 needs special handling
    if io_dtype == "bfloat16":
        # bfloat16 can't be converted directly to numpy
        # Return as int16 view, add_initializer will handle proper ONNX storage
        cos_np = cos_cache.view(torch.int16).numpy()
        sin_np = sin_cache.view(torch.int16).numpy()
        return cos_np, sin_np, config
    else:
        return cos_cache.numpy(), sin_cache.numpy(), config
