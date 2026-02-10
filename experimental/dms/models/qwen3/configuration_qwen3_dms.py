# Adapted from https://github.com/huggingface/transformers/blob/47b0e478f324b54f177ea7998a0791870fdd0324/src/transformers/models/qwen3/configuration_qwen3.py

# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
"""Qwen3 model configuration."""

from dms.core import setup_compile_limit_for_dms
from transformers import Qwen3Config


class Qwen3ConfigDMS(Qwen3Config):
    """DMS configuration for Qwen3 model."""

    def __init__(
        self,
        dms_alpha_scale: float = 100.0,
        dms_initial_alpha_offset: float = 5.0,
        dms_window_size: int = 512,
        dms_paged_attention_block_size: int = 256,
        dms_cr: int = 8,
        dms_disable_eviction: bool = False,
        dms_separate_alpha: bool = False,
        dms_alpha_per: str = "head",
        dms_tau: float = 0.1,
        dms_compile_limit: int | None = 72,
        dms_manual_inference_mode: bool = False,
        dms_chunked_prefill: int | None = None,
        dms_preallocate_for_tokens: int = 4096,
        **kwargs,
    ):
        """Initialize the Qwen3ConfigDMS model.

        Args:
        dms_alpha_scale: scaling factor for DMS decision logits.
        dms_initial_alpha_offset: initial offset for DMS decision logits.
        dms_window_size: sliding window size for DMS.
        dms_paged_attention_block_size: block size for paged cache.
        dms_cr: compression ratio for DMS. For documentation purposes only.
        dms_disable_eviction: turns adapter DMS models into vanilla models.
        dms_separate_alpha: True -> We initialise new parameters, False -> DMS uses query parameters.
        dms_alpha_per: Whether to make per head or per layer for DMS eviction decisions.
        dms_tau: Temperature for DMS decision logits.
        dms_compile_limit: Torch.compile limit.
        dms_manual_inference_mode: Whether to use inference with manual prefill/inference switching for kv-cache.
        dms_chunked_prefill: Chunk size for prefill.
        dms_preallocate_for_tokens: Preallocate space for tokens in kv-cache.
        """
        self.dms_alpha_scale = dms_alpha_scale
        self.dms_initial_alpha_offset = dms_initial_alpha_offset
        self.dms_window_size = dms_window_size
        self.dms_paged_attention_block_size = dms_paged_attention_block_size
        self.dms_cr = dms_cr
        self.dms_disable_eviction = dms_disable_eviction
        self.dms_separate_alpha = dms_separate_alpha
        self.dms_alpha_per = dms_alpha_per
        self.dms_tau = dms_tau
        self.dms_manual_inference_mode = dms_manual_inference_mode
        self.dms_chunked_prefill = dms_chunked_prefill
        self.dms_preallocate_for_tokens = dms_preallocate_for_tokens

        assert self.dms_paged_attention_block_size > 0, (
            f"dms_paged_attention_block_size: {self.dms_paged_attention_block_size} is not greater than 0"
        )
        assert self.dms_window_size > self.dms_paged_attention_block_size, (
            f"dms_window_size: {self.dms_window_size} "
            f"is not greater than dms_paged_attention_block_size: {self.dms_paged_attention_block_size}"
        )
        assert self.dms_alpha_per in ["head", "layer"], (
            f"dms_alpha_per: {self.dms_alpha_per} is not supported"
        )
        if dms_compile_limit is not None:
            setup_compile_limit_for_dms(compile_limit=dms_compile_limit)
        super().__init__(
            **kwargs,
        )
