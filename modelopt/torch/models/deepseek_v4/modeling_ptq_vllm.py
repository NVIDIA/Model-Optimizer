# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""DeepSeek-V4 PTQ modeling for vLLM (HF model type ``deepseek_v4``)."""

import torch
from vllm.forward_context import get_forward_context

from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.vllm_indexer import (
    _fake_quantize_fp8_rows,
    _get_arg,
    _native_positional_index,
    _QuantVLLMIndexerBase,
    _requantize_fp8_indexer_k_cache,
)
from modelopt.torch.quantization.plugins.vllm_mega_moe import _QuantVLLMMegaMoEExperts

__all__: list[str] = []

# NotImplementedError: some vLLM model packages reject unsupported platforms at import.
try:
    from vllm.models.deepseek_v4.attention import DeepseekV4Indexer as VllmDeepseekV4Indexer
except (ImportError, NotImplementedError):
    VllmDeepseekV4Indexer = None
try:
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts
except (ImportError, NotImplementedError):
    DeepseekV4MegaMoEExperts = None


class _QuantVLLMDeepseekV4Indexer(_QuantVLLMIndexerBase):
    """DeepSeek-V4 indexer: fused kernels FP8-quantize the query and cache the compressed key.

    After the forward, the cache entries the compressor wrote (one per ``compress_ratio`` tokens)
    are re-quantized through ``indexer_k_quantizer`` and the returned query through
    ``indexer_q_quantizer``. vLLM uses both without the Hadamard rotation that the DeepSeek
    reference applies.
    """

    def _setup(self):
        super()._setup()
        self._positions_pos = _native_positional_index(self, "forward", "positions")
        self._indexer_weights_pos = _native_positional_index(self, "forward", "indexer_weights")

    def forward(self, *args, **kwargs):
        self._validate_scorer_quantizers()
        q, q_scale, weights = super().forward(*args, **kwargs)
        if self.indexer_k_quantizer.is_enabled:
            self._check_fp8_indexer()
            self._requantize_written_keys(_get_arg(args, kwargs, "positions", self._positions_pos))
        if self.indexer_q_quantizer.is_enabled and q is not None:  # None: short-context shortcut
            self._check_fp8_indexer()
            indexer_weights = _get_arg(args, kwargs, "indexer_weights", self._indexer_weights_pos)
            q, weights = self._requantize_query(q, weights, indexer_weights)
        return q, q_scale, weights

    def _check_fp8_indexer(self) -> None:
        if getattr(self, "use_fp4_kv", False):
            raise NotImplementedError(
                "The indexer quantizers re-quantize the FP8 indexer query and cache; serve with "
                "the default indexer_kv_dtype instead of 'mxfp4'."
            )

    def _requantize_query(
        self, q: torch.Tensor, weights: torch.Tensor, indexer_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-quantize the FP8 query; the kernel folds its power-of-two scale into ``weights``.

        The kernel returns ``indexer_weights * q_scale * softmax_scale * n_head**-0.5`` as
        ``weights`` and no ``q_scale``: divide the other factors out, round to the power of two and
        swap in the new scale.
        """
        base = indexer_weights.float() * self.softmax_scale * self.n_head**-0.5
        # A head with zero weight does not score; zero its row instead of recovering its scale.
        live = (weights != 0) & (base != 0)
        ratio = torch.where(live, weights / torch.where(live, base, 1.0), 1.0)
        old_scale = torch.exp2(torch.round(torch.log2(ratio)))
        x = torch.where(live[..., None], q.float() * old_scale[..., None], 0.0)
        q, new_scale = _fake_quantize_fp8_rows(x, self.indexer_q_quantizer)
        return q, weights * (new_scale / old_scale)

    def _requantize_written_keys(self, positions: torch.Tensor) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):  # profiling run: nothing was written
            return
        # The compress kernel skips tokens without a compressor-state slot or an indexer-cache
        # slot, and only the last token of each compression group produces an entry.
        state_slots = attn_metadata[self.compressor.state_cache.prefix].slot_mapping
        num_tokens = state_slots.shape[0]
        k_slots = attn_metadata[self.k_cache.prefix].slot_mapping[:num_tokens]
        positions = positions[:num_tokens]
        valid = (k_slots >= 0) & (state_slots >= 0) & ((positions + 1) % self.compress_ratio == 0)
        _requantize_fp8_indexer_k_cache(
            self.k_cache.kv_cache, k_slots, valid, self.indexer_k_quantizer
        )


if VllmDeepseekV4Indexer is not None:
    QuantModuleRegistry.register({VllmDeepseekV4Indexer: "vllm_DeepseekV4Indexer"})(
        _QuantVLLMDeepseekV4Indexer
    )

if DeepseekV4MegaMoEExperts is not None:
    QuantModuleRegistry.register({DeepseekV4MegaMoEExperts: "vllm_DeepseekV4MegaMoEExperts"})(
        _QuantVLLMMegaMoEExperts
    )
