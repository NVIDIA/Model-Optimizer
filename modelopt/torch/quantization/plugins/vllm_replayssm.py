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

"""Opt-in two-precision GDN serving with vLLM-owned ReplaySSM cache pages."""

import weakref
from dataclasses import replace
from math import prod

import torch
import vllm
from packaging.version import Version
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    causal_conv1d_update,
    is_conv_state_dim_first,
)

from modelopt.torch.kernels.quantization.linear_attention.replayssm_cache import (
    _gdn_replay_state_spec,
    _GDNReplayCache,
)

from ..linear_attention._two_precision import validate_two_precision
from ..nn import QuantModuleRegistry
from .linear_attention import _LinearAttentionQuantMixin
from .vllm import create_parallel_state

__all__ = ["configure_vllm_replayssm"]


def configure_vllm_replayssm(model_runner, config, quant_config):
    """Apply a dynamic replay recipe after model loading and before cache allocation.

    This initial engine integration supports one request at a time, including
    chunked prompt prefill and request-slot reuse. HF weights must be accompanied
    by the same explicit replay recipe used for training.
    """
    # Import here to avoid the quantization API/plugin import cycle.
    from ..model_quant import quantize

    if Version(vllm.__version__).release[:2] != (0, 30):
        raise NotImplementedError("ReplaySSM serving requires vLLM 0.30.x")
    if (
        not config.model_config.enforce_eager
        or config.model_config.dtype != torch.bfloat16
        or config.scheduler_config.async_scheduling
        or config.scheduler_config.max_num_seqs != 1
    ):
        raise ValueError("ReplaySSM requires BF16, eager synchronous execution and max_num_seqs=1")
    if (
        config.speculative_config is not None
        or config.cache_config.enable_prefix_caching
        or config.cache_config.mamba_cache_mode != "none"
        or config.kv_transfer_config is not None
        or config.ec_transfer_config is not None
        or config.parallel_config.world_size != 1
        or config.parallel_config.data_parallel_size != 1
        or getattr(config.parallel_config, "prefill_context_parallel_size", 1) != 1
    ):
        raise ValueError(
            "ReplaySSM requires one GPU, no prefix caching, speculation or cache transfer"
        )
    if quant_config.get("algorithm") is not None:
        raise ValueError("ReplaySSM's dynamic read recipe does not require calibration")
    model = model_runner.model
    layers = [module for module in model.modules() if type(module) is QwenGatedDeltaNetAttention]
    if not layers:
        raise ValueError("ReplaySSM requires Qwen GDN attention layers")
    for layer in layers:
        if layer.head_k_dim != 128 or layer.gdn_prefill_backend != "triton":
            raise ValueError("ReplaySSM requires K=128 and gdn_prefill_backend='triton'")
        if layer.get_state_dtype()[1] != torch.float32:
            raise ValueError("ReplaySSM requires mamba_ssm_cache_dtype='float32'")
    if QwenGatedDeltaNetAttention in QuantModuleRegistry:
        raise ValueError("Register ReplaySSM before converting GDN modules with another adapter")
    QuantModuleRegistry.register({QwenGatedDeltaNetAttention: "vllm_ReplaySSMGDN"})(_ReplaySSMGDN)
    quantize(model, quant_config)
    for layer in layers:
        if layer.linear_attention_config.precision != "replayssm_two_precision":
            raise ValueError("Every GDN layer must select precision='replayssm_two_precision'")
        layer.enable_fused_gdn_decode = False
        layer._replay_runner = weakref.proxy(model_runner)


class _ReplaySSMGDN(_LinearAttentionQuantMixin):
    linear_attention_quantizer_names = ("gdn_state_quantizer", "gdn_read_quantizer")

    def _setup(self):
        super()._setup()
        self.parallel_state = create_parallel_state()
        self._register_temp_attribute("_replay_cache", None)
        self._register_temp_attribute("_replay_runner", None)

    def _replay_spec(self):
        return _gdn_replay_state_spec(
            self.num_k_heads // self.tp_size, self.num_v_heads // self.tp_size, self.head_v_dim
        )

    def get_kv_cache_spec(self, vllm_config):
        spec = super().get_kv_cache_spec(vllm_config)
        extra = self._replay_spec()[1:]
        return replace(
            spec,
            shapes=(*spec.shapes, *(shape for shape, _ in extra)),
            dtypes=(*spec.dtypes, *(dtype for _, dtype in extra)),
        )

    def bind_kv_cache(self, cache):
        # Preserve the native two-state warmup interface; only the engine cache
        # specification and its page views include the replay buffers.
        spec = (*zip(self.get_state_shape(), self.get_state_dtype()), *self._replay_spec()[1:])
        pages = cache.flatten(1)
        offset, buffers = 0, []
        for shape, dtype in spec:
            size = prod(shape) * dtype.itemsize
            buffers.append(pages[:, offset : offset + size].view(dtype).view(-1, *shape))
            offset += size
        self.kv_cache = tuple(buffers)
        self._replay_cache = _GDNReplayCache(
            self.kv_cache[1:],
            self.num_k_heads // self.tp_size,
            quantized_read=validate_two_precision(
                self.gdn_state_quantizer, self.gdn_read_quantizer
            ),
        )

    def _prompt_progress(self):
        runner = self._replay_runner
        if hasattr(runner, "req_states"):
            # vLLM 0.30 V2 keeps request metadata in indexed CPU arrays.
            state = runner.req_states
            index = next(iter(state.req_id_to_index.values()))
            return state.num_computed_tokens_np[index], state.prefill_len.np[index]
        batch = runner.input_batch
        return batch.num_computed_tokens_cpu[0], batch.num_prompt_tokens[0]

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        metadata = get_forward_context().attn_metadata
        metadata = metadata.get(self.prefix) if isinstance(metadata, dict) else None
        if metadata is None:
            return super()._forward_core(mixed_qkv, b, a, core_attn_out)
        read_enabled = self.gdn_read_quantizer.is_enabled and self.gdn_read_quantizer._if_quant
        if self.gdn_state_quantizer.is_enabled or read_enabled != self._replay_cache.quantized_read:
            raise ValueError("Restart the ReplaySSM worker to change checkpoint precision")
        if metadata.spec_sequence_masks is not None or (
            metadata.num_prefills and metadata.num_decodes
        ):
            raise ValueError(
                "ReplaySSM does not yet support speculative or mixed prefill/decode batches"
            )
        indices = metadata.non_spec_state_indices_tensor[
            : metadata.num_decodes + metadata.num_prefills
        ]
        computed, prompt_length = self._prompt_progress()
        is_prompt = computed < prompt_length
        if is_prompt:
            if computed > 0 and torch.any(self._replay_cache.steps.index_select(0, indices) != 0):
                raise ValueError("Continuation prefill after replay decode is not supported")
            # A one-token prompt/tail is labeled decode by GDN metadata, so use
            # the runner's CPU prompt boundary to avoid advancing replay early.
            if metadata.has_initial_state is None:
                if computed == 0:
                    self.kv_cache[0].index_fill_(0, indices.long(), 0)
                    self.kv_cache[1].index_fill_(0, indices.long(), 0)
                super()._forward_core(mixed_qkv, b, a, core_attn_out)
                self._replay_cache.reset(indices)
                return
            super()._forward_core(mixed_qkv, b, a, core_attn_out)
            self._replay_cache.reset(indices)
            return
        count = metadata.num_actual_tokens
        conv = self.kv_cache[0]
        if not is_conv_state_dim_first():
            conv = conv.transpose(-1, -2)
        packed = causal_conv1d_update(
            mixed_qkv[:count],
            conv,
            self.conv1d.weight.flatten(1),
            self.conv1d.bias,
            self.activation,
            conv_state_indices=indices,
            validate_data=False,
        )
        self._replay_cache.step(
            packed.contiguous(),
            a[:count].contiguous(),
            b[:count].contiguous(),
            self.A_log.float().contiguous(),
            self.dt_bias.float().contiguous(),
            indices,
            core_attn_out[:count],
        )
