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

"""DeepSeek-V4 Pro native MTP construction and streaming checkpoint loading."""

import json
import math
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed.fsdp as fsdp
from safetensors import safe_open
from torch import nn
from transformers import DynamicCache
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4HyperHead,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
)

from modelopt.torch.export.shard_cast_utils import (
    dequantize_fp8_ue8m0_to_bf16,
    quantize_bf16_to_fp8_ue8m0,
)
from modelopt.torch.quantization.qtensor import MXFP4QTensor
from modelopt.torch.quantization.utils import replace_function
from modelopt.torch.utils.plugins.hf_checkpoint_utils import indexed_weight_map

from ..mtp.adapter import NativeMTPAdapter, NativeMTPCheckpointError, register_native_mtp_adapter
from .dflash_flex_attention import _tiles, build_draft_block_mask
from .modeling_fakebase import FakeBaseConfig, FakeBaseModel
from .modeling_mtp import MTPModule, MTPModuleOutput, create_mtp_model

__all__ = ["DeepSeekV4MTPAdapter", "DeepSeekV4MTPModule"]


_DIRECT_STATE_KEYS = {
    "attn.attn_sink": "decoder.self_attn.sinks",
    "attn.kv_norm.weight": "decoder.self_attn.kv_norm.weight",
    "attn.q_norm.weight": "decoder.self_attn.q_a_norm.weight",
    "attn.wkv.weight": "decoder.self_attn.kv_proj.weight",
    "attn.wo_a.weight": "decoder.self_attn.o_a_proj.weight",
    "attn.wo_b.weight": "decoder.self_attn.o_b_proj.weight",
    "attn.wq_a.weight": "decoder.self_attn.q_a_proj.weight",
    "attn.wq_b.weight": "decoder.self_attn.q_b_proj.weight",
    "attn_norm.weight": "decoder.input_layernorm.weight",
    "e_proj.weight": "e_proj.weight",
    "enorm.weight": "enorm.weight",
    "ffn.gate.bias": "decoder.mlp.gate.e_score_correction_bias",
    "ffn.gate.weight": "decoder.mlp.gate.weight",
    "ffn.shared_experts.w1.weight": "decoder.mlp.shared_experts.gate_proj.weight",
    "ffn.shared_experts.w2.weight": "decoder.mlp.shared_experts.down_proj.weight",
    "ffn.shared_experts.w3.weight": "decoder.mlp.shared_experts.up_proj.weight",
    "ffn_norm.weight": "decoder.post_attention_layernorm.weight",
    "h_proj.weight": "h_proj.weight",
    "hc_attn_base": "decoder.attn_hc.base",
    "hc_attn_fn": "decoder.attn_hc.fn",
    "hc_attn_scale": "decoder.attn_hc.scale",
    "hc_ffn_base": "decoder.ffn_hc.base",
    "hc_ffn_fn": "decoder.ffn_hc.fn",
    "hc_ffn_scale": "decoder.ffn_hc.scale",
    "hc_head_base": "hc_head.hc_base",
    "hc_head_fn": "hc_head.hc_fn",
    "hc_head_scale": "hc_head.hc_scale",
    "hnorm.weight": "hnorm.weight",
    "norm.weight": "norm.weight",
}


class DeepSeekV4MTPModule(MTPModule):
    """One native MTP layer; the shared EAGLE loop supplies aligned tokens and HC states."""

    def __init__(self, config: DeepseekV4Config):
        """Construct the native projections and reuse Transformers' DSV4 decoder."""
        super().__init__(config, state_shape=(config.hc_mult, config.hidden_size))
        self.e_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.h_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.enorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.decoder = DeepseekV4DecoderLayer(config, 0)
        self.hc_head = DeepseekV4HyperHead(config)
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb: DeepseekV4RotaryEmbedding | None = None
        # FSDP2 must broadcast the fixed routing correction with the other persistent state.
        gate = self.decoder.mlp.gate
        bias = gate.e_score_correction_bias
        del gate.e_score_correction_bias
        gate.register_parameter("e_score_correction_bias", nn.Parameter(bias, requires_grad=False))

    @contextmanager
    def fsdp2_context(self):
        """Keep native controls in FP32 while FSDP2 gathers large projections in BF16."""
        fully_shard = fsdp.fully_shard

        def wrap(module, **kwargs):
            if module is self:
                policy = kwargs.get("mp_policy", fsdp.MixedPrecisionPolicy())
                attention = self.decoder.self_attn
                fully_shard(list(attention.children()), **kwargs)
                fully_shard(
                    [attention, self.decoder.attn_hc, self.decoder.ffn_hc, self.hc_head],
                    **{
                        **kwargs,
                        "mp_policy": replace(
                            policy, param_dtype=torch.float32, cast_forward_inputs=False
                        ),
                    },
                )
                fully_shard(
                    self.decoder.mlp.gate,
                    **{
                        **kwargs,
                        "mp_policy": replace(policy, param_dtype=torch.float32),
                    },
                )
            return fully_shard(module, **kwargs)

        with replace_function(fsdp, "fully_shard", wrap, "_modelopt_mtp_fully_shard"):
            yield

    def _maybe_init_rope(self, device):
        if self.rotary_emb is None:
            with torch.device(device):
                self.rotary_emb = DeepseekV4RotaryEmbedding(self.config)
        else:
            self.rotary_emb.to(device)

    def _forward(
        self,
        hidden_states,
        inputs_embeds,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        input_ids,
    ):
        if not use_cache:
            past_key_values = None
        tokens = self.enorm(inputs_embeds)
        hidden_states = self.e_proj(tokens).unsqueeze(2) + self.h_proj(self.hnorm(hidden_states))
        length = tokens.shape[1]
        if position_ids is None:
            position_ids = torch.arange(length, device=tokens.device).unsqueeze(0)
        # EAGLE's TTT mask addresses complete previous rounds. A sliding cache would evict them.
        if past_key_values is not None and past_key_values.get_seq_length() == 0:
            past_key_values = DynamicCache()
        cached = past_key_values.get_seq_length() if past_key_values is not None else 0
        backend = self.config._attn_implementation if tokens.is_cuda else "eager"
        mask = self._prepare_attention_mask(tokens, attention_mask, cached, backend)
        _, kernel_options = _tiles(self.config.head_dim)
        self._maybe_init_rope(tokens.device)
        assert self.rotary_emb is not None
        # Restore the requested backend after CPU execution so later CUDA placement uses it.
        configured_backend = self.config._attn_implementation
        self.config._attn_implementation = backend
        try:
            raw = self.decoder(
                hidden_states,
                input_ids=input_ids,
                position_embeddings={
                    "main": self.rotary_emb(tokens, position_ids=position_ids, layer_type="main")
                },
                position_ids=position_ids,
                attention_mask=mask,
                kernel_options=kernel_options,
                past_key_values=past_key_values if use_cache else None,
            )
        finally:
            self.config._attn_implementation = configured_backend
        return MTPModuleOutput(
            self.norm(self.hc_head(raw)), raw, past_key_values if use_cache else None
        )

    def _prepare_attention_mask(self, tokens, attention_mask, cached, backend):
        length = tokens.shape[1]
        kv_length = cached + length
        query_pos = torch.arange(length, device=tokens.device)[:, None]
        key_pos = torch.arange(kv_length, device=tokens.device)[None, :] % length
        local = (query_pos >= key_pos) & (query_pos - key_pos < self.config.sliding_window)
        mask = (
            tokens.new_zeros((1, 1, length, kv_length))
            if attention_mask is None
            else attention_mask
        )
        mask = mask.masked_fill(~local, torch.finfo(tokens.dtype).min)
        if backend != "flex_attention":
            return mask

        def allowed(batch, head, q, kv):
            # Flex evaluates whole tiles, including indices beyond an unaligned sequence end.
            return (
                (q < length)
                & (kv < kv_length)
                & (mask[batch, 0, q.clamp(max=length - 1), kv.clamp(max=kv_length - 1)] == 0)
            )

        return build_draft_block_mask(
            allowed, mask.shape[0], length, kv_length, tokens.device, {self.config.head_dim}
        )


@register_native_mtp_adapter
class DeepSeekV4MTPAdapter(NativeMTPAdapter):
    """Load local native DSV4 Pro MTP and frozen target endpoints without the backbone."""

    name = "deepseek_v4"

    @classmethod
    def supports(cls, model_path):
        """Identify native checkpoint names using metadata only."""
        try:
            with (Path(model_path) / "config.json").open() as stream:
                data = json.load(stream)
            return data.get(
                "model_type"
            ) == "deepseek_v4" and "mtp.0.e_proj.weight" in indexed_weight_map(model_path)
        except (OSError, ValueError):
            return False

    @classmethod
    def load_mtp(cls, model_path, *, dtype, device):
        """Load BF16 MTP projections with native FP32 controls, or construct their meta layout."""
        return cls._load_native(model_path, dtype, device, with_base=False)[0]

    @classmethod
    def create(cls, model_path, *, dtype, device, config):
        """Attach native checkpoint state to FakeBase and the shared EAGLE loop."""
        mtp, base = cls._load_native(model_path, dtype, device, with_base=True)
        return create_mtp_model(mtp, base, config)

    @classmethod
    def _load_native(cls, model_path, dtype, device, *, with_base):
        if dtype != torch.bfloat16:
            raise ValueError("DSV4 native MTP training requires BF16 projection weights")
        device = torch.device("cpu" if device is None else device)
        config, base_layers = cls._native_config(model_path)
        with torch.device("meta"):
            mtp = DeepSeekV4MTPModule(config).to(dtype=dtype)
            base = cls._create_base(config, base_layers, dtype) if with_base else None

        weight_map = indexed_weight_map(model_path)
        with ExitStack() as stack, torch.no_grad():
            targets = cls._load_targets(mtp, base, weight_map)
            sources = cls._checkpoint_tensors(model_path, weight_map, targets, stack)
            # Native FP32 controls must not lose precision before loading or meta sharding.
            for native_name, parameter_name in _DIRECT_STATE_KEYS.items():
                if sources["mtp.0." + native_name][0].get_dtype() == "F32":
                    parameter = mtp.get_parameter(parameter_name)
                    parameter.data = parameter.data.float()
            if device.type != "meta":
                mtp.to_empty(device=device)
                if base is not None:
                    base.to_empty(device=device)
                # Rebuild destination views after replacing the meta parameters.
                for key, target in cls._load_targets(mtp, base, weight_map).items():
                    cls._load_weight(target, *sources[key])
                if base is not None and config.tie_word_embeddings:
                    if not torch.equal(base.embed_tokens.weight, base.lm_head.weight):
                        raise NativeMTPCheckpointError("Tied embedding and head weights disagree")
                    base.lm_head.weight = base.embed_tokens.weight

        # Non-persistent RoPE must be real even on meta-only FSDP ranks.
        mtp._maybe_init_rope("cpu" if device.type == "meta" else device)
        if base is not None:
            base.requires_grad_(False)
        return mtp, base

    @staticmethod
    def _create_base(config, base_layers, dtype):
        return FakeBaseModel(
            FakeBaseConfig(
                num_hidden_layers=base_layers,
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                max_position_embeddings=config.max_position_embeddings,
                dtype=dtype,
                tie_word_embeddings=config.tie_word_embeddings,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                intermediate_size=config.moe_intermediate_size,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
                final_norm_type="rmsnorm",
            )
        )

    @staticmethod
    def _native_config(model_path):
        with (Path(model_path) / "config.json").open() as stream:
            data = json.load(stream)
        if data.get("model_type") != "deepseek_v4" or data.get("num_nextn_predict_layers") != 1:
            raise NativeMTPCheckpointError("Expected DeepSeek-V4 with exactly one native MTP layer")
        base_layers = data["num_hidden_layers"]
        schedule = data.get("layer_types")
        if schedule is None:
            schedule = [
                "sliding_attention" if ratio == 0 else "compressed"
                for ratio in data.get("compress_ratios", [])
            ]
        if len(schedule) != base_layers + 1 or schedule[-1] != "sliding_attention":
            raise NativeMTPCheckpointError(
                "Expected a sliding-attention MTP entry after the backbone layers"
            )
        mlp_schedule = data.get("mlp_layer_types")
        if mlp_schedule is not None and (
            len(mlp_schedule) != base_layers + 1 or mlp_schedule[-1] != "moe"
        ):
            raise NativeMTPCheckpointError("The native MTP layer must use score-routed MoE")
        quant = data.get("quantization_config", {})
        if quant and (
            quant.get("quant_method") != "fp8"
            or quant.get("scale_fmt") != "ue8m0"
            or quant.get("weight_block_size") != [128, 128]
        ):
            raise NativeMTPCheckpointError(
                "Expected native DSV4 Pro FP8 weights with 128x128 UE8M0 scales"
            )
        config_data = {
            k: v for k, v in data.items() if k not in ("quantization_config", "compress_ratios")
        }
        config_data.update(
            num_hidden_layers=1, layer_types=["sliding_attention"], mlp_layer_types=["moe"]
        )
        config = DeepseekV4Config(**config_data)
        config._attn_implementation = "flex_attention"
        config._experts_implementation = "grouped_mm"
        return config, base_layers

    @staticmethod
    def _native_tensors(state, config):
        """Map module state or lazy checkpoint slices to one native tensor at a time."""
        for source, target in _DIRECT_STATE_KEYS.items():
            yield "mtp.0." + source, state[target][:]
        gate_up = state["decoder.mlp.experts.gate_up_proj"]
        down = state["decoder.mlp.experts.down_proj"]
        width = config.moe_intermediate_size
        for i in range(config.n_routed_experts):
            yield f"mtp.0.ffn.experts.{i}.w1.weight", gate_up[i, :width]
            yield f"mtp.0.ffn.experts.{i}.w3.weight", gate_up[i, width:]
            yield f"mtp.0.ffn.experts.{i}.w2.weight", down[i]

    @classmethod
    def _load_targets(cls, mtp, base, weight_map):
        targets = dict(cls._native_tensors(mtp.state_dict(), mtp.config))
        if base is not None:
            targets["embed.weight"] = base.embed_tokens.weight
            head = "head.weight" if "head.weight" in weight_map else "embed.weight"
            if head == "embed.weight" and not base.config.tie_word_embeddings:
                raise NativeMTPCheckpointError(
                    "Checkpoint is missing head.weight and does not tie embeddings"
                )
            if head == "embed.weight":
                base.lm_head.weight = base.embed_tokens.weight
            else:
                targets[head] = base.lm_head.weight
            targets["norm.weight"] = base.norm.weight
        return targets

    @staticmethod
    def _checkpoint_tensors(model_path, weight_map, targets, stack):
        """Open lazy weight/scale pairs and validate all headers before allocating weights."""
        handles = {}

        def tensor_slice(key):
            if key not in weight_map:
                raise NativeMTPCheckpointError(f"Missing checkpoint tensor {key}")
            shard = weight_map[key]
            if shard not in handles:
                handles[shard] = stack.enter_context(
                    safe_open(str(Path(model_path) / shard), framework="pt", device="cpu")
                )
            return handles[shard].get_slice(key)

        sources = {}
        expected = set(targets)
        for key, target in targets.items():
            source = tensor_slice(key)
            shape, fmt = source.get_shape(), source.get_dtype()
            if fmt in ("I8", "U8"):
                shape[-1] *= 2
            if tuple(shape) != tuple(target.shape) or fmt not in (
                "I8",
                "U8",
                "F8_E4M3",
                "BF16",
                "F32",
                "F16",
            ):
                raise NativeMTPCheckpointError(
                    f"Unsupported tensor {key}: {fmt} {shape}, expected {tuple(target.shape)}"
                )
            scale = None
            if fmt in ("I8", "U8", "F8_E4M3"):
                if not key.endswith(".weight") or len(shape) != 2:
                    raise NativeMTPCheckpointError(f"Expected a quantized matrix at {key}")
                scale_key = key.removesuffix(".weight") + ".scale"
                scale = tensor_slice(scale_key)
                expected.add(scale_key)
                scale_shape = (
                    [math.ceil(shape[0] / 128), math.ceil(shape[1] / 128)]
                    if fmt == "F8_E4M3"
                    else [shape[0], shape[1] // 32]
                )
                if (
                    (fmt != "F8_E4M3" and shape[1] % 32)
                    or scale.get_shape() != scale_shape
                    or scale.get_dtype() not in ("U8", "F8_E8M0")
                ):
                    raise NativeMTPCheckpointError(f"Invalid native block scales for {key}")
            sources[key] = source, scale
        extra = {k for k in weight_map if k.startswith("mtp.")} - expected
        if extra:
            raise NativeMTPCheckpointError(f"Unrecognized native MTP tensors: {sorted(extra)}")
        return sources

    @staticmethod
    def _load_weight(target, source, scale):
        """Load one tensor or expert, using ModelOpt to dequantize native weights."""
        weight = source[:]
        if scale is not None:
            weight = weight.to(target.device)
            scale = scale[:].to(target.device).contiguous()
            if weight.dtype == torch.float8_e4m3fn:
                weight = dequantize_fp8_ue8m0_to_bf16(weight, scale)
            else:
                weight = MXFP4QTensor(
                    target.shape, target.dtype, weight.contiguous().view(torch.uint8)
                ).dequantize(scale=scale.view(torch.uint8), block_sizes={-1: 32})
        target.copy_(weight)

    @classmethod
    def export_tensors(cls, model_path, mtp, state, *, device):
        """Yield trained tensors in native DSV4 layouts, including quantization scales."""
        native_map = indexed_weight_map(model_path)
        with ExitStack() as stack:
            native = cls._checkpoint_tensors(
                model_path, native_map, cls._load_targets(mtp, None, native_map), stack
            )
            for key, value in cls._native_tensors(state, mtp.config):
                source_tensor, source_scale = native[key]
                fmt = source_tensor.get_dtype()
                # Only BF16 compute weights need rounding from Trainer's FP32 masters.
                dtype = torch.float32 if fmt == "F32" else torch.bfloat16
                value = value.to(device=device, dtype=dtype).contiguous()
                if source_scale is None:
                    encoded = value.to(
                        {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[fmt]
                    )
                elif fmt == "F8_E4M3":
                    encoded, scale = quantize_bf16_to_fp8_ue8m0(value)
                else:
                    quantized, scale = MXFP4QTensor.quantize(value, block_size=32)
                    encoded = quantized._quantized_data.view(
                        torch.int8 if fmt == "I8" else torch.uint8
                    )
                yield key, encoded
                if source_scale is not None:
                    scale = scale.reshape(source_scale.get_shape()).view(
                        torch.uint8 if source_scale.get_dtype() == "U8" else torch.float8_e8m0fnu
                    )
                    yield key.removesuffix(".weight") + ".scale", scale
