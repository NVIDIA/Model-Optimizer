# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiMo-V2.6 release metadata on the upstream differentiable MiMo implementation."""

import copy
import re

import torch
from transformers import AutoConfig, AutoModelForCausalLM, MiMoV2FlashConfig, MiMoV2FlashForCausalLM
from transformers.integrations import use_experts_implementation
from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashAttention,
    MiMoV2FlashExperts,
)

__all__ = ["MiMoV2Config", "MiMoV2ForCausalLM"]


@use_experts_implementation
class _MiMoV26Experts(MiMoV2FlashExperts):
    """Keep the release's FP32 routing accumulator in the eager fallback too."""

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states, dtype=top_k_weights.dtype)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            mask = mask.permute(2, 1, 0)
            active = mask.sum((-1, -2)).nonzero().flatten()
        for expert in active:
            if expert == self.num_experts:
                continue
            slot, tokens = torch.where(mask[expert])
            gate, up = torch.nn.functional.linear(
                hidden_states[tokens], self.gate_up_proj[expert]
            ).chunk(2, -1)
            values = torch.nn.functional.linear(self.act_fn(gate) * up, self.down_proj[expert])
            output.index_add_(0, tokens, values * top_k_weights[tokens, slot, None])
        return output.to(hidden_states.dtype)


class MiMoV2Config(MiMoV2FlashConfig):
    """Normalize the released Pro/Flash patterns without changing their geometry."""

    model_type = "mimo_v2"
    base_model_ep_plan = {
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "ep_dispatch_experts",
    }

    def __init__(self, **kwargs):
        """Translate released layer patterns and per-attention rotary metadata."""
        self.checkpoint_quantization_config = copy.deepcopy(
            kwargs.pop("quantization_config", kwargs.pop("checkpoint_quantization_config", None))
        )
        patterns = kwargs.get("hybrid_layer_pattern")
        if patterns is not None and "layer_types" not in kwargs:
            kwargs["layer_types"] = [
                "sliding_attention" if kind else "full_attention" for kind in patterns
            ]
        moe = kwargs.get("moe_layer_freq")
        if isinstance(moe, list) and "mlp_layer_types" not in kwargs:
            kwargs["mlp_layer_types"] = ["sparse" if active else "dense" for active in moe]
        if (
            not isinstance(kwargs.get("rope_parameters"), dict)
            or "full_attention" not in kwargs["rope_parameters"]
        ):
            factor = kwargs.get("partial_rotary_factor", 0.334)
            kwargs["rope_parameters"] = {
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": kwargs.get("rope_theta", 5_000_000.0),
                    "partial_rotary_factor": factor,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": kwargs.get("swa_rope_theta", 10_000.0),
                    "partial_rotary_factor": factor,
                },
            }
        if "layernorm_epsilon" in kwargs:
            kwargs.setdefault("rms_norm_eps", kwargs["layernorm_epsilon"])
        super().__init__(**kwargs)
        for parameters in self.rope_parameters.values():
            width = int(self.head_dim * parameters.get("partial_rotary_factor", 0.334))
            if width % 2:
                raise ValueError("MiMo release rotary dimension must be even.")


class MiMoV2ForCausalLM(MiMoV2FlashForCausalLM):
    """Use native fused experts while preserving each release's SWA KV-head count.

    This class accepts reconstructed floating weights. Raw MXFP4 bytes and fused
    release checkpoint projections require their explicit importer first.
    """

    config_class = MiMoV2Config

    def __init__(self, config):
        """Build upstream modules with the release's per-layer KV geometry."""
        super().__init__(config)
        wanted = getattr(config, "swa_num_key_value_heads", config.num_key_value_heads * 2)
        for index, layer in enumerate(self.model.layers):
            gate = getattr(layer.mlp, "gate", None)
            if gate is not None and hasattr(gate, "e_score_correction_bias"):
                # Native construction, unlike checkpoint loading, does not apply buffer dtype plans.
                gate.e_score_correction_bias = gate.e_score_correction_bias.float()
            if hasattr(layer.mlp, "experts"):
                experts = _MiMoV26Experts(config)
                experts.load_state_dict(layer.mlp.experts.state_dict(), assign=True)
                layer.mlp.experts = experts
            sliding = config.layer_types[index] == "sliding_attention"
            if sliding and wanted != config.num_key_value_heads * 2:
                if wanted % 2:
                    raise ValueError("MiMo native SWA KV-head count must be even.")
                local = copy.copy(config)
                local.num_key_value_heads = wanted // 2
                layer.self_attn = MiMoV2FlashAttention(local, index)
                layer.self_attn.apply(self._init_weights)
            sink = (
                getattr(config, "add_swa_attention_sink_bias", True)
                if sliding
                else getattr(config, "add_full_attention_sink_bias", False)
            )
            if not sink:
                layer.self_attn.sinks = None

    def load_published_state_dict(self, state_dict, *, strict=True, assign=False):
        """Load reconstructed floating text tensors from the release's fused-QKV layout.

        Packed MXFP4 and block-FP8 tensors need explicit weight reconstruction first;
        this method does not claim original quantized-kernel/activation parity.
        """
        expected, converted, experts = self.state_dict(), {}, {}

        def put(name, tensor):
            if name in converted:
                raise ValueError(f"Duplicate MiMo checkpoint mapping for {name!r}.")
            converted[name] = tensor

        for name, tensor in state_dict.items():
            if tensor.dtype not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            ) or name.endswith(("weight_scale", "weight_scale_inv", "weight_packed")):
                raise ValueError("MiMo loading requires reconstructed floating text weights.")
            match = re.fullmatch(r"(.+\.self_attn)\.qkv_proj\.(weight|bias)", name)
            if match:
                owner, kind = match.groups()
                keys = [owner + "." + part + "_proj." + kind for part in ("q", "k", "v")]
                if any(key not in expected for key in keys):
                    raise ValueError(f"Unexpected MiMo fused attention tensor {name!r}.")
                shapes = [expected[key].shape for key in keys]
                if tensor.shape != (sum(shape[0] for shape in shapes), *shapes[0][1:]):
                    raise ValueError(f"MiMo fused QKV shape mismatch for {name!r}.")
                for key, value in zip(
                    keys, tensor.split([shape[0] for shape in shapes], 0), strict=True
                ):
                    put(key, value)
                continue
            match = re.fullmatch(
                r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", name
            )
            if match:
                owner, expert, part = match.groups()
                experts.setdefault(owner, {})[int(expert), part] = tensor
            else:
                put(name.replace(".attention_sink_bias", ".sinks"), tensor)
        for owner, weights in experts.items():
            gate_up = owner + ".gate_up_proj"
            down = owner + ".down_proj"
            if gate_up not in expected or down not in expected:
                raise ValueError(f"Unexpected MiMo expert owner {owner!r}.")
            count, width, hidden = expected[gate_up].shape
            keys = {
                (i, part) for i in range(count) for part in ("gate_proj", "up_proj", "down_proj")
            }
            if weights.keys() != keys:
                raise ValueError(f"Incomplete or unexpected MiMo expert tensors for {owner!r}.")
            for (_, part), value in weights.items():
                shape = (hidden, width // 2) if part == "down_proj" else (width // 2, hidden)
                if tuple(value.shape) != shape:
                    raise ValueError(f"MiMo expert {part} must have shape {shape}.")
            gate_dtype, down_dtype = weights[0, "gate_proj"].dtype, weights[0, "down_proj"].dtype
            for i in range(count):
                gate_dtype = torch.promote_types(gate_dtype, weights[i, "gate_proj"].dtype)
                gate_dtype = torch.promote_types(gate_dtype, weights[i, "up_proj"].dtype)
                down_dtype = torch.promote_types(down_dtype, weights[i, "down_proj"].dtype)
            # Fill each bank once instead of retaining a second bank of concatenated experts.
            combined = weights[0, "gate_proj"].new_empty(expected[gate_up].shape, dtype=gate_dtype)
            projected = weights[0, "down_proj"].new_empty(expected[down].shape, dtype=down_dtype)
            for i in range(count):
                combined[i, : width // 2].copy_(weights[i, "gate_proj"])
                combined[i, width // 2 :].copy_(weights[i, "up_proj"])
                projected[i].copy_(weights[i, "down_proj"])
            put(gate_up, combined)
            put(down, projected)
        if strict and expected.keys() != converted.keys():
            missing, extra = expected.keys() - converted.keys(), converted.keys() - expected.keys()
            raise ValueError(
                f"MiMo checkpoint keys differ: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for name, tensor in converted.items():
            if name in expected and tensor.shape != expected[name].shape:
                raise ValueError(f"MiMo checkpoint shape mismatch for {name!r}.")
        return self.load_state_dict(converted, strict=strict, assign=assign)


AutoConfig.register("mimo_v2", MiMoV2Config, exist_ok=True)
AutoModelForCausalLM.register(MiMoV2Config, MiMoV2ForCausalLM, exist_ok=True)
