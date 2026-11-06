# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit conversion of published V4.1 text tensors to the native fused layout."""

import copy
import re

import torch

from ._hf_engram import V41NgramHash
from .configuration_hf import DeepseekV41Config
from .moe import _decode_weight

__all__ = []


def _canonical_key(name):
    name = name.removeprefix("model.")
    name = name.replace("self_attn", "attn").replace(".mlp.", ".ffn.")
    name = name.replace("weight_scale_inv", "scale").replace("e_score_correction_bias", "bias")
    if name.startswith("embed_tokens."):
        name = name.replace("embed_tokens.", "embed.", 1)
    if name.startswith(("head.", "lm_head.")):
        return "lm_head." + name.split(".", 1)[1]
    return "model." + name


def load_published_text_checkpoint(
    cls, config, source, *, tokenizer, dtype, device, allow_auxiliary
):
    config = (
        copy.deepcopy(config)
        if isinstance(config, DeepseekV41Config)
        else DeepseekV41Config(**config)
    )
    canonical, auxiliary = {}, []
    for name, tensor in source.items():
        raw = name.removeprefix("model.")
        if raw.startswith(("vision.", "aligner.", "image_", "mtp.")):
            auxiliary.append(name)
            continue
        key = _canonical_key(name)
        if key in canonical:
            raise ValueError(f"Duplicate V4.1 checkpoint mapping for {key!r}.")
        canonical[key] = tensor
    if auxiliary and not allow_auxiliary:
        raise ValueError(
            "Source checkpoint contains auxiliary vision/DSpark tensors; "
            "explicitly set allow_auxiliary=True to discard them."
        )
    with torch.device("meta"):
        model = cls(config)
    target_device = torch.device(device or "cpu")
    decoded, consumed, quantized = {}, set(), []

    def read(name, shape, *, table=False):
        if name not in canonical:
            raise ValueError(f"Missing V4.1 text checkpoint tensor {name!r}.")
        consumed.add(name)
        weight = canonical[name]
        scale_name = name.removesuffix("weight") + "scale"
        scale = canonical.get(scale_name) if name.endswith("weight") else None
        if scale is not None:
            consumed.add(scale_name)
        if table and weight.dtype == torch.float8_e4m3fn:
            if weight.shape != shape or scale is None or scale.shape != (shape[0], shape[1] // 32):
                raise ValueError(
                    "Engram FP8 tables require one scale per row and group of 32 columns."
                )
            value = weight.float().unflatten(-1, (-1, 32)) * scale.float().unsqueeze(-1)
            return value.flatten(-2), False
        if len(shape) == 3:
            if (
                weight.shape != shape
                or scale is not None
                or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            ):
                raise ValueError(
                    "A native fused checkpoint tensor must be floating and have its full logical shape."
                )
            return weight, False
        return _decode_weight(weight, scale, shape)

    for name, parameter in model.named_parameters():
        fused = re.fullmatch(r"(.+\.experts)\.(gate_up_proj|down_proj)", name)
        if fused:
            prefix, projection = fused.groups()
            value = torch.empty(parameter.shape, device=target_device, dtype=dtype)
            if name in canonical:
                native, _ = read(name, parameter.shape)
                value.copy_(native.to(device=target_device, dtype=dtype))
            else:
                for expert in range(parameter.shape[0]):
                    if projection == "gate_up_proj":
                        width = config.moe_intermediate_size
                        gate, q_gate = read(
                            f"{prefix}.{expert}.w1.weight", (width, config.hidden_size)
                        )
                        up, q_up = read(f"{prefix}.{expert}.w3.weight", (width, config.hidden_size))
                        if q_gate != q_up:
                            raise ValueError(
                                "Fusing source experts requires matching gate/up activation precision."
                            )
                        value[expert, :width].copy_(gate.to(device=target_device, dtype=dtype))
                        value[expert, width:].copy_(up.to(device=target_device, dtype=dtype))
                        if q_gate:
                            quantized.extend((f"{prefix}.{expert}.w1", f"{prefix}.{expert}.w3"))
                    else:
                        down, q_down = read(
                            f"{prefix}.{expert}.w2.weight",
                            (config.hidden_size, config.moe_intermediate_size),
                        )
                        value[expert].copy_(down.to(device=target_device, dtype=dtype))
                        if q_down:
                            quantized.append(f"{prefix}.{expert}.w2")
            decoded[name] = value
        else:
            value, rounded = read(name, parameter.shape, table=".engram.embed." in name)
            fp32 = any(
                part.startswith("hc_") or part == "gate" for part in name.split(".")
            ) or name.endswith("attn_sink")
            if ".compressor.wgate." in name:
                fp32 = True
            if ".compressor.wkv." in name:
                layer_id = int(name.split(".")[2])
                fp32 = config.compress_ratios[layer_id] > 1
            decoded[name] = (
                value.to(device=target_device, dtype=torch.float32 if fp32 else dtype)
                .detach()
                .clone()
            )
            if rounded and not name.endswith("wo_a.weight"):
                quantized.append(name.removesuffix(".weight"))
    if model.model.engram_hash is not None:
        model.model.engram_hash = V41NgramHash(config).to(target_device)
        if tokenizer is not None:
            model.set_engram_tokenizer(tokenizer)
        if not model.model.engram_hash.ready:
            raise ValueError(
                "A published Engram checkpoint requires its tokenizer or an exact engram_token_map."
            )
        key = "model.engram_hash.token_map"
        decoded[key] = model.model.engram_hash.token_map
        if key in canonical:
            consumed.add(key)
            if not torch.equal(canonical[key].to(target_device), decoded[key]):
                raise ValueError("Checkpoint Engram token map disagrees with its configuration.")
    unexpected = set(canonical) - consumed
    if unexpected:
        raise ValueError(f"Unexpected V4.1 text checkpoint tensors: {sorted(unexpected)}")
    model.load_state_dict(decoded, assign=True)
    config.checkpoint_quantized_projections = sorted(
        set(quantized) | set(config.checkpoint_quantized_projections)
    )
    model._apply_checkpoint_precision()
    model.discarded_auxiliary_checkpoint_keys = tuple(auxiliary)
    return model
