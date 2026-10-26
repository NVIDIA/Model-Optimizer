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

"""Derive per-rank benchmark shapes from a Transformers model on meta tensors.

The script walks the instantiated model's Linear modules, fuses Q/K/V and
gate/up projections, recognizes Mamba 2, GatedDeltaNet, and common
routed-expert layouts, and applies the common serving/export TP layout. It
never calls a checkpoint weight loader.
When a decoder layout is unsupported, the derived shapes are still printed and
the script exits nonzero; benchmark the missing shapes directly with
benchmark_via_builtin.py.
"""

import argparse
import importlib.util
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ShapeError(ValueError):
    """A model layer cannot be represented by the supported benchmark layout."""


@dataclass(frozen=True)
class _MoeShape:
    hidden: int
    intermediate: int
    experts: int
    top_k: int
    activation: str | None = None
    # The expert container's module path. Metadata, not identity: two layouts
    # that differ only by path are still one shape for dedup purposes.
    name: str = field(default="experts", compare=False)


@dataclass(frozen=True)
class _ExpertShape:
    hidden: int
    intermediate: int
    gated: bool


@dataclass(frozen=True)
class _Kernel:
    """One derived per-rank GEMM: N x K, labeled by its source module path(s)."""

    n: int
    k: int
    label: str


@dataclass(frozen=True)
class _MambaLayout:
    in_shape: tuple[int, int]
    out_shape: tuple[int, int]
    intermediate: int
    heads: int
    groups: int
    state: int


@dataclass(frozen=True)
class _GdnLayout:
    out_shape: tuple[int, int]
    num_k_heads: int
    num_v_heads: int
    key_dim: int
    value_dim: int


_PROJECTIONS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "gate_up_proj",
    "down_proj",
}
_RESERVED = {
    "--nks",
    "--moe_name",
    "--moe_hidden_size",
    "--moe_intermediate_size",
    "--moe_num_experts",
    "--moe_top_k",
    "--moe_activation_type",
}

# Modules deliberately outside the benchmarked GEMM list fall into two
# exclusion mechanisms:
# 1. Positional: embeddings, the LM head, and anything else outside the
#    decoder blocks never enter the audit (the `layers.<i>` position filter
#    in _unsupported_decoder_linears).
# 2. Name-based: routing and gating projections that live inside decoder
#    blocks are excluded by these names. They are never quantized in
#    deployment recipes and vLLM dispatches them through specialized (often
#    FP32-output) paths, so a standard GEMM row would not model them anyway.
_ROUTER_PATH_PARTS = {"router", "routers"}
_GATING_LEAF_NAMES = {"gate", "router", "router_proj", "shared_expert_gate"}


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _load_meta_model(model_ref: str, trust_remote_code: bool, revision: str | None):
    # Transformers and Accelerate are optional, heavy ModelOpt dependencies.
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise ShapeError("install ModelOpt with the 'hf' extra") from exc

    path = Path(model_ref).expanduser()
    ref = str(path) if path.exists() else model_ref
    try:
        config = AutoConfig.from_pretrained(
            ref, trust_remote_code=trust_remote_code, revision=revision
        )
        model_kwargs = {"trust_remote_code": trust_remote_code}
        auto_map = getattr(config, "auto_map", {}) or {}
        if revision and trust_remote_code and "AutoModelForCausalLM" in auto_map:
            model_kwargs["code_revision"] = revision
        with init_empty_weights(include_buffers=True):
            try:
                model = AutoModelForCausalLM.from_config(config, **model_kwargs)
            except Exception:
                text_config = getattr(config, "text_config", None)
                if text_config is None:
                    raise
                # Multimodal wrapper configs build their text decoder
                # directly; vision towers are outside the benchmark scope.
                model = AutoModelForCausalLM.from_config(text_config, **model_kwargs)
    except Exception as exc:
        raise ShapeError(f"could not construct {model_ref!r} on meta tensors: {exc}") from exc

    tensors = list(model.named_parameters()) + list(model.named_buffers())
    materialized = [name for name, tensor in tensors if not tensor.is_meta]
    if materialized:
        raise ShapeError(f"model construction allocated tensors: {', '.join(materialized[:3])}")
    return config, model


def _linear_shape(module: Any) -> tuple[int, int] | None:
    if hasattr(module, "out_features") and hasattr(module, "in_features"):
        return int(module.out_features), int(module.in_features)
    return None


def _divide(value: int, size: int, label: str) -> int:
    if value % size:
        raise ShapeError(f"{label}={value} is not divisible by {size}")
    return value // size


def _path_label(parent: str, leaf: str) -> str:
    """Label a GEMM by its module path with layer indices normalized to ``*``."""
    path = f"{parent}.{leaf}" if parent else leaf
    return re.sub(r"(?<=\.)\d+(?=\.|$)", "*", path)


def _fused_qkv(
    q: tuple[int, int],
    k: tuple[int, int],
    v: tuple[int, int],
    head_dim: int,
    tp: int,
    parent: str,
) -> _Kernel:
    if q[1] != k[1] or q[1] != v[1] or k[0] != v[0]:
        raise ShapeError(f"unsupported Q/K/V shapes under {parent}")
    q_heads = _divide(q[0], head_dim, f"{parent}.q_proj output")
    kv_heads = _divide(k[0], head_dim, f"{parent}.k_proj output")
    if kv_heads >= tp:
        _divide(q_heads, tp, f"{parent}.q_proj heads")
        _divide(kv_heads, tp, f"{parent}.k_proj heads")
        local_n = _divide(q[0] + k[0] + v[0], tp, f"{parent}.qkv")
    else:
        local_q = _divide(q_heads, tp, f"{parent}.q_proj heads") * head_dim
        _divide(tp, kv_heads, "TP/KV replication ratio")
        local_n = local_q + 2 * head_dim
    label = "|".join(_path_label(parent, leaf) for leaf in ("q_proj", "k_proj", "v_proj"))
    return _Kernel(local_n, q[1], label)


def _dense_kernels(model: Any, config: Any, tp: int) -> list[_Kernel]:
    groups: dict[str, dict[str, tuple[int, int]]] = {}
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        parts = name.split(".")
        # Routed experts are excluded here and handled by _moe_shapes. Shared
        # experts (for example `.shared_experts.up_proj`) intentionally do not
        # match the filter: they run densely for every token, so their
        # projections belong in the dense GEMM list.
        if (
            leaf not in _PROJECTIONS
            or ".experts." in name
            or ".local_experts." in name
            or any(part in _ROUTER_PATH_PARTS for part in parts[:-1])
        ):
            continue
        shape = _linear_shape(module)
        if shape:
            groups.setdefault(name.rpartition(".")[0], {})[leaf] = shape

    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    kernels = []
    for parent, layers in groups.items():
        kernels += _parent_kernels(parent, layers, int(head_dim), tp)
    return list(dict.fromkeys(kernels))


def _parent_kernels(
    parent: str, layers: dict[str, tuple[int, int]], head_dim: int, tp: int
) -> list[_Kernel]:
    """Derive one parent module's GEMMs from its recognized projections."""
    kernels = []
    qkv = {"q_proj", "k_proj", "v_proj"}
    present_qkv = qkv.intersection(layers)
    if present_qkv:
        if present_qkv != qkv:
            raise ShapeError(f"incomplete Q/K/V projections under {parent}")
        kernels.append(
            _fused_qkv(
                layers["q_proj"],
                layers["k_proj"],
                layers["v_proj"],
                head_dim,
                tp,
                parent,
            )
        )
    if "o_proj" in layers:
        # Row-parallel: TP shards K.
        n, k = layers["o_proj"]
        kernels.append(
            _Kernel(n, _divide(k, tp, f"{parent}.o_proj"), _path_label(parent, "o_proj"))
        )

    if "gate_proj" in layers and "up_proj" in layers:
        gate_n, gate_k = layers["gate_proj"]
        up_n, up_k = layers["up_proj"]
        if gate_k != up_k:
            raise ShapeError(f"gate/up inputs differ under {parent}")
        # vLLM shards gate and up individually before merging them, so
        # each output must divide by TP, not just their sum.
        n = _divide(gate_n, tp, f"{parent}.gate_proj") + _divide(up_n, tp, f"{parent}.up_proj")
        label = "|".join(_path_label(parent, leaf) for leaf in ("gate_proj", "up_proj"))
        kernels.append(_Kernel(n, gate_k, label))
    elif "gate_proj" in layers:
        raise ShapeError(f"gate projection has no matching up projection under {parent}")
    elif "up_proj" in layers:
        # Column-parallel: TP shards N.
        n, k = layers["up_proj"]
        kernels.append(
            _Kernel(_divide(n, tp, f"{parent}.up_proj"), k, _path_label(parent, "up_proj"))
        )
    if "gate_up_proj" in layers:
        # Column-parallel: TP shards N (the checkpoint pre-fused gate and up).
        n, k = layers["gate_up_proj"]
        kernels.append(
            _Kernel(
                _divide(n, tp, f"{parent}.gate_up_proj"), k, _path_label(parent, "gate_up_proj")
            )
        )
    if "down_proj" in layers:
        # Row-parallel: TP shards K.
        n, k = layers["down_proj"]
        kernels.append(
            _Kernel(n, _divide(k, tp, f"{parent}.down_proj"), _path_label(parent, "down_proj"))
        )
    return kernels


def _mamba_layout(module: Any) -> _MambaLayout | None:
    in_shape = _linear_shape(getattr(module, "in_proj", None))
    out_shape = _linear_shape(getattr(module, "out_proj", None))
    intermediate = getattr(module, "intermediate_size", None)
    heads = getattr(module, "num_heads", None)
    groups = getattr(module, "n_groups", None)
    state = getattr(module, "ssm_state_size", None)
    if (
        in_shape is None
        or out_shape is None
        or intermediate is None
        or heads is None
        or groups is None
        or state is None
    ):
        return None
    return _MambaLayout(in_shape, out_shape, int(intermediate), int(heads), int(groups), int(state))


def _mamba_kernels(model: Any, tp: int) -> list[_Kernel]:
    kernels = []
    for name, module in model.named_modules():
        layout = _mamba_layout(module)
        if layout is None:
            continue
        hidden = layout.in_shape[1]
        expected_in = 2 * layout.intermediate + 2 * layout.groups * layout.state + layout.heads
        if layout.in_shape[0] != expected_in or layout.out_shape != (hidden, layout.intermediate):
            raise ShapeError(f"unsupported Mamba projection shapes under {name}")

        local_intermediate = _divide(layout.intermediate, tp, f"{name}.intermediate_size")
        local_heads = _divide(layout.heads, tp, f"{name}.num_heads")
        if layout.groups % tp == 0:
            local_groups = layout.groups // tp
        elif layout.groups == 1:
            local_groups = 1
        else:
            raise ShapeError(f"{name}.n_groups={layout.groups} is not divisible by TP={tp}")
        local_in = 2 * local_intermediate + 2 * local_groups * layout.state + local_heads
        kernels.extend(
            [
                _Kernel(local_in, hidden, _path_label(name, "in_proj")),
                _Kernel(hidden, local_intermediate, _path_label(name, "out_proj")),
            ]
        )
    return list(dict.fromkeys(kernels))


def _gdn_layout(module: Any) -> _GdnLayout | None:
    out_shape = _linear_shape(getattr(module, "out_proj", None))
    num_k_heads = getattr(module, "num_k_heads", None)
    num_v_heads = getattr(module, "num_v_heads", None)
    key_dim = getattr(module, "key_dim", None)
    value_dim = getattr(module, "value_dim", None)
    has_input = (
        getattr(module, "in_proj_qkvz", None) is not None
        or getattr(module, "in_proj_qkv", None) is not None
    )
    if (
        out_shape is None
        or not has_input
        or num_k_heads is None
        or num_v_heads is None
        or key_dim is None
        or value_dim is None
    ):
        return None
    return _GdnLayout(out_shape, int(num_k_heads), int(num_v_heads), int(key_dim), int(value_dim))


def _gdn_kernels(model: Any, tp: int) -> list[_Kernel]:
    kernels = []
    for name, module in model.named_modules():
        layout = _gdn_layout(module)
        if layout is None:
            continue
        key_dim, value_dim = layout.key_dim, layout.value_dim
        num_v_heads = layout.num_v_heads
        hidden = layout.out_shape[0]
        fused_qkvz = _linear_shape(getattr(module, "in_proj_qkvz", None))
        qkvz_label = _path_label(name, "in_proj_qkvz")
        ba_label = _path_label(name, "in_proj_ba")
        if fused_qkvz is not None:
            # Qwen3-Next stores qkvz and ba pre-fused.
            ba = _linear_shape(getattr(module, "in_proj_ba", None))
            valid = fused_qkvz == (2 * key_dim + 2 * value_dim, hidden) and ba == (
                2 * num_v_heads,
                hidden,
            )
        else:
            # Qwen3.5 stores them split, but vLLM's shared GDN mixer merges
            # qkv+z and b+a into the same two per-rank GEMMs either way.
            qkvz_label = "|".join(_path_label(name, leaf) for leaf in ("in_proj_qkv", "in_proj_z"))
            ba_label = "|".join(_path_label(name, leaf) for leaf in ("in_proj_b", "in_proj_a"))
            shapes = {
                leaf: _linear_shape(getattr(module, leaf, None))
                for leaf in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
            }
            valid = (
                shapes["in_proj_qkv"] == (2 * key_dim + value_dim, hidden)
                and shapes["in_proj_z"] == (value_dim, hidden)
                and shapes["in_proj_b"] == (num_v_heads, hidden)
                and shapes["in_proj_a"] == (num_v_heads, hidden)
            )
        if not valid or layout.out_shape != (hidden, value_dim):
            raise ShapeError(f"unsupported GatedDeltaNet projection shapes under {name}")
        _divide(layout.num_k_heads, tp, f"{name}.linear_num_key_heads")
        local_v_heads = _divide(num_v_heads, tp, f"{name}.linear_num_value_heads")
        kernels.extend(
            [
                _Kernel(
                    _divide(2 * key_dim + 2 * value_dim, tp, f"{name}.qkvz"),
                    hidden,
                    qkvz_label,
                ),
                _Kernel(2 * local_v_heads, hidden, ba_label),
                _Kernel(
                    hidden,
                    _divide(value_dim, tp, f"{name}.value_dim"),
                    _path_label(name, "out_proj"),
                ),
            ]
        )
    return list(dict.fromkeys(kernels))


def _expert_shape(module: Any) -> _ExpertShape | None:
    gate = getattr(module, "gate_proj", None)
    up = getattr(module, "up_proj", None)
    down = getattr(module, "down_proj", None)
    if gate is None and getattr(module, "w1", None) is not None:
        gate = getattr(module, "w1", None)
        up = getattr(module, "w3", None)
        down = getattr(module, "w2", None)
    if gate is not None:
        gate_shape, up_shape, down_shape = map(_linear_shape, (gate, up, down))
        if gate_shape is None or up_shape is None or down_shape is None:
            raise ShapeError("incomplete gated expert Linear layout")
        if gate_shape != up_shape or down_shape != (gate_shape[1], gate_shape[0]):
            raise ShapeError("unsupported gated expert Linear shapes")
        return _ExpertShape(gate_shape[1], gate_shape[0], True)

    up_shape, down_shape = map(_linear_shape, (up, down))
    if up_shape is None and down_shape is None:
        return None
    if up_shape is None or down_shape is None or down_shape != (up_shape[1], up_shape[0]):
        raise ShapeError("unsupported non-gated expert Linear shapes")
    return _ExpertShape(up_shape[1], up_shape[0], False)


def _stacked_expert_shape(
    first: Any, down: Any, factor: int, name: str, expected_hidden: int | None
) -> _ExpertShape:
    if first.ndim != 3 or down.ndim != 3 or first.shape[0] != down.shape[0]:
        raise ShapeError(f"unsupported stacked experts at {name}")

    first_shape = tuple(int(value) for value in first.shape)
    down_shape = tuple(int(value) for value in down.shape)
    candidates = []
    if first_shape[1] % factor == 0:
        hidden, intermediate = first_shape[2], first_shape[1] // factor
        if down_shape[1:] == (hidden, intermediate):
            candidates.append((hidden, intermediate))
    if first_shape[2] % factor == 0:
        hidden, intermediate = first_shape[1], first_shape[2] // factor
        if down_shape[1:] == (intermediate, hidden):
            candidates.append((hidden, intermediate))
    if expected_hidden is not None:
        candidates = [candidate for candidate in candidates if candidate[0] == expected_hidden]
    if len(set(candidates)) != 1:
        raise ShapeError(f"unsupported stacked expert projection shapes at {name}")
    hidden, intermediate = candidates[0]
    return _ExpertShape(hidden, intermediate, factor == 2)


def _moe_activation(config: Any, gated: bool) -> str | None:
    configured = getattr(config, "mlp_hidden_act", None) or getattr(config, "hidden_act", None)
    normalized = str(configured).lower().replace("-", "_")
    if gated:
        if normalized in {"silu", "swiglu", "swish"}:
            return "Swiglu"
        # vLLM's FlashInfer MoE path serves only the tanh-approximation GELU
        # ("gelu_tanh", "gelu_pytorch_tanh", and "gelu_new" share that
        # formula). Exact gelu and quick_gelu take non-FlashInfer backends in
        # vLLM, so they are rejected here rather than timed as a proxy.
        if normalized in {"gelu_tanh", "gelu_pytorch_tanh", "gelu_new"}:
            return "Geglu"
        raise ShapeError(
            f"unsupported gated MoE activation {configured!r}; vLLM's FlashInfer MoE "
            "serves only SiLU/SwiGLU and tanh-GELU"
        )
    activations = {
        "gelu": "Gelu",
        "identity": "Identity",
        "relu": "Relu",
        "relu2": "Relu2",
        "relu_squared": "Relu2",
        "silu": "Silu",
    }
    if normalized not in activations:
        raise ShapeError(f"unsupported non-gated MoE activation {configured!r}")
    return activations[normalized]


# Fallback copy of ModelOpt's _ACTIVE_MOE_TOP_K_ATTRS for environments without
# ModelOpt installed; a test asserts it stays in sync with the canonical list.
_MOE_TOP_K_ATTRS_FALLBACK = (
    "num_experts_per_tok",
    "num_experts_per_token",
    "moe_top_k",
    "top_k",
    "num_selected_experts",
)


def _top_k(config: Any) -> int | None:
    # ModelOpt's AutoQuantize cost model owns the canonical attribute list, so
    # benchmark rows and AutoQuantize agree on how a config declares top_k.
    try:
        from modelopt.torch.quantization._auto_quantize_cost import _ACTIVE_MOE_TOP_K_ATTRS

        attrs = _ACTIVE_MOE_TOP_K_ATTRS
    except ImportError:
        attrs = _MOE_TOP_K_ATTRS_FALLBACK

    for attr in attrs:
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    return None


def _moe_shapes(model: Any, config: Any) -> set[_MoeShape]:
    shapes = set()
    top_k = _top_k(config)
    for name, module in model.named_modules():
        expert_container = name.rsplit(".", 1)[-1] in {"experts", "local_experts"}
        if expert_container:
            expert_modules = list(module.children())
            shape = _expert_shape(expert_modules[0]) if expert_modules else None
            if shape:
                if top_k is None:
                    raise ShapeError("could not determine MoE top_k")
                if any(_expert_shape(expert) != shape for expert in expert_modules[1:]):
                    raise ShapeError(f"experts under {name} do not share one Linear layout")
                shapes.add(
                    _MoeShape(
                        shape.hidden,
                        shape.intermediate,
                        len(expert_modules),
                        top_k,
                        _moe_activation(config, shape.gated),
                        _path_label(*name.rpartition(".")[::2]),
                    )
                )

        params = dict(module.named_parameters(recurse=False))
        down = params.get("down_proj")
        if params.get("gate_up_proj") is not None:
            first, factor = params["gate_up_proj"], 2
        elif params.get("up_proj") is not None:
            first, factor = params["up_proj"], 1
        else:
            expert_params = [
                f"{param_name}{tuple(param.shape)}"
                for param_name, param in params.items()
                if param.ndim >= 2
            ]
            if expert_container and expert_params:
                raise ShapeError(
                    f"unsupported stacked expert parameters at {name}: " + ", ".join(expert_params)
                )
            continue
        if down is None:
            raise ShapeError(f"stacked experts at {name} have no down projection")
        if top_k is None:
            raise ShapeError("could not determine MoE top_k")
        expected_hidden = getattr(config, "moe_latent_size", None) or getattr(
            config, "hidden_size", None
        )
        shape = _stacked_expert_shape(
            first,
            down,
            factor,
            name,
            int(expected_hidden) if expected_hidden is not None else None,
        )
        shapes.add(
            _MoeShape(
                shape.hidden,
                shape.intermediate,
                int(first.shape[0]),
                top_k,
                _moe_activation(config, shape.gated),
                _path_label(*name.rpartition(".")[::2]),
            )
        )
    return shapes


def _declared_expert_count(config: Any) -> int | None:
    if _top_k(config) is None:
        return None
    for attr in ("n_routed_experts", "num_local_experts", "num_experts"):
        value = getattr(config, attr, None)
        if value is not None and int(value) > 0:
            return int(value)
    return None


def _mixer_claimed_projections(model: Any) -> set[str]:
    """Full paths of the Linears the Mamba and GDN recognizers already derive."""
    claimed: set[str] = set()
    for parent, module in model.named_modules():
        if _mamba_layout(module) is not None:
            claimed.update({f"{parent}.in_proj", f"{parent}.out_proj"})
        if _gdn_layout(module) is not None:
            claimed.update(
                f"{parent}.{leaf}"
                for leaf in (
                    "in_proj_qkvz",
                    "in_proj_ba",
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_b",
                    "in_proj_a",
                    "out_proj",
                )
            )
    return claimed


def _unsupported_decoder_linears(
    model: Any, routed_experts_handled: bool = False
) -> list[tuple[str, int, int]]:
    claimed = _mixer_claimed_projections(model)
    layouts: dict[tuple[str, int, int], str] = {}
    for name, module in model.named_modules():
        shape = _linear_shape(module)
        if shape is None:
            continue
        parts = name.split(".")
        in_decoder = any(
            part in {"block", "blocks", "h", "layer", "layers"}
            and index + 1 < len(parts)
            and parts[index + 1].isdigit()
            for index, part in enumerate(parts)
        )
        leaf = parts[-1]
        if not in_decoder:
            continue
        if any(part in _ROUTER_PATH_PARTS for part in parts):
            continue
        if routed_experts_handled and any(part in {"experts", "local_experts"} for part in parts):
            continue
        if leaf in _PROJECTIONS or leaf in _GATING_LEAF_NAMES or name in claimed:
            continue
        layouts.setdefault((leaf, *shape), name)
    return [(name, n, k) for (leaf, n, k), name in layouts.items()]


def _audited_moe_shape(model: Any, config: Any) -> tuple[_MoeShape | None, bool, list[str]]:
    """Derive the model's global routed-expert shape, or its audit problems.

    Returns ``(shape, experts_recognized, problems)``; the shape is ``None``
    whenever any problem is found, so the audit findings are reported instead
    of a masking per-rank ShapeError.
    """
    problems: list[str] = []
    try:
        moe_shapes = _moe_shapes(model, config)
    except ShapeError as exc:
        moe_shapes = set()
        problems.append(str(exc))
    declared_experts = _declared_expert_count(config)
    if not problems and declared_experts is not None:
        if not moe_shapes:
            problems.append(
                f"model declares {declared_experts} routed experts but no supported expert "
                "GEMM layout was found"
            )
        elif any(shape.experts != declared_experts for shape in moe_shapes):
            found = sorted({shape.experts for shape in moe_shapes})
            problems.append(
                f"model declares {declared_experts} routed experts but instantiated layouts "
                f"have expert counts {found}"
            )
    experts_recognized = bool(moe_shapes)
    if len(moe_shapes) > 1:
        problems.append("model contains multiple routed-expert layouts")
    if problems:
        moe_shapes = set()
    return next(iter(moe_shapes), None), experts_recognized, problems


def _shard_moe(moe: _MoeShape, tp: int, ep: int) -> _MoeShape:
    """Apply vLLM's EP/TP partitioning to the global MoE shape."""
    if ep != 1 and ep % tp:
        raise ShapeError(
            f"EP={ep} is not a multiple of TP={tp}; vLLM expert parallelism spans TP x DP, "
            "so no modeled serving layout matches this combination — if it is intentional "
            "(e.g. Megatron-style EP), benchmark the per-rank expert shape directly with "
            "benchmark_via_builtin.py"
        )
    local_experts = _divide(moe.experts, ep, "expert count")
    intermediate = moe.intermediate
    if ep == 1:
        intermediate = _divide(intermediate, tp, "expert intermediate size")
    if moe.top_k > local_experts:
        raise ShapeError("top_k exceeds the per-rank expert count")
    return _MoeShape(moe.hidden, intermediate, local_experts, moe.top_k, moe.activation, moe.name)


def _inspect_model(
    model: Any, config: Any, tp: int, ep: int
) -> tuple[list[_Kernel], _MoeShape | None, list[str]]:
    config = getattr(config, "text_config", None) or config
    kernels = (
        _dense_kernels(model, config, tp) + _mamba_kernels(model, tp) + _gdn_kernels(model, tp)
    )
    moe, experts_recognized, problems = _audited_moe_shape(model, config)
    if moe is None:
        if ep != 1 and not problems:
            raise ShapeError("EP requires routed experts")
    else:
        moe = _shard_moe(moe, tp, ep)
    unsupported = _unsupported_decoder_linears(model, routed_experts_handled=experts_recognized)
    if unsupported:
        details = ", ".join(f"{name} ({n}x{k})" for name, n, k in unsupported)
        problems.append(f"unsupported decoder Linear GEMM layout(s): {details}")
    if not kernels and moe is None and not problems:
        raise ShapeError("no dense benchmark shapes found")
    return kernels, moe, problems


def _command(
    kernels: list[_Kernel],
    moe: _MoeShape | None,
    passthrough: list[str],
) -> list[str]:
    command: list[str] = []
    if kernels:
        # One N,K,NAME argument per derived kernel: same-shape kernels from
        # different modules keep separate names and become duplicated rows.
        command += ["--nks", *(f"{kernel.n},{kernel.k},{kernel.label}" for kernel in kernels)]
    if moe:
        command += [
            "--moe_hidden_size",
            str(moe.hidden),
            "--moe_intermediate_size",
            str(moe.intermediate),
            "--moe_num_experts",
            str(moe.experts),
            "--moe_top_k",
            str(moe.top_k),
            "--moe_name",
            moe.name,
        ]
        if moe.activation:
            command += ["--moe_activation_type", moe.activation]
    return command + passthrough


_RUNNER_PATH = Path(__file__).with_name("benchmark_via_builtin.py")


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("benchmark_via_builtin", _RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _print_preview(
    model: Any,
    config: Any,
    tp: int,
    ep: int,
    kernels: list[_Kernel],
    moe: _MoeShape | None,
    problems: list[str],
) -> None:
    print(f"# {type(model).__name__} ({getattr(config, 'model_type', '?')}), TP={tp}, EP={ep}")
    print(
        "# layout: Transformers meta model; fused QKV and gate/up; "
        "Mamba 2, GatedDeltaNet, and routed experts"
    )
    for kernel in dict.fromkeys(kernels):
        print(f"# {kernel.n}x{kernel.k} <- {kernel.label}")
    if moe:
        activation = f" activation={moe.activation}" if moe.activation else ""
        print(
            f"# MoE: H={moe.hidden} F={moe.intermediate} E={moe.experts} "
            f"top_k={moe.top_k}{activation}"
        )
        if ep > 1:
            print(
                f"# MoE sharding: EP={ep} partitions whole experts; "
                "expert width stays intact (expert-TP=1)"
            )
        elif tp > 1:
            print(f"# MoE sharding: TP={tp} shards the expert intermediate width (EP=1)")
    for problem in problems:
        print(f"# unsupported: {problem}")


def main() -> None:
    """Parse arguments, derive shapes, and optionally run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hub ID, local model directory, or config.json")
    parser.add_argument("--tp", type=_positive_int, default=1, help="tensor parallel size, e.g. 8")
    parser.add_argument("--ep", type=_positive_int, default=1, help="expert parallel size, e.g. 8")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision", help="Hugging Face branch, tag, or commit")
    parser.add_argument("--print_only", action="store_true")
    args, passthrough = parser.parse_known_args()
    for token in passthrough:
        if token.split("=", 1)[0] in _RESERVED:
            parser.error("derived --nks/--nk_names/--moe_* shapes cannot be overridden")

    try:
        config, model = _load_meta_model(args.model, args.trust_remote_code, args.revision)
        kernels, moe, problems = _inspect_model(model, config, args.tp, args.ep)
    except ShapeError as exc:
        parser.error(str(exc))

    _print_preview(model, config, args.tp, args.ep, kernels, moe, problems)
    if problems:
        parser.error(
            "the derived shapes above are incomplete; validate each unsupported layout's "
            "TP/EP sharding and benchmark it directly with benchmark_via_builtin.py"
        )
    command = _command(kernels, moe, passthrough)
    print(">>> " + shlex.join([sys.executable, str(_RUNNER_PATH), *command]))
    if not args.print_only:
        _load_runner().main(command)


if __name__ == "__main__":
    main()
