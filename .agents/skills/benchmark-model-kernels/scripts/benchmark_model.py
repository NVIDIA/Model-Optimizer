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
gate/up projections, recognizes Mamba 2 and common routed-expert layouts, maps
config routing fields to FlashInfer's MoE routing method, and applies the
common serving/export TP layout. It never calls a checkpoint weight loader.
When a decoder layout is unsupported, the derived shapes are still printed and
the script exits nonzero; benchmark the missing shapes directly with
benchmark_via_builtin.py.
"""

import argparse
import importlib.util
import shlex
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _ExpertShape:
    hidden: int
    intermediate: int
    gated: bool


@dataclass(frozen=True)
class _MoeRouting:
    method: str
    num_expert_group: int | None = None
    topk_group: int | None = None
    routed_scaling_factor: float | None = None
    use_routing_bias: bool = False


_Kernel = tuple[int, int, str]

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
    "--nk_names",
    "--moe_hidden_size",
    "--moe_intermediate_size",
    "--moe_num_experts",
    "--moe_top_k",
    "--moe_activation_type",
    "--moe_routing_method",
    "--moe_num_expert_group",
    "--moe_topk_group",
    "--moe_routed_scaling_factor",
    "--moe_use_routing_bias",
}

_ROUTER_NAMES = {"gate", "router", "router_proj"}


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
            model = AutoModelForCausalLM.from_config(config, **model_kwargs)
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
    return local_n, q[1], "fused_qkv"


def _dense_kernels(model: Any, config: Any, tp: int) -> list[_Kernel]:
    groups: dict[str, dict[str, tuple[int, int]]] = {}
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        parts = name.split(".")
        if (
            leaf not in _PROJECTIONS
            or ".experts." in name
            or ".local_experts." in name
            or any(part in {"router", "routers"} for part in parts[:-1])
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
                    int(head_dim),
                    tp,
                    parent,
                )
            )
        if "o_proj" in layers:
            n, k = layers["o_proj"]
            kernels.append((n, _divide(k, tp, f"{parent}.o_proj"), "attention_out"))

        if "gate_proj" in layers and "up_proj" in layers:
            gate_n, gate_k = layers["gate_proj"]
            up_n, up_k = layers["up_proj"]
            if gate_k != up_k:
                raise ShapeError(f"gate/up inputs differ under {parent}")
            n = _divide(gate_n + up_n, tp, f"{parent}.gate_up")
            kernels.append((n, gate_k, "fused_gate_up"))
        elif "gate_proj" in layers:
            raise ShapeError(f"gate projection has no matching up projection under {parent}")
        elif "up_proj" in layers:
            n, k = layers["up_proj"]
            kernels.append((_divide(n, tp, f"{parent}.up_proj"), k, "up"))
        if "gate_up_proj" in layers:
            n, k = layers["gate_up_proj"]
            kernels.append((_divide(n, tp, f"{parent}.gate_up_proj"), k, "fused_gate_up"))
        if "down_proj" in layers:
            n, k = layers["down_proj"]
            kernels.append((n, _divide(k, tp, f"{parent}.down_proj"), "down"))
    return list(dict.fromkeys(kernels))


def _mamba_layout(
    module: Any,
) -> tuple[tuple[int, int], tuple[int, int], int, int, int, int] | None:
    in_shape = _linear_shape(getattr(module, "in_proj", None))
    out_shape = _linear_shape(getattr(module, "out_proj", None))
    attrs = (
        getattr(module, "intermediate_size", None),
        getattr(module, "num_heads", None),
        getattr(module, "n_groups", None),
        getattr(module, "ssm_state_size", None),
    )
    if in_shape is None or out_shape is None or any(value is None for value in attrs):
        return None
    intermediate, heads, groups, state = (int(value) for value in attrs if value is not None)
    return in_shape, out_shape, intermediate, heads, groups, state


def _mamba_kernels(model: Any, tp: int) -> list[_Kernel]:
    kernels = []
    for name, module in model.named_modules():
        layout = _mamba_layout(module)
        if layout is None:
            continue
        in_shape, out_shape, intermediate, heads, groups, state = layout
        hidden = in_shape[1]
        expected_in = 2 * intermediate + 2 * groups * state + heads
        if in_shape[0] != expected_in or out_shape != (hidden, intermediate):
            raise ShapeError(f"unsupported Mamba projection shapes under {name}")

        local_intermediate = _divide(intermediate, tp, f"{name}.intermediate_size")
        local_heads = _divide(heads, tp, f"{name}.num_heads")
        if groups % tp == 0:
            local_groups = groups // tp
        elif groups == 1:
            local_groups = 1
        else:
            raise ShapeError(f"{name}.n_groups={groups} is not divisible by TP={tp}")
        local_in = 2 * local_intermediate + 2 * local_groups * state + local_heads
        kernels.extend(
            [
                (local_in, hidden, "mamba_in"),
                (hidden, local_intermediate, "mamba_out"),
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
        if normalized == "geglu" or normalized.startswith("gelu") or normalized == "quick_gelu":
            return "Geglu"
        raise ShapeError(f"unsupported gated MoE activation {configured!r}")
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


def _top_k(config: Any) -> int | None:
    for attr in ("num_experts_per_tok", "moe_top_k"):
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
            )
        )
    return shapes


def _moe_routing(model: Any, config: Any) -> _MoeRouting:
    # DeepSeek-style group-limited routing is declared by these three config
    # fields together; the score-correction bias is a tensor, not a config field.
    groups = getattr(config, "n_group", None)
    topk_group = getattr(config, "topk_group", None)
    scaling = getattr(config, "routed_scaling_factor", None)
    if groups is not None and topk_group is not None and scaling is not None:
        tensors = list(model.named_parameters()) + list(model.named_buffers())
        use_bias = any(name.rsplit(".", 1)[-1] == "e_score_correction_bias" for name, _ in tensors)
        return _MoeRouting("deepseek_v3", int(groups), int(topk_group), float(scaling), use_bias)
    if getattr(config, "norm_topk_prob", False):
        return _MoeRouting("renormalize")
    return _MoeRouting("topk")


def _declared_expert_count(config: Any) -> int | None:
    if _top_k(config) is None:
        return None
    for attr in ("n_routed_experts", "num_local_experts", "num_experts"):
        value = getattr(config, attr, None)
        if value is not None and int(value) > 0:
            return int(value)
    return None


def _unsupported_decoder_linears(
    model: Any, routed_experts_handled: bool = False
) -> list[tuple[str, int, int]]:
    mamba_projections = set()
    for parent, module in model.named_modules():
        if _mamba_layout(module) is not None:
            mamba_projections.update({f"{parent}.in_proj", f"{parent}.out_proj"})

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
        if any(part in {"router", "routers"} for part in parts):
            continue
        if routed_experts_handled and any(part in {"experts", "local_experts"} for part in parts):
            continue
        if leaf in _PROJECTIONS or leaf in _ROUTER_NAMES or name in mamba_projections:
            continue
        layouts.setdefault((leaf, *shape), name)
    return [(name, n, k) for (leaf, n, k), name in layouts.items()]


def _inspect_model(
    model: Any, config: Any, tp: int, ep: int
) -> tuple[list[_Kernel], _MoeShape | None, _MoeRouting | None, list[str]]:
    config = getattr(config, "text_config", None) or config
    kernels = _dense_kernels(model, config, tp) + _mamba_kernels(model, tp)
    problems = []
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
        # The expert audit is unresolved, so a derived MoE tuple is suspect;
        # skip its per-rank validation so the audit findings are reported
        # instead of a masking ShapeError.
        moe_shapes = set()
    moe = next(iter(moe_shapes), None)
    routing = None
    if moe is None:
        if ep != 1 and not problems:
            raise ShapeError("EP requires routed experts")
    else:
        local_experts = _divide(moe.experts, ep, "expert count")
        intermediate = moe.intermediate
        if ep == 1:
            intermediate = _divide(intermediate, tp, "expert intermediate size")
        if moe.top_k > local_experts:
            raise ShapeError("top_k exceeds the per-rank expert count")
        moe = _MoeShape(moe.hidden, intermediate, local_experts, moe.top_k, moe.activation)
        routing = _moe_routing(model, config)
    unsupported = _unsupported_decoder_linears(model, routed_experts_handled=experts_recognized)
    if unsupported:
        details = ", ".join(f"{name} ({n}x{k})" for name, n, k in unsupported)
        problems.append(f"unsupported decoder Linear GEMM layout(s): {details}")
    if not kernels and moe is None and not problems:
        raise ShapeError("no dense benchmark shapes found")
    return kernels, moe, routing, problems


def _command(
    kernels: list[_Kernel],
    moe: _MoeShape | None,
    routing: _MoeRouting | None,
    passthrough: list[str],
) -> list[str]:
    names: dict[tuple[int, int], list[str]] = {}
    for n, k, label in kernels:
        labels = names.setdefault((n, k), [])
        if label not in labels:
            labels.append(label)
    command: list[str] = []
    if names:
        command += ["--nks", *(f"{n},{k}" for n, k in names)]
        command += ["--nk_names", *("/".join(labels) for labels in names.values())]
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
        ]
        if moe.activation:
            command += ["--moe_activation_type", moe.activation]
    if routing:
        command += ["--moe_routing_method", routing.method]
        if routing.num_expert_group is not None:
            command += ["--moe_num_expert_group", str(routing.num_expert_group)]
        if routing.topk_group is not None:
            command += ["--moe_topk_group", str(routing.topk_group)]
        if routing.routed_scaling_factor is not None:
            command += ["--moe_routed_scaling_factor", str(routing.routed_scaling_factor)]
        if routing.use_routing_bias:
            command.append("--moe_use_routing_bias")
    return command + passthrough


def _load_runner() -> Any:
    path = Path(__file__).with_name("benchmark_via_builtin.py")
    spec = importlib.util.spec_from_file_location("benchmark_via_builtin", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
        kernels, moe, routing, problems = _inspect_model(model, config, args.tp, args.ep)
    except ShapeError as exc:
        parser.error(str(exc))

    print(
        f"# {type(model).__name__} ({getattr(config, 'model_type', '?')}), "
        f"TP={args.tp}, EP={args.ep}"
    )
    print("# layout: Transformers meta model; fused QKV and gate/up; Mamba 2 and routed experts")
    for n, k, label in dict.fromkeys(kernels):
        print(f"# {n}x{k} <- {label}")
    if moe:
        activation = f" activation={moe.activation}" if moe.activation else ""
        print(
            f"# MoE: H={moe.hidden} F={moe.intermediate} E={moe.experts} "
            f"top_k={moe.top_k}{activation}"
        )
    if routing:
        groups = (
            f" n_group={routing.num_expert_group} topk_group={routing.topk_group}"
            f" scaling={routing.routed_scaling_factor} bias={routing.use_routing_bias}"
            if routing.method == "deepseek_v3"
            else ""
        )
        print(f"# MoE routing: {routing.method}{groups}")
    for problem in problems:
        print(f"# unsupported: {problem}")
    if problems:
        parser.error(
            "the derived shapes above are incomplete; validate each unsupported layout's "
            "TP/EP sharding and benchmark it directly with benchmark_via_builtin.py"
        )
    command = _command(kernels, moe, routing, passthrough)
    runner = Path(__file__).with_name("benchmark_via_builtin.py")
    print(">>> " + shlex.join([sys.executable, str(runner), *command]))
    if not args.print_only:
        _load_runner().main(command)


if __name__ == "__main__":
    main()
