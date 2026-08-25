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

"""Layer-wise KV-cache AutoQuant using isolated forward KL sensitivity."""

from __future__ import annotations

import fnmatch
import math
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from modelopt.torch.opt.searcher import LPS
from modelopt.torch.utils import print_rank_0, safe_load, safe_save

from .config import QuantizeConfig
from .conversion import set_quantizer_by_cfg
from .nn import TensorQuantizer

__all__ = ["auto_quantize_kv_cache"]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

_KV_QUANTIZER_ATTRS = ("k_bmm_quantizer", "v_bmm_quantizer")
_KV_AUTOQUANT_SCHEMA_VERSION = 1
_KV_CANDIDATE_HOLDER_NAME = "layer"
_KV_CANDIDATE_NAMES = {f"{_KV_CANDIDATE_HOLDER_NAME}.{attr}" for attr in _KV_QUANTIZER_ATTRS}
_NON_KV_PROBE_NAMES = {
    f"{_KV_CANDIDATE_HOLDER_NAME}.{name}"
    for name in (
        "q_bmm_quantizer",
        "p_bmm_quantizer",
        "input_quantizer",
        "output_quantizer",
        "q_proj.input_quantizer",
        "q_proj.weight_quantizer",
        "q_proj.output_quantizer",
    )
}


def _disabled_quantizer() -> TensorQuantizer:
    quantizer = TensorQuantizer()
    quantizer.disable()
    return quantizer


def _candidate_quantizers(config: QuantizeConfig) -> dict[str, TensorQuantizer]:
    """Build a candidate with the same qualified K/V names used by a full model."""
    _validate_candidate_patterns(config)
    root = nn.Module()
    holder = nn.Module()
    root.add_module(_KV_CANDIDATE_HOLDER_NAME, holder)
    for attr in _KV_QUANTIZER_ATTRS:
        setattr(holder, attr, _disabled_quantizer())
    set_quantizer_by_cfg(root, config.quant_cfg)
    quantizers = {attr: getattr(holder, attr) for attr in _KV_QUANTIZER_ATTRS}
    for attr, quantizer in quantizers.items():
        if not isinstance(quantizer, TensorQuantizer) or not quantizer.is_enabled:
            raise ValueError(
                f"KV-cache candidate must enable {attr}; got {type(quantizer).__name__}."
            )
    return quantizers


def _validate_candidate_patterns(config: QuantizeConfig) -> None:
    """Require every ordered config entry to match only a qualified K/V name."""
    matched_names: set[str] = set()
    probe_names = _KV_CANDIDATE_NAMES | _NON_KV_PROBE_NAMES
    for entry in config.quant_cfg:
        if entry.parent_class is not None:
            raise ValueError("KV-cache AutoQuant candidates do not support parent_class filters.")
        matches = {name for name in probe_names if fnmatch.fnmatch(name, entry.quantizer_name)}
        if not matches:
            raise ValueError(
                "KV-cache AutoQuant candidate pattern "
                f"{entry.quantizer_name!r} does not match a supported qualified K/V quantizer."
            )
        non_kv_matches = matches - _KV_CANDIDATE_NAMES
        if non_kv_matches:
            raise ValueError(
                "KV-cache AutoQuant candidates may configure only k_bmm_quantizer and "
                f"v_bmm_quantizer; pattern {entry.quantizer_name!r} also matches "
                f"{sorted(non_kv_matches)}."
            )
        matched_names.update(matches)
    if matched_names != _KV_CANDIDATE_NAMES:
        raise ValueError(
            "KV-cache AutoQuant candidates must completely configure both "
            "k_bmm_quantizer and v_bmm_quantizer."
        )


def _algorithm_method(config: QuantizeConfig) -> str | None:
    algorithm = config.algorithm
    if algorithm is None or isinstance(algorithm, str):
        return algorithm
    if isinstance(algorithm, dict):
        return algorithm.get("method")
    return getattr(algorithm, "method", None)


def _deployable_kv_bits(quantizer: TensorQuantizer) -> float:
    """Return storage bits for the narrow K/V formats supported by unified export."""
    if quantizer.bias is not None:
        raise ValueError("KV-cache AutoQuant does not support affine candidates yet.")
    if quantizer.is_fp8:
        return 8.0
    if quantizer.is_nvfp4_dynamic and quantizer.block_sizes.get(-1) == 16:
        return 4.5
    raise ValueError(
        "KV-cache AutoQuant candidates must use unified-export-compatible per-tensor FP8 "
        "or block-16 dynamic NVFP4 quantizers."
    )


def _candidate_kv_bits(config: QuantizeConfig) -> tuple[float, float]:
    quantizers = _candidate_quantizers(config)
    return (
        _deployable_kv_bits(quantizers["k_bmm_quantizer"]),
        _deployable_kv_bits(quantizers["v_bmm_quantizer"]),
    )


def _validate_deployable_candidate(config: QuantizeConfig) -> None:
    quantizers = _candidate_quantizers(config)
    k_quantizer = quantizers["k_bmm_quantizer"]
    v_quantizer = quantizers["v_bmm_quantizer"]
    k_bits = _deployable_kv_bits(k_quantizer)
    v_bits = _deployable_kv_bits(v_quantizer)
    if k_bits != v_bits and (k_bits, v_bits) != (8.0, 4.5):
        raise ValueError(
            "Unified export supports only uniform FP8, uniform NVFP4, or FP8-K/NVFP4-V "
            "KV-cache AutoQuant candidates."
        )

    algorithm_method = _algorithm_method(config)
    for attr, quantizer in quantizers.items():
        will_calibrate = algorithm_method == "max" and not quantizer._use_constant_amax
        if not hasattr(quantizer, "_amax") and not will_calibrate:
            raise ValueError(
                f"KV-cache AutoQuant candidate {attr} has no persistent export scale. "
                "Use max calibration or constant_amax; dynamic and use_constant_amax-only "
                "candidates cannot be exported."
            )

    assert config.effective_bits is not None
    actual_effective_bits = (k_bits + v_bits) / 2.0
    if not math.isclose(config.effective_bits, actual_effective_bits, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "KV-cache AutoQuant candidate effective_bits does not match its configured K/V "
            f"storage cost: declared {config.effective_bits}, actual {actual_effective_bits}."
        )


def _validate_kv_only_config(config: QuantizeConfig) -> None:
    if config.effective_bits is None:
        raise ValueError(
            "Each KV-cache AutoQuant candidate must declare config-level effective_bits."
        )
    algorithm_method = _algorithm_method(config)
    if algorithm_method != "max":
        if algorithm_method is not None:
            raise ValueError(
                "KV-cache AutoQuant supports only non-structural calibration algorithms "
                f"None and 'max'; got {algorithm_method!r}."
            )
    _validate_deployable_candidate(config)


def _validate_search_inputs(
    constraints: dict[str, Any],
    quantization_formats: list[tuple[dict[str, Any], str]],
    num_calib_steps: int,
    num_score_steps: int,
) -> tuple[float, list[tuple[str, QuantizeConfig]]]:
    """Validate a KV-cache search before the caller converts the model."""
    if set(constraints) != {"kv_effective_bits"}:
        raise ValueError(
            "KV-cache AutoQuant constraints must contain only kv_effective_bits; "
            f"got {sorted(constraints)}."
        )
    target_bits = float(constraints["kv_effective_bits"])
    if not (0 < target_bits <= 16):
        raise ValueError(f"kv_effective_bits must be in (0, 16], got {target_bits}.")
    if num_calib_steps <= 0:
        raise ValueError("num_calib_steps must be positive.")
    if num_score_steps <= 0:
        raise ValueError("num_score_steps must be positive.")

    candidates = []
    seen_names = set()
    for raw_config, name in quantization_formats:
        if name in seen_names:
            raise ValueError(f"Duplicate KV-cache AutoQuant candidate name: {name!r}.")
        config = QuantizeConfig(**raw_config)
        _validate_kv_only_config(config)
        candidates.append((name, config))
        seen_names.add(name)
    if not candidates:
        raise ValueError("KV-cache AutoQuant requires at least one candidate format.")
    return target_bits, candidates


def _projection_width(module: nn.Module, side: str) -> int | None:
    projection = getattr(module, f"{side}_proj", None)
    out_features = getattr(projection, "out_features", None)
    if isinstance(out_features, int) and out_features > 0:
        return out_features

    config = getattr(module, "config", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    head_dim = getattr(module, "head_dim", None) or getattr(config, "head_dim", None)
    if not isinstance(head_dim, int):
        hidden_size = getattr(config, "hidden_size", None)
        num_heads = getattr(config, "num_attention_heads", None)
        if isinstance(hidden_size, int) and isinstance(num_heads, int) and num_heads > 0:
            head_dim = hidden_size // num_heads
    if isinstance(num_kv_heads, int) and isinstance(head_dim, int):
        return num_kv_heads * head_dim
    return None


def _kv_scalar_weight(module: nn.Module, name: str) -> int:
    k_width = _projection_width(module, "k")
    v_width = _projection_width(module, "v")
    if k_width is None or v_width is None:
        raise ValueError(
            "Cannot determine exact KV width for eligible attention layer "
            f"{name!r}. Expected k_proj/v_proj.out_features or config "
            "num_key_value_heads plus head_dim."
        )
    return k_width + v_width


def _validate_candidate_cost_geometry(
    candidates: list[tuple[str, QuantizeConfig]],
    layers: list[tuple[str, nn.Module, int]],
) -> None:
    candidate_bits = [_candidate_kv_bits(config) for _, config in candidates]
    if all(k_bits == v_bits for k_bits, v_bits in candidate_bits):
        return

    unequal_width_layers = []
    for name, module, _ in layers:
        k_width = _projection_width(module, "k")
        v_width = _projection_width(module, "v")
        if k_width != v_width:
            unequal_width_layers.append(f"{name} (K={k_width}, V={v_width})")
    if unequal_width_layers:
        raise ValueError(
            "KV-cache AutoQuant cannot cost asymmetric K/V candidates on layers with unequal "
            "K/V widths: " + ", ".join(unequal_width_layers) + "."
        )


def _eligible_layers(
    model: nn.Module, disabled_layers: list[str] | str | None
) -> list[tuple[str, nn.Module, int]]:
    patterns = [disabled_layers] if isinstance(disabled_layers, str) else disabled_layers or []
    boundaries = []
    names_by_identity: dict[int, list[str]] = {}
    for name, module in model.named_modules(remove_duplicate=False):
        if not all(hasattr(module, attr) for attr in _KV_QUANTIZER_ATTRS):
            continue
        boundaries.append((name, module))
        names_by_identity.setdefault(id(module), []).append(name)
    aliases = [names for names in names_by_identity.values() if len(names) > 1]
    if aliases:
        raise ValueError(
            f"KV-cache attention boundaries are registered through aliases: {aliases}."
        )

    layers = []
    for name, module in boundaries:
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            continue
        layers.append((name, module, _kv_scalar_weight(module, name)))
    if not layers:
        raise ValueError("KV-cache AutoQuant found no eligible attention layers.")
    return layers


def _apply_layer_quantizers(module: nn.Module, quantizers: dict[str, TensorQuantizer]) -> None:
    for attr, quantizer in quantizers.items():
        setattr(module, attr, quantizer)


@contextmanager
def _freeze_existing_quantizers(model: nn.Module, candidate_quantizers: list[TensorQuantizer]):
    """Freeze calibration without changing existing quantizers' execution mode."""
    candidate_ids = {id(quantizer) for quantizer in candidate_quantizers}
    states = []
    for module in model.modules():
        if (
            not isinstance(module, TensorQuantizer)
            or id(module) in candidate_ids
            or not module.is_enabled
        ):
            continue
        states.append((module, module._if_calib))
        module.disable_calib()
    try:
        yield
    finally:
        for quantizer, if_calib in states:
            quantizer._if_calib = if_calib


def _get_logits(
    forward_step: Callable[[nn.Module, Any], torch.Tensor], model: nn.Module, data: Any
) -> torch.Tensor:
    logits = forward_step(model, data)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("KV-cache AutoQuant forward_step must return a logits tensor.")
    if logits.ndim < 2 or logits.shape[-1] == 0:
        raise ValueError(
            "KV-cache AutoQuant forward_step must return logits with a non-empty vocabulary "
            "dimension."
        )
    if not torch.isfinite(logits).all():
        raise ValueError("KV-cache AutoQuant encountered NaN or Inf logits.")
    return logits


def _solve_additive_recipe(
    layer_names: list[str],
    scalar_weights: list[int],
    candidate_names: list[str],
    candidate_bits: list[float],
    scores: list[list[float]],
    target_bits: float,
    verbose: bool,
) -> tuple[list[int], str]:
    denominator = float(sum(scalar_weights))
    candidate_costs = [
        [weight * bits / 16.0 for bits in candidate_bits] for weight in scalar_weights
    ]
    max_cost = denominator * target_bits / 16.0
    lps = LPS(
        name="KVCacheAutoQuant",
        constraints={"kv_cache_size_after_compression": max_cost},
        constraints_to_candidate_costs={"kv_cache_size_after_compression": candidate_costs},
        candidate_scores=scores,
        objective_type="minimize",
        verbose=verbose,
    )
    selections, status = lps()
    if status != "Optimal":
        minimum_bits = sum(weight * min(candidate_bits) for weight in scalar_weights) / denominator
        raise ValueError(
            f"KV-cache AutoQuant could not satisfy kv_effective_bits={target_bits}; "
            f"minimum achievable value is {minimum_bits:.4f}. Solver status: {status}."
        )
    if len(selections) != len(layer_names):
        raise RuntimeError(
            "KV-cache AutoQuant solver returned an invalid selection count: "
            f"{len(selections)} for {len(layer_names)} layers and candidates {candidate_names}."
        )
    return selections, status


def _search_signature(
    candidates: list[tuple[str, QuantizeConfig]],
    layers: list[tuple[str, nn.Module, int]],
    target_bits: float,
    num_calib_steps: int,
    num_score_steps: int,
) -> dict[str, Any]:
    return {
        "schema_version": _KV_AUTOQUANT_SCHEMA_VERSION,
        "kv_effective_bits": target_bits,
        "num_calib_steps": num_calib_steps,
        "num_score_steps": num_score_steps,
        "candidates": [
            {
                "name": name,
                "config": config.model_dump(mode="json", exclude_none=True),
            }
            for name, config in candidates
        ],
        "layers": [{"name": name, "kv_scalar_weight": weight} for name, _, weight in layers],
    }


def _checkpoint_state_is_compatible(state: dict[str, Any], signature: dict[str, Any]) -> bool:
    return state.get("search_signature") == signature


def _quantizer_state_dict(
    candidate_quantizers: dict[str, dict[str, dict[str, TensorQuantizer]]],
) -> dict[str, dict[str, dict[str, dict[str, torch.Tensor]]]]:
    return {
        layer_name: {
            candidate_name: {
                attr: quantizer.state_dict() for attr, quantizer in layer_quantizers.items()
            }
            for candidate_name, layer_quantizers in layer_candidates.items()
        }
        for layer_name, layer_candidates in candidate_quantizers.items()
    }


def _restore_quantizer_state_dict(
    candidate_quantizers: dict[str, dict[str, dict[str, TensorQuantizer]]],
    state: dict[str, dict[str, dict[str, dict[str, torch.Tensor]]]],
) -> None:
    """Restore calibration buffers into config-created candidate quantizers."""
    for layer_name, layer_candidates in candidate_quantizers.items():
        for candidate_name, layer_quantizers in layer_candidates.items():
            for attr, quantizer in layer_quantizers.items():
                quantizer_state = state[layer_name][candidate_name][attr]
                for key, value in quantizer_state.items():
                    if "." not in key and key not in quantizer._buffers:
                        quantizer.register_buffer(key, torch.empty_like(value))
                quantizer.load_state_dict(quantizer_state)


def _report_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON-safe search report, excluding calibration tensors."""
    return {key: value for key, value in state.items() if key != "quantizer_state"}


@torch.inference_mode()
def auto_quantize_kv_cache(
    model: nn.Module,
    constraints: dict[str, Any],
    quantization_formats: list[tuple[dict[str, Any], str]],
    data_loader: Iterable,
    forward_step: Callable[[nn.Module, Any], torch.Tensor],
    *,
    num_calib_steps: int,
    num_score_steps: int,
    disabled_layers: list[str] | str | None = None,
    verbose: bool = False,
    checkpoint: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Select one supplied K/V format per attention layer using isolated forward KL.

    Candidate formats are format-agnostic ``QuantizeConfig`` dictionaries. Each must
    configure K and V together and declare ``effective_bits`` matching its packed
    storage per K-or-V scalar, including scale overhead. Candidate calibration is scoped
    to the candidate K/V quantizers, while pre-existing fixed quantizers keep executing
    with frozen state. Persistent ``constant_amax`` formats may skip calibration forwards.
    """
    target_bits, candidates = _validate_search_inputs(
        constraints, quantization_formats, num_calib_steps, num_score_steps
    )

    layers = _eligible_layers(model, disabled_layers)
    _validate_candidate_cost_geometry(candidates, layers)
    signature = _search_signature(
        candidates,
        layers,
        target_bits,
        num_calib_steps,
        num_score_steps,
    )
    candidate_names = [name for name, _ in candidates]
    candidate_bits = []
    for _, config in candidates:
        assert config.effective_bits is not None
        candidate_bits.append(config.effective_bits)

    original_quantizers = {
        name: {attr: getattr(module, attr) for attr in _KV_QUANTIZER_ATTRS}
        for name, module, _ in layers
    }
    disabled_quantizers = {
        name: {attr: _disabled_quantizer() for attr in _KV_QUANTIZER_ATTRS} for name, _, _ in layers
    }
    candidate_quantizers = {
        name: {
            candidate_name: _candidate_quantizers(config) for candidate_name, config in candidates
        }
        for name, _, _ in layers
    }

    is_training = model.training
    model.eval()
    try:
        for name, module, _ in layers:
            _apply_layer_quantizers(module, disabled_quantizers[name])

        state: dict[str, Any] | None = None
        if checkpoint is not None and os.path.exists(checkpoint):
            restored = safe_load(checkpoint)
            if not isinstance(restored, dict):
                raise ValueError(
                    "KV-cache AutoQuant checkpoint must contain a search-state dictionary."
                )
            if _checkpoint_state_is_compatible(restored, signature):
                state = restored
                if verbose:
                    print_rank_0(f"KV-cache AutoQuant restored search state from {checkpoint}.")
            else:
                raise ValueError(
                    "KV-cache AutoQuant checkpoint does not match the current candidates "
                    "or eligible layers. Use a different checkpoint path."
                )

        if state is not None and state.get("calibration_complete"):
            quantizer_state = state.get("quantizer_state")
            if quantizer_state is None:
                raise ValueError(
                    "KV-cache AutoQuant checkpoint is missing calibrated quantizer state. "
                    "Use a different checkpoint path."
                )
            _restore_quantizer_state_dict(candidate_quantizers, quantizer_state)
        else:
            from .model_quant import calibrate

            for candidate_name, config in candidates:
                for layer_name, module, _ in layers:
                    _apply_layer_quantizers(
                        module, candidate_quantizers[layer_name][candidate_name]
                    )

                if config.algorithm is not None:

                    def calibration_loop(calibration_model):
                        for step, data in enumerate(data_loader):
                            if step >= num_calib_steps:
                                break
                            _get_logits(forward_step, calibration_model, data)

                    active_quantizers = [
                        quantizer
                        for layer_name, _, _ in layers
                        for quantizer in candidate_quantizers[layer_name][candidate_name].values()
                    ]
                    calibration_proxy = nn.Module()
                    calibration_proxy.quantizers = nn.ModuleList(active_quantizers)
                    with _freeze_existing_quantizers(model, active_quantizers):
                        calibrate(
                            calibration_proxy,
                            algorithm=config.algorithm,
                            forward_loop=lambda _: calibration_loop(model),
                        )

                for layer_name, module, _ in layers:
                    _apply_layer_quantizers(module, disabled_quantizers[layer_name])

            state = {
                "schema_version": _KV_AUTOQUANT_SCHEMA_VERSION,
                "search_signature": signature,
                "calibration_complete": True,
                "num_calib_steps": num_calib_steps,
                "quantizer_state": _quantizer_state_dict(candidate_quantizers),
            }
            if checkpoint is not None:
                checkpoint_dir = os.path.dirname(checkpoint)
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                safe_save(state, checkpoint)

        assert state is not None
        if not state.get("layers"):
            score_sums: dict[str, dict[str, torch.Tensor | None]] = {
                layer_name: dict.fromkeys(candidate_names) for layer_name, _, _ in layers
            }
            scored_tokens = 0
            scored_steps = 0
            iterator = tqdm(
                data_loader,
                total=num_score_steps,
                desc="Estimating KV-cache KL sensitivity",
                disable=not verbose,
            )
            for data in iterator:
                if scored_steps >= num_score_steps:
                    break
                logits_ref = _get_logits(forward_step, model, data)
                log_prob_ref = torch.log_softmax(logits_ref.float(), dim=-1)
                scored_tokens += logits_ref.numel() // logits_ref.shape[-1]

                for layer_name, module, _ in layers:
                    for candidate_name, _ in candidates:
                        _apply_layer_quantizers(
                            module, candidate_quantizers[layer_name][candidate_name]
                        )
                        logits_quant = _get_logits(forward_step, model, data)
                        if logits_quant.shape != logits_ref.shape:
                            raise ValueError(
                                "KV-cache AutoQuant forward_step returned different reference and "
                                f"candidate logits shapes: {tuple(logits_ref.shape)} and "
                                f"{tuple(logits_quant.shape)}."
                            )
                        score = F.kl_div(
                            torch.log_softmax(logits_quant.float(), dim=-1),
                            log_prob_ref,
                            reduction="sum",
                            log_target=True,
                        )
                        previous_score = score_sums[layer_name][candidate_name]
                        score_sums[layer_name][candidate_name] = (
                            score if previous_score is None else previous_score + score
                        )
                        _apply_layer_quantizers(module, disabled_quantizers[layer_name])
                scored_steps += 1

            if scored_steps == 0 or scored_tokens == 0:
                raise ValueError("KV-cache AutoQuant data_loader produced no scoring batches.")
            scores = []
            for layer_name, _, _ in layers:
                layer_scores = []
                for candidate_name in candidate_names:
                    score_sum = score_sums[layer_name][candidate_name]
                    if score_sum is None:
                        raise RuntimeError(
                            "KV-cache AutoQuant did not collect a score for "
                            f"{layer_name!r}/{candidate_name!r}."
                        )
                    if not torch.isfinite(score_sum):
                        raise ValueError(
                            "KV-cache AutoQuant produced a non-finite KL score for "
                            f"{layer_name!r}/{candidate_name!r}."
                        )
                    layer_scores.append(float(score_sum.item()) / scored_tokens)
                scores.append(layer_scores)
            selections, status = _solve_additive_recipe(
                [name for name, _, _ in layers],
                [weight for _, _, weight in layers],
                candidate_names,
                candidate_bits,
                scores,
                target_bits,
                verbose,
            )
            denominator = float(sum(weight for _, _, weight in layers))
            achieved_bits = (
                sum(
                    weight * candidate_bits[selected]
                    for selected, (_, _, weight) in zip(selections, layers)
                )
                / denominator
            )
            selected_score = sum(
                layer_scores[selected] for selected, layer_scores in zip(selections, scores)
            )
            state.update(
                {
                    "method": "kl_div",
                    "score_reduction": "mean_per_scored_token",
                    "constraints": {"kv_effective_bits": target_bits},
                    "num_score_steps": scored_steps,
                    "num_scored_tokens": scored_tokens,
                    "candidates": [
                        {
                            "name": name,
                            "effective_bits": effective_bits,
                            "config": config.model_dump(mode="json", exclude_none=True),
                        }
                        for (name, config), effective_bits in zip(candidates, candidate_bits)
                    ],
                    "layers": {
                        layer_name: {
                            "kv_scalar_weight": weight,
                            "scores": dict(zip(candidate_names, layer_scores)),
                            "selected": candidate_names[selected],
                        }
                        for selected, layer_scores, (layer_name, _, weight) in zip(
                            selections, scores, layers
                        )
                    },
                    "best": {
                        "effective_bits": achieved_bits,
                        "score": selected_score,
                        "is_satisfied": achieved_bits <= target_bits + 1e-12,
                        "solver_status": status,
                    },
                }
            )
            if checkpoint is not None:
                checkpoint_dir = os.path.dirname(checkpoint)
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                safe_save(state, checkpoint)
                if verbose:
                    print_rank_0(f"Saved KV-cache AutoQuant report to {checkpoint}.")

        for layer_name, module, _ in layers:
            selected_name = state["layers"][layer_name]["selected"]
            _apply_layer_quantizers(module, candidate_quantizers[layer_name][selected_name])
            if verbose:
                print_rank_0(f"KV-cache AutoQuant selected {selected_name} for {layer_name}.")
        report = _report_state(state)
        model._modelopt_kv_cache_auto_quantize_state = report
        return model, report
    except Exception:
        for layer_name, module, _ in layers:
            _apply_layer_quantizers(module, original_quantizers[layer_name])
        raise
    finally:
        model.train(is_training)
