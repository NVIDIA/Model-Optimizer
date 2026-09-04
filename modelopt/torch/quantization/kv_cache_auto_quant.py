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
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from modelopt.torch.opt.hparam import Hparam
from modelopt.torch.opt.searcher import LPS, BaseSearcher, SearchConfig, SearchStateDict
from modelopt.torch.utils import print_rank_0

from ._auto_quantize_cost import (
    COST_MODEL_KV_CACHE,
    KVCacheCostModel,
    get_auto_quantize_cost_model,
    normalize_auto_quantize_constraints,
)
from .config import QuantizeConfig
from .conversion import set_quantizer_by_cfg
from .nn import TensorQuantizer

__all__ = ["AutoQuantizeKVSearcher"]

if TYPE_CHECKING:
    from collections.abc import Callable

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
        if quantizer._dynamic:
            raise ValueError(
                f"KV-cache AutoQuant candidate {attr} uses top-level dynamic quantization, "
                "which does not retain a persistent export scale."
            )
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
    if (
        set(constraints) - {"effective_bits", "cost_model", "cost"}
        or constraints.get("cost_model") != COST_MODEL_KV_CACHE
        or "effective_bits" not in constraints
        or constraints.get("cost") not in (None, {})
    ):
        raise ValueError(
            "KV-cache AutoQuant requires an effective_bits target with "
            f"cost_model='kv_cache'; got {constraints}."
        )
    target_bits = float(constraints["effective_bits"])
    if not (0 < target_bits <= 16):
        raise ValueError(f"effective_bits must be in (0, 16], got {target_bits}.")
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
    forward_step: Callable[[nn.Module, Any], torch.Tensor],
    model: nn.Module,
    data: Any,
    *,
    validate_finite: bool = True,
) -> torch.Tensor:
    logits = forward_step(model, data)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("KV-cache AutoQuant forward_step must return a logits tensor.")
    if logits.ndim < 2 or logits.shape[-1] == 0:
        raise ValueError(
            "KV-cache AutoQuant forward_step must return logits with a non-empty vocabulary "
            "dimension."
        )
    if validate_finite and not torch.isfinite(logits).all():
        raise ValueError("KV-cache AutoQuant encountered NaN or Inf logits.")
    return logits


def _solve_additive_recipe(
    layer_names: list[str],
    layer_widths: list[tuple[int, int]],
    candidate_names: list[str],
    candidate_kv_bits: list[tuple[float, float]],
    scores: list[list[float]],
    target_bits: float,
    verbose: bool,
) -> tuple[list[int], str]:
    cost_model = get_auto_quantize_cost_model(COST_MODEL_KV_CACHE)
    assert isinstance(cost_model, KVCacheCostModel)
    denominator = float(sum(k_width + v_width for k_width, v_width in layer_widths))
    candidate_costs = [
        [
            cost_model.candidate_cost(k_width, v_width, k_bits, v_bits) / 16.0
            for k_bits, v_bits in candidate_kv_bits
        ]
        for k_width, v_width in layer_widths
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
        minimum_bits = sum(min(costs) for costs in candidate_costs) * 16.0 / denominator
        raise ValueError(
            f"KV-cache AutoQuant could not satisfy effective_bits={target_bits}; "
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
    num_calib_steps: int,
    num_score_steps: int,
) -> dict[str, Any]:
    return {
        "schema_version": _KV_AUTOQUANT_SCHEMA_VERSION,
        "num_calib_steps": num_calib_steps,
        "num_score_steps": num_score_steps,
        "candidates": [
            {
                "name": name,
                "config": config.model_dump(mode="json", exclude_none=True),
            }
            for name, config in candidates
        ],
        "layers": [
            {
                "name": name,
                "k_width": _projection_width(module, "k"),
                "v_width": _projection_width(module, "v"),
            }
            for name, module, _ in layers
        ],
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


def _validate_persistent_candidate_scales(
    candidate_quantizers: dict[str, dict[str, dict[str, TensorQuantizer]]],
) -> None:
    """Require every calibrated candidate scale to be persistent in its state dict."""
    for layer_name, layer_candidates in candidate_quantizers.items():
        for candidate_name, layer_quantizers in layer_candidates.items():
            for attr, quantizer in layer_quantizers.items():
                if "_amax" not in quantizer.state_dict():
                    raise ValueError(
                        f"KV-cache AutoQuant candidate {candidate_name!r} for "
                        f"{layer_name!r}/{attr} has no persistent export scale after calibration."
                    )


def _report_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON-safe search report, excluding calibration tensors."""
    return {key: value for key, value in state.items() if key != "quantizer_state"}


class QuantKVRecipeHparam(Hparam):
    """One paired K/V format decision for an attention layer."""

    def __init__(
        self,
        name: str,
        module: nn.Module,
        candidates: list[tuple[str, QuantizeConfig]],
    ) -> None:
        super().__init__(range(len(candidates)), original=0)
        self.name = name
        self.module = module
        self.candidates = candidates
        self.original_quantizers = {attr: getattr(module, attr) for attr in _KV_QUANTIZER_ATTRS}
        self.reference_quantizers = {attr: _disabled_quantizer() for attr in _KV_QUANTIZER_ATTRS}
        self.candidate_quantizers = {
            index: _candidate_quantizers(config) for index, (_, config) in enumerate(candidates)
        }
        k_width = _projection_width(module, "k")
        v_width = _projection_width(module, "v")
        assert k_width is not None and v_width is not None
        self.k_width = k_width
        self.v_width = v_width
        self.use_reference()

    @property
    def active(self) -> int:
        """Return the selected candidate index."""
        assert isinstance(self._active, int)
        return self._active

    @active.setter
    def active(self, value: int | None) -> None:
        if value is None:
            assert isinstance(self.original, int)
            value = self.original
        assert value in self.choices
        self._active = value
        _apply_layer_quantizers(self.module, self.candidate_quantizers[value])

    def use_reference(self) -> None:
        """Use BF16/no-quant K/V as the scoring reference, never as a solver choice."""
        _apply_layer_quantizers(self.module, self.reference_quantizers)

    def restore_original(self) -> None:
        """Restore the K/V quantizer objects present before search."""
        _apply_layer_quantizers(self.module, self.original_quantizers)

    def candidate_name(self, index: int) -> str:
        return self.candidates[index][0]

    def candidate_bits(self, index: int) -> tuple[float, float]:
        return _candidate_kv_bits(self.candidates[index][1])

    def candidate_cost(self, index: int, cost_model: KVCacheCostModel) -> float:
        k_bits, v_bits = self.candidate_bits(index)
        return cost_model.candidate_cost(self.k_width, self.v_width, k_bits, v_bits)


class AutoQuantizeKVSearcher(BaseSearcher):
    """KV-cache AutoQuant backend using the shared search/checkpoint lifecycle."""

    method_name = "kl_div"

    @property
    def default_search_config(self) -> SearchConfig:
        """Return KV-specific fields layered on the shared search configuration."""
        config = super().default_search_config
        config.update(
            {
                "quantization_formats": [],
                "forward_step": None,
                "num_calib_steps": 512,
                "num_score_steps": 128,
                "disabled_layers": None,
            }
        )
        return config

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Return the checkpointed KV search state."""
        return {
            "schema_version": _KV_AUTOQUANT_SCHEMA_VERSION,
            "method": self.method_name,
            "cost_model": COST_MODEL_KV_CACHE,
            "search_signature": None,
            "calibration_complete": False,
            "num_calib_steps": 0,
            "score_reduction": "mean_per_scored_token",
            "num_score_steps": 0,
            "num_scored_tokens": 0,
            "candidates": [],
            "layers": {},
            "quantizer_state": {},
            "requested_constraints": {},
            "best": {
                "recipe": {},
                "constraints": {},
                "score": float("inf"),
                "is_satisfied": False,
                "solver_status": None,
            },
        }

    def sanitize_search_config(self, config: SearchConfig | None) -> SearchConfig:
        """Validate the data inputs required by isolated forward-KL scoring."""
        config = super().sanitize_search_config(config)
        if config["data_loader"] is None:
            raise ValueError("data_loader must be provided for KV-cache AutoQuant.")
        if config["forward_step"] is None:
            raise ValueError("forward_step must be provided for KV-cache AutoQuant.")
        return config

    def load_search_checkpoint(self) -> bool:
        """Load compatible fields before validating the resolved KV search signature."""
        return super().load_search_checkpoint(strict=False)

    @property
    def _candidate_quantizer_map(
        self,
    ) -> dict[str, dict[str, dict[str, TensorQuantizer]]]:
        return {
            hparam.name: {
                hparam.candidate_name(index): quantizers
                for index, quantizers in hparam.candidate_quantizers.items()
            }
            for hparam in self._hparams
        }

    def restore_original_quantizers(self) -> None:
        """Restore pre-search K/V objects after a failed search."""
        for hparam in getattr(self, "_hparams", []):
            hparam.restore_original()

    def _calibrate_candidates(self) -> None:
        from .model_quant import calibrate

        data_loader = self.config["data_loader"]
        forward_step = self.config["forward_step"]
        num_calib_steps = self.config["num_calib_steps"]
        for candidate_index, (_, config) in enumerate(self._candidates):
            for hparam in self._hparams:
                hparam.active = candidate_index

            if config.algorithm is not None:

                def calibration_loop(calibration_model):
                    for step, data in enumerate(data_loader):
                        if step >= num_calib_steps:
                            break
                        _get_logits(forward_step, calibration_model, data)

                active_quantizers = [
                    quantizer
                    for hparam in self._hparams
                    for quantizer in hparam.candidate_quantizers[candidate_index].values()
                ]
                calibration_proxy = nn.Module()
                calibration_proxy.quantizers = nn.ModuleList(active_quantizers)
                with _freeze_existing_quantizers(self.model, active_quantizers):
                    calibrate(
                        calibration_proxy,
                        algorithm=config.algorithm,
                        forward_loop=lambda _: calibration_loop(self.model),
                    )

            for hparam in self._hparams:
                hparam.use_reference()

        candidate_quantizers = self._candidate_quantizer_map
        _validate_persistent_candidate_scales(candidate_quantizers)
        self.quantizer_state = _quantizer_state_dict(candidate_quantizers)
        self.calibration_complete = True
        self.num_calib_steps = num_calib_steps
        self.save_search_checkpoint(verbose=self.config["verbose"])

    def before_search(self) -> None:
        """Resolve attention decisions and calibrate or restore candidate scales."""
        super().before_search()
        self.constraints = normalize_auto_quantize_constraints(self.model, self.constraints)
        target_bits, self._candidates = _validate_search_inputs(
            self.constraints,
            self.config["quantization_formats"],
            self.config["num_calib_steps"],
            self.config["num_score_steps"],
        )
        self._target_bits = target_bits
        layers = _eligible_layers(self.model, self.config["disabled_layers"])
        _validate_candidate_cost_geometry(self._candidates, layers)
        signature = _search_signature(
            self._candidates,
            layers,
            self.config["num_calib_steps"],
            self.config["num_score_steps"],
        )
        if self.search_signature is not None and not _checkpoint_state_is_compatible(
            self.state_dict(), signature
        ):
            raise ValueError(
                "KV-cache AutoQuant checkpoint does not match the current candidates, scoring "
                "setup, or eligible layers. Use a different checkpoint path."
            )
        self.search_signature = signature
        self._hparams = [
            QuantKVRecipeHparam(name, module, self._candidates) for name, module, _ in layers
        ]
        self._cost_model = get_auto_quantize_cost_model(COST_MODEL_KV_CACHE)
        assert isinstance(self._cost_model, KVCacheCostModel)
        self.candidates = [
            {
                "name": name,
                "effective_bits": config.effective_bits,
                "k_bits": _candidate_kv_bits(config)[0],
                "v_bits": _candidate_kv_bits(config)[1],
                "config": config.model_dump(mode="python", exclude_none=True),
            }
            for name, config in self._candidates
        ]
        self.model.eval()

        if self.calibration_complete:
            if not self.quantizer_state:
                raise ValueError(
                    "KV-cache AutoQuant checkpoint is missing calibrated quantizer state. "
                    "Use a different checkpoint path."
                )
            candidate_quantizers = self._candidate_quantizer_map
            _restore_quantizer_state_dict(candidate_quantizers, self.quantizer_state)
            _validate_persistent_candidate_scales(candidate_quantizers)
            if self.config["verbose"]:
                print_rank_0("KV-cache AutoQuant restored calibration from checkpoint.")
        else:
            self._calibrate_candidates()

    def _estimate_sensitivity_scores(self) -> None:
        candidate_names = [name for name, _ in self._candidates]
        score_sums: dict[str, dict[str, torch.Tensor | None]] = {
            hparam.name: dict.fromkeys(candidate_names) for hparam in self._hparams
        }
        scored_tokens = 0
        scored_steps = 0
        all_logits_finite: torch.Tensor | None = None
        iterator = tqdm(
            self.config["data_loader"],
            total=self.config["num_score_steps"],
            desc="Estimating KV-cache KL sensitivity",
            disable=not self.config["verbose"],
        )
        for data in iterator:
            if scored_steps >= self.config["num_score_steps"]:
                break
            logits_ref = _get_logits(
                self.config["forward_step"], self.model, data, validate_finite=False
            )
            batch_logits_finite = torch.isfinite(logits_ref).all()
            log_prob_ref = torch.log_softmax(logits_ref.float(), dim=-1)
            scored_tokens += logits_ref.numel() // logits_ref.shape[-1]

            for hparam in self._hparams:
                for candidate_index, candidate_name in enumerate(candidate_names):
                    hparam.active = candidate_index
                    logits_quant = _get_logits(
                        self.config["forward_step"], self.model, data, validate_finite=False
                    )
                    batch_logits_finite.logical_and_(torch.isfinite(logits_quant).all())
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
                    previous_score = score_sums[hparam.name][candidate_name]
                    score_sums[hparam.name][candidate_name] = (
                        score if previous_score is None else previous_score + score
                    )
                    hparam.use_reference()
            if all_logits_finite is None:
                all_logits_finite = batch_logits_finite
            else:
                all_logits_finite.logical_and_(batch_logits_finite)
            scored_steps += 1

        if scored_steps == 0 or scored_tokens == 0:
            raise ValueError("KV-cache AutoQuant data_loader produced no scoring batches.")
        assert all_logits_finite is not None
        if not all_logits_finite:
            raise ValueError("KV-cache AutoQuant encountered NaN or Inf logits.")

        self.layers = {}
        for hparam in self._hparams:
            layer_score_sums = []
            for candidate_name in candidate_names:
                score_sum = score_sums[hparam.name][candidate_name]
                if score_sum is None:
                    raise RuntimeError(
                        "KV-cache AutoQuant did not collect a score for "
                        f"{hparam.name!r}/{candidate_name!r}."
                    )
                layer_score_sums.append(score_sum)
            layer_scores = (torch.stack(layer_score_sums) / scored_tokens).tolist()
            scores = {}
            for candidate_name, score in zip(candidate_names, layer_scores):
                if not math.isfinite(score):
                    raise ValueError(
                        "KV-cache AutoQuant produced a non-finite KL score for "
                        f"{hparam.name!r}/{candidate_name!r}."
                    )
                scores[candidate_name] = score
            self.layers[hparam.name] = {
                "k_width": hparam.k_width,
                "v_width": hparam.v_width,
                "scores": scores,
            }
        self.num_score_steps = scored_steps
        self.num_scored_tokens = scored_tokens
        self.save_search_checkpoint(verbose=self.config["verbose"])

    def _solve(self) -> None:
        candidate_names = [name for name, _ in self._candidates]
        candidate_kv_bits = [_candidate_kv_bits(config) for _, config in self._candidates]
        layers = cast("dict[str, dict[str, Any]]", self.layers)
        scores = [
            [layers[hparam.name]["scores"][name] for name in candidate_names]
            for hparam in self._hparams
        ]
        selections, status = _solve_additive_recipe(
            [hparam.name for hparam in self._hparams],
            [(hparam.k_width, hparam.v_width) for hparam in self._hparams],
            candidate_names,
            candidate_kv_bits,
            scores,
            self._target_bits,
            self.config["verbose"],
        )
        denominator = float(sum(hparam.k_width + hparam.v_width for hparam in self._hparams))
        cost_model = self._cost_model
        assert isinstance(cost_model, KVCacheCostModel)
        total_cost = sum(
            (
                hparam.candidate_cost(selected, cost_model)
                for hparam, selected in zip(self._hparams, selections)
            ),
            start=0.0,
        )
        achieved_bits = total_cost / denominator
        selected_score = sum(
            layer_scores[selected] for selected, layer_scores in zip(selections, scores)
        )
        recipe = {}
        for hparam, selected in zip(self._hparams, selections):
            hparam.active = selected
            selected_name = hparam.candidate_name(selected)
            self.layers[hparam.name]["selected"] = selected_name
            recipe[hparam.name] = selected_name
            if self.config["verbose"]:
                print_rank_0(f"KV-cache AutoQuant selected {selected_name} for {hparam.name}.")
        self.requested_constraints = {
            "effective_bits": self._target_bits,
            "cost_model": COST_MODEL_KV_CACHE,
        }
        self.best = {
            "recipe": recipe,
            "constraints": {
                "effective_bits": achieved_bits,
                "cost_model": COST_MODEL_KV_CACHE,
            },
            "score": selected_score,
            "is_satisfied": achieved_bits <= self._target_bits + 1e-12,
            "solver_status": status,
        }

    @torch.inference_mode()
    def run_search(self) -> None:
        """Score candidates when needed, solve the budget, and apply the selection."""
        if not self.layers:
            self._estimate_sensitivity_scores()
        self._solve()
        self.save_search_checkpoint(verbose=self.config["verbose"])

    def after_search(self) -> None:
        """Attach the JSON-safe sensitivity report used by unified export."""
        self.model._modelopt_kv_cache_auto_quantize_state = _report_state(self.state_dict())


def _config_entry_dict(entry: Any) -> dict[str, Any]:
    if hasattr(entry, "model_dump"):
        return entry.model_dump(mode="python", exclude_none=True)
    return dict(entry)


def get_kv_cache_auto_quantize_config(
    search_state: dict[str, Any],
    constraints: dict[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Build a flat K/V quantization config, optionally re-solving at a new target."""
    requested = constraints or search_state.get("requested_constraints")
    if not isinstance(requested, dict):
        raise ValueError("KV-cache AutoQuant search state has no requested constraints.")
    target_bits = float(requested["effective_bits"])
    if requested.get("cost_model", COST_MODEL_KV_CACHE) != COST_MODEL_KV_CACHE:
        raise ValueError("KV-cache search state can only be re-solved with cost_model='kv_cache'.")

    candidates = search_state["candidates"]
    candidate_names = [candidate["name"] for candidate in candidates]
    candidate_kv_bits = [
        (float(candidate["k_bits"]), float(candidate["v_bits"])) for candidate in candidates
    ]
    layers = search_state["layers"]
    layer_names = list(layers)
    selections, _ = _solve_additive_recipe(
        layer_names,
        [(layers[name]["k_width"], layers[name]["v_width"]) for name in layer_names],
        candidate_names,
        candidate_kv_bits,
        [
            [layers[name]["scores"][candidate] for candidate in candidate_names]
            for name in layer_names
        ],
        target_bits,
        verbose,
    )

    quant_cfg: list[dict[str, Any]] = [{"quantizer_name": "*", "enable": False}]
    needs_max_calibration = False
    for layer_name, selected in zip(layer_names, selections):
        selected_config = candidates[selected]["config"]
        # ``algorithm=None`` is omitted from the JSON-safe search state, while calibrated
        # candidates retain their validated ``max`` algorithm.
        needs_max_calibration |= selected_config.get("algorithm") is not None
        config = QuantizeConfig(**selected_config)
        for entry in config.quant_cfg:
            entry_dict = _config_entry_dict(entry)
            pattern = entry_dict["quantizer_name"]
            for attr in _KV_QUANTIZER_ATTRS:
                if fnmatch.fnmatch(f"{_KV_CANDIDATE_HOLDER_NAME}.{attr}", pattern):
                    resolved = dict(entry_dict)
                    resolved["quantizer_name"] = f"{layer_name}.{attr}"
                    quant_cfg.append(resolved)
    return {"quant_cfg": quant_cfg, "algorithm": "max" if needs_max_calibration else None}
