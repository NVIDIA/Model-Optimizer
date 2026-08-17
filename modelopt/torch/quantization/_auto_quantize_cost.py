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

"""Cost models for AutoQuantize effective-bits accounting."""

import fnmatch
from collections.abc import Sequence
from typing import Any, Final

import regex as re
import torch.nn as nn

# Default target used by historical AutoQuantize calls when no explicit effective-bits
# constraint is supplied. The value is intentionally kept for backward compatibility.
DEFAULT_AUTO_QUANTIZE_EFFECTIVE_BITS: Final = 4.8

AUTO_QUANTIZE_CONSTRAINT_KEYS: Final = frozenset(
    {"effective_bits", "cost_model", "cost", "latency"}
)
ACTIVE_MOE_EXPERT_RATIO_KEY: Final = "active_moe_expert_ratio"
EXCLUDED_MODULE_NAME_PATTERNS_KEY: Final = "excluded_module_name_patterns"
COST_MODEL_WEIGHT: Final = "weight"
COST_MODEL_ACTIVE_MOE: Final = "active_moe"
COST_MODEL_LATENCY: Final = "latency"

# Keys inside constraints['cost'] for the latency cost model.
LATENCY_LUT_PATH_KEY: Final = "lut_path"
LATENCY_DEPLOYMENT_PROFILE_KEY: Final = "deployment_profile"
LATENCY_M_KEY: Final = "m"
# Key inside the top-level constraints['latency'] block.
LATENCY_RELATIVE_TO_MIN_KEY: Final = "relative_to_min"

_ROUTED_MOE_EXPERT_NAME_RE = re.compile(r"(^|\.)experts(\.|$)")
_ACTIVE_MOE_TOP_K_ATTRS = (
    "num_experts_per_tok",
    "num_experts_per_token",
    "moe_top_k",
    "top_k",
    "num_selected_experts",
)
_ACTIVE_MOE_NUM_EXPERTS_ATTRS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_num_experts",
    "num_routed_experts",
)


def _iter_model_configs(model: nn.Module):
    seen = set()
    for obj in (model, getattr(model, "model", None), getattr(model, "language_model", None)):
        config = getattr(obj, "config", None)
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        yield config
        for nested_attr in ("text_config", "language_config"):
            nested_config = getattr(config, nested_attr, None)
            if nested_config is None or id(nested_config) in seen:
                continue
            seen.add(id(nested_config))
            yield nested_config


def _get_first_numeric_config_attr(config: Any, attr_names: tuple[str, ...]) -> float | None:
    for attr_name in attr_names:
        value = getattr(config, attr_name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def infer_active_moe_expert_ratio(model: nn.Module) -> float | None:
    """Infer top-k / num-experts from a single model config object when possible."""
    for config in _iter_model_configs(model):
        num_active_experts = _get_first_numeric_config_attr(config, _ACTIVE_MOE_TOP_K_ATTRS)
        num_experts = _get_first_numeric_config_attr(config, _ACTIVE_MOE_NUM_EXPERTS_ATTRS)
        if num_active_experts is None or num_experts is None or num_experts <= 0:
            continue
        ratio = num_active_experts / num_experts
        if ratio <= 0.0:
            continue
        return min(ratio, 1.0)
    return None


def is_routed_moe_module_name(name: str) -> bool:
    """Return True for routed MoE expert modules, excluding shared experts."""
    return "shared_expert" not in name and _ROUTED_MOE_EXPERT_NAME_RE.search(name) is not None


def _get_module_weight_numel(module: nn.Module) -> int:
    """Return the parameter count for a module's quantizable weights.

    Standard quantized linear modules have a single ``weight`` parameter. Fused
    MoE expert containers expose projection tensors directly instead, so both
    fused projections contribute to AutoQuantize cost accounting.
    """
    weight = getattr(module, "weight", None)
    if weight is not None:
        return weight.numel()

    # Fused MoE expert containers expose projection tensors directly instead of
    # a single ``weight`` parameter.
    return sum(
        param.numel()
        for attr in ("gate_up_proj", "down_proj")
        if (param := getattr(module, attr, None)) is not None
    )


class AutoQuantizeCostModel:
    """Base class for AutoQuantize effective-bits cost accounting."""

    name: str
    supported_cost_keys: frozenset[str] = frozenset({EXCLUDED_MODULE_NAME_PATTERNS_KEY})

    def normalize_cost_constraints(
        self, model: nn.Module, cost_constraints: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate and normalize cost-model-specific constraints."""
        unknown_cost_keys = set(cost_constraints) - self.supported_cost_keys
        if unknown_cost_keys:
            raise ValueError(f"Unsupported auto_quantize cost constraints: {unknown_cost_keys}.")
        excluded_patterns = cost_constraints.get(EXCLUDED_MODULE_NAME_PATTERNS_KEY)
        if excluded_patterns is None:
            return cost_constraints
        if isinstance(excluded_patterns, str):
            excluded_patterns = [excluded_patterns]
        if not isinstance(excluded_patterns, Sequence) or not all(
            isinstance(pattern, str) for pattern in excluded_patterns
        ):
            raise ValueError(
                f"constraints['cost']['{EXCLUDED_MODULE_NAME_PATTERNS_KEY}'] must be a string "
                "or a sequence of strings."
            )
        cost_constraints[EXCLUDED_MODULE_NAME_PATTERNS_KEY] = list(excluded_patterns)
        return cost_constraints

    def module_cost_weight(
        self, module_names: Sequence[str], cost_constraints: dict[str, Any]
    ) -> float:
        """Return the cost multiplier for a group of modules."""
        excluded_patterns = cost_constraints.get(EXCLUDED_MODULE_NAME_PATTERNS_KEY, [])
        if (
            module_names
            and excluded_patterns
            and all(
                any(fnmatch.fnmatch(name, pattern) for pattern in excluded_patterns)
                for name in module_names
            )
        ):
            return 0.0
        return 1.0


class WeightCostModel(AutoQuantizeCostModel):
    """Count all quantizable weights equally."""

    name = COST_MODEL_WEIGHT


class ActiveMoECostModel(AutoQuantizeCostModel):
    """Scale routed MoE expert weights by the active experts per-token ratio."""

    name = COST_MODEL_ACTIVE_MOE
    supported_cost_keys = frozenset(
        {ACTIVE_MOE_EXPERT_RATIO_KEY, EXCLUDED_MODULE_NAME_PATTERNS_KEY}
    )

    def normalize_cost_constraints(
        self, model: nn.Module, cost_constraints: dict[str, Any]
    ) -> dict[str, Any]:
        cost_constraints = super().normalize_cost_constraints(model, cost_constraints)
        active_moe_expert_ratio = cost_constraints.get(ACTIVE_MOE_EXPERT_RATIO_KEY)
        if active_moe_expert_ratio is None:
            active_moe_expert_ratio = infer_active_moe_expert_ratio(model)
            if active_moe_expert_ratio is None:
                raise ValueError(
                    "Could not infer active_moe_expert_ratio from model.config. "
                    "Pass it via constraints['cost']['active_moe_expert_ratio']."
                )

        if not (
            isinstance(active_moe_expert_ratio, (int, float))
            and not isinstance(active_moe_expert_ratio, bool)
            and 0.0 < active_moe_expert_ratio <= 1.0
        ):
            raise ValueError(
                "constraints['cost']['active_moe_expert_ratio'] must be in (0.0, 1.0]."
            )
        cost_constraints[ACTIVE_MOE_EXPERT_RATIO_KEY] = float(active_moe_expert_ratio)
        return cost_constraints

    def module_cost_weight(
        self, module_names: Sequence[str], cost_constraints: dict[str, Any]
    ) -> float:
        base_weight = super().module_cost_weight(module_names, cost_constraints)
        if base_weight == 0.0:
            return 0.0
        if any(is_routed_moe_module_name(n) for n in module_names):
            return cost_constraints[ACTIVE_MOE_EXPERT_RATIO_KEY]
        return base_weight


class LatencyCostModel(AutoQuantizeCostModel):
    """Price candidates by measured kernel latency from a ``haq_latency_v1`` LUT.

    Unlike the weight-based cost models, latency is a per-``(group, recipe)`` cost
    looked up from the LUT by the searcher; this class only validates the
    latency-specific ``constraints['cost']`` block (``lut_path``,
    ``deployment_profile``, ``m``) and continues to support
    ``excluded_module_name_patterns`` for cost-excluded groups. The top-level
    ``constraints['latency']`` block (``relative_to_min``) is validated in
    :func:`normalize_auto_quantize_constraints`.
    """

    name = COST_MODEL_LATENCY
    supported_cost_keys = frozenset(
        {
            LATENCY_LUT_PATH_KEY,
            LATENCY_DEPLOYMENT_PROFILE_KEY,
            LATENCY_M_KEY,
            EXCLUDED_MODULE_NAME_PATTERNS_KEY,
        }
    )

    def normalize_cost_constraints(
        self, model: nn.Module, cost_constraints: dict[str, Any]
    ) -> dict[str, Any]:
        cost_constraints = super().normalize_cost_constraints(model, cost_constraints)
        lut_path = cost_constraints.get(LATENCY_LUT_PATH_KEY)
        if not isinstance(lut_path, str) or not lut_path:
            raise ValueError(
                "constraints['cost']['lut_path'] must be a non-empty path to a "
                "haq_latency_v1 CSV for cost_model: latency."
            )
        deployment_profile = cost_constraints.get(LATENCY_DEPLOYMENT_PROFILE_KEY)
        if not isinstance(deployment_profile, str) or not deployment_profile:
            raise ValueError(
                "constraints['cost']['deployment_profile'] must be a non-empty string "
                "for cost_model: latency."
            )
        m = cost_constraints.get(LATENCY_M_KEY)
        if not (isinstance(m, int) and not isinstance(m, bool) and m > 0):
            raise ValueError(
                "constraints['cost']['m'] must be a positive integer for cost_model: latency."
            )
        return cost_constraints


_COST_MODELS: Final = {
    COST_MODEL_WEIGHT: WeightCostModel(),
    COST_MODEL_ACTIVE_MOE: ActiveMoECostModel(),
    COST_MODEL_LATENCY: LatencyCostModel(),
}


def get_auto_quantize_cost_model(name: str) -> AutoQuantizeCostModel:
    """Return the registered AutoQuantize cost model."""
    try:
        return _COST_MODELS[name]
    except KeyError as e:
        raise ValueError(
            f"Invalid constraints['cost_model']: {name}. Valid options are {tuple(_COST_MODELS)}."
        ) from e


def normalize_auto_quantize_constraints(
    model: nn.Module, constraints: dict[str, Any] | None
) -> dict[str, Any]:
    """Validate and normalize AutoQuantize constraints."""
    constraints = (
        {"effective_bits": DEFAULT_AUTO_QUANTIZE_EFFECTIVE_BITS}
        if constraints is None
        else dict(constraints)
    )
    unexpected_constraint_keys = set(constraints) - AUTO_QUANTIZE_CONSTRAINT_KEYS
    if unexpected_constraint_keys:
        raise ValueError(
            f"Unsupported auto_quantize constraints: {unexpected_constraint_keys}. Supported "
            "constraints are 'effective_bits', 'cost_model', 'cost', and 'latency'."
        )

    cost_model_name = constraints.get("cost_model", COST_MODEL_WEIGHT)
    if not isinstance(cost_model_name, str):
        raise ValueError("constraints['cost_model'] must be a string when provided.")
    cost_model = get_auto_quantize_cost_model(cost_model_name)

    cost_constraints = constraints.get("cost", {})
    if cost_constraints is None:
        cost_constraints = {}
    if not isinstance(cost_constraints, dict):
        raise ValueError("constraints['cost'] must be a dict when provided.")
    cost_constraints = cost_model.normalize_cost_constraints(model, dict(cost_constraints))

    # The latency cost model uses measured latency as the budget dimension and is
    # mutually exclusive with the effective-bits target. It requires the top-level
    # 'latency' block; every other cost model forbids it.
    if cost_model.name == COST_MODEL_LATENCY:
        if "effective_bits" in constraints:
            raise ValueError(
                "'effective_bits' and cost_model: latency are mutually exclusive. Provide the "
                "latency budget via constraints['latency']['relative_to_min'] instead."
            )
        constraints["latency"] = _normalize_latency_budget(constraints.get("latency"))
    elif constraints.get("latency") is not None:
        raise ValueError(
            "constraints['latency'] is only valid with cost_model: latency; "
            f"got cost_model={cost_model.name!r}."
        )

    constraints["cost_model"] = cost_model.name
    if cost_constraints or cost_model.name in (COST_MODEL_ACTIVE_MOE, COST_MODEL_LATENCY):
        constraints["cost"] = cost_constraints
    else:
        constraints.pop("cost", None)
    return constraints


def _normalize_latency_budget(latency_block: Any) -> dict[str, float]:
    """Validate the top-level ``constraints['latency']`` block for cost_model: latency."""
    if not isinstance(latency_block, dict):
        raise ValueError(
            "cost_model: latency requires a 'latency' block, e.g. "
            "constraints['latency'] = {'relative_to_min': 1.2}."
        )
    unknown = set(latency_block) - {LATENCY_RELATIVE_TO_MIN_KEY}
    if unknown:
        raise ValueError(f"Unsupported constraints['latency'] keys: {sorted(unknown)}.")
    relative_to_min = latency_block.get(LATENCY_RELATIVE_TO_MIN_KEY)
    if not (
        isinstance(relative_to_min, (int, float))
        and not isinstance(relative_to_min, bool)
        and relative_to_min >= 1.0
    ):
        raise ValueError(
            "constraints['latency']['relative_to_min'] must be a number >= 1.0 (the latency "
            "budget is relative_to_min * minimum achievable latency)."
        )
    return {LATENCY_RELATIVE_TO_MIN_KEY: float(relative_to_min)}
