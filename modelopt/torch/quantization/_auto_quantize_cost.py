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

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final

import regex as re
import torch.nn as nn

# Default target used by historical AutoQuantize calls when no explicit effective-bits
# constraint is supplied. The value is intentionally kept for backward compatibility.
DEFAULT_AUTO_QUANTIZE_EFFECTIVE_BITS: Final = 4.8

AUTO_QUANTIZE_CONSTRAINT_KEYS: Final = frozenset({"effective_bits", "cost_model", "cost"})
ACTIVE_MOE_EXPERT_RATIO_KEY: Final = "active_moe_expert_ratio"
COST_MODEL_WEIGHT: Final = "weight"
COST_MODEL_ACTIVE_MOE: Final = "active_moe"

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


class AutoQuantizeCostModel:
    """Base class for AutoQuantize effective-bits cost accounting."""

    name: str
    supported_cost_keys: frozenset[str] = frozenset()

    def normalize_cost_constraints(
        self, model: nn.Module, cost_constraints: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate and normalize cost-model-specific constraints."""
        unknown_cost_keys = set(cost_constraints) - self.supported_cost_keys
        if unknown_cost_keys:
            raise ValueError(f"Unsupported auto_quantize cost constraints: {unknown_cost_keys}.")
        return cost_constraints

    def module_cost_weight(
        self, module_names: Sequence[str], cost_constraints: dict[str, Any]
    ) -> float:
        """Return the cost multiplier for a group of modules."""
        return 1.0

    def total_weight_size(
        self,
        named_modules: Iterable[tuple[str, nn.Module]],
        is_auto_quantize_module: Callable[[nn.Module], bool],
        cost_constraints: dict[str, Any],
    ) -> float:
        """Return the cost denominator for the effective-bits constraint."""
        return sum(
            module.weight.numel() * self.module_cost_weight([name], cost_constraints)
            for name, module in named_modules
            if is_auto_quantize_module(module)
        )


class WeightCostModel(AutoQuantizeCostModel):
    """Count all quantizable weights equally."""

    name = COST_MODEL_WEIGHT


class ActiveMoECostModel(AutoQuantizeCostModel):
    """Scale routed MoE expert weights by the active experts per-token ratio."""

    name = COST_MODEL_ACTIVE_MOE
    supported_cost_keys = frozenset({ACTIVE_MOE_EXPERT_RATIO_KEY})

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
        if any(is_routed_moe_module_name(n) for n in module_names):
            return cost_constraints[ACTIVE_MOE_EXPERT_RATIO_KEY]
        return 1.0


_COST_MODELS: Final = {
    COST_MODEL_WEIGHT: WeightCostModel(),
    COST_MODEL_ACTIVE_MOE: ActiveMoECostModel(),
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
            f"Unsupported auto_quantize constraints: {unexpected_constraint_keys}. "
            "Supported constraints are 'effective_bits', 'cost_model', and 'cost'."
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

    constraints["cost_model"] = cost_model.name
    if cost_constraints or cost_model.name == COST_MODEL_ACTIVE_MOE:
        constraints["cost"] = cost_constraints
    else:
        constraints.pop("cost", None)
    return constraints
