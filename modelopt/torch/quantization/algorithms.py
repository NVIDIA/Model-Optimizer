# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Module for advanced quantization algorithms."""

import csv
import fnmatch
import gc
import hashlib
import math
import types
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from modelopt.torch.opt.conversion import ModeloptStateManager
from modelopt.torch.opt.hparam import CustomHPType, Hparam, HPType
from modelopt.torch.opt.searcher import LPS, BaseSearcher, SearchConfig, SearchStateDict
from modelopt.torch.opt.utils import get_hparam, named_hparams
from modelopt.torch.utils import create_param_grad_clear_hook, print_rank_0, report_memory
from modelopt.torch.utils.distributed import DistributedProcessGroup, ParallelState, is_master

from . import config as mtq_config
from . import model_calib
from ._auto_quantize_cost import (
    ACTIVE_MOE_EXPERT_RATIO_KEY,
    AUTO_QUANTIZE_CONSTRAINT_KEYS,
    AUTO_QUANTIZE_SCORE_MODEL_ACTIVE_WEIGHTED,
    AUTO_QUANTIZE_SCORE_MODEL_PER_ACTIVE,
    AUTO_QUANTIZE_SCORE_MODEL_PER_ELEMENT,
    AUTO_QUANTIZE_SCORE_MODEL_RAW,
    COST_MODEL_ACTIVE_MOE,
    COST_MODEL_WEIGHT,
    _get_module_weight_numel,
    get_auto_quantize_cost_model,
    normalize_auto_quantize_constraints,
)
from .config import QuantizeConfig, QuantizerAttributeConfig, QuantizerCfgEntry
from .conversion import set_quantizer_by_cfg
from .nn import QuantLinearConvBase, QuantModule, SequentialQuantizer, TensorQuantizer
from .utils import is_quantized_linear


def _is_fused_experts_module(module: nn.Module) -> bool:
    """Return True if ``module`` is a quantized fused-MoE-experts container.

    These modules expose plural ``*_input_quantizer`` and ``*_weight_quantizers``
    (an ``nn.ModuleList`` of per-expert quantizers) instead of the singular
    ``input_quantizer`` / ``weight_quantizer`` attrs found on standard
    ``nn.Linear``-derived QuantModules. AutoQuantize hparam discovery and cost
    accounting need to recognize this layout to enumerate fused experts as
    search dimensions.
    """
    # Late import to avoid a circular import at module load time.
    try:
        from .plugins.huggingface import _QuantFusedExperts
    except ImportError:
        return False
    return isinstance(module, _QuantFusedExperts)


# Quantizer attribute names that participate in AutoQuantize snapshot/restore.
_STD_QUANTIZER_ATTRS = ("input_quantizer", "weight_quantizer", "output_quantizer")
_FUSED_EXPERTS_QUANTIZER_ATTRS = (
    "gate_up_proj_input_quantizer",
    "gate_up_proj_weight_quantizers",
    "down_proj_input_quantizer",
    "down_proj_weight_quantizers",
)


def _get_quantizer_attrs(module: nn.Module) -> tuple[str, ...]:
    """Return the quantizer attribute names that AutoQuantize must snapshot/restore.

    For fused MoE experts, this returns the four plural quantizer attrs (two
    shared input quantizers + two ``ModuleList`` of per-expert weight quantizers).
    For standard Linear-derived QuantModules, returns the canonical trio.
    """
    if _is_fused_experts_module(module):
        return _FUSED_EXPERTS_QUANTIZER_ATTRS
    return _STD_QUANTIZER_ATTRS


def _make_fresh_quantizer_for_attr(module: nn.Module, attr_name: str) -> nn.Module:
    """Return a fresh, default quantizer object suitable to overwrite ``module.<attr_name>``.

    For ModuleList attrs (per-expert quantizers on fused-experts modules), the
    returned ModuleList preserves the original list length so per-expert
    enumeration stays consistent across recipes.
    """
    current = getattr(module, attr_name, None)
    if isinstance(current, nn.ModuleList):
        return nn.ModuleList(TensorQuantizer() for _ in range(len(current)))
    return TensorQuantizer()


def estimate_quant_compression(quant_cfg: QuantizeConfig) -> float:
    """Estimate the compression ratio of a quantization configuration.

    Right now, we find the minimum compression ratio across all quantizer attribute configs.
    This is not perfect but is a good proxy for the overall compression ratio. We will improve
    this in future releases.

    Args:
        quant_cfg: The quantization configuration to estimate compression for.

    Returns:
        float: The estimated compression ratio (0.0 to 1.0).
    """

    def estimate_quant_compression_for_quantizer(quantizer_attr_cfg):
        if isinstance(quantizer_attr_cfg, list):
            if not quantizer_attr_cfg:
                return 1.0
            return min(estimate_quant_compression_for_quantizer(q) for q in quantizer_attr_cfg)
        if isinstance(quantizer_attr_cfg, dict):
            # Handle raw quantizer cfg dicts (e.g. {"num_bits": (4, 3), "axis": None})
            if not quantizer_attr_cfg.get("enable", True):
                return 1.0
            num_bits = quantizer_attr_cfg.get("num_bits")
            if num_bits is None:
                return 1.0
            if isinstance(num_bits, tuple):
                return (sum(num_bits) + 1) / 16
            elif isinstance(num_bits, int):
                return num_bits / 16
            else:
                raise ValueError(f"Unknown quantization config {num_bits}")

        if isinstance(quantizer_attr_cfg, QuantizerAttributeConfig):
            if not quantizer_attr_cfg.enable:
                return 1.0
            if not hasattr(quantizer_attr_cfg, "num_bits"):
                return 1.0
            if isinstance(quantizer_attr_cfg.num_bits, tuple):
                return (sum(quantizer_attr_cfg.num_bits) + 1) / 16
            elif isinstance(quantizer_attr_cfg.num_bits, int):
                return quantizer_attr_cfg.num_bits / 16
            else:
                raise ValueError(f"Unknown quantization config {quantizer_attr_cfg.num_bits}")

        raise ValueError(f"Unknown type {type(quantizer_attr_cfg)}, {quantizer_attr_cfg}")

    cfgs = []
    for e in quant_cfg.quant_cfg:
        if e.get("enable", True) is False:
            continue
        c = e.get("cfg")
        if c is not None:
            cfgs.append(c)
    return estimate_quant_compression_for_quantizer(cfgs) if cfgs else 1.0


class QuantRecipe(CustomHPType):
    """A subclass of QuantizeConfig enabling auto_quantize specific configurations.

    Args:
        quant_cfg: str or dict or None. dict is used for custom quantization formats.
        name: name for custom quantization formats. Only used if quantization format is a custom
            format not available in :mod:`modelopt.torch.quantization.config`.
    """

    def __init__(self, quant_cfg: str | dict[str, Any] | None = None, name: str | None = None):
        """Initialize the QuantRecipe with the quantization configuration."""
        name = self.get_auto_name_for_config(quant_cfg) or name

        if quant_cfg is None:
            quant_cfg = {"quant_cfg": [{"quantizer_name": "*", "enable": False}]}
        elif isinstance(quant_cfg, str):
            assert hasattr(mtq_config, quant_cfg), f"Unknown quantization format {quant_cfg}"
            quant_cfg = getattr(mtq_config, quant_cfg)
        else:
            assert name is not None, "name must be provided for custom quantization formats"

        self.config = mtq_config.QuantizeConfig(**quant_cfg)  # type: ignore [arg-type]

        # Disable KV Cache quantization
        # Currently KV Cache quantization is enabled for some quantization formats and disabled for others
        # This breaks the monotonicity of the quantization formats in terms of weight compression Vs accuracy
        self.config.quant_cfg.append(
            QuantizerCfgEntry(quantizer_name="*output_quantizer", enable=False)
        )

        self.compression = estimate_quant_compression(self.config)

        self._str_repr: str = f"{name}(effective-bits: {self.compression * 16})"

    @staticmethod
    def get_auto_name_for_config(quant_cfg: str | dict[str, Any] | None) -> str | None:
        """Get a name for the quantization configuration."""
        if quant_cfg is None:
            return "NONE"
        if isinstance(quant_cfg, str):
            return quant_cfg
        for quant_cfg_name in mtq_config.choices:
            if quant_cfg == getattr(mtq_config, quant_cfg_name):
                return quant_cfg_name
        return None

    @property
    def num_bits(self) -> int:
        """Get the number of bits for the quantization format."""
        return int(self.compression * 16)

    def __str__(self) -> str:
        return self._str_repr

    def __repr__(self) -> str:
        return self._str_repr

    def __lt__(self, other: "QuantRecipe"):
        return self.compression < other.compression

    def __eq__(self, other: object):
        assert isinstance(other, QuantRecipe)
        return self._str_repr == other._str_repr

    def __hash__(self) -> int:
        return hash(self._str_repr)

    @staticmethod
    def disable_folding_pqs_to_weights():
        """Disable the folding of pre_quant_scale to weights."""
        model_calib._ENABLE_FOLDING_PQS_TO_WEIGHTS = False

    @staticmethod
    def fold_pqs_to_weights(model):
        """Fold the pre_quant_scale in weight_quantizers to weights."""
        model_calib._ENABLE_FOLDING_PQS_TO_WEIGHTS = True
        for name, module in model.named_modules():
            if is_quantized_linear(module):
                with SequentialQuantizer.convert_to_single_quantizer(module):
                    if module.weight_quantizer.pre_quant_scale is not None:
                        weight_pqs = module.weight_quantizer.pre_quant_scale
                        delattr(module.weight_quantizer, "_pre_quant_scale")
                        model_calib._apply_weight_pre_quant_scale(module, weight_pqs)


class QuantRecipeHparam(Hparam):
    """An Hparam for quantization recipes.

    See :class:`Hparam <modelopt.torch.opt.hparam.Hparam>` for more details. In addition, this Hparam also:

    * Keeps a link to its ``quant_modules`` and ``score_modules`` and sets the quantizers for the
      ``quant_modules`` based on the active recipe.
    * Provides ``get_score()`` and ``get_cost()`` methods to evaluate recipes.
    * Registers itself with each ``score_module`` via the ``_hparams_for_scoring`` attribute.
    """

    def __init__(
        self,
        choices: Sequence[QuantRecipe] | None = None,
        quant_modules: list[nn.Module] | None = None,
        score_modules: list[nn.Module] | None = None,
        name: str | None = None,
        quant_module_names: list[str] | None = None,
        cost_weight: float = 1.0,
    ) -> None:
        """Initializes Hparam with original value and choices."""
        choices = sorted({*(choices if choices else []), QuantRecipe(quant_cfg=None)})
        super().__init__(choices, original=choices[0])

        self.name = name
        self.quant_module_names = quant_module_names or []
        assert cost_weight >= 0.0, "cost_weight must be non-negative."
        self.cost_weight = cost_weight

        self.quant_modules = list(set(quant_modules or []))
        self.score_modules = list(set(score_modules or self.quant_modules))

        # This is a hack; We dont want to make the input_quantizer, weight_quantizer, output_quantizer
        # a dynamic attribute for backward compatibility with the model_calib.py
        # TODO: Make input_quantizer, weight_quantizer, output_quantizer a dynamic attribute and get rid of this hack
        # NOTE: For fused-experts modules, the relevant attrs are plural
        # (``*_input_quantizer`` + ``*_weight_quantizers`` ModuleList) — see
        # ``_get_quantizer_attrs``. Both layouts share the same snapshot dict
        # shape so ``active.setter`` swaps the right child modules.
        self._all_quantizer_choices = {quant_recipe: {} for quant_recipe in self.choices}

        quant_recipe: QuantRecipe
        for quant_recipe in self.choices:
            for quant_module in self.quant_modules:
                attr_names = _get_quantizer_attrs(quant_module)
                for attr_name in attr_names:
                    setattr(
                        quant_module,
                        attr_name,
                        _make_fresh_quantizer_for_attr(quant_module, attr_name),
                    )

                set_quantizer_by_cfg(quant_module, quant_recipe.config.quant_cfg)
                self._all_quantizer_choices[quant_recipe][quant_module] = {
                    attr_name: getattr(quant_module, attr_name) for attr_name in attr_names
                }

        self.active = self.original

        # Importance dict is keyed by score_module (where the score is computed)
        self._importance_dict = {
            quant_recipe: dict.fromkeys(self.score_modules) for quant_recipe in self.choices
        }

        # Attach this hparam to each score_module's set of hparams it scores
        for score_module in self.score_modules:
            if not hasattr(score_module, "_hparams_for_scoring"):
                score_module._hparams_for_scoring = set()
            score_module._hparams_for_scoring.add(self)

    @property
    def active(self) -> HPType:
        """Return the currently active value."""
        return self._active

    @active.setter
    def active(self, val: HPType | None):
        """Set the active value with a sanity check for choices and dynamic hparams."""
        val = self.original if val is None else val
        assert val in self._choices, f"val = {val}, choices = {self.choices}"
        if self.is_configurable:
            self._active = val
        else:
            assert self._active == val

        for nn_module, quantizer_choices in self._all_quantizer_choices[val].items():
            for quantizer_attr_name, quantizer in quantizer_choices.items():
                setattr(nn_module, quantizer_attr_name, quantizer)

    @property
    def importance(self) -> dict:
        """Raises an error since this is not a useful abstraction for AutoQuantize."""
        raise NotImplementedError

    def get_score(self, recipe: QuantRecipe) -> float:
        """Get the score for a given recipe."""
        total_score = 0
        for score_module in self.score_modules:
            importance = self._importance_dict[recipe][score_module]
            if importance is None:
                continue

            parallel_state = getattr(score_module, "parallel_state", None)

            if parallel_state is None:
                total_score += importance.cpu().item()
                continue

            if parallel_state.expert_model_parallel_group.is_initialized():
                # TODO: Support expert model parallelism for score estimation
                warnings.warn("AutoQuantize does not support expert model parallelism yet.")
            importance = importance.cpu()
            importance = DistributedProcessGroup.get_dist_syncd_obj(
                importance,
                [parallel_state.tensor_parallel_group, parallel_state.data_parallel_group],
                sum,
            )
            total_score += importance.item()
        return total_score

    def get_cost(self, recipe: QuantRecipe, cost_weight: float | None = None) -> float:
        """Get the cost for a given recipe.

        The cost is the total weight size of the quantizable modules multiplied by
        the compression ratio of the recipe.
        """
        cost_weight = self.cost_weight if cost_weight is None else cost_weight
        cost = 0
        for quant_module in self.quant_modules:
            weight_size = (
                _AutoQuantizeBaseSearcher._get_total_weight_size([quant_module]) * cost_weight
            )
            parallel_state = getattr(quant_module, "parallel_state", None)

            if parallel_state is None:
                cost += weight_size * recipe.compression
                continue

            if parallel_state.expert_model_parallel_group.is_initialized():
                # TODO: Support expert model parallelism
                warnings.warn("AutoQuantize does not support expert model parallelism yet.")

            weight_size = DistributedProcessGroup.get_dist_syncd_obj(
                weight_size,
                [parallel_state.tensor_parallel_group],
                sum,
            )

            # Across data parallel groups, the weight size is the same for all the ranks.
            weight_size = DistributedProcessGroup.get_dist_syncd_obj(
                weight_size,
                [parallel_state.data_parallel_group],
                lambda a: a[0],
            )
            cost += weight_size * recipe.compression

        return cost

    @property
    def attrs(self) -> list[str]:
        """Return the attributes of the hparam for repr."""
        return ["name", "cost_weight", *super().attrs]


_LINEAR_ATTN_QKVZ_RE = re.compile(r"^(.*?\.linear_attn)\.(?:in_proj_qkv|in_proj_z)$")
_LINEAR_ATTN_BA_RE = re.compile(r"^(.*?\.linear_attn)\.(?:in_proj_a|in_proj_b)$")
_LINEAR_ATTN_LAYER_RE = re.compile(r"^(.*?\.linear_attn)\.(?:in_proj_qkv|in_proj_z|out_proj)$")
_SELF_ATTN_LAYER_RE = re.compile(r"^(.*?\.self_attn)\.(?:q_proj|k_proj|v_proj|o_proj)$")
_SELF_ATTN_SCORE_RE = re.compile(r"^(.*?\.self_attn)\.(?:q_proj|k_proj|v_proj|o_proj)$")
_LINEAR_ATTN_SCORE_RE = re.compile(
    r"^(.*?\.linear_attn)\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)$"
)
_FUSED_ROUTED_EXPERTS_RE = re.compile(r"^((?:.*\.)?mlp)\.experts$")
_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")
_AUTO_QUANTIZE_RESPONSE_RISK_CATEGORY_ALIASES = {
    "linear_attn": {"linear_attn_layer"},
    "self_attn": {"self_attn_layer"},
    "shared_expert": {"shared_moe"},
    "routed_expert": {"routed_moe"},
}
AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED = "runtime_fused"
AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED_LINEAR_ATTN_LAYER = "runtime_fused+linear_attn_layer"
AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED_SELF_ATTN_LAYER = "runtime_fused+self_attn_layer"
AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED_LINEAR_SELF_ATTN_LAYER = (
    "runtime_fused+linear_attn_layer+self_attn_layer"
)
AUTO_QUANTIZE_GROUPING_SCHEME_ALIASES = {
    "default": AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED,
    AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED: AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED,
    AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED_LINEAR_ATTN_LAYER: (
        AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED_LINEAR_ATTN_LAYER
    ),
    AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED_SELF_ATTN_LAYER: (
        AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED_SELF_ATTN_LAYER
    ),
    AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED_LINEAR_SELF_ATTN_LAYER: (
        AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED_LINEAR_SELF_ATTN_LAYER
    ),
}
AUTO_QUANTIZE_GROUPING_SCHEMES = frozenset(AUTO_QUANTIZE_GROUPING_SCHEME_ALIASES)
AUTO_QUANTIZE_SCORE_COMPONENT_TRACKING_NONE = "none"
AUTO_QUANTIZE_SCORE_COMPONENT_TRACKING_BATCH = "batch"
AUTO_QUANTIZE_SCORE_COMPONENT_TRACKING_MODES = frozenset(
    {
        AUTO_QUANTIZE_SCORE_COMPONENT_TRACKING_NONE,
        AUTO_QUANTIZE_SCORE_COMPONENT_TRACKING_BATCH,
    }
)
AUTO_QUANTIZE_HIDDEN_RECON_SCORE_REDUCE_MEAN = "mean"
AUTO_QUANTIZE_HIDDEN_RECON_SCORE_REDUCE_MAX = "max"
AUTO_QUANTIZE_HIDDEN_RECON_SCORE_REDUCERS = frozenset(
    {
        AUTO_QUANTIZE_HIDDEN_RECON_SCORE_REDUCE_MEAN,
        AUTO_QUANTIZE_HIDDEN_RECON_SCORE_REDUCE_MAX,
    }
)
_AUTO_QUANTIZE_SCORE_COMPONENT_SOURCE_KEYS = (
    "source_id",
    "source",
    "dataset",
    "dataset_name",
    "source_name",
)


def _linear_attn_qkvz_group_key(_model, name: str) -> str | None:
    m = _LINEAR_ATTN_QKVZ_RE.match(name)
    return f"{m.group(1)}/qkvz" if m else None


def _linear_attn_ba_group_key(_model, name: str) -> str | None:
    m = _LINEAR_ATTN_BA_RE.match(name)
    return f"{m.group(1)}/ba" if m else None


def _linear_attn_layer_group_key(_model, name: str) -> str | None:
    m = _LINEAR_ATTN_LAYER_RE.match(name)
    return f"{m.group(1)}/layer" if m else None


def _self_attn_layer_group_key(_model, name: str) -> str | None:
    m = _SELF_ATTN_LAYER_RE.match(name)
    return f"{m.group(1)}/layer" if m else None


def _self_attn_score_module(_model, name: str) -> str | None:
    m = _SELF_ATTN_SCORE_RE.match(name)
    return m.group(1) if m else None


def _linear_attn_score_module(_model, name: str) -> str | None:
    m = _LINEAR_ATTN_SCORE_RE.match(name)
    return m.group(1) if m else None


def _fused_routed_experts_score_module(model, name: str) -> str | None:
    m = _FUSED_ROUTED_EXPERTS_RE.match(name)
    if not m:
        return None
    try:
        module = model.get_submodule(name)
    except AttributeError:
        return None
    return m.group(1) if _is_fused_experts_module(module) else None


def _auto_quantize_recipe_name(recipe: Any) -> str:
    text = str(recipe)
    upper = text.upper()
    if "NONE" in upper:
        return "BF16"
    if "FP8" in upper:
        return "FP8"
    if "W4A16" in upper and "NVFP4" in upper:
        return "W4A16_NVFP4"
    if "NVFP4" in upper:
        return "NVFP4"
    return text.replace("\n", " ")


def _auto_quantize_candidate_signature(recipe_info: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for hparam_name in sorted(recipe_info):
        digest.update(hparam_name.encode("utf-8"))
        digest.update(b"\t")
        digest.update(
            _auto_quantize_recipe_name(recipe_info[hparam_name]["format"]).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _auto_quantize_module_category(name: str) -> str:
    if "lm_head" in name:
        return "lm_head"
    if ".mlp.experts" in name:
        return "routed_expert"
    if ".mlp.shared_expert" in name:
        return "shared_expert"
    if ".linear_attn." in name and name.rsplit(".", 1)[-1] in {"in_proj_a", "in_proj_b"}:
        return "linear_attn_ab"
    if ".linear_attn." in name:
        return "linear_attn"
    if ".self_attn." in name:
        return "self_attn"
    if ".conv" in name:
        return "conv"
    return "other"


def _auto_quantize_stats_categories(candidate_stat: dict[str, Any]) -> set[str]:
    categories = {
        _auto_quantize_module_category(name)
        for name in list(candidate_stat.get("module_names") or [])
    }
    categories.discard("")
    if not categories:
        categories.add("other")
    for category in list(categories):
        categories.update(_AUTO_QUANTIZE_RESPONSE_RISK_CATEGORY_ALIASES.get(category, set()))
    return categories


def _auto_quantize_stats_layer(candidate_stat: dict[str, Any]) -> str:
    layers = sorted(
        {
            match.group(1)
            for name in list(candidate_stat.get("module_names") or [])
            if (match := _LAYER_INDEX_RE.search(name))
        }
    )
    return layers[0] if len(layers) == 1 else ""


def _load_response_risk_rows(response_risk: dict[str, Any]) -> list[dict[str, Any]]:
    entries = response_risk.get("entries")
    if entries is not None:
        return entries

    source_path = response_risk.get("source_path")
    if source_path is None:
        return []

    path = Path(source_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Empty AutoQuantize response_risk table: {path}")
        return list(reader)


def _load_candidate_rerank_rows(candidate_rerank: dict[str, Any]) -> list[dict[str, Any]]:
    entries = candidate_rerank.get("entries")
    if entries is not None:
        return entries

    source_path = candidate_rerank.get("source_path")
    if source_path is None:
        return []

    path = Path(source_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Empty AutoQuantize candidate_rerank table: {path}")
        return list(reader)


def _load_candidate_rerank_signal(candidate_rerank: dict[str, Any] | None) -> dict[str, float]:
    """Load optional candidate-packet score adjustments by packet signature."""
    if not candidate_rerank:
        return {}

    id_field = candidate_rerank.get("id_field", "signature")
    score_field = candidate_rerank.get("score_field", "rerank_score")
    scale = float(candidate_rerank.get("scale", 1.0))
    signal: dict[str, float] = {}
    for line_no, row in enumerate(_load_candidate_rerank_rows(candidate_rerank), start=2):
        packet_id = str(row.get(id_field) or "").strip()
        if not packet_id and (row.get("family") or row.get("category")):
            continue
        if not packet_id:
            raise ValueError(f"AutoQuantize candidate_rerank row {line_no} requires '{id_field}'.")
        raw_score = str(row.get(score_field) or "").strip()
        if not raw_score:
            raise ValueError(
                f"AutoQuantize candidate_rerank row {line_no} requires '{score_field}'."
            )
        value = float(raw_score) * scale
        if not math.isfinite(value):
            raise ValueError(f"AutoQuantize candidate_rerank row {line_no} is non-finite.")
        signal[packet_id] = signal.get(packet_id, 0.0) + value
    return signal


def _load_candidate_family_rerank_rules(
    candidate_rerank: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Load optional candidate-level family allocation rerank rules.

    These rules are applied after LP packet enumeration, using packet-level
    ``family_format_counts``. They are intended for validation-derived triage
    and prioritization, not for metadata-only promotion.
    """
    if not candidate_rerank:
        return []

    rows = list(candidate_rerank.get("family_entries") or [])
    rows.extend(
        row
        for row in _load_candidate_rerank_rows(candidate_rerank)
        if row.get("family") or row.get("category")
    )
    scale = float(candidate_rerank.get("family_scale", candidate_rerank.get("scale", 1.0)))
    rules: list[dict[str, Any]] = []
    for line_no, row in enumerate(rows, start=2):
        family = str(row.get("family") or row.get("category") or "").strip()
        if not family:
            raise ValueError(
                f"AutoQuantize candidate_rerank family row {line_no} requires family/category."
            )
        fmt = str(row.get("format") or row.get("selected_format") or "*").strip() or "*"
        raw_score = str(
            row.get("rerank_score") or row.get("risk") or row.get("score") or ""
        ).strip()
        if not raw_score:
            raise ValueError(
                f"AutoQuantize candidate_rerank family row {line_no} requires rerank_score/risk/score."
            )
        score = float(raw_score) * scale
        if not math.isfinite(score):
            raise ValueError(f"AutoQuantize candidate_rerank family row {line_no} is non-finite.")
        rules.append(
            {
                "family": family,
                "format": _auto_quantize_recipe_name(fmt) if fmt != "*" else "*",
                "score": score,
                "mode": str(row.get("mode") or "packet").strip() or "packet",
                "count_level": str(row.get("count_level") or "module").strip() or "module",
                "min_count": row.get("min_count"),
                "max_count": row.get("max_count"),
                "min_frac": row.get("min_frac"),
                "max_frac": row.get("max_frac"),
            }
        )
    return rules


def _load_response_risk_signal(
    response_risk: dict[str, Any] | None,
) -> dict[tuple[str, str, str, str], float]:
    """Load hparam/category response-risk penalties from a descriptor.

    Supported row schemas mirror ``scripts/replay_autoq_lps_from_state.py``:
    ``hparam,format,risk`` and ``category,layer,format,risk``. Candidate-level
    rows without hparam/category are ignored by the LP objective; they remain
    useful as packet/backtest gates.
    """
    if not response_risk:
        return {}

    risk_metric_filter = response_risk.get("risk_metric")
    scale = float(response_risk.get("scale", 1.0))
    signal: dict[tuple[str, str, str, str], float] = {}
    for line_no, row in enumerate(_load_response_risk_rows(response_risk), start=2):
        if risk_metric_filter:
            risk_metric = str(row.get("risk_metric") or "").strip()
            if risk_metric and risk_metric != risk_metric_filter:
                continue
        fmt = str(row.get("format") or "").strip()
        raw_risk = str(row.get("risk") or "").strip()
        if not fmt or not raw_risk:
            raise ValueError(f"AutoQuantize response_risk row {line_no} requires format and risk.")
        value = float(raw_risk) * scale
        if not math.isfinite(value):
            raise ValueError(f"AutoQuantize response_risk row {line_no} is non-finite.")
        hparam = str(row.get("hparam") or "").strip()
        category = str(row.get("category") or row.get("family") or "").strip()
        layer = str(row.get("layer") or "").strip()
        if hparam:
            signal[("hparam", hparam, "", fmt)] = (
                signal.get(("hparam", hparam, "", fmt), 0.0) + value
            )
        if category and category != "candidate_response":
            if layer == "*":
                category_key = ("category", category, "", fmt)
                signal[category_key] = signal.get(category_key, 0.0) + value
            elif layer:
                layer_key = ("category_layer", category, layer, fmt)
                signal[layer_key] = signal.get(layer_key, 0.0) + value
            else:
                category_key = ("category", category, "", fmt)
                signal[category_key] = signal.get(category_key, 0.0) + value
    return signal


def _response_risk_penalty(
    hparam_name: str,
    candidate_stat: dict[str, Any],
    fmt: str,
    response_risk_signal: dict[tuple[str, str, str, str], float],
) -> float:
    if not response_risk_signal:
        return 0.0

    hparam_names = {hparam_name}
    if hparam_name.endswith(".quant_recipe"):
        hparam_names.add(hparam_name.removesuffix(".quant_recipe"))
    categories = _auto_quantize_stats_categories(candidate_stat)
    layer = _auto_quantize_stats_layer(candidate_stat)
    keys: list[tuple[str, str, str, str]] = []
    keys.extend(("hparam", name, "", fmt) for name in hparam_names if name)
    keys.extend(("hparam", name, "", "*") for name in hparam_names if name)
    for category in categories:
        keys.append(("category_layer", category, layer, fmt))
        keys.append(("category", category, "", fmt))
        keys.append(("category_layer", category, layer, "*"))
        keys.append(("category", category, "", "*"))
    keys.extend(
        [
            ("category_layer", "*", layer, fmt),
            ("category", "*", "", fmt),
            ("category", "*", "", "*"),
        ]
    )
    return sum(response_risk_signal.get(key, 0.0) for key in keys)


class _AutoQuantizeBaseSearcher(BaseSearcher, ABC):
    """Base searcher for AutoQuantize algorithm."""

    # This searcher finds optimal per-layer quantization by searching across quantization formats
    # for each quantizable module (quant module). Optionally, quant grouping rules can restrict
    # certain modules to share the same format. Sensitivity scores are computed from perturbations
    # at score modules. See AutoQuantizeGradientSearcher for detailed documentation.

    candidate_stats: dict[str, dict[str, Any]]
    best: dict[str, Any]
    quantizer_states: dict
    last_candidate_packets: list[dict[str, Any]]
    method_name: str | None = None

    quant_grouping_rules = [
        r"^(.*?)\.(q_proj|k_proj|v_proj)$",  # q_proj, k_proj, v_proj for llama like models
        # gate_proj, up_proj, down_proj for Qwen3 like MoE models
        r"^(.*?\.mlp\.experts)\.\d+\.(gate_proj|up_proj|down_proj)$",
        # Qwen3 shared experts are also fused/semantic groups. Keep gate/up/down
        # together instead of letting the generic gate/up rule split down_proj.
        r"^((?:.*\.)?mlp\.shared_expert)\.(gate_proj|up_proj|down_proj)$",
        r"^(.*?\.mixer\.experts)\.\d+\.(up_proj|down_proj)$",  # NemotronH MoE experts
        r"^(.*?)\.(gate_proj|up_proj)$",  # gate_proj, up_proj for llama like models
        r"^(.*?)\.(\d+\.(w1|w2|w3))$",  # mixtral experts
        r"^(.*?)\.((w1_linear|w2_linear|w3_linear)\.\d+)$",  # dbrx experts
        # Qwen3.5/3.6 hybrid linear_attn: vLLM fuses (in_proj_qkv, in_proj_z)
        # into ``in_proj_qkvz`` and (in_proj_a, in_proj_b) into ``in_proj_ba`` and
        # requires fused shards to share quant_algo. Two callables (not one
        # regex) so qkv+z and a+b produce DIFFERENT group keys; each pair
        # stays with its own fusion partner.
        _linear_attn_qkvz_group_key,
        _linear_attn_ba_group_key,
    ]

    score_module_rules = []

    @property
    def default_search_config(self):
        """Get the default config for the searcher."""
        return {
            "quantization_formats": ["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG"],
            "data_loader": None,
            "score_data_loader": None,
            "num_calib_steps": 512,
            "num_score_steps": 128,
            "data_signature": None,
            "deployment": None,
            "disabled_layers": None,
            "verbose": is_master(),
            "checkpoint": None,
            "cost_model": COST_MODEL_WEIGHT,
            "cost": {},
            "active_moe_expert_ratio": None,
            "quant_grouping_scheme": AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED,
            "score_component_tracking": AUTO_QUANTIZE_SCORE_COMPONENT_TRACKING_NONE,
            "hidden_recon_score_windows": ["full"],
            "hidden_recon_score_reduce": AUTO_QUANTIZE_HIDDEN_RECON_SCORE_REDUCE_MEAN,
        }

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Get the default state dict for AutoQuantize."""
        return {
            "method": self.method_name,
            "scoring_signature": None,
            "cost_model": "weight",
            "cost": {},
            "active_moe_expert_ratio": None,
            "cost_denominator": None,
            "disabled_layers": None,
            "quant_grouping_scheme": AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED,
            "score_component_tracking": AUTO_QUANTIZE_SCORE_COMPONENT_TRACKING_NONE,
            "hidden_recon_score_windows": ["full"],
            "hidden_recon_score_reduce": AUTO_QUANTIZE_HIDDEN_RECON_SCORE_REDUCE_MEAN,
            "score_component_metadata": {},
            "candidate_stats": defaultdict(dict),
            "quantizer_states": {},
            "best": {"recipe": {}, "constraints": {}, "score": float("inf"), "is_satisfied": False},
        }

    def sanitize_search_config(self, config: SearchConfig | None) -> SearchConfig:
        """Sanitize the search config dict."""
        config = config or {}
        config = super().sanitize_search_config(config)
        assert config["data_loader"] is not None, (
            "`data_loader` must be provided for `auto_quantize`."
        )
        if config["score_data_loader"] is None:
            config["score_data_loader"] = config["data_loader"]
        assert config["forward_step"] is not None, (
            "`forward_step` must be provided for `auto_quantize`."
        )
        quant_grouping_scheme = config["quant_grouping_scheme"]
        if quant_grouping_scheme not in AUTO_QUANTIZE_GROUPING_SCHEMES:
            raise ValueError(
                f"quant_grouping_scheme must be one of {sorted(AUTO_QUANTIZE_GROUPING_SCHEMES)}."
            )
        config["quant_grouping_scheme"] = AUTO_QUANTIZE_GROUPING_SCHEME_ALIASES[
            quant_grouping_scheme
        ]
        if config["score_component_tracking"] not in AUTO_QUANTIZE_SCORE_COMPONENT_TRACKING_MODES:
            raise ValueError(
                "score_component_tracking must be one of "
                f"{sorted(AUTO_QUANTIZE_SCORE_COMPONENT_TRACKING_MODES)}."
            )
        config["hidden_recon_score_windows"] = _normalize_hidden_recon_score_windows(
            config.get("hidden_recon_score_windows", ["full"])
        )
        if config["hidden_recon_score_reduce"] not in AUTO_QUANTIZE_HIDDEN_RECON_SCORE_REDUCERS:
            raise ValueError(
                "hidden_recon_score_reduce must be one of "
                f"{sorted(AUTO_QUANTIZE_HIDDEN_RECON_SCORE_REDUCERS)}."
            )
        return config

    def load_search_checkpoint(self) -> bool:
        return super().load_search_checkpoint(strict=False)

    @staticmethod
    def _is_auto_quantize_module(module):
        if (is_quantized_linear(module) or isinstance(module, QuantLinearConvBase)) and isinstance(
            module, QuantModule
        ):
            return True
        # Fused MoE experts: a single ``QuantModule`` that owns N per-expert
        # weight quantizers in an ``nn.ModuleList`` plus shared input quantizers.
        # All N experts in a layer share one search dimension (one recipe per
        # fused module).
        return _is_fused_experts_module(module) and isinstance(module, QuantModule)

    @staticmethod
    def _get_search_recipes(quantization_formats):
        return sorted(
            {
                QuantRecipe(quant_cfg=q[0], name=q[1])
                if isinstance(q, tuple)
                else QuantRecipe(quant_cfg=q)
                for q in quantization_formats
            }
        )

    def _apply_quant_group_rule(self, name: str, rule) -> str | None:
        """Apply a single quant_group_rule to a module name.

        Args:
            name: Module name
            rule: Either a regex pattern string or a callable that returns a unique key;
                If callable, it should take the model and the name as input and return the unique key

        Returns:
            The group key if the rule matches, None otherwise
        """
        if callable(rule):
            return rule(self.model, name)
        else:
            # Regex pattern
            pattern = re.compile(rule)
            match = pattern.match(name)
            if match:
                return match.group(1)
        return None

    def _apply_score_group_rule(self, name: str, rule) -> str | None:
        """Apply a single score_group_rule to a module name.

        Args:
            name: Module name
            rule: Either a regex pattern string or a callable that returns the score module name.
                If callable, it should take the model and the name as input and return the score module name

        Returns:
            The score module name if the rule matches, None otherwise
        """
        if callable(rule):
            return rule(self.model, name)
        else:
            # Regex pattern - return the matched name or full match
            pattern = re.compile(rule)
            match = pattern.match(name)
            if match:
                # For score rules, return the full match or first group
                return match.group(0) if match.lastindex is None else match.group(1)
        return None

    def _get_quant_grouping_rules(self):
        rules: list[Any] = []
        scheme = self.config.get(
            "quant_grouping_scheme",
            AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED,
        )
        if "linear_attn_layer" in scheme:
            # Group deployable linear-attention projections into one functional
            # decision. Keep A/B on the existing fused pair so disabled A/B do
            # not disable qkv/z/out search.
            rules.append(_linear_attn_layer_group_key)
        if "self_attn_layer" in scheme:
            rules.append(_self_attn_layer_group_key)
        rules.extend(self.quant_grouping_rules)
        return rules

    @staticmethod
    def _rule_signature(rule) -> str:
        if callable(rule):
            module = getattr(rule, "__module__", "")
            qualname = getattr(rule, "__qualname__", repr(rule))
            return f"{module}.{qualname}"
        return str(rule)

    @staticmethod
    def _callable_signature(func) -> str | None:
        if func is None:
            return None
        module = getattr(func, "__module__", "")
        qualname = getattr(func, "__qualname__", repr(func))
        return f"{module}.{qualname}"

    @staticmethod
    def _data_loader_signature(data_loader) -> dict[str, Any] | None:
        if data_loader is None:
            return None
        signature: dict[str, Any] = {"type": type(data_loader).__qualname__}
        with suppress(TypeError):
            signature["len"] = len(data_loader)
        for attr_name in ("batch_size", "drop_last"):
            if hasattr(data_loader, attr_name):
                value = getattr(data_loader, attr_name)
                if isinstance(value, (int, float, str, bool, type(None))):
                    signature[attr_name] = value
        dataset = getattr(data_loader, "dataset", None)
        if dataset is not None:
            signature["dataset_type"] = type(dataset).__qualname__
            with suppress(TypeError):
                signature["dataset_len"] = len(dataset)
        return signature

    def _build_scoring_signature(self, search_recipes: Sequence[QuantRecipe]) -> dict[str, Any]:
        """Return the state signature that decides if saved scores are reusable."""
        return {
            "schema_version": 1,
            "method": self.method_name,
            "quantization_formats": [
                {
                    "name": _auto_quantize_recipe_name(recipe),
                    "num_bits": recipe.num_bits,
                    "algorithm": repr(recipe.config.algorithm),
                    "quant_cfg": repr(recipe.config.quant_cfg),
                }
                for recipe in search_recipes
            ],
            "num_calib_steps": self.config["num_calib_steps"],
            "num_score_steps": self.config["num_score_steps"],
            "data_signature": self.config["data_signature"],
            "disabled_layers": self.config["disabled_layers"],
            "cost_model": self.config["cost_model"],
            "cost": self.config["cost"],
            "active_moe_expert_ratio": self.config["active_moe_expert_ratio"],
            "quant_grouping_scheme": self.config["quant_grouping_scheme"],
            "score_component_tracking": self.config["score_component_tracking"],
            "hidden_recon_score_windows": self.config["hidden_recon_score_windows"],
            "hidden_recon_score_reduce": self.config["hidden_recon_score_reduce"],
            "quant_grouping_rules": [
                self._rule_signature(rule) for rule in self._get_quant_grouping_rules()
            ],
            "score_module_rules": [self._rule_signature(rule) for rule in self.score_module_rules],
            "data_loader": self._data_loader_signature(self.config["data_loader"]),
            "score_data_loader": self._data_loader_signature(self.config["score_data_loader"]),
            "forward_step": self._callable_signature(self.config["forward_step"]),
            "loss_func": self._callable_signature(self.config.get("loss_func")),
            "forward_backward_step": self._callable_signature(
                self.config.get("forward_backward_step")
            ),
        }

    @staticmethod
    def _signature_diff(restored: dict[str, Any] | None, current: dict[str, Any]) -> str:
        if restored is None:
            return "checkpoint has no scoring_signature"
        restored_normalized = _AutoQuantizeBaseSearcher._normalize_scoring_signature(restored)
        current_normalized = _AutoQuantizeBaseSearcher._normalize_scoring_signature(current)
        assert restored_normalized is not None and current_normalized is not None
        changed = [
            key
            for key in sorted(set(restored_normalized) | set(current_normalized))
            if restored_normalized.get(key) != current_normalized.get(key)
        ]
        return ", ".join(changed)

    @staticmethod
    def _normalize_scoring_signature(signature: dict[str, Any] | None) -> dict[str, Any] | None:
        if signature is None:
            return None
        signature = dict(signature)
        signature.setdefault(
            "score_component_tracking", AUTO_QUANTIZE_SCORE_COMPONENT_TRACKING_NONE
        )
        signature.setdefault("hidden_recon_score_windows", ["full"])
        signature.setdefault(
            "hidden_recon_score_reduce", AUTO_QUANTIZE_HIDDEN_RECON_SCORE_REDUCE_MEAN
        )
        signature.setdefault("score_data_loader", signature.get("data_loader"))
        return signature

    def _get_score_module_from_name(
        self, model: nn.Module, score_module_name: str, quant_module: nn.Module
    ) -> nn.Module:
        """Get the actual score module object from its name.

        Args:
            model: The model containing all modules
            score_module_name: The name of the score module to retrieve
            quant_module: The quantized module for which the score is estimated

        Returns:
            The score module object, or the quantized module itself if the score module is not found
        """
        try:
            score_module = model.get_submodule(score_module_name)
            return score_module
        except AttributeError:
            warnings.warn(
                f"Score module '{score_module_name}' not found. Score will estimated from the quantized module itself."
            )
            return quant_module

    def insert_hparams_after_merge_rules(self, model, quant_recipes, disabled_layers=None):
        """Restrict the search space using the merge rules and insert the hparams for the model."""
        # TRTLLM fuses linear layers such as q_proj, k_proj, v_proj into same layer
        # Hence we need to restrict the search space so that all these layers share the same recipe
        # Lets group the modules based on the rules and insert the same hparam for all the modules in the group

        if disabled_layers is None:
            disabled_layers = []
        elif isinstance(disabled_layers, str):
            disabled_layers = [disabled_layers]

        # Map from group key to list of (quant_module, name, disabled, score_module)
        search_map: dict[str, list[tuple[nn.Module, str, bool, nn.Module]]] = {}

        for name, module in model.named_modules():
            if not self._is_auto_quantize_module(module):
                continue

            # Skip layers that match disabled_layers patterns
            disabled = False
            for pattern in disabled_layers:
                if fnmatch.fnmatch(name, pattern):
                    disabled = True
                    break

            # Apply quant_grouping_rules to determine the group key
            group_key = name  # Default: each module in its own group
            for rule in self._get_quant_grouping_rules():
                result = self._apply_quant_group_rule(name, rule)
                if result is not None:
                    group_key = result
                    # We support only one rule for matching per module
                    break

            # Apply score_module_rules to determine the score module name, then get the actual module
            score_module_name = name  # Default: score from same module
            for rule in self.score_module_rules:
                result = self._apply_score_group_rule(name, rule)
                if result is not None:
                    score_module_name = result
                    # We support only one rule for matching per module
                    break

            # Get the actual score module object immediately
            score_module = self._get_score_module_from_name(model, score_module_name, module)

            if group_key not in search_map:
                search_map[group_key] = [(module, name, disabled, score_module)]
            else:
                search_map[group_key].append((module, name, disabled, score_module))

        for group_key, module_info_list in search_map.items():
            quant_modules = [module for module, _, _, _ in module_info_list]
            disabled = any(disabled for _, _, disabled, _ in module_info_list)
            score_modules = [score_module for _, _, _, score_module in module_info_list]
            quant_module_names = [name for _, name, _, _ in module_info_list]
            cost_weight = self._cost_model.module_cost_weight(
                quant_module_names, self.config["cost"]
            )

            _quant_recipes = None if disabled else quant_recipes
            hparam = QuantRecipeHparam(
                _quant_recipes,
                quant_modules=quant_modules,
                score_modules=score_modules,
                name=str(group_key),
                quant_module_names=quant_module_names,
                cost_weight=cost_weight,
            )

            for module in quant_modules:
                module._register_hparam("quant_recipe", hparam)

    def _get_formatted_weight_compression_constraint(self):
        effective_bits = self.constraints["effective_bits"]
        assert effective_bits > 0 and effective_bits <= 16, (
            "effective_bits should be between 0 and 16."
        )
        weight_compression = self.constraints["effective_bits"] / 16.0

        return weight_compression

    def _verify_constraint(self, search_recipes):
        assert self.constraints["effective_bits"] >= search_recipes[0].num_bits, (
            f"The effective_bits {self.constraints['effective_bits']} constraint cannot be lower than the "
            f"num_bits of most aggressive quantization format for this search which is "
            f"{search_recipes[0]} whose num_bits = {search_recipes[0].num_bits}."
        )

    @abstractmethod
    def estimate_sensitivity_scores(self) -> None:
        """Estimate sensitivity scores and track them with Hparam."""

    def initialize_candidate_stats(self):
        """Initialize the candidate stats for the model."""
        for name, hparam in named_hparams(self.model, unique=True):
            if not isinstance(hparam, QuantRecipeHparam):
                continue

            formats, scores, costs = [], [], []
            element_costs = []
            prev_score = float("inf")
            for recipe in hparam.choices:
                formats.append(recipe)

                score = hparam.get_score(recipe)  # type: ignore [arg-type]
                cost = hparam.get_cost(recipe)  # type: ignore [arg-type]
                element_cost = hparam.get_cost(recipe, cost_weight=1.0)  # type: ignore [arg-type]

                score = min(score, prev_score)  # TODO: Should we get rid of this?
                scores.append(score)
                costs.append(cost)
                element_costs.append(element_cost)
                prev_score = score

            stats = self.candidate_stats.setdefault(name, {})
            stats["formats"] = formats
            stats["scores"] = scores
            stats["costs"] = costs
            stats["element_costs"] = element_costs
            stats["module_names"] = hparam.quant_module_names
            stats["cost_weight"] = hparam.cost_weight
            for component_key in (name, str(hparam.name), f"{hparam.name}.quant_recipe"):
                score_components = self._score_component_records_by_hparam.get(component_key)
                if score_components:
                    stats["score_components"] = score_components
                    break

    def _run_func(self, func, num_iters=1, desc="", data_loader=None):
        if data_loader is None:
            data_loader = self.config["data_loader"]
        try:
            for i, data in tqdm(
                zip(range(num_iters), data_loader),
                desc=desc,
                total=num_iters,
            ):
                self._score_component_current_batch = self._score_component_batch_metadata(i, data)
                func(self.model, data)
        finally:
            self._score_component_current_batch = None

    def _score_component_tracking_enabled(self) -> bool:
        return (
            getattr(
                self,
                "score_component_tracking",
                AUTO_QUANTIZE_SCORE_COMPONENT_TRACKING_NONE,
            )
            != AUTO_QUANTIZE_SCORE_COMPONENT_TRACKING_NONE
        )

    @staticmethod
    def _score_component_scalar(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            if value.ndim == 1 and value.numel() <= 16:
                return value.detach().cpu().tolist()
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)) and len(value) <= 16:
            values = []
            for item in value:
                if isinstance(item, (str, int, float, bool)):
                    values.append(item)
                elif isinstance(item, torch.Tensor) and item.numel() == 1:
                    values.append(item.detach().cpu().item())
                else:
                    return None
            return values
        return None

    def _score_component_batch_metadata(self, batch_index: int, data: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "batch_index": batch_index,
            "component_id": f"batch:{batch_index}",
        }
        if isinstance(data, dict):
            for key in _AUTO_QUANTIZE_SCORE_COMPONENT_SOURCE_KEYS:
                if key not in data:
                    continue
                value = self._score_component_scalar(data[key])
                if value is None:
                    continue
                metadata["source_id"] = value
                metadata["component_id"] = f"{key}:{value}"
                break
        return metadata

    def _score_component_module_name(self, module: nn.Module) -> str:
        module_names = getattr(self, "_score_component_module_names", None)
        if module_names is None:
            module_names = {module: name for name, module in self.model.named_modules()}
            self._score_component_module_names = module_names
        return module_names.get(module, "")

    def _record_score_component(
        self,
        hparam: QuantRecipeHparam,
        recipe: QuantRecipe,
        score_module: nn.Module,
        score: torch.Tensor,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._score_component_tracking_enabled():
            return

        batch_metadata = getattr(self, "_score_component_current_batch", None) or {}
        record = {
            **batch_metadata,
            **(extra_metadata or {}),
            "hparam": hparam.name,
            "format": _auto_quantize_recipe_name(recipe),
            "score_module_name": self._score_component_module_name(score_module),
            "score": float(score.detach().float().cpu().item()),
        }
        self._score_component_records_by_hparam[str(hparam.name)].append(record)

    def before_search(self):
        """Prepare the model for search by calibrating the quantizers  and collecting ``AutoQuantize`` score."""
        # Import here to avoid circular import
        from modelopt.torch.quantization.model_quant import calibrate

        from .conversion import restore_quantizer_state, update_quantize_metadata
        from .utils import get_quantizer_state_dict, set_quantizer_state_dict

        super().before_search()
        self.constraints = normalize_auto_quantize_constraints(self.model, self.constraints)
        self.config["cost_model"] = self.constraints["cost_model"]
        self.config["cost"] = self.constraints.get("cost", {})
        self.config["active_moe_expert_ratio"] = self.config["cost"].get(
            ACTIVE_MOE_EXPERT_RATIO_KEY
        )
        cost_model = get_auto_quantize_cost_model(self.config["cost_model"])
        restored_method = getattr(self, "method", None)
        if self.candidate_stats and restored_method not in (None, self.method_name):
            raise ValueError(
                f"Checkpoint method '{restored_method}' does not match current method "
                f"'{self.method_name}'. Use a different checkpoint path."
            )
        restored_cost_model = getattr(self, "cost_model", "weight")
        restored_active_moe_expert_ratio = getattr(self, "active_moe_expert_ratio", None)
        restored_quant_grouping_scheme = getattr(
            self,
            "quant_grouping_scheme",
            AUTO_QUANTIZE_GROUPING_SCHEME_RUNTIME_FUSED,
        )
        if self.candidate_stats and (
            restored_cost_model != self.config["cost_model"]
            or restored_active_moe_expert_ratio != self.config["active_moe_expert_ratio"]
        ):
            raise ValueError(
                "Checkpoint AutoQuantize cost model does not match current search config: "
                f"checkpoint=({restored_cost_model}, {restored_active_moe_expert_ratio}), "
                f"current=({self.config['cost_model']}, {self.config['active_moe_expert_ratio']}). "
                "Use a different checkpoint path."
            )
        if self.candidate_stats and (
            restored_quant_grouping_scheme != self.config["quant_grouping_scheme"]
        ):
            raise ValueError(
                "Checkpoint AutoQuantize grouping scheme does not match current search config: "
                f"checkpoint={restored_quant_grouping_scheme}, "
                f"current={self.config['quant_grouping_scheme']}. "
                "Use a different checkpoint path."
            )
        self.method = self.method_name
        self.cost_model = self.config["cost_model"]
        self.cost = self.config["cost"]
        self.active_moe_expert_ratio = self.config["active_moe_expert_ratio"]
        self.disabled_layers = self.config["disabled_layers"]
        self.quant_grouping_scheme = self.config["quant_grouping_scheme"]
        self.score_component_tracking = self.config["score_component_tracking"]
        self.hidden_recon_score_windows = self.config["hidden_recon_score_windows"]
        self.hidden_recon_score_reduce = self.config["hidden_recon_score_reduce"]
        self.score_component_metadata = {
            "schema_version": 1,
            "mode": self.score_component_tracking,
            "source_keys": list(_AUTO_QUANTIZE_SCORE_COMPONENT_SOURCE_KEYS),
            "hidden_recon_score_windows": self.hidden_recon_score_windows,
            "hidden_recon_score_reduce": self.hidden_recon_score_reduce,
        }
        self._score_component_records_by_hparam = defaultdict(list)
        self._score_component_current_batch = None
        self._score_component_module_names = None
        self.cost_denominator = getattr(self, "cost_denominator", None)

        search_recipes = self._get_search_recipes(self.config["quantization_formats"])
        self._verify_constraint(search_recipes)
        self._cost_model = cost_model
        current_scoring_signature = self._build_scoring_signature(search_recipes)
        restored_scoring_signature = self._normalize_scoring_signature(self.scoring_signature)
        compare_scoring_signature = self._normalize_scoring_signature(current_scoring_signature)
        if self.candidate_stats and restored_scoring_signature != compare_scoring_signature:
            diff = self._signature_diff(self.scoring_signature, current_scoring_signature)
            raise ValueError(
                "Checkpoint AutoQuantize scoring signature does not match current search "
                f"config ({diff}). Use a different checkpoint path or rerun scoring from scratch."
            )
        self.scoring_signature = current_scoring_signature
        self.insert_hparams_after_merge_rules(
            self.model, search_recipes, self.config["disabled_layers"]
        )

        QuantRecipe.disable_folding_pqs_to_weights()

        # Iterate over the search recipes and calibrate the quantizers for each recipe
        calibrated_new = False
        for recipe in search_recipes:
            if recipe == QuantRecipe(quant_cfg=None):  # No-quant format
                continue

            for name, hparam in named_hparams(self.model, configurable=True):
                if not isinstance(hparam, QuantRecipeHparam):
                    continue
                hparam.active = recipe

            if recipe in self.quantizer_states:
                saved = self.quantizer_states[recipe]
                # config is unused by restore_quantizer_state
                restore_quantizer_state(
                    self.model, QuantizeConfig(), {"quantizer_state": saved["metadata"]}
                )
                set_quantizer_state_dict(self.model, saved["state_dict"])
                if self.config["verbose"]:
                    print_rank_0(f"AutoQuantize: Restored calibration for {recipe}")
                continue

            # Lets reduce the number of calibration steps for AWQ since it takes longer
            num_calib_steps = (
                self.config["num_calib_steps"]
                if "awq" not in str(recipe.config.algorithm)
                else max(1, self.config["num_calib_steps"] // 4)
            )

            def forward_loop(model):
                self._run_func(
                    self.config["forward_step"],
                    num_iters=num_calib_steps,
                    desc=f"Calibrating for {recipe}",
                )

            calibrate(
                self.model,
                algorithm=recipe.config.algorithm,
                forward_loop=forward_loop,
            )
            # Calibrate adds a new mode to the model. Since auto_quantize mixes the quantization recipes
            # across layers, lets not save this new mode in the modelopt state.
            # TODO: This is a hack. We need to create a mode for auto_quantize to handle this in a clean way.
            ModeloptStateManager(self.model).state_dict().pop()
            metadata: dict = {}
            # config is unused by update_quantize_metadata
            update_quantize_metadata(self.model, QuantizeConfig(), metadata)
            self.quantizer_states[recipe] = {
                "metadata": metadata["quantizer_state"],
                "state_dict": get_quantizer_state_dict(self.model),
            }
            calibrated_new = True

        if calibrated_new:
            self.save_search_checkpoint(verbose=self.config["verbose"])

        if self.candidate_stats:
            if self.config["verbose"]:
                print_rank_0("AutoQuantize: Restored from checkpoint, skipping scoring")
            return

        self.estimate_sensitivity_scores()
        self.initialize_candidate_stats()
        self.save_search_checkpoint(verbose=self.config["verbose"])

    @staticmethod
    def _print_recipe_summary(best_recipe, total_cost, total_weight_size, prefix="AutoQuantize"):
        for name, recipe in best_recipe.items():
            print_rank_0(f"{prefix} best recipe for {name.replace('.quant_recipe', '')}: {recipe}")
        effective_bits = (total_cost / total_weight_size) * 16
        print_rank_0(f"{prefix} effective bits: {effective_bits:.2f}")
        return effective_bits

    @staticmethod
    def _get_total_weight_size(modules):
        return sum(
            _get_module_weight_numel(module)
            if _AutoQuantizeBaseSearcher._is_auto_quantize_module(module)
            else 0
            for module in modules
        )

    def _get_constraints_for_search(self, max_weight_size, lower_bound=None):
        constraints = {
            "weight_size_after_compression": (
                lower_bound * max_weight_size if lower_bound else lower_bound,
                max_weight_size,
            )
        }
        return constraints, "weight_size_after_compression"

    def _get_search_lower_bounds(self):
        constraints = getattr(self, "constraints", {})
        if constraints.get("cost_lower_bound") is not None:
            return [constraints["cost_lower_bound"]]
        cost_model = getattr(self, "cost_model", getattr(self, "config", {}).get("cost_model"))
        if cost_model == COST_MODEL_ACTIVE_MOE:
            return [0.99, 0.90, None]
        return [None, 0.99, 0.90]

    @abstractmethod
    def run_search_with_stats(self, max_weight_size, verbose=False):
        """Run the search with stats to get the best recipe and whether the constraints are satisfied."""

    def _get_response_risk_signal(self) -> dict[tuple[str, str, str, str], float]:
        """Return cached response-risk penalties for this searcher."""
        response_risk = getattr(self, "constraints", {}).get("response_risk")
        if not response_risk:
            return {}

        cache_key = repr(sorted(response_risk.items(), key=lambda item: item[0]))
        if getattr(self, "_response_risk_cache_key", None) != cache_key:
            self._response_risk_signal_cache = _load_response_risk_signal(response_risk)
            self._response_risk_cache_key = cache_key
        return self._response_risk_signal_cache

    def _get_candidate_rerank_signal(self) -> dict[str, float]:
        """Return cached packet-level rerank adjustments keyed by packet signature."""
        candidate_rerank = getattr(self, "constraints", {}).get("candidate_rerank")
        if not candidate_rerank or not candidate_rerank.get("enabled", False):
            return {}

        cache_key = repr(sorted(candidate_rerank.items(), key=lambda item: item[0]))
        if getattr(self, "_candidate_rerank_cache_key", None) != cache_key:
            self._candidate_rerank_signal_cache = _load_candidate_rerank_signal(candidate_rerank)
            self._candidate_rerank_cache_key = cache_key
        return self._candidate_rerank_signal_cache

    def _get_candidate_family_rerank_rules(self) -> list[dict[str, Any]]:
        """Return cached packet-level family allocation rerank rules."""
        candidate_rerank = getattr(self, "constraints", {}).get("candidate_rerank")
        if not candidate_rerank or not candidate_rerank.get("enabled", False):
            return []

        cache_key = repr(sorted(candidate_rerank.items(), key=lambda item: item[0]))
        if getattr(self, "_candidate_family_rerank_cache_key", None) != cache_key:
            self._candidate_family_rerank_rules_cache = _load_candidate_family_rerank_rules(
                candidate_rerank
            )
            self._candidate_family_rerank_cache_key = cache_key
        return self._candidate_family_rerank_rules_cache

    @staticmethod
    def _candidate_family_rerank_score(
        packet: dict[str, Any], rules: list[dict[str, Any]]
    ) -> float:
        """Score packet-level family allocation rules against packet metadata."""
        if not rules:
            return 0.0
        total_score = 0.0
        for rule in rules:
            count_level = rule.get("count_level", "module")
            if count_level == "hparam":
                family_counts = packet.get("hparam_family_format_counts") or {}
            elif count_level == "module":
                family_counts = packet.get("family_format_counts") or {}
            else:
                raise ValueError(
                    f"AutoQuantize candidate_rerank count_level must be 'module' or 'hparam', got {count_level!r}."
                )
            counts = family_counts.get(rule["family"], {})
            if not counts:
                continue
            if rule["format"] == "*":
                count = sum(int(value) for value in counts.values())
            else:
                count = int(counts.get(rule["format"], 0) or 0)
            total = sum(int(value) for value in counts.values())
            frac = count / total if total else 0.0

            min_count = rule.get("min_count")
            max_count = rule.get("max_count")
            min_frac = rule.get("min_frac")
            max_frac = rule.get("max_frac")
            if min_count not in {None, ""} and count < float(str(min_count)):
                continue
            if max_count not in {None, ""} and count > float(str(max_count)):
                continue
            if min_frac not in {None, ""} and frac < float(str(min_frac)):
                continue
            if max_frac not in {None, ""} and frac > float(str(max_frac)):
                continue

            if rule["mode"] == "per_count":
                total_score += rule["score"] * count
            elif rule["mode"] == "per_fraction":
                total_score += rule["score"] * frac
            else:
                total_score += rule["score"]
        return total_score

    def _rerank_candidate_packets(self, packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidate_rerank = getattr(self, "constraints", {}).get("candidate_rerank")
        if not packets or not candidate_rerank or not candidate_rerank.get("enabled", False):
            return packets

        rerank_signal = self._get_candidate_rerank_signal()
        family_rules = self._get_candidate_family_rerank_rules()

        reranked_packets: list[dict[str, Any]] = []
        for packet in packets:
            signature_score = rerank_signal.get(str(packet.get("signature") or ""), 0.0)
            family_score = self._candidate_family_rerank_score(packet, family_rules)
            rerank_score = signature_score + family_score
            reranked_packets.append(
                {
                    **packet,
                    "lp_packet_id": packet["packet_id"],
                    "rerank_score": rerank_score,
                    "rerank_signature_score": signature_score,
                    "rerank_family_score": family_score,
                    "rerank_objective_scores": packet["objective_scores"] + rerank_score,
                    "rerank_source_match": signature_score != 0.0,
                    "rerank_family_match": family_score != 0.0,
                }
            )

        if rerank_signal or family_rules:
            reranked_packets.sort(
                key=lambda packet: (
                    packet["rerank_objective_scores"],
                    packet["objective_scores"],
                    packet["lp_packet_id"],
                )
            )
        for rerank_rank, packet in enumerate(reranked_packets):
            packet["rerank_rank"] = rerank_rank
        return reranked_packets

    def _candidate_scores_for_search(
        self, candidate_stat: dict[str, list[float]], hparam_name: str = ""
    ) -> list[float]:
        constraints = getattr(self, "constraints", {})
        score_model = constraints.get("score_model", AUTO_QUANTIZE_SCORE_MODEL_RAW)
        if score_model == AUTO_QUANTIZE_SCORE_MODEL_RAW:
            scores = candidate_stat["scores"]
        elif score_model == AUTO_QUANTIZE_SCORE_MODEL_PER_ELEMENT:
            element_costs = candidate_stat.get("element_costs")
            if not element_costs:
                cost_weight = candidate_stat.get("cost_weight", 1.0)
                element_costs = [
                    cost / cost_weight if cost_weight > 0 else cost
                    for cost in candidate_stat["costs"]
                ]
            scores = [
                score / cost if cost > 0 else score
                for score, cost in zip(candidate_stat["scores"], element_costs)
            ]
        elif score_model == AUTO_QUANTIZE_SCORE_MODEL_PER_ACTIVE:
            scores = [
                score / cost if cost > 0 else score
                for score, cost in zip(candidate_stat["scores"], candidate_stat["costs"])
            ]
        elif score_model == AUTO_QUANTIZE_SCORE_MODEL_ACTIVE_WEIGHTED:
            cost_weight = candidate_stat.get("cost_weight")
            if cost_weight is None:
                element_costs = candidate_stat.get("element_costs")
                if element_costs:
                    ratios = [
                        cost / element_cost
                        for cost, element_cost in zip(candidate_stat["costs"], element_costs)
                        if element_cost > 0
                    ]
                    cost_weight = max(ratios) if ratios else 1.0
                else:
                    cost_weight = 1.0
            scores = [score * cost_weight for score in candidate_stat["scores"]]
        else:
            raise ValueError(f"Unsupported AutoQuantize score_model: {score_model}")

        response_risk_signal = self._get_response_risk_signal()
        if not response_risk_signal:
            return scores

        return [
            score
            + _response_risk_penalty(
                hparam_name,
                candidate_stat,
                _auto_quantize_recipe_name(fmt),
                response_risk_signal,
            )
            for score, fmt in zip(scores, candidate_stat["formats"])
        ]

    def run_search(self):
        """Search for the best per-layer quantization configuration and return the best model and configuration."""
        verbose = self.config["verbose"]
        assert "effective_bits" in self.constraints and (
            set(self.constraints) <= AUTO_QUANTIZE_CONSTRAINT_KEYS
        ), (
            "`constraints` must contain 'effective_bits' and may contain 'cost_model', "
            "'cost', 'cost_lower_bound', 'score_model', 'response_risk', and "
            "'candidate_rerank'. "
            f"Got {self.constraints.keys()}."
        )

        compression = self._get_formatted_weight_compression_constraint()
        total_weight_size = self._cost_model.total_weight_size(
            self.model.named_modules(), self._is_auto_quantize_module, self.config["cost"]
        )
        self.cost_denominator = total_weight_size
        max_weight_size = total_weight_size * compression
        if verbose:
            print_rank_0(
                "AutoQuantize cost model: "
                f"{self.config['cost_model']}"
                + (
                    f" (active_moe_expert_ratio={self.config['active_moe_expert_ratio']})"
                    if self.config["cost_model"] == COST_MODEL_ACTIVE_MOE
                    else ""
                )
            )

        # Run the search with stats to get the best recipe and whether the constraints are satisfied
        best_recipe_info, is_satisfied = self.run_search_with_stats(max_weight_size, verbose)
        self.best["is_satisfied"] = is_satisfied
        if getattr(self, "last_candidate_packets", None):
            candidate_rerank = self.constraints.get("candidate_rerank") or {}
            if isinstance(candidate_rerank, dict) and candidate_rerank.get("enabled", False):
                launch_authority = candidate_rerank.get("launch_authority_default", "no")
                self.best["candidate_packets"] = [
                    {**packet, "launch_authority": launch_authority}
                    for packet in self.last_candidate_packets
                ]

        best_recipe = {}
        best_constraints, best_scores = 0, 0
        for name, best_hparam_recipe_info in best_recipe_info.items():
            # Solvers could give different solutions for the same layer across DP/TP groups even though
            # the scores and costs are the same. Lets make sure the same recipe is selected across DP/TP
            _ps = self.model.get_submodule(name.split(".quant_recipe")[0]).parallel_state
            best_format = DistributedProcessGroup.get_dist_syncd_obj(
                best_hparam_recipe_info["format"],
                [_ps.data_parallel_group, _ps.tensor_parallel_group],
                lambda a: a[0],
            )

            best_recipe[name] = best_format
            get_hparam(self.model, name).active = best_format
            best_constraints += best_hparam_recipe_info["costs"]
            best_scores += best_hparam_recipe_info["scores"]

        if verbose:
            effective_bits_from_search = self._print_recipe_summary(
                best_recipe, best_constraints, total_weight_size
            )
        else:
            effective_bits_from_search = (best_constraints / total_weight_size) * 16

        self.best["recipe"] = best_recipe
        self.best["constraints"] = {"effective_bits": effective_bits_from_search}
        self.best["score"] = best_scores
        response_risk = self.constraints.get("response_risk")
        if isinstance(response_risk, dict):
            self.best["response_risk_source"] = {
                "source_path": response_risk.get("source_path"),
                "scale": response_risk.get("scale", 1.0),
                "risk_metric": response_risk.get("risk_metric"),
                "provenance": response_risk.get("provenance"),
                "loaded_signal_entries": len(self._get_response_risk_signal()),
            }
        candidate_rerank = self.constraints.get("candidate_rerank")
        if isinstance(candidate_rerank, dict):
            self.best["candidate_rerank_source"] = {
                "source_path": candidate_rerank.get("source_path"),
                "scale": candidate_rerank.get("scale", 1.0),
                "id_field": candidate_rerank.get("id_field", "signature"),
                "score_field": candidate_rerank.get("score_field", "rerank_score"),
                "provenance": candidate_rerank.get("provenance"),
                "loaded_signal_entries": len(self._get_candidate_rerank_signal()),
            }

        self.save_search_checkpoint(verbose=verbose)

        QuantRecipe.fold_pqs_to_weights(self.model)


_AUTO_QUANTIZE_SCORE_CHUNK_SIZE = 16 * 1024 * 1024


def _get_auto_quantize_score(grad_output, output_diff):
    grad_output_flat = grad_output.reshape(-1)
    output_diff_flat = output_diff.reshape(-1)
    if grad_output_flat.numel() != output_diff_flat.numel():
        raise ValueError(
            "`grad_output` and `output_diff` must have the same number of elements, "
            f"got {grad_output_flat.numel()} and {output_diff_flat.numel()}."
        )

    score = torch.zeros((), dtype=torch.float32, device=grad_output.device)
    for start in range(0, grad_output_flat.numel(), _AUTO_QUANTIZE_SCORE_CHUNK_SIZE):
        end = min(start + _AUTO_QUANTIZE_SCORE_CHUNK_SIZE, grad_output_flat.numel())
        x = grad_output_flat[start:end].float() * output_diff_flat[start:end].float()
        score.add_(x.clamp(-1e10, 1e10).square().sum())
    return score


def _add_auto_quantize_score(grad_output, output_diff, score_tensor):
    score_tensor += _get_auto_quantize_score(grad_output, output_diff)


def _get_primary_output_tensor(output: Any) -> torch.Tensor:
    """Return the primary tensor from a module output for reconstruction scoring."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(
        "AutoQuantize hidden_recon scoring expects score-module outputs to be "
        f"Tensors or tuple/list with a Tensor first element, got {type(output)!r}."
    )


def _normalize_hidden_recon_score_windows(windows: Any) -> list[str]:
    """Normalize and validate hidden-reconstruction score windows."""
    if windows is None or windows == "":
        windows = ["full"]
    elif isinstance(windows, str):
        windows = [window.strip() for window in windows.split(",") if window.strip()]
    else:
        windows = [str(window).strip() for window in windows if str(window).strip()]
    if not windows:
        windows = ["full"]

    normalized = []
    for window in windows:
        if window == "full":
            normalized.append(window)
            continue
        match = re.fullmatch(r"(?:last|tail):?(\d+)", window)
        if match is None:
            raise ValueError(
                "hidden_recon_score_windows entries must be 'full', 'last:N', "
                f"'tail:N', 'lastN', or 'tailN', got {window!r}."
            )
        count = int(match.group(1))
        if count <= 0:
            raise ValueError(f"hidden_recon score window must be positive, got {window!r}.")
        normalized.append(f"last:{count}")
    return normalized


def _slice_hidden_recon_score_window(tensor: torch.Tensor, window: str) -> torch.Tensor:
    if window == "full" or tensor.ndim < 2:
        return tensor
    count = int(window.split(":", 1)[1])
    seq_dim = -2
    if tensor.shape[seq_dim] <= count:
        return tensor
    return tensor.narrow(seq_dim, tensor.shape[seq_dim] - count, count)


def _get_hidden_recon_score_from_tensors(
    reference: torch.Tensor, quantized: torch.Tensor
) -> torch.Tensor:
    if reference.shape != quantized.shape:
        raise ValueError(
            "Reference and quantized score-module outputs must have the same shape, "
            f"got {tuple(reference.shape)} and {tuple(quantized.shape)}."
        )
    if reference.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=reference.device)

    reference = reference.float()
    quantized = quantized.float()
    mse = (quantized - reference).square().mean()
    denom = reference.square().mean().clamp_min(1e-12)
    return mse / denom


def _get_hidden_recon_score_components(
    reference_output: Any, quantized_output: Any, windows: Sequence[str] | None = None
) -> list[tuple[str, torch.Tensor]]:
    """Return normalized parent hidden-state reconstruction errors per score window."""
    reference = _get_primary_output_tensor(reference_output)
    quantized = _get_primary_output_tensor(quantized_output)
    normalized_windows = _normalize_hidden_recon_score_windows(windows or ["full"])
    return [
        (
            window,
            _get_hidden_recon_score_from_tensors(
                _slice_hidden_recon_score_window(reference, window),
                _slice_hidden_recon_score_window(quantized, window),
            ),
        )
        for window in normalized_windows
    ]


def _reduce_hidden_recon_scores(
    window_scores: Sequence[tuple[str, torch.Tensor]], reducer: str
) -> torch.Tensor:
    scores = [score for _, score in window_scores]
    if not scores:
        return torch.zeros((), dtype=torch.float32)
    if reducer == AUTO_QUANTIZE_HIDDEN_RECON_SCORE_REDUCE_MEAN:
        return torch.stack(scores).mean()
    if reducer == AUTO_QUANTIZE_HIDDEN_RECON_SCORE_REDUCE_MAX:
        return torch.stack(scores).max()
    raise ValueError(
        f"Unsupported hidden_recon_score_reduce={reducer!r}; "
        f"expected one of {sorted(AUTO_QUANTIZE_HIDDEN_RECON_SCORE_REDUCERS)}."
    )


def _get_hidden_recon_score(reference_output: Any, quantized_output: Any) -> torch.Tensor:
    """Return normalized parent hidden-state reconstruction error."""
    return _reduce_hidden_recon_scores(
        _get_hidden_recon_score_components(reference_output, quantized_output, ["full"]),
        AUTO_QUANTIZE_HIDDEN_RECON_SCORE_REDUCE_MEAN,
    )


def _config_like_objects(model: nn.Module) -> list[Any]:
    """Return unique config-like objects that may carry generation cache defaults."""
    configs = []
    seen = set()
    for module in model.modules():
        for attr in ("config", "generation_config"):
            config = getattr(module, attr, None)
            if config is None or id(config) in seen:
                continue
            seen.add(id(config))
            configs.append(config)
            for nested_attr in ("text_config", "language_config"):
                nested_config = getattr(config, nested_attr, None)
                if nested_config is not None and id(nested_config) not in seen:
                    seen.add(id(nested_config))
                    configs.append(nested_config)
    return configs


def _set_model_use_cache(model: nn.Module, use_cache: bool) -> list[tuple[Any, Any]]:
    """Set ``use_cache`` on model configs and return original values for restore."""
    originals = []
    for config in _config_like_objects(model):
        if hasattr(config, "use_cache"):
            originals.append((config, getattr(config, "use_cache")))
            setattr(config, "use_cache", use_cache)
    return originals


def _restore_model_use_cache(originals: list[tuple[Any, Any]]) -> None:
    for config, use_cache in originals:
        setattr(config, "use_cache", use_cache)


class AutoQuantizeGradientSearcher(_AutoQuantizeBaseSearcher):
    """A searcher for AutoQuantize algorithm that uses gradient based score estimation.

    In AutoQuantize, we search for the best per-layer quantization configuration that minimizes the sum of per-layer
    scores while meeting the specified constraint. AutoQuantize uses Linear Programming Solver to find the
    optimal quantization configuration.

    The auto_quantize score for a layer quantization configuration is an approximation of model loss change due
    to quantizing the particular layer with the particular configuration.
    The approximation is based on taylor expansion of the loss function wrt to the quantized output of the layer and
    substitution of Fisher information for Hessian.
    This approximation is mathematically correct for models where the loss
    is a log likelihood loss such as BERT, GPT, etc. However, the auto_quantize score can still be used as a proxy
    for other models such as ResNet.

    **Quant Modules:**

    This searcher operates on quantizable modules (quant modules), which are typically Linear or Conv layers
    that support quantization. Optionally, grouping rules can be applied to ensure certain layers share the same
    quantization format (e.g., Q, K, V projections in the same attention layer). For details on quant_grouping_rules
    and customization, see the :meth:`auto_quantize <modelopt.torch.quantization.model_quant.auto_quantize>`
    API documentation.

    **Score Modules:**

    By default, for each quant module, its sensitivity score is estimated using that module's output perturbation.
    However, the sensitivity can also be estimated by looking at perturbation at a separate point in the neural
    network (score module). This is helpful in some cases such as MoEs for speed and lower memory consumption.
    Since all experts are already restricted to the same quant format by quant grouping rules, their sensitivity
    can be estimated together at a single point (e.g., the MLP output level).
    """

    method_name = "gradient"

    score_module_rules = [
        # Score attention projection recipes at the whole attention module output so
        # downstream norm/position/gating/state transitions are reflected in the
        # sensitivity estimate. This changes scoring only; quant grouping still
        # controls which modules must share a recipe for deployment compatibility.
        _self_attn_score_module,
        _linear_attn_score_module,
        # HF fused routed experts are one deployment-compatible bank-level
        # hparam. Score them at parent MLP output, matching the unfused/shared
        # expert rules so router/shared-expert combination is included.
        _fused_routed_experts_score_module,
        # Use MLP layer output for gate_proj, up_proj, down_proj for Qwen3 like MoE models (local and shared experts)
        r"^(.*?\.mlp)\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",
        r"^((?:.*\.)?mlp)\.shared_expert\.(gate_proj|up_proj|down_proj)$",
        r"^(.*?\.mixer)\.experts\.\d+\.(up_proj|down_proj)$",  # NemotronH MoE experts
        r"^(.*?)\.(\d+\.(w1|w2|w3))$",  # mixtral experts
        r"^(.*?)\.((w1_linear|w2_linear|w3_linear)\.\d+)$",  # dbrx experts
    ]

    # See `register_custom_support` for details
    _custom_support: list[tuple[Callable, Callable, Callable]] = []

    @property
    def default_search_config(self):
        """Get the default config for the searcher."""
        config = super().default_search_config
        config.update(
            {
                "forward_step": None,
                "loss_func": None,
                "forward_backward_step": None,
            }
        )
        return config

    def sanitize_search_config(self, config: SearchConfig | None) -> SearchConfig:
        """Sanitize the search config dict."""
        config = config or {}
        if "score_func" in config:
            warnings.warn("`score_func` is ignored for gradient based `auto_quantize`.")
            config.pop("score_func")
        config = super().sanitize_search_config(config)
        if config["forward_backward_step"] is None:
            assert config["loss_func"] is not None, (
                "`loss_func` or `forward_backward_step` must be provided for `auto_quantize`."
            )
            config["forward_backward_step"] = self._get_default_forward_backward_step()

        return config

    @classmethod
    def register_custom_support(
        cls,
        is_supported_checker: Callable,
        grad_ckpt_context: Callable,
        is_param_grad_enabled: Callable,
    ) -> None:
        """(Optional) Register custom support for `AutoQuantize` score estimation.

        This custom support is used to enable memory/compute efficient backward gradient propagation. This involves:

        - `grad_ckpt_context`: backward pass with gradient checkpointing enabled
        - `is_param_grad_enabled`: AutoQuantize only needs activation gradients to be computed (not weight
          gradients). `is_param_grad_enabled` is used to select which parameters should have gradients enabled,
          limiting gradient computation to only what's needed for activation gradients. For LLMs, to trigger all
          activation gradient computation, just enabling the embedding layer weight gradient is sufficient. This will
          enable gradient computation for all the activation gradients downstream.

        If the `is_supported_checker(model)` returns True, the `grad_ckpt_context(model)` will be
        used to enable gradient checkpointing and `is_param_grad_enabled(pname, model)`
        will be used to select which parameters have gradients enabled to minimize gradient computation.
        """
        cls._custom_support.append((is_supported_checker, grad_ckpt_context, is_param_grad_enabled))

    def _get_default_forward_backward_step(self):
        def forward_backward_step(model, data):
            output = self.config["forward_step"](model, data)
            loss = self.config["loss_func"](output, data)
            try:
                loss.backward()
            except RuntimeError as e:
                raise RuntimeError(
                    "AutoQuantize: Error while calling `backward()` on the loss returned by `loss_func`. "
                    "Please fix this!"
                    f"error: {e}"
                ) from e

        return forward_backward_step

    @torch.enable_grad()
    def _estimate_auto_quantize_scores(self, is_param_grad_enabled):
        # TODO: remove the no-quant recipe
        def auto_quantize_score_estimate_forward(module, *args, **kwargs):
            for hparam in module._hparams_for_scoring:
                if hparam.is_configurable:
                    hparam.active = QuantRecipe(quant_cfg=None)

            output = module._forward_original(*args, **kwargs)

            # If gradient checkpointing is enabled, gradient will not be enabled in the global forward pass.
            # With gradient checkpointing, gradients are computed in the local forward pass during backward pass

            # Lets compute the output_diff and save it in memory only if gradient is enabled to be memory efficient
            if not torch.is_grad_enabled():
                return output

            module.output_diff_dict = {hparam: {} for hparam in module._hparams_for_scoring}
            with torch.no_grad():
                for hparam in module._hparams_for_scoring:
                    if not hparam.is_configurable:
                        continue
                    for recipe in hparam.choices:
                        if recipe == QuantRecipe(quant_cfg=None):
                            continue
                        hparam.active = recipe
                        output_diff = module._forward_original(*args, **kwargs)

                        if isinstance(output_diff, tuple):
                            output_diff = output_diff[0] - output[0]
                        else:
                            output_diff -= output
                        module.output_diff_dict[hparam][recipe] = output_diff.detach()

                    # Disable the configurable hparam now that we have computed the diff
                    hparam.active = QuantRecipe(quant_cfg=None)

            return output

        def backward_hook(module, grad_input, grad_output):
            for hparam, output_diff_dict in module.output_diff_dict.items():
                for recipe, output_diff in output_diff_dict.items():
                    score = _get_auto_quantize_score(grad_output[0], output_diff)
                    if hparam._importance_dict[recipe][module] is None:
                        hparam._importance_dict[recipe][module] = score
                    else:
                        hparam._importance_dict[recipe][module] += score
                    self._record_score_component(hparam, recipe, module, score)

        def setup_params_for_score_estimation(name, param, params_metadata, enable_grad=True):
            # Let us delete the gradient as soon as they are computed to save memory
            params_metadata[name] = {"requires_grad": param.requires_grad}
            param.requires_grad = enable_grad
            if not enable_grad:
                return
            if self.config.get("verbose", False):
                print_rank_0(f"AutoQuantize: Enabling gradient for param {name}.")
            accum_grad, handle = create_param_grad_clear_hook(param)
            params_metadata[name]["accum_grad"] = accum_grad  # We need to keep the accum_grad alive
            params_metadata[name]["handle"] = handle

        def setup_module_for_score_estimation(module):
            module._forward_original = module.forward
            module.forward = types.MethodType(auto_quantize_score_estimate_forward, module)
            module._backward_hook_handle = module.register_full_backward_hook(backward_hook)

        def cleanup_module_after_score_estimation(module):
            module.forward = module._forward_original
            del module._forward_original

            module._backward_hook_handle.remove()

        def cleanup_params_after_score_estimation(name, param, params_metadata):
            param.requires_grad = params_metadata[name]["requires_grad"]
            handle = params_metadata[name].get("handle", None)
            if handle is not None:
                handle.remove()

        score_modules = set()
        for name, module in self.model.named_modules():
            if (
                hasattr(module, "_hparams_for_scoring")
                and any(hparam.is_configurable for hparam in module._hparams_for_scoring)
                and module not in score_modules
            ):
                # Monkey patch the forward methods to cache (Q(Y) - Y)
                setup_module_for_score_estimation(module)
                score_modules.add(module)

        params_metadata = {}
        for name, param in self.model.named_parameters():
            setup_params_for_score_estimation(
                name, param, params_metadata, is_param_grad_enabled(name, self.model)
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            report_memory("AutoQuantize: starting score estimation, ")

        self._run_func(
            self.config["forward_backward_step"],
            num_iters=self.config["num_score_steps"],
            desc="Estimating auto_quantize scores",
            data_loader=self.config["score_data_loader"],
        )

        if torch.cuda.is_available():
            report_memory("AutoQuantize: After score estimation")

        for module in score_modules:
            cleanup_module_after_score_estimation(module)

        for name, param in self.model.named_parameters():
            cleanup_params_after_score_estimation(name, param, params_metadata)

        # Delete the params_metadata
        del params_metadata
        gc.collect()

    def estimate_sensitivity_scores(self) -> None:
        """Estimate sensitivity scores using hessian approximation."""
        self.model.eval()

        def _default_is_param_grad_enabled(pname, model):
            return True

        grad_checkpointing_ctxt = None
        is_param_grad_enabled = _default_is_param_grad_enabled
        for is_supported_checker, ctxt_candidate, grad_enabled_candidate in self._custom_support:
            if is_supported_checker(self.model):
                grad_checkpointing_ctxt = ctxt_candidate
                is_param_grad_enabled = grad_enabled_candidate
                break

        with grad_checkpointing_ctxt(self.model) if grad_checkpointing_ctxt else nullcontext():
            self._estimate_auto_quantize_scores(is_param_grad_enabled)

    def run_search_with_stats(self, max_weight_size, verbose=False):
        """Linear Programming Solve for gradient based auto_quantize.

        AutoQuantize uses Linear Programming Solver to find the optimal quantization configuration which
        minimizes the sum of per-layer auto_quantize scores while meeting the specified constraint.
        """
        candidate_rerank = getattr(self, "constraints", {}).get("candidate_rerank") or {}
        top_k = candidate_rerank.get("top_k", 1) if candidate_rerank.get("enabled", False) else 1
        packets, is_satisfied = self.run_search_candidate_packets_with_stats(
            max_weight_size, top_k, verbose
        )
        packets = self._rerank_candidate_packets(packets)
        self.last_candidate_packets = packets
        if not packets:
            return {}, is_satisfied
        return packets[0]["recipe_info"], is_satisfied

    def _build_lps(self, max_weight_size, lower_bound, verbose=False):
        constraints, constraint_name = self._get_constraints_for_search(
            max_weight_size, lower_bound
        )
        return LPS(
            name="AutoQuantize",
            constraints=constraints,
            constraints_to_candidate_costs={
                constraint_name: [
                    candidate_stat["costs"] for candidate_stat in self.candidate_stats.values()
                ]
            },
            candidate_scores=[
                self._candidate_scores_for_search(candidate_stat, name)
                for name, candidate_stat in self.candidate_stats.items()
            ],
            objective_type="minimize",
            verbose=verbose,
        )

    def _recipe_info_from_selection(self, selections: list[int]) -> dict[str, dict[str, Any]]:
        best_recipes = {}
        for name, selected_idx in zip(self.candidate_stats.keys(), selections):
            objective_scores = self._candidate_scores_for_search(self.candidate_stats[name], name)
            best_recipes[name] = {
                "format": self.candidate_stats[name]["formats"][selected_idx],
                "costs": self.candidate_stats[name]["costs"][selected_idx],
                "scores": self.candidate_stats[name]["scores"][selected_idx],
                "objective_scores": objective_scores[selected_idx],
            }
        return best_recipes

    def _candidate_family_format_counts(
        self, recipe_info: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, dict[str, int]]]:
        """Summarize selected formats by module family for candidate-packet triage.

        ``family_format_counts`` expands grouped hparams by their covered module
        names, while ``hparam_family_format_counts`` counts the hparam once for
        each family it touches. Both are metadata only: they are meant for
        reject/triage gates, not for final candidate promotion.
        """
        module_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        hparam_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for hparam_name, info in recipe_info.items():
            fmt = _auto_quantize_recipe_name(info["format"])
            candidate_stat = self.candidate_stats.get(hparam_name, {})
            module_names = [str(name) for name in candidate_stat.get("module_names") or []]
            if not module_names:
                module_names = [hparam_name.removesuffix(".quant_recipe")]

            categories = {
                _auto_quantize_module_category(module_name) for module_name in module_names
            }
            categories.discard("")
            if not categories:
                categories.add("other")

            for module_name in module_names:
                category = _auto_quantize_module_category(module_name) or "other"
                module_counts[category][fmt] += 1
            for category in categories:
                hparam_counts[category][fmt] += 1

        return {
            "family_format_counts": {
                category: dict(sorted(counts.items()))
                for category, counts in sorted(module_counts.items())
            },
            "hparam_family_format_counts": {
                category: dict(sorted(counts.items()))
                for category, counts in sorted(hparam_counts.items())
            },
        }

    def _candidate_packet_from_selection(
        self, packet_id: int, selections: list[int], status: str
    ) -> dict[str, Any]:
        recipe_info = self._recipe_info_from_selection(selections)
        total_cost = sum(info["costs"] for info in recipe_info.values())
        total_score = sum(info["scores"] for info in recipe_info.values())
        total_objective_score = sum(info["objective_scores"] for info in recipe_info.values())
        total_weight_size = getattr(self, "cost_denominator", None)
        if total_weight_size is None:
            total_weight_size = sum(
                max(candidate_stat["costs"]) for candidate_stat in self.candidate_stats.values()
            )
        effective_bits = (total_cost / total_weight_size) * 16 if total_weight_size else None
        return {
            "packet_id": packet_id,
            "status": status,
            "selection": {
                name: int(selected_idx)
                for name, selected_idx in zip(self.candidate_stats.keys(), selections)
            },
            "recipe": {name: info["format"] for name, info in recipe_info.items()},
            "recipe_info": recipe_info,
            "signature": _auto_quantize_candidate_signature(recipe_info),
            "costs": total_cost,
            "scores": total_score,
            "objective_scores": total_objective_score,
            "effective_bits": effective_bits,
            **self._candidate_family_format_counts(recipe_info),
        }

    def run_search_candidate_packets_with_stats(
        self, max_weight_size, top_k: int = 1, verbose=False
    ) -> tuple[list[dict[str, Any]], bool]:
        """Enumerate real AutoQuant LP candidate packets without changing scoring.

        The first packet is identical to the normal best recipe. Later packets are
        neighboring feasible LP optima generated by no-good cuts, which lets
        diagnostic workflows inspect automatic alternatives before deciding
        which candidates warrant downstream validation.
        """
        # TODO: Do this only for rank 0 in the respective pipeline group

        solutions: list[tuple[list[int], str]] = []
        for lower_bound in self._get_search_lower_bounds():
            # The LP solver for auto_quantize sometimes fails to find a solution if a lower bound is not
            # specified. I dont know why this happens.
            # As a workaround, lets specify a lower bound for the weight compression if previous
            # search without lower bound fails.
            lps = self._build_lps(max_weight_size, lower_bound, verbose)
            solutions = lps.solve_top_k(top_k)
            self.status = solutions[0][1] if solutions else "Infeasible"
            if self.status == "Optimal":
                break

        if self.status != "Optimal":
            warnings.warn(
                "AutoQuantize FAILED to find a solution! The searched model might not meet all constraints. "
            )
            is_satisfied = False
        else:
            is_satisfied = True

        packets = [
            self._candidate_packet_from_selection(packet_id, selections, status)
            for packet_id, (selections, status) in enumerate(solutions)
            if status == "Optimal"
        ]
        return packets, is_satisfied


class AutoQuantizeHiddenReconSearcher(AutoQuantizeGradientSearcher):
    """AutoQuantize searcher using parent hidden-state reconstruction scores.

    This searcher reuses the gradient searcher's parent score-module rules and
    LPS selector, but it scores each candidate by direct normalized parent
    output reconstruction error instead of weighting the output perturbation by
    downstream loss gradients.
    """

    method_name = "hidden_recon"

    def sanitize_search_config(self, config: SearchConfig | None) -> SearchConfig:
        """Sanitize config and ignore backward-only inputs."""
        config = config or {}
        if "score_func" in config:
            warnings.warn("`score_func` is ignored for hidden_recon based `auto_quantize`.")
            config.pop("score_func")
        for ignored_key in ("loss_func", "forward_backward_step"):
            if config.get(ignored_key) is not None:
                warnings.warn(f"`{ignored_key}` is ignored for hidden_recon based `auto_quantize`.")
            config[ignored_key] = None
        return _AutoQuantizeBaseSearcher.sanitize_search_config(self, config)

    @torch.inference_mode()
    def _estimate_hidden_recon_scores(self):
        def hidden_recon_score_estimate_forward(module, *args, **kwargs):
            hparams = [hparam for hparam in module._hparams_for_scoring if hparam.is_configurable]
            if not hparams:
                return module._forward_original(*args, **kwargs)

            for hparam in hparams:
                hparam.active = QuantRecipe(quant_cfg=None)

            reference_output = module._forward_original(*args, **kwargs)

            for hparam in hparams:
                for recipe in hparam.choices:
                    if recipe == QuantRecipe(quant_cfg=None):
                        continue
                    hparam.active = recipe
                    quantized_output = module._forward_original(*args, **kwargs)
                    window_scores = _get_hidden_recon_score_components(
                        reference_output,
                        quantized_output,
                        self.config["hidden_recon_score_windows"],
                    )
                    score = _reduce_hidden_recon_scores(
                        window_scores, self.config["hidden_recon_score_reduce"]
                    )
                    if hparam._importance_dict[recipe][module] is None:
                        hparam._importance_dict[recipe][module] = score
                    else:
                        hparam._importance_dict[recipe][module] += score
                    for window, window_score in window_scores:
                        self._record_score_component(
                            hparam,
                            recipe,
                            module,
                            window_score,
                            {
                                "score_window": window,
                                "score_reduce": self.config["hidden_recon_score_reduce"],
                            },
                        )
                    if len(window_scores) > 1:
                        self._record_score_component(
                            hparam,
                            recipe,
                            module,
                            score,
                            {
                                "score_window": "aggregate",
                                "score_reduce": self.config["hidden_recon_score_reduce"],
                            },
                        )

                hparam.active = QuantRecipe(quant_cfg=None)

            return reference_output

        def setup_module_for_score_estimation(module):
            module._forward_original = module.forward
            module.forward = types.MethodType(hidden_recon_score_estimate_forward, module)

        def cleanup_module_after_score_estimation(module):
            module.forward = module._forward_original
            del module._forward_original

        score_modules = set()
        for _, module in self.model.named_modules():
            if (
                hasattr(module, "_hparams_for_scoring")
                and any(hparam.is_configurable for hparam in module._hparams_for_scoring)
                and module not in score_modules
            ):
                setup_module_for_score_estimation(module)
                score_modules.add(module)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            report_memory("AutoQuantize hidden_recon: starting score estimation, ")

        use_cache_originals = []
        try:
            # Hidden-reconstruction scoring replays parent attention modules
            # multiple times inside a single model forward. Disable generation
            # caches so mutable HF cache objects do not grow between recipe
            # probes while the attention mask still reflects the original
            # sequence length.
            use_cache_originals = _set_model_use_cache(self.model, False)
            self._run_func(
                self.config["forward_step"],
                num_iters=self.config["num_score_steps"],
                desc="Estimating hidden_recon scores",
                data_loader=self.config["score_data_loader"],
            )
        finally:
            _restore_model_use_cache(use_cache_originals)
            for module in score_modules:
                cleanup_module_after_score_estimation(module)

        if torch.cuda.is_available():
            report_memory("AutoQuantize hidden_recon: After score estimation")

        gc.collect()

    def estimate_sensitivity_scores(self) -> None:
        """Estimate sensitivity scores from normalized parent reconstruction error."""
        self.model.eval()
        self._estimate_hidden_recon_scores()


@torch.compile(dynamic=True)
def _get_log_softmax_dist(logits: torch.Tensor, tp_group) -> torch.Tensor:
    dtype = logits.dtype
    max_logits = torch.amax(logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(max_logits, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    logits = (logits - max_logits).float()
    sum_exp_logits = torch.exp(torch.logsumexp(logits, dim=-1, keepdim=True))
    torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    return (logits - torch.log(sum_exp_logits)).to(dtype)


def _get_log_prob(logits: torch.Tensor, lm_head: nn.Module = None) -> torch.Tensor:
    parallel_state: ParallelState | None = (
        getattr(lm_head, "parallel_state", None) if lm_head is not None else None
    )
    if parallel_state is not None and parallel_state.tensor_parallel_group.is_initialized():
        return _get_log_softmax_dist(logits, parallel_state.tensor_parallel_group.group)
    return torch.log_softmax(logits.float(), dim=-1)


def _get_kl_div_loss(
    log_prob_unquant: torch.Tensor, logits_quant: torch.Tensor, lm_head: nn.Module = None
) -> torch.Tensor:
    log_prob_quant = _get_log_prob(logits_quant, lm_head=lm_head)
    return F.kl_div(log_prob_quant, log_prob_unquant, reduction="sum", log_target=True)


def _get_lm_head(model: nn.Module) -> nn.Module:
    # HF models do allgather of logits to at lm_head
    # Hence lm_head outputs are not TP sharded - so we dont need to return the lm_head for TP KLDiv
    # Loss
    for name, module in model.named_modules():
        if name.endswith("output_layer"):  # Megatron models
            return module
    return None


class AutoQuantizeKLDivSearcher(_AutoQuantizeBaseSearcher):
    """A searcher for AutoQuantize algorithm that uses KL-Divergence loss based score estimation."""

    method_name = "kl_div"

    @property
    def default_search_config(self):
        """Get the default config for the searcher."""
        config = super().default_search_config
        config.update(
            {
                "forward_step": None,
            }
        )
        return config

    def sanitize_search_config(self, config: SearchConfig | None) -> SearchConfig:
        """Sanitize the search config dict."""
        config = config or {}
        for ignored_key in ["score_func", "loss_func", "forward_backward_step"]:
            if ignored_key in config:
                if config[ignored_key] is not None:
                    warnings.warn(
                        f"`{ignored_key}` is ignored for KL-Divergence loss based `auto_quantize`."
                    )
                config.pop(ignored_key)
        config = super().sanitize_search_config(config)
        assert config["forward_step"] is not None, (
            "`forward_step` must be provided for KL-Divergence loss based `auto_quantize`. "
            "`forward_step(model, data)` should return model logits."
        )
        return config

    @torch.inference_mode()
    def estimate_sensitivity_scores(self):
        """Estimate the sensitivity scores for the model.

        Higher score means more sensitive to quantization.
        """

        def set_to_unquantized():
            for name, hparam in named_hparams(self.model, unique=True):
                if not isinstance(hparam, QuantRecipeHparam):
                    continue
                if hparam.is_configurable:
                    hparam.active = QuantRecipe(quant_cfg=None)

        self.model.eval()
        num_iters = self.config["num_score_steps"]
        try:
            for i, data in tqdm(
                zip(range(num_iters), self.config["score_data_loader"]),
                desc="Estimating KLDivergence loss",
                total=num_iters,
            ):
                self._score_component_current_batch = self._score_component_batch_metadata(i, data)
                set_to_unquantized()
                logits_unquant = self.config["forward_step"](self.model, data)
                log_prob_unquant = _get_log_prob(logits_unquant, lm_head=_get_lm_head(self.model))

                for name, hparam in tqdm(
                    list(named_hparams(self.model, configurable=True)), desc="Evaluating hparams"
                ):
                    if not isinstance(hparam, QuantRecipeHparam):
                        continue
                    for recipe in hparam.choices:
                        if not isinstance(recipe, QuantRecipe):
                            continue
                        if recipe == QuantRecipe(quant_cfg=None):
                            continue
                        hparam.active = recipe
                        logits_quant = self.config["forward_step"](self.model, data)
                        score = _get_kl_div_loss(
                            log_prob_unquant, logits_quant, _get_lm_head(self.model)
                        )
                        if hparam._importance_dict[recipe][hparam.score_modules[0]] is None:
                            hparam._importance_dict[recipe][hparam.score_modules[0]] = score
                        else:
                            hparam._importance_dict[recipe][hparam.score_modules[0]] += score
                        self._record_score_component(hparam, recipe, hparam.score_modules[0], score)
                    hparam.active = QuantRecipe(quant_cfg=None)
        finally:
            self._score_component_current_batch = None

    def run_search_with_stats(self, max_weight_size, verbose=False):
        """Run threshold-based binary search for KLDivergence loss based auto_quantize.

        We use binary search to minimize the max(per-layer score) while meeting the constraint.
        """
        # Collect all sensitivity scores to determine initial threshold bounds
        all_scores = [
            score
            for name in self.candidate_stats
            for score in self._candidate_scores_for_search(self.candidate_stats[name], name)
        ]

        if not all_scores:
            warnings.warn("No scores available for threshold-based search!")
            is_satisfied = False
            return {}, is_satisfied

        # Initialize binary search bounds
        min_score = min(all_scores)
        max_score = max(all_scores)
        threshold = (min_score + max_score) / 2.0
        lower_bound = min_score
        upper_bound = max_score

        # Run for fixed number of iterations
        max_iterations = 100

        if verbose:
            print_rank_0("AutoQuantize: Starting threshold-based binary search")
            print_rank_0(f"  Score range: [{min_score:.6e}, {max_score:.6e}]")
            print_rank_0(f"  Target weight size: {max_weight_size:.2f}")

        for iteration in range(max_iterations):
            # Select recipes based on current threshold
            best_recipes = {}
            total_weight_size = 0.0

            for name in self.candidate_stats:
                formats = self.candidate_stats[name]["formats"]
                scores = self._candidate_scores_for_search(self.candidate_stats[name], name)
                costs = self.candidate_stats[name]["costs"]

                selected_idx = 0
                for idx in range(len(formats)):
                    if scores[idx] <= threshold:
                        selected_idx = idx
                        break

                best_recipes[name] = {
                    "format": formats[selected_idx],
                    "costs": costs[selected_idx],
                    "scores": self.candidate_stats[name]["scores"][selected_idx],
                    "objective_scores": scores[selected_idx],
                }
                total_weight_size += costs[selected_idx]

            # Check if we meet the constraint
            meets_constraint = total_weight_size <= max_weight_size

            if verbose:
                print_rank_0(
                    f"  Iteration {iteration + 1}: threshold={threshold:.6e}, "
                    f"weight_size={total_weight_size:.2f}, "
                    f"meets_constraint={meets_constraint}"
                )

            # Update binary search bounds
            if meets_constraint:
                upper_bound = threshold  # Threshold was too aggressive, relax it
            else:
                lower_bound = threshold  # Threshold was too lax, tighten it

            # Update threshold for next iteration
            threshold = (lower_bound + upper_bound) / 2.0

        # Final check if constraint is satisfied
        is_satisfied = total_weight_size <= max_weight_size

        if verbose:
            print_rank_0(
                f"AutoQuantize: Search complete. "
                f"Final weight size: {total_weight_size:.2f} "
                f"(target: {max_weight_size:.2f}), "
                f"constraint satisfied: {is_satisfied}"
            )

        return best_recipes, is_satisfied


# Backward compatibility alias (defaults to gradient-based searcher)
AutoQuantizeSearcher = AutoQuantizeGradientSearcher


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def get_auto_quantize_config(search_state, constraints=None, verbose=False):
    """Build a flat quant config dict from auto_quantize search_state.

    Re-solves for ``constraints`` if provided, otherwise uses the best recipe from the search.

    Args:
        search_state: The state dict returned by :func:`auto_quantize`.
        constraints: Optional dict with ``effective_bits`` key to re-solve for a new target.
        verbose: If True, prints the per-layer recipe assignments.

    Returns:
        A config dict suitable for :func:`quantize`.
    """
    if constraints is not None:
        best_recipe = _resolve_best_recipe(search_state, constraints, verbose=verbose)
    else:
        best_recipe = search_state["best"]["recipe"]

    def _cfg_to_dict(v):
        if isinstance(v, mtq_config.QuantizerAttributeConfig):
            return {
                "num_bits": v.num_bits,
                **v.model_dump(exclude_defaults=True),
            }
        if isinstance(v, list):
            return [_cfg_to_dict(c) for c in v]
        return v

    quant_cfg: list[dict] = [{"quantizer_name": "*", "enable": False}]
    quant_cfg.extend(
        {"quantizer_name": pattern, "enable": False}
        for pattern in _as_list(search_state.get("disabled_layers"))
    )
    per_module_entries: list[dict] = []
    _per_module_attrs = ("input_quantizer", "weight_quantizer", "output_quantizer")
    # Track global (non per-module) recipe entries.  Last recipe wins for each pattern.
    global_entries: dict[str, dict] = {}

    for hparam_name, recipe in best_recipe.items():
        if recipe == QuantRecipe(quant_cfg=None):
            continue
        module_names = search_state["candidate_stats"][hparam_name]["module_names"]
        for module_name in module_names:
            for quantizer_attr in _per_module_attrs:
                matched_cfg, matched_enable = _match_quantizer_cfg(
                    recipe.config.quant_cfg, quantizer_attr
                )
                if matched_enable is not None:
                    entry: dict[str, Any] = {
                        "quantizer_name": f"{module_name}.{quantizer_attr}",
                        "enable": matched_enable,
                    }
                    if matched_cfg is not None:
                        entry["cfg"] = _cfg_to_dict(matched_cfg)
                    per_module_entries.append(entry)

        # Collect non-per-module entries (e.g. *[kv]_bmm_quantizer) from winning recipes.
        for recipe_entry in recipe.config.quant_cfg:
            pattern = recipe_entry["quantizer_name"]
            if pattern == "*" or any(
                fnmatch.fnmatch(attr, pattern) or pattern.endswith(attr)
                for attr in _per_module_attrs
            ):
                continue
            cfg = recipe_entry.get("cfg")
            enable = recipe_entry.get("enable", True)
            ge: dict[str, Any] = {"quantizer_name": pattern, "enable": enable}
            if cfg is not None:
                ge["cfg"] = _cfg_to_dict(cfg)
            global_entries[pattern] = ge

    # Keep path-scoped recipe entries before explicit module entries so selected
    # modules override default disables such as ``*lm_head*``.
    quant_cfg.extend(global_entries.values())
    quant_cfg.extend(per_module_entries)
    warnings.warn(
        "get_auto_quantize_config: returned config uses algorithm='max'. "
        "Per-recipe calibration algorithms (e.g. smoothquant, awq) are not preserved. "
        "Update config['algorithm'] if a different calibration algorithm is needed (e.g. 'gptq')."
    )
    return {"quant_cfg": quant_cfg, "algorithm": "max"}


def _get_search_replay_constraints(search_state, constraints):
    constraints = dict(constraints or search_state.get("best", {}).get("constraints", {}))
    if "effective_bits" not in constraints:
        raise ValueError(
            "constraints must contain 'effective_bits' when replaying an AutoQuantize search_state."
        )
    return constraints


def _build_searcher_from_search_state(search_state, constraints):
    effective_bits = constraints["effective_bits"]
    compression = effective_bits / 16.0
    candidate_stats = search_state["candidate_stats"]
    total_weight_size = search_state.get("cost_denominator") or sum(
        s["costs"][-1] for s in candidate_stats.values()
    )
    max_weight_size = total_weight_size * compression
    method = search_state["method"]

    if method == "gradient":
        searcher = AutoQuantizeGradientSearcher()
    elif method in {"hidden_recon", "parent_recon"}:
        searcher = AutoQuantizeHiddenReconSearcher()
    elif method == "kl_div":
        searcher = AutoQuantizeKLDivSearcher()
    else:
        raise ValueError(
            f"Unknown autoquant search method: {method!r}. "
            "Expected 'gradient', 'hidden_recon', 'parent_recon', or 'kl_div'."
        )

    searcher.candidate_stats = candidate_stats
    searcher.cost_model = search_state.get("cost_model", COST_MODEL_WEIGHT)
    searcher.cost = search_state.get("cost", {})
    searcher.active_moe_expert_ratio = search_state.get("active_moe_expert_ratio")
    if (
        searcher.cost_model == COST_MODEL_ACTIVE_MOE
        and not searcher.cost
        and searcher.active_moe_expert_ratio is not None
    ):
        searcher.cost = {ACTIVE_MOE_EXPERT_RATIO_KEY: searcher.active_moe_expert_ratio}
    searcher.config = {
        **searcher.default_search_config,
        "cost_model": searcher.cost_model,
        "cost": searcher.cost,
        "active_moe_expert_ratio": searcher.active_moe_expert_ratio,
    }
    searcher.constraints = {
        "effective_bits": effective_bits,
        "cost_model": searcher.cost_model,
        "cost": searcher.cost,
        "score_model": constraints.get("score_model", AUTO_QUANTIZE_SCORE_MODEL_RAW),
    }
    if constraints.get("cost_lower_bound") is not None:
        searcher.constraints["cost_lower_bound"] = constraints["cost_lower_bound"]
    if constraints.get("response_risk") is not None:
        searcher.constraints["response_risk"] = constraints["response_risk"]
    if constraints.get("candidate_rerank") is not None:
        searcher.constraints["candidate_rerank"] = constraints["candidate_rerank"]
    return searcher, max_weight_size, total_weight_size


def get_auto_quantize_candidate_packets(
    search_state, constraints=None, top_k: int = 1, verbose=False
):
    """Replay an AutoQuantize search_state and return top-k candidate packets.

    This is intended for fast offline inspection of saved AutoQuant states.  It
    reuses stored candidate stats, enumerates real LP candidates with no-good
    cuts, and annotates packets with launch_authority="no" by default so callers
    do not confuse metadata/screen triage with final benchmark promotion.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")

    constraints = _get_search_replay_constraints(search_state, constraints)
    candidate_rerank = dict(constraints.get("candidate_rerank") or {})
    candidate_rerank["enabled"] = True
    candidate_rerank["top_k"] = top_k
    candidate_rerank.setdefault("launch_authority_default", "no")
    constraints["candidate_rerank"] = candidate_rerank

    searcher, max_weight_size, _ = _build_searcher_from_search_state(search_state, constraints)
    _, is_satisfied = searcher.run_search_with_stats(max_weight_size, verbose=verbose)
    launch_authority = candidate_rerank.get("launch_authority_default", "no")
    packets = [
        {
            **packet,
            "launch_authority": launch_authority,
            "source_provenance": "restored_auto_quantize_state",
        }
        for packet in getattr(searcher, "last_candidate_packets", [])
    ]

    return {
        "is_satisfied": is_satisfied,
        "constraints": constraints,
        "candidate_packets": packets,
        "candidate_rerank_source": {
            "source_path": candidate_rerank.get("source_path"),
            "scale": candidate_rerank.get("scale", 1.0),
            "id_field": candidate_rerank.get("id_field", "signature"),
            "score_field": candidate_rerank.get("score_field", "rerank_score"),
            "provenance": candidate_rerank.get("provenance"),
            "loaded_signal_entries": len(searcher._get_candidate_rerank_signal()),
        },
    }


def _resolve_best_recipe(search_state, constraints, verbose=False):
    constraints = _get_search_replay_constraints(search_state, constraints)
    searcher, max_weight_size, total_weight_size = _build_searcher_from_search_state(
        search_state, constraints
    )
    best_recipe_info, _ = searcher.run_search_with_stats(max_weight_size, verbose=verbose)

    best_recipe = {name: info["format"] for name, info in best_recipe_info.items()}
    if verbose:
        total_cost = sum(info["costs"] for info in best_recipe_info.values())
        _AutoQuantizeBaseSearcher._print_recipe_summary(
            best_recipe, total_cost, total_weight_size, prefix="get_auto_quantize_config"
        )

    return best_recipe


def _match_quantizer_cfg(quant_cfg, quantizer_attr):
    # Last-match-wins to mirror set_quantizer_by_cfg behavior.
    # Patterns may be path-scoped (e.g. "*mlp*weight_quantizer") while quantizer_attr
    # is a bare name like "weight_quantizer".  We match if the bare name matches directly
    # OR if the pattern ends with the bare quantizer_attr (path-scoped match).
    matched = None
    matched_enable = None
    for entry in quant_cfg:
        parent_class = entry.get("parent_class") if hasattr(entry, "get") else entry.parent_class
        if parent_class is not None:
            continue
        pattern = entry["quantizer_name"]
        cfg = entry.get("cfg")
        enable = entry.get("enable", True)
        # Direct match: the bare quantizer_attr matches the whole pattern (e.g. "*weight_quantizer")
        if fnmatch.fnmatch(quantizer_attr, pattern) or pattern.endswith(quantizer_attr):
            matched = cfg
            matched_enable = enable

    return matched, matched_enable
