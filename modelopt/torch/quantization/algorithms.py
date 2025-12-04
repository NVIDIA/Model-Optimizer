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

import fnmatch
import gc
import types
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Any

import regex as re
import torch
import torch.nn as nn
from tqdm import tqdm

from modelopt.torch.opt.conversion import ModeloptStateManager
from modelopt.torch.opt.hparam import CustomHPType, Hparam, HPType
from modelopt.torch.opt.searcher import LPS, BaseSearcher, SearchConfig, SearchStateDict
from modelopt.torch.opt.utils import get_hparam, named_hparams
from modelopt.torch.utils import create_param_grad_clear_hook, print_rank_0, report_memory
from modelopt.torch.utils.distributed import DistributedProcessGroup, ParallelState, is_master

from . import config as mtq_config
from . import model_calib
from .config import QuantizeConfig, QuantizerAttributeConfig
from .conversion import set_quantizer_by_cfg
from .nn import QuantLinearConvBase, QuantModule, SequentialQuantizer, TensorQuantizer
from .utils import is_quantized_linear


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
            return min(estimate_quant_compression_for_quantizer(q) for q in quantizer_attr_cfg)
        if isinstance(quantizer_attr_cfg, dict):
            return estimate_quant_compression_for_quantizer(list(quantizer_attr_cfg.values()))

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

    return estimate_quant_compression_for_quantizer(list(quant_cfg.quant_cfg.values()))


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
            quant_cfg = {"quant_cfg": {"*": {"enable": False}}}
        elif isinstance(quant_cfg, str):
            assert hasattr(mtq_config, quant_cfg), f"Unknown quantization format {quant_cfg}"
            quant_cfg = getattr(mtq_config, quant_cfg)
        else:
            assert name is not None, "name must be provided for custom quantization formats"

        self.config = mtq_config.QuantizeConfig(**quant_cfg)  # type: ignore [arg-type]

        # Disable KV Cache quantization
        # Currently KV Cache quantization is enabled for some quantization formats and disabled for others
        # This breaks the monotonicity of the quantization formats in terms of weight compression Vs accuracy
        self.config.quant_cfg["*output_quantizer"] = mtq_config.QuantizerAttributeConfig(
            enable=False
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
                with SequentialQuantizer.convert_to_single_quantizer(model):
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
    ) -> None:
        """Initializes Hparam with original value and choices."""
        choices = sorted({*(choices if choices else []), QuantRecipe(quant_cfg=None)})
        super().__init__(choices, original=choices[0])

        self.name = name

        self.quant_modules = list(set(quant_modules or []))
        self.score_modules = list(set(score_modules or self.quant_modules))

        # This is a hack; We dont want to make the input_quantizer, weight_quantizer, output_quantizer
        # a dynamic attribute for backward compatibility with the model_calib.py
        # TODO: Make input_quantizer, weight_quantizer, output_quantizer a dynamic attribute and get rid of this hack
        self._all_quantizer_choices = {quant_recipe: {} for quant_recipe in self.choices}

        quant_recipe: QuantRecipe
        for quant_recipe in self.choices:
            for quant_module in self.quant_modules:
                for quantizer_attr_name in [
                    "input_quantizer",
                    "weight_quantizer",
                    "output_quantizer",
                ]:
                    setattr(quant_module, quantizer_attr_name, TensorQuantizer())

                set_quantizer_by_cfg(quant_module, quant_recipe.config.quant_cfg)
                self._all_quantizer_choices[quant_recipe][quant_module] = {
                    quantizer_attr_name: getattr(quant_module, quantizer_attr_name)
                    for quantizer_attr_name in [
                        "input_quantizer",
                        "weight_quantizer",
                        "output_quantizer",
                    ]
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

    def get_cost(self, recipe: QuantRecipe) -> float:
        """Get the cost for a given recipe.

        The cost is the total weight size of the quantizable modules multiplied by
        the compression ratio of the recipe.
        """
        cost = 0
        for quant_module in self.quant_modules:
            weight_size = _AutoQuantizeBaseSearcher._get_total_weight_size([quant_module])
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
        return ["name", *super().attrs]


class _AutoQuantizeBaseSearcher(BaseSearcher, ABC):
    """Base searcher for AutoQuantize algorithm."""

    # This searcher finds optimal per-layer quantization by searching across quantization formats
    # for each quantizable module (quant module). Optionally, quant grouping rules can restrict
    # certain modules to share the same format. Sensitivity scores are computed from perturbations
    # at score modules. See AutoQuantizeGradientSearcher for detailed documentation.

    candidate_stats: dict[str, dict[str, list[float]]]
    best: dict[str, Any]

    quant_grouping_rules = [
        r"^(.*?)\.(q_proj|k_proj|v_proj)$",  # q_proj, k_proj, v_proj for llama like models
        # gate_proj, up_proj, down_proj for Qwen3 like MoE models
        r"^(.*?\.mlp\.experts)\.\d+\.(gate_proj|up_proj|down_proj)$",
        r"^(.*?)\.(gate_proj|up_proj)$",  # gate_proj, up_proj for llama like models
        r"^(.*?)\.(\d+\.(w1|w2|w3))$",  # mixtral experts
        r"^(.*?)\.((w1_linear|w2_linear|w3_linear)\.\d+)$",  # dbrx experts
    ]

    score_module_rules = []

    @property
    def default_search_config(self):
        """Get the default config for the searcher."""
        return {
            "quantization_formats": ["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG"],
            "data_loader": None,
            "num_calib_steps": 512,
            "num_score_steps": 128,
            "deployment": None,
            "disabled_layers": None,
            "verbose": is_master(),
            "checkpoint": None,
        }

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Get the default state dict for AutoQuantize."""
        return {
            "candidate_stats": defaultdict(dict),
            "best": {"recipe": {}, "constraints": {}, "score": float("inf"), "is_satisfied": False},
        }

    def sanitize_search_config(self, config: SearchConfig | None) -> SearchConfig:
        """Sanitize the search config dict."""
        config = config or {}
        config = super().sanitize_search_config(config)
        assert config["data_loader"] is not None, (
            "`data_loader` must be provided for `auto_quantize`."
        )
        assert config["forward_step"] is not None, (
            "`forward_step` must be provided for `auto_quantize`."
        )
        return config

    @staticmethod
    def _is_auto_quantize_module(module):
        return (
            is_quantized_linear(module) or isinstance(module, QuantLinearConvBase)
        ) and isinstance(module, QuantModule)

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
            for rule in self.quant_grouping_rules:
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

            quant_recipes = None if disabled else quant_recipes
            hparam = QuantRecipeHparam(
                quant_recipes,
                quant_modules=quant_modules,
                score_modules=score_modules,
                name=str(group_key),
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
            prev_score = float("inf")
            for recipe in hparam.choices:
                formats.append(recipe)

                score = hparam.get_score(recipe)  # type: ignore [arg-type]
                cost = hparam.get_cost(recipe)  # type: ignore [arg-type]

                score = min(score, prev_score)  # TODO: Should we get rid of this?
                scores.append(score)
                costs.append(cost)
                prev_score = score

            self.candidate_stats[name]["formats"] = formats
            self.candidate_stats[name]["scores"] = scores
            self.candidate_stats[name]["costs"] = costs

    def _run_func(self, func, num_iters=1, desc=""):
        for i, data in tqdm(
            zip(range(num_iters), self.config["data_loader"]),
            desc=desc,
            total=num_iters,
        ):
            func(self.model, data)

    def before_search(self):
        """Prepare the model for search by calibrating the quantizers  and collecting ``AutoQuantize`` score."""
        # Import here to avoid circular import
        from modelopt.torch.quantization.model_quant import calibrate

        super().before_search()

        search_recipes = self._get_search_recipes(self.config["quantization_formats"])
        self._verify_constraint(search_recipes)
        self.insert_hparams_after_merge_rules(
            self.model, search_recipes, self.config["disabled_layers"]
        )

        QuantRecipe.disable_folding_pqs_to_weights()

        # Iterate over the search recipes and calibrate the quantizers for each recipe
        for recipe in search_recipes:
            if recipe == QuantRecipe(quant_cfg=None):  # No-quant format
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

            for name, hparam in named_hparams(self.model, configurable=True):
                if not isinstance(hparam, QuantRecipeHparam):
                    continue
                hparam.active = recipe

            # Now calibrate the quantizers for the recipe
            calibrate(
                self.model,
                algorithm=recipe.config.algorithm,
                forward_loop=forward_loop,
            )
            # Calibrate adds a new mode to the model. Since auto_quantize mixes the quantization recipes
            # across layers, lets not save this new mode in the modelopt state.
            # TODO: This is a hack. We need to create a mode for auto_quantize to handle this in a clean way.
            ModeloptStateManager(self.model).state_dict().pop()

        if self.candidate_stats:
            if self.config["verbose"]:
                print_rank_0("AutoQuantize: Restored from checkpoint, skipping scoring")
            return

        self.estimate_sensitivity_scores()
        self.initialize_candidate_stats()
        # Save checkpoint after successful score estimation
        self.save_search_checkpoint(verbose=self.config["verbose"])

    @staticmethod
    def _get_total_weight_size(modules):
        return sum(
            (
                module.weight.numel()
                if _AutoQuantizeBaseSearcher._is_auto_quantize_module(module)
                else 0
            )
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

    @abstractmethod
    def run_search_with_stats(self, max_weight_size, verbose=False):
        """Run the search with stats to get the best recipe and whether the constraints are satisfied."""

    def run_search(self):
        """Search for the best per-layer quantization configuration and return the best model and configuration."""
        verbose = self.config["verbose"]
        assert len(self.constraints) == 1 and "effective_bits" in self.constraints, (
            f"`constraints` must contain only 'effective_bits' constraint. "
            f"Got {self.constraints.keys()}"
        )

        compression = self._get_formatted_weight_compression_constraint()
        total_weight_size = self._get_total_weight_size(self.model.modules())
        max_weight_size = total_weight_size * compression

        # Run the search with stats to get the best recipe and whether the constraints are satisfied
        best_recipe_info, is_satisfied = self.run_search_with_stats(max_weight_size, verbose)
        self.best["is_satisfied"] = is_satisfied

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
                print_rank_0(
                    f"AutoQuantize best recipe for {name.replace('.quant_recipe', '')}: {best_recipe[name]}"
                )

        effective_bits_from_search = (best_constraints / total_weight_size) * 16
        if verbose:
            print_rank_0(
                f"AutoQuantize effective bits from search: {effective_bits_from_search: .2f}"
            )

        self.best["recipe"] = best_recipe
        self.best["constraints"] = {"effective_bits": effective_bits_from_search}
        self.best["score"] = best_scores

        QuantRecipe.fold_pqs_to_weights(self.model)


def _get_auto_quantize_score(grad_output, output_diff):
    return ((grad_output.float() ** 2) * (output_diff.float() ** 2)).sum()


def _add_auto_quantize_score(grad_output, output_diff, score_tensor):
    score_tensor += _get_auto_quantize_score(grad_output, output_diff)


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

    score_module_rules = [
        # Use MLP layer output for gate_proj, up_proj, down_proj for Qwen3 like MoE models (local and shared experts)
        r"^(.*?\.mlp)\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",
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
        def auto_quantize_score_estimate_forward(module, input, *args, **kwargs):
            for hparam in module._hparams_for_scoring:
                if hparam.is_configurable:
                    hparam.active = QuantRecipe(quant_cfg=None)

            output = module._forward_original(input, *args, **kwargs)

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
                        output_diff = module._forward_original(input, *args, **kwargs)

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
                    if hparam._importance_dict[recipe][module] is None:
                        hparam._importance_dict[recipe][module] = _get_auto_quantize_score(
                            grad_output[0], output_diff
                        )
                    else:
                        _add_auto_quantize_score(
                            grad_output[0], output_diff, hparam._importance_dict[recipe][module]
                        )

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
        # TODO: Do this only for rank 0 in the respective pipeline group

        for lower_bound in [None, 0.99, 0.90]:
            # The LP solver for auto_quantize sometimes fails to find a solution if a lower bound is not
            # specified. I dont know why this happens.
            # As a workaround, lets specify a lower bound for the weight compression if previous
            # search without lower bound fails.
            constraints, constraint_name = self._get_constraints_for_search(
                max_weight_size, lower_bound
            )

            lps = LPS(
                name="AutoQuantize",
                constraints=constraints,
                constraints_to_candidate_costs={
                    constraint_name: [
                        candidate_stat["costs"] for candidate_stat in self.candidate_stats.values()
                    ]
                },
                candidate_scores=[
                    candidate_stat["scores"] for candidate_stat in self.candidate_stats.values()
                ],
                objective_type="minimize",
                verbose=verbose,
            )
            selections, self.status = lps()
            if self.status == "Optimal":
                break

        if self.status != "Optimal":
            warnings.warn(
                "AutoQuantize FAILED to find a solution! The searched model might not meet all constraints. "
            )
            is_satisfied = False
        else:
            is_satisfied = True

        best_recipes = {}
        for name, selected_idx in zip(self.candidate_stats.keys(), selections):
            best_recipes[name] = {
                "format": self.candidate_stats[name]["formats"][selected_idx],
                "costs": self.candidate_stats[name]["costs"][selected_idx],
                "scores": self.candidate_stats[name]["scores"][selected_idx],
            }

        return best_recipes, is_satisfied


# TODO: Enable torch compile for this function
# Currently modelopt.onnx is breaking this
def _get_softmax_dist(
    logits: torch.Tensor, tp_group, return_log_prob: bool = False
) -> torch.Tensor:
    # TODO: test this
    dtype = logits.dtype
    max_logits = torch.amax(logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(max_logits, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    logits = (logits - max_logits).float()
    sum_exp_logits = torch.exp(torch.logsumexp(logits, dim=-1, keepdim=True))
    torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    logits = logits - torch.log(sum_exp_logits)
    if return_log_prob:
        return logits.to(dtype)
    else:
        return torch.exp(logits).to(dtype)


def _get_softmax(logits: torch.Tensor, return_log_prob: bool = False) -> torch.Tensor:
    # TODO: do we need to do log_softmax in float32?
    # log_softmax is supposed to be numerically stable implementation
    log_prob = torch.log_softmax(logits.float(), dim=-1)
    if return_log_prob:
        return log_prob
    else:
        return torch.exp(log_prob)


def _get_p_log_q(p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    return torch.sum(p * log_q).float()


def _get_prob_from_logits(
    logits: torch.Tensor, return_log_prob: bool = False, lm_head: nn.Module = None
) -> torch.Tensor:
    parallel_state: ParallelState | None = (
        getattr(lm_head, "parallel_state", None) if lm_head is not None else None
    )
    if parallel_state is not None and parallel_state.tensor_parallel_group.is_initialized():
        return _get_softmax_dist(
            logits, parallel_state.tensor_parallel_group.group, return_log_prob
        )
    return _get_softmax(logits, return_log_prob)


def _get_kl_div_loss(
    prob_unquant: torch.Tensor, logits_quant: torch.Tensor, lm_head: nn.Module = None
) -> torch.Tensor:
    log_prob_quant = _get_prob_from_logits(logits_quant, return_log_prob=True, lm_head=lm_head)
    # We dont need to calculate the full kl div loss here, just get - p*log_q
    return -_get_p_log_q(prob_unquant, log_prob_quant)


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
        for _, data in tqdm(
            zip(range(num_iters), self.config["data_loader"]),
            desc="Estimating KLDivergence loss",
            total=num_iters,
        ):
            set_to_unquantized()
            logits_unquant = self.config["forward_step"](self.model, data)
            prob_unquant = _get_prob_from_logits(
                logits_unquant,
                return_log_prob=False,
                lm_head=_get_lm_head(self.model),
            )

            for name, hparam in tqdm(
                list(named_hparams(self.model, configurable=True)), desc="Evaluating hparams"
            ):
                if not isinstance(hparam, QuantRecipeHparam):
                    continue
                for recipe in hparam.choices:
                    if recipe == QuantRecipe(quant_cfg=None):
                        continue
                    hparam.active = recipe
                    logits_quant = self.config["forward_step"](self.model, data)
                    score = _get_kl_div_loss(prob_unquant, logits_quant, _get_lm_head(self.model))
                    if hparam._importance_dict[recipe][hparam.score_modules[0]] is None:
                        hparam._importance_dict[recipe][hparam.score_modules[0]] = score
                    else:
                        hparam._importance_dict[recipe][hparam.score_modules[0]] += score
                hparam.active = QuantRecipe(quant_cfg=None)

    def run_search_with_stats(self, max_weight_size, verbose=False):
        """Run threshold-based binary search for KLDivergence loss based auto_quantize.

        We use binary search to minimize the max(per-layer score) while meeting the constraint.
        """
        # Collect all sensitivity scores to determine initial threshold bounds
        all_scores = [
            score for name in self.candidate_stats for score in self.candidate_stats[name]["scores"]
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
                scores = self.candidate_stats[name]["scores"]
                costs = self.candidate_stats[name]["costs"]

                selected_idx = 0
                for idx in range(len(formats)):
                    if scores[idx] <= threshold:
                        selected_idx = idx
                        break

                best_recipes[name] = {
                    "format": formats[selected_idx],
                    "costs": costs[selected_idx],
                    "scores": scores[selected_idx],
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
