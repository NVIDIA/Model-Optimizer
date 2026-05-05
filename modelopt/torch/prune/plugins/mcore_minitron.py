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

"""Module implementing top-level ``mcore_minitron`` pruning handler for NVIDIA Megatron-Core models.

Minitron pruning algorithm uses activation magnitudes to estimate importance of neurons / attention heads / mamba heads
in the model.
More details on Minitron pruning algorithm can be found here: https://arxiv.org/pdf/2407.14679

Supports both GPT (attention-based) and Mamba (state-space) models, as well as hybrid models with both types of layers.

Actual dynamic module implementations are at :mod:`modelopt.torch.nas.plugins.megatron`.
"""

import io
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import Any
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from pydantic import create_model
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

from modelopt.torch.nas.conversion import NASModeRegistry
from modelopt.torch.nas.plugins.megatron import (
    HAS_MAMBA,
    SUPPORTED_MODELS,
    _DynamicMambaLayer,
    _DynamicMambaMixer,
    _DynamicMCoreLanguageModel,
    _DynamicMLP,
    _DynamicMoELayer,
    _DynamicSelfAttention,
    _DynamicSequentialMLP,
    _DynamicTransformerLayer,
)
from modelopt.torch.nas.plugins.megatron_model_stats import (
    mcore_memory_footprint_mb,
    mcore_param_count,
    parse_main_layer_chars,
    print_mcore_model_stats,
)
from modelopt.torch.nas.registry import DMRegistry
from modelopt.torch.nas.utils import get_subnet_config, sample, sort_parameters
from modelopt.torch.opt.config import ModeloptBaseConfig, get_kwargs_for_create_model_with_rules
from modelopt.torch.opt.conversion import ApplyModeError
from modelopt.torch.opt.dynamic import DynamicModule, DynamicSpace
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ConvertReturnType,
    ModeDescriptor,
    RestoreEntrypoint,
)
from modelopt.torch.opt.searcher import BaseSearcher, SearchConfig, SearchStateDict
from modelopt.torch.opt.utils import named_hparams
from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils import get_module_device, num2hrb, print_rank_0

from ..pruning import PruneModeRegistry

SUPPORTED_HPARAMS = {
    # 1. Width pruning
    "hidden_size",
    # MLP
    "ffn_hidden_size",
    # Attention
    "num_attention_heads",
    # Mamba
    "mamba_num_heads",
    "mamba_head_dim",
    # MoE
    "moe_ffn_hidden_size",
    "moe_shared_expert_intermediate_size",
    "num_moe_experts",
    # 2. Depth pruning
    "num_layers",
}

__all__ = [
    "SUPPORTED_HPARAMS",
    "MCoreMinitronConfig",
    "MCoreMinitronModeDescriptor",
    "MCoreMinitronSearcher",
    "drop_mcore_language_model_layers",
    "get_mcore_minitron_config",
]


def drop_mcore_language_model_layers(model: nn.Module, *, layers_to_drop: list[int]) -> None:
    """Remove given layers (1-indexed) of the model (works with TP and/or PP).

    If model is a wrapper around GPTModel or MambaModel, it will be unwrapped.
    """
    layers_to_drop = sorted(layers_to_drop)
    assert layers_to_drop[0] >= 1, (
        f"Layers to drop should be in range 1 to {model.config.num_layers}, got {layers_to_drop}."
    )

    supported_model_types = tuple(SUPPORTED_MODELS.keys())
    for n, m in model.named_modules():
        if isinstance(m, supported_model_types):
            model = m
            break
    assert isinstance(model, supported_model_types), (
        f"Model should have one of {supported_model_types} submodule, got {model}"
    )
    print_rank_0(f"Dropping decoder layers {layers_to_drop} from model.")

    # get the number of layers remaining in each pp rank
    layers_remaining_per_pp = torch.zeros(
        get_pipeline_model_parallel_world_size(),
        dtype=torch.int,
        device=get_module_device(model),
    )
    layers_remaining = torch.tensor(
        sum(1 for layer in model.decoder.layers if layer.layer_number not in layers_to_drop),
        dtype=torch.int,
        device=get_module_device(model),
    )

    # Below distributed gather requires tensors to be on cuda
    layers_remaining_per_pp = layers_remaining_per_pp.cuda()
    layers_remaining = layers_remaining.cuda()
    torch.distributed.all_gather_into_tensor(
        layers_remaining_per_pp, layers_remaining, group=get_pipeline_model_parallel_group()
    )
    layers_remaining_per_pp = [i.item() for i in layers_remaining_per_pp]
    new_num_layers = sum(layers_remaining_per_pp)

    # reindex kept layers, exclude sharded state dict for dropped layers
    layer_number = sum(layers_remaining_per_pp[: get_pipeline_model_parallel_rank()]) + 1
    kept_layers = []
    for layer in model.decoder.layers:
        if layer.layer_number not in layers_to_drop:
            layer.layer_number = layer_number
            layer_number += 1
            kept_layers.append(layer)
    model.decoder.layers = nn.ModuleList(kept_layers)

    model.config.num_layers = new_num_layers


def _rprint(*renderables: Any) -> None:
    """Render rich renderables and print on rank 0 only."""
    buf = io.StringIO()
    Console(file=buf, highlight=False, force_terminal=sys.stdout.isatty(), width=160).print(
        *renderables
    )
    print_rank_0()
    print_rank_0(buf.getvalue())


# Constraint keys that trigger the grid-search path in MCoreMinitronSearcher.
# Order defines priority: first active key is used as the primary display/sort metric.
_METRIC_CONSTRAINT_PRIORITY = ("active_params", "params", "memory_mb")
_METRIC_CONSTRAINTS = frozenset(_METRIC_CONSTRAINT_PRIORITY)


@dataclass
class CandidateSubnet:
    ss_config: dict
    metrics: dict[str, float]
    score: float | None


torch.serialization.add_safe_globals([CandidateSubnet])


class MCoreMinitronSearcher(BaseSearcher):
    """Searcher for Minitron pruning algorithm.

    Supported constraint keys: ``export_config``, ``params``, ``active_params``, ``memory_mb``.

    Available additional config options (used when a metric constraint is provided):
    - `max_width_pruning`: Maximum fraction per width hyperparameter to prune (default: 0.40).
        Only top (1 - max_width_pruning) choices will be considered.
    - `max_depth_pruning`: Maximum fraction per depth hyperparameter to prune (default: 0.20).
        Only top (1 - max_depth_pruning) choices will be considered.
    - `hparams_to_skip`: List of hparams to skip during the search (default: None).
    - `top_k`: Number of candidates to consider for score_func validation (default: 10).
    - `seq_length`: Sequence length for KV-cache memory estimate (default: 4096).
        Only used with the ``memory_mb`` constraint.
    - `batch_size`: Batch size for KV-cache and Mamba-state memory estimate (default: 1).
        Only used with the ``memory_mb`` constraint.
    """

    local_activations: dict[str, torch.Tensor]
    layer_scores: dict[int, torch.Tensor]
    sorted_layers: list[int] | None  # 1-indexed sorted list of layer numbers
    # Dict from params constraint to list of all CandidateSubnets fitting that constraint
    all_candidates_per_constraint: dict[tuple, list[CandidateSubnet]]

    @property
    def default_search_config(self) -> SearchConfig:
        """Get the default config for the searcher."""
        return {
            **super().default_search_config,
            "max_iter_data_loader": 1024,
            "skip_sorting": False,
            "scores_path": None,
            # Additional search config for metric-based pruning
            "max_width_pruning": 0.40,
            "max_depth_pruning": 0.20,
            "hparams_to_skip": None,
            "top_k": 10,
            # Memory footprint config (only used with memory_mb constraint)
            "seq_length": 4096,
            "batch_size": 1,
        }

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Return default state dict for importance scores and activations from forward loop."""
        return {
            "local_activations": {},
            "layer_scores": {},
            "sorted_layers": None,
            "all_candidates_per_constraint": {},
        }

    def sanitize_search_config(self, config: SearchConfig | None) -> SearchConfig:
        """Sanitize the search config dict."""
        config = super().sanitize_search_config(config)
        if config["scores_path"]:
            config["checkpoint"] = config["scores_path"]
        config["verbose"] = True  # Print for all ranks
        return config

    def before_search(self) -> None:
        """Optional pre-processing steps before the search."""
        super().before_search()

        # Check that the constraint is valid.
        # export_config must be the sole key; metric constraints can be combined freely.
        active_metric_keys = self.constraints.keys() & _METRIC_CONSTRAINTS
        assert self.constraints.keys() <= {"export_config"} | _METRIC_CONSTRAINTS, (
            f"Only {sorted({'export_config'} | _METRIC_CONSTRAINTS)} constraints are supported!"
        )
        assert not ("export_config" in self.constraints and active_metric_keys), (
            "export_config cannot be combined with metric constraints!"
        )
        assert self.constraints, "At least one constraint must be provided!"

        if "export_config" in self.constraints:
            export_config = self.constraints["export_config"]
            assert isinstance(export_config, dict)  # to keep mypy happy
            if "num_query_groups" in export_config:
                warn("num_query_groups is no longer supported (since 0.41)! It will be ignored.")
                if export_config["num_query_groups"] != self.model.config.num_query_groups:
                    raise ValueError(
                        f"num_query_groups must be {self.model.config.num_query_groups}!"
                    )
                export_config.pop("num_query_groups")
            assert export_config.keys() <= SUPPORTED_HPARAMS, (
                f"Only {SUPPORTED_HPARAMS} are supported for pruning! Received: {export_config=}"
            )

            # Only sort the parameters that are to be pruned
            # If a user only prunes depth, we should not sort width parameters
            self.hps_to_sort = set(export_config.keys())
        else:
            for k in active_metric_keys:
                assert isinstance(self.constraints[k], (int, float)), f"{k} must be a float!"
            assert self.has_score, "score_func (e.g. MMLU) is required for metric-based pruning!"
            export_config = None
            # Sort all parameters for metric-based pruning
            self.hps_to_sort = SUPPORTED_HPARAMS

        for n, hp in named_hparams(self.model, unique=True):
            hp_name = n.split(".")[-1]
            if hp.is_configurable:
                # Make sure configurable hparams are the ones with right names else implementation needs to be fixed!
                assert hp_name in SUPPORTED_HPARAMS, f"[ImplError] Invalid hparam {hp_name}!"
                if export_config is not None and hp_name in export_config:
                    assert export_config[hp_name] in hp.choices, (
                        f"Invalid choice {export_config[hp_name]} for {n}! Available choices: {hp.choices}"
                    )
            hp.reset_choices()  # Make sure ConcatHparam choices are updated after modify()

        assert isinstance(self.model, _DynamicMCoreLanguageModel), (
            "Input should be unwrapped MCore model!"
        )

    def run_search(self) -> None:
        """Run forward loop to collect activations, sort parameters, and prune the model."""
        print_mcore_model_stats(
            self.model, "Original Model", self.config["seq_length"], self.config["batch_size"]
        )
        registry = ImportanceEstimatorRegistry(self.model)
        if self.local_activations and self.layer_scores:  # Available from per-rank checkpoint
            registry.set_local_activations_and_layer_scores(
                self.local_activations, self.layer_scores
            )
        elif not self.config["skip_sorting"]:
            assert self.forward_loop is not None
            is_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                self.forward_loop(self.model)
            self.model.train(is_training)

            # Store activations and layer scores for re-pruning with different export configs
            self.local_activations, self.layer_scores = (
                registry.get_local_activations_and_layer_scores()
            )
            self.save_search_checkpoint(verbose=True)

        if self.config["skip_sorting"]:
            print_rank_0("Skipping sorting parameters...")
        else:
            sort_parameters(self.model, self.hps_to_sort, verbose=False)
        registry.cleanup()

        if self.layer_scores:
            # sort layers by scores and drop the lowest ones
            self.sorted_layers = [
                layer
                for layer, _ in sorted(self.layer_scores.items(), key=lambda x: x[1], reverse=True)
            ]
            assert sorted(self.sorted_layers) == list(range(1, self.model.config.num_layers + 1))
        else:
            assert (
                self.constraints.keys() == {"export_config"}
                and "num_layers" not in self.constraints["export_config"]
            ), "Cannot prune `num_layers` without collecting layer scores!"
            self.sorted_layers = None

        if self.constraints.keys() & _METRIC_CONSTRAINTS:
            export_config = self.search_best_arch_by_metrics()
        else:
            export_config = self.constraints["export_config"]

        # Prune homogeneously
        self._prune(export_config, prune_depth=True)

        # TODO: Rename to hybrid_layer_pattern after MCore 0.17 and nemo:26.04 is released (for M-LM PR #3377)
        # Update hybrid_override_pattern if pruning is done on a hybrid model
        if isinstance(self.model, MambaModel):
            hybrid_key = (
                "hybrid_override_pattern"
                if hasattr(self.model, "hybrid_override_pattern")
                else "hybrid_layer_pattern"
            )
            print_rank_0(f"Original {hybrid_key}: {getattr(self.model, hybrid_key)}")
            new_num_layers = self.model.config.num_layers
            assert self.sorted_layers is not None
            kept_layers_numbers = self.sorted_layers[:new_num_layers]
            setattr(
                self.model,
                hybrid_key,
                "".join(
                    c
                    for i, c in enumerate(getattr(self.model, hybrid_key))
                    if i + 1 in kept_layers_numbers
                ),
            )
            print_rank_0(f"Pruned {hybrid_key}: {getattr(self.model, hybrid_key)}")

        print_mcore_model_stats(
            self.model, "Pruned Model", self.config["seq_length"], self.config["batch_size"]
        )

    def _prune(self, export_config: dict, prune_depth: bool = True) -> None:
        """Prune the model homogeneously based on the export_config by setting active choices for configurable hparams.

        Args:
            export_config: Dictionary mapping hyperparameter names to their pruned values.
            prune_depth: Whether to drop layers based on sorted_layers (default: True).
        """
        # Prune homogeneously
        for n, hp in named_hparams(self.model, configurable=True):
            hp_name = n.split(".")[-1]
            if hp_name in export_config:
                hp.active = export_config[hp_name]

        # Drop layers if depth pruning is enabled
        if prune_depth:
            num_layers_hp = self.model.get_hparam("num_layers")
            if num_layers_hp.active != num_layers_hp.max:
                assert self.sorted_layers is not None
                layers_to_drop = self.sorted_layers[num_layers_hp.active :]
                drop_mcore_language_model_layers(self.model, layers_to_drop=layers_to_drop)

        # Update model config with pruned architecture
        # kv_channels can be None so we need to save from original hidden_size and num_attention_heads
        if self.model.config.kv_channels is None:
            self.model.config.kv_channels = (
                self.model.config.hidden_size // self.model.config.num_attention_heads
            )
        # num_query_groups can be None so we need to save from original num_attention_heads
        if self.model.config.num_query_groups is None:
            self.model.config.num_query_groups = self.model.config.num_attention_heads
        # moe_ffn_hidden_size can be None so we need to save from original ffn_hidden_size
        if (
            self.model.config.moe_ffn_hidden_size is None
            and self.model.config.num_moe_experts is not None
        ):
            self.model.config.moe_ffn_hidden_size = self.model.config.ffn_hidden_size
        # Now set hparam active choices
        for hp_name, hp_value in export_config.items():
            setattr(self.model.config, hp_name, hp_value)

        # Reinitialize the MoE token dispatcher after pruning
        for m in self.model.modules():
            if isinstance(m, _DynamicMoELayer):
                m._export_reinit_token_dispatcher()

    def search_best_arch_by_metrics(self) -> dict:
        """Search for the best architecture based on the given metric constraint.

        Supports ``params``, ``active_params``, and ``memory_mb`` constraints.
        Performs a grid-search over the search space to find subnets fitting the constraint,
        then validates the top-k candidates using ``score_func`` (e.g. MMLU).

        Returns:
            export_config: Dictionary mapping hyperparameter names to their pruned values.
        """
        assert self.sorted_layers is not None
        # Ordered list of active metric keys; primary (first) is used for sorting/display.
        active_metric_keys = [k for k in _METRIC_CONSTRAINT_PRIORITY if k in self.constraints]
        primary_key = active_metric_keys[0]
        max_metrics: dict[str, float] = {k: float(self.constraints[k]) for k in active_metric_keys}  # type: ignore[arg-type]
        max_width_pruning = self.config["max_width_pruning"]
        max_depth_pruning = self.config["max_depth_pruning"]
        hparams_to_skip = self.config["hparams_to_skip"]
        top_k = self.config["top_k"]
        constraints_str = ", ".join(f"{self._fmt_metric(v, k)} {k}" for k, v in max_metrics.items())
        print_rank_0(f"\nSearching for the best pruned architecture under {constraints_str}...")

        # 1. Find available search space choices (across all PP ranks)
        hp_choices = {}
        for n, hp in named_hparams(self.model, configurable=True):
            hp_name = n.split(".")[-1]
            hp_choices[hp_name] = hp.choices
        pp_group = dist.DistributedProcessGroup(get_pipeline_model_parallel_group())
        hp_choices = dist.DistributedProcessGroup.get_dist_syncd_obj(
            hp_choices,
            pp_group,
            op=lambda all_pp_search_spaces: {
                k: v for d in all_pp_search_spaces for k, v in d.items()
            },
        )

        # 2. Perform grid-search over the search space to find subnets fitting all constraints
        constraints_cache_key = tuple((k, max_metrics[k]) for k in active_metric_keys)
        if constraints_cache_key not in self.all_candidates_per_constraint:
            max_num_layers = self.model.get_hparam("num_layers").max
            search_space_configs = MCoreMinitronSearcher._generate_search_space_combos(
                hp_choices,
                max_width_pruning,
                max_depth_pruning,
                hparams_to_skip,
            )
            selected = []
            for ss_config in tqdm(
                search_space_configs,
                desc="Finding all candidates fitting the constraints...",
                disable=not dist.is_master(),
            ):
                candidate_metrics = self._compute_candidate_metrics(ss_config, max_num_layers)
                if all(candidate_metrics[k] <= max_metrics[k] for k in active_metric_keys):
                    selected.append(
                        CandidateSubnet(
                            ss_config, {k: candidate_metrics[k] for k in active_metric_keys}, None
                        )
                    )
            assert len(selected) > 0, "No subnets found fitting the constraints!"
            print_rank_0(f"Found {len(selected)} candidates fitting the constraints!")
            self.all_candidates_per_constraint[constraints_cache_key] = sorted(
                selected, key=lambda x: x.metrics[primary_key], reverse=True
            )
            self.save_search_checkpoint(verbose=True)
        else:
            print_rank_0(f"\nUsing top {top_k} candidates from checkpoint")
        top_k_candidates = self.all_candidates_per_constraint[constraints_cache_key][:top_k]

        table = Table(title=f"Top {top_k} Candidates", show_header=True, header_style="bold")
        table.add_column("#", justify="right", style="dim", no_wrap=True)
        table.add_column("export_config", overflow="fold")
        for k in active_metric_keys:
            table.add_column(k, justify="right")
        for i, candidate in enumerate(top_k_candidates, 1):
            row = [str(i), rich_escape(str(candidate.ss_config))]
            row += [self._fmt_metric(candidate.metrics[k], k) for k in active_metric_keys]
            table.add_row(*row)
        _rprint(table)

        # 3. Optional Knowledge Distillation (KD) step for all top-k candidates
        _rprint(
            f"[yellow]\nSkipping optional Knowledge Distillation (KD) step for candidates as it is a manual step. "
            "As per the original paper (https://arxiv.org/pdf/2407.14679), ideally we need to perform a short "
            f"Knowledge Distillation on ~2B tokens for all top {top_k} candidates before evaluating the "
            "`score_func`, which will take a lot longer to prune, require splitting the pruning process into multiple "
            "stages and a lot more compute for pruning but can lead to better pruned model selection. If you are "
            f"interested to do this, you can take the top {top_k} candidates' `export_config` from the logs above and "
            "then export all models separately and perform Knowledge Distillation on each of them before evaluating "
            f"the `score_func`.\n[/yellow]"
        )

        # 4. Validate top-k candidates using the score_func and return the best subnet
        # WAR for Nemotron-3-Nano-30B-A3B-BF16. Disable expert bias during candidate eval to prevent in-place
        # __setattr__ on dynamically-sliced buffers from corrupting their shape (128 -> 120 elements).
        _routers_with_expert_bias = []
        for n, m in self.model.named_modules():
            if hasattr(m, "enable_expert_bias") and m.enable_expert_bias:
                print(
                    f"Temporarily disabling expert bias for {n} on rank {dist.rank()} for candidate evaluation..."
                )
                m.enable_expert_bias = False
                _routers_with_expert_bias.append(m)

        for candidate in tqdm(
            top_k_candidates,
            desc=f"Validating top {top_k} candidates on given score_func (this will take some time)...",
            disable=not dist.is_master(),
            smoothing=0.7,
        ):
            if candidate.score is None:  # not restored from checkpoint
                all_layers = self.model.decoder.layers
                start_layer_number = all_layers[0].layer_number

                self._prune(candidate.ss_config, prune_depth=True)
                candidate.score = self.eval_score(silent=False)
                self.save_search_checkpoint(verbose=False)

                # reset to max subnet and revert dropped layers
                sample(self.model, sample_func=max)
                for layer in all_layers:
                    layer.layer_number = start_layer_number
                    start_layer_number += 1
                self.model.decoder.layers = all_layers
            metrics_str = ", ".join(
                f"{self._fmt_metric(v, k)} {k}" for k, v in candidate.metrics.items()
            )
            print_rank_0(f"\t{candidate.ss_config} -> {metrics_str}, {candidate.score:.4f} score\n")

        for m in _routers_with_expert_bias:
            m.enable_expert_bias = True

        scored_table = Table(
            title=f"Top {top_k} Candidates with Scores", show_header=True, header_style="bold"
        )
        scored_table.add_column("#", justify="right", style="dim", no_wrap=True)
        scored_table.add_column("export_config")
        for k in active_metric_keys:
            scored_table.add_column(k, justify="right")
        scored_table.add_column("score", justify="right")
        for i, candidate in enumerate(top_k_candidates, 1):
            row = [str(i), rich_escape(str(candidate.ss_config))]
            row += [self._fmt_metric(candidate.metrics[k], k) for k in active_metric_keys]
            row.append(f"{candidate.score:.4f}")
            scored_table.add_row(*row)
        _rprint(scored_table)

        dist.barrier()
        best = max(top_k_candidates, key=lambda x: x.score)  # type: ignore[arg-type, return-value]
        best_grid = Table.grid(padding=(0, 2))
        best_grid.add_column(style="bold green", no_wrap=True)
        best_grid.add_column()
        best_grid.add_row("export_config", rich_escape(str(best.ss_config)))
        for k, v in best.metrics.items():
            best_grid.add_row(k, self._fmt_metric(v, k))
        best_grid.add_row("score", f"{best.score:.4f}")
        _rprint(
            Panel(best_grid, title="[bold green]Best Subnet[/bold green]", border_style="green")
        )
        return best.ss_config

    def _fmt_metric(self, value: float, constraint_key: str) -> str:
        """Format a metric value for display."""
        return f"{value:.3f} MB" if constraint_key == "memory_mb" else num2hrb(value)

    @staticmethod
    def _generate_search_space_combos(
        search_space: dict[str, list],
        max_width_pruning: float = 0.40,
        max_depth_pruning: float = 0.20,
        hparams_to_skip: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate all possible combinations of hyperparameters from the search space.

        Args:
            search_space: Dictionary mapping hyperparameter names to their possible sorted choices.
                        Example: {"hidden_size": [1024, 2048, 3072, 4096], "num_layers": [1, 2, ..., 31, 32]}
            max_width_pruning: Maximum fraction of width hyperparameters to prune (default: 0.40).
                            Only top (1 - max_width_pruning) choices will be considered.
            max_depth_pruning: Maximum fraction of depth hyperparameters to prune (default: 0.20).
                            Only top (1 - max_depth_pruning) choices will be considered.
            hparams_to_skip: List of hparams to skip during the search (default: None).

        Returns:
            List of configuration dictionaries, where each dictionary maps hyperparameter
            names to their chosen values. Example:
            [
                {"hidden_size": 1024, "num_layers": 1},
                {"hidden_size": 1024, "num_layers": 2},
                ...
                {"hidden_size": 4096, "num_layers": 32},
            ]
        """
        if hparams_to_skip:
            search_space = dict(search_space)  # Avoid modifying the original search space
            print_rank_0(f"Skipping {hparams_to_skip=} during search space generation...")
            for hparam in hparams_to_skip:
                if hparam in search_space:
                    search_space.pop(hparam)
                else:
                    warn(f"Hparam {hparam} not found in search space! Skipping...")

        filtered_ss = {
            k: (
                sorted(v)[int((1 - max_depth_pruning) * len(v)) :]
                if k == "num_layers"
                else sorted(v)[int((1 - max_width_pruning) * len(v)) :]
            )
            for k, v in search_space.items()
            if len(v) > 1
        }

        ss_size = 1
        table = Table(
            title=f"Search Space \n(≤{max_width_pruning * 100:.0f}% width / ≤{max_depth_pruning * 100:.0f}% depth pruning)",  # noqa: E501
            show_header=True,
            header_style="bold",
        )
        table.add_column("Hyperparameter")
        table.add_column("Choices", overflow="fold")
        for k, v in filtered_ss.items():
            table.add_row(k, rich_escape(str(v)))
            ss_size *= len(v)
        table.add_section()
        table.add_row("Search space size", f"{ss_size}")
        _rprint(table)

        hparam_names = list(filtered_ss.keys())
        hparam_choices_lists = [filtered_ss[name] for name in hparam_names]

        search_space_combos = [
            dict(zip(hparam_names, choices)) for choices in product(*hparam_choices_lists)
        ]
        assert len(search_space_combos) == ss_size

        return search_space_combos

    def _compute_candidate_metrics(self, ss_config: dict, max_num_layers: int) -> dict[str, float]:
        """Compute all active metric constraint values for a candidate config analytically.

        Handles depth pruning by filtering the hybrid layer pattern to the kept (best) layers.
        """
        model = self.model
        active_metric_keys = self.constraints.keys() & _METRIC_CONSTRAINTS

        # Get hybrid layer pattern for MambaModel (None for pure GPT)
        hybrid_layer_pattern: str | None = None
        if isinstance(model, MambaModel):
            hybrid_key = (
                "hybrid_override_pattern"
                if hasattr(self.model, "hybrid_override_pattern")
                else "hybrid_layer_pattern"
            )
            hybrid_layer_pattern = getattr(model, hybrid_key)

        # If depth pruning on a hybrid model, filter the pattern to only the kept layers.
        # sorted_layers gives layer numbers (1-indexed) ordered best-first; we keep the top N.
        num_layers_target: int = ss_config.get("num_layers", max_num_layers)
        if hybrid_layer_pattern is not None and num_layers_target < max_num_layers:
            assert self.sorted_layers is not None
            kept = set(self.sorted_layers[:num_layers_target])
            layer_chars = parse_main_layer_chars(hybrid_layer_pattern)
            hybrid_layer_pattern = "".join(c for i, c in enumerate(layer_chars) if (i + 1) in kept)

        metrics: dict[str, float] = {}

        if active_metric_keys & {"params", "active_params"}:
            total, active = mcore_param_count(
                model.config,
                model.vocab_size,
                model.share_embeddings_and_output_weights,
                hybrid_layer_pattern=hybrid_layer_pattern,
                **ss_config,
            )
            if "params" in active_metric_keys:
                metrics["params"] = total
            if "active_params" in active_metric_keys:
                metrics["active_params"] = active

        if "memory_mb" in active_metric_keys:
            _, _, _, metrics["memory_mb"] = mcore_memory_footprint_mb(
                model.config,
                model.vocab_size,
                model.share_embeddings_and_output_weights,
                hybrid_layer_pattern=hybrid_layer_pattern,
                dtype_bytes=2,  # assume BF16 input
                sequence_length=self.config["seq_length"],
                batch_size=self.config["batch_size"],
                **ss_config,
            )

        return metrics


MCoreMinitronConfig: type[ModeloptBaseConfig] = create_model(
    "MCoreMinitronConfig",
    **get_kwargs_for_create_model_with_rules(
        registry=DMRegistry,
        default_rules={
            "megatron.core.models.gpt.GPTModel": {
                "hidden_size_divisor": 256,
                "ffn_hidden_size_divisor": 512,
                "num_moe_experts_divisor": 8,
                "num_layers_divisor": 2,
            },
            **(
                {
                    "megatron.core.models.mamba.MambaModel": {
                        "hidden_size_divisor": 256,
                        "ffn_hidden_size_divisor": 512,
                        "mamba_head_dim_divisor": 8,
                        "num_moe_experts_divisor": 8,
                        "num_layers_divisor": 2,
                    }
                }
                if HAS_MAMBA
                else {}
            ),
        },
        doc='Configuration for the ``"mcore_minitron"`` mode.',
    ),
)


def get_mcore_minitron_config(
    *,
    hidden_size_divisor: int = 256,
    ffn_hidden_size_divisor: int = 512,
    mamba_head_dim_divisor: int = 8,
    num_moe_experts_divisor: int = 8,
    num_layers_divisor: int = 2,
) -> ModeloptBaseConfig:
    """Get a MCoreMinitronConfig with the given divisors instead of default."""
    config = MCoreMinitronConfig()

    def _set_divisors(c):
        for k, v in c.items():
            if isinstance(v, dict):
                _set_divisors(v)
            elif k == "hidden_size_divisor":
                c[k] = hidden_size_divisor
            elif k == "ffn_hidden_size_divisor":
                c[k] = ffn_hidden_size_divisor
            elif k == "mamba_head_dim_divisor":
                c[k] = mamba_head_dim_divisor
            elif k == "num_moe_experts_divisor":
                c[k] = num_moe_experts_divisor
            elif k == "num_layers_divisor":
                c[k] = num_layers_divisor

    _set_divisors(config)
    return config


def _convert_model_to_dynamic_space(
    model: nn.Module, config: ModeloptBaseConfig | None = None
) -> DynamicSpace:
    """Create a dynamic space for the model (in-place)."""
    dynamic_space = DynamicSpace(model)
    dynamic_space._should_be_converted = lambda mod: isinstance(mod, tuple(SUPPORTED_MODELS.keys()))
    dynamic_space.convert_to_dynamic(config.model_dump() if config else None, DMRegistry)
    if not dynamic_space.is_configurable():
        raise ApplyModeError(
            "The model does not contain any configurable hyperparameters! Please check the"
            " documentation for modules and config and how to get a configurable model."
        )

    return dynamic_space


def convert_mcore_minitron(model: nn.Module, config: ModeloptBaseConfig) -> ConvertReturnType:
    """Convert the model to the dynamic search space (in-place) and return the converted model and metadata.

    This is a simplified version of convert_fastnas_searchspace that removes the automated recursive tracing
    and instead directly converts the top-level model to a DynamicModule. Submodules should not need to be explicitly
    converted as that happens from the top-level model.
    """
    _convert_model_to_dynamic_space(model, config)

    # store current config in metadata
    metadata = {"subnet_config": get_subnet_config(model)}

    # return converted model as well as metadata
    return model, metadata


def restore_mcore_minitron(
    model: nn.Module, config: ModeloptBaseConfig, metadata: dict
) -> nn.Module:
    """Restore the model (no-op since we don't want to convert again which forces TP=1)."""
    return model


@NASModeRegistry.register_mode
@PruneModeRegistry.register_mode
class MCoreMinitronModeDescriptor(ModeDescriptor):
    """Class to describe the ``"mcore_minitron"`` mode.

    The properties of this mode can be inspected via the source code.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "mcore_minitron"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return MCoreMinitronConfig

    @property
    def next_modes(self) -> set[str] | None:
        """Modes that must immediately follow this mode."""
        return {"export_nas", "kd_loss", "quantize", "sparse_magnitude", "sparse_gpt"}

    @property
    def export_mode(self) -> str | None:
        """The mode that corresponds to the export mode of this mode."""
        return "export_nas"

    @property
    def search_algorithm(self) -> type[BaseSearcher]:
        """Specifies the search algorithm to use for this mode."""
        return MCoreMinitronSearcher

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model to a search space."""
        return convert_mcore_minitron

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model with the modelopt_state."""
        return restore_mcore_minitron


class ImportanceEstimatorRegistry:
    """Register importance estimators and forward hooks for all supported modules in the model.

    This class should be instantiated after converting the model to DynamicModule but before
    running the forward loop for importance estimation.
    """

    def __init__(self, model: DynamicModule):
        """Initialize the registry."""
        assert isinstance(model, _DynamicMCoreLanguageModel), "Model must be a DynamicModule"
        self.model = model
        self._hooks: list[tuple[nn.Module, Any]] = []  # List of (module, hook_handle) tuples

        print_rank_0("Registering importance estimators and forward hooks...")
        for module in self.model.modules():
            if isinstance(module, _DynamicMCoreLanguageModel):
                _register_hidden_size_importance(module, self)
            elif isinstance(module, (_DynamicTransformerLayer, _DynamicMambaLayer)):
                _register_depth_cosine_importance(module, self)
            elif isinstance(module, _DynamicSelfAttention):
                _register_self_attention_importance(module, self)
            elif isinstance(module, _DynamicMLP):
                _register_mlp_importance(module, self)
            elif isinstance(module, _DynamicSequentialMLP):
                _register_sequential_mlp_importance(module, self)
            elif isinstance(module, _DynamicMambaMixer):
                _register_mamba_mixer_importance(module, self)

    def register_hook(
        self,
        module: nn.Module,
        hook_fn: Callable,
        hook_type: str = "forward",
        **hook_kwargs,
    ) -> None:
        """Register a forward or forward_pre hook on a module.

        Args:
            module: The module to register the hook on.
            hook_fn: The hook function to register.
            hook_type: Type of hook ("forward" or "forward_pre").
            **hook_kwargs: Additional kwargs for hook registration.
        """
        if hook_type == "forward":
            handle = module.register_forward_hook(hook_fn, **hook_kwargs)
        elif hook_type == "forward_pre":
            handle = module.register_forward_pre_hook(hook_fn, **hook_kwargs)
        else:
            raise ValueError(f"Unsupported hook_type: {hook_type}")

        self._hooks.append((module, handle))

    def register_importance(
        self,
        dynamic_module: DynamicModule,
        hparam_name: str,
        importance_fn: Callable,
        importance_is_order: bool = False,
    ) -> None:
        """Register an importance estimator for a hyperparameter.

        Args:
            dynamic_module: The DynamicModule instance.
            hparam_name: Name of the hyperparameter to register importance for.
            importance_fn: Function that returns importance scores.
            importance_is_order: Whether the importance is a ranking order.
        """
        hp = dynamic_module.get_hparam(hparam_name)
        if importance_is_order:
            hp._importance_is_order = True
        hp.register_importance(importance_fn)

    def cleanup(self) -> None:
        """Remove all registered hooks and temporary attributes."""
        # Remove all hooks
        for _, handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        # Unpatch return_layernorm_output on fused TELayerNormColumnParallelLinear modules
        for m in self.model.modules():
            if isinstance(m, TELayerNormColumnParallelLinear):
                m.return_layernorm_output = False

    def get_layer_scores(self) -> dict[int, torch.Tensor]:
        """Get the layer scores (1-indexed) from the model.

        Returns:
            Dictionary mapping layer number to layer score.
        """
        num_layers_hp = self.model.get_hparam("num_layers")

        for layer in self.model.decoder.layers:
            assert layer._scores > 0, "No scores collected for importance estimation."

        # gather layer scores from all PP ranks
        layer_scores = {}
        for layer in self.model.decoder.layers:
            layer_scores[layer.layer_number] = layer._scores
        pp_group = dist.DistributedProcessGroup(get_pipeline_model_parallel_group())
        layer_scores = dist.DistributedProcessGroup.get_dist_syncd_obj(
            layer_scores,
            pp_group,
            op=lambda all_pp_layer_scores: {
                k: v for d in all_pp_layer_scores for k, v in d.items()
            },
        )
        print_rank_0(f"Layerwise scores (1-indexed, higher is better): {layer_scores}")
        assert sorted(layer_scores.keys()) == list(range(1, num_layers_hp.max + 1))  # type: ignore[arg-type]

        return layer_scores

    def get_local_activations_and_layer_scores(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[int, torch.Tensor]]:
        """Get this rank's local activations and global layer scores from the model.

        Each rank saves its own activations to its per-rank checkpoint file (no allgather needed).
        Layer scores are gathered across all PP ranks to produce a global ranking.
        """
        local_activations = {
            n: m._activations for n, m in self.model.named_modules() if hasattr(m, "_activations")
        }
        layer_scores = self.get_layer_scores()

        return local_activations, layer_scores

    def set_local_activations_and_layer_scores(
        self,
        local_activations: dict[str, torch.Tensor],
        layer_scores: dict[int, torch.Tensor],
    ) -> None:
        """Set the pre-computed layer_scores and local activations instead of running forward.

        Args:
            local_activations: Dict from module name to activations for this rank.
            layer_scores: Dict from layer_number (1-indexed) to score (global across all PP ranks).
        """
        print_rank_0("Loading activations and scores from per-rank checkpoint...")
        for layer in self.model.decoder.layers:
            layer._scores = layer_scores[layer.layer_number]
        for n, m in self.model.named_modules():
            if hasattr(m, "_activations"):
                m._activations = local_activations[n]


# Module-specific registration functions
def _register_hidden_size_importance(
    module: _DynamicMCoreLanguageModel, registry: ImportanceEstimatorRegistry
) -> None:
    """Register importance estimators for Language Model (GPT/Mamba) modules."""
    module._register_temp_attribute("_activations", {})

    def _collect_activations(mod, module_id, activations_tensor):
        """Accumulate activation importance scores for a given module."""
        activations_tensor = activations_tensor.to(torch.float32)
        activations = activations_tensor.abs().mean(dim=0)  # [batch_size, hidden_size]
        activations = activations.pow(2).sum(dim=0)
        if module_id not in mod._activations:
            mod._activations[module_id] = activations
        else:
            mod._activations[module_id] += (
                activations  # aggregate sum instead of mean of scores for simplicity
            )

    def _fused_ln_linear_forward_hook(mod, module_inner, input, output):
        """Hook on TELayerNormColumnParallelLinear with return_layernorm_output=True.

        Extracts the exact layernorm output from TE's fused kernel and restores
        the normal return format so downstream code is not affected.
        """
        # Output format with return_layernorm_output=True:
        #   te_return_bias=True:  MCore returns (linear_out, bias, ln_out)
        #   te_return_bias=False: MCore returns ((linear_out, ln_out), None)
        if module_inner.te_return_bias:
            linear_out, bias, ln_out = output
            fixed_output = (linear_out, bias)
        else:
            (linear_out, ln_out), bias = output
            fixed_output = (linear_out, bias)

        # Gather over all TP regions
        # NOTE: This is not used at the moment since we restrict to TP=1
        ln_out = gather_from_tensor_model_parallel_region(ln_out).detach()
        _collect_activations(mod, id(module_inner), ln_out)

        # Return the normal output format so downstream code (e.g. SelfAttention) is not affected
        return fixed_output

    def _layernorm_forward_hook(mod, module_inner, input, output):
        """Hook on separate layernorm modules (e.g. TENorm for MoE pre_mlp_layernorm)."""
        # Gather output [seq_len, batch_size, hidden_size] over all TP regions
        # NOTE: This is not used at the moment since we restrict to TP=1
        output = gather_from_tensor_model_parallel_region(output).detach()
        _collect_activations(mod, id(module_inner), output)

    def _estimate_hidden_size_importance(mod):
        """Return the activation magnitude-based importance of the hidden_size."""
        assert mod._activations, "No activations collected for importance estimation."
        # Convert squared sum to L2 norm over global batch size per hook
        aggregated_activations = [act.pow(0.5) for act in mod._activations.values()]
        activations = torch.stack(aggregated_activations).sum(dim=0)  # [hidden_size]

        # Reduce over all PP ranks
        activations = activations.clone()
        torch.distributed.all_reduce(activations, op=torch.distributed.ReduceOp.SUM)
        return activations

    # Register hooks to collect post-layernorm activations for hidden_size importance.
    # Layernorms are fused into TELayerNormColumnParallelLinear. We temporarily
    # patch return_layernorm_output=True so TE's fused kernel returns the layernorm output.
    # For MoE layers, pre_mlp_layernorm is a separate TENorm — use a regular forward hook.
    for m in module.modules():
        if isinstance(m, TELayerNormColumnParallelLinear):
            m.return_layernorm_output = True

    for layer in module.decoder.layers:
        if isinstance(layer, _DynamicTransformerLayer):
            if isinstance(layer.self_attention, _DynamicSelfAttention):
                # input_layernorm is fused into self_attention.linear_qkv
                registry.register_hook(
                    layer.self_attention.linear_qkv,
                    partial(_fused_ln_linear_forward_hook, module),
                    hook_type="forward",
                )

            if isinstance(layer.mlp, _DynamicMoELayer):
                # MoE layers have a separate pre_mlp_layernorm (TENorm, not IdentityOp)
                registry.register_hook(
                    layer.pre_mlp_layernorm,
                    partial(_layernorm_forward_hook, module),
                    hook_type="forward",
                )
            elif isinstance(layer.mlp, _DynamicMLP):
                # Dense MLP: pre_mlp_layernorm is fused into mlp.linear_fc1
                registry.register_hook(
                    layer.mlp.linear_fc1,
                    partial(_fused_ln_linear_forward_hook, module),
                    hook_type="forward",
                )
        elif isinstance(layer, _DynamicMambaLayer):
            # Mamba norm is fused into mixer.in_proj
            registry.register_hook(
                layer.mixer.in_proj,
                partial(_fused_ln_linear_forward_hook, module),
                hook_type="forward",
            )

    registry.register_importance(
        module, "hidden_size", lambda: _estimate_hidden_size_importance(module)
    )


def _register_depth_cosine_importance(
    module: _DynamicTransformerLayer | _DynamicMambaLayer, registry: ImportanceEstimatorRegistry
) -> None:
    """Register importance estimators for TransformerLayer and MambaLayer modules."""
    module._register_temp_attribute("_scores", 0.0)

    def _layer_imp_forward_hook(mod, module_inner, args, kwargs, output):
        """Hook to collect cosine similarity between input and output to rank layers for depth pruning."""
        hidden_states = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]

        if isinstance(mod, _DynamicTransformerLayer):
            output, _ = output  # [seq_len, batch_size, hidden_size]

        # use full precision to avoid overflow
        hidden_states = hidden_states.to(torch.float32)
        output = output.to(torch.float32)

        with torch.no_grad():
            # Lower cosine_similarity means higher importance hence use 1 - cosine_similarity
            score = 1 - F.cosine_similarity(hidden_states, output, dim=2).mean()
            # TODO: Check if we need to reduce over TP regions (seems like all TP have same scores anyway)
            global_score = reduce_from_tensor_model_parallel_region(score).item()
            mod._scores += global_score  # aggregate sum instead of mean of scores for simplicity

    registry.register_hook(
        module,
        partial(_layer_imp_forward_hook, module),
        hook_type="forward",
        with_kwargs=True,
    )


def _register_self_attention_importance(
    module: _DynamicSelfAttention, registry: ImportanceEstimatorRegistry
) -> None:
    """Register importance estimators for SelfAttention modules."""
    module._register_temp_attribute("_activations", None)

    def _linear_proj_forward_hook(mod, module_inner, input, output):
        """Hook to collect activations for importance estimation.

        Activations are computed as mean over seq_len and then squared and summed over batch_size.
        Later we take the square root of the sum to get the L2 norm.
        """
        # Gather input [seq_len, batch_size, query_projection_size] over all TP regions
        # NOTE: This is not used at the moment since we restrict to TP=1
        input = gather_from_tensor_model_parallel_region(input[0]).detach()

        input = input.to(torch.float32)  # use full precision to avoid overflow
        activations = input.abs().mean(dim=0)
        activations = activations.pow(2).sum(dim=0)  # [query_projection_size]
        if mod._activations is None:
            mod._activations = activations
        else:
            mod._activations += activations

    def _estimate_head_ranking(mod):
        """Return the importance for num_attention_heads."""
        assert mod._activations is not None, "No activations collected for importance estimation."
        n_groups = mod.num_query_groups_per_partition
        max_nheads = mod.get_hparam("num_attention_heads").max
        max_heads_per_group = max_nheads // n_groups

        # Convert squared sum to L2 norm
        scores = mod._activations.pow(0.5).view(max_nheads, mod.config.kv_channels).cpu()
        attn_head_importance = torch.linalg.vector_norm(scores, ord=2, dim=1)
        # group_importance = torch.linalg.vector_norm(scores, ord=2, dim=(0, 2))

        # Convert to global indices by adding offset
        attn_head_ranking_per_group = attn_head_importance.view(
            n_groups, max_heads_per_group
        ).argsort(dim=1, descending=True)

        attn_head_ranking_global = (
            attn_head_ranking_per_group + torch.arange(0, max_nheads, max_heads_per_group)[:, None]
        ).flatten()
        assert torch.equal(attn_head_ranking_global.sort().values, torch.arange(max_nheads))
        # Return group-aware ranking of all attention heads
        # Actual group-aware trimming happens in NumAttentionHeadsHp
        return attn_head_ranking_global

    registry.register_hook(
        module.linear_proj,
        partial(_linear_proj_forward_hook, module),
        hook_type="forward",
    )
    # [HACK] Return ranking (group-aware) instead of importance for sort_parameters()
    # NOTE: Trimming should also happen within each group. This is handled in NumAttentionHeadsHp
    registry.register_importance(
        module,
        "num_attention_heads",
        lambda: _estimate_head_ranking(module),
        importance_is_order=True,
    )


def _register_mlp_importance(module: _DynamicMLP, registry: ImportanceEstimatorRegistry) -> None:
    """Register importance estimators for MLP modules."""
    module._register_temp_attribute("_activations", None)

    def _linear_fc2_forward_hook(mod, module_inner, input, output):
        """Hook to collect activations for importance estimation.

        Activations are computed as mean over seq_len and then squared and summed over batch_size.
        Later we take the square root of the sum to get the L2 norm.
        """
        # Gather input [seq_len, batch_size, ffn_hidden_size] over all TP regions
        # NOTE: This is not used at the moment since we restrict to TP=1
        input = gather_from_tensor_model_parallel_region(input[0]).detach()
        if input.dim() == 2:
            # For sparse experts, there is no batch dimension.
            input = input[:, None, :]

        input = input.to(torch.float32)  # use full precision to avoid overflow
        activations = input.abs().mean(dim=0)  # [batch_size, ffn_hidden_size]
        activations = activations.pow(2).sum(dim=0)  # [ffn_hidden_size]
        if mod._activations is None:
            mod._activations = activations
        else:
            mod._activations += activations

    def _estimate_importance(mod):
        """Return the activation magnitude-based importance of the ffn_hidden_size."""
        assert mod._activations is not None, "No activations collected for importance estimation."
        # Convert squared sum to L2 norm
        return mod._activations.pow(0.5)

    registry.register_hook(
        module.linear_fc2, partial(_linear_fc2_forward_hook, module), hook_type="forward"
    )
    registry.register_importance(module, module.hparam_name, lambda: _estimate_importance(module))


def _register_sequential_mlp_importance(
    module: _DynamicSequentialMLP, registry: ImportanceEstimatorRegistry
) -> None:
    """Register importance estimators for SequentialMLP (MoE experts) modules."""
    module._register_temp_attribute(
        "_activations",
        {
            "expert_l2_scores": torch.zeros(module.num_local_experts),
            "expert_sample_counts": torch.zeros(module.num_local_experts),
        },
    )

    def _expert_l2_imp_forward_hook(mod, module_inner, input, output):
        """Track expert importance based on L2 norms of expert outputs."""
        # Split output back to per-expert outputs using torch.split
        tokens_per_expert_list = input[1].tolist()
        # use full precision to avoid overflow
        output_local = output[0].to(torch.float32).detach()
        output_local_list = torch.split(output_local, tokens_per_expert_list)

        # Compute L2 norm for each expert's output
        for expert_idx, expert_output in enumerate(output_local_list):
            # Guard: if expert_output is empty tensor, add zero score
            if expert_output.numel() == 0:
                l2_norm = 0.0
            else:
                # Compute L2 norm of expert output (router_prob * expert_output)
                l2_norm = torch.linalg.vector_norm(expert_output, ord=2, dim=-1).sum().item()

            # Accumulate L2 scores and sample counts
            mod._activations["expert_l2_scores"][expert_idx] += l2_norm
            mod._activations["expert_sample_counts"][expert_idx] += tokens_per_expert_list[
                expert_idx
            ]

    def _estimate_expert_importance(mod):
        """Estimate expert importance based on accumulated L2 norms."""
        assert mod._activations["expert_sample_counts"].sum() > 0, (
            "No activations collected for importance estimation."
        )
        # Average L2 scores across samples (avoid division by zero if some experts have no samples)
        return mod._activations["expert_l2_scores"] / (
            mod._activations["expert_sample_counts"] + 1e-8
        )

    registry.register_hook(
        module,
        partial(_expert_l2_imp_forward_hook, module),
        hook_type="forward",
    )
    registry.register_importance(
        module,
        "num_local_experts",
        lambda: _estimate_expert_importance(module),
    )


def _register_mamba_mixer_importance(
    module: _DynamicMambaMixer, registry: ImportanceEstimatorRegistry
) -> None:
    """Register importance estimators for MambaMixer modules."""
    module._register_temp_attribute("_activations", None)

    def _mamba_in_proj_forward_hook(mod, module_inner, input, output):
        """Hook to collect activations for importance estimation.

        Activations are computed as mean over seq_len and then squared and summed over batch_size.
        Later we take the square root of the sum to get the L2 norm.
        """
        # Gather output [seq_len, batch_size, d_inner] over all TP regions
        # NOTE: This is not used at the moment since we restrict to TP=1
        output = gather_from_tensor_model_parallel_region(output[0]).detach()

        output = output.to(torch.float32)  # use full precision to avoid overflow
        activations = output.abs().mean(dim=0)
        activations = activations.pow(2).sum(dim=0)  # [d_inner]
        if mod._activations is None:
            mod._activations = activations
        else:
            mod._activations += activations

    def _estimate_head_and_head_dim_rankings(mod):
        """Get the rankings of Mamba heads and head dimensions."""
        # Convert squared sum to L2 norm
        scores = mod._activations.pow(0.5)
        assert scores is not None, "No activations collected for importance estimation."

        max_nheads = mod.get_hparam("mamba_num_heads").max
        max_headdim = mod.get_hparam("mamba_head_dim").max
        max_d_inner = mod.get_hparam("d_inner").max
        target_headdim = mod.headdim
        nheads_per_group = max_nheads // mod.ngroups

        # While there can be many ways of computing the ranking out of z, x, and dt,
        # based on ablations in the paper, using `x` is the best way to compute the ranking.
        x_indices = torch.arange(max_d_inner, 2 * max_d_inner)
        scores_x = scores[x_indices]  # shape = [max_d_inner] i.e. [max_nheads * max_headdim]

        # Get ranking of all head and target head dimensions (same for each head)
        all_head_dim_importance = torch.linalg.vector_norm(  # shape = [max_headdim]
            scores_x.view(max_nheads, max_headdim), ord=2, dim=0
        )
        all_head_dim_ranking = all_head_dim_importance.argsort(descending=True).cpu()
        target_head_dim_ranking = all_head_dim_ranking[:target_headdim]

        # Get ranking of all heads with target head dimensions
        target_head_dim_indices_per_head = torch.cat(  # shape = [max_nheads * target_headdim]
            [i * max_headdim + target_head_dim_ranking for i in range(max_nheads)]
        )

        # Get ranking of heads (sorted within their group)
        groupwise_head_importance = torch.linalg.vector_norm(  # shape = [ngroups, nheads_per_group]
            scores_x[target_head_dim_indices_per_head].view(
                mod.ngroups, nheads_per_group, target_headdim
            ),
            ord=2,
            dim=2,
        )
        groupwise_head_ranking = groupwise_head_importance.argsort(dim=1, descending=True).cpu()
        group_offsets = torch.arange(mod.ngroups).unsqueeze(1) * nheads_per_group
        all_head_ranking = (groupwise_head_ranking + group_offsets).flatten()

        return all_head_ranking, all_head_dim_ranking

    registry.register_hook(
        module.in_proj,
        partial(_mamba_in_proj_forward_hook, module),
        hook_type="forward",
    )
    # [HACK] Return ranking (group-aware) instead of importance for sort_parameters()
    # NOTE: Trimming should also happen within each group. This is handled in MambaNumHeadsHp.
    registry.register_importance(
        module,
        "mamba_num_heads",
        lambda: _estimate_head_and_head_dim_rankings(module)[0],
        importance_is_order=True,
    )
    registry.register_importance(
        module,
        "mamba_head_dim",
        lambda: _estimate_head_and_head_dim_rankings(module)[1],
        importance_is_order=True,
    )
