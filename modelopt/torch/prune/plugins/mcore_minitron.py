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

"""Module implementing top-level ``mcore_minitron`` pruning handler for NVIDIA Megatron-Core / NeMo models.

Minitron pruning algorithm uses activation magnitudes to estimate importance of neurons / attention heads / mamba heads
in the model.
More details on Minitron pruning algorithm can be found here: https://arxiv.org/pdf/2407.14679

Supports both GPT (attention-based) and Mamba (state-space) models, as well as hybrid models with both types of layers.

Actual dynamic module implementations are at :mod:`modelopt.torch.nas.plugins.megatron`.
"""

import copy
from collections.abc import Callable
from functools import partial
from typing import Any
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from modelopt.torch.nas.conversion import NASModeRegistry
from modelopt.torch.nas.plugins.megatron import (
    HAS_MAMBA,
    SUPPORTED_MODELS,
    _DynamicMambaLayer,
    _DynamicMambaMixer,
    _DynamicMCoreLanguageModel,
    _DynamicMLP,
    _DynamicSelfAttention,
    _DynamicSequentialMLP,
    _DynamicTransformerLayer,
)
from modelopt.torch.nas.registry import DMRegistry
from modelopt.torch.nas.utils import get_subnet_config, sort_parameters
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
from modelopt.torch.utils import get_module_device, print_rank_0

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
    print_rank_0(f"Dropping layers {layers_to_drop} from {n} ({type(model)}).")

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
    layer_offset = sum(layers_remaining_per_pp[: get_pipeline_model_parallel_rank()])
    layer_number = layer_offset + 1
    dropped_layers = []
    for layer in model.decoder.layers:
        if layer.layer_number in layers_to_drop:
            layer.layer_number = -1  # should not be used
            # layer.sharded_state_dict = lambda prefix, sharded_offsets, metadata: {}
            dropped_layers.append(layer)
        else:
            layer.layer_number = layer_number
            layer.get_transformer_layer_offset = lambda: layer_offset
            layer_number += 1

    # remove dropped layers from the modulelist
    model.decoder.layers = nn.ModuleList(
        [layer for layer in model.decoder.layers if layer.layer_number != -1]
    )
    for layer in dropped_layers:
        del layer

    model.config.num_layers = new_num_layers


class MCoreMinitronSearcher(BaseSearcher):
    """Searcher for Minitron pruning algorithm."""

    activations_per_rank: list[dict[str, torch.Tensor]]
    layer_scores: dict[int, torch.Tensor]

    @property
    def default_search_config(self) -> SearchConfig:
        """Get the default config for the searcher."""
        return {
            **super().default_search_config,
            "max_iter_data_loader": 1024,
            "skip_sorting": False,
            "scores_path": None,
        }

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Return default state dict for importance scores and activations from forward loop."""
        return {"activations_per_rank": [], "layer_scores": {}}

    def sanitize_search_config(self, config: SearchConfig | None) -> SearchConfig:
        """Sanitize the search config dict."""
        config = super().sanitize_search_config(config)
        config["checkpoint"] = config["scores_path"]
        config["verbose"] = True  # Print for all ranks
        return config

    def before_search(self) -> None:
        """Optional pre-processing steps before the search."""
        super().before_search()

        # Check that the constraint is valid
        assert self.constraints.keys() == {"export_config"}, (
            "Only `export_config` constraint is supported for pruning!"
        )

        self.constraints["export_config"] = copy.deepcopy(self.constraints["export_config"])
        export_config = self.constraints["export_config"]
        if "num_query_groups" in export_config:
            warn("num_query_groups is no longer supported (since 0.41)! It will be ignored.")
            if export_config["num_query_groups"] != self.model.config.num_query_groups:  # type: ignore[index]
                raise ValueError(f"num_query_groups must be {self.model.config.num_query_groups}!")
            export_config.pop("num_query_groups")  # type: ignore[union-attr]
        assert isinstance(export_config, dict)  # to keep mypy happy
        assert export_config.keys() <= SUPPORTED_HPARAMS, (
            f"Only {SUPPORTED_HPARAMS} are supported for pruning! Received: {export_config.keys()}"
        )

        # Only sort the parameters that are to be pruned
        # If a user only prunes depth, we should not sort width parameters
        self.hps_to_sort = SUPPORTED_HPARAMS & export_config.keys()

        for n, hp in named_hparams(self.model, unique=True):
            hp_name = n.split(".")[-1]
            if hp.is_configurable:
                # Make sure configurable hparams are the ones with right names else implementation needs to be fixed!
                assert hp_name in SUPPORTED_HPARAMS, f"[ImplError] Invalid hparam {hp_name}!"
                if hp_name in export_config:
                    assert export_config[hp_name] in hp.choices, (
                        f"Invalid choice {export_config[hp_name]} for {n}! Available choices: {hp.choices}"
                    )
            hp.reset_choices()  # Make sure ConcatHparam choices are updated after modify()

    def run_search(self) -> None:
        """Run actual search."""
        # Run forward loop to collect activations and sort parameters
        unwrapped_model = self.model
        for m in self.model.modules():
            if isinstance(m, _DynamicMCoreLanguageModel):
                unwrapped_model = m
                break
        assert isinstance(unwrapped_model, _DynamicMCoreLanguageModel), "Model not supported!"

        registry = ImportanceEstimatorRegistry(unwrapped_model)
        if self.layer_scores and self.activations_per_rank:  # Available from checkpoint
            print_rank_0("Loading activations and scores per rank from checkpoint...")
            registry.set_activations_and_layer_scores(self.activations_per_rank, self.layer_scores)
        elif not self.config["skip_sorting"]:
            print_rank_0("Running forward loop...")
            assert self.forward_loop is not None
            is_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                self.forward_loop(self.model)
            self.model.train(is_training)

            # Store activations and layer scores for re-pruning with different export configs
            self.activations_per_rank, self.layer_scores = (
                registry.get_activations_and_layer_scores()
            )
            self.save_search_checkpoint(verbose=True)

        if self.config["skip_sorting"]:
            print_rank_0("Skipping sorting parameters...")
        else:
            sort_parameters(self.model, self.hps_to_sort, verbose=True)

        # Prune homogeneously
        export_config = self.constraints["export_config"]
        assert isinstance(export_config, dict)  # to keep mypy happy
        for n, hp in named_hparams(self.model, configurable=True):
            hp_name = n.split(".")[-1]
            if hp_name in export_config:
                hp.active = export_config[hp_name]

        # Drop layers if depth pruning is enabled
        num_layers_hp = unwrapped_model.get_hparam("num_layers")
        if num_layers_hp.active != num_layers_hp.max:
            # sort layers by scores and drop the lowest ones
            sorted_layers = sorted(self.layer_scores.items(), key=lambda x: x[1], reverse=True)
            layers_to_drop = [layer for layer, _ in sorted_layers[num_layers_hp.active :]]  # type: ignore[misc]
            drop_mcore_language_model_layers(self.model, layers_to_drop=layers_to_drop)

        # kv_channels can be None so we need to save original from original hidden_size and num_attention_heads
        model_cfg = self.model.config
        orig_kv_channels = getattr(model_cfg, "kv_channels")
        if orig_kv_channels is None:
            orig_kv_channels = getattr(model_cfg, "hidden_size") // getattr(
                model_cfg, "num_attention_heads"
            )
        setattr(model_cfg, "kv_channels", orig_kv_channels)
        for n in SUPPORTED_HPARAMS:
            if n in export_config:
                setattr(model_cfg, n, export_config[n])

        registry.cleanup()


MCoreMinitronConfig: type[ModeloptBaseConfig] = create_model(
    "MCoreMinitronConfig",
    **get_kwargs_for_create_model_with_rules(
        registry=DMRegistry,
        default_rules={
            "megatron.core.models.gpt.GPTModel": {
                "hidden_size_divisor": 64,
                "ffn_hidden_size_divisor": 64,
                "num_moe_experts_divisor": 1,
            },
            **(
                {
                    "megatron.core.models.mamba.MambaModel": {
                        "hidden_size_divisor": 64,
                        "ffn_hidden_size_divisor": 64,
                        "mamba_head_dim_divisor": 4,
                        "num_moe_experts_divisor": 1,
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
    channel_divisor: int = 64,
    mamba_head_dim_divisor: int = 4,
    num_moe_experts_divisor: int = 1,
) -> ModeloptBaseConfig:
    """Get a MCoreMinitronConfig with the given channel divisor instead of default."""
    config = MCoreMinitronConfig()

    def _set_divisors(c):
        for k, v in c.items():
            if isinstance(v, dict):
                _set_divisors(v)
            elif k in ["hidden_size_divisor", "ffn_hidden_size_divisor"]:
                c[k] = channel_divisor
            elif k == "mamba_head_dim_divisor":
                c[k] = mamba_head_dim_divisor
            elif k == "num_moe_experts_divisor":
                c[k] = num_moe_experts_divisor

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
        all_pp_layer_scores = [None] * get_pipeline_model_parallel_world_size()
        torch.distributed.all_gather_object(
            all_pp_layer_scores, layer_scores, group=get_pipeline_model_parallel_group()
        )
        layer_scores = {k: v for d in all_pp_layer_scores for k, v in d.items()}  # type: ignore[attr-defined]
        print_rank_0(f"Layerwise scores (1-indexed, higher is better): {layer_scores}")
        assert sorted(layer_scores.keys()) == list(range(1, num_layers_hp.max + 1))  # type: ignore[arg-type]

        return layer_scores

    def get_activations_and_layer_scores(
        self,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[int, torch.Tensor]]:
        """Get the per-rank activations and layer scores from the model."""
        local_activations = {}
        for n, m in self.model.named_modules():
            if hasattr(m, "_activations"):
                local_activations[n] = m._activations
        activations_per_rank = dist.allgather(
            local_activations, group=get_pipeline_model_parallel_group()
        )
        assert len(activations_per_rank) == get_pipeline_model_parallel_world_size()

        layer_scores = self.get_layer_scores()

        return activations_per_rank, layer_scores

    def set_activations_and_layer_scores(
        self,
        activations_per_rank: list[dict[str, torch.Tensor]],
        layer_scores: dict[int, torch.Tensor],
    ) -> None:
        """Set the pre-computed layer_scores and per-rank activations instead of running forward.

        Args:
            activations_per_rank: List of dicts from module name to activations. Should match PP size.
            layer_scores: Dict from layer_number (1-indexed) to score.
        """
        rank = get_pipeline_model_parallel_rank()
        pp_size = get_pipeline_model_parallel_world_size()
        assert len(activations_per_rank) == pp_size, (
            f"Expected same PP size for stored pruning scores ({len(activations_per_rank)}) as current ({pp_size})!"
        )
        for layer in self.model.decoder.layers:
            layer._scores = layer_scores[layer.layer_number]
        for n, m in self.model.named_modules():
            if hasattr(m, "_activations"):
                m._activations = activations_per_rank[rank][n]


# Module-specific registration functions
def _register_hidden_size_importance(
    module: _DynamicMCoreLanguageModel, registry: ImportanceEstimatorRegistry
) -> None:
    """Register importance estimators for Language Model (GPT/Mamba) modules."""
    module._register_temp_attribute("_activations", {})

    def _emb_layernorm_forward_hook(mod, module_inner, input, output):
        """Hook to collect activations for importance estimation.

        Activations are computed as mean over seq_len and then squared and summed over batch_size.
        Later we take the square root of the sum to get the L2 norm.
        """
        # Gather output [seq_len, batch_size, hidden_size] over all TP regions
        # NOTE: This is not used at the moment since we restrict to TP=1
        output = gather_from_tensor_model_parallel_region(output).detach()

        output = output.to(torch.float32)  # use full precision to avoid overflow
        activations = output.abs().mean(dim=0)  # [batch_size, hidden_size]
        activations = activations.pow(2).sum(dim=0)
        if id(module_inner) not in mod._activations:
            mod._activations[id(module_inner)] = activations
        else:
            mod._activations[id(module_inner)] += (
                activations  # aggregate sum instead of mean of scores for simplicity
            )

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

    # Register hooks for all layers
    for layer in module.decoder.layers:
        if isinstance(layer, _DynamicTransformerLayer):
            if isinstance(layer.self_attention, _DynamicSelfAttention):
                registry.register_hook(
                    layer.input_layernorm,
                    partial(_emb_layernorm_forward_hook, module),
                    hook_type="forward",
                )

            if isinstance(layer.mlp, (_DynamicMLP, _DynamicSequentialMLP)):
                registry.register_hook(
                    layer.pre_mlp_layernorm,
                    partial(_emb_layernorm_forward_hook, module),
                    hook_type="forward",
                )
        elif isinstance(layer, _DynamicMambaLayer):
            registry.register_hook(
                layer.norm, partial(_emb_layernorm_forward_hook, module), hook_type="forward"
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
