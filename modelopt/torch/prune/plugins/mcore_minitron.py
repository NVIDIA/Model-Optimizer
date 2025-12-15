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
from warnings import warn

import torch
import torch.nn as nn
from pydantic import create_model

# isort: off
# import nas plugin to check if it is enabled else raises an Exception and disables the plugin
from modelopt.torch.nas.plugins.megatron import *  # noqa: F403
from modelopt.torch.nas.plugins.megatron import (
    HAS_MAMBA,
    _DynamicMCoreLanguageModel,
    SUPPORTED_MODELS,
    drop_mcore_language_model_layers,
)
# isort: on

from modelopt.torch.nas.conversion import NASModeRegistry
from modelopt.torch.nas.registry import DMRegistry
from modelopt.torch.nas.utils import get_subnet_config, sort_parameters
from modelopt.torch.opt.config import ModeloptBaseConfig, get_kwargs_for_create_model_with_rules
from modelopt.torch.opt.conversion import ApplyModeError
from modelopt.torch.opt.dynamic import DynamicSpace
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ConvertReturnType,
    ModeDescriptor,
    RestoreEntrypoint,
)
from modelopt.torch.opt.searcher import BaseSearcher, SearchConfig, SearchStateDict
from modelopt.torch.opt.utils import named_hparams
from modelopt.torch.utils import print_rank_0

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
]


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

        if self.layer_scores and self.activations_per_rank:  # Available from checkpoint
            print_rank_0("Loading activations and scores per rank from checkpoint...")
            unwrapped_model.set_activations_and_layer_scores(
                self.activations_per_rank, self.layer_scores
            )
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
                unwrapped_model.get_activations_and_layer_scores()
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
                        "mamba_num_heads_divisor": 4,
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
