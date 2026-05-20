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

"""Module implementing ``fasnas`` pruning algorithm for search."""

import warnings
from collections.abc import Callable
from typing import Any

import torch.nn as nn
import tqdm
from pydantic import create_model

from modelopt.torch.nas.autonas import (
    AutoNASPatchManager,
    IterativeSearcher,
    convert_searchspace,
    restore_searchspace,
    update_autonas_metadata,
)
from modelopt.torch.nas.conversion import NASModeRegistry
from modelopt.torch.nas.hparams import ConcatTracedHp
from modelopt.torch.nas.registry import DMRegistry
from modelopt.torch.nas.utils import get_subnet_config, sample, select, sort_parameters
from modelopt.torch.opt.config import ModeloptBaseConfig, get_kwargs_for_create_model_with_rules
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ConvertReturnType,
    MetadataDict,
    ModeDescriptor,
    RestoreEntrypoint,
    UpdateEntrypoint,
)
from modelopt.torch.opt.searcher import BaseSearcher, SearchStateDict
from modelopt.torch.opt.utils import named_hparams
from modelopt.torch.utils import random

from .pruning import PruneModeRegistry

ConstraintsRes = dict[str, float]
ConstraintEvalFunc = Callable[[ConstraintsRes | None], float]


class FastNASPatchManager(AutoNASPatchManager):
    """A patch manager for FastNAS (same as AutoNAS except no sampling during training)."""

    @property
    def sample_during_training(self):
        """Indicates whether we should sample a new subnet during training."""
        return False


update_fastnas_metadata = update_autonas_metadata


def convert_fastnas_searchspace(model: nn.Module, config: ModeloptBaseConfig) -> ConvertReturnType:
    """Convert search space for FastNAS mode with correct patch manager."""
    return convert_searchspace(model, config, FastNASPatchManager)


def restore_fastnas_searchspace(
    model: nn.Module, config: ModeloptBaseConfig, metadata: MetadataDict
) -> nn.Module:
    """Restore search space for FastNAS mode with correct patch manager."""
    return restore_searchspace(model, config, metadata, FastNASPatchManager)


class BinarySearcher(IterativeSearcher):
    """An iterative searcher that uses binary search to find the best configuration."""

    sensitivity_map: dict[str, dict[int, float]]
    min_degrade: float
    max_degrade: float
    middle_value: float
    original_score: float

    @property
    def hparam_names_for_search(self) -> set[str]:
        """We can only optimize over certain types of hparams in binary search."""
        # TODO: eventually improve this!
        return {"out_channels", "out_features", "num_attention_heads"}

    @property
    def hparam_types_for_search(self) -> tuple[type]:
        """We can only optimize over certain types of hparams in binary search."""
        # TODO: eventually improve this!
        return (ConcatTracedHp,)

    def before_search(self) -> None:
        """Build sensitivity map before search that we use to approximate the cost function."""
        # run super method
        super().before_search()

        # sort parameters in model
        sort_parameters(self.model)

        # for binary search, we currently only support 1 constraint
        assert len(self.constraints_func.effective_limits) == 1, (
            f"Currently, we only support one constraint for `{type(self).__name__}`, but got"
            f" {len(self.constraints_func.effective_limits)}"
        )

        # compute and register the construction of sensitivity map
        self._build_sensitivity_map(self.config["verbose"])
        self.max_degrade = max([max(v.values()) for v in self.sensitivity_map.values()])
        self.min_degrade = min([min(v.values()) for v in self.sensitivity_map.values()])

        # overwrite the score function to be a fake function, returning the -max degrade
        def max_degrade(_model):
            return -max(
                self.sensitivity_map[k][active]
                for k, active in get_subnet_config(_model, True).items()
                if k in self.sensitivity_map
            )

        self.config["score_func"] = max_degrade

    def before_step(self) -> None:
        """Check what the middle value is to determine where we recurse."""
        self.middle_value = 0.5 * (self.max_degrade + self.min_degrade)

    def _apply_fastnas_according_to_threshold(self, threshold):
        cfg = {
            name: min([k for k, v in sensitivity.items() if v <= threshold])
            for name, sensitivity in self.sensitivity_map.items()
        }
        select(self.model, cfg, strict=False)

    def sample(self) -> dict[str, Any]:
        """Check in which interval we should recurse and sets the corresponding subnet."""
        self._apply_fastnas_according_to_threshold(self.middle_value)
        return {"config": get_subnet_config(self.model)}

    def after_step(self) -> None:
        """Update boundaries of the interval after recursing."""
        if self.candidate["is_satisfied"]:
            self.min_degrade, self.max_degrade = self.min_degrade, self.middle_value
        else:
            self.min_degrade, self.max_degrade = self.middle_value, self.max_degrade

    def early_stop(self) -> bool:
        """Early stop if the interval is small enough."""
        return abs(0.5 * (self.min_degrade + self.max_degrade) - self.middle_value) < 1e-5

    @property
    def default_state_dict(self) -> SearchStateDict:
        """We also store the sensitivity map and related arguments."""
        return {
            **super().default_state_dict,
            "sensitivity_map": {},
            "middle_value": None,
            "min_degrade": None,
            "max_degrade": None,
            "original_score": None,
        }

    def load_search_checkpoint(self) -> bool:
        """We only want to load sensitivity map and original_score here and keep the rest."""
        state_dict = self.state_dict()
        keys_to_load = ("sensitivity_map", "original_score")
        is_loaded = super().load_search_checkpoint()
        if is_loaded:
            for k, v in state_dict.items():
                if k not in keys_to_load:
                    setattr(self, k, v)
        return is_loaded

    def _get_binary_search_hps(self):
        hps_configurable = dict(named_hparams(self.model, configurable=True))

        # TODO: eventually improve this!
        binary_search_hps = {
            k: hp
            for k, hp in hps_configurable.items()
            if k.split(".")[-1] in self.hparam_names_for_search
            or isinstance(hp, self.hparam_types_for_search)
        }

        # Sanity check if there are other searchable hparams not supported by the searcher.
        if len(binary_search_hps) < len(hps_configurable):
            warnings.warn(
                f"`{type(self).__name__}` does not support search for"
                f" {hps_configurable.keys() - binary_search_hps.keys()}"
            )

        return binary_search_hps

    def _build_sensitivity_map(self, verbose=False) -> None:
        if self.original_score is None:
            sample(self.model, random.original)
            self.original_score = self.eval_score()

        binary_search_hps = self._get_binary_search_hps()

        # Measure sensitivity

        # Getting the number of choices needed to validate
        total_choices_to_validate = sum(
            [len(hparam.choices) for hparam in binary_search_hps.values()]
        )

        assert total_choices_to_validate != 0, f"{type(self).__name__}: no searchable hparams found"

        binary_search_hps = {
            k: hp
            for k, hp in binary_search_hps.items()
            if (
                k not in self.sensitivity_map
                or sorted(hp.choices) != sorted(self.sensitivity_map[k].keys())
            )
        }

        remaining_choices_to_validate = sum(
            [len(hparam.choices) for hparam in binary_search_hps.values()]
        )

        if remaining_choices_to_validate == 0:
            return

        print(
            "\nBeginning pre-search estimation. If the runtime of score function is longer than",
            "a few minutes, consider subsampling the dataset used in score function.",
            "\nA PyTorch dataset can be subsampled using torch.utils.data.Subset",
            "(https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset) as following:\n",
            "subset_dataset = torch.utils.data.Subset(dataset, indices)",
        )
        with tqdm.tqdm(
            initial=total_choices_to_validate - remaining_choices_to_validate,
            total=total_choices_to_validate,
            desc="Collecting pre-search statistics",
            smoothing=0.1,
        ) as pbar:
            sample(self.model, sample_func=max)  # reset for sanity
            for name, hparam in binary_search_hps.items():
                reach_zero = False  # if already reach 0 or <0, we can skip later computations
                cur_sensitivity_scores = {}
                for choice in sorted(hparam.choices):
                    select(self.model, {name: choice}, strict=False)
                    if hparam.active == hparam.max or reach_zero:
                        cur_sensitivity_scores[hparam.active] = 0.0
                    else:
                        score = self.eval_score()
                        cur_sensitivity_scores[hparam.active] = self.original_score - score
                        if self.original_score - score < 1e-5:
                            reach_zero = True  # will skip the computation for later choices
                    # better visualization for debug
                    if verbose:
                        pbar.set_postfix(
                            {
                                "cur": (
                                    f"{name}({hparam.active}/{hparam.max}):"
                                    f" {cur_sensitivity_scores[hparam.active]:.2f}"
                                )
                            }
                        )
                    pbar.update(1)

                # make sure the sensitivity is monotonic
                prev_max = 0.0
                for k in list(cur_sensitivity_scores.keys())[::-1]:
                    cur_sensitivity_scores[k] = prev_max = max(cur_sensitivity_scores[k], prev_max)

                self.sensitivity_map[name] = cur_sensitivity_scores
                self.save_search_checkpoint()


def _conv_config():
    return {
        "channels_ratio": tuple(0.05 * i for i in range(1, 21)),
        "kernel_size": (),
        "channel_divisor": 32,
    }


def _norm_lin_config():
    return {
        "features_ratio": tuple(0.05 * i for i in range(1, 21)),
        "feature_divisor": 32,
    }


def _get_fastnas_default_rules():
    return {
        "nn.Conv1d": _conv_config(),
        "nn.Conv2d": _conv_config(),
        "nn.Conv3d": _conv_config(),
        "nn.ConvTranspose1d": _conv_config(),
        "nn.ConvTranspose2d": _conv_config(),
        "nn.ConvTranspose3d": _conv_config(),
        "nn.Linear": _norm_lin_config(),
        "nn.BatchNorm1d": _norm_lin_config(),
        "nn.BatchNorm2d": _norm_lin_config(),
        "nn.BatchNorm3d": _norm_lin_config(),
        "nn.SyncBatchNorm": _norm_lin_config(),
        "nn.InstanceNorm1d": _norm_lin_config(),
        "nn.InstanceNorm2d": _norm_lin_config(),
        "nn.InstanceNorm3d": _norm_lin_config(),
        "nn.LayerNorm": _norm_lin_config(),
        "nn.GroupNorm": {k: v for k, v in _conv_config().items() if k != "kernel_size"},
    }


FastNASConfig: type[ModeloptBaseConfig] = create_model(
    "FastNASConfig",
    **get_kwargs_for_create_model_with_rules(
        registry=DMRegistry,
        default_rules=_get_fastnas_default_rules(),
        doc='Configuration for the ``"fastnas"`` mode.',
    ),
)


@NASModeRegistry.register_mode
@PruneModeRegistry.register_mode
class FastNASModeDescriptor(ModeDescriptor):
    """Class to describe the ``"fastnas"`` mode.

    The properties of this mode can be inspected via the source code.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "fastnas"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return FastNASConfig

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
        """Specifies the search algorithm to use for this mode (if any)."""
        return BinarySearcher

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return convert_fastnas_searchspace

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_fastnas_searchspace

    @property
    def update_for_save(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before saving."""
        return update_fastnas_metadata

    @property
    def update_for_new_mode(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before new mode."""
        return update_fastnas_metadata
