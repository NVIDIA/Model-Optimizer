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

"""Entrypoints for AutoNAS mode."""

import copy
import hashlib
import json
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import tqdm
from pydantic import create_model
from torch.nn.modules.batchnorm import _BatchNorm

from modelopt.torch.opt.config import ModeloptBaseConfig, get_kwargs_for_create_model_with_rules
from modelopt.torch.opt.conversion import ApplyModeError, ModelLikeModule
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ConvertReturnType,
    MetadataDict,
    ModeDescriptor,
    RestoreEntrypoint,
    UpdateEntrypoint,
)
from modelopt.torch.opt.searcher import BaseSearcher, ForwardLoop, SearchConfig, SearchStateDict
from modelopt.torch.opt.utils import is_configurable
from modelopt.torch.utils import (
    compare_dict,
    get_model_attributes,
    is_channels_last,
    num2hrb,
    random,
    run_forward_loop,
    stats,
    torch_detach,
    torch_to,
)

from .algorithms import ConstraintsFunc, get_constraints_func
from .conversion import NASModeRegistry
from .patch import PatchData, PatchManager, _modelopt_eval_recursion_guard, prep_for_eval
from .registry import DMRegistry
from .search_space import generate_search_space
from .utils import get_subnet_config, sample, select

__all__ = [
    "AutoNASConfig",
    "AutoNASModeDescriptor",
    "AutoNASPatchManager",
    "EvolveSearcher",
    "IterativeSearcher",
    "RandomSearcher",
    "convert_autonas_searchspace",
    "convert_searchspace",
    "restore_autonas_searchspace",
    "restore_searchspace",
    "update_autonas_metadata",
]

# we have two different numbers here since during training it might take longer to stabilize
MODELOPT_QUEUE_MAXLEN = 50  # indicates length of modelopt data queue for BN calib
MODELOPT_BN_CALIB_ITERS = (
    100  # indicates # iters in train mode 'til we trust BN stats without calib
)


def _get_ratio_list():
    return (0.5, 0.67, 1.0)


def _conv_config():
    return {
        "channels_ratio": _get_ratio_list(),
        "kernel_size": (),
        "channel_divisor": 32,
    }


def _norm_lin_config():
    return {
        "features_ratio": _get_ratio_list(),
        "feature_divisor": 32,
    }


AutoNASConfig: type[ModeloptBaseConfig] = create_model(
    "AutoNASConfig",
    **get_kwargs_for_create_model_with_rules(
        registry=DMRegistry,
        default_rules={
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
            "nn.Sequential": {"min_depth": 0},
        },
        doc='Configuration for the ``"autonas"`` mode.',
    ),
)


class AutoNASPatchManager(PatchManager):
    """A class to handle the monkey patching of the model for automode."""

    def _get_default_patch_data(self) -> PatchData:
        """Return the default patch data of the model."""
        return {
            "queue": deque(maxlen=MODELOPT_QUEUE_MAXLEN),  # queue for reset_bn data batches cache
            "fill": False,  # force push to queue if set to True
            "count": 0,  # internal count for number of forward passes
            "train_count_since_sample": MODELOPT_BN_CALIB_ITERS,  # step count since last sample
            "autosampled": False,  # indicator whether current sub-net was auto-sampled
        }

    def _hook_post_eval(self, forward_loop: ForwardLoop | None = None) -> None:
        """Optional hook that is called after eval() (or train(False)) to calibrate the model."""
        # get the model and patch_data reference
        model = self._model
        patch_data = self.patch_data

        # automode-related attributes
        fill = patch_data["fill"]

        # BN calibration related in autonas/fastnas mode
        if is_configurable(model):
            # indicates that model is being trained in which case we want max sub-net for eval
            if patch_data["autosampled"]:
                sample(model, sample_func=max)

            # check current count since last sample operation
            count_since_sample = patch_data["train_count_since_sample"]

            # logic to determine whether and how to calibrate BN statistics
            queue = patch_data["queue"]
            if forward_loop is None and count_since_sample < MODELOPT_BN_CALIB_ITERS:
                if len(queue) < queue.maxlen and self._is_modelopt_queue_needed(model):
                    warnings.warn(
                        "modelopt data queue not filled! Inference results can be inaccurate."
                    )
                else:
                    forward_loop = partial(
                        run_forward_loop,
                        data_loader=queue,
                        collect_func=lambda x: x,
                    )
            elif forward_loop is not None:
                # in this case we want to fill up the queue since new data is provided.
                patch_data["fill"] = True

        # call reset_bn in eval mode
        if forward_loop is not None:
            with _modelopt_eval_recursion_guard(model):
                self._reset_bn(model, forward_loop)
            patch_data["train_count_since_sample"] = MODELOPT_BN_CALIB_ITERS

        # resetting modelopt_fill
        patch_data["fill"] = fill

    @staticmethod
    def _is_modelopt_queue_needed(model) -> bool:
        return any(isinstance(m, _BatchNorm) and m.track_running_stats for m in model.modules())

    @classmethod
    def _reset_bn(cls, model: nn.Module, forward_loop: ForwardLoop):
        """Calibrate BN statistics by the provided data loader.

        Args:
            model: The model to be calibrated
            forward_loop: A ``Callable`` that takes a model as input and runs a pre-defined forward
                loop on it using data that is suitable for BN calibration.
        """
        # skip if there is no batch normalization layer in the network that
        # requires tracking running stats
        if not cls._is_modelopt_queue_needed(model):
            return

        is_training = model.training
        momentum = {}

        # modify all batch norms to track cumulative moving average
        for name, m in model.named_modules():
            if isinstance(m, _BatchNorm):
                m.reset_running_stats()
                m.train()
                momentum[name] = m.momentum
                m.momentum = None

        # do some forward passes to track BN statistics
        with torch.no_grad():
            forward_loop(model)

        # reset network into its original state
        model.train(is_training)
        for name, m in model.named_modules():
            if name in momentum:
                m.momentum = momentum[name]

    def _hook_pre_sample(self) -> None:
        """Optional hook to be called before sample-related operations (sample & select)."""
        patch_data = self.patch_data_or_empty
        patch_data["train_count_since_sample"] = 0
        patch_data["autosampled"] = False

    @property
    def sample_during_training(self) -> bool:
        """Indicates whether we should sample a new subnet during training."""
        return True

    def _hook_pre_forward(self, *args, **kwargs) -> None:
        """Optional hook that is called before the original forward function is called."""
        patch_data = self.patch_data
        mod = self._model
        if not (mod.training or patch_data["fill"]):
            return

        # run sample step and update metadata related to sampling
        def process_data(data):
            return torch_to(torch_detach(data), device="cpu", non_blocking=True)

        if mod.training or patch_data["fill"]:
            # sample new model configuration in training mode and indicate it was auto-sampled
            if mod.training:
                if self.sample_during_training:
                    sample(mod)
                    patch_data["autosampled"] = True
                patch_data["train_count_since_sample"] += 1  # update train mode count
            # push detached CPU copy of data to queue (once full only update periodically)
            queue = patch_data["queue"]
            if len(queue) < queue.maxlen or patch_data["count"] % 100 == 0 or patch_data["fill"]:
                queue.append((*process_data(args), process_data(kwargs)))
            patch_data["count"] += 1


class IterativeSearcher(BaseSearcher, ABC):
    """Base class for iterative search algorithms."""

    iter_num: int
    num_satisfied: int
    constraints_func: ConstraintsFunc
    candidate: dict[str, Any]
    best: SearchStateDict
    samples: dict[str, Any]
    history: dict[str, Any]
    best_history: dict[str, Any]

    @property
    def default_search_config(self) -> SearchConfig:
        """Get the default config for the searcher."""
        return {
            **super().default_search_config,
            "num_iters": 5000,
            "max_iter_data_loader": 50,  # plenty to calibrate BN statistics
        }

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Return default state dict."""
        return {
            "iter_num": 0,
            "num_satisfied": 0,
            "candidate": {},
            "best": {"metric": -float("inf"), "constraints": None},
            "samples": {},
            "history": {"metric": [], "constraints": defaultdict(list)},
            "best_history": {"iter_num": [], "candidates": []},
        }

    def sanitize_search_config(self, config: SearchConfig | None) -> SearchConfig:
        """Sanitize the search config dict."""
        config = super().sanitize_search_config(config)
        assert config["score_func"] is not None, "Please provide `score_func`!"
        return config

    def _sample(self) -> dict[str, Any]:
        return {"config": sample(self.model)}

    @abstractmethod
    def sample(self) -> dict[str, Any]:
        """Sample and select new sub-net configuration and return configuration."""
        raise NotImplementedError

    def before_search(self) -> None:
        """Ensure that the model is actually configurable and ready for eval."""
        super().before_search()

        # Initialize the constraints functor
        self.constraints_func = get_constraints_func(
            self.model, self.constraints, self.dummy_input, deployment=self.deployment
        )

        # Set the model to max subnet and put model into eval mode with potential calibaration
        sample(self.model, max)
        prep_for_eval(self.model, self.forward_loop)

        # check for configurability of the models; otherwise we don't need to run the search.
        if not is_configurable(self.model):
            warnings.warn(
                "Provided model does not contain configurable hparams with multiple choices! "
                "Running only one iteration."
            )
            self.config["num_iters"] = 1

        # do a sanity check and profile the constraints
        # This way we also fill up the interpolation table for latency with min, centroid, max!
        from .algorithms import profile  # TODO: hack until we refactor files

        profile(
            self.model,
            constraints=self.constraints_func,
            strict=True,
            verbose=self.config["verbose"],
            use_centroid=True,
        )

    def run_search(self) -> None:
        """Run iterative search loop."""
        num_iters = self.config["num_iters"]
        verbose = self.config["verbose"]

        # run search loop
        if verbose:
            pbar = tqdm.trange(
                self.iter_num,
                num_iters,
                initial=self.iter_num,
                total=num_iters,
                position=0,
                leave=True,
            )

        for self.iter_num in range(self.iter_num, num_iters):
            self.before_step()
            self.run_step()
            self.after_step()

            if verbose:
                info = {
                    "num_satisfied": self.num_satisfied,
                    "metric": self.candidate["metric"],
                    "constraints": self.candidate["constraints"],
                    "best_subnet_metric": self.best["metric"],
                    "best_subnet_constraints": self.best["constraints"],
                }

                # display the full stats only once a while
                if len(self.history["metric"]) == 100:
                    info["metric/stats"] = stats(self.history["metric"])
                    info["constraints/stats"] = {
                        name: stats(vals) for name, vals in self.history["constraints"].items()
                    }
                    self.history["metric"].clear()
                    self.history["constraints"].clear()

                def _recursive_format(obj, fmt):
                    if isinstance(obj, float):
                        return num2hrb(obj)
                    if isinstance(obj, dict):
                        return {k: _recursive_format(v, fmt) for k, v in obj.items()}
                    return obj

                pbar.update()
                info = _recursive_format(info, fmt="{:.4g}")
                pbar.set_description(f"[num_satisfied] = {info['num_satisfied']}")

            if self.early_stop():
                break

        if verbose:
            pbar.close()
            print(f"[best_subnet_constraints] = {info['best_subnet_constraints']}")

    def after_search(self) -> None:
        """Select best model."""
        super().after_search()
        select(self.model, self.best["config"])
        # if self.has_score:
        #     final_score = self.eval_score(silent=False)
        #     print(f"Final score for the searched model: {final_score}")

    def before_step(self) -> None:
        """Run before each iterative step."""

    def run_step(self) -> None:
        """The main routine of each iterative step."""
        # sample and select the candidate
        self.candidate = self.sample()

        # obtain the independent config and hparams
        # TODO: consider whether we want to eventually bring back "active_config" from omnimizer
        # to improve the efficiency of the search and avoid redundant sampling. This was used to
        # avoid testing two architectures that only differed in modules that were not active, i.e.,
        # due to reduced depth.
        independent_config = self._configurable_config(self.model)

        # serialize and hash the candidate
        buffer = json.dumps({"config": independent_config}, sort_keys=True)
        ckey = hashlib.sha256(buffer.encode()).hexdigest()

        if ckey not in self.samples:
            # check constraints
            self.candidate["is_satisfied"], self.candidate["constraints"] = self.constraints_func()

            # evaluate the metric
            if self.candidate["is_satisfied"]:
                self.candidate["metric"] = self.eval_score()
            else:
                self.candidate["metric"] = -float("inf")

            self.samples[ckey] = copy.deepcopy(self.candidate)
        else:
            self.candidate = copy.deepcopy(self.samples[ckey])

        self.num_satisfied += int(self.candidate["is_satisfied"])
        is_best = self.candidate["metric"] >= self.best["metric"] or self.iter_num == 0
        self.candidate["is_best"] = is_best

        # update the stats if satisfied
        if self.candidate["is_satisfied"]:
            self.history["metric"].append(self.candidate["metric"])
            for name, val in self.candidate["constraints"].items():
                self.history["constraints"][name].append(val)

        # update the best if necessary
        if is_best:
            self.best = copy.deepcopy(self.candidate)
            self.best_history["iter_num"].append(self.iter_num)
            self.best_history["candidates"].append(self.best)

        # save the state dict
        if self.iter_num % 100 == 0 or self.iter_num == self.config["num_iters"] - 1 or is_best:
            self.save_search_checkpoint()

    def after_step(self) -> None:
        """Run after each iterative step."""

    def early_stop(self) -> bool:
        """Check if we should early stop the search if possible."""
        return False

    def _configurable_config(self, model) -> dict[str, Any]:
        """Returns the config dict of the configurable hyperparameters of the model."""
        return get_subnet_config(model, configurable=True)


class RandomSearcher(IterativeSearcher):
    """An iterative searcher that samples subnets randomly."""

    def sample(self) -> dict[str, Any]:
        """Random sample new subset during each steo."""
        return self._sample()


class EvolveSearcher(IterativeSearcher):
    """An iterative searcher that uses an evolutionary algorithm to optimize the subnet config."""

    population: list[dict[str, Any]]
    candidates: list[dict[str, Any]]

    @property
    def default_search_config(self) -> SearchConfig:
        """Default search config contains additional algorithm parameters."""
        return {
            **super().default_search_config,
            "population_size": 100,
            "candidate_size": 25,
            "mutation_prob": 0.1,
        }

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Return default state dict."""
        return {
            **super().default_state_dict,
            "population": [],
            "candidates": [],
        }

    def _mutate(self, input: dict[str, Any]) -> dict[str, Any]:
        output = self._sample()
        # only considers independent hparams
        output["config"] = self._configurable_config(self.model)
        for var in output:  # pylint: disable=C0206
            for name in output[var]:
                if random.random() > self.config["mutation_prob"]:
                    output[var][name] = input[var][name]
        # returns reduced, independent config
        return output

    def _crossover(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        output = {"config": {}}
        # only considers independent hparams
        for name in self._configurable_config(self.model):
            output["config"][name] = random.choice(inputs)["config"][name]
        # returns reduced, independent config
        return output

    def sample(self) -> dict[str, Any]:
        """Sampling a new subnet involves random sampling, mutation, and crossover."""
        if not self.candidates:
            return self._sample()
        if len(self.population) < self.config["population_size"] // 2:
            output = self._mutate(random.choice(self.candidates))
        else:
            output = self._crossover(random.sample(self.candidates, 2))
        # returns full config
        select(self.model, output["config"])
        output["config"] = get_subnet_config(self.model)
        return output

    def before_search(self) -> None:
        """Set the lower bound of the constraints to 0.85 * upper bound before search."""
        super().before_search()
        self.constraints_func.set_rel_lower_bounds(rel_lower=0.85)

    def before_step(self) -> None:
        """Update candidates and population before each iterative step."""
        if len(self.population) >= self.config["population_size"]:
            all_candidates = sorted(self.population, key=lambda x: x["metric"], reverse=True)
            self.candidates = all_candidates[: self.config["candidate_size"]]
            self.population = []

    def after_step(self) -> None:
        """Update population after each iterative step."""
        if self.candidate["is_satisfied"]:
            self.population.append(copy.deepcopy(self.candidate))


def convert_searchspace(
    model: nn.Module, config: ModeloptBaseConfig, patch_manager_type: type[PatchManager]
) -> ConvertReturnType:
    """Convert given model into a search space."""
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    # check if the model is in channels-last format
    # NOTE: not supported because we have not implemented channel sorting for channels-last format
    if is_channels_last(model):
        raise RuntimeError(f"Channels-last format in {type(model).__name__} is not supported!")

    # put together metadata before starting to modify model
    metadata = {"model_attributes": get_model_attributes(model)}

    # now convert the model to a search space
    search_space = generate_search_space(model, rules=config.model_dump())

    # sanity check if any module has been converted
    if not search_space.is_configurable():
        raise ApplyModeError(
            "The model does not contain any configurable hyperparameters! Please check the"
            " documentation for modules and config and how to get a configurable model."
        )

    # activate the max subnet that can be used for subsequent tasks like training
    # (note it should be activated already but we do it here again to be sure)
    sample(model, max)

    # search space requires a patch
    patch_manager_type(model).patch()

    # get current config store in metadata
    metadata["subnet_config"] = get_subnet_config(model)

    # return converted model as well as metadata
    return model, metadata


def convert_autonas_searchspace(model: nn.Module, config: ModeloptBaseConfig) -> ConvertReturnType:
    """Convert search space for AutoNAS mode with correct patch manager."""
    return convert_searchspace(model, config, AutoNASPatchManager)


def restore_autonas_searchspace(
    model: nn.Module, config: ModeloptBaseConfig, metadata: MetadataDict
) -> nn.Module:
    """Restore search space for AutoNAS mode with correct patch manager."""
    return restore_searchspace(model, config, metadata, AutoNASPatchManager)


def restore_searchspace(
    model: nn.Module,
    config: ModeloptBaseConfig,
    metadata: MetadataDict,
    patch_manager: type[PatchManager],
) -> nn.Module:
    """Restore a search space from the given model."""
    # retrieve metadata model attributes
    model_attributes = metadata["model_attributes"]

    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    # set up train/eval mode accordingly
    is_training = model.training
    model.train(model_attributes["training"])

    # run regular convert entrypoint
    model, metadata_new = convert_searchspace(model, config, patch_manager)

    # compare new model attributes with provided model attributes
    new_attributes = metadata_new["model_attributes"]
    unmatched_keys = compare_dict(model_attributes, new_attributes)
    if len(unmatched_keys) > 0:
        error_msg = ["Following keys in your model do not match the checkpoint:"]
        error_msg += [
            f"{k}: Original={model_attributes.get(k, 'N/A')}; New={new_attributes.get(k, 'N/A')}"
            for k in unmatched_keys
        ]
        raise ApplyModeError("\n\t".join(error_msg))

    # ensure model is back in mode it started out in
    model.train(is_training)

    # select subnet config stored in metadata
    select(model, metadata["subnet_config"])

    # return converted model
    return model


def update_autonas_metadata(
    model: nn.Module, config: ModeloptBaseConfig, metadata: MetadataDict
) -> None:
    """Update subnet config to current subnet config of model."""
    metadata["subnet_config"] = get_subnet_config(model)


@NASModeRegistry.register_mode
class AutoNASModeDescriptor(ModeDescriptor):
    """Class to describe the ``"autonas"`` mode.

    The properties of this mode can be inspected via the source code.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "autonas"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return AutoNASConfig

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
        return EvolveSearcher

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return convert_autonas_searchspace

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_autonas_searchspace

    @property
    def update_for_save(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before saving."""
        return update_autonas_metadata

    @property
    def update_for_new_mode(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before new mode."""
        return update_autonas_metadata
