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

"""Thin PDD objective integration for AutoModel's diffusion recipe."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy

try:
    import nemo_automodel.recipes.diffusion.train as automodel_diffusion_train
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
    from nemo_automodel.components.training.rng import ScopedRNG
    from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe
except ImportError as exc:
    raise ImportError(
        "The PDD example requires nemo_automodel. Install "
        "examples/diffusers/fastgen/requirements.txt."
    ) from exc

from modelopt.torch.fastgen import PDDConfig, PDDPipeline
from modelopt.torch.fastgen.plugins.qwen_image_pdd import (
    QwenImagePDDAdapter,
    adopt_qwen_image_mr210_forward,
)

from .training import PDDFlowMatchingStepAdapter


@contextmanager
def _preserve_fp32_timestep_inputs() -> Iterator[None]:
    """Keep Qwen's continuous timestep in FP32 while FSDP computes in BF16."""
    original_builder = automodel_diffusion_train._build_diffusion_parallel_manager_args

    def build_manager_args(**kwargs: Any) -> dict[str, Any]:
        manager_args = original_builder(**kwargs)
        if manager_args.get("_manager_type") != "fsdp2":
            return manager_args

        compute_dtype = kwargs.get("compute_dtype") or kwargs["dtype"]
        current_policy = manager_args.get("mp_policy")
        manager_args["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=getattr(
                current_policy,
                "param_dtype",
                None if kwargs["lora_enabled"] else compute_dtype,
            ),
            reduce_dtype=getattr(current_policy, "reduce_dtype", torch.float32),
            output_dtype=getattr(current_policy, "output_dtype", compute_dtype),
            cast_forward_inputs=False,
        )
        return manager_args

    automodel_diffusion_train._build_diffusion_parallel_manager_args = build_manager_args
    try:
        yield
    finally:
        automodel_diffusion_train._build_diffusion_parallel_manager_args = original_builder


def _config_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value)


def _validate_prepared_student(model: nn.Module, config: PDDConfig) -> None:
    """Require the widened PDD projection to exist before AutoModel setup."""
    try:
        projection = model.get_submodule("proj_out")
    except AttributeError as error:
        raise ValueError("The prepared Qwen student is missing proj_out.") from error
    if not isinstance(projection, nn.Linear):
        raise TypeError("The prepared Qwen student proj_out must be linear.")

    model_config = getattr(model, "config", None)
    in_channels = (
        model_config.get("in_channels")
        if isinstance(model_config, Mapping)
        else getattr(model_config, "in_channels", None)
    )
    if not isinstance(in_channels, int) or in_channels <= 0:
        raise ValueError("The prepared Qwen student has invalid in_channels.")
    expected_out_features = config.grid_size * in_channels
    if projection.out_features != expected_out_features:
        raise ValueError(
            "Prepare the Qwen PDD student before training: expected proj_out.out_features="
            f"{expected_out_features}, got {projection.out_features}."
        )


class PDDDiffusionRecipe(TrainDiffusionRecipe):
    """Use AutoModel's native lifecycle with a PDD loss and frozen teacher."""

    def setup(self) -> None:
        with _preserve_fp32_timestep_inputs():
            super().setup()

            raw_pdd = _config_mapping(self.cfg.get("pdd", {}))
            self.pdd_config = PDDConfig.model_validate(raw_pdd)

            # The student artifact is widened before AutoModel creates FSDP and optimizer state.
            # Binding the MR210 forward here changes behavior only; it creates no parameters.
            adopt_qwen_image_mr210_forward(self.model)
            _validate_prepared_student(self.model, self.pdd_config)
            self.model.enable_gradient_checkpointing()

            # ``teacher_model`` is the BaseRecipe-recognized frozen reference-model name; native
            # checkpoint save/load deliberately excludes it while tracking every student state.
            self.teacher_model = self._load_teacher()
        pdd_pipeline = PDDPipeline(
            self.model,
            self.teacher_model,
            self.pdd_config,
            QwenImagePDDAdapter(self.pdd_config, compute_dtype=self.compute_dtype),
        )
        self.flow_matching_pipeline = PDDFlowMatchingStepAdapter(pdd_pipeline)
        logging.info("[PDD] AutoModel lifecycle enabled; grid_size=%d", self.pdd_config.grid_size)

    def _load_teacher(self) -> nn.Module:
        """Load the frozen PDD target model with AutoModel's diffusion parallelizer."""
        fsdp_cfg = self.cfg.get("fsdp", None)
        ddp_cfg = self.cfg.get("ddp", None)
        manager_args = automodel_diffusion_train._build_diffusion_parallel_manager_args(
            fsdp_cfg=fsdp_cfg,
            ddp_cfg=ddp_cfg,
            world_size=self.world_size,
            dtype=self.model_dtype,
            compute_dtype=self.compute_dtype,
            lora_enabled=False,
        )
        teacher_source = self.cfg.get(
            "model.teacher_model_name_or_path",
            "Qwen/Qwen-Image",
        )
        if not Path(teacher_source).expanduser().is_dir():
            teacher_source = snapshot_download(
                teacher_source,
                revision=self.cfg.get("model.teacher_revision", None),
            )
        with ScopedRNG(seed=self.seed + 1, ranked=dist.is_initialized()):
            pipe, _ = NeMoAutoDiffusionPipeline.from_pretrained(
                teacher_source,
                torch_dtype=self.model_dtype,
                device=self.device,
                parallel_scheme={"transformer": manager_args},
                components_to_load=["transformer"],
                load_for_training=False,
                low_cpu_mem_usage=True,
            )
        teacher = pipe.transformer
        del pipe
        adopt_qwen_image_mr210_forward(teacher)
        teacher.eval().requires_grad_(False)
        return teacher
