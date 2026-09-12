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

"""Pinned AutoModel compatibility seam for Qwen-Image PDD setup."""

from __future__ import annotations

import inspect
import logging
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import nemo_automodel
import nemo_automodel.recipes.diffusion.train as automodel_diffusion_train
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy

from modelopt.torch.fastgen.plugins.qwen_image_pdd import freeze_qwen_image_pdd_unused_parameters

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["automodel_pdd_setup"]

_SUPPORTED_AUTOMODEL_RELEASE = "0.5.0"
_SETUP_PATCH_LOCK = threading.RLock()


def _accepts_parameters(function: Any, required: set[str]) -> bool:
    parameters = inspect.signature(function).parameters
    return required.issubset(parameters) or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _validate_automodel_setup_api() -> None:
    version = str(getattr(nemo_automodel, "__version__", ""))
    release = version.partition("+")[0]
    if release != _SUPPORTED_AUTOMODEL_RELEASE:
        raise RuntimeError(
            "The Qwen PDD example requires nemo_automodel release "
            f"{_SUPPORTED_AUTOMODEL_RELEASE}; found {version or '<unknown>'}."
        )

    builder = getattr(automodel_diffusion_train, "_build_diffusion_parallel_manager_args", None)
    required_builder_parameters = {
        "fsdp_cfg",
        "ddp_cfg",
        "world_size",
        "dtype",
        "compute_dtype",
        "lora_enabled",
    }
    if not callable(builder) or not _accepts_parameters(builder, required_builder_parameters):
        raise RuntimeError("AutoModel diffusion parallel-manager API is incompatible with PDD.")

    pipeline_cls = getattr(automodel_diffusion_train, "NeMoAutoDiffusionPipeline", None)
    if not isinstance(pipeline_cls, type):
        raise RuntimeError("AutoModel diffusion pipeline loading API is incompatible with PDD.")
    descriptor = inspect.getattr_static(pipeline_cls, "from_pretrained", None)
    if not isinstance(descriptor, classmethod) or not _accepts_parameters(
        pipeline_cls.from_pretrained,
        {"load_for_training"},
    ):
        raise RuntimeError("AutoModel diffusion pipeline loading API is incompatible with PDD.")


@contextmanager
def automodel_pdd_setup() -> Iterator[None]:
    """Scope the two AutoModel 0.5.0 setup adaptations required by Qwen PDD.

    AutoModel does not yet expose public hooks for preserving FP32 forward inputs or
    freezing model parameters before optimizer construction. The lock serializes the
    narrow process-global setup window; both module attributes are restored on exit.
    """
    with _SETUP_PATCH_LOCK:
        _validate_automodel_setup_api()
        original_builder = automodel_diffusion_train._build_diffusion_parallel_manager_args
        original_pipeline_cls = automodel_diffusion_train.NeMoAutoDiffusionPipeline

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

        class PDDSetupPipeline(automodel_diffusion_train.NeMoAutoDiffusionPipeline):
            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
                del cls
                pipe, managers = original_pipeline_cls.from_pretrained(*args, **kwargs)
                if kwargs.get("load_for_training", False):
                    frozen_names = freeze_qwen_image_pdd_unused_parameters(pipe.transformer)
                    logging.info(
                        "[PDD] Full training excludes %d unused final-block text-output "
                        "tensors: %s",
                        len(frozen_names),
                        ", ".join(frozen_names),
                    )
                return pipe, managers

        automodel_diffusion_train._build_diffusion_parallel_manager_args = build_manager_args
        automodel_diffusion_train.NeMoAutoDiffusionPipeline = PDDSetupPipeline
        try:
            yield
        finally:
            automodel_diffusion_train.NeMoAutoDiffusionPipeline = original_pipeline_cls
            automodel_diffusion_train._build_diffusion_parallel_manager_args = original_builder
