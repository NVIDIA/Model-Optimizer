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

"""Student artifact restore and train-scope handling for the QAD example.

The generic AutoModel diffusion builder intentionally owns FSDP and optimizer
construction. QAD only needs two narrowly-scoped hooks around that builder:

* validate the ModelOpt topology restored by a native Diffusers training bundle
  before FSDP;
* after FSDP, optionally freeze everything except ModelOpt SVDQuant's HF PEFT A/B
  parameters and rebuild AdamW from the live sharded parameters.

Quantization itself is never calibrated here. The complete topology, weights,
and quantizer buffers must already be present in the student bundle.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import logging
import re
from typing import TYPE_CHECKING, Any

import modelopt.torch.opt as mto
from modelopt.torch.quantization.nn import TensorQuantizer

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch
    from torch import nn

_SVDQUANT_PARAMETER_RE = re.compile(r"(?:^|\.)lora_[AB]\.modelopt_svdquant\.weight$")
_SUPPORTED_STUDENT_MODES = frozenset({"nvfp4", "nvfp4_svdquant"})
_SUPPORTED_TRAIN_SCOPES = frozenset({"all", "lora_only"})


@dataclasses.dataclass(frozen=True)
class StudentSettings:
    """Resolved ``qad.student`` configuration."""

    mode: str
    model_name_or_path: str
    train_scope: str = "all"

    def validate(self) -> None:
        if self.mode not in _SUPPORTED_STUDENT_MODES:
            raise ValueError(
                f"qad.student.mode must be one of {sorted(_SUPPORTED_STUDENT_MODES)}, "
                f"got {self.mode!r}."
            )
        if not self.model_name_or_path:
            raise ValueError("model.pretrained_model_name_or_path is required for the student.")
        if self.train_scope not in _SUPPORTED_TRAIN_SCOPES:
            raise ValueError(
                f"qad.student.train_scope must be 'all' or 'lora_only', got {self.train_scope!r}."
            )

        if self.mode == "nvfp4" and self.train_scope != "all":
            raise ValueError("Regular NVFP4 supports only qad.student.train_scope=all.")


@dataclasses.dataclass
class StudentBuildState:
    """Information captured while AutoModel builds the student."""

    parallel_scheme: dict[str, dict[str, Any]] | None = None
    quantizer_count: int = 0
    svdquant_parameter_names: tuple[str, ...] = ()


def _is_block16_nvfp4(quantizer: TensorQuantizer) -> bool:
    block_sizes = quantizer.block_sizes or {}
    return bool(
        (quantizer.is_nvfp4_dynamic or quantizer.is_nvfp4_static) and block_sizes.get(-1) == 16
    )


def _enabled_quantizer_leaves(module: Any) -> tuple[TensorQuantizer, ...]:
    if module is None or not hasattr(module, "modules"):
        return ()
    return tuple(
        child
        for child in module.modules()
        if isinstance(child, TensorQuantizer) and child.is_enabled
    )


def _validate_nvfp4_quantizers(
    model: nn.Module,
    *,
    artifact_name: str,
    required_targets: tuple[str, ...] = (),
) -> None:
    """Reject non-NVFP4 artifacts before FSDP obscures their module topology."""
    enabled_by_slot: dict[str, list[tuple[str, TensorQuantizer]]] = {
        "weight": [],
        "input": [],
    }
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer) or not module.is_enabled:
            continue
        path_parts = name.split(".")
        for slot in enabled_by_slot:
            if f"{slot}_quantizer" in path_parts:
                enabled_by_slot[slot].append((name, module))

    missing_slots = [slot for slot, entries in enabled_by_slot.items() if not entries]
    if missing_slots:
        raise RuntimeError(
            f"{artifact_name} is not an NVFP4 W4A4 training artifact: no enabled "
            + "/".join(missing_slots)
            + " quantizers were found."
        )

    incompatible = [
        name
        for entries in enabled_by_slot.values()
        for name, quantizer in entries
        if not _is_block16_nvfp4(quantizer)
    ]
    if incompatible:
        raise RuntimeError(
            f"{artifact_name} contains enabled GEMM quantizers that are not block-16 NVFP4 "
            "(E2M1 values with E4M3 scales): " + ", ".join(incompatible[:5])
        )

    for target_name in required_targets:
        target = model.get_submodule(target_name)
        get_base_layer = getattr(target, "get_base_layer", None)
        base_layer = get_base_layer() if callable(get_base_layer) else target
        for slot in ("weight", "input"):
            leaves = _enabled_quantizer_leaves(getattr(base_layer, f"{slot}_quantizer", None))
            if not leaves or any(not _is_block16_nvfp4(quantizer) for quantizer in leaves):
                raise RuntimeError(
                    f"SVDQuant target {target_name!r} does not have an enabled block-16 "
                    f"NVFP4 {slot}_quantizer."
                )

    logging.info(
        "[QAD] validated block-16 NVFP4 W4A4 quantizers before FSDP: %d weight, %d input",
        len(enabled_by_slot["weight"]),
        len(enabled_by_slot["input"]),
    )


def _modelopt_mode_states(model: nn.Module) -> dict[str, dict[str, Any]]:
    if not mto.ModeloptStateManager.is_converted(model):
        return {}
    return dict(mto.modelopt_state(model)["modelopt_state_dict"])


def _reject_non_training_modes(mode_states: dict[str, dict[str, Any]]) -> None:
    if "real_quantize" in mode_states:
        raise RuntimeError(
            "QAD cannot train a compressed real-quantized bundle. Recalibrate without "
            "quantize.py --compress and provide the resulting fake-quantized training bundle."
        )


def _validate_regular_bundle(model: nn.Module) -> int:
    mode_states = _modelopt_mode_states(model)
    if not mode_states:
        raise RuntimeError(
            "qad.student.mode=nvfp4 requires a ModelOpt-aware Diffusers training bundle. "
            "Calibrate it with quantize.py --output-bundle before starting QAD."
        )
    _reject_non_training_modes(mode_states)
    if "svdquant_calibrate" in mode_states:
        raise RuntimeError(
            "qad.student.mode=nvfp4 received an SVDQuant bundle; use mode=nvfp4_svdquant."
        )
    quantizers = [module for module in model.modules() if isinstance(module, TensorQuantizer)]
    if not quantizers:
        raise RuntimeError("The regular NVFP4 student bundle restored no TensorQuantizers.")
    _validate_nvfp4_quantizers(model, artifact_name="The regular NVFP4 student bundle")
    logging.info(
        "[QAD] validated regular ModelOpt NVFP4 bundle before FSDP: %d quantizers",
        len(quantizers),
    )
    return len(quantizers)


def _validate_svdquant_bundle(model: nn.Module) -> tuple[str, ...]:
    mode_states = _modelopt_mode_states(model)
    _reject_non_training_modes(mode_states)
    mode_state = mode_states.get("svdquant_calibrate")
    if mode_state is None:
        raise RuntimeError(
            "qad.student.mode=nvfp4_svdquant requires a bundle containing the "
            "svdquant_calibrate ModelOpt mode."
        )
    metadata = mode_state.get("metadata", {}).get("svdquant_peft")
    if not metadata:
        raise RuntimeError(
            "The SVDQuant bundle is malformed or predates the HF PEFT contract: its "
            "svdquant_calibrate mode has no svdquant_peft metadata. Recalibrate it "
            "with quantize.py --output-bundle."
        )

    expected_targets = tuple(metadata.get("target_modules", ()))
    names = tuple(
        name for name, _ in model.named_parameters() if _SVDQUANT_PARAMETER_RE.search(name)
    )
    expected_names = {
        f"{target_name}.lora_{factor}.modelopt_svdquant.weight"
        for target_name in expected_targets
        for factor in ("A", "B")
    }
    if not expected_targets or set(names) != expected_names:
        raise RuntimeError(
            "The SVDQuant bundle did not restore a complete pair of "
            "lora_A/lora_B.modelopt_svdquant weights for every target module. "
            "A weight-free quantizer state or a deployment export is not a valid "
            "QAD training bundle."
        )
    _validate_nvfp4_quantizers(
        model,
        artifact_name="The SVDQuant student bundle",
        required_targets=expected_targets,
    )
    missing_pre_quant_scale_buffers: list[str] = []
    for target_name in expected_targets:
        target = model.get_submodule(target_name)
        get_base_layer = getattr(target, "get_base_layer", None)
        base_layer = get_base_layer() if callable(get_base_layer) else target
        input_quantizer = getattr(base_layer, "input_quantizer", None)
        pre_quant_scale = getattr(input_quantizer, "_pre_quant_scale", None)
        if (
            pre_quant_scale is None
            or getattr(input_quantizer, "_buffers", {}).get("_pre_quant_scale")
            is not pre_quant_scale
        ):
            missing_pre_quant_scale_buffers.append(target_name)
    if missing_pre_quant_scale_buffers:
        raise RuntimeError(
            "SVDQuant pre_quant_scale must be restored as frozen TensorQuantizer buffer "
            "state for every target; missing or non-buffer targets: "
            + ", ".join(missing_pre_quant_scale_buffers[:5])
        )
    logging.info(
        "[QAD] validated SVDQuant training bundle before FSDP: %d targets, %d A/B tensors",
        len(expected_targets),
        len(names),
    )
    return names


def _apply_train_scope(model: nn.Module, scope: str) -> list[nn.Parameter]:
    if scope == "lora_only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(_SVDQUANT_PARAMETER_RE.search(name) is not None)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"qad.student.train_scope={scope!r} selected no parameters.")

    if scope == "lora_only":
        live_names = tuple(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )
        invalid = [name for name in live_names if not _SVDQUANT_PARAMETER_RE.search(name)]
        if invalid:
            raise RuntimeError(
                "lora_only left non-SVDQuant parameters trainable: " + ", ".join(invalid[:5])
            )

    parameter_pre_scales = [
        name for name, _ in model.named_parameters() if "pre_quant_scale" in name
    ]
    if parameter_pre_scales:
        raise RuntimeError(
            "pre_quant_scale must remain a buffer and must never enter the optimizer: "
            + ", ".join(parameter_pre_scales[:5])
        )
    return trainable


def _rebuild_optimizer_from_live_parameters(
    optimizer: torch.optim.Optimizer,
    parameters: list[nn.Parameter],
) -> torch.optim.Optimizer:
    """Recreate the just-built optimizer without carrying stale parameter refs."""
    if optimizer.state:
        raise RuntimeError("QAD expected a newly-created optimizer with no state.")
    if len(optimizer.param_groups) != 1:
        raise RuntimeError(
            "QAD lora_only currently expects AutoModel to create one optimizer parameter group."
        )
    return type(optimizer)(parameters, **dict(optimizer.defaults))


def _validate_optimizer_membership(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual_list = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    actual = {id(parameter) for parameter in actual_list}
    if len(actual) != len(actual_list):
        raise RuntimeError("The student optimizer contains duplicate parameter references.")
    if actual != expected:
        raise RuntimeError(
            "Student optimizer membership does not exactly match the live post-FSDP "
            f"trainable parameters (missing={len(expected - actual)}, extra={len(actual - expected)})."
        )


def _guard_automodel_hooks(diffusion_train: Any, auto_pipeline: Any) -> None:
    builder_parameters = inspect.signature(diffusion_train.build_model_and_optimizer).parameters
    required_builder_parameters = {
        "model_id",
        "learning_rate",
        "device",
        "dtype",
        "optimizer_cfg",
    }
    missing = required_builder_parameters - set(builder_parameters)
    if missing:
        raise RuntimeError(
            "Unsupported nemo_automodel diffusion builder; missing parameters: "
            + ", ".join(sorted(missing))
        )
    if not hasattr(auto_pipeline, "_apply_parallelization"):
        raise RuntimeError(
            "Unsupported nemo_automodel: auto_diffusion_pipeline._apply_parallelization is missing."
        )


@contextlib.contextmanager
def patch_student_build(
    settings: StudentSettings,
) -> Iterator[StudentBuildState]:
    """Patch the two example-local seams needed during the parent ``setup`` call.

    Both module globals are restored in ``finally``. The patch is active only while
    the one student is being constructed; teacher construction happens afterwards.
    """
    from nemo_automodel._diffusers import auto_diffusion_pipeline as auto_pipeline
    from nemo_automodel.recipes.diffusion import train as diffusion_train

    _guard_automodel_hooks(diffusion_train, auto_pipeline)
    original_apply_parallelization = auto_pipeline._apply_parallelization
    original_build_model_and_optimizer = diffusion_train.build_model_and_optimizer
    state = StudentBuildState()
    apply_calls = 0

    def apply_parallelization(pipe, parallel_scheme):
        nonlocal apply_calls
        apply_calls += 1
        if apply_calls != 1:
            raise RuntimeError(
                "QAD's guarded student build expected exactly one parallelized component load."
            )
        state.parallel_scheme = parallel_scheme
        transformer = pipe.transformer
        if settings.mode == "nvfp4":
            state.quantizer_count = _validate_regular_bundle(transformer)
        else:
            state.svdquant_parameter_names = _validate_svdquant_bundle(transformer)
            state.quantizer_count = sum(
                isinstance(module, TensorQuantizer) for module in transformer.modules()
            )
        return original_apply_parallelization(pipe, parallel_scheme)

    def build_model_and_optimizer(**kwargs):
        pipe, optimizer, device_mesh = original_build_model_and_optimizer(**kwargs)
        trainable = _apply_train_scope(pipe.transformer, settings.train_scope)
        if settings.train_scope == "lora_only":
            optimizer = _rebuild_optimizer_from_live_parameters(optimizer, trainable)
            logging.info(
                "[QAD] rebuilt optimizer after FSDP for lora_only: %d live A/B tensors",
                len(trainable),
            )
        _validate_optimizer_membership(pipe.transformer, optimizer)
        return pipe, optimizer, device_mesh

    auto_pipeline._apply_parallelization = apply_parallelization
    diffusion_train.build_model_and_optimizer = build_model_and_optimizer
    try:
        yield state
    finally:
        diffusion_train.build_model_and_optimizer = original_build_model_and_optimizer
        auto_pipeline._apply_parallelization = original_apply_parallelization

    if apply_calls != 1 or state.parallel_scheme is None:
        raise RuntimeError(
            "QAD did not observe the expected pre-FSDP student parallelization point."
        )
