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

"""ModelOpt conversion and restoration for DASC policy metadata."""

import copy
import warnings
from collections.abc import Iterable

from pydantic import ValidationError
from torch import nn

from modelopt.torch.opt.conversion import ApplyModeError, ModeloptStateManager
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict
from modelopt.torch.utils import unwrap_model

from .config import DASCCalibrationMeasurement, DASCConfig, DASCPolicy
from .policy import build_dasc_policy, validate_dasc_decay_parameters, validate_dasc_model_structure

__all__ = []

_DASC_POLICY_ATTRIBUTE = "_modelopt_dasc_policy"


def _attach_policy(model: nn.Module, policy: DASCPolicy) -> None:
    """Attach a JSON-safe DASC policy to an unwrapped model."""
    setattr(model, _DASC_POLICY_ATTRIBUTE, policy.model_dump(mode="json"))


def convert_dasc_model(
    model: nn.Module,
    config: DASCConfig,
    *,
    measurements: Iterable[DASCCalibrationMeasurement | dict] | None = None,
) -> ConvertReturnType:
    """Analyze GDN decay and attach the selected DASC policy without changing execution."""
    if measurements is None:
        raise ApplyModeError(
            "DASC requires calibration measurements; use "
            "modelopt.torch.sparsity.state_sparsity.calibrate(model, config, measurements)"
        )
    policy = build_dasc_policy(model, config, measurements)
    _attach_policy(model, policy)
    return model, {"policy": policy.model_dump(mode="json")}


def restore_dasc_model(model: nn.Module, config: DASCConfig, metadata: MetadataDict) -> nn.Module:
    """Restore and structurally validate a serialized DASC policy."""
    if set(metadata) != {"policy"}:
        raise ApplyModeError("DASC metadata must contain only the policy field")
    try:
        policy = DASCPolicy(**metadata["policy"])
    except (TypeError, ValidationError) as error:
        raise ApplyModeError(f"Invalid DASC policy metadata: {error}") from error

    expected_config = {
        "variant": config.variant,
        "epsilon": config.epsilon,
        "static_gate_input": config.static_gate_input,
        "decay_parameter_storage_dtype": config.decay_parameter_storage_dtype,
        "wmax_candidates": config.wmax_candidates,
        "quality_gates": {
            "min_perplexity_retention": config.min_perplexity_retention,
            "min_top1_agreement": config.min_top1_agreement,
            "min_checkpoint_savings": config.min_checkpoint_savings,
        },
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "model_config_id": config.model_config_id,
        "calibration_data_id": config.calibration_data_id,
        "granularity": config.granularity,
        "preserve_convolution_state": config.preserve_convolution_state,
    }
    mismatched = {
        key: (value, getattr(policy, key))
        for key, value in expected_config.items()
        if value != getattr(policy, key)
    }
    if mismatched:
        raise ApplyModeError(f"DASC policy metadata does not match its mode config: {mismatched}")

    try:
        validate_dasc_model_structure(model, policy)
    except ApplyModeError as error:
        warnings.warn(
            f"{error}. The restored DASC policy is stale; re-run calibrate() before deployment",
            stacklevel=2,
        )
    _attach_policy(model, policy)
    return model


def update_dasc_metadata(model: nn.Module, config: DASCConfig, metadata: MetadataDict) -> None:
    """Refresh metadata without making unrelated ModelOpt save or compose paths unusable."""
    policy = get_attached_dasc_policy(model)
    try:
        validate_dasc_model_structure(model, policy)
        validate_dasc_decay_parameters(model, policy)
    except ApplyModeError as error:
        warnings.warn(
            f"{error}. The saved DASC policy is stale; re-run calibrate() before deployment",
            stacklevel=2,
        )
    metadata.clear()
    metadata["policy"] = copy.deepcopy(policy.model_dump(mode="json"))


def replace_dasc_mode(
    model: nn.Module,
    config: DASCConfig,
    measurements: Iterable[DASCCalibrationMeasurement | dict],
) -> nn.Module:
    """Replace existing DASC mode state in place with a newly derived policy."""
    model = unwrap_model(model, force_unwrap=True)
    manager = ModeloptStateManager(model)
    state = manager.state_dict()
    dasc_indices = [index for index, (mode, _) in enumerate(state) if mode == "dasc"]
    if not dasc_indices:
        raise ApplyModeError("Cannot replace DASC mode because the model has no DASC state")
    policy = build_dasc_policy(model, config, measurements)
    if dasc_indices[-1] != len(state) - 1:
        manager.update_last_state_before_new_mode(model)

    first_index = dasc_indices[0]
    state[first_index] = (
        "dasc",
        {
            "config": config.model_dump(),
            "metadata": {"policy": policy.model_dump(mode="json")},
        },
    )
    for index in reversed(dasc_indices[1:]):
        del state[index]
    _attach_policy(model, policy)
    return model


def get_attached_dasc_policy(model: nn.Module) -> DASCPolicy:
    """Return the validated policy attached by conversion or restoration."""
    model = unwrap_model(model, force_unwrap=True)
    try:
        return DASCPolicy(**getattr(model, _DASC_POLICY_ATTRIBUTE))
    except (AttributeError, TypeError, ValidationError) as error:
        raise ApplyModeError("Model has no valid attached DASC policy") from error
