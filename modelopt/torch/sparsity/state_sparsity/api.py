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

"""Public DASC state-sparsity APIs."""

import copy
from collections.abc import Iterable
from typing import Any

from torch import nn

from modelopt.torch.opt.conversion import apply_mode

from .config import DASCCalibrationMeasurement, DASCConfig
from .conversion import get_attached_dasc_policy
from .mode import DASCModeRegistry
from .policy import validate_dasc_decay_parameters, validate_dasc_model_structure

__all__ = ["calibrate", "export_policy"]


def calibrate(
    model: nn.Module,
    config: dict[str, Any] | DASCConfig,
    measurements: Iterable[DASCCalibrationMeasurement | dict],
) -> nn.Module:
    """Calibrate and attach a DASC policy without changing model execution.

    ``measurements`` must contain exactly one entry for every configured ``Wmax`` candidate.
    The largest candidate passing every quality, lifecycle, and storage gate is selected.

    Example::

        import modelopt.torch.sparsity.state_sparsity as mtss

        model = mtss.calibrate(model, config, measurements)
        deployment_policy = mtss.export_policy(model)

    Args:
        model: Model containing GatedDeltaNet modules with one-dimensional ``A_log`` and
            ``dt_bias`` tensors.
        config: Checkpoint provenance, candidate windows, and quality gates.
        measurements: Quality and checkpoint-storage results produced by the caller's paired
            dense-versus-DASC calibration workflow.

    Returns:
        The input model with a serializable DASC policy attached through ModelOpt state.
    """
    config_dict = config.model_dump() if isinstance(config, DASCConfig) else config
    return apply_mode(
        model,
        mode=[("dasc", config_dict)],
        registry=DASCModeRegistry,
        mode_kwargs={"measurements": measurements},
    )


def export_policy(model: nn.Module) -> dict[str, Any]:
    """Export a JSON-safe DASC policy after validating model structure and decay parameters.

    This policy does not implement checkpoint packing or recovery. A serving backend must preserve
    convolution state, store retained complete GDN heads, recover omitted heads according to the
    declared variant, and materialize the ordinary dense runtime state before continuation.
    """
    policy = get_attached_dasc_policy(model)
    validate_dasc_model_structure(model, policy)
    validate_dasc_decay_parameters(model, policy)
    return copy.deepcopy(policy.model_dump(mode="json"))
