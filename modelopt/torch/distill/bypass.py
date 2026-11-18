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

"""One-call Bypass repair of an existing IQ-quantized student."""

import logging

import torch
from torch import nn

from modelopt.torch.quantization.repair import plan_iq_repair

from ._bypass_capture import ModuleBypassAdapter
from ._bypass_runner import _BypassRunner
from .config import BypassConfig

__all__ = ["bypass"]


def bypass(
    student: nn.Module,
    teacher: nn.Module,
    train_dataloader,
    config=None,
    *,
    validation_dataloader=None,
    device_mesh=None,
) -> nn.Module:
    """Optimize IQ native scales on streamed frozen teacher boundary inputs.

    Args:
        student: Existing ModelOpt IQ PTQ model; frozen formats may coexist.
        teacher: Separate original-weight callable with matching boundary paths.
        train_dataloader: Unique sample batches that exactly fill each global update.
            Dict batches accept boolean/integer ``valid_mask`` or ``loss_mask``,
            in that order, with nonzero meaning valid rather than a token weight.
            Otherwise, ``attention_mask`` must be 2D boolean/integer. Floating or
            4D attention masks require an explicit validity mask. Weighted masks
            are unsupported; other metadata does not define the loss mask.
        config: A :class:`BypassConfig`, a dict, or None for the established defaults.
        validation_dataloader: Optional held-out batches for finalized-student valid-token MSE.
            Diagnostic failures propagate after committing training updates to the student.
            Zero-step preflight skips validation as well as training.
        device_mesh: Caller-owned placement mesh, adopted by supported integrations.

    Returns:
        The supplied student, finalized for ordinary ModelOpt save/export. Temporary
        fields, hooks, modes and gradient flags are restored on every exit.
    """
    config = config if isinstance(config, BypassConfig) else BypassConfig(**(config or {}))
    if device_mesh is not None or (
        torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
    ):
        raise ValueError("Distributed Bypass requires its native placement integration")
    adapter: ModuleBypassAdapter
    if getattr(getattr(student, "config", None), "model_type", None):
        from .plugins.transformers import TransformersBypassAdapter

        adapter = TransformersBypassAdapter()
    else:
        adapter = ModuleBypassAdapter()
    units = adapter.resolve_units(student, teacher, config)
    plan = plan_iq_repair(
        [student.get_submodule(u.student_path) for u in units], root_model=student
    )
    logging.getLogger(__name__).info("Bypass selected boundaries: %s", [u.name for u in units])
    if config.num_steps == 0:
        return student
    return _BypassRunner(student, teacher, units, plan, config, adapter).run(
        train_dataloader, validation_dataloader
    )
