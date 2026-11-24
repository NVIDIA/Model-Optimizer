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
        train_dataloader: Per-rank batches that exactly fill each update's sample budget:
            ``config.global_batch_size`` without a mesh, or ``global_batch_size // dp_size``
            with a mesh, where ``dp_size`` counts actual data lanes (including FSDP,
            excluding TP and CP). The global batch must divide exactly over those lanes.
            Data lanes supply unique samples; TP peers supply identical samples and masks.
            CP peers replay identical full sequences and masks; valid-token ownership is
            partitioned across CP ranks, so CP does not create additional sample lanes.
            Ranks must execute the same number of microbatches per update.
            Dict batches accept boolean/integer ``valid_mask`` or ``loss_mask``,
            in that order, with nonzero meaning valid rather than a token weight.
            Otherwise, ``attention_mask`` must be 2D boolean/integer. Floating or
            4D attention masks require an explicit validity mask. Weighted masks
            are unsupported; other metadata does not define the loss mask.
            Batches are snapshotted before the next reader call. Named model inputs
            and loss masks have a leading sample axis; other 1D dict metadata is
            shared across samples. Represent auxiliary sample vectors as ``[batch, 1]``.
            Global Python/NumPy/CPU RNG and CUDA RNG on resident model devices are
            seeded and restored. Readers using other CUDA devices must own their
            generators; reader generators and worker state advance independently.
            The HF adapter additionally derives default validity from ``_packed_seq_ids``:
            zero denotes padding and positive IDs identify documents. Explicit loss masks win.
            HF 2D attention masks must be Boolean/integer; floating additive bias uses 4D.
        config: A :class:`BypassConfig`, a dict, or None for the established defaults.
        validation_dataloader: Optional held-out batches for finalized-student valid-token MSE.
            Batches obey the same sequence length, nonempty batch and validity-mask rules
            as training, with identical TP/CP samples and unique samples across data lanes.
            Ranks must have matching batch-presence and microbatch-count schedules;
            validation need not fill the training global-batch budget.
            Diagnostic failures propagate after committing training updates to the student.
            Zero-step preflight skips validation as well as training.
        device_mesh: Caller-owned placement mesh, adopted by supported integrations.
            It determines the per-rank data-lane budget and TP/CP replication contract
            described above; use a data-lane sampler, rather than sharding across TP/CP.

    Finalization commits boundaries in order. Any failure during finalization may
    leave earlier boundaries committed; reload a saved checkpoint before retrying.
    Persistent owner restoration failures may also retain temporary repair state;
    discard the affected model and reload it before reuse. Cleanup failures are
    attached to the primary exception when one is already propagating.

    For HF PTQ, use
    :meth:`TransformersBypassAdapter.weight_patterns
    <modelopt.torch.distill.plugins.transformers.TransformersBypassAdapter.weight_patterns>`
    with the same selectors to obtain ordinary ModelOpt ``weight_quantizer`` globs
    before quantizing the student. These patterns exclude frozen routing gates.

    Returns:
        The supplied student, finalized for ordinary ModelOpt save/export. Temporary
        fields and hooks are removed unless owner restoration fails; caller modes
        and teacher gradient flags are restored on every exit.
    """
    config = config if isinstance(config, BypassConfig) else BypassConfig(**(config or {}))
    if (
        torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and device_mesh is None
    ):
        raise ValueError("Distributed Bypass requires the caller's actual device_mesh")
    adapter: ModuleBypassAdapter
    if getattr(getattr(student, "config", None), "model_type", None):
        from .plugins.transformers import TransformersBypassAdapter

        adapter = TransformersBypassAdapter(device_mesh)
        adapter.adopt(student)
    else:
        adapter = ModuleBypassAdapter()
        if device_mesh is not None:
            from ._bypass_parallel import torch_mesh_context

            adapter.context = torch_mesh_context(device_mesh)
    context = getattr(adapter, "context", None)
    runtime = None
    error = None
    try:
        if context is not None:
            from ._bypass_runtime import DistributedBypassRuntime

            runtime = DistributedBypassRuntime(context, config)
        units = adapter.resolve_units(student, teacher, config)
        plan = plan_iq_repair(
            [student.get_submodule(u.student_path) for u in units], root_model=student
        )
    except Exception as exc:
        if context is None:
            raise
        error = f"{type(exc).__name__}: {exc}"
    if context is not None:
        from ._bypass_parallel import _all_rank_values

        # Constructor refusals must agree even on ranks that have no runtime yet.
        errors = sorted({value for value in _all_rank_values(context, error) if value})
        if errors:
            raise ValueError("; ".join(errors))
        assert runtime is not None
        schedules = runtime.gather_values(
            (config.model_dump(), [u.name for u in units], validation_dataloader is not None)
        )
        runtime.synchronize_error(
            "Bypass recipes, boundaries and validation presence must agree across model ranks"
            if any(value != schedules[0] for value in schedules)
            else None
        )
    logging.getLogger(__name__).info("Bypass selected boundaries: %s", [u.name for u in units])
    if config.num_steps == 0:
        return student
    return _BypassRunner(student, teacher, units, plan, config, adapter, runtime).run(
        train_dataloader, validation_dataloader
    )
