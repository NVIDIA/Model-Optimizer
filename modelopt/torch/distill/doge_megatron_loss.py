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

"""Megatron-Bridge loss helpers for DoGE distillation."""

from functools import partial

import torch
from megatron.bridge.training.gpt_step import _create_loss_function_modelopt
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import get_model_config

from modelopt.torch.distill.doge_megatron_data import _GPTBatch

__all__ = ["_weighted_source_forward_step"]


def _forward_batch(batch: _GPTBatch, model: GPTModel) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a Megatron GPT forward pass from an already-prepared DoGE batch.

    Mirrors the forward half of ``megatron.bridge.training.gpt_step._forward_step_common()``,
    but uses a DoGE batch that was already sampled from a source-specific iterator.
    """
    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_metadata = batch
    config = get_model_config(model)
    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if packed_seq_metadata is not None:
        if getattr(config, "is_hybrid_model", False):
            if tokens is not None:
                packed_seq_metadata["total_tokens"] = tokens.size(1)
            elif labels is not None:
                packed_seq_metadata["total_tokens"] = labels.size(1)
            else:
                packed_seq_metadata["total_tokens"] = getattr(config, "seq_length", None)
        forward_args["packed_seq_params"] = get_packed_seq_params(packed_seq_metadata)

    return model(**forward_args), loss_mask


def _weighted_loss(
    loss: torch.Tensor,
    num_tokens: torch.Tensor,
    report: dict[str, torch.Tensor],
    _output_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Adapt the precomputed weighted DoGE loss to Megatron-Bridge's loss API.

    Megatron-Bridge forward steps return ``(output_tensor, loss_function)`` and later call
    ``loss_function(output_tensor)``. DoGE computes KD losses immediately after each source
    forward, so this function ignores ``output_tensor`` and returns the already-combined
    ``(loss, num_tokens, report)`` tuple expected by the training loop. The caller passes
    ``loss`` itself as the dummy ``output_tensor`` because returning ``None`` would be a more
    fragile violation of the tensor-valued forward-step contract.
    """
    return loss, num_tokens, report


def _weighted_source_forward_step(
    state: GlobalState,
    source_batches: dict[str, _GPTBatch],
    model: GPTModel,
    blend_weights: dict[str, float],
    return_schedule_plan: bool,
) -> tuple[torch.Tensor, partial]:
    """Return Megatron's inner-loop weighted source loss.

    This function computes one source KD loss per batch in ``source_batches`` and combines them
    using ``blend_weights``. Megatron-Bridge then backpropagates the returned loss and updates the
    student with its normal optimizer step.

    DoGE runs one forward pass per training source. The ModelOpt ``DistillationModel`` wrapper
    stores teacher/student activations from its latest forward pass on the wrapped modules, so each
    source's KD loss must be computed immediately after that source forward. Otherwise, the next
    source forward would overwrite the activations needed by ``compute_kd_loss``.
    """
    if return_schedule_plan:
        raise NotImplementedError(
            "DoGE weighted source forward step does not support schedule plans yet."
        )

    total_loss = None
    loss_num_tokens = None
    total_report: dict[str, torch.Tensor] = {}

    for path, batch in source_batches.items():
        output, loss_mask = _forward_batch(batch, model)
        # Same ModelOpt KD loss builder used by
        # megatron.bridge.training.gpt_step.forward_step_modelopt().
        loss_function = _create_loss_function_modelopt(
            loss_mask,
            model,
            check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
            check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
        )
        source_loss, source_num_tokens, source_report = loss_function(output)
        weight = blend_weights[path]
        # Convert each source loss to a per-token average before weighting. ``clamp`` avoids
        # divide-by-zero if a batch has no valid loss tokens.
        weighted_loss = weight * source_loss / torch.clamp(source_num_tokens, min=1)
        weighted_report_tokens = weight * source_num_tokens

        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        # Megatron-Core expects integer-like token counts when aggregating the returned loss.
        # Since ``total_loss`` is already normalized above, return a denominator of one with the
        # same shape/device/dtype as ``source_num_tokens`` instead of a weighted float token count.
        # Real token counts are still preserved in ``total_report`` for logging.
        loss_num_tokens = (
            torch.ones_like(source_num_tokens) if loss_num_tokens is None else loss_num_tokens
        )
        # Match the report format consumed by Megatron-Bridge train_step(), which expects each
        # metric value to contain ``[loss_numerator, num_tokens]`` and reduces it as
        # ``sum(loss_numerator) / sum(num_tokens)`` for logging.
        for name, value in source_report.items():
            weighted_value = torch.cat(
                [(weight * value[0]).view(1), weighted_report_tokens.view(1)]
            )
            total_report[name] = (
                weighted_value if name not in total_report else total_report[name] + weighted_value
            )

    if total_loss is None or loss_num_tokens is None:
        raise RuntimeError("DoGE weighted source loss requires at least one source batch.")

    # Bridge requires an ``output_tensor`` plus a loss function. The real weighted KD loss is
    # already computed above, so the scalar ``total_loss`` is also used as the ignored dummy output
    # tensor. This is only valid for the current non-pipeline-parallel PoC.
    return total_loss, partial(_weighted_loss, total_loss, loss_num_tokens, total_report)
