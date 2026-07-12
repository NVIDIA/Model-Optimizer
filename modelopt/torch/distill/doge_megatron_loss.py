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

import contextlib
from functools import partial

import torch
from megatron.bridge.training.gpt_step import _create_loss_function_modelopt
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import get_model_config

from modelopt.torch.distill.doge_megatron_data import _GPTBatch

__all__ = ["calc_alignment_gradient_vector", "weighted_source_forward_step"]


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


def weighted_source_forward_step(
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


# The functions below compute selected-parameter gradients for the DoGE outer loop.
# They reuse the same Megatron-Bridge GPT forward and ModelOpt KD-loss path as the training loss,
# but call ``torch.autograd.grad`` for scoring instead of backpropagating the full training loss.
def _clear_model_grads(model: GPTModel) -> None:
    """Clear gradients that should not leak into the Megatron optimizer step."""
    if hasattr(model, "zero_grad_buffer"):
        # Mirrors Megatron-Bridge ``training.train.train_step()``, which clears DDP gradient
        # buffers with ``model_chunk.zero_grad_buffer()`` before calling ``optimizer.zero_grad()``.
        # This is needed because Megatron DDP stores gradients in communication buffers, not only
        # in ``parameter.grad``.
        model.zero_grad_buffer()
    for parameter in model.parameters():
        parameter.grad = None
        if hasattr(parameter, "main_grad") and parameter.main_grad is not None:
            parameter.main_grad.zero_()


def _get_alignment_parameters(model: GPTModel) -> list[torch.nn.Parameter]:
    """Return hardcoded Qwen3-8B final-MLP projection parameters for DoGE scoring."""
    # TODO: Temporary DoGE PoC selector for Qwen3-8B. DoGE computes source and target KD gradients
    # on this parameter, compares their directions, and uses the score to update data-blend
    # weights. Make this configurable before using other models.
    alignment_param_suffix = "decoder.layers.35.mlp.linear_fc2.weight"
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.endswith(alignment_param_suffix)
    ]
    if not parameters:
        raise RuntimeError(
            "DoGE alignment parameter not found: expected a trainable parameter ending with "
            f"{alignment_param_suffix!r}."
        )
    return parameters


def calc_alignment_gradient_vector(
    state: GlobalState,
    batch: _GPTBatch,
    model: GPTModel,
) -> torch.Tensor:
    """Return a flattened selected-parameter KD gradient vector for one DoGE batch.

    The PoC uses Qwen3-8B's final MLP output projection as a cheap approximation to full-model
    DoGE gradients. ``torch.autograd.grad`` computes gradients only for that selected parameter and
    avoids populating optimizer-visible ``.grad`` fields; gradients are still cleared before and
    after scoring as a safety guard.

    The forward/loss construction mirrors ``megatron.bridge.training.gpt_step.forward_step_modelopt``:
    run the GPT forward pass, build the ModelOpt KD loss with ``_create_loss_function_modelopt()``,
    and call the returned loss function. The gradient clearing mirrors
    ``megatron.bridge.training.train.train_step()``. DoGE differs from Bridge's normal train step by
    using ``torch.autograd.grad`` on selected parameters for scoring instead of calling backward on
    the full training loss.

    TODO: This repeats the forward/KD-loss construction used by ``weighted_source_forward_step()``.
    Refactor the shared loss computation so DoGE scoring and weighted source training use one code
    path.
    """
    _clear_model_grads(model)
    parameters = _get_alignment_parameters(model)
    no_sync = model.no_sync() if hasattr(model, "no_sync") else contextlib.nullcontext()
    with no_sync:
        output, loss_mask = _forward_batch(batch, model)
        loss_function = _create_loss_function_modelopt(
            loss_mask,
            model,
            check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
            check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
        )
        loss, num_tokens, _ = loss_function(output)
        loss = loss / torch.clamp(num_tokens, min=1)
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    vector = torch.cat(
        [
            torch.zeros_like(parameter, dtype=torch.float32).reshape(-1)
            if gradient is None
            else gradient.detach().float().reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
    )
    _clear_model_grads(model)
    # BF16 probe backward can overflow a small number of entries in this large projection layer.
    # DoGE scoring only needs a direction; zero non-finite entries so they do not poison the score.
    return torch.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
