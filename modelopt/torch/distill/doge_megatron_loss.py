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
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Literal, TypedDict

import torch
import torch.distributed as dist
from megatron.bridge.training.gpt_step import _create_loss_function_modelopt
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import get_model_config

from modelopt.torch.distill.doge_megatron_data import _GPTBatch

DoGEAlignmentParamScope = Literal["final_mlp", "all_trainable"]

__all__ = [
    "DoGEAlignmentDiagnostics",
    "DoGEAlignmentParamScope",
    "DoGEVirtualStepDiagnostic",
    "compute_alignment_scores",
    "compute_virtual_step_diagnostics",
    "sampled_source_forward_step",
    "weighted_source_forward_step",
    "zero_sampled_source_forward_step",
    "zero_weighted_source_forward_step",
]


@dataclass(frozen=True)
class DoGEAlignmentDiagnostics:
    """Gradient-alignment diagnostics for one DoGE outer-loop step."""

    scores: dict[str, float]
    alignment_debug: dict[str, dict[str, float | int]]
    source_probe_kd_loss: dict[str, float]
    target_probe_kd_loss: float
    source_gradients: dict[str, torch.Tensor]
    target_gradient: torch.Tensor


class DoGEVirtualStepDiagnostic(TypedDict):
    """Target-probe result after virtual selected-scope steps."""

    blend_weights: dict[str, float]
    target_probe_kd_before: float
    target_probe_kd_after: float
    delta_target_probe_kd: float
    virtual_gradient_norm: float
    virtual_update_norm: float
    virtual_total_update_norm: float
    virtual_step_lr: float
    virtual_step_num_steps: int


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


def _zero_loss(
    loss_function: partial,
    output_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return a zeroed loss while preserving the backward graph from the real source loss."""
    loss, num_tokens, report = loss_function(output_tensor)
    return loss * 0.0, num_tokens, report


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


def sampled_source_forward_step(
    state: GlobalState,
    source_batches: dict[str, _GPTBatch],
    selected_source_path: str,
    model: GPTModel,
    return_schedule_plan: bool,
) -> tuple[torch.Tensor, partial]:
    """Return Megatron's inner-loop loss for one sampled source.

    Unlike ``weighted_source_forward_step()``, this path computes exactly one source loss and
    returns it unweighted. Use it when DoGE blend weights should behave like real sampled data
    probabilities instead of per-step loss coefficients.
    """
    if return_schedule_plan:
        raise NotImplementedError(
            "DoGE sampled source forward step does not support schedule plans yet."
        )
    if selected_source_path not in source_batches:
        raise ValueError(f"sampled source batch not found: {selected_source_path}")

    output, loss_mask = _forward_batch(source_batches[selected_source_path], model)
    loss_function = _create_loss_function_modelopt(
        loss_mask,
        model,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
    )
    return output, loss_function


def zero_weighted_source_forward_step(
    state: GlobalState,
    source_batches: dict[str, _GPTBatch],
    model: GPTModel,
    blend_weights: dict[str, float],
    return_schedule_plan: bool,
) -> tuple[torch.Tensor, partial]:
    """Return the weighted source graph with a zero loss for frozen-student DoGE runs.

    Megatron DDP expects the backward pass to visit the same parameters as a real training step.
    A scalar zero loss attached to only one parameter leaves DDP's gradient-ready state incomplete,
    so frozen-student mode builds the normal weighted source loss and zeroes it at the loss-function
    boundary.
    """
    output_tensor, loss_function = weighted_source_forward_step(
        state,
        source_batches,
        model,
        blend_weights,
        return_schedule_plan,
    )
    return output_tensor, partial(_zero_loss, loss_function)


def zero_sampled_source_forward_step(
    state: GlobalState,
    source_batches: dict[str, _GPTBatch],
    selected_source_path: str,
    model: GPTModel,
    return_schedule_plan: bool,
) -> tuple[torch.Tensor, partial]:
    """Return a sampled-source graph with a zero loss for frozen-student DoGE runs."""
    output_tensor, loss_function = sampled_source_forward_step(
        state,
        source_batches,
        selected_source_path,
        model,
        return_schedule_plan,
    )
    return output_tensor, partial(_zero_loss, loss_function)


# The functions below compute selected-scope gradients for the DoGE outer loop.
# They reuse the same Megatron-Bridge GPT forward and ModelOpt KD-loss path as the training loss,
# but read only the selected gradients before clearing them so scoring does not affect training.
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


def _get_alignment_parameters(
    model: GPTModel, alignment_param_scope: DoGEAlignmentParamScope
) -> list[torch.nn.Parameter]:
    """Return parameters used for DoGE gradient-alignment scoring."""
    if alignment_param_scope == "all_trainable":
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("DoGE alignment found no trainable parameters.")
        return parameters

    if alignment_param_scope == "final_mlp":
        # TODO: Temporary DoGE PoC selector for Qwen3-8B. DoGE computes source and target KD
        # gradients on this parameter, compares their directions, and uses the score to update
        # data-blend weights.
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

    raise ValueError(f"Unsupported DoGE alignment parameter scope: {alignment_param_scope!r}")


def _read_parameter_gradient(parameter: torch.nn.Parameter) -> torch.Tensor:
    """Return a flattened gradient from Megatron's grad buffer or ``parameter.grad``."""
    gradient = getattr(parameter, "main_grad", None)
    if gradient is None:
        gradient = parameter.grad
    if gradient is None:
        return torch.zeros_like(parameter, dtype=torch.float32).reshape(-1)
    return gradient.detach().float().reshape(-1)


def _calc_alignment_gradient_vector(
    state: GlobalState,
    batch: _GPTBatch,
    model: GPTModel,
    alignment_param_scope: DoGEAlignmentParamScope,
) -> tuple[torch.Tensor, float]:
    """Return a KD gradient vector and normalized KD loss for the selected parameter scope.

    The PoC uses Qwen3-8B's final MLP output projection as an approximation to full-model DoGE
    gradients by default. ``all_trainable`` is an experiment-only diagnostic scope that reads
    gradients for every trainable local parameter shard. Gradients are cleared before and after
    scoring so they do not leak into the optimizer step.

    The forward/loss construction mirrors ``megatron.bridge.training.gpt_step.forward_step_modelopt``:
    run the GPT forward pass, build the ModelOpt KD loss with ``_create_loss_function_modelopt()``,
    and call the returned loss function. The gradient clearing mirrors
    ``megatron.bridge.training.train.train_step()``. DoGE differs from Bridge's normal train step by
    calling backward for scoring and then reading only selected parameter gradients before clearing
    them.

    TODO: This repeats the forward/KD-loss construction used by ``weighted_source_forward_step()``.
    Refactor the shared loss computation so DoGE scoring and weighted source training use one code
    path.
    """
    _clear_model_grads(model)
    parameters = _get_alignment_parameters(model, alignment_param_scope)
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
        normalized_loss = loss / torch.clamp(num_tokens, min=1)
        normalized_loss.backward()
    probe_loss = _reduce_probe_loss(loss, num_tokens)
    vector = torch.cat([_read_parameter_gradient(parameter) for parameter in parameters])
    _clear_model_grads(model)
    # BF16 probe backward can overflow a small number of entries in the selected gradient vector.
    # DoGE scoring only needs a direction; zero non-finite entries so they do not poison the score.
    return torch.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0), probe_loss


def _calc_probe_kd_loss(state: GlobalState, batch: _GPTBatch, model: GPTModel) -> float:
    """Return normalized KD loss for one already-prepared DoGE batch."""
    _clear_model_grads(model)
    with torch.no_grad():
        output, loss_mask = _forward_batch(batch, model)
        loss_function = _create_loss_function_modelopt(
            loss_mask,
            model,
            check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
            check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
        )
        loss, num_tokens, _ = loss_function(output)
    probe_loss = _reduce_probe_loss(loss, num_tokens)
    _clear_model_grads(model)
    return probe_loss


def _reduce_probe_loss(loss: torch.Tensor, num_tokens: torch.Tensor) -> float:
    """Return the globally normalized KD loss for one DoGE probe batch."""
    components = torch.stack(
        [
            loss.detach().float().reshape(()),
            num_tokens.detach().float().reshape(()),
        ]
    )
    if dist.is_available() and dist.is_initialized():
        # PoC synchronization matches alignment-score reduction. With the current setup PP=DP=CP=1
        # and the default group is TP-only. TODO: reduce over exact model/data-parallel groups when
        # adding support for other parallelism layouts.
        dist.all_reduce(components, op=dist.ReduceOp.SUM)
    return (components[0] / torch.clamp(components[1], min=1)).item()


def _calc_alignment_score(
    source_gradient: torch.Tensor, target_gradient: torch.Tensor
) -> tuple[float, dict[str, float]]:
    """Return one DoGE source-to-target alignment score.

    Args:
        source_gradient: Flattened selected-scope KD gradient for one training source.
        target_gradient: Flattened selected-scope KD gradient for the target objective.

    Returns:
        A cosine-similarity score and debug values used to compute it. Higher scores mean the
        source gradient points more toward the target gradient and should increase that source's
        blend weight.

    The gradients are scaled before dot/norm computation only for numeric stability. Cosine
    similarity is unchanged by dividing both vectors by the same positive scale.
    """
    scale = torch.maximum(source_gradient.abs().max(), target_gradient.abs().max())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(scale, op=dist.ReduceOp.MAX)
    scaled_source_gradient = source_gradient / scale if scale.item() != 0 else source_gradient
    scaled_target_gradient = target_gradient / scale if scale.item() != 0 else target_gradient

    # Compute a global cosine similarity:
    # dot(source, target) / (norm(source) * norm(target)).
    # Reduce dot/norm components before forming the cosine so TP-sharded parameter slices
    # contribute to one score instead of averaging per-rank local cosines.
    score_components = torch.stack(
        [
            torch.dot(scaled_source_gradient, scaled_target_gradient),
            torch.dot(scaled_source_gradient, scaled_source_gradient),
            torch.dot(scaled_target_gradient, scaled_target_gradient),
        ]
    )
    if dist.is_available() and dist.is_initialized():
        # PoC synchronization: with the current Qwen3-8B setup PP=DP=CP=1 and the default
        # group is TP-only. TODO: reduce over the exact model-parallel group when adding
        # support for other parallelism layouts.
        dist.all_reduce(score_components, op=dist.ReduceOp.SUM)
    source_norm = torch.sqrt(score_components[1])
    target_norm = torch.sqrt(score_components[2])
    denominator = source_norm * target_norm
    score = (
        torch.zeros((), dtype=source_gradient.dtype, device=source_gradient.device)
        if denominator.item() == 0
        else score_components[0] / denominator
    )
    return score.item(), {
        "dot": score_components[0].item(),
        "source_norm": source_norm.item(),
        "target_norm": target_norm.item(),
    }


def compute_alignment_scores(
    state: GlobalState,
    source_batches: dict[str, _GPTBatch],
    target_batch: _GPTBatch,
    model: GPTModel,
    blend_weights: dict[str, float],
    alignment_param_scope: DoGEAlignmentParamScope,
) -> DoGEAlignmentDiagnostics:
    """Compute source-to-target gradient-alignment scores for DoGE weight updates.

    Scores are keyed by the same dataset paths as ``source_batches``. Higher scores should increase
    a source's DoGE blend weight. The debug dictionary records the reduced cosine-similarity
    components used to compute each score plus dot-product diagnostics for the current blend.
    Probe KD losses are one-batch diagnostics from the same batches used for alignment scoring.
    """
    target_gradient, target_probe_kd_loss = _calc_alignment_gradient_vector(
        state, target_batch, model, alignment_param_scope
    )
    target_token_sum = int(target_batch[0].sum().item())
    scores = {}
    alignment_debug = {}
    source_probe_kd_loss = {}
    source_gradients = {}
    for path, batch in source_batches.items():
        source_gradient, probe_loss = _calc_alignment_gradient_vector(
            state, batch, model, alignment_param_scope
        )
        score, debug = _calc_alignment_score(source_gradient, target_gradient)
        source_gradients[path] = source_gradient
        scores[path] = score
        source_probe_kd_loss[path] = probe_loss
        alignment_debug[path] = {
            "source_token_sum": int(batch[0].sum().item()),
            "target_token_sum": target_token_sum,
            **debug,
        }
    advantage_debug = _calc_advantage_debug(source_gradients, target_gradient, blend_weights)
    for path, debug in advantage_debug.items():
        alignment_debug[path].update(debug)
    return DoGEAlignmentDiagnostics(
        scores=scores,
        alignment_debug=alignment_debug,
        source_probe_kd_loss=source_probe_kd_loss,
        target_probe_kd_loss=target_probe_kd_loss,
        source_gradients=source_gradients,
        target_gradient=target_gradient,
    )


def compute_virtual_step_diagnostics(
    state: GlobalState,
    source_batches: Mapping[str, _GPTBatch],
    source_gradients: Mapping[str, torch.Tensor],
    target_batch: _GPTBatch,
    model: GPTModel,
    candidate_blend_weights: Mapping[str, Mapping[str, float]],
    virtual_step_lr: float,
    target_probe_kd_loss: float,
    alignment_param_scope: DoGEAlignmentParamScope,
    virtual_step_num_steps: int,
) -> dict[str, DoGEVirtualStepDiagnostic]:
    """Measure target-KD change after virtual selected-scope steps for candidate blends.

    For one-step diagnostics, this helper mixes the already-computed source gradients, temporarily
    applies ``param -= virtual_step_lr * mixed_gradient`` to the selected DoGE alignment parameters,
    evaluates target KD on the same target batch, and restores the original parameter values.
    For multi-step diagnostics, it recomputes source gradients on the same source batches after
    every virtual parameter update before applying the next virtual update.
    The real model and real blend weights are unchanged.
    """
    if virtual_step_num_steps < 1:
        raise ValueError("DoGE virtual-step diagnostics require at least one virtual step.")

    diagnostics: dict[str, DoGEVirtualStepDiagnostic] = {}
    for label, weights in candidate_blend_weights.items():
        with _preserve_alignment_parameters(model, alignment_param_scope) as parameters:
            virtual_gradient_norm = 0.0
            virtual_total_update_norm = 0.0
            for _ in range(virtual_step_num_steps):
                step_source_gradients = (
                    source_gradients
                    if virtual_step_num_steps == 1
                    else _calc_source_gradient_vectors(
                        state, source_batches, model, alignment_param_scope
                    )
                )
                mixed_gradient = _mix_source_gradients(step_source_gradients, weights)
                virtual_gradient_norm = _reduced_vector_norm(mixed_gradient)
                virtual_update_norm = virtual_step_lr * virtual_gradient_norm
                virtual_total_update_norm += virtual_update_norm
                _apply_alignment_parameter_step(parameters, mixed_gradient, virtual_step_lr)

            target_probe_kd_after = _calc_probe_kd_loss(state, target_batch, model)
        diagnostics[label] = {
            "blend_weights": dict(weights),
            "target_probe_kd_before": target_probe_kd_loss,
            "target_probe_kd_after": target_probe_kd_after,
            "delta_target_probe_kd": target_probe_kd_after - target_probe_kd_loss,
            "virtual_gradient_norm": virtual_gradient_norm,
            "virtual_update_norm": virtual_step_lr * virtual_gradient_norm,
            "virtual_total_update_norm": virtual_total_update_norm,
            "virtual_step_lr": virtual_step_lr,
            "virtual_step_num_steps": virtual_step_num_steps,
        }
    return diagnostics


def _calc_source_gradient_vectors(
    state: GlobalState,
    source_batches: Mapping[str, _GPTBatch],
    model: GPTModel,
    alignment_param_scope: DoGEAlignmentParamScope,
) -> dict[str, torch.Tensor]:
    """Recompute selected-scope source KD gradients for the current virtual parameters."""
    return {
        path: _calc_alignment_gradient_vector(state, batch, model, alignment_param_scope)[0]
        for path, batch in source_batches.items()
    }


def _mix_source_gradients(
    source_gradients: Mapping[str, torch.Tensor], blend_weights: Mapping[str, float]
) -> torch.Tensor:
    """Return ``sum_i weight_i * source_gradient_i`` for one candidate blend."""
    mixed_gradient = None
    for path, gradient in source_gradients.items():
        weighted_gradient = blend_weights[path] * gradient
        mixed_gradient = (
            weighted_gradient if mixed_gradient is None else mixed_gradient + weighted_gradient
        )
    if mixed_gradient is None:
        raise RuntimeError("DoGE virtual-step diagnostics require at least one source gradient.")
    return mixed_gradient


def _reduced_vector_norm(vector: torch.Tensor) -> float:
    """Return global vector norm for the currently TP-sharded DoGE gradient vector."""
    squared_norm = torch.dot(vector, vector)
    if dist.is_available() and dist.is_initialized():
        # Same TP-only PoC reduction as the alignment-score path. TODO: reduce over the exact
        # model-parallel group when adding support for other parallelism layouts.
        dist.all_reduce(squared_norm, op=dist.ReduceOp.SUM)
    return torch.sqrt(squared_norm).item()


@contextlib.contextmanager
def _temporary_alignment_parameter_step(
    model: GPTModel,
    gradient_vector: torch.Tensor,
    virtual_step_lr: float,
    alignment_param_scope: DoGEAlignmentParamScope,
):
    """Temporarily apply a selected-scope SGD step and restore original values."""
    with _preserve_alignment_parameters(model, alignment_param_scope) as parameters:
        _apply_alignment_parameter_step(parameters, gradient_vector, virtual_step_lr)
        yield


@contextlib.contextmanager
def _preserve_alignment_parameters(
    model: GPTModel,
    alignment_param_scope: DoGEAlignmentParamScope,
):
    """Preserve selected-scope parameters while virtual diagnostic updates run."""
    parameters = _get_alignment_parameters(model, alignment_param_scope)
    original_values = [parameter.detach().clone() for parameter in parameters]
    try:
        yield parameters
    finally:
        with torch.no_grad():
            for parameter, original_value in zip(parameters, original_values):
                parameter.copy_(original_value)
        _clear_model_grads(model)


def _apply_alignment_parameter_step(
    parameters: list[torch.nn.Parameter],
    gradient_vector: torch.Tensor,
    virtual_step_lr: float,
) -> None:
    """Apply one virtual SGD step to selected parameters."""
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            numel = parameter.numel()
            # ``gradient_vector`` concatenates gradients for all selected parameters. Slice this
            # parameter's shard and apply one virtual SGD step:
            #     theta <- theta - virtual_step_lr * grad
            # PyTorch ``add_(grad, alpha=-lr)`` performs ``theta += -lr * grad`` in place.
            gradient = gradient_vector[offset : offset + numel].view_as(parameter)
            parameter.add_(gradient.to(dtype=parameter.dtype), alpha=-virtual_step_lr)
            offset += numel
        if offset != gradient_vector.numel():
            raise RuntimeError(
                "DoGE virtual-step gradient vector does not match selected parameter size: "
                f"used {offset} values from {gradient_vector.numel()}."
            )


def _calc_advantage_debug(
    source_gradients: dict[str, torch.Tensor],
    target_gradient: torch.Tensor,
    blend_weights: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Return the predicted target-loss benefit of shifting weight toward each source.

    ``advantage_scaled_dot`` is a numerically scaled version of
    ``dot(g_target, g_source - g_mix)``. Under the first-order Taylor approximation, positive
    values mean moving weight toward that source is predicted to reduce target loss more than
    keeping the current blend. The shared positive scale preserves signs and ordering but not
    absolute magnitudes.
    """
    mix_gradient = None
    for path, gradient in source_gradients.items():
        weighted_gradient = blend_weights[path] * gradient
        mix_gradient = (
            weighted_gradient if mix_gradient is None else mix_gradient + weighted_gradient
        )
    if mix_gradient is None:
        raise RuntimeError("DoGE advantage diagnostics require at least one source gradient.")

    scale = torch.maximum(target_gradient.abs().max(), mix_gradient.abs().max())
    for gradient in source_gradients.values():
        scale = torch.maximum(scale, gradient.abs().max())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(scale, op=dist.ReduceOp.MAX)

    if scale.item() != 0:
        scaled_target_gradient = target_gradient / scale
        scaled_mix_gradient = mix_gradient / scale
        scaled_source_gradients = {
            path: gradient / scale for path, gradient in source_gradients.items()
        }
    else:
        scaled_target_gradient = target_gradient
        scaled_mix_gradient = mix_gradient
        scaled_source_gradients = source_gradients

    mix_dot = torch.dot(scaled_target_gradient, scaled_mix_gradient)
    mix_norm = torch.dot(scaled_mix_gradient, scaled_mix_gradient)
    target_norm = torch.dot(scaled_target_gradient, scaled_target_gradient)
    components = [mix_dot, mix_norm, target_norm]
    for gradient in scaled_source_gradients.values():
        components.extend(
            [
                torch.dot(scaled_target_gradient, gradient),
                torch.dot(scaled_target_gradient, gradient - scaled_mix_gradient),
            ]
        )
    reduced = torch.stack(components)
    if dist.is_available() and dist.is_initialized():
        # Same TP-only PoC reduction as cosine scoring above.
        # TODO: reduce over exact model/data-parallel groups for other parallelism layouts.
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

    mix_dot = reduced[0]
    mix_norm = torch.sqrt(reduced[1])
    target_norm = torch.sqrt(reduced[2])
    mix_denominator = mix_norm * target_norm
    mix_cosine = (
        torch.zeros((), dtype=mix_dot.dtype, device=mix_dot.device)
        if mix_denominator.item() == 0
        else mix_dot / mix_denominator
    )

    debug = {}
    offset = 3
    for path in scaled_source_gradients:
        source_target_dot = reduced[offset]
        advantage_dot = reduced[offset + 1]
        debug[path] = {
            "source_target_scaled_dot": source_target_dot.item(),
            "mix_target_scaled_dot": mix_dot.item(),
            "advantage_scaled_dot": advantage_dot.item(),
            "mix_target_cosine": mix_cosine.item(),
        }
        offset += 2
    return debug
