# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Model validation and loss calculation utilities for single-GPU and multi-GPU setups.

Also provides helper functions for loss metrics, KL divergence, JS divergence,
and similarity losses for knowledge distillation.
"""

# mypy: ignore-errors
import functools
import math
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
from typing_extensions import Self

from modelopt.torch.puzzletron.tools import kd_model


class UnshardedLowMemorySparseTensor:
    def __init__(self, x: torch.Tensor):
        inds_dtype = self._infer_inds_dtype(x)
        x_sparse = x.to_sparse_coo()
        self._values = x_sparse.values()
        self._indices = x_sparse.indices().to(inds_dtype)
        self._size = x_sparse.size()

    @staticmethod
    def _infer_inds_dtype(x: torch.Tensor) -> torch.dtype:
        max_dim = max(x.shape)
        for inds_dtype in [torch.int16, torch.int32, torch.int64]:
            if torch.iinfo(inds_dtype).max >= max_dim:
                return inds_dtype

    def to_sparse_coo(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(values=self._values, indices=self._indices, size=self._size)

    def to_dense(self) -> torch.Tensor:
        return self.to_sparse_coo().to_dense()

    def to(self, *args) -> Self:
        self._values = self._values.to(*args)
        for arg in args:
            if isinstance(arg, torch.device) or isinstance(arg, str):
                self._indices = self._indices.to(arg)
        return self


class LowMemorySparseTensor:
    _max_sparse_size = torch.iinfo(torch.int32).max

    def __init__(self, x: torch.Tensor):
        num_chunks = math.ceil(x.numel() / self._max_sparse_size)
        self._chunk_dim = np.argmax(x.shape)
        self._chunks = [
            UnshardedLowMemorySparseTensor(chunk)
            for chunk in torch.chunk(x, num_chunks, dim=self._chunk_dim)
        ]

    def to(self, *args) -> Self:
        for chunk in self._chunks:
            chunk.to(*args)
        return self

    def to_dense(self) -> torch.Tensor:
        return torch.concat([chunk.to_dense() for chunk in self._chunks], dim=self._chunk_dim)


@torch.no_grad()
def calculate_losses(
    model: nn.Module,
    dataloader: DataLoader,
    target_probs: None = None,
    return_probs: bool = False,
    checkpoint_manager=None,
) -> tuple[dict[str, dict], None] | tuple[None, None]:
    """Do model forward on each batch and calculate LM loss.
    Works on lit-llama models (single gpu) and huggingface models (can be multi gpu).
    Does not support data-parallel.

    ### Anything related to probs and hidden states is not supported currently! ###
    calculate_losses() isn't updated according to the major refactor in
    calculate_losses_pipeline() regarding hidden states.

    Returns:
        outputs = {
            "lm_loss": list[float],
            "token_accuracy_top_1": list[float],
            "token_accuracy_top_5": list[float],
            "token_accuracy_top_10": list[float],
        }
    """
    if (target_probs is not None) or return_probs:
        raise NotImplementedError(
            "calculate_losses() isn't updated according to the major refactor in "
            "calculate_losses_pipeline() regarding hidden states."
        )

    model_device = next(model.parameters()).device
    outputs = []

    try:
        num_batches = len(dataloader)
    except:
        num_batches = None

    # Adjust progress bar for resume
    start_batch = checkpoint_manager.current_batch if checkpoint_manager else 0
    progress_bar = tqdm(
        enumerate(dataloader),
        total=num_batches,
        desc=f"calculate_losses({(target_probs is None)=}, {return_probs=})",
    )
    if start_batch > 0:
        progress_bar.update(start_batch)

    for i_batch, batch in progress_bar:
        # Skip batch if resuming from checkpoint
        if checkpoint_manager and checkpoint_manager.should_skip_batch(i_batch):
            continue

        input_ids = batch["input_ids"].to(model_device)
        logits = model(input_ids)
        if hasattr(logits, "logits"):
            logits = logits.logits
        # logits = logits.float()

        targets = batch["targets"].to(model_device)

        batch_outputs = calculate_batch_outputs(
            hidden_states=None,
            target_hidden_states=None,
            logits=logits,
            target_logits=None,
            targets=targets,
            return_hidden_states=False,
            calculate_full_score_ablations=False,
            calc_on_cpu=False,
        )
        outputs.append(batch_outputs)

        # Update checkpoint progress periodically
        if checkpoint_manager:
            checkpoint_manager.update_progress(i_batch + 1, num_batches)

    losses, _ = _organize_outputs(outputs)
    return losses, None


def calculate_batch_outputs(
    hidden_states: torch.Tensor | None,
    target_hidden_states: torch.Tensor | None,
    logits: torch.Tensor,
    target_logits: torch.Tensor | None,
    targets: torch.Tensor,
    return_hidden_states: bool,
    calculate_full_score_ablations: bool,
    calc_on_cpu: bool,
) -> dict:
    if calc_on_cpu:
        if hidden_states is not None:
            hidden_states = hidden_states.cpu()
        if target_hidden_states is not None:
            target_hidden_states = target_hidden_states.cpu()
        if logits is not None:
            logits = logits.cpu()
        if target_logits is not None:
            target_logits = target_logits.cpu()
        if targets is not None:
            targets = targets.cpu()

    batch_outputs = _calculate_ground_truth_based_scores(logits, targets)

    if (target_hidden_states is not None) or (target_logits is not None):
        batch_outputs.update(
            _calculate_teacher_similarity_scores(
                hidden_states,
                target_hidden_states,
                logits,
                target_logits,
                calculate_full_score_ablations,
            )
        )

    if return_hidden_states:
        batch_outputs["hidden_states_per_batch"] = hidden_states.cpu()

    return batch_outputs


def _organize_outputs(
    outputs_per_batch: list[dict],
) -> tuple[dict[str, dict], list[torch.Tensor] | None]:
    outputs = _concatenate_batch_outputs(outputs_per_batch)
    hidden_states_per_batch = outputs.pop("hidden_states_per_batch", None)
    losses = {
        loss_name: {
            "avg": sum(loss_per_sample) / len(loss_per_sample),
            "per_sample": loss_per_sample,
        }
        for loss_name, loss_per_sample in outputs.items()
    }
    return losses, hidden_states_per_batch


def _concatenate_batch_outputs(outputs_per_batch: list[dict]) -> dict[str, list]:
    outputs = {}
    for output_name in outputs_per_batch[0]:  # Regular dict is directly iterable
        item_list = []
        for batch_outputs in outputs_per_batch:
            batch_items = batch_outputs[output_name]
            if isinstance(batch_items, list | tuple):
                item_list.extend(batch_items)
            else:
                item_list.append(batch_items)
        outputs[output_name] = item_list
    return outputs


def _calculate_per_sample_lm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> list[float]:
    per_sample_lm_loss = (
        torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), targets, ignore_index=-1, reduction="none"
        )
        .mean(dim=-1)
        .tolist()
    )
    return per_sample_lm_loss


def _calculate_ground_truth_based_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, list[float]]:
    scores = {"lm_loss": _calculate_per_sample_lm_loss(logits, targets)}

    for top_k in (1, 5, 10):
        top_k_predictions = logits.topk(top_k, dim=-1).indices  # [b, t, top_k]
        is_target_in_predictions = (targets.unsqueeze(-1) == top_k_predictions).any(
            dim=-1
        )  # [b, t]
        fraction_model_predicted_target = is_target_in_predictions.float().mean(dim=-1)  # [b]
        scores[f"token_accuracy_top_{top_k}"] = fraction_model_predicted_target.tolist()

    return scores


def cosine_embedding_loss(
    hidden_states: torch.Tensor,
    target_hidden_states: torch.Tensor,
) -> list[float]:
    return kd_model.cosine_embedding_loss_batched(hidden_states, target_hidden_states).tolist()


def normalized_mse_loss(
    hidden_states: torch.Tensor,
    target_hidden_states: torch.Tensor,
) -> list[float]:
    return [
        kd_model.normalized_mse_loss(hidden_states[i_sample], target_hidden_states[i_sample]).item()
        for i_sample in range(hidden_states.shape[0])
    ]


def mse_loss(
    hidden_states: torch.Tensor,
    target_hidden_states: torch.Tensor,
) -> list[float]:
    return [
        F.mse_loss(hidden_states[i_sample], target_hidden_states[i_sample]).item()
        for i_sample in range(hidden_states.shape[0])
    ]


def mae_loss(
    hidden_states: torch.Tensor,
    target_hidden_states: torch.Tensor,
) -> list[float]:
    return [
        F.l1_loss(hidden_states[i_sample], target_hidden_states[i_sample]).item()
        for i_sample in range(hidden_states.shape[0])
    ]


def _calculate_teacher_similarity_scores(
    hidden_states: torch.Tensor,
    target_hidden_states: torch.Tensor,
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    calculate_full_score_ablations: bool,
) -> dict[str, list[float]]:
    """hidden_states: [batch, tokens, n_embd]
    target_hidden_states: [batch, tokens, n_embd]
    logits: [batch, tokens, vocab]
    target_logits: [batch, tokens, vocab]
    """

    def calc_per_sample(func, logits, target_probs):
        return [
            func(logits=logits[i_sample], target_probs=target_probs[i_sample])
            for i_sample in range(logits.shape[0])
        ]

    score_ablations = {}

    if (target_hidden_states is not None) and (hidden_states.shape == target_hidden_states.shape):
        for func in (cosine_embedding_loss, normalized_mse_loss, mse_loss, mae_loss):
            score_name = f"{func.__name__}_hidden_states"
            score_ablations[score_name] = func(hidden_states, target_hidden_states)

    if target_logits is not None:
        for func in (cosine_embedding_loss, normalized_mse_loss, mse_loss, mae_loss):
            score_name = f"{func.__name__}_logits"
            score_ablations[score_name] = func(logits, target_logits)

        for top_p in (0.99, 0.95, None) if calculate_full_score_ablations else (None,):
            transformed_logits = (
                logits if (top_p is None) else top_p_top_k(logits, top_p=top_p, top_k=None)
            )
            transformed_target_logits = (
                target_logits
                if (top_p is None)
                else top_p_top_k(target_logits, top_p=top_p, top_k=None)
            )
            target_probs = transformed_target_logits.softmax(-1)

            for func in (kl_div, js_div, tv_dist):
                for clip_epsilon in (
                    (
                        ClipEpsilon.NO_CLIP,
                        ClipEpsilon.CLIP_NO_RENORMALIZE,
                        ClipEpsilon.CLIP_RENORMALIZE,
                    )
                    if calculate_full_score_ablations
                    else (ClipEpsilon.NO_CLIP,)
                ):
                    epsilon_factors = (
                        (1.0, 0.1, 0.01) if not clip_epsilon == ClipEpsilon.NO_CLIP else (None,)
                    )

                    for epsilon_factor in epsilon_factors:
                        score_name = (
                            f"{func.__name__}--top_p_{top_p}--clip_epsilon_{clip_epsilon.name}"
                            f"--epsilon_factor_{epsilon_factor}"
                        )
                        func_with_args = functools.partial(
                            func, clip_epsilon=clip_epsilon, epsilon_factor=epsilon_factor
                        )
                        score_ablations[score_name] = calc_per_sample(
                            func_with_args, transformed_logits, target_probs
                        )
                        if (top_p is None) and (clip_epsilon == ClipEpsilon.NO_CLIP):
                            short_score_name = func.__name__
                            score_ablations[short_score_name] = score_ablations[score_name]

        for top_k in (1, 5, 10):
            teacher_greedy_prediction = target_logits.argmax(dim=-1, keepdim=True)  # [b,t,1]
            student_top_k_predictions = logits.topk(top_k, dim=-1).indices  # [b,t,k]
            is_teacher_prediction_in_student_predictions = (
                teacher_greedy_prediction == student_top_k_predictions
            ).any(dim=-1)  # [b,t]
            fraction_student_predicted_teacher = (
                is_teacher_prediction_in_student_predictions.float().mean(dim=-1)
            )  # [b]
            score_ablations[f"greedy_teacher_prediction_in_student_top_{top_k}"] = (
                fraction_student_predicted_teacher.tolist()
            )

        if calculate_full_score_ablations:
            for top_p in (0.99, 0.95, 0.50, None):
                # student
                transformed_logits = logits.clone()

                # teacher
                transformed_target_logits = (
                    target_logits.clone()
                    if (top_p is None)
                    else top_p_top_k(target_logits, top_p=top_p, top_k=None)
                )

                target_probs = transformed_target_logits.softmax(-1)
                mask = transformed_target_logits == -1000
                if torch.any(mask):
                    transformed_logits[mask] = 0
                    transformed_target_logits[mask] = 0
                    target_probs[mask] = 0

                for func in (mse_loss, mae_loss):
                    score_name = f"{func.__name__}_logits_top_p_{top_p}"
                    score_ablations[score_name] = func(
                        transformed_logits, transformed_target_logits
                    )

                if top_p is not None and top_p > 0.9:
                    func = kl_div
                    clip_epsilon = ClipEpsilon.NO_CLIP
                    score_name = (
                        f"{func.__name__}--top_p_{top_p}--clip_epsilon_no_clip_student_unfiltered"
                    )
                    func_with_args = functools.partial(
                        func, clip_epsilon=clip_epsilon, epsilon_factor=epsilon_factor
                    )
                    score_ablations[score_name] = calc_per_sample(
                        func_with_args, logits, target_probs
                    )
                    # score_name = f"{func.__name__}_abs--top_p_{top_p}--clip_epsilon_no_clip_student_unfiltered"
                    # score_ablations[score_name] = [s.abs() for s in score_ablations[score_name]]

    return score_ablations


class ClipEpsilon(Enum):
    NO_CLIP = "NO_CLIP"
    CLIP_RENORMALIZE = "CLIP_RENORMALIZE"
    CLIP_NO_RENORMALIZE = "CLIP_NO_RENORMALIZE"


def _logits_to_logprobs(
    logits: torch.Tensor, clip_epsilon: ClipEpsilon, epsilon_factor: float
) -> torch.Tensor:
    """logits: [tokens, vocab]"""
    logprobs = logits.log_softmax(
        -1
    )  # must normalize logits before clipping otherwise log(1/voacb) means nothing
    if clip_epsilon == ClipEpsilon.NO_CLIP:
        return logprobs
    vocab_size = logprobs.shape[-1]
    epsilon = math.log(epsilon_factor * 1 / vocab_size)
    logprobs = torch.clip(logprobs, min=epsilon)
    if clip_epsilon == ClipEpsilon.CLIP_RENORMALIZE:
        logprobs = logprobs.log_softmax(
            -1
        )  # we do log_softmax again to retain legitimate distributions
    return logprobs


def kl_div(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    clip_epsilon: ClipEpsilon = ClipEpsilon.NO_CLIP,
    epsilon_factor: float = 1.0,
) -> float:
    """Kullback-Leibler Divergence for a single sample.
    logits: [tokens, vocab]
    target_probs: [tokens, vocab]
    """
    num_tokens = logits.shape[0]
    logprobs = _logits_to_logprobs(logits, clip_epsilon, epsilon_factor)

    _kl_div = (
        F.kl_div(logprobs, target_probs, reduction="sum", log_target=False).item() / num_tokens
    )
    return _kl_div


def js_div(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    clip_epsilon: ClipEpsilon = ClipEpsilon.NO_CLIP,
    epsilon_factor: float = 1.0,
) -> float:
    """Jensen-Shannon Divergence for a single sample.
    logits: [tokens, vocab]
    target_probs: [tokens, vocab]
    """
    probs = logits.softmax(-1)
    mixture_probs = (probs + target_probs) / 2
    mixture_logprobs = mixture_probs.log().clip(min=-1000)

    pred_kl_div = kl_div(mixture_logprobs, probs, clip_epsilon, epsilon_factor)
    target_kl_div = kl_div(mixture_logprobs, target_probs, clip_epsilon, epsilon_factor)
    _js_div = 0.5 * (pred_kl_div + target_kl_div)
    return _js_div


def tv_dist(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    clip_epsilon: ClipEpsilon = ClipEpsilon.NO_CLIP,
    epsilon_factor: float = 1.0,
) -> float:
    """Total Variation Distance (L1-loss) for a single sample.
    logits: [tokens, vocab]
    target_probs: [tokens, vocab]
    """
    num_tokens, vocab_size = logits.shape
    probs = logits.softmax(-1)

    if clip_epsilon != ClipEpsilon.NO_CLIP:
        epsilon = epsilon_factor * 1 / vocab_size
        probs = probs.clip(min=epsilon)
        target_probs = target_probs.clip(min=epsilon)
        if clip_epsilon == ClipEpsilon.CLIP_RENORMALIZE:
            probs = probs / probs.sum(-1, keepdim=True)
            target_probs = target_probs / target_probs.sum(-1, keepdim=True)

    _tv_dist = 0.5 * (probs - target_probs).abs().sum().item() / num_tokens
    return _tv_dist


DEFAULT_TOP_P = 0.999
# WestLake model:
# 700 = percentile 0.9 for top_p=0.99
# 1700 = percentile 0.95 for top_p=0.99 and percentile 0.75 for top_p=0.999
# For top_p=0.999 and top_k=1700 you take about 75 GB for 2048*8192 tokens
DEFAULT_TOP_K = 1000


def top_p_top_k(
    logits: torch.Tensor,
    top_p: float | None = DEFAULT_TOP_P,
    top_k: int | None = DEFAULT_TOP_K,
    filter_value=-1000,
) -> torch.Tensor:
    logit_warpers = []
    if top_p is not None:
        logit_warpers.append(TopPLogitsWarper(top_p=top_p, filter_value=filter_value))
    if top_k is not None:
        logit_warpers.append(TopKLogitsWarper(top_k=top_k, filter_value=filter_value))

    warped_logits = []
    for sample_logits in logits:
        for warper in logit_warpers:
            sample_logits = warper(input_ids=None, scores=sample_logits)
        warped_logits.append(sample_logits)
    warped_logits = torch.stack(warped_logits)

    return warped_logits
