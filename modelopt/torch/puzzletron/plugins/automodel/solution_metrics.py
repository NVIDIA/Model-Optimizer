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

"""Per-solution quality metrics for AutoModel replace-1-block scoring.

The FlashKLD backend evaluates the exact full-vocabulary LM head with FLA fused
cross-entropy and KL kernels while the vocabulary weights remain tensor-parallel.
Sequence-local metric contributions are reduced across CP before sample values are
gathered across the data axis. Hidden cosine and normalized-MSE use additive sufficient
statistics so their CP reduction is the exact full-sequence value, not an average of
per-segment ratios.
"""

import torch
import torch.distributed as torch_dist

from ...utils.validation import _organize_outputs
from .flash_dual import (
    flash_dual_ce_kd,
    flash_kld_ce_topk,
    topk_accuracy_and_agreement_from_hidden,
)

__all__ = ["score_batch", "aggregate_solution_scores", "retain_teacher_channels"]

_HIDDEN_STAT_PREFIX = "_cp_hidden_"
_METRIC_STAT_PREFIX = "_cp_metric_"


def retain_teacher_channels(
    candidate_hidden: torch.Tensor,
    candidate_lm_head_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_lm_head_weight: torch.Tensor,
    *,
    channel_indices: torch.Tensor | tuple[int, ...] | list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a full-width teacher target onto the student's retained basis.

    A physically sorted checkpoint stores its active prefix in a permuted basis.
    ``channel_indices`` maps that local prefix back to the original teacher basis;
    unsorted checkpoints retain the ordinary prefix behavior.
    """
    candidate_width = int(candidate_hidden.shape[-1])
    if int(candidate_lm_head_weight.shape[-1]) != candidate_width:
        raise ValueError(
            "candidate hidden/head width mismatch: "
            f"hidden={candidate_width} head={candidate_lm_head_weight.shape[-1]}"
        )
    if teacher_hidden.shape[:-1] != candidate_hidden.shape[:-1]:
        raise ValueError(
            "teacher/candidate token shape mismatch: "
            f"teacher={tuple(teacher_hidden.shape)} candidate={tuple(candidate_hidden.shape)}"
        )
    if int(teacher_hidden.shape[-1]) < candidate_width:
        raise ValueError(
            f"teacher hidden width {teacher_hidden.shape[-1]} is smaller than candidate "
            f"width {candidate_width}"
        )
    if int(teacher_lm_head_weight.shape[0]) != int(candidate_lm_head_weight.shape[0]):
        raise ValueError(
            "teacher/candidate vocabulary size mismatch: "
            f"teacher={teacher_lm_head_weight.shape[0]} "
            f"candidate={candidate_lm_head_weight.shape[0]}"
        )
    if int(teacher_lm_head_weight.shape[-1]) < candidate_width:
        raise ValueError(
            f"teacher LM-head width {teacher_lm_head_weight.shape[-1]} is smaller than "
            f"candidate width {candidate_width}"
        )
    if channel_indices is None:
        return teacher_hidden[..., :candidate_width], teacher_lm_head_weight[:, :candidate_width]
    indices = torch.as_tensor(channel_indices, dtype=torch.long, device=teacher_hidden.device)
    if indices.ndim != 1 or indices.numel() != candidate_width:
        raise ValueError(
            "retained teacher channel indices must contain exactly the candidate width: "
            f"indices={tuple(indices.shape)} candidate_width={candidate_width}"
        )
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= teacher_hidden.shape[-1]):
        raise ValueError(
            f"retained teacher channel indices are outside width {teacher_hidden.shape[-1]}"
        )
    head_indices = indices.to(teacher_lm_head_weight.device)
    return (
        teacher_hidden.index_select(-1, indices),
        teacher_lm_head_weight.index_select(-1, head_indices),
    )


def score_batch(
    candidate_hidden: torch.Tensor,
    candidate_lm_head_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 1.0,
    ignore_index: int = -100,
    chunk_size: int = 16384,
    lm_head_backend: str = "streaming",
    tp_group=None,
    flash_kld_token_chunk_size: int | None = None,
    flash_kld_reduction_backend: str = "fla",
    ce_mask: torch.Tensor | None = None,
    kd_mask: torch.Tensor | None = None,
    hidden_mask: torch.Tensor | None = None,
    hidden_metric_teacher: torch.Tensor | None = None,
) -> dict:
    """Per-sample metrics for one batch, streaming both LM heads over the vocab.

    Inputs are the captured final hidden states ``[b, t, d]`` and LM-head weights
    ``[vocab, d]`` for candidate (student) and the cached teacher, plus next-token
    ``targets`` ``[b, t]``. Returns ``{metric: [per_sample, ...]}`` with the legacy keys:
    ``lm_loss``, ``kl_div``, label-based ``token_accuracy_top_{1,5,10}``, explicit
    ``token_accuracy_top_{1,5,10}_consistency``, and the ``*_hidden_states`` similarity
    metrics. ``hidden_metric_teacher`` may project the original teacher into the
    student's retained basis for hidden-state metrics without changing the
    full-width teacher logits used for KL. No full ``[b, t, vocab]`` logits are
    ever materialized.
    """
    b, t, _ = candidate_hidden.shape
    sh = candidate_hidden.reshape(b * t, -1)
    th = teacher_hidden.reshape(b * t, -1).to(device=sh.device, dtype=sh.dtype)
    sw = candidate_lm_head_weight
    tw = teacher_lm_head_weight.to(device=sh.device, dtype=sw.dtype)
    flat_targets = targets.reshape(b * t)
    ce_valid = (
        ce_mask.reshape(b * t).to(device=sh.device, dtype=torch.bool)
        if ce_mask is not None
        else flat_targets.ge(0)
    )
    kd_valid = (
        kd_mask.reshape(b * t).to(device=sh.device, dtype=torch.bool)
        if kd_mask is not None
        else ce_valid
    )
    hidden_valid = (
        hidden_mask.reshape(b, t).to(device=sh.device, dtype=torch.bool)
        if hidden_mask is not None
        else torch.ones((b, t), device=sh.device, dtype=torch.bool)
    )
    loss_targets = flat_targets.masked_fill(~ce_valid, ignore_index)

    if lm_head_backend == "flash_kld":
        ce, kd, consistency_hits, accuracy_hits = flash_kld_ce_topk(
            sh,
            sw,
            th,
            tw,
            loss_targets,
            temperature=temperature,
            ignore_index=ignore_index,
            teacher_identity=(candidate_hidden is teacher_hidden),
            tp_group=tp_group,
            token_chunk_size=flash_kld_token_chunk_size,
            reduction_backend=flash_kld_reduction_backend,
        )
    elif lm_head_backend == "streaming":
        ce, kd = flash_dual_ce_kd(
            sh, sw, th, tw, loss_targets,
            temperature=temperature, ignore_index=ignore_index, chunk_size=chunk_size,
        )
        consistency_hits, accuracy_hits = topk_accuracy_and_agreement_from_hidden(
            sh,
            sw,
            th,
            tw,
            flat_targets,
            chunk_size=chunk_size,
        )
    else:
        raise ValueError(f"Unknown replace-block lm_head_backend={lm_head_backend!r}")

    def metric_payload(name: str, values: torch.Tensor, mask: torch.Tensor) -> dict[str, list]:
        values = values.reshape(b, t)
        mask = mask.reshape(b, t)
        sums = (values * mask.to(values.dtype)).sum(dim=1)
        counts = mask.sum(dim=1).to(values.dtype)
        means = torch.where(counts > 0, sums / counts.clamp_min(1), torch.zeros_like(sums))
        return {
            name: means.tolist(),
            f"{_METRIC_STAT_PREFIX}{name}_sum": sums.tolist(),
            f"{_METRIC_STAT_PREFIX}{name}_count": counts.tolist(),
        }

    out = {}
    out.update(metric_payload("lm_loss", ce, ce_valid))
    out.update(metric_payload("kl_div", kd, kd_valid))
    for top_k, hit in consistency_hits.items():
        metric_name = f"top_{top_k}_logit_agreement"
        payload = metric_payload(metric_name, hit.float(), kd_valid)
        out.update(payload)
        out.update(
            metric_payload(
                f"token_accuracy_top_{top_k}_consistency",
                hit.float(),
                kd_valid,
            )
        )
        out.update(
            metric_payload(
                f"token_accuracy_top_{top_k}",
                accuracy_hits[top_k].float(),
                ce_valid,
            )
        )

    metric_teacher = (
        teacher_hidden if hidden_metric_teacher is None else hidden_metric_teacher
    )
    if candidate_hidden.shape == metric_teacher.shape:
        candidate = candidate_hidden.float()
        teacher = metric_teacher.to(device=candidate.device).float()
        feature_mask = hidden_valid.unsqueeze(-1).to(candidate.dtype)
        candidate = candidate * feature_mask
        teacher = teacher * feature_mask
        diff = candidate - teacher
        reduce_dims = tuple(range(1, candidate.ndim))
        out.update(
            {
                f"{_HIDDEN_STAT_PREFIX}dot": (candidate * teacher).sum(reduce_dims).tolist(),
                f"{_HIDDEN_STAT_PREFIX}candidate_sq": candidate.square().sum(reduce_dims).tolist(),
                f"{_HIDDEN_STAT_PREFIX}teacher_sq": teacher.square().sum(reduce_dims).tolist(),
                f"{_HIDDEN_STAT_PREFIX}diff_sq": diff.square().sum(reduce_dims).tolist(),
                f"{_HIDDEN_STAT_PREFIX}target_eps_sq": (
                    (teacher - 1e-6).square() * feature_mask
                ).sum(reduce_dims).tolist(),
                f"{_HIDDEN_STAT_PREFIX}abs_diff": diff.abs().sum(reduce_dims).tolist(),
                f"{_HIDDEN_STAT_PREFIX}count": (
                    hidden_valid.sum(dim=1) * candidate.shape[-1]
                ).tolist(),
            }
        )
        eps = 1e-8
        dot = (candidate * teacher).sum(reduce_dims)
        candidate_sq = candidate.square().sum(reduce_dims)
        teacher_sq = teacher.square().sum(reduce_dims)
        diff_sq = diff.square().sum(reduce_dims)
        target_eps_sq = ((teacher - 1e-6).square() * feature_mask).sum(reduce_dims)
        count = (hidden_valid.sum(dim=1) * candidate.shape[-1]).to(diff_sq.dtype).clamp_min(1)
        out.update(
            {
                "cosine_embedding_loss_hidden_states": (
                    1.0 - dot / (candidate_sq.sqrt().clamp_min(eps) * teacher_sq.sqrt().clamp_min(eps))
                ).tolist(),
                "normalized_mse_loss_hidden_states": (
                    diff_sq / target_eps_sq.clamp_min(eps)
                ).tolist(),
                "mse_loss_hidden_states": (diff_sq / count).tolist(),
                "mae_loss_hidden_states": (diff.abs().sum(reduce_dims) / count).tolist(),
                "raw_replacement_loss": (diff_sq / count).tolist(),
            }
        )
    return out


def _gather_per_sample(per_sample: list[float], token_group) -> list[float]:
    """Concatenate this rank's per-sample values with the other token-group ranks'.

    The token group (``dp_cp``) is the data-partition axis: each rank scored a disjoint
    sample subset, so the union is the full set. Uses ``all_gather_object`` (small lists
    of floats). A no-op without an initialized group or with group size 1.
    """
    if token_group is None or not torch_dist.is_initialized():
        return list(per_sample)
    world = torch_dist.get_world_size(group=token_group)
    if world == 1:
        return list(per_sample)
    gathered: list[list[float]] = [None] * world  # type: ignore[list-item]
    torch_dist.all_gather_object(gathered, list(per_sample), group=token_group)
    out: list[float] = []
    for chunk in gathered:
        out.extend(chunk)
    return out


def _reduce_cp_sample_segments(
    per_sample: list[float], cp_group, *, average: bool = True
) -> list[float]:
    """Combine equal-length sequence segments into complete per-sample values."""
    if cp_group is None or not torch_dist.is_initialized():
        return list(per_sample)
    cp_size = torch_dist.get_world_size(cp_group)
    if cp_size == 1:
        return list(per_sample)
    values = torch.tensor(per_sample, dtype=torch.float64, device="cuda")
    torch_dist.all_reduce(values, group=cp_group)
    if average:
        values.div_(cp_size)
    if torch_dist.get_rank(cp_group) != 0:
        return []
    return values.cpu().tolist()


def aggregate_solution_scores(per_batch_outputs: list[dict], token_group=None, cp_group=None) -> dict:
    """Assemble ``{metric: {avg, per_sample}}`` from per-batch outputs, reduced over dp_cp.

    Mirrors ``utils.validation._organize_outputs`` (concatenate per-batch lists -> avg),
    then gathers each metric's per-sample list across the token group so the result is
    independent of the data-parallel / node count.
    """
    losses, _ = _organize_outputs(per_batch_outputs)
    reduced = {}
    hidden_stats = {}
    metric_stats = {}
    for name, entry in losses.items():
        is_hidden_stat = name.startswith(_HIDDEN_STAT_PREFIX)
        is_metric_stat = name.startswith(_METRIC_STAT_PREFIX)
        cp_reduced = _reduce_cp_sample_segments(
            entry["per_sample"],
            cp_group,
            average=not (is_hidden_stat or is_metric_stat),
        )
        per_sample = _gather_per_sample(cp_reduced, token_group)
        if is_hidden_stat:
            hidden_stats[name.removeprefix(_HIDDEN_STAT_PREFIX)] = per_sample
            continue
        if is_metric_stat:
            metric_stats[name.removeprefix(_METRIC_STAT_PREFIX)] = per_sample
            continue
        avg = sum(per_sample) / len(per_sample) if per_sample else float("nan")
        reduced[name] = {"avg": avg, "per_sample": per_sample}

    if hidden_stats:
        eps = 1e-8
        cosine = [
            1.0 - dot / (max(candidate_sq**0.5, eps) * max(teacher_sq**0.5, eps))
            for dot, candidate_sq, teacher_sq in zip(
                hidden_stats["dot"],
                hidden_stats["candidate_sq"],
                hidden_stats["teacher_sq"],
            )
        ]
        normalized_mse = [
            diff_sq / max(target_eps_sq, eps)
            for diff_sq, target_eps_sq in zip(
                hidden_stats["diff_sq"], hidden_stats["target_eps_sq"]
            )
        ]
        mse = [
            diff_sq / count
            for diff_sq, count in zip(hidden_stats["diff_sq"], hidden_stats["count"])
        ]
        mae = [
            abs_diff / count
            for abs_diff, count in zip(hidden_stats["abs_diff"], hidden_stats["count"])
        ]
        for name, per_sample in (
            ("cosine_embedding_loss_hidden_states", cosine),
            ("normalized_mse_loss_hidden_states", normalized_mse),
            ("mse_loss_hidden_states", mse),
            ("mae_loss_hidden_states", mae),
        ):
            avg = sum(per_sample) / len(per_sample) if per_sample else float("nan")
            reduced[name] = {"avg": avg, "per_sample": per_sample}
        # This is the unadjusted model-3 versus retained model-1 hidden MSE used
        # by MIP.  The sliced-teacher loss is reported separately and is never
        # subtracted from this score.
        reduced["raw_replacement_loss"] = dict(reduced["mse_loss_hidden_states"])
    for stat_name, sums in metric_stats.items():
        if not stat_name.endswith("_sum"):
            continue
        metric_name = stat_name.removesuffix("_sum")
        counts = metric_stats.get(f"{metric_name}_count")
        if counts is None:
            raise RuntimeError(f"missing distributed count for metric {metric_name}")
        per_sample = [
            total / count if count > 0 else 0.0
            for total, count in zip(sums, counts)
        ]
        reduced[metric_name] = {
            "avg": sum(per_sample) / len(per_sample) if per_sample else float("nan"),
            "per_sample": per_sample,
        }
    return reduced
