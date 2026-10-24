# Adapted from https://github.com/sgl-project/SpecForge/blob/8ea5ca6/specforge/core/dflash.py
# and https://github.com/deepseek-ai/DeepSpec/blob/add63ba/deepspec/modeling/dspark/loss.py
# Copyright (c) 2025 sgl-project
# Copyright (c) 2026 The DeepSpec Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

"""Loss components of the DFlash speculative-decoding family, independent of any variant.

A training step's supervision: which token each block position is trained on and where
(:func:`_block_targets`), how the positions are weighted (:func:`_block_position_weights`),
and :class:`LossContext`, which carries them with the draft's logits to the loss.
"""

import functools
from typing import NamedTuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

__all__ = ["DraftLogits", "LossContext", "LossOutput"]


def _block_positions(anchor_positions, block_size, offset, seq_len):
    """Index ``anchor + offset + k`` of each block position ``k``, clamped onto the sequence.

    Returns ``(positions [B, N, bs], inside)``; ``inside`` is False past the sequence end.
    """
    k = torch.arange(block_size, device=anchor_positions.device).view(1, 1, -1)
    unclamped = anchor_positions.unsqueeze(-1) + k + offset
    return unclamped.clamp(min=0, max=seq_len - 1), unclamped < seq_len


def _block_gather(x, positions):
    """``x[b, positions[b, n, k]]`` for ``x`` [B, seq] and ``positions`` [B, N, bs]."""
    return torch.gather(x.unsqueeze(1).expand(-1, positions.shape[1], -1), 2, positions)


def _block_tokens(input_ids, anchor_positions, block_size, offset):
    """The token at ``anchor + offset + k`` for each block position ``k``: [B, N, bs]."""
    positions, _ = _block_positions(anchor_positions, block_size, offset, input_ids.shape[1])
    return _block_gather(input_ids, positions)


class _BlockTargets(NamedTuple):
    """What each block position is trained on: the token at ``anchor + k + label_shift``."""

    token_ids: torch.Tensor  # [B, N, bs]
    positions: torch.Tensor  # [B, N, bs] their sequence indices, clamped onto the sequence
    mask: torch.Tensor  # [B, N, bs] 1.0 where the position is supervised


def _block_targets(input_ids, anchor_positions, block_keep_mask, loss_mask, block_size, shift):
    """Labels of every block position and where they are supervised.

    A position is supervised when its block is kept and its label lies inside the sequence
    and the loss mask, and is not the anchor itself, which the draft is given as input.
    """
    positions, inside = _block_positions(anchor_positions, block_size, shift, input_ids.shape[1])
    k = torch.arange(block_size, device=input_ids.device).view(1, 1, -1)
    mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, block_size).float()
    mask = mask * inside.float()
    mask = mask * (k + shift > 0).float()
    mask = mask * _block_gather(loss_mask, positions)
    return _BlockTargets(_block_gather(input_ids, positions), positions, mask)


def _dpace_position_weights(
    confidences: torch.Tensor, alpha: float, valid_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute detached D-PACE per-position weights from draft confidences.

    Derived from D-PACE (arXiv:2605.18810). The paper factorizes the per-position
    weight (Fig. 2 / Eq. 8) into a *cumulative confidence* times a *continuation
    value*, which is equivalently the suffix sum of the cumulative confidences::

        C_j = prod_{i<=j} q~_i                 # cumulative confidence (Eq. 8)
        w_j = sum_{m>=j} C_m                    # = C_j * continuation value f~_j

    Each confidence is asymmetrically smoothed toward 1 (Eq. 7)::

        q~_i = (1 - alpha) * q_i + alpha,   alpha in (0, 1],

    so the floor ``q~_i >= alpha`` keeps every cumulative product (hence every
    weight) strictly positive. We evaluate the suffix sum from its definition as
    ``total - exclusive_prefix_sum`` of ``C`` rather than reversing the tensor.
    Positions with ``valid_mask == 0`` are multiplicative no-ops in ``C`` and
    contribute nothing to the sum, matching the per-token loss mask. Weights are
    detached (Eq. 9): they reweight the cross-entropy without adding gradient.

    Args:
        confidences: ``[..., L]`` draft confidence ``q_i = exp(-CE_i)`` per position.
        alpha: smoothing factor in (0, 1]; raises if outside that range.
        valid_mask: optional ``[..., L]`` 0/1 mask; ``None`` treats all positions valid.

    Returns:
        Detached weights with the same shape and dtype as ``confidences``.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"dflash_dpace_alpha must be in (0, 1], got {alpha}")

    with torch.no_grad():
        smoothed = alpha + (1.0 - alpha) * confidences.float()  # Eq. 7
        if valid_mask is not None:
            keep = valid_mask.to(torch.bool)
            smoothed = torch.where(keep, smoothed, torch.ones_like(smoothed))
        cum_conf = torch.cumprod(smoothed, dim=-1)  # Eq. 8 cumulative confidence C_j
        if valid_mask is not None:
            cum_conf = cum_conf * keep.to(cum_conf.dtype)
        # Suffix sum w_j = sum_{m>=j} C_m, written as total minus the exclusive
        # prefix sum so no axis reversal is needed (Eq. 8).
        inclusive = torch.cumsum(cum_conf, dim=-1)
        weights = inclusive[..., -1:] - inclusive + cum_conf
        return weights.to(dtype=confidences.dtype)


def _block_position_weights(mask, scheme, first, decay_factor, dpace_alpha, target_ce=None):
    """``mask`` weighted per block position, by D-PACE or by exponential decay.

    ``first`` is the first predicted block position. ``target_ce`` returns the draft's
    cross-entropy on its labels ([B, N, bs]); only D-PACE calls it.
    """
    block_size = mask.shape[-1]
    if scheme == "dpace" and block_size > 1:
        with torch.no_grad():
            # Draft confidence q_i = exp(-CE) on the label; detached, as in the paper (Eq. 9).
            confidences = torch.exp(-target_ce()[..., first:].float())
            dpace = torch.ones_like(mask)
            dpace[..., first:] = _dpace_position_weights(
                confidences, dpace_alpha, valid_mask=mask[..., first:]
            )
        return mask * dpace
    if decay_factor > 0:
        k = torch.arange(block_size, device=mask.device).view(1, 1, -1)
        return mask * torch.exp(-(k - first).clamp(min=0).float() / decay_factor)
    return mask


def _masked_top1(logits, target_ids, mask):
    """Top-1 accuracy of ``logits`` against ``target_ids`` where ``mask`` is set (0-dim)."""
    correct = (logits.argmax(dim=-1) == target_ids) & (mask > 0.5)
    return correct.sum().float() / (mask.sum() + 1e-6)


def _tvd_chunk(a, b):
    """Per-token TVD for one row chunk: ``(softmax(a) - softmax(b)).abs().sum(-1)``."""
    return (torch.softmax(a.float(), dim=-1) - torch.softmax(b.float(), dim=-1)).abs().sum(dim=-1)


def _tvd_per_token(final_logits, teacher_logits, chunk_size=1024, chunk_fn=None):
    """Total-variation distance ||softmax(a)-softmax(b)||_1 / ... per token, memory-lean.

    Materializing both [N, vocab] float32 softmax tensors at once OOMs at large
    N (num_anchors*block_size) and a 150k vocab. We compute it in row chunks and
    gradient-checkpoint each chunk so the wide softmaxes are recomputed in backward
    rather than held — peak memory ~ chunk_size*vocab instead of N*vocab. The math
    is identical to ``(softmax(final)-softmax(teacher)).abs().sum(-1)``.

    Chunks with ``Tensor.split``, not slicing (each slice's backward zero-fills a full
    [N, vocab] tensor) nor ``torch.chunk`` (different chunk shapes, so recompiles).
    """
    _chunk = chunk_fn or _tvd_chunk
    outs = []
    for a, b in zip(
        final_logits.split(chunk_size, dim=0),
        teacher_logits.split(chunk_size, dim=0),
        strict=True,
    ):
        if torch.is_grad_enabled() and a.requires_grad:
            outs.append(torch.utils.checkpoint.checkpoint(_chunk, a, b, use_reentrant=False))
        else:
            outs.append(_chunk(a, b))
    return torch.cat(outs, dim=0)


class DraftLogits(NamedTuple):
    """The draft's logits per block position: [B, N, bs, V]."""

    final: torch.Tensor  # the distribution the draft serves
    backbone: torch.Tensor  # before any head; ``final`` itself for a head-less variant


class LossOutput(NamedTuple):
    """A training step's loss, the accuracy ``train_acc`` reports, and its metrics."""

    loss: torch.Tensor
    accuracy: float
    metrics: dict  # name -> detached 0-dim tensor


class LossContext:
    """One training step's supervision; costly per-token quantities are computed lazily, once.

    ``weights`` are the per-position weights after decay or D-PACE, which reads the draft's
    confidence off ``ce("final")``. ``ce`` keeps its gradient only when ``ce_grad`` says the
    objective backpropagates through it; otherwise it runs under ``no_grad``.
    """

    def __init__(self, batch, logits, targets, *, weighting, teacher, ce_grad):
        """Weigh the step's block positions.

        Args:
            batch: The forward's ``DraftBatch``.
            logits: The draft's ``DraftLogits``.
            targets: The labels and their mask, from ``_block_targets``.
            weighting: ``_block_position_weights``' scheme, first, decay_factor and dpace_alpha.
            teacher: Returns the base model's logits for every label position.
            ce_grad: Whether the objective backpropagates through ``ce``.
        """
        self.batch = batch
        self.logits = logits
        self.targets = targets
        self._teacher_fn = teacher
        self._ce_grad = ce_grad and torch.is_grad_enabled()
        self._ce = {}
        self._teacher = None
        self.weights = _block_position_weights(
            targets.mask,
            **weighting,
            target_ce=lambda: self.ce("final").detach().view(targets.mask.shape),
        )
        self.weight_sum = self.weights.sum() + 1e-6

    @functools.cached_property
    def trainable(self) -> bool:
        """Whether the weights leave anything to train on (reads the device once)."""
        return bool(self.weight_sum > 1.0)

    def flat_logits(self, source):
        """``logits.<source>`` as [B * N * bs, V]."""
        logits = getattr(self.logits, source)
        return logits.reshape(-1, logits.shape[-1])

    def ce(self, source):
        """Per-token cross-entropy of ``logits.<source>`` on the labels: [B * N * bs]."""
        if source not in self._ce:
            with torch.set_grad_enabled(self._ce_grad):
                self._ce[source] = F.cross_entropy(
                    self.flat_logits(source), self.targets.token_ids.view(-1), reduction="none"
                )
        return self._ce[source]

    def teacher(self):
        """The base model's logits for every label position: [B, N, bs, V]."""
        if self._teacher is None:
            self._teacher = self._teacher_fn()
        return self._teacher
