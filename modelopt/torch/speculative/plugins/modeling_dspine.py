# Adapted from https://github.com/sgl-project/SpecForge/blob/2dadea1/specforge/modeling/draft/dspine.py
# Copyright (c) 2025 sgl-project
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

"""DSpine draft module: the DFlash backbone with causal blocks and adjacent causal injection.

DSpine (Tencent, "Draft in Parallel, Condition Through Depth: Adjacent Causal Injection for
Speculative Decoding", https://github.com/Tencent/DSpine) drafts a whole block in one parallel
pass like DFlash, with causal in-block attention, and after every draft layer ``l`` writes each
position's predecessor code into it::

    z_t = sqrt(C) * normalize(R_l u_t)
    h_t = u_t + rms(u_t) * W_l (sigmoid(G_q z_t + G_p z_{t-1} + b) * A z_{t-1})

Position 1's predecessor is the anchor token's fixed codebook code, and the anchor is never
written. Because attention is causal, a message written at layer ``l`` reaches the next
position at layer ``l + 1``. Decoding redoes the last write with the exact code of the token
chosen for the previous position (:meth:`DSpineModule.refine`) and rescores that position's
top-k backbone candidates with it.

Parameter and buffer names follow the released checkpoints (e.g. ``tencent/Qwen3-8B-DSpine-b16``,
read by Tencent/DSpine@0cc92d8 ``sglang_dspine/models/dspine.py``), so ``DSpineExporter`` writes
what SGLang's ``--speculative-algorithm DSPINE`` loads without conversion.
"""

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from .modeling_dflash import DFlashModule

__all__ = ["DSpineCodeBridge", "DSpineDraftOutput", "DSpineModule"]


class DSpineDraftOutput(NamedTuple):
    """One DSpine draft forward over ``[B, N * block_size]`` draft positions."""

    hidden_states: torch.Tensor  # final-normed draft hidden states [B, N*bs, H]
    pre_injection: torch.Tensor  # last layer's output before its write [B, N*bs, H]
    layer_features: torch.Tensor  # each layer's post-write codes [L, B, N, bs, C]
    refined: torch.Tensor | None = None  # refine() of positions 1..bs-1 [B, N, bs-1, H]


class _ZeroTerminalProjection(nn.Module):
    """The terminal projection the released layout carries and the serving loader requires all zero."""

    def __init__(self, rank: int, code_size: int):
        super().__init__()
        self.register_buffer("weight", torch.zeros(code_size, rank))


class DSpineCodeBridge(nn.Module):
    """Adjacent causal injection between the draft's decoder layers.

    ``early_read_weights`` [L-1, C, H] and ``read_weight`` [C, H] are the per-layer readouts
    ``R_l``; ``feature_write_weights`` [L, H, R] the per-layer writes ``W_l``; ``prev`` (``A``),
    ``gate`` (``G_q``, ``b``) and ``predecessor_gate`` (``G_p``) are shared across layers. The
    ``codebook`` [V, C] holds fixed unit-norm token codes derived from the target head.
    """

    codebook: torch.Tensor

    def __init__(self, *, hidden_size: int, num_layers: int, vocab_size: int, code_size, rank):
        """Build the bridge on its identity start (zero writes and predecessor gate)."""
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"DSpine needs at least 2 draft layers, got {num_layers}.")
        if not 0 < code_size <= min(hidden_size, vocab_size - 1):
            raise ValueError(
                f"DSpine code_size must be in [1, min(hidden_size, vocab_size - 1)] = "
                f"[1, {min(hidden_size, vocab_size - 1)}], got {code_size}."
            )
        if rank < 1:
            raise ValueError(f"DSpine code_rank must be >= 1, got {rank}.")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.code_size = code_size
        self.rank = rank
        self.code_scale = math.sqrt(code_size)
        self.early_read_weights = nn.Parameter(torch.empty(num_layers - 1, code_size, hidden_size))
        self.read_weight = nn.Parameter(torch.empty(code_size, hidden_size))
        self.feature_write_weights = nn.Parameter(torch.zeros(num_layers, hidden_size, rank))
        self.prev = nn.Linear(code_size, rank, bias=False)
        self.gate = nn.Linear(code_size, rank, bias=True)
        self.predecessor_gate = nn.Linear(code_size, rank, bias=False)
        self.out = _ZeroTerminalProjection(rank, code_size)
        self.register_buffer("codebook", torch.zeros(vocab_size, code_size))
        self.register_buffer("independent_write_version", torch.ones((), dtype=torch.int64))
        self.register_buffer("terminal_disabled_version", torch.ones((), dtype=torch.int64))
        with torch.no_grad():
            # Placeholders until initialize_transfer_space sets the readouts from the target head.
            for weight in (*self.early_read_weights, self.read_weight):
                nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            # Zero writes make an untrained DSpine numerically a causal DFlash.
            self.gate.bias.zero_()
            self.predecessor_gate.weight.zero_()

    def layer_read_weight(self, layer: int) -> torch.Tensor:
        """Readout ``R_l`` of draft layer ``layer``: [C, H]."""
        return self.read_weight if layer == self.num_layers - 1 else self.early_read_weights[layer]

    def feature(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        """Each row's code ``sqrt(C) * normalize(R_l h)``: [..., H] -> [..., C]."""
        projected = F.linear(hidden, self.layer_read_weight(layer))
        return (self.code_scale * F.normalize(projected.float(), dim=-1)).to(hidden.dtype)

    def token_codes(self, token_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Scaled codebook codes ``sqrt(C) * E[token_ids]``: [...] -> [..., C]."""
        return (self.code_scale * F.embedding(token_ids, self.codebook)).to(dtype)

    def write(self, hidden, receiver, predecessor, layer: int) -> torch.Tensor:
        """``h + rms(h) * W_l (sigmoid(G_q receiver + G_p predecessor + b) * A predecessor)``.

        ``rms`` has no epsilon and no learned weight, as in the serving kernels.
        """
        predecessor = predecessor.to(receiver.dtype)
        gate = torch.sigmoid(self.gate(receiver) + self.predecessor_gate(predecessor))
        delta = F.linear(gate * self.prev(predecessor), self.feature_write_weights[layer])
        rms = torch.linalg.vector_norm(hidden.float(), dim=-1, keepdim=True)
        return hidden + (rms / math.sqrt(hidden.shape[-1])).to(hidden.dtype) * delta

    def forward(self, hidden, layer: int, block_token_ids, replace_mask=None) -> torch.Tensor:
        """Write each position's predecessor code after decoder layer ``layer``.

        Args:
            hidden: Layer output per block, [B, N, bs, H].
            layer: Index of the decoder layer just applied.
            block_token_ids: [B, N, bs]; ``[..., 0]`` is each block's anchor, whose code is
                position 1's predecessor. ``[..., t > 0]`` are read only for blocks in
                ``replace_mask``.
            replace_mask: Optional [B, N] bool; those blocks take the true previous tokens'
                codes as predecessors (the teacher-forcing curriculum).
        """
        features = self.feature(hidden, layer)
        anchor = self.token_codes(block_token_ids[..., :1], features.dtype)
        predecessors = torch.cat((anchor, features[..., 1:-1, :]), dim=-2)
        if replace_mask is not None:
            teacher = self.token_codes(block_token_ids[..., :-1], features.dtype)
            predecessors = torch.where(replace_mask[..., None, None], teacher, predecessors)
        successors = self.write(hidden[..., 1:, :], features[..., 1:, :], predecessors, layer)
        return torch.cat((hidden[..., :1, :], successors), dim=-2)

    @torch.no_grad()
    def initialize_transfer_space(self, head_weight: torch.Tensor, chunk_size: int = 4096):
        """Build the fixed codebook and the readouts from the target's [V, H] output head.

        Normalizes and centres the head rows, takes the top-``code_size`` principal directions
        of their covariance, and whitens and re-normalizes the projected rows into the codebook;
        every readout starts as the principal directions. Computed in float32.
        """
        vocab_size = self.codebook.shape[0]
        if tuple(head_weight.shape) != (vocab_size, self.hidden_size):
            raise ValueError(
                f"DSpine needs the target head's [vocab, hidden] = [{vocab_size}, "
                f"{self.hidden_size}] weight, got {tuple(head_weight.shape)}."
            )
        covariance = head_weight.new_zeros(self.hidden_size, self.hidden_size, dtype=torch.float32)
        total = covariance.new_zeros(self.hidden_size)
        for rows in head_weight.split(chunk_size):
            normalized = F.normalize(rows.float(), dim=-1)
            total.add_(normalized.sum(dim=0))
            covariance.add_(normalized.T @ normalized)
        mean = total / vocab_size
        covariance.sub_(vocab_size * torch.outer(mean, mean))
        values, directions = torch.linalg.eigh(covariance)
        values, directions = values[-self.code_size :], directions[:, -self.code_size :]
        # Eigenvectors are defined up to sign, and devices may disagree on it; fix each one's
        # largest component positive so every data-parallel rank derives the same transfer space.
        signs = directions.gather(0, directions.abs().argmax(dim=0, keepdim=True)).sign()
        directions = directions * torch.where(signs == 0, 1.0, signs)
        # A rank-deficient head leaves float noise, not exact zeros, where eigenvalues vanish.
        tolerance = values[-1] * self.hidden_size * torch.finfo(torch.float32).eps
        if not torch.isfinite(values).all() or (values <= tolerance).any():
            raise ValueError(
                f"The target head has fewer than code_size={self.code_size} positive principal "
                "directions; lower dflash_architecture_config.code_size."
            )
        inv_sqrt_values = values.rsqrt()
        for start, rows in zip(range(0, vocab_size, chunk_size), head_weight.split(chunk_size)):
            codes = (F.normalize(rows.float(), dim=-1) - mean) @ directions * inv_sqrt_values
            self.codebook[start : start + rows.shape[0]] = F.normalize(codes, dim=-1).to(
                self.codebook
            )
        readout = directions.T.to(self.read_weight)
        self.read_weight.copy_(readout)
        self.early_read_weights.copy_(readout.expand_as(self.early_read_weights))


class DSpineModule(DFlashModule):
    """DFlash draft backbone with causal blocks and an adjacent causal injection after every layer.

    DSpine-specific ``dflash_architecture_config`` keys (defaults reproduce the released drafters):
    ``code_size`` (1024), ``code_rank`` (512), ``refine_top_k`` (16), and
    ``teacher_forcing_layers`` (3): how many leading layers the curriculum may teacher-force.
    """

    def __init__(self, config):
        """Initialize the DFlash backbone, then the code bridge and the serving version markers."""
        super().__init__(config)
        num_layers = config.num_hidden_layers
        if self.block_size < 2:
            raise ValueError(f"DSpine needs block_size >= 2, got {self.block_size}.")
        self.head_topk = int(getattr(config, "refine_top_k", 16))
        # The serving kernels support power-of-two widths in [2, 64] only.
        if self.head_topk & (self.head_topk - 1) or not 2 <= self.head_topk <= 64:
            raise ValueError(
                f"DSpine refine_top_k must be a power of two in [2, 64], got {self.head_topk}."
            )
        if self.head_topk > config.vocab_size:
            raise ValueError(f"DSpine refine_top_k={self.head_topk} exceeds the vocabulary.")
        teacher_layers = int(getattr(config, "teacher_forcing_layers", 3))
        if not 0 <= teacher_layers <= num_layers:
            raise ValueError(
                f"DSpine teacher_forcing_layers must be in [0, {num_layers}], got {teacher_layers}."
            )
        self.teacher_forcing_layers = frozenset(range(teacher_layers))
        self.selected_prefix_bridge = DSpineCodeBridge(
            hidden_size=config.hidden_size,
            num_layers=num_layers,
            vocab_size=config.vocab_size,
            code_size=int(getattr(config, "code_size", 1024)),
            rank=int(getattr(config, "code_rank", 512)),
        )
        # Version markers the serving loader checks for the causal draft and the refinement.
        self.register_buffer("draft_block_causal_version", torch.ones((), dtype=torch.int64))
        self.register_buffer("last_write_refine_version", torch.ones((), dtype=torch.int64))

    def forward(
        self,
        noise_embedding,
        target_hidden,
        position_ids,
        attention_mask=None,
        *,
        block_token_ids,
        replace_mask=None,
        refine_predecessor_ids=None,
    ) -> DSpineDraftOutput:
        """Run the draft over ``N`` blocks; ``block_token_ids`` [B, N, bs] as in :class:`DSpineCodeBridge`.

        ``refine_predecessor_ids`` [B, N, bs - 1], the tokens before positions ``1..bs-1``, also
        returns their :meth:`refine` states. Training refines here so that FSDP2, which gathers
        the draft's weights only around ``forward``, covers it.
        """
        # Outside the compiled body: lazy rotary init mutates the module.
        self._maybe_init_rotary_emb(device=noise_embedding.device)
        return self._body()(
            noise_embedding,
            target_hidden,
            position_ids,
            attention_mask,
            block_token_ids,
            replace_mask,
            refine_predecessor_ids,
        )

    def _forward_body(
        self,
        noise_embedding,
        target_hidden,
        position_ids,
        attention_mask,
        block_token_ids,
        replace_mask=None,
        refine_predecessor_ids=None,
    ):
        """Feature fusion, then each decoder layer followed by its code write, then the final norm."""
        bsz, n_blocks, block_size = block_token_ids.shape
        bridge = self.selected_prefix_bridge
        hidden = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden, position_ids)

        features = []
        for index, layer in enumerate(self.layers):
            pre_injection = layer(hidden, target_hidden, position_embeddings, attention_mask)
            injected = bridge(
                pre_injection.view(bsz, n_blocks, block_size, -1),
                index,
                block_token_ids,
                replace_mask if index in self.teacher_forcing_layers else None,
            )
            features.append(bridge.feature(injected, index))
            hidden = injected.view(bsz, n_blocks * block_size, -1)
        refined = None
        if refine_predecessor_ids is not None:
            successors = pre_injection.view(bsz, n_blocks, block_size, -1)[:, :, 1:]
            refined = self.refine(successors.contiguous(), refine_predecessor_ids)
        return DSpineDraftOutput(self.norm(hidden), pre_injection, torch.stack(features), refined)

    def refine(self, pre_injection: torch.Tensor, predecessor_ids: torch.Tensor) -> torch.Tensor:
        """Redo the last layer's write with each row's exact predecessor token, then final-norm.

        Args:
            pre_injection: Last-layer outputs before their write, [..., H].
            predecessor_ids: The token each row's predecessor committed, [...].
        """
        bridge = self.selected_prefix_bridge
        last = len(self.layers) - 1
        receiver = bridge.feature(pre_injection, last)
        predecessor = bridge.token_codes(predecessor_ids, receiver.dtype)
        return self.norm(bridge.write(pre_injection, receiver, predecessor, last))
