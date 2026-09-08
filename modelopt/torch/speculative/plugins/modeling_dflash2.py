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

# Adapted from https://github.com/sgl-project/SpecForge/pull/772
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

"""DFlash2 draft module — DFlash backbone plus local convolution and candidate selection.

DFlash2 (Inco AI / Z Lab, https://inco.ai/blog/dflash2/) keeps DFlash's one-pass
parallel backbone and adds two small components that address the two ways a
purely parallel draft loses acceptance:

- :class:`DFlashGroupedConv` — a grouped *dynamic* depthwise convolution wrapped
  around every attention and MLP sublayer. Each block position mixes in its
  predecessors inside the block, which injects the intra-block sequential
  dependency the parallel backbone lacks (mitigating suffix acceptance decay)
  without a second backbone pass. Taps do not cross the block boundary.

- :class:`CandidateSelector` — a low-rank transition scorer. Instead of an
  independent argmax per block position, the drafter keeps the target head's
  top-k candidates per position and scores adjacent transitions, so serving can
  walk one coherent path through the block.

Where Domino uses a GRU and DSpark a Markov transition bias, DFlash2 spends its
extra capacity on these two pieces: both are cheap (a few percent of draft
parameters, ~1% of serving step latency in the reference measurements).

This module owns the parameters only; the training wrapper (``HFDFlash2Model``
in ``hf_dflash2.py``) orchestrates the forward and the selector loss. Module and
parameter names (``attention_conv`` / ``mlp_conv`` / ``base_kernel`` /
``kernel_projection`` / ``candidate_selector`` / ``predecessor_codebook`` /
``successor_codebook`` / ``hidden_projection``) match the SGLang and vLLM
``DFlash2DraftModel`` loaders so an exported checkpoint is served directly.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .modeling_dflash import DFlashModule

__all__ = ["CandidateSelector", "DFlash2Module", "DFlashGroupedConv"]


class DFlashGroupedConv(nn.Module):
    """Grouped dynamic depthwise convolution over positions within a proposal block.

    Wraps one sublayer: :meth:`prepare` convolves the sublayer input and emits the
    dynamic kernel for the output side, :meth:`finish` convolves the sublayer
    output. One projection of the sublayer input produces both sides' kernel
    deltas.

    ``base_kernel`` starts as an identity (tap 0 weight 1, later taps 0), so a
    freshly built DFlash2 draft computes exactly what its DFlash backbone would.
    That makes the convolution a stable extension rather than a perturbation, and
    lets a DFlash checkpoint warm-start a DFlash2 run.
    """

    def __init__(self, hidden_size: int, block_size: int, taps: int, group_size: int):
        """Build the identity-initialized base kernel and the dynamic-kernel projection."""
        super().__init__()
        if taps < 1:
            raise ValueError(f"DFlash2 conv_kernel_size must be >= 1, got {taps}.")
        if taps > block_size:
            raise ValueError(
                f"DFlash2 conv_kernel_size ({taps}) must not exceed "
                f"dflash_block_size ({block_size})."
            )
        if group_size < 1 or hidden_size % group_size:
            raise ValueError(
                f"DFlash2 conv_group_size ({group_size}) must be >= 1 and divide "
                f"hidden_size ({hidden_size})."
            )

        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = int(hidden_size) // self.group_size

        # [input/output side, tap, channel]; identity at tap 0. Layout matches the
        # SGLang/vLLM DFlash2 weight loader.
        base_kernel = torch.zeros(2, self.taps, int(hidden_size))
        base_kernel[:, 0] = 1.0
        self.base_kernel = nn.Parameter(base_kernel)
        self.kernel_projection = nn.Linear(
            int(hidden_size), 2 * self.taps * self.num_groups, bias=False
        )

    def _convolve(self, hidden_states, delta, side: int):
        """Apply the depthwise convolution for one side, with taps clipped at block starts."""
        bsz, seq_len, hidden_size = hidden_states.shape
        if seq_len % self.block_size:
            raise ValueError(
                f"DFlash2 convolution needs a sequence length divisible by "
                f"block_size ({self.block_size}), got {seq_len}."
            )

        n_blocks = seq_len // self.block_size
        blocks = hidden_states.reshape(
            bsz, n_blocks, self.block_size, self.num_groups, self.group_size
        )
        dynamic = delta.reshape(bsz, n_blocks, self.block_size, self.taps, self.num_groups)
        base = self.base_kernel[side].reshape(1, 1, 1, self.taps, self.num_groups, self.group_size)
        # Per-position, per-group coefficients: static base plus the dynamic delta.
        coefficients = base + dynamic.unsqueeze(-1)

        output = coefficients[:, :, :, 0] * blocks
        for tap in range(1, self.taps):
            # Shift within the block only: position k reads k-tap, and the first
            # `tap` positions of each block read zeros rather than the previous block.
            shifted = F.pad(blocks[:, :, : self.block_size - tap], (0, 0, 0, 0, tap, 0))
            output = output + coefficients[:, :, :, tap] * shifted
        return output.reshape(bsz, seq_len, hidden_size)

    def prepare(self, hidden_states):
        """Convolve the sublayer input; return it with the output side's dynamic kernel."""
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[..., 0, :, :], side=0), coefficients[
            ..., 1, :, :
        ]

    def finish(self, hidden_states, state):
        """Convolve the sublayer output using the kernel produced by :meth:`prepare`."""
        return self._convolve(hidden_states, state, side=1)


class CandidateSelector(nn.Module):
    """Low-rank scorer for transitions between adjacent block positions' candidates.

    Scores an edge from a predecessor token ``p`` to a candidate token ``c`` at a
    block position with hidden state ``h`` as::

        edge(p -> c) = <predecessor_codebook[p] * hidden_projection(h),
                        successor_codebook[c]> + unary_logit[c]

    i.e. a bilinear form between the two token codebooks, gated by the context.
    Training scores each position's candidate set independently under teacher
    forcing (:meth:`score_candidates`); serving walks the resulting lattice.
    """

    def __init__(self, hidden_size: int, vocab_size: int, rank: int, top_k: int, std: float):
        """Build the predecessor/successor codebooks and the context projection."""
        super().__init__()
        if rank < 1:
            raise ValueError(f"DFlash2 selector_rank must be >= 1, got {rank}.")
        if not 1 <= top_k <= vocab_size:
            raise ValueError(
                f"DFlash2 selector_top_k must be in [1, vocab_size={vocab_size}], got {top_k}."
            )
        self.top_k = int(top_k)
        self.rank = int(rank)
        self.predecessor_codebook = nn.Parameter(torch.empty(int(vocab_size), int(rank)))
        self.successor_codebook = nn.Parameter(torch.empty(int(vocab_size), int(rank)))
        self.hidden_projection = nn.Linear(int(hidden_size), int(rank), bias=False)
        nn.init.normal_(self.predecessor_codebook, std=std)
        nn.init.normal_(self.successor_codebook, std=std)

    def score_candidates(self, candidate_ids, unary_logits, hidden_states, predecessor_ids):
        """Add the predecessor transition score to a candidate set's unary logits.

        Args:
            candidate_ids: Candidate token ids ``[..., K]``.
            unary_logits: Backbone logits for those candidates ``[..., K]``.
            hidden_states: Backbone hidden at this position ``[..., H]``.
            predecessor_ids: Teacher-forced predecessor token ids ``[...]``.

        Returns:
            Selector logits over the candidate set ``[..., K]``.
        """
        predecessor = self.predecessor_codebook[predecessor_ids]
        successor = self.successor_codebook[candidate_ids]
        context = predecessor * self.hidden_projection(hidden_states)
        transition = torch.einsum("...r,...kr->...k", context.to(successor.dtype), successor)
        return unary_logits + transition

    @torch.no_grad()
    def greedy_path(self, candidate_ids, unary_logits, hidden_states, anchor_token_ids):
        """Walk the candidate lattice greedily, mirroring the serving-side path walk.

        Args:
            candidate_ids: ``[B, L, K]`` candidate ids per block position.
            unary_logits: ``[B, L, K]`` backbone logits for those candidates.
            hidden_states: ``[B, L, H]`` backbone hidden per block position.
            anchor_token_ids: ``[B]`` the verified token preceding the block.

        Returns:
            Selected token ids ``[B, L]``.
        """
        predecessor_ids = anchor_token_ids
        path = []
        for position in range(candidate_ids.shape[1]):
            scores = self.score_candidates(
                candidate_ids[:, position],
                unary_logits[:, position],
                hidden_states[:, position],
                predecessor_ids,
            )
            selected = scores.argmax(dim=-1, keepdim=True)
            predecessor_ids = candidate_ids[:, position].gather(1, selected)[:, 0]
            path.append(predecessor_ids)
        return torch.stack(path, dim=1)


class DFlash2Module(DFlashModule):
    """DFlash draft backbone with per-sublayer convolutions and a candidate selector."""

    def __init__(self, config):
        """Initialize the DFlash backbone, then attach the convolutions and the selector."""
        super().__init__(config)

        self.projector_type = getattr(config, "projector_type", "dflash2")

        def required_int(name: str) -> int:
            """Read an int architecture field, rejecting missing values and bools."""
            value = getattr(config, name, None)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"DFlash2 (projector_type='dflash2') requires an integer "
                    f"'{name}' in dflash_architecture_config, got {value!r}."
                )
            return value

        taps = required_int("conv_kernel_size")
        group_size = required_int("conv_group_size")
        rank = required_int("selector_rank")
        top_k = required_int("selector_top_k")

        std = getattr(config, "initializer_range", 0.02)

        # Replace each layer's no-op sublayer wrappers with real convolutions. The
        # backbone layer forward already calls prepare()/finish() around attention
        # and the MLP, so nothing else in the layer changes.
        for layer in self.layers:
            for wrapper_name in ("attention_conv", "mlp_conv"):
                setattr(
                    layer,
                    wrapper_name,
                    DFlashGroupedConv(
                        hidden_size=config.hidden_size,
                        block_size=self.block_size,
                        taps=taps,
                        group_size=group_size,
                    ),
                )

        self.candidate_selector = CandidateSelector(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            rank=rank,
            top_k=top_k,
            std=std,
        )

        # DFlashModule.__init__ already ran _init_weights before these modules
        # existed, so initialize the new Linear layers explicitly. base_kernel and
        # the codebooks keep the init set in their own constructors.
        self._init_head_weights(std)

    def _init_head_weights(self, std: float):
        """Initialize the convolution and selector Linear layers (matching HF _init_weights)."""
        linears = [self.candidate_selector.hidden_projection]
        for layer in self.layers:
            linears += [layer.attention_conv.kernel_projection, layer.mlp_conv.kernel_projection]
        for module in linears:
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
