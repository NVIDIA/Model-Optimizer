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

# Adapted from https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/modeling/dspark/markov_head.py
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

"""DSpark draft module — DFlash backbone plus a lightweight sequential (Markov) head.

DSpark (DeepSeek-AI, "DSpark: Confidence-Scheduled Speculative Decoding with
Semi-Autoregressive Generation") shares Domino's idea: keep the parallel DFlash
backbone for speed and add a lightweight sequential head that injects the
intra-block causal dependency the parallel backbone lacks (mitigating suffix
acceptance decay). Where Domino uses a GRU over base-model token embeddings,
DSpark adds a *prefix-dependent transition bias* ``B_k`` to the backbone's base
logits, inducing a causal block distribution
``p_k(x_k | x_0, x_<k) = softmax(U_k + B_k)``.

This module owns the head parameters (and an optional confidence head); the
training wrapper (``HFDSparkModel`` in ``hf_dspark.py``) orchestrates the
teacher-forced forward and the loss. Three head variants are supported:

- ``vanilla``: ``B(x_{k-1}) = W2(W1[x_{k-1}])`` — a memoryless first-order
  transition, factorized low-rank (``markov_w1: vocab->rank``,
  ``markov_w2: rank->vocab``). Cheapest; uses neither the backbone hidden nor
  recurrence.
- ``gated``: gates the previous-token embedding by the backbone hidden before
  projecting: ``B = W2(sigmoid(gate_proj([h_k; W1[x_{k-1}]])) * W1[x_{k-1}])``.
- ``rnn``: a GRU-like recurrent head carrying a state ``s_k`` across positions in
  the block, so position ``k`` sees the full prefix ``x_<k`` (closest analogue to
  Domino's GRU).

The head uses its OWN embedding table (``markov_w1``), not the base model's, so
the bias computation is fully self-contained on this module. Submodule names
(``markov_w1`` / ``markov_w2`` / ``gate_proj`` / ``joint_proj`` /
``confidence_proj``) match the upstream DeepSpec checkpoint layout so exported
checkpoints stay portable.
"""

import torch
from torch import nn

from .modeling_dflash import DFlashModule

__all__ = ["DSparkModule"]


class DSparkModule(DFlashModule):
    """DFlash draft backbone augmented with the DSpark sequential (Markov) head."""

    def __init__(self, config):
        """Initialize the DFlash backbone, then add the Markov + (optional) confidence head."""
        super().__init__(config)

        self.projector_type = getattr(config, "projector_type", "dspark")
        self.markov_rank = int(config.markov_rank)
        self.markov_head_type = str(getattr(config, "markov_head_type", "vanilla")).lower()
        # DSpark treats the anchor as the first prediction position (next-token
        # alignment), matching Domino's shift_label path; kept as an attribute so
        # the wrapper and exporter can read it uniformly.
        self.shift_label = True
        if self.markov_rank <= 0:
            raise ValueError(f"DSpark requires markov_rank > 0, got {self.markov_rank}.")
        if self.markov_head_type not in ("vanilla", "gated", "rnn"):
            raise ValueError(
                f"Unsupported markov_head_type: {self.markov_head_type!r}. "
                "Expected 'vanilla', 'gated' or 'rnn'."
            )

        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        r = self.markov_rank

        # Low-rank first-order transition: W1 is an embedding lookup over the
        # previous token, W2 projects the rank-r state back to a vocab logit bias.
        self.markov_w1 = nn.Embedding(vocab_size, r)
        self.markov_w2 = nn.Linear(r, vocab_size, bias=False)
        if self.markov_head_type == "gated":
            self.gate_proj = nn.Linear(hidden_size + r, r)
        elif self.markov_head_type == "rnn":
            # Joint [gate; candidate; output] projection over [s_{k-1}; W1[x_{k-1}]; h_k].
            self.joint_proj = nn.Linear(2 * r + hidden_size, 3 * r)

        # Optional confidence head: predicts the per-position acceptance probability
        # (supervised in the wrapper by the DSpark confidence BCE loss).
        self.use_confidence_head = bool(getattr(config, "use_confidence_head", False))
        if self.use_confidence_head:
            self.confidence_proj = nn.Linear(hidden_size + r, 1)

        # DFlashModule.__init__ already ran _init_weights before these modules
        # existed, so initialize the new layers explicitly.
        self._init_head_weights(config)

    def _init_head_weights(self, config):
        """Initialize the head Linear/Embedding layers (matching HF _init_weights std)."""
        std = getattr(config, "initializer_range", 0.02)
        modules = [self.markov_w1, self.markov_w2]
        if self.markov_head_type == "gated":
            modules.append(self.gate_proj)
        elif self.markov_head_type == "rnn":
            modules.append(self.joint_proj)
        if self.use_confidence_head:
            modules.append(self.confidence_proj)
        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def prev_token_embeddings(self, prev_ids: torch.Tensor) -> torch.Tensor:
        """Look up the Markov embedding ``W1[x_{k-1}]`` of the teacher-forced prev tokens."""
        return self.markov_w1(prev_ids.long())

    def compute_markov_bias(self, prev_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Compute the transition bias ``B_k`` added to the backbone base logits.

        Args:
            prev_ids: Teacher-forced previous-token ids per block position [B, N, block_size].
            hidden: Backbone hidden states [B, N, block_size, H] (used by gated/rnn heads).

        Returns:
            Logit bias [B, N, block_size, vocab].
        """
        prev_emb = self.prev_token_embeddings(prev_ids)  # [B, N, bs, r]

        if self.markov_head_type == "vanilla":
            return self.markov_w2(prev_emb)

        if self.markov_head_type == "gated":
            gate = torch.sigmoid(self.gate_proj(torch.cat([hidden, prev_emb], dim=-1)))
            return self.markov_w2(gate.to(prev_emb.dtype) * prev_emb)

        # rnn: unroll the gated recurrence over the block dimension.
        block_size = prev_ids.shape[-1]
        leading = prev_emb.shape[:-2]  # [B, N]
        state = torch.zeros(*leading, self.markov_rank, device=prev_emb.device, dtype=hidden.dtype)
        biases = []
        for k in range(block_size):
            state, bias = self._rnn_step(state, prev_emb[..., k, :], hidden[..., k, :])
            biases.append(bias)
        return torch.stack(biases, dim=-2)

    def _rnn_step(self, state, prev_emb, hidden):
        """One GRU-like recurrent step. Returns (new_state [.., r], bias [.., vocab])."""
        z = torch.cat([state, prev_emb, hidden], dim=-1)
        gate_raw, candidate_raw, output_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.markov_w2(torch.tanh(output_raw))
        return new_state, bias

    def markov_step(self, prev_token: torch.Tensor, hidden: torch.Tensor, state=None):
        """One autoregressive Markov step (inference): bias for a single position.

        Args:
            prev_token: Previously decoded token ids [B].
            hidden: Backbone hidden at this position [B, H] (used by gated/rnn).
            state: Recurrent state [B, r] (rnn head only); None initializes to zero.

        Returns:
            (bias [B, vocab], new_state [B, r] | None) — ``new_state`` is the input
            ``state`` unchanged for the memoryless heads.
        """
        prev_emb = self.prev_token_embeddings(prev_token)
        if self.markov_head_type == "vanilla":
            return self.markov_w2(prev_emb), state
        if self.markov_head_type == "gated":
            gate = torch.sigmoid(self.gate_proj(torch.cat([hidden, prev_emb], dim=-1)))
            return self.markov_w2(gate.to(prev_emb.dtype) * prev_emb), state
        # rnn
        if state is None:
            state = torch.zeros(
                prev_emb.shape[0], self.markov_rank, device=prev_emb.device, dtype=hidden.dtype
            )
        state, bias = self._rnn_step(state, prev_emb, hidden)
        return bias, state

    def compute_confidence_logits(
        self, prev_ids: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        """Per-position acceptance-probability logits ``c_k = w^T[h_k; W1[x_{k-1}]]``.

        Returns logits [B, N, block_size] (pass through sigmoid for a probability).
        """
        prev_emb = self.prev_token_embeddings(prev_ids)
        return self.confidence_proj(torch.cat([hidden, prev_emb], dim=-1)).squeeze(-1)
