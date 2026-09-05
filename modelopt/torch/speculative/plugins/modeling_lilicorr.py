# Adapted from https://github.com/sgl-project/SpecForge
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

"""LiLiCorr draft module — DFlash backbone plus a lattice reranker over its candidates.

LiLiCorr (NVIDIA, "LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts
for Speculative Decoding", https://arxiv.org/abs/2608.20530) addresses the failure
mode a purely parallel drafter has: DFlash is trained on per-position *marginals*
rather than the joint block distribution, so the tokens it emits are individually
plausible yet jointly incoherent.

Instead of committing the per-slot argmax, the drafter keeps its top-``k``
candidates at each of the ``block_size - 1`` slots and LiLiCorr scores the
resulting lattice:

- Each candidate is embedded, projected, and combined with an **anchor** — one
  projected row of the committed target hidden state, the position immediately
  before the block. Rank and log-probability features from the drafter's own
  distribution are concatenated in, so the head sees how confident the drafter was
  and how close the near-ties are.
- A small bidirectional transformer over the ``slots x k`` lattice produces one
  hidden state per candidate (:class:`LiLiCorrLayer`).
- Two heads read it: ``out_head`` and ``in_head``. Adjacent candidates *match*
  when the earlier one's ``out`` vector has high cosine similarity with the later
  one's ``in`` vector, so the block's joint structure is captured without the
  joint distribution ever being materialized.

The factors are bounded by construction: every bilinear operand is L2-normalized
along ``factor_dim``, so each score is a cosine in ``[-1, 1]`` and the factor
magnitude is set solely by a fixed, non-learnable ``logit_scale``. That is what
makes the selection stable without a clamp, a squash, or a separate grad clip.

Serving commits a path greedily, left to right. Greedy is deliberate, not an
approximation: speculative decoding accepts the longest correct *prefix*, so every
prefix event contributes to ``E[tau]`` and the early ones dominate. Maximizing the
whole-block score instead (exactly solvable by Viterbi over this chain) trades an
early slot for a better tail, which is rational for the path score and
self-defeating for prefix acceptance.

This module owns the head parameters only; the training wrapper
(``HFLiLiCorrModel`` in ``hf_lilicorr.py``) orchestrates the forward, the
three-term objective and the metrics. Submodule names (``token_proj`` /
``pass_hidden_proj`` / ``feature_mlp`` / ``slot_embedding`` / ``rank_embedding`` /
``relative_slot_bias`` / ``same_slot_bias`` / ``context_proj`` / ``output_norm`` /
``anchor_norm`` / ``factor_input_proj`` / ``out_head`` / ``in_head`` /
``anchor_out_head``) match the serving loader, so an exported checkpoint is served
directly.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm as _NORM_CLS  # noqa: N814

from .modeling_dflash import DFlashModule

__all__ = ["LiLiCorrHead", "LiLiCorrLayer", "LiLiCorrModule"]

# Per-candidate input features, in the order the first Linear expects:
#   soft rank: [log_probs, probs, logprob_gap]
#   hard rank: [rank_frac, is_top1]
_NUM_CANDIDATE_FEATURES = 5


class LiLiCorrLayer(nn.Module):
    """One bidirectional attention layer over the candidate lattice.

    Attention is unmasked across the whole ``slots x k`` lattice — a candidate may
    look at any other, in either direction — because the lattice is a scoring
    structure, not a sequence being generated. Position enters only through the
    slot/rank embeddings and the relative-slot bias.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, rms_norm_eps: float):
        """Build the pre-norm attention and MLP sublayers."""
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"LiLiCorr hidden_size={hidden_size} must be divisible by num_heads={num_heads}."
            )
        self.attn_norm = _NORM_CLS(hidden_size, eps=rms_norm_eps)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=0.0)
        self.mlp_norm = _NORM_CLS(hidden_size, eps=rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor, attention_bias: torch.Tensor) -> torch.Tensor:
        """Pre-norm attention with an additive bias, then the MLP."""
        attn_input = self.attn_norm(hidden_states)
        attn_output, _ = self.attn(
            attn_input, attn_input, attn_input, attn_mask=attention_bias, need_weights=False
        )
        hidden_states = hidden_states + attn_output
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class LiLiCorrHead(nn.Module):
    """Anchor-conditioned chain scorer over the drafter's per-slot top-k candidates.

    Produces the two factors a chain needs: a ``start`` score for slot 0 against
    the anchor, and a ``pair`` score for every transition between adjacent slots.
    Both are cosine similarities scaled by ``logit_scale``; the wrapper composes
    them into log-potentials via :meth:`compose_log_factors`.

    Every geometry and scaling argument is required. ``candidate_topk`` sets the
    lattice width and the shape of ``rank_embedding``, while ``logit_scale`` and
    ``vector_eps`` change the score without changing any shape — so a defaulted
    value would build a head that loads cleanly and scores a different function.
    """

    def __init__(
        self,
        *,
        model_hidden_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        block_size: int,
        candidate_topk: int,
        factor_dim: int,
        rms_norm_eps: float,
        vector_eps: float,
        logit_scale: float,
    ):
        """Validate the geometry, then build the lattice tower and the two factor heads."""
        super().__init__()
        if block_size < 2:
            raise ValueError(f"LiLiCorr block_size must be >= 2, got {block_size}.")
        if candidate_topk < 1:
            raise ValueError(f"LiLiCorr candidate_topk must be >= 1, got {candidate_topk}.")
        if num_layers <= 0:
            raise ValueError(f"LiLiCorr num_layers must be positive, got {num_layers}.")
        if factor_dim <= 0:
            raise ValueError(f"LiLiCorr factor_dim must be positive, got {factor_dim}.")
        if not 0.0 < vector_eps < 0.5:
            raise ValueError(f"LiLiCorr vector_eps must be in (0, 0.5), got {vector_eps}.")
        if logit_scale <= 0.0:
            raise ValueError(f"LiLiCorr logit_scale must be positive, got {logit_scale}.")

        self.block_size = int(block_size)
        self.num_candidate_slots = self.block_size - 1
        self.candidate_topk = int(candidate_topk)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.factor_dim = int(factor_dim)
        self.vector_eps = float(vector_eps)
        self.logit_scale = float(logit_scale)

        # Identity when the head is as wide as the draft, so no redundant matmul.
        self.token_proj = (
            nn.Identity()
            if model_hidden_size == hidden_size
            else nn.Linear(model_hidden_size, hidden_size)
        )
        self.pass_hidden_proj = nn.Linear(model_hidden_size, hidden_size)
        self.feature_mlp = nn.Sequential(
            nn.LayerNorm(_NUM_CANDIDATE_FEATURES),
            nn.Linear(_NUM_CANDIDATE_FEATURES, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.slot_embedding = nn.Parameter(
            torch.zeros(1, 1, self.num_candidate_slots, 1, hidden_size)
        )
        # Rank-ordered, so its first `topk` rows are exactly the rows that apply to
        # ranks 0..topk-1. That is what lets a head trained at K serve any k <= K.
        self.rank_embedding = nn.Parameter(torch.zeros(1, 1, 1, self.candidate_topk, hidden_size))
        self.relative_slot_bias = nn.Parameter(torch.zeros(num_heads, 2 * self.block_size - 1))
        self.same_slot_bias = nn.Parameter(torch.zeros(num_heads))
        # The anchor is a row of the target hidden state, hence model_hidden_size
        # wide. One row has no sequence to position, so it carries no position
        # embedding and does not join the lattice attention: it feeds the factor
        # heads only.
        self.context_proj = nn.Linear(model_hidden_size, hidden_size)

        self.layers = nn.ModuleList(
            [
                LiLiCorrLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = _NORM_CLS(hidden_size, eps=rms_norm_eps)
        self.anchor_norm = _NORM_CLS(hidden_size, eps=rms_norm_eps)
        # The factor heads read [self, anchor, self * anchor].
        self.factor_input_proj = nn.Linear(hidden_size * 3, hidden_size)
        # Named by edge direction: a transition runs OUT of the earlier candidate
        # and IN to the later one, so pair = out[s] . in[s+1] and start = anchor_out . in[0].
        self.out_head = nn.Linear(hidden_size, factor_dim)
        self.in_head = nn.Linear(hidden_size, factor_dim)
        self.anchor_out_head = nn.Linear(hidden_size, factor_dim)
        self.reset_factor_heads()

    def reset_factor_heads(self) -> None:
        """Re-seed the three factor heads with a small non-zero std.

        Must run after any ``_init_weights`` sweep. The factors are dot products of
        the out/in embeddings, so zero-init on both sides makes the product's
        gradient vanish at init — a dead saddle. A small std keeps the initial
        scores near zero (near-uniform) while leaving gradient to break symmetry.
        """
        for head in (self.out_head, self.in_head, self.anchor_out_head):
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    def compose_log_factors(
        self, *, start_scores: torch.Tensor, pair_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn the cosine scores into the chain's log-potentials, ``logit_scale * cos``.

        ``logit_scale`` is a fixed, non-learnable constant that sets how sharp the chain
        may become, since a cosine in ``[-1, 1]`` is too flat to commit on. fp32 because
        the decode's argmax runs on these values.
        """
        return self.logit_scale * start_scores.float(), self.logit_scale * pair_scores.float()

    def _attention_bias(
        self, batch_blocks: int, *, device: torch.device, dtype: torch.dtype, topk: int
    ) -> torch.Tensor:
        """Build the per-head additive attention bias over the lattice.

        Relative slot distance, plus a same-slot term. This is pure lattice geometry with
        no token content in it, so one bias serves every block and is simply broadcast.

        Shapes below use a running example of ``block_size`` 4 (hence 3 slots) and
        ``topk`` 2, giving ``S = slots * topk = 6`` lattice positions.
        """
        # Which slot each of the S flattened positions belongs to: [0, 0, 1, 1, 2, 2] —
        # slot 0's two candidates, then slot 1's, then slot 2's.
        slot_ids = torch.arange(
            self.num_candidate_slots, device=device, dtype=torch.long
        ).repeat_interleave(topk)
        # Signed slot distance from query i to key j, [S, S]. A slot-1 row reads
        # [1, 1, 0, 0, -1, -1]: slot 0 behind it, itself, then slot 2 ahead.
        # The clamp is defensive and never binds — slots span at most block_size-2.
        rel = (slot_ids.view(-1, 1) - slot_ids.view(1, -1)).clamp(
            min=-(self.block_size - 1), max=self.block_size - 1
        )
        # One learned scalar per (attention head, distance); num_heads is the lattice
        # attention's head count (lilicorr_num_heads), not the draft's and not the out/in
        # factor heads. A column index cannot be negative, so adding block_size-1 centres
        # the signed distance over the table's 2*block_size-1 columns, putting distance 0
        # in the middle: with the running example's offset of 3, rel -2 -> column 1,
        # 0 -> column 3, +2 -> column 5. Gives [num_heads, S, S].
        bias = self.relative_slot_bias[:, rel + self.block_size - 1]
        # The topk x topk diagonal blocks: candidates competing for the same slot. This
        # coincides with rel == 0, which relative_slot_bias already covers, but it carries
        # its own learned scalar per head.
        same_slot = slot_ids.view(-1, 1) == slot_ids.view(1, -1)
        bias = bias + same_slot.unsqueeze(0).to(dtype=bias.dtype) * self.same_slot_bias.view(
            -1, 1, 1
        )
        bias = bias.to(device=device, dtype=dtype)
        # [batch_blocks * num_heads, S, S] is the layout nn.MultiheadAttention wants.
        return (
            bias.unsqueeze(0)
            .expand(batch_blocks, -1, -1, -1)
            .reshape(batch_blocks * self.num_heads, bias.shape[-2], bias.shape[-1])
        )

    def _project_anchor(
        self,
        *,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor | None,
        bsz: int,
        n_blocks: int,
    ) -> torch.Tensor:
        """Project one committed target row per block.

        Shape is checked rather than adapted: the anchor is the whole context the
        factor heads see, so a differently shaped one would load and score a
        different function of the same parameters.
        """
        expected = (bsz, n_blocks, 1)
        if tuple(anchor_hidden.shape[:3]) != expected:
            raise ValueError(
                f"anchor_hidden must be {expected} + [dim] — one row per block — "
                f"got {tuple(anchor_hidden.shape)}."
            )
        anchor_state = self.context_proj(
            anchor_hidden.reshape(bsz * n_blocks, anchor_hidden.shape[-1])
        )
        if anchor_valid is not None:
            # A block anchored at position 0 has no preceding committed token. Its input
            # row arrives already zeroed, but context_proj has a bias, so a zeroed input
            # still projects to that bias — zero the output to keep the anchor at zero.
            valid = anchor_valid.reshape(bsz * n_blocks).to(
                device=anchor_state.device, dtype=anchor_state.dtype
            )
            anchor_state = anchor_state * valid.unsqueeze(-1)
        return anchor_state

    def forward(
        self,
        *,
        candidate_token_ids: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        pass_hidden: torch.Tensor,
        embed_tokens: nn.Module,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score the lattice.

        Shapes below use ``B`` for the batch, ``N`` for the blocks per sequence, ``slots``
        for the ``block_size - 1`` predicted positions in a block, ``k`` for the candidates
        per slot and ``H`` for the draft's hidden size.

        Args:
            candidate_token_ids: Candidate ids ``[B, N, slots, k]``.
            candidate_log_probs: Their log-probs under the drafter ``[B, N, slots, k]``.
            pass_hidden: Draft hidden per slot ``[B, N, slots, H]``.
            embed_tokens: Embedding table for the candidate ids (target's).
            anchor_hidden: One committed target row per block ``[B, N, 1, H]``.
            anchor_valid: Whether each block's anchor is in range ``[B, N, 1]``.

        Returns:
            ``(start_scores, pair_scores)`` — ``[B, N, k]`` and
            ``[B, N, slots-1, k, k]``, both cosine similarities in ``[-1, 1]``.
        """
        if candidate_token_ids.ndim != 4:
            raise ValueError(
                "candidate_token_ids must be [batch, blocks, slots, k], got "
                f"{tuple(candidate_token_ids.shape)}."
            )
        bsz, n_blocks, n_slots, topk = candidate_token_ids.shape
        if n_slots != self.num_candidate_slots or topk > self.candidate_topk:
            raise ValueError(
                f"lattice slots={n_slots}, k={topk} is not compatible with "
                f"slots={self.num_candidate_slots}, trained k={self.candidate_topk} "
                "(need 1 <= k <= trained)."
            )
        if candidate_log_probs.shape != candidate_token_ids.shape:
            raise ValueError(
                "candidate_log_probs must match candidate_token_ids, got "
                f"{tuple(candidate_log_probs.shape)} vs {tuple(candidate_token_ids.shape)}."
            )
        if pass_hidden.shape[:3] != (bsz, n_blocks, n_slots):
            raise ValueError(
                f"pass_hidden must be [batch, blocks, slots, dim], got {tuple(pass_hidden.shape)}."
            )

        # Detached: the candidate embeddings are an input to the head, not a path
        # the head trains the target's embedding table through.
        token_states = self.token_proj(embed_tokens(candidate_token_ids).detach())
        pass_states = self.pass_hidden_proj(pass_hidden).unsqueeze(-2)

        log_probs = candidate_log_probs.float()
        probs = log_probs.exp()
        logprob_gap = log_probs - log_probs.max(dim=-1, keepdim=True).values
        # The candidate's rank as a fraction, top-1 at 0 and last at 1: [0, 1/3, 2/3, 1]
        # at topk 4. Identical for every slot and block, rank being a position within a
        # slot's own candidate list. At topk 1 there is no range to normalize over.
        if topk == 1:
            rank_frac = torch.zeros_like(log_probs)
        else:
            rank_frac = (
                torch.arange(topk, device=log_probs.device, dtype=log_probs.dtype).view(
                    1, 1, 1, topk
                )
                / float(topk - 1)
            ).expand_as(log_probs)
        is_top1 = torch.zeros_like(log_probs)
        is_top1[..., 0] = 1.0
        features = torch.stack([log_probs, probs, logprob_gap, rank_frac, is_top1], dim=-1)

        hidden_states = token_states + pass_states
        hidden_states = hidden_states + self.feature_mlp(features.to(dtype=token_states.dtype))
        hidden_states = hidden_states + self.slot_embedding.to(
            device=token_states.device, dtype=token_states.dtype
        )
        hidden_states = hidden_states + self.rank_embedding[:, :, :, :topk, :].to(
            device=token_states.device, dtype=token_states.dtype
        )
        hidden_states = hidden_states.reshape(
            bsz * n_blocks, n_slots * topk, token_states.shape[-1]
        )

        anchor_state = self._project_anchor(
            anchor_hidden=anchor_hidden, anchor_valid=anchor_valid, bsz=bsz, n_blocks=n_blocks
        )
        attention_bias = self._attention_bias(
            bsz * n_blocks, device=hidden_states.device, dtype=hidden_states.dtype, topk=topk
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_bias=attention_bias)
        hidden_states = self.output_norm(hidden_states).reshape(bsz, n_blocks, n_slots, topk, -1)
        anchor_state = self.anchor_norm(anchor_state).reshape(bsz, n_blocks, -1)

        anchor_for_candidate = anchor_state[:, :, None, None, :].expand_as(hidden_states)
        factor_hidden = F.silu(
            self.factor_input_proj(
                torch.cat(
                    [
                        hidden_states,
                        anchor_for_candidate,
                        hidden_states * anchor_for_candidate,
                    ],
                    dim=-1,
                )
            )
        )
        # L2-normalize every bilinear operand so each score is a cosine in [-1, 1]:
        # the factor's magnitude is then set solely by the fixed logit_scale, and
        # direction carries all the learned signal. Bounding the direction rather
        # than squashing the raw score keeps the gradient largest at the
        # near-uniform operating point instead of saturating there.
        out_vec = F.normalize(self.out_head(factor_hidden), dim=-1, eps=self.vector_eps)
        in_vec = F.normalize(self.in_head(factor_hidden), dim=-1, eps=self.vector_eps)
        anchor_out = F.normalize(self.anchor_out_head(anchor_state), dim=-1, eps=self.vector_eps)

        start_scores = (anchor_out[:, :, None, :] * in_vec[:, :, 0, :, :]).sum(dim=-1)
        pair_scores = (out_vec[:, :, :-1, :, None, :] * in_vec[:, :, 1:, None, :, :]).sum(dim=-1)
        return start_scores, pair_scores


class LiLiCorrModule(DFlashModule):
    """DFlash draft backbone augmented with the LiLiCorr lattice reranker."""

    def __init__(self, config):
        """Initialize the DFlash backbone, then attach the reranker head."""
        super().__init__(config)

        self.projector_type = getattr(config, "projector_type", "lilicorr")

        def required(name: str, cast):
            """Read an architecture field the recipe must set.

            Deliberately not defaulted — see :class:`LiLiCorrHead`. Some of these
            change tensor shapes and would be caught at load; ``logit_scale`` and
            ``vector_eps`` do not.
            """
            value = getattr(config, name, None)
            if value is None or isinstance(value, bool):
                raise ValueError(
                    f"LiLiCorr (projector_type='lilicorr') requires '{name}' in "
                    "dflash_architecture_config; it defines the head that was trained. "
                    "Warm-starting reproduces a head only if every field here matches the "
                    "checkpoint's own: dflash_init_checkpoint restores weights and takes the "
                    "geometry from this config, so a mismatch in one of the two shape-free "
                    "fields (lilicorr_logit_scale, lilicorr_vector_eps) loads cleanly and "
                    "scores a different function."
                )
            return cast(value)

        # Defaults to the draft width, which is what omitting the key records.
        head_hidden_size = int(getattr(config, "lilicorr_hidden_size", 0) or config.hidden_size)

        self.lilicorr = LiLiCorrHead(
            model_hidden_size=config.hidden_size,
            hidden_size=head_hidden_size,
            num_layers=required("lilicorr_num_layers", int),
            num_heads=required("lilicorr_num_heads", int),
            mlp_ratio=required("lilicorr_mlp_ratio", float),
            block_size=self.block_size,
            candidate_topk=required("lilicorr_candidate_topk", int),
            factor_dim=required("lilicorr_factor_dim", int),
            rms_norm_eps=config.rms_norm_eps,
            vector_eps=required("lilicorr_vector_eps", float),
            logit_scale=required("lilicorr_logit_scale", float),
        )
        # DFlashModule.__init__ ran _init_weights before this head existed, so the
        # head's own Linears still carry PyTorch's default init. Re-run the draft's
        # convention over them, then re-seed the factor heads: a std sweep over those
        # three would leave the bilinear product at a dead saddle.
        self._init_head_weights(config)
        self.lilicorr.reset_factor_heads()

        # Optional: wrap every draft sublayer in DFlash2's grouped dynamic convolution,
        # installed only when the recipe asks for it, so this module still builds the
        # plain reranker when the two geometry keys are absent. Last in __init__ on
        # purpose -- _init_head_weights above iterates self.lilicorr.modules(), and
        # keeping the convolutions out of its reach is what makes the init below
        # authoritative.
        taps = getattr(config, "conv_kernel_size", None)
        group_size = getattr(config, "conv_group_size", None)
        if taps is not None and group_size is not None:
            self._install_sublayer_convs(config, int(taps), int(group_size))

    def _install_sublayer_convs(self, config, taps: int, group_size: int) -> None:
        """Replace each backbone layer's no-op sublayer wrappers with grouped convs.

        Reuses ``DFlash2``'s :class:`DFlashGroupedConv` and the no-op sublayer seam
        ``DFlashDecoderLayer`` already exposes, so the convolution itself is shared code
        rather than a second implementation of the same arithmetic.

        The initialization is the one deliberate difference. DFlash2 draws
        ``kernel_projection`` from ``normal_(0, initializer_range)``, so its convolution
        is not the identity at step 0. Here it is zero by default, and since
        ``base_kernel`` is identity at tap 0 the whole wrapper is then an *exact*
        identity at init: ``prepare`` emits a zero dynamic kernel, ``coefficients ==
        base``, and the convolution returns its input unchanged. That makes the
        difference between a conv and a non-conv run attributable to the convolutions
        rather than to a perturbed starting point.

        ``conv_projection_init_std`` is a separate key from ``initializer_range`` on
        purpose: the latter also seeds the reranker, so overloading it would couple two
        unrelated initializations.

        The import is deferred to here rather than taken at module scope because
        ``modeling_dflash2`` is the only part of LiLiCorr that needs DFlash2 at all.
        Importing it eagerly would make the whole plugin -- including the plain
        reranker, which shares no code with DFlash2 -- unimportable wherever DFlash2 is
        absent. Deferring it keeps that cost on the one recipe that asks for it.
        """
        try:
            from .modeling_dflash2 import DFlashGroupedConv
        except ImportError as exc:
            raise ImportError(
                "LiLiCorr's optional sublayer convolutions reuse DFlash2's "
                "DFlashGroupedConv, which is not available in this installation. Remove "
                "conv_kernel_size and conv_group_size from dflash_architecture_config to "
                "train the plain LiLiCorr reranker, or install a version of ModelOpt that "
                "provides the DFlash2 draft variant."
            ) from exc

        std = float(getattr(config, "conv_projection_init_std", 0.0))
        for layer in self.layers:
            for wrapper_name in ("attention_conv", "mlp_conv"):
                conv = DFlashGroupedConv(
                    hidden_size=config.hidden_size,
                    block_size=self.block_size,
                    taps=taps,
                    group_size=group_size,
                )
                with torch.no_grad():
                    if std > 0.0:
                        conv.kernel_projection.weight.normal_(mean=0.0, std=std)
                    else:
                        conv.kernel_projection.weight.zero_()
                setattr(layer, wrapper_name, conv)

    def _init_head_weights(self, config):
        """Initialize the head's Linear layers to the draft's own convention."""
        std = getattr(config, "initializer_range", 0.02)
        for module in self.lilicorr.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project captured target hidden states into the draft's hidden space.

        The same fusion :meth:`DFlashModule.forward` applies to the injected context,
        exposed as a method because the head's anchor is a row of that projected
        sequence and must be produced by the same function.
        """
        return self.hidden_norm(self.fc(target_hidden))

    def build_anchor_rows(
        self, *, target_hidden: torch.Tensor, anchor_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather one committed target row per block and project it.

        The anchor is the position immediately *before* the block — the last row the
        target has committed when the block is drafted. ``anchor_positions`` is
        required rather than defaulted: falling back to "the last row of the sequence"
        produces a different anchor from the trained one, which loads cleanly and
        scores plausibly wrong.

        Returns ``(anchor_hidden [B, N, 1, H], anchor_valid [B, N, 1])``, for ``B`` the
        batch, ``N`` the blocks per sequence and ``H`` the draft's hidden size. A block
        whose anchor falls before the sequence start is marked invalid and zeroed.
        """
        if anchor_positions.ndim != 2:
            raise ValueError(
                f"anchor_positions must be [batch, blocks], got {tuple(anchor_positions.shape)}."
            )
        bsz, seq_len, capture_dim = target_hidden.shape
        if anchor_positions.shape[0] != bsz:
            raise ValueError(
                f"anchor_positions batch {anchor_positions.shape[0]} does not match "
                f"target_hidden batch {bsz}."
            )

        # anchor-1, not anchor: block position 0 carries the anchor token itself, given
        # rather than predicted. The target has no row conditioned on that token — it is
        # the freshly committed one — so the row before it is the newest usable context,
        # and the one serving can supply without a further target pass.
        indices = anchor_positions.to(target_hidden.device) - 1
        valid = (indices >= 0) & (indices < seq_len)
        safe_indices = indices.clamp(min=0, max=max(seq_len - 1, 0)).long()
        rows = torch.gather(
            target_hidden, 1, safe_indices.unsqueeze(-1).expand(-1, -1, capture_dim)
        )
        # `fc` and `hidden_norm` are both row-wise, so projecting the gathered rows is
        # the same function as projecting the sequence and gathering afterwards — and
        # it avoids a second full-sequence projection for the few rows in use.
        anchor_hidden = self.project_target_hidden(rows)
        anchor_hidden = torch.where(
            valid.unsqueeze(-1), anchor_hidden, torch.zeros_like(anchor_hidden)
        )
        return anchor_hidden.unsqueeze(2), valid.unsqueeze(-1)

    def score_lattice(
        self,
        *,
        candidate_token_ids: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        pass_hidden: torch.Tensor,
        embed_tokens: nn.Module,
        target_hidden: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score the candidate lattice, resolving the anchor from the target hidden.

        Returns ``(start_scores, pair_scores)`` — see :meth:`LiLiCorrHead.forward`.
        """
        anchor_hidden, anchor_valid = self.build_anchor_rows(
            target_hidden=target_hidden, anchor_positions=anchor_positions
        )
        return self.lilicorr(
            candidate_token_ids=candidate_token_ids,
            candidate_log_probs=candidate_log_probs,
            pass_hidden=pass_hidden,
            embed_tokens=embed_tokens,
            anchor_hidden=anchor_hidden,
            anchor_valid=anchor_valid,
        )
