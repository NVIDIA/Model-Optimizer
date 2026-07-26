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

"""Generic (AnyModel) dynamic single-block pruning — no realized weights.

Makes one block behave as pruned during a forward pass, used by replace-1-block scoring (load the
sorted teacher once, prune one block per candidate). The mechanism is descriptor-driven by
module name:

* **Removal** (FFN top-K, attention head/group removal): a forward-pre-hook **masks** the dropped
  channels/heads at the ``down_proj`` / ``o_proj`` input. Exact — the masked-forward output equals
  the pruned model's output (the dropped channels/heads contribute zero downstream). Works on a
  plain tensor or a sharded ``DTensor`` (the mask is distributed to the activation's placement).
The masks/keep-sets come from :mod:`.attention_ffn_surgery`; the descriptor supplies the
``down_proj``/``o_proj``/``k_proj``/``v_proj`` module names, so this file is model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .attention_ffn_surgery import attention_keep_mask, ffn_keep_mask, sorted_attention_keep_indices

__all__ = [
    "FFNRemovalSpec",
    "AttnRemovalSpec",
    "apply_prune_hooks",
    "register_mask_hook",
    "build_block_prune_specs",
]


def _apply_feature_mask(x: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Multiply ``x[..., features]`` by a boolean ``[features]`` keep mask (plain or DTensor)."""
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

    if isinstance(x, DTensor):
        mask_shape = (1,) * (x.ndim - 1) + (keep_mask.numel(),)
        mask = keep_mask.to(dtype=x.dtype).reshape(mask_shape)
        feature_dim = x.ndim - 1
        mask_placements = tuple(
            p if isinstance(p, Shard) and p.dim == feature_dim else Replicate()
            for p in x.placements
        )
        m = distribute_tensor(mask, x.device_mesh, mask_placements)
        return x * m
    return x * keep_mask.to(dtype=x.dtype, device=x.device)


@dataclass
class FFNRemovalSpec:
    """Mask the ``down_proj`` input to keep ``keep_mask`` intermediate channels."""

    module_name: str  # e.g. "...layers.5.mlp.down_proj"
    keep_mask: torch.Tensor  # bool [intermediate]


@dataclass
class AttnRemovalSpec:
    """Mask the ``o_proj`` input to keep ``keep_mask`` query-head columns."""

    module_name: str  # e.g. "...layers.5.self_attn.o_proj"
    keep_mask: torch.Tensor  # bool [num_q * head_dim]


def _mask_prehook(keep_mask):
    def hook(module, args):
        return (_apply_feature_mask(args[0], keep_mask), *args[1:])

    return hook


def register_mask_hook(module, keep_mask):
    """Register the input-masking pre-hook on a resolved module object; returns the handle."""
    return module.register_forward_pre_hook(_mask_prehook(keep_mask))


def apply_prune_hooks(model, specs) -> list:
    """Register removal masks for the given specs; returns hook handles to ``.remove()``.

    ``model.get_submodule(name)`` resolves each module (descriptor-supplied names), so this is
    model-agnostic.
    """
    handles = []
    for spec in specs:
        if isinstance(spec, (FFNRemovalSpec, AttnRemovalSpec)):
            module = model.get_submodule(spec.module_name)
            handles.append(module.register_forward_pre_hook(_mask_prehook(spec.keep_mask)))
    return handles


def build_block_prune_specs(
    *,
    down_proj_name: str | None,
    o_proj_name: str | None,
    orig_intermediate: int | None,
    target_intermediate: int | None,
    orig_num_q: int | None,
    orig_num_kv: int | None,
    target_num_q: int | None,
    target_num_kv: int | None,
    head_dim: int | None,
) -> list:
    """Map a per-block target onto the right D3 spec(s) for a *sorted* block.

    FFN target K (< orig) -> mask down_proj to the prefix ``[:K]``. Attention
    removal keeps the first ``target_num_kv`` sorted groups and the first
    ``target_num_q/target_num_kv`` sorted query heads in each kept group.  A
    q-preserving KV merge is intentionally not supported in the new Puzzletron
    attention contract.
    Module names are the *loaded model's* paths (caller resolves them). Returns the spec list.
    """
    specs: list = []
    if target_intermediate is not None and orig_intermediate and target_intermediate < orig_intermediate:
        specs.append(
            FFNRemovalSpec(down_proj_name, ffn_keep_mask(orig_intermediate, torch.arange(target_intermediate)))
        )

    # A zero-sized attention target is the typed ``no_op`` representation.  It
    # is handled by the enclosing block runtime hook, which zeros the complete
    # attention sublayer output.  Group/head slicing only applies to positive
    # attention shapes; attempting GQA arithmetic on the no-op's 0/0 shape
    # would divide by zero before that hook is installed.
    if (
        target_num_kv is not None
        and target_num_q is not None
        and target_num_kv > 0
        and target_num_q > 0
        and orig_num_kv
    ):
        # Every kept KV group keeps an equal number of query heads (regular GQA).
        assert target_num_q % target_num_kv == 0, (
            f"target_num_q {target_num_q} not divisible by target_num_kv {target_num_kv}"
        )
        if target_num_q < orig_num_q or target_num_kv < orig_num_kv:
            m = target_num_q // target_num_kv
            orig_m = orig_num_q // orig_num_kv
            assert m <= orig_m, (
                "Attention pruning removes whole KV groups and/or a uniform number of query heads "
                f"per group; got target_heads_per_group={m} > orig_heads_per_group={orig_m}. "
                "Use target_num_q = target_num_kv * orig_heads_per_group when reducing KV groups."
            )
            keep_q, _ = sorted_attention_keep_indices(target_num_kv, m, orig_num_q // orig_num_kv)
            specs.append(AttnRemovalSpec(o_proj_name, attention_keep_mask(orig_num_q, keep_q, head_dim)))
    return specs
