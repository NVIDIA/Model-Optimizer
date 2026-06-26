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

"""Monkey patch for Megatron router expert-bias paths on heterogeneous MoE.

Megatron's default ``_update_router_expert_bias`` stacks all MoE layers together:

    torch.stack(tokens_per_expert_list, dim=0)

This fails when layer-local expert counts differ (e.g. 16/64/96/128 experts across layers).
The patch below groups layers by expert-vector length and updates each group independently.

With ``moe_router_enable_expert_bias=True``, ``reset_model_temporary_tensors`` also calls
``module.local_tokens_per_expert.zero_()`` for every module that has ``expert_bias``. On
hybrid / distillation models some of those buffers are ``None`` (eval-only teacher, dense
layers, or experts not touched on a rank), which raises ``AttributeError``. The patch skips
modules whose token counter was never allocated.

Usage
-----
Call once before training starts (for example from :meth:`PuzzletronHooks.before_load_models`):

    from router_expert_bias_patch import apply_patch as apply_router_expert_bias_patch
    apply_router_expert_bias_patch()
"""

from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from importlib import import_module
from typing import Any

import torch

logger = logging.getLogger(__name__)

_PATCH_SENTINEL = "_puzzletron_router_expert_bias_patch_applied"
_ORIG_FN_ATTR = "_puzzletron_orig_update_router_expert_bias"
_ORIG_RESET_FN_ATTR = "_puzzletron_orig_reset_model_temporary_tensors"


def _has_active_expert_bias_counters(module: torch.nn.Module) -> bool:
    """True when a module participates in router expert-bias bookkeeping."""
    return (
        hasattr(module, "expert_bias")
        and module.expert_bias is not None
        and module.local_tokens_per_expert is not None
    )


def _safe_reset_model_temporary_tensors(
    config: Any,
    model: list[torch.nn.Module],
) -> None:
    """Heterogeneous-safe replacement for Megatron's temporary MoE tensor reset."""
    from megatron.core.utils import get_attr_wrapped_model

    for model_chunk in model:
        for module in get_attr_wrapped_model(model_chunk, "modules")():
            if config.moe_router_enable_expert_bias and _has_active_expert_bias_counters(module):
                module.local_tokens_per_expert.zero_()
            if (
                config.moe_router_load_balancing_type == "global_aux_loss"
                or "global_aux_loss" in config.moe_router_load_balancing_type
            ) and hasattr(module, "reset_global_aux_loss_tracker"):
                module.reset_global_aux_loss_tracker()


def _grouped_update_router_expert_bias(
    model: list[torch.nn.Module],
    config: Any,
    tp_dp_cp_group: torch.distributed.ProcessGroup | None = None,
) -> None:
    """Heterogeneous-safe replacement for Megatron's router expert-bias update."""
    from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias
    from megatron.core.utils import get_attr_wrapped_model

    per_layer_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for model_chunk in model:
        # Keep the same train/eval behavior as upstream:
        # update only modules in train mode (teacher may be eval-only during KD).
        per_layer_pairs.extend(
            (module.local_tokens_per_expert, module.expert_bias)
            for module in get_attr_wrapped_model(model_chunk, "modules")()
            if module.training and _has_active_expert_bias_counters(module)
        )

    # Hybrid models may include no MoE layers on some stages.
    if not per_layer_pairs:
        return

    grouped_pairs: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    for tokens_per_expert, expert_bias in per_layer_pairs:
        if tokens_per_expert.shape != expert_bias.shape:
            raise RuntimeError(
                "router expert bias shape mismatch: "
                f"tokens_per_expert.shape={tokens_per_expert.shape}, "
                f"expert_bias.shape={expert_bias.shape}"
            )
        grouped_pairs[tokens_per_expert.numel()].append((tokens_per_expert, expert_bias))

    # Deterministic order is important so all ranks launch collectives identically.
    for num_experts in sorted(grouped_pairs):
        pairs = grouped_pairs[num_experts]
        stacked_tokens_per_expert = torch.stack([p[0] for p in pairs], dim=0)
        stacked_expert_bias = torch.stack([p[1] for p in pairs], dim=0)
        update_kwargs = {}
        # Megatron versions differ: some expose tp_dp_cp_group, some don't.
        if "tp_dp_cp_group" in inspect.signature(get_updated_expert_bias).parameters:
            update_kwargs["tp_dp_cp_group"] = tp_dp_cp_group

        stacked_updated_expert_bias = get_updated_expert_bias(
            stacked_tokens_per_expert,
            stacked_expert_bias,
            config.moe_router_bias_update_rate,
            **update_kwargs,
        )

        for (_, expert_bias), updated_expert_bias in zip(pairs, stacked_updated_expert_bias):
            expert_bias.copy_(updated_expert_bias)


def apply_patch() -> None:
    """Patch Megatron router expert-bias helpers with heterogeneous-safe logic."""
    from megatron.bridge.utils.common_utils import get_rank_safe

    fmg = import_module("megatron.core.distributed.finalize_model_grads")

    if getattr(fmg, _PATCH_SENTINEL, False):
        logger.debug("router_expert_bias_patch.apply_patch: already patched; skipping")
        if get_rank_safe() == 0:
            print(
                "[RouterExpertBiasPatch-install] already patched; skipping",
                flush=True,
            )
        return

    setattr(fmg, _ORIG_FN_ATTR, fmg._update_router_expert_bias)
    fmg._update_router_expert_bias = _grouped_update_router_expert_bias
    setattr(fmg, _ORIG_RESET_FN_ATTR, fmg.reset_model_temporary_tensors)
    fmg.reset_model_temporary_tensors = _safe_reset_model_temporary_tensors
    setattr(fmg, _PATCH_SENTINEL, True)
    logger.info(
        "router_expert_bias_patch.apply_patch: patched "
        "megatron.core.distributed.finalize_model_grads._update_router_expert_bias "
        "and reset_model_temporary_tensors"
    )
    # Use print (not logger) so the message survives early startup before Megatron
    # reconfigures logging — same pattern as MoEAuxFix-install in hooks.py.
    if get_rank_safe() == 0:
        print(
            "[RouterExpertBiasPatch-install] patched "
            "finalize_model_grads._update_router_expert_bias and "
            "reset_model_temporary_tensors",
            flush=True,
        )


def remove_patch() -> None:
    """Restore Megatron's original router expert-bias helpers."""
    fmg = import_module("megatron.core.distributed.finalize_model_grads")

    if not getattr(fmg, _PATCH_SENTINEL, False):
        return

    orig = getattr(fmg, _ORIG_FN_ATTR, None)
    if orig is not None:
        fmg._update_router_expert_bias = orig
        delattr(fmg, _ORIG_FN_ATTR)

    orig_reset = getattr(fmg, _ORIG_RESET_FN_ATTR, None)
    if orig_reset is not None:
        fmg.reset_model_temporary_tensors = orig_reset
        delattr(fmg, _ORIG_RESET_FN_ATTR)

    if hasattr(fmg, _PATCH_SENTINEL):
        delattr(fmg, _PATCH_SENTINEL)

    logger.info(
        "router_expert_bias_patch.remove_patch: restored original "
        "megatron.core.distributed.finalize_model_grads router expert-bias helpers"
    )
