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

"""Runtime patches for ``megatron.bridge.models.conversion.model_bridge``.

Replaces the previous approach of bind-mounting a full copy of the upstream
``model_bridge.py`` over ``/opt/Megatron-Bridge/...`` with two narrow,
idempotent monkey-patches.

Patches
-------

Patch A — ``_megatron_local_name_to_global`` (module-level helper):

    Upstream resolves ``layer_module`` only when PP > 1 and computes
    ``num_experts_per_rank = config.num_moe_experts // ep_size``. For
    Puzzletron heterogeneous students the per-layer expert count may differ
    from ``config.num_moe_experts``, so the global division gives the wrong
    rank-local expert numbering and EP weights end up swapped/dropped.

    The patched version always resolves ``layer_module`` when ``"layers."``
    appears in the parameter name (the EP branch needs it regardless of PP)
    and reads ``layer_module.mlp.num_local_experts`` directly.

Patch B — ``MegatronModelBridge.load_weights_hf_to_megatron`` (post-load
diagnostic):

    Lists every Megatron parameter that no conversion task touched. Untouched
    params are left at their default ``init_method`` and silently NaN at
    iter 1 for heterogeneous students (e.g. unloaded Mamba ``mixer.A_log``,
    ``dt_bias``, ``conv1d``, ``in_proj``).

    To capture the conversion-task list reliably, the wrapper temporarily
    replaces ``self.build_conversion_tasks`` for the duration of the call so
    the *upstream* method invokes it under its own ``hide_teacher_model()``
    context — calling ``build_conversion_tasks`` again outside that context
    would incorrectly include ModelOpt's frozen ``_teacher_model.*`` shadow
    tree.

    The diagnostic skips ``_teacher_model.*`` for the same reason: the
    teacher is loaded by a separate bridge call.

    Behaviour controlled by the env var ``STRICT_HF_TO_MEGATRON_LOAD``:

    * ``raise`` — raise ``RuntimeError`` listing untouched params
    * ``log`` — print a rank-0 warning (default)
    * ``off`` — disable the diagnostic

Usage
-----

Call :func:`apply_patch` once during process startup, before any
``AutoBridge.from_hf_pretrained`` / ``bridge.to_megatron_provider`` call.
Subsequent calls are no-ops.
"""

from __future__ import annotations

import logging
import os
import re

from megatron.bridge.models.conversion import model_bridge as _mb
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name
from megatron.bridge.utils.common_utils import print_rank_0
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_size

logger = logging.getLogger(__name__)

_PATCH_SENTINEL = "_puzzletron_model_bridge_patch_applied"
_STRICT_MODES = {"raise", "log", "off"}


# ---------------------------------------------------------------------------
# Patch A: heterogeneous-aware _megatron_local_name_to_global
# ---------------------------------------------------------------------------
def _megatron_local_name_to_global_hetero(
    models,
    config,
    param_name: str,
    vp_stage: int | None = None,
) -> str:
    """Heterogeneous-aware replacement for the upstream helper.

    Differences from upstream:
      * Resolves ``layer_module`` whenever ``"layers."`` is in the param name,
        regardless of PP size, so the EP branch can read the per-layer expert
        count from ``layer_module.mlp.num_local_experts``.
      * Uses the layer's local expert count (which may differ per layer for
        Puzzletron heterogeneous students) instead of the global
        ``config.num_moe_experts // ep_size``.
    """
    layer_module = None

    pp_group = parallel_state.get_pipeline_model_parallel_group()
    if "layers." in param_name:
        match = re.match(r"^(.+?\.layers\.\d+)", param_name)
        assert match is not None
        layer_prefix = match.group(1)
        _, layer_module = get_module_and_param_from_name(
            models=models, param_name=layer_prefix, vp_stage=vp_stage
        )

        if get_pg_size(pp_group) > 1:
            local_layer_number = int(param_name.split("layers.")[1].split(".")[0])
            if isinstance(layer_module, MegatronModule):
                global_layer_number = layer_module.layer_number - 1
                param_name = param_name.replace(
                    f"layers.{local_layer_number}.",
                    f"layers.{global_layer_number}.",
                )

    ep_group = parallel_state.get_expert_model_parallel_group()
    if (
        ".mlp.experts.linear_fc" in param_name
        and get_pg_size(ep_group) > 1
        and ".adapter." not in param_name
    ):
        assert layer_module is not None, (
            "layer_module is not found. MoE expert params should live under a layer."
        )
        num_experts_per_rank = layer_module.mlp.num_local_experts

        def _update_expert_number(name: str, param_type: str) -> str:
            local_expert_number = int(name.split(f".{param_type}")[-1])
            global_expert_number = num_experts_per_rank * ep_group.rank() + local_expert_number
            return name.replace(
                f".{param_type}{local_expert_number}",
                f".{param_type}{global_expert_number}",
            )

        if ".weight" in param_name:
            param_name = _update_expert_number(param_name, "weight")
        elif ".bias" in param_name:
            param_name = _update_expert_number(param_name, "bias")

    return param_name


# ---------------------------------------------------------------------------
# Patch B: post-load strict diagnostic on load_weights_hf_to_megatron
# ---------------------------------------------------------------------------
def _build_strict_load_wrapper(orig_load):
    """Return a wrapper around ``MegatronModelBridge.load_weights_hf_to_megatron``.

    The wrapper temporarily intercepts ``self.build_conversion_tasks`` so it
    can capture the task list the upstream method actually used (which runs
    under ``hide_teacher_model()``). After the load returns, it walks the
    Megatron model and reports any parameter whose ``id()`` is not in the
    captured task set, excluding ModelOpt's ``_teacher_model.*`` subtree.
    """

    def load_weights_hf_to_megatron(
        self,
        hf_pretrained,
        megatron_model,
        allowed_mismatched_params=None,
    ):
        captured_tasks: list = []
        original_bct = type(self).build_conversion_tasks  # unbound; bound via self

        def _capturing_bct(self_, *args, **kwargs):
            tasks = original_bct(self_, *args, **kwargs)
            if not captured_tasks:
                captured_tasks.extend(tasks)
            return tasks

        # Bind on the instance so the upstream method's ``self.build_conversion_tasks``
        # resolves to our capturing wrapper, while leaving the class method intact.
        self.build_conversion_tasks = _capturing_bct.__get__(self, type(self))
        try:
            result = orig_load(self, hf_pretrained, megatron_model, allowed_mismatched_params)
        finally:
            del self.build_conversion_tasks  # restore class-level method

        mode = os.environ.get("STRICT_HF_TO_MEGATRON_LOAD", "log").lower()
        if mode not in _STRICT_MODES:
            logger.warning("Unknown STRICT_HF_TO_MEGATRON_LOAD=%r; falling back to 'log'.", mode)
            mode = "log"
        if mode == "off" or not captured_tasks:
            return result

        loaded_param_ids = {
            id(t.param_weight)
            for t in captured_tasks
            if t.megatron_module is not None and t.param_weight is not None
        }

        megatron_model_list = (
            megatron_model if isinstance(megatron_model, list) else [megatron_model]
        )

        unloaded: list[str] = []
        for vp_idx, sub_model in enumerate(megatron_model_list):
            for name, param in sub_model.named_parameters():
                if id(param) in loaded_param_ids:
                    continue
                # ModelOpt distillation: teacher params are loaded by a
                # separate bridge call, not this one.
                if "_teacher_model." in name:
                    continue
                unloaded.append(f"vp={vp_idx} {name} shape={tuple(param.shape)}")

        if unloaded:
            msg = "\n".join(
                [
                    f"HF->Megatron load left {len(unloaded)} parameter(s) untouched"
                    " (excluding mtd._teacher_model.*):",
                    *(f"  {n}" for n in unloaded),
                ]
            )
            if mode == "raise":
                raise RuntimeError(msg)
            print_rank_0(f"WARNING: {msg}")

        return result

    load_weights_hf_to_megatron.__wrapped__ = orig_load
    load_weights_hf_to_megatron.__name__ = orig_load.__name__
    load_weights_hf_to_megatron.__qualname__ = orig_load.__qualname__
    load_weights_hf_to_megatron.__doc__ = orig_load.__doc__
    return load_weights_hf_to_megatron


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def apply_patch() -> None:
    """Install Patches A and B on ``megatron.bridge.models.conversion.model_bridge``.

    Idempotent: subsequent calls are no-ops.
    """
    if getattr(_mb, _PATCH_SENTINEL, False):
        return

    _mb._megatron_local_name_to_global = _megatron_local_name_to_global_hetero
    logger.info(
        "Patched megatron.bridge.models.conversion.model_bridge."
        "_megatron_local_name_to_global (heterogeneous EP)"
    )

    base = _mb.MegatronModelBridge
    base.load_weights_hf_to_megatron = _build_strict_load_wrapper(base.load_weights_hf_to_megatron)
    logger.info(
        "Patched MegatronModelBridge.load_weights_hf_to_megatron (STRICT_HF_TO_MEGATRON_LOAD=%s)",
        os.environ.get("STRICT_HF_TO_MEGATRON_LOAD", "log"),
    )

    setattr(_mb, _PATCH_SENTINEL, True)
