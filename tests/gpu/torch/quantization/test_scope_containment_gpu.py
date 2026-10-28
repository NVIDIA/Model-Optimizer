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

"""Scope containment for the NVFP4 layouts, which only fake-quantize on a device.

The same property the CPU suite checks -- an accepted plan writes nothing outside its scope --
over the cases it cannot run. `tests/unit` is a CPU-only CI job, so without this file the NVFP4
cases skip everywhere and the layouts that carry the shared `global_amax` and the MLA
per-tensor ranges go untested.
"""

import pytest
import torch
from _test_utils.torch.quantization.scope_containment import (
    CASES,
    changed,
    forward_loop,
    is_leak,
    make_distinguishable,
    model_with,
    needs_gpu,
    seed_loop,
    snapshot,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.algo_cfg import AlgoCfgValidationError, compile_algo_cfg
from modelopt.torch.quantization.model_calib import _finalize_with_shared_state, max_calibrate
from modelopt.torch.quantization.model_quant import _apply_calibration_plan
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils.shared_input import (
    SHARED_PATTERNS,
    SharedWeightGlobalAmaxState,
    iter_shared_quant_states,
)

GPU_CASES = [c for c in CASES if needs_gpu(c[0])]


@pytest.mark.parametrize(
    ("case_id", "part", "quant_cfg", "selector", "scope"),
    GPU_CASES,
    ids=[c[0] for c in GPU_CASES],
)
def test_an_accepted_plan_writes_only_inside_its_scope(case_id, part, quant_cfg, selector, scope):
    model = mtq.quantize(model_with(part), {"quant_cfg": quant_cfg, "algorithm": "max"}, seed_loop)
    config = {"algo_cfg": [{selector: scope, "cfg": ["max"]}], "algorithm": None}

    try:
        plan = compile_algo_cfg(config, model)
    except AlgoCfgValidationError:
        pytest.skip("refused at compile time; a refusal is safe by construction")

    from modelopt.torch.quantization.algo_cfg import stage_targets

    admitted = set()
    for stage in plan:
        modules, quantizers = stage_targets(model, stage)
        admitted |= modules | quantizers

    make_distinguishable(model)
    before = snapshot(model)
    _apply_calibration_plan(model, config["algo_cfg"], None, forward_loop)
    leaked = [k for k in changed(before, snapshot(model)) if is_leak(k, admitted)]

    assert not leaked, (
        f"{case_id}: an accepted plan wrote outside its scope.\n"
        f"  scope  : {selector}={scope!r}\n"
        f"  leaked : {sorted(leaked)}"
    )


@pytest.mark.parametrize(
    "cfg_name",
    ["NVFP4_DEFAULT_CFG", "NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG", "W4A8_NVFP4_FP8_CFG"],
)
def test_finalization_does_not_move_a_settled_group(cfg_name):
    """Re-running finalization must not change a group that is already reconciled.

    ``_finalize_with_shared_state`` runs over the whole model and recomputes each group's
    ``global_amax`` unconditionally, so every stage of a plan re-finalizes every group --
    including ones an earlier stage already settled. That is only safe if the recompute is a
    no-op on a settled group, which is what this pins. The MSE sweep config is here because it
    is the shape of the shipped recipe and needs Triton, so nothing else exercises it.
    """
    from _test_utils.torch.quantization.scope_containment import _Attn

    quant_cfg = getattr(mtq, cfg_name)["quant_cfg"]
    model = mtq.quantize(model_with(_Attn), {"quant_cfg": quant_cfg, "algorithm": "max"}, seed_loop)
    assert any(iter_shared_quant_states(model, SharedWeightGlobalAmaxState)), (
        f"{cfg_name} attached no shared state; the test would be vacuous"
    )

    _finalize_with_shared_state(model, list(SHARED_PATTERNS))
    settled = snapshot(model)
    _finalize_with_shared_state(model, list(SHARED_PATTERNS))

    assert not changed(settled, snapshot(model)), (
        f"{cfg_name}: re-finalizing a settled group moved state"
    )


def test_finalization_reconciles_a_group_a_stage_disturbed():
    """The other half: when a group *is* inconsistent, finalization does rewrite its members.

    This is what makes the split-group refusals load-bearing rather than theoretical -- without
    it, a green idempotency test above could be read as "finalization never writes".
    """
    from _test_utils.torch.quantization.scope_containment import _Attn

    quant_cfg = mtq.NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG["quant_cfg"]
    model = mtq.quantize(model_with(_Attn), {"quant_cfg": quant_cfg, "algorithm": "max"}, seed_loop)
    _finalize_with_shared_state(model, list(SHARED_PATTERNS))

    # Stand in for an earlier stage having searched one sibling's range.
    disturbed = model.block.q_proj.weight_quantizer
    disturbed.amax = disturbed.amax * 0.5
    before = snapshot(model)
    _finalize_with_shared_state(model, list(SHARED_PATTERNS))

    assert changed(before, snapshot(model)), (
        "finalization left a disturbed group alone; the split-group refusals would be moot"
    )
