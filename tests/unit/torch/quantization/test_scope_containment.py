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

"""One property, measured: an accepted plan writes nothing outside its own scope.

The compile-time refusals in ``algo_cfg`` are hand-written mirrors of runtime reconciliation --
``SharedWeightGlobalAmaxState``, ``_sync_mla_fused_down_proj_amax``, ``sync_moe_expert_amax``.
A mirror has to agree with the function it mirrors about which quantizer *roles* it writes, how
it *finds* the group's members, and which *formats* take part. Checking that agreement by reading
both is what has repeatedly gone wrong.

So this checks the property those refusals exist to guarantee, rather than the refusals:

    for every scope the compiler accepts, run the plan and assert nothing outside it changed.

A scope the compiler *refuses* is skipped -- a refusal is always safe, possibly over-strict, and
over-strictness is a separate question from containment. A failure here is therefore always a
real leak, never a style disagreement, and it names the quantizer that moved.

What a pass does *not* establish: cases the compiler refuses are not exercised at all, and the
measurement is "did a value change", so it needs every quantizer to start from a distinct amax
or an idempotent write is invisible -- see `_make_distinguishable`.
"""

import pytest
import torch

from _test_utils.torch.quantization.scope_containment import (
    CASES,
    INT8,
    _Attn,
    DEVICE,
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
from modelopt.torch.quantization.algo_cfg import (
    AlgoCfgValidationError,
    compile_algo_cfg,
    stage_targets,
)
from modelopt.torch.quantization.model_quant import _apply_calibration_plan
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.mark.parametrize(
    ("case_id", "part", "quant_cfg", "selector", "scope"),
    CASES,
    ids=[c[0] for c in CASES],
)
def test_an_accepted_plan_writes_only_inside_its_scope(case_id, part, quant_cfg, selector, scope):
    """The property every compile-time group refusal exists to guarantee.

    ``algorithm=None`` so the scope's complement gets no fallback stage: anything that moves
    outside the entry's own targets is a write the mask failed to confine.
    """
    if needs_gpu(case_id) and DEVICE == "cpu":
        pytest.skip("NVFP4 fake-quant has no CPU kernel")
    # Calibrate the whole model first, the way an earlier stage would. Two reasons: it is the
    # scenario the group refusals exist for (a later stage overwriting an earlier one's work),
    # and it means every quantizer starts with a real amax -- so a difference below is an
    # overwrite, not the None-to-something of a first calibration.
    model = mtq.quantize(model_with(part), {"quant_cfg": quant_cfg, "algorithm": "max"}, seed_loop)
    config = {"algo_cfg": [{selector: scope, "cfg": ["max"]}], "algorithm": None}

    try:
        plan = compile_algo_cfg(config, model)
    except AlgoCfgValidationError:
        pytest.skip("refused at compile time; a refusal is safe by construction")

    admitted = set()
    for stage in plan:
        modules, quantizers = stage_targets(model, stage)
        admitted |= modules | quantizers

    make_distinguishable(model)
    before = snapshot(model)
    _apply_calibration_plan(model, config["algo_cfg"], None, forward_loop)
    moved = changed(before, snapshot(model))

    leaked = [k for k in moved if is_leak(k, admitted)]
    assert not leaked, (
        f"{case_id}: an accepted plan wrote outside its scope.\n"
        f"  scope    : {selector}={scope!r}\n"
        f"  admitted : {sorted(admitted)}\n"
        f"  leaked   : {sorted(leaked)}"
    )


def test_the_harness_can_see_a_leak():
    """Non-vacuity: with the mask removed, the same measurement must report a leak.

    Otherwise a green run above would mean only that the snapshot is blind.
    """
    model = mtq.quantize(model_with(_Attn), {"quant_cfg": INT8, "algorithm": "max"}, seed_loop)
    plan = compile_algo_cfg({"algo_cfg": [{"module_name": "*q_proj*", "cfg": ["max"]}]}, model)

    modules, quantizers = stage_targets(model, plan[0])
    admitted = modules | quantizers
    assert admitted, "fixture resolved no targets"

    from modelopt.torch.quantization.model_calib import max_calibrate

    before = snapshot(model)
    max_calibrate(model, forward_loop, should_process=None)  # deliberately unmasked
    moved = changed(before, snapshot(model))

    leaked = [k for k in moved if is_leak(k, admitted)]
    assert leaked, "the snapshot cannot detect an out-of-scope write; the property test is vacuous"


def test_every_case_resolves_a_partial_scope():
    """Each case must actually split something, or it tests nothing.

    A scope covering the whole model trivially cannot leak, so a case that resolves to
    everything would pass without exercising any reconciliation.
    """
    for case_id, part, quant_cfg, selector, scope in CASES:
        if needs_gpu(case_id) and DEVICE == "cpu":
            continue
        model = mtq.quantize(
            model_with(part), {"quant_cfg": quant_cfg, "algorithm": "max"}, seed_loop
        )
        try:
            plan = compile_algo_cfg({"algo_cfg": [{selector: scope, "cfg": ["max"]}]}, model)
        except AlgoCfgValidationError:
            continue  # refused: the compiler judged it a split, which is the point
        _, quantizers = stage_targets(model, plan[0])
        everything = {
            name
            for name, module in model.named_modules()
            if isinstance(module, TensorQuantizer) and module.is_enabled
        }
        assert quantizers and quantizers < everything, (
            f"{case_id}: scope covers everything ({len(quantizers)}/{len(everything)}), "
            "so it cannot witness a split"
        )
