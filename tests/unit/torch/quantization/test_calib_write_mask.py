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

"""Tests for the ``should_process`` write-mask on the calibration algorithms."""

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.model_calib import (
    awq_clip,
    awq_lite,
    gptq,
    imatrix_calibrate,
    local_hessian_calibrate,
    max_calibrate,
    mse_calibrate,
    smoothquant,
)
from modelopt.torch.quantization.nn import TensorQuantizer

# Weight-side block quantization, the default shape for most of these algorithms.
BLOCK_CFG = [
    {"quantizer_name": "*", "enable": False},
    {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "block_sizes": {-1: 32}}},
    {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
]

# SmoothQuant migrates per-channel activation scale into the weight, so it needs a per-channel
# weight quantizer and a per-tensor input one -- the shape of INT8_SMOOTHQUANT_CFG.
SMOOTHQUANT_CFG = [
    {"quantizer_name": "*", "enable": False},
    {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
    {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
]

# imatrix only encodes weights whose backend registered an encoder; "ggml" is the one in tree.
IMATRIX_CFG = [
    {"quantizer_name": "*", "enable": False},
    {
        "quantizer_name": "*weight_quantizer",
        "cfg": {"num_bits": "iq2_xxs", "block_sizes": {-1: 256}, "backend": "ggml"},
    },
    {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
]


class _Model(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.keep = nn.Linear(d, d, bias=False)
        self.skip = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.skip(self.keep(x))


@dataclass
class _Case:
    """One calibration algorithm with the config and input width it needs."""

    algo: Callable
    quant_cfg: list = field(default_factory=lambda: BLOCK_CFG)
    width: int = 32

    @property
    def id(self):
        return self.algo.__name__

    def quantized(self):
        torch.manual_seed(0)
        model = _Model(self.width).eval()
        model = mtq.quantize(model, {"quant_cfg": self.quant_cfg, "algorithm": None}, None)
        # One forward settles each quantizer's internal `block_sizes` -> `axis` normalization,
        # which happens on first use regardless of scope -- out-of-scope quantizers keep
        # fake-quantizing by design. Snapshotting after it keeps that out of the diff, so a
        # later `axis` change is a real write rather than first-use bookkeeping.
        with torch.no_grad():
            model(torch.randn(2, 8, self.width))
        return model

    def forward_loop(self, model):
        torch.manual_seed(1)
        for _ in range(2):
            model(torch.randn(2, 8, self.width))


# Every algorithm the diff gates separately. Each is a distinct guard in the source, so each is a
# distinct claim; sharing one parametrization keeps them from drifting apart.
CASES = [
    _Case(max_calibrate),
    _Case(mse_calibrate),
    _Case(awq_lite),
    _Case(awq_clip),
    _Case(gptq),
    _Case(local_hessian_calibrate),
    _Case(smoothquant, quant_cfg=SMOOTHQUANT_CFG),
    _Case(imatrix_calibrate, quant_cfg=IMATRIX_CFG, width=256),
]

# The four that can inherit a previous stage's amax instead of re-deriving it.
HANDOFF_CASES = [
    c for c in CASES if c.algo in (mse_calibrate, local_hessian_calibrate, gptq, imatrix_calibrate)
]


def _writable_state(model):
    """Everything a calibration algorithm may write.

    Not just amax: the algorithms also move ``axis``, stash ``_amax_for_smoothing``, register
    ``pre_quant_scale`` and toggle enable state. A snapshot that misses those cannot tell a
    guarded algorithm from an unguarded one -- it is what let two rounds of leaf guards look
    complete while sites remained, so the snapshot is deliberately wider than the obvious.
    """
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            amax = getattr(module, "_amax", None)
            state[f"amax:{name}"] = None if amax is None else module.amax.clone()
            state[f"axis:{name}"] = module.axis
            state[f"enabled:{name}"] = bool(module.is_enabled)
            state[f"epqs:{name}"] = bool(getattr(module, "_enable_pre_quant_scale", False))
            afs = getattr(module, "_amax_for_smoothing", None)
            state[f"afs:{name}"] = None if afs is None else afs.detach().clone()
        if hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
            state[f"weight:{name}"] = module.weight.detach().clone()
        pqs = getattr(module, "pre_quant_scale", None)
        if pqs is not None:
            state[f"pqs:{name}"] = pqs.detach().clone()
    return state


def _under(name, prefix):
    """Same boundary ``_only`` uses, so an assertion cannot be weaker than the predicate."""
    return name == prefix or name.startswith(prefix + ".")


def _only(model, prefix):
    """Write-mask admitting one subtree, keyed on identity exactly as the real one is."""
    admitted = {
        id(m)
        for name, m in model.named_modules()
        if name == prefix or name.startswith(prefix + ".")
    }
    return lambda module: id(module) in admitted


def _input_amaxes(model):
    return {
        name: getattr(module, "_amax", None)
        for name, module in model.named_modules()
        if isinstance(module, TensorQuantizer) and name.endswith("input_quantizer")
    }


def _changed(before, after):
    return {n for n in set(before) | set(after) if not _equal(before.get(n), after.get(n))}


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_the_write_mask_confines_an_algorithm_to_its_scope(case):
    model = case.quantized()
    before = _writable_state(model)
    case.algo(model, forward_loop=case.forward_loop, should_process=_only(model, "keep"))
    after = _writable_state(model)

    changed = _changed(before, after)
    assert changed, f"{case.id} wrote nothing at all -- the test would pass vacuously"
    outside = sorted(n for n in changed if not _under(n.split(":", 1)[1], "keep"))
    assert not outside, f"{case.id} wrote outside its scope: {outside}"


def test_no_mask_means_the_whole_model():
    model = _Case(max_calibrate).quantized()
    max_calibrate(model, forward_loop=_Case(max_calibrate).forward_loop)
    calibrated = {
        n for n, v in _writable_state(model).items() if n.startswith("amax:") and v is not None
    }
    assert any("keep" in n for n in calibrated)
    assert any("skip" in n for n in calibrated)


def test_the_mask_confines_the_weight_only_path_too():
    # No forward loop routes through `weight_only_quantize`, a separate loop from the one the
    # forward-loop path uses; it needs its own guard.
    model = _Case(max_calibrate).quantized()
    before = _writable_state(model)
    max_calibrate(model, forward_loop=None, should_process=_only(model, "keep"))
    after = _writable_state(model)

    changed = _changed(before, after)
    assert changed, "weight-only calibration wrote nothing -- the test would pass vacuously"
    outside = sorted(n for n in changed if not _under(n.split(":", 1)[1], "keep"))
    assert not outside, f"weight-only calibration wrote outside its scope: {outside}"


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_the_mask_never_toggles_enable_state(case):
    """``_in_scope`` claims this model-wide, so it is pinned for every algorithm, not just max.

    Several algorithms disable and re-enable a quantizer around their own forward; the claim is
    that a scoped run leaves every quantizer's enable state as it found it.
    """
    model = case.quantized()

    def enabled():
        return {
            n: bool(m.is_enabled)
            for n, m in model.named_modules()
            if isinstance(m, TensorQuantizer)
        }

    before = enabled()
    case.algo(model, forward_loop=case.forward_loop, should_process=_only(model, "keep"))
    assert before == enabled()


@pytest.mark.parametrize("admitted", ["keep", "skip"])
def test_the_mask_survives_reparenting(admitted):
    """The claim that justifies keying on identity rather than name.

    ``layerwise_calibrate`` calls an algorithm on one decoder layer at a time, so
    ``named_modules()`` yields subtree-relative names that no full-model name would match. Here
    the predicate is built from full-model names but the algorithm runs on the submodule, which
    is exactly that situation: a name-keyed mask would admit nothing in either direction.
    """
    case = _Case(mse_calibrate)
    model = case.quantized()
    predicate = _only(model, admitted)
    # MSE refines an existing amax, so seed one first -- as the real chain does.
    max_calibrate(model, forward_loop=case.forward_loop)

    before = _writable_state(model)
    mse_calibrate(model.keep, forward_loop=None, skip_max_init=True, should_process=predicate)
    changed = _changed(before, _writable_state(model))

    if admitted == "keep":
        assert changed, "a full-model-name mask admitted nothing once the subtree was reparented"
        assert all(_under(n.split(":", 1)[1], "keep") for n in changed), changed
    else:
        assert not changed, f"wrote under a mask that admits no module in this subtree: {changed}"


def test_the_mask_keeps_local_hessian_from_hooking_out_of_scope_weights():
    """Hessian capture is a cost, not only a write: each hook allocates and accumulates H.

    Scoping only the search would still build -- and discard -- a Hessian for every weight
    outside the scope. Observable through the per-weight fallback warning, which is emitted once
    per *hooked* weight, so a warning naming an out-of-scope module means its Hessian was built.
    """
    case = _Case(local_hessian_calibrate)
    model = case.quantized()
    with pytest.warns(UserWarning) as record:
        local_hessian_calibrate(
            model, forward_loop=case.forward_loop, should_process=_only(model, "keep")
        )

    hooked = [str(w.message) for w in record if "Hessian weighting" in str(w.message)]
    assert hooked, "no per-weight warning at all -- the test would pass vacuously"
    assert not [m for m in hooked if "skip" in m], f"hooked out-of-scope weights: {hooked}"


@pytest.mark.parametrize("case", HANDOFF_CASES, ids=[c.id for c in HANDOFF_CASES])
def test_skip_max_init_elides_the_bootstrap(case):
    """``skip_max_init`` must actually drop the bootstrap ``max_calibrate``.

    Pinned via the input quantizers, which only that bootstrap calibrates: if they come back
    calibrated the flag did nothing, and a chained stage's search would have been discarded.
    """
    model = case.quantized()
    case.algo(model, forward_loop=case.forward_loop)
    assert any(v is not None for v in _input_amaxes(model).values()), (
        f"{case.id} did not seed input amax even without the flag -- test is vacuous"
    )

    model = case.quantized()
    case.algo(model, forward_loop=case.forward_loop, skip_max_init=True)
    still_none = [n for n, v in _input_amaxes(model).items() if v is None]
    assert sorted(still_none) == sorted(_input_amaxes(model)), (
        f"{case.id} re-derived amax despite skip_max_init=True"
    )


def test_skip_max_init_feeds_gptqs_solve_the_inherited_amax():
    """The retained amax must reach GPTQ's solve, not merely survive the call.

    GPTQ is the hand-off that motivates the flag: it re-derives amax unconditionally, so without
    the skip a preceding range search is discarded before the solve ever sees it. (The weight
    search in ``mse_calibrate`` / ``local_hessian_calibrate`` builds its own calibrator from the
    weight instead, so for those the flag only governs activation seeding -- covered above.)

    Two runs identical but for the amax handed in: if the quantized weights agree, the inherited
    value never reached the solve and the chain is a no-op.
    """
    case = _Case(gptq)

    def run(scale):
        model = case.quantized()
        max_calibrate(model, forward_loop=case.forward_loop)
        for module in model.modules():
            if isinstance(module, TensorQuantizer) and getattr(module, "_amax", None) is not None:
                module.amax = module.amax * scale
        gptq(model, forward_loop=case.forward_loop, skip_max_init=True)
        return model.keep.weight.detach().clone()

    assert not torch.equal(run(1.0), run(0.5)), (
        "gptq ignored the inherited amax -- skip_max_init is a no-op"
    )


def _equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        return a == b  # axis, enable state and the pre-quant-scale flag are plain scalars
    return a.shape == b.shape and torch.equal(a, b)


def _all_but_leaf(model, leaf_name):
    """Admit every module except one quantizer leaf."""
    excluded = {id(m) for name, m in model.named_modules() if name == leaf_name}
    return lambda module: id(module) not in excluded


def _leaves_a_scope_could_drop(algo):
    """The quantizer leaves each algorithm writes, and so must not write out of scope.

    AWQ migrates scale between the input and weight quantizers, so dropping either must stop it.
    """
    if algo in (smoothquant, awq_lite, awq_clip):
        # These migrate scale between the input and weight quantizers, so dropping either leaf
        # must stop them.
        return ["skip.weight_quantizer", "skip.input_quantizer"]
    return ["skip.weight_quantizer"]


LEAF_CASES = [
    (c, leaf)
    for c in CASES
    if c.algo is not max_calibrate
    for leaf in _leaves_a_scope_could_drop(c.algo)
]


@pytest.mark.parametrize(
    ("case", "leaf"), LEAF_CASES, ids=[f"{c.id}-{leaf.split('.')[-1]}" for c, leaf in LEAF_CASES]
)
def test_the_mask_excludes_a_single_quantizer_leaf_under_an_admitted_parent(case, leaf):
    """A scope may drop one leaf while keeping its parent, as a ``quantizer_name`` exclusion does.

    The unscoped run is the control: it establishes that the algorithm writes this leaf at all,
    so the scoped run's silence is evidence rather than a vacuous pass.
    """
    parent = leaf.rsplit(".", 1)[0]

    def region(model):
        """What the excluded leaf drives: its own amax, and the parent state its search writes.

        The parent's *other* quantizers stay in scope, so their amax is excluded -- an algorithm
        is entitled to write those.
        """
        state = _writable_state(model)
        own = {k: v for k, v in state.items() if k.endswith(f":{leaf}")}
        own.update({k: v for k, v in state.items() if k in (f"weight:{parent}", f"pqs:{parent}")})
        return own

    def run(scoped):
        model = case.quantized()
        before = region(model)
        case.algo(
            model,
            forward_loop=case.forward_loop,
            should_process=_all_but_leaf(model, leaf) if scoped else None,
        )
        return _changed(before, region(model))

    if not run(scoped=False):
        # The toy model does not drive this algorithm to write this leaf, so a scoped run proving
        # silence would prove nothing. The guard is still there; this test cannot witness it.
        pytest.skip(f"{case.id} writes nothing under {leaf} even unscoped")
    leaked = sorted(run(scoped=True))
    assert not leaked, f"{case.id} wrote through the excluded leaf {leaf}: {leaked}"
