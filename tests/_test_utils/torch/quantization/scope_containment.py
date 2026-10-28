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

"""Shared machinery for the scope-containment property, used by the CPU and GPU suites.

The property: for every scope the compiler accepts, running the plan must leave everything
outside that scope untouched. The CPU suite exercises the layouts that fake-quantize on CPU;
the GPU suite exercises the NVFP4 ones, which have no CPU kernel and would otherwise never
run anywhere -- `tests/unit` is a CPU-only CI job.
"""

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.algo_cfg import (
    AlgoCfgValidationError,
    compile_algo_cfg,
    stage_predicate,
    stage_targets,
)
from modelopt.torch.quantization.model_quant import _apply_calibration_plan
from modelopt.torch.quantization.nn import TensorQuantizer

# --- Configs that put real shared state on the model -------------------------------------

# The shipped configs, not hand-rolled equivalents: these are what a user passes, and an
# invented one that merely looks the same can fail on a kernel the real one never reaches.
NVFP4_STATIC = mtq.NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG["quant_cfg"]
NVFP4_DYNAMIC = mtq.NVFP4_DEFAULT_CFG["quant_cfg"]

INT8 = [
    {"quantizer_name": "*", "enable": False},
    {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
    {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
]


# --- Models that carry each kind of reconciliation group ----------------------------------


class _MLP(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.gate_proj = nn.Linear(d, d, bias=False)
        self.up_proj = nn.Linear(d, d, bias=False)
        self.down_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class _Attn(nn.Module):
    """q/k/v are the canonical fused sibling group: one shared weight ``global_amax``."""

    def __init__(self, d=32):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class _MLA(nn.Module):
    """The pair ``_sync_mla_fused_down_proj_amax`` reconciles, by attribute name."""

    def __init__(self, d=32):
        super().__init__()
        self.linear_q_down_proj = nn.Linear(d, d, bias=False)
        self.linear_kv_down_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.linear_q_down_proj(x) + self.linear_kv_down_proj(x)


class _Experts(nn.ModuleList):
    pass


class _MoE(nn.Module):
    """Carries the hook ``max_calibrate`` calls to reconcile a layer's local experts."""

    def __init__(self, d=32, n=2, container="experts"):
        super().__init__()
        setattr(self, container, _Experts([_MLP(d) for _ in range(n)]))
        self._container = container

    def forward(self, x):
        return sum(e(x) for e in getattr(self, self._container))

    def layer_sync_moe_local_experts_amax(self, sync_weight_amax=False):
        from modelopt.torch.quantization.utils import sync_moe_expert_amax

        sync_moe_expert_amax(getattr(self, self._container), sync_weight_amax=sync_weight_amax)


#: NVFP4 fake-quant has no CPU kernel ("Only support E=4 & M=3"), so those cases need a device.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def model_with(part):
    class _Model(nn.Module):
        def __init__(self, d=32):
            super().__init__()
            self.block = part(d)

        def forward(self, x):
            return self.block(x)

    torch.manual_seed(0)
    return _Model().eval().to(DEVICE)


#: Tags of cases whose numerics have no CPU fake-quant kernel. Named rather than sniffed from
#: the config: `scale_bits` sits inside `block_sizes`, so a top-level check never fires and the
#: case reaches a kernel that raises instead of skipping.
GPU_ONLY_TAGS = ("nvfp4_static", "nvfp4_dynamic")


def needs_gpu(case_id):
    return any(tag in case_id for tag in GPU_ONLY_TAGS)


# (id, model factory, quant cfg, scope selector, scope glob)
CASES = [
    ("fused_qkv/nvfp4_static/one_sibling", _Attn, NVFP4_STATIC, "module_name", "*q_proj*"),
    (
        "fused_qkv/nvfp4_static/inputs_only",
        _Attn,
        NVFP4_STATIC,
        "quantizer_name",
        "*input_quantizer",
    ),
    ("fused_qkv/int8/one_sibling", _Attn, INT8, "module_name", "*q_proj*"),
    ("mla_pair/nvfp4_static/one_half", _MLA, NVFP4_STATIC, "module_name", "*linear_q_down_proj*"),
    ("mla_pair/nvfp4_dynamic/one_half", _MLA, NVFP4_DYNAMIC, "module_name", "*linear_q_down_proj*"),
    (
        "mla_pair/nvfp4_dynamic/weights_only",
        _MLA,
        NVFP4_DYNAMIC,
        "quantizer_name",
        "*weight_quantizer",
    ),
    ("mla_pair/int8/one_half", _MLA, INT8, "module_name", "*linear_q_down_proj*"),
    ("local_experts/int8/one_expert", _MoE, INT8, "module_name", "*experts.0*"),
    ("local_experts/int8/inputs_only", _MoE, INT8, "quantizer_name", "*input_quantizer"),
    ("local_experts/int8/elsewhere", _MoE, INT8, "module_name", "*down_proj*"),
    ("local_experts/nvfp4_static/one_expert", _MoE, NVFP4_STATIC, "module_name", "*experts.0*"),
    # Both halves of the MLA pair, but only their input role. The sync reconciles every role
    # that holds a per-tensor range, so with dynamic NVFP4 it writes the weights too.
    (
        "mla_pair/nvfp4_dynamic/inputs_only",
        _MLA,
        NVFP4_DYNAMIC,
        "quantizer_name",
        "*input_quantizer",
    ),
]


def seed_loop(model):
    """Data for the prior whole-model calibration."""
    torch.manual_seed(1)
    for _ in range(2):
        model(torch.randn(2, 8, 32, device=DEVICE))


def forward_loop(model):
    """Data for the scoped stage -- deliberately larger than the seed's.

    If both stages saw the same data, max calibration would recompute the same amax and an
    out-of-scope write would be invisible. The scale factor makes any recalibration show up,
    which is what keeps the property test from passing vacuously.
    """
    torch.manual_seed(2)
    for _ in range(2):
        model(torch.randn(2, 8, 32, device=DEVICE) * 10.0)


def make_distinguishable(model):
    """Give every quantizer a distinct amax, so a reconciliation that writes is observable.

    Without this the harness is much weaker than it looks. The seed calibration runs the same
    reconciliations the scoped stage will, so a group already holds one shared value -- and
    recomputing a max over equal values writes the value that was already there. The write
    happens, the snapshot cannot see it, and the case passes vacuously. Scaling each amax by a
    distinct factor keeps the shape and makes any cross-quantizer write change something.
    """
    for i, (_, module) in enumerate(model.named_modules()):
        if isinstance(module, TensorQuantizer) and getattr(module, "_amax", None) is not None:
            module._amax = module._amax * (1.0 + 0.05 * (i + 1))


def snapshot(model):
    """Every piece of quantizer state a calibration stage can move."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            for attr in ("_amax", "_global_amax", "_pre_quant_scale"):
                value = getattr(module, attr, None)
                state[f"{attr}:{name}"] = None if value is None else value.detach().clone()
            state[f"axis:{name}"] = module.axis
            state[f"enabled:{name}"] = bool(module.is_enabled)
        if isinstance(module, nn.Linear):
            state[f"weight:{name}"] = module.weight.detach().clone()
    return state


def changed(before, after):
    moved = []
    for key, old in before.items():
        new = after[key]
        if (old is None) != (new is None):
            moved.append(key)
        elif isinstance(old, torch.Tensor):
            if old.shape != new.shape or not torch.equal(old, new):
                moved.append(key)
        elif old != new:
            moved.append(key)
    return moved


#: Snapshot keys naming a quantizer; the rest name the module that owns a weight.
#:
#: `_global_amax` is deliberately absent. It is *derived* from the member `_amax` values of a
#: fused group, so finalization recomputes it for every group on the model. That is not an
#: independent write: if no out-of-scope `_amax` moved, the recomputed value carries no new
#: information. Counting it would report a leak on every scoped plan over a static-NVFP4 model.
_QUANTIZER_ATTRS = ("_amax", "_pre_quant_scale", "axis", "enabled")


def is_leak(key, admitted):
    """Whether this change happened outside the scope.

    A quantizer's state counts as in scope only when that *quantizer* was admitted, not when
    its owning module was: a scope naming one role of a module leaves the other role excluded,
    and treating the module as cover is exactly the confusion these rules keep making.
    """
    attr, name = key.split(":", 1)
    if attr not in _QUANTIZER_ATTRS and attr != "weight":
        return False
    return name not in admitted
