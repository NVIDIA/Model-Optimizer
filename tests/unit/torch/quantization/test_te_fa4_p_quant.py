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

from types import SimpleNamespace

import torch

from modelopt.torch.kernels.quantization.attention import te_fa4_p_quant
from modelopt.torch.quantization.conversion import quantizer_state, restore_quantizer_state
from modelopt.torch.quantization.nn import TensorQuantizer


def test_scope_reaches_forward_calls_only_and_windows_are_translated(monkeypatch):
    """The mode is given to the FA4 forward names inside a scope and to nothing else; TE's
    unbounded window sides (-1) reach FA4 as None, forward and backward."""
    monkeypatch.setattr(te_fa4_p_quant, "fa4_p_quant_formats", lambda: ("fp8_e4m3", "nvfp4"))
    seen = {}

    def record(name):
        return lambda *args, **kwargs: seen.__setitem__(name, kwargs)

    names = te_fa4_p_quant.FORWARD_NAMES + te_fa4_p_quant.BACKWARD_NAMES
    backends = SimpleNamespace(**{name: record(name) for name in names})
    assert te_fa4_p_quant.install(backends)
    wrapped = {name: getattr(backends, name) for name in names}
    assert te_fa4_p_quant.install(backends)  # a second install does not wrap twice
    assert all(getattr(backends, name) is wrapped[name] for name in names)

    with te_fa4_p_quant.p_quant_scope("nvfp4") as scope:
        backends.flash_attn_func_v4(window_size=(-1, 0))
        backends._flash_attn_fwd_v4(window_size_left=-1, window_size_right=0)
        backends._flash_attn_bwd_v4(window_size_left=-1, window_size_right=0)
    assert scope.calls == 2
    assert seen["flash_attn_func_v4"] == {"window_size": (None, 0), "p_quant": "nvfp4"}
    assert seen["_flash_attn_fwd_v4"] == {
        "window_size_left": None,
        "window_size_right": 0,
        "p_quant": "nvfp4",
    }
    assert seen["_flash_attn_bwd_v4"] == {"window_size_left": None, "window_size_right": 0}

    # Outside a scope (a teacher model, or the student after its forward returned).
    backends.flash_attn_varlen_func_v4(window_size=(4, 0))
    assert seen["flash_attn_varlen_func_v4"] == {"window_size": (4, 0)}

    # Only -1 is "unbounded": another negative side is a window of its own.
    backends.flash_attn_func_v4(window_size=(-2, 0))
    assert seen["flash_attn_func_v4"] == {"window_size": (-2, 0)}


def test_a_checkpoint_without_p_bmm_quantizer_state_restores_with_it_disabled():
    """A checkpoint saved before Megatron attention had a p_bmm_quantizer keeps unquantized P,
    even when its saved config matches the name (it configured nothing at the time)."""
    model = torch.nn.Module()
    model.k_bmm_quantizer = TensorQuantizer()
    model.p_bmm_quantizer = TensorQuantizer()  # enabled, as the saved config would leave it
    saved = {k: v for k, v in quantizer_state(model).items() if k != "p_bmm_quantizer"}

    restore_quantizer_state(model, None, {"quantizer_state": saved})

    assert model.k_bmm_quantizer.is_enabled
    assert not model.p_bmm_quantizer.is_enabled
