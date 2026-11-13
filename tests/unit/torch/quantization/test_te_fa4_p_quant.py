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

from modelopt.torch.kernels.quantization.attention import te_fa4_p_quant


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


def test_nothing_is_wrapped_without_a_p_quant_capable_fa4(monkeypatch):
    monkeypatch.setattr(te_fa4_p_quant, "fa4_p_quant_formats", lambda: ())
    backends = SimpleNamespace(flash_attn_func_v4=print)
    assert not te_fa4_p_quant.install(backends)
    assert backends.flash_attn_func_v4 is print
