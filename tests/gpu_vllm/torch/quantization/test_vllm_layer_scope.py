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

"""The owner of a vLLM layer, in scope while an op that names the layer runs."""

import functools
import gc
from types import SimpleNamespace

import pytest

from modelopt.torch.quantization.plugins import vllm_layer_scope


class _Owner:
    pass


def _decorate(fn):
    """Stand-in for vLLM's ``eager_break_during_capture``."""

    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        return fn(*args, **kwargs)

    return decorated


@pytest.fixture
def ops(monkeypatch):
    """An op that names its layer, the kernel it calls, and vLLM's registry of named layers."""
    layers = {"layer.0": SimpleNamespace(), "layer.1": SimpleNamespace()}
    monkeypatch.setattr(
        vllm_layer_scope, "get_forward_context", lambda: SimpleNamespace(no_compile_layers=layers)
    )
    module = SimpleNamespace()

    def kernel(x, scale=2):
        return x * scale

    def op(x, layer_name, run=None):
        return (run or module.kernel)(x)

    module.kernel, module.op = kernel, _decorate(op)
    return SimpleNamespace(module=module, layers=layers, kernel=kernel, op=op)


def test_callee_runs_for_the_owner_of_the_layer_whose_op_runs(ops):
    owner = _Owner()
    vllm_layer_scope.register_layer_owner(ops.layers["layer.0"], owner)
    calls, outside = [], []

    def call(called_owner, original, bound):
        calls.append((called_owner, dict(bound.arguments)))
        return original(*bound.args, **bound.kwargs) + 1

    vllm_layer_scope.wrap_layer_op(ops.module, "op", "layer_name", decorator=_decorate)
    vllm_layer_scope.wrap_layer_callee(
        ops.module, "kernel", call, applies=lambda o: o.active, outside=lambda: outside.append(1)
    )
    owner.active = True
    assert ops.module.op(3, "layer.0") == 7  # bound with the default scale
    assert calls == [(owner, {"x": 3, "scale": 2})]
    owner.active = False
    assert ops.module.op(3, "layer.0") == 6
    assert ops.module.op(3, SimpleNamespace(value="layer.1")) == 6  # a layer without an owner
    assert ops.module.kernel(3) == 6  # outside an op
    assert outside == [1] and len(calls) == 1
    assert vllm_layer_scope.layer_owner() is None


def test_scope_nests_and_ends_with_the_op(ops):
    owner = _Owner()
    vllm_layer_scope.register_layer_owner(ops.layers["layer.0"], owner)
    vllm_layer_scope.wrap_layer_op(ops.module, "op", "layer_name")
    seen = []

    def record(x):
        seen.append(vllm_layer_scope.layer_owner())

    def inner(x):
        record(x)
        return ops.module.op(x, "layer.1", run=record)

    ops.module.op(1, layer_name="layer.0", run=inner)
    assert seen == [owner, None]

    def fail(x):
        raise ValueError(x)

    with pytest.raises(ValueError):
        ops.module.op(1, "layer.0", run=fail)
    assert vllm_layer_scope.layer_owner() is None


def test_op_wrap_goes_under_its_decorator_once(ops):
    owner = _Owner()
    vllm_layer_scope.register_layer_owner(ops.layers["layer.0"], owner)
    for _ in range(2):
        vllm_layer_scope.wrap_layer_op(ops.module, "op", "layer_name", decorator=_decorate)
        vllm_layer_scope.wrap_layer_callee(
            ops.module, "kernel", lambda *args: None, applies=lambda o: True
        )
    assert ops.module.op.__wrapped__.__wrapped__ is ops.op
    assert ops.module.kernel.__wrapped__ is ops.kernel
    # A breakable-cudagraph replay calls the op's __wrapped__, which must still set the scope.
    seen = []

    def record(x):
        seen.append(vllm_layer_scope.layer_owner())

    ops.module.op.__wrapped__(1, "layer.0", run=record)
    assert seen == [owner]
    # Without the decorator, the wrap goes around the decorated op.
    module = SimpleNamespace(op=_decorate(ops.op), kernel=ops.kernel)
    vllm_layer_scope.wrap_layer_op(module, "op", "layer_name")
    assert module.op.__wrapped__.__wrapped__ is ops.op


def test_owner_is_held_weakly(ops):
    owner = _Owner()
    vllm_layer_scope.register_layer_owner(ops.layers["layer.0"], owner)
    vllm_layer_scope.wrap_layer_op(ops.module, "op", "layer_name")
    seen = []

    def run(x):
        seen.append(vllm_layer_scope.layer_owner() is not None)

    ops.module.op(1, "layer.0", run=run)
    del owner
    gc.collect()
    ops.module.op(1, "layer.0", run=run)
    assert seen == [True, False]
