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

"""The layer whose op runs and its owner, seen from the functions that the op calls."""

import functools
import gc
import weakref
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
    """An op that names its layer and the kernel it calls, with an empty registry of owners."""
    monkeypatch.setattr(vllm_layer_scope, "_owners", weakref.WeakValueDictionary())
    module = SimpleNamespace()

    def kernel(x, scale=2):
        return x * scale

    def op(x, layer_name, run=None):
        return (run or module.kernel)(x)

    module.kernel, module.op = kernel, _decorate(op)
    return SimpleNamespace(module=module, kernel=kernel, op=op)


def _scope():
    return vllm_layer_scope.current_layer(), vllm_layer_scope.current_layer_owner()


def test_patched_callee_sees_the_layer_whose_op_runs_and_its_owner(ops):
    owner = _Owner()
    vllm_layer_scope.register_layer_owner("layer.0", owner)
    calls = []

    def scaled_up(original, x, scale=2):
        calls.append((*_scope(), x, scale))
        return original(x, scale) + 1

    vllm_layer_scope.wrap_layer_op(ops.module, "op", "layer_name", decorator=_decorate)
    vllm_layer_scope.patch_function(ops.module, "kernel", scaled_up)
    assert ops.module.op(3, "layer.0") == 7
    # A layer without an owner, named by vLLM's LayerName.
    assert ops.module.op(3, SimpleNamespace(value="layer.1")) == 7
    assert ops.module.kernel(3, scale=1) == 4  # outside an op
    assert calls == [("layer.0", owner, 3, 2), ("layer.1", None, 3, 2), (None, None, 3, 1)]
    # A later registration of the name replaces the owner.
    other = _Owner()
    vllm_layer_scope.register_layer_owner("layer.0", other)
    ops.module.op(3, "layer.0")
    assert calls[-1][:2] == ("layer.0", other)
    assert _scope() == (None, None)


def test_scope_nests_and_ends_with_the_op(ops):
    owner = _Owner()
    vllm_layer_scope.register_layer_owner("layer.0", owner)
    vllm_layer_scope.wrap_layer_op(ops.module, "op", "layer_name")
    seen = []

    def record(x):
        seen.append(_scope())

    def inner(x):
        record(x)
        return ops.module.op(x, "layer.1", run=record)

    ops.module.op(1, layer_name="layer.0", run=inner)
    assert seen == [("layer.0", owner), ("layer.1", None)]

    def fail(x):
        raise ValueError(x)

    with pytest.raises(ValueError):
        ops.module.op(1, "layer.0", run=fail)
    assert _scope() == (None, None)


def test_op_wrap_goes_under_its_decorator_once(ops):
    owner = _Owner()
    vllm_layer_scope.register_layer_owner("layer.0", owner)
    for _ in range(2):
        vllm_layer_scope.wrap_layer_op(ops.module, "op", "layer_name", decorator=_decorate)
    assert ops.module.op.__wrapped__.__wrapped__ is ops.op
    # A breakable-cudagraph replay calls the op's __wrapped__, which must still set the scope.
    seen = []

    def record(x):
        seen.append(_scope())

    ops.module.op.__wrapped__(1, "layer.0", run=record)
    assert seen == [("layer.0", owner)]
    # Without the decorator, the wrap goes around the decorated op.
    module = SimpleNamespace(op=_decorate(ops.op))
    vllm_layer_scope.wrap_layer_op(module, "op", "layer_name")
    assert module.op.__wrapped__.__wrapped__ is ops.op


def test_patch_function_applies_each_wrapper_once(ops):
    def plus_one(original, *args, **kwargs):
        return original(*args, **kwargs) + 1

    def doubled(original, *args, **kwargs):
        return original(*args, **kwargs) * 2

    for _ in range(2):
        vllm_layer_scope.patch_function(ops.module, "kernel", plus_one)
    assert ops.module.kernel.__wrapped__ is ops.kernel
    assert ops.module.kernel(3) == 7
    # Another wrapper wraps the patched function; patching with either again changes nothing.
    vllm_layer_scope.patch_function(ops.module, "kernel", doubled)
    vllm_layer_scope.patch_function(ops.module, "kernel", plus_one)
    vllm_layer_scope.patch_function(ops.module, "kernel", doubled)
    assert ops.module.kernel(3) == 14


def test_owner_is_held_weakly(ops):
    owner = _Owner()
    vllm_layer_scope.register_layer_owner("layer.0", owner)
    vllm_layer_scope.wrap_layer_op(ops.module, "op", "layer_name")
    seen = []

    def run(x):
        seen.append(
            (vllm_layer_scope.current_layer(), vllm_layer_scope.current_layer_owner() is not None)
        )

    ops.module.op(1, "layer.0", run=run)
    del owner
    gc.collect()
    ops.module.op(1, "layer.0", run=run)
    assert seen == [("layer.0", True), ("layer.0", False)]
