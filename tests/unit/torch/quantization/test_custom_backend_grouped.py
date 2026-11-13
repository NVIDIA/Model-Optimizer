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

"""Tests for the grouped entrypoint of a custom quantization backend."""

import contextlib

import pytest
import torch
from torch import nn
from torch.nn.modules import module as torch_module

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import (
    TensorQuantizer,
    is_registered_quant_backend,
    register_quant_backend,
    unregister_quant_backend,
)
from modelopt.torch.quantization.nn.modules import tensor_quantizer
from modelopt.torch.quantization.nn.modules.tensor_quantizer import _try_grouped_fake_quant
from modelopt.torch.quantization.utils import (
    enable_weight_cache,
    export_torch_mode,
    weight_cache_scope,
)

# Every call of the backend, in order: the quantizer of an entrypoint call, or the list of
# quantizers of a grouped call. Pre-hooks of the tests log here too.
LOG: list = []
GROUPED: dict = {"decline": False}


def _toy_fake_quant(x, scale):
    """Round ``x`` to multiples of ``scale``, straight-through to ``x``, LSQ-like to ``scale``."""
    t = x / scale
    return ((t.round() - t).detach() + t) * scale


def _toy_entrypoint(x, quantizer):
    LOG.append(quantizer)
    return _toy_fake_quant(x, quantizer.toy_scale).to(x.dtype)


def _toy_grouped(inputs, quantizers):
    """All inputs in one computation, as a fused kernel would."""
    LOG.append(list(quantizers))
    assert all(x.is_contiguous() for x in inputs)
    if GROUPED["decline"]:
        return None
    flat_x = torch.cat([x.reshape(-1) for x in inputs])
    flat_s = torch.cat([q.toy_scale.expand_as(x).reshape(-1) for x, q in zip(inputs, quantizers)])
    flat_y = _toy_fake_quant(flat_x, flat_s)
    sizes = [x.numel() for x in inputs]
    return [y.view_as(x).to(x.dtype) for y, x in zip(flat_y.split(sizes), inputs)]


@pytest.fixture(autouse=True)
def toy_backend():
    register_quant_backend("toy", _toy_entrypoint, grouped_entrypoint=_toy_grouped)
    LOG.clear()
    GROUPED["decline"] = False
    yield
    unregister_quant_backend("toy")


SHAPES = [(4, 8), (6, 8), (2, 16)]


def _setup(seed=0):
    """Quantizers with a learned per-row scale, and their weights.

    All values are dyadic so that every result is exact, whatever the order of a reduction.
    """
    gen = torch.Generator().manual_seed(seed)
    quantizers, weights, inputs = [], [], []
    for i, (rows, cols) in enumerate(SHAPES):
        q = TensorQuantizer(QuantizerAttributeConfig(backend="toy"))
        exponents = torch.randint(-2, 1, (rows, 1), generator=gen).float()
        q.toy_scale = nn.Parameter(2.0**exponents)
        quantizers.append(q)
        # The second input is a transposed view, which forward makes contiguous.
        shape = (cols, rows) if i == 1 else (rows, cols)
        weights.append(nn.Parameter(torch.randint(-32, 33, shape, generator=gen) / 8))
        inputs.append(weights[-1].t() if i == 1 else weights[-1])
    return quantizers, weights, inputs


def _loss(outputs):
    gen = torch.Generator().manual_seed(1)
    return sum((o * (torch.randint(-4, 5, o.shape, generator=gen) / 4)).sum() for o in outputs)


def _grads(quantizers, weights, outputs):
    params = [*weights, *(q.toy_scale for q in quantizers)]
    return torch.autograd.grad(_loss(outputs), params, allow_unused=True, materialize_grads=True)


def _assert_equal(results, expected):
    assert len(results) == len(expected)
    for result, exp in zip(results, expected):
        assert torch.equal(result, exp)


def _reference(seed=0, hooks=None):
    quantizers, weights, inputs = _setup(seed)
    if hooks is not None:
        hooks(quantizers)
    outputs = [q(x) for q, x in zip(quantizers, inputs)]
    return outputs, _grads(quantizers, weights, outputs)


def test_grouped_matches_per_quantizer():
    ref_outputs, ref_grads = _reference()
    quantizers, weights, inputs = _setup()
    LOG.clear()
    outputs = _try_grouped_fake_quant(quantizers, inputs)
    assert [quantizers] == LOG  # one grouped call, no per-quantizer call
    _assert_equal(outputs, ref_outputs)
    grads = _grads(quantizers, weights, outputs)
    assert all(g.abs().sum() > 0 for g in grads)
    _assert_equal(grads, ref_grads)

    with torch.no_grad():  # the caller's grad mode
        outputs = _try_grouped_fake_quant(quantizers, inputs)
    assert not any(o.requires_grad for o in outputs)
    _assert_equal(outputs, ref_outputs)


def _log_hook(name, result=None):
    def hook(module, args):
        LOG.append(name)
        return result(args) if result is not None else None

    return hook


def _pre_hooks(quantizers):
    q0, q1, q2 = quantizers
    q0.register_forward_pre_hook(_log_hook("q0.a"))

    def gather(module, args):  # like Megatron's param all-gather: the parameter lands here
        LOG.append("q0.b")
        with torch.no_grad():
            module.toy_scale.fill_(0.5)

    q0.register_forward_pre_hook(gather)
    q1.register_forward_pre_hook(_log_hook("q1", lambda args: args[0] * 2))  # a new input
    q2.register_forward_pre_hook(_log_hook("q2", lambda args: (args[0],)))


def test_pre_hooks_run_once_in_order_before_backend():
    ref_outputs, ref_grads = _reference(hooks=_pre_hooks)
    quantizers, weights, inputs = _setup()
    _pre_hooks(quantizers)
    LOG.clear()
    outputs = _try_grouped_fake_quant(quantizers, inputs)
    assert ["q0.a", "q0.b", "q1", "q2", quantizers] == LOG
    _assert_equal(outputs, ref_outputs)
    _assert_equal(_grads(quantizers, weights, outputs), ref_grads)


def test_declined_falls_back_to_forward():
    ref_outputs, ref_grads = _reference(hooks=_pre_hooks)
    quantizers, weights, inputs = _setup()
    _pre_hooks(quantizers)
    GROUPED["decline"] = True
    LOG.clear()
    outputs = _try_grouped_fake_quant(quantizers, inputs)
    # The pre-hooks run once, then each quantizer's forward.
    assert ["q0.a", "q0.b", "q1", "q2", quantizers, *quantizers] == LOG
    _assert_equal(outputs, ref_outputs)
    _assert_equal(_grads(quantizers, weights, outputs), ref_grads)


def _empty_input(quantizers):
    quantizers[1].register_forward_pre_hook(_log_hook("q1", lambda args: args[0][:0]))


def _disable(quantizers):
    def hook(module, args):
        LOG.append("q1")
        module.disable()

    quantizers[1].register_forward_pre_hook(hook)


@pytest.mark.parametrize("hooks", [_empty_input, _disable])
def test_pre_hook_leaving_ungroupable_falls_back_to_forward(hooks):
    ref_outputs, ref_grads = _reference(hooks=hooks)
    quantizers, weights, inputs = _setup()
    hooks(quantizers)
    LOG.clear()
    outputs = _try_grouped_fake_quant(quantizers, inputs)
    # Each forward: q1 returns its empty or unquantized input without calling the backend.
    assert ["q1", quantizers[0], quantizers[2]] == LOG
    _assert_equal(outputs, ref_outputs)
    _assert_equal(_grads(quantizers, weights, outputs), ref_grads)


class _SubclassTensor(torch.Tensor):
    pass


def _rule_out(case, quantizers, inputs, stack: contextlib.ExitStack):
    """Make ``case`` rule out the grouped call; return the (possibly new) quantizers and inputs."""
    q0, q1, _ = quantizers
    if case == "forward_hook":
        q1.register_forward_hook(lambda module, args, output: None)
    elif case == "backward_hook":
        q1.register_full_backward_hook(lambda module, grad_in, grad_out: None)
    elif case == "backward_pre_hook":
        q1.register_full_backward_pre_hook(lambda module, grad_out: None)
    elif case == "kwargs_pre_hook":
        q1.register_forward_pre_hook(lambda module, args, kwargs: None, with_kwargs=True)
    elif case == "global_pre_hook":
        stack.callback(torch_module.register_module_forward_pre_hook(lambda m, a: None).remove)
    elif case == "global_hook":
        stack.callback(torch_module.register_module_forward_hook(lambda m, a, o: None).remove)
    elif case == "disabled":
        q1.disable()
    elif case == "calibrating":
        q1.enable_calib()
    elif case == "real_quant":
        q1._fake_quant = False
    elif case == "constant_amax":
        q1._use_constant_amax = True
    elif case == "mixed_backends":
        q1.backend = None
    elif case == "subclass_quantizer":
        quantizers[1] = type("ToyQuantizer", (TensorQuantizer,), {})(
            QuantizerAttributeConfig(backend="toy")
        )
        quantizers[1].toy_scale = q1.toy_scale
    elif case == "block_sizes":
        q1.block_sizes = {-1: 4}
    elif case == "pre_quant_scale":
        q1.pre_quant_scale = torch.ones(8)
    elif case == "duplicate":
        quantizers[2] = q0
        inputs[2] = inputs[0]
    elif case == "count":
        inputs.pop()
    elif case == "subclass_input":
        inputs[1] = inputs[1].detach().as_subclass(_SubclassTensor)
    elif case == "empty_input":
        inputs[1] = inputs[1][:0]
    elif case == "devices":
        inputs[1] = inputs[1].to("meta")
    elif case == "no_grouped_entrypoint":
        unregister_quant_backend("toy")
        register_quant_backend("toy", _toy_entrypoint)
    elif case == "export":
        stack.enter_context(export_torch_mode())
    return quantizers, inputs


NOT_GROUPED = [
    "forward_hook",
    "backward_hook",
    "backward_pre_hook",
    "kwargs_pre_hook",
    "global_pre_hook",
    "global_hook",
    "disabled",
    "calibrating",
    "real_quant",
    "constant_amax",
    "mixed_backends",
    "subclass_quantizer",
    "block_sizes",
    "pre_quant_scale",
    "duplicate",
    "count",
    "subclass_input",
    "empty_input",
    "devices",
    "no_grouped_entrypoint",
    "export",
]


@pytest.mark.parametrize("case", NOT_GROUPED)
def test_not_grouped(case):
    quantizers, _, inputs = _setup()
    quantizers[0].register_forward_pre_hook(_log_hook("q0"))
    with contextlib.ExitStack() as stack:
        quantizers, inputs = _rule_out(case, quantizers, inputs, stack)
        LOG.clear()
        assert _try_grouped_fake_quant(quantizers, inputs) is None
    assert LOG == []  # neither a pre-hook nor the backend ran


@pytest.mark.parametrize("case", ["compiling", "onnx_export", "tracing"])
def test_not_grouped_when_exporting(monkeypatch, case):
    target = {
        "compiling": (torch.compiler, "is_compiling"),
        "onnx_export": (torch.onnx, "is_in_onnx_export"),
        "tracing": (torch.jit, "is_tracing"),
    }[case]
    quantizers, _, inputs = _setup()
    monkeypatch.setattr(*target, lambda: True)
    assert _try_grouped_fake_quant(quantizers, inputs) is None
    assert LOG == []


def test_not_grouped_in_weight_cache_scope():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"backend": "toy"}, "enable": True},
        ],
        "algorithm": "max",
    }
    mtq.quantize(model, cfg, lambda m: m(torch.randn(2, 8)))
    quantizers = [model[0].weight_quantizer, model[1].weight_quantizer]
    weights = [model[0].weight, model[1].weight]
    for q in quantizers:  # the weight cache takes no learned scales
        q.toy_scale = nn.Parameter(torch.ones(8, 1), requires_grad=False)
    enable_weight_cache(model)
    LOG.clear()
    with weight_cache_scope(model):
        assert _try_grouped_fake_quant(quantizers, weights) is None
    assert LOG == []
    assert _try_grouped_fake_quant(quantizers, weights) is not None
    assert [quantizers] == LOG


@pytest.mark.parametrize(
    "case", ["count", "not_sequence", "not_tensor", "shape", "dtype", "device"]
)
def test_bad_outputs_raise(case):
    def bad(inputs, quantizers):
        outputs = _toy_grouped(inputs, quantizers)
        if case == "count":
            return outputs[:-1]
        if case == "not_sequence":
            return outputs[0]
        if case == "not_tensor":
            outputs[1] = outputs[1].tolist()
        elif case == "shape":
            outputs[1] = outputs[1][:1]
        elif case == "dtype":
            outputs[1] = outputs[1].double()
        elif case == "device":
            outputs[1] = outputs[1].to("meta")
        return outputs

    register_quant_backend("toy", _toy_entrypoint, grouped_entrypoint=bad)
    quantizers, _, inputs = _setup()
    with pytest.raises((TypeError, ValueError), match="grouped entrypoint of quant backend 'toy'"):
        _try_grouped_fake_quant(quantizers, inputs)


def test_registration():
    quantizers, _, inputs = _setup()
    with pytest.raises(TypeError):
        register_quant_backend("toy", _toy_entrypoint, grouped_entrypoint="not callable")
    assert _try_grouped_fake_quant(quantizers, inputs) is not None  # the old one is kept

    # Registering again without a grouped entrypoint removes it.
    with pytest.warns(UserWarning, match="Overwriting"):
        register_quant_backend("toy", _toy_entrypoint)
    assert _try_grouped_fake_quant(quantizers, inputs) is None

    register_quant_backend("toy", _toy_entrypoint, grouped_entrypoint=_toy_grouped)
    unregister_quant_backend("toy")
    assert not is_registered_quant_backend("toy")
    assert "toy" not in tensor_quantizer._QUANT_GROUPED_BACKENDS
    register_quant_backend("toy", _toy_entrypoint)
    LOG.clear()
    assert _try_grouped_fake_quant(quantizers, inputs) is None
    assert LOG == []
