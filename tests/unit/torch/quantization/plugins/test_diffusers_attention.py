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

import copy
import io

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ModelMixin = pytest.importorskip("diffusers").ModelMixin

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins.diffusion.diffusers import _fp8_mha_disabled

_SDPA_ALIAS = F.scaled_dot_product_attention


def _fake_attention(q, k, v):
    return torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v


def _wan_attention(q, k, v):
    if torch.onnx.is_in_onnx_export():
        return F.scaled_dot_product_attention(q, k, v)
    return _fake_attention(q, k, v)


class DelegatedAttention(nn.Module):
    def __init__(self, style="keyword"):
        super().__init__()
        self.style = style
        self.q, self.k, self.v, self.o = (nn.Linear(32, 32) for _ in range(4))

    def _qkv(self, x):
        qkv = (self.q(x), self.k(x), self.v(x))
        return tuple(tensor.view(2, 4, 2, 16).transpose(1, 2) for tensor in qkv)

    def forward(self, x):
        q, k, v = self._qkv(x)
        if self.style == "keyword":
            output = _wan_attention(q=q, k=k, v=v)
        else:
            output = _wan_attention(q, k, v)
            if self.style == "repeated":
                output = _wan_attention(output, k, v)
        return self.o(output.transpose(1, 2).flatten(2))


class AliasedAttention(DelegatedAttention):
    def forward(self, x):
        return self.o(_SDPA_ALIAS(*self._qkv(x)).transpose(1, 2).flatten(2))


class NonSDPAAttention(DelegatedAttention):
    def forward(self, x):
        return self.o(_fake_attention(*self._qkv(x)).transpose(1, 2).flatten(2))


class DelegatedModel(ModelMixin):
    def __init__(self, attention_cls=DelegatedAttention, **kwargs):
        super().__init__()
        self.attn = attention_cls(**kwargs)

    def forward(self, x):
        return self.attn(x)


@pytest.fixture(autouse=True)
def _clean_registrations():
    yield
    for cls in (DelegatedAttention, AliasedAttention, NonSDPAAttention):
        if cls in mtq.QuantModuleRegistry:
            mtq.unregister(cls)


def _quantize(model, enabled=True):
    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    attrs = {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"}
    config["quant_cfg"].extend(
        {"quantizer_name": name, "cfg": attrs, "enable": enabled}
        for name in ("*[qkv]_bmm_quantizer", "*softmax_quantizer")
    )
    inputs = torch.randn(2, 4, 32)
    mtq.quantize(model, config, lambda quant_model: quant_model(inputs))
    return inputs


def _export(attention_cls):
    onnx = pytest.importorskip("onnx")
    model = DelegatedModel(attention_cls).eval()
    inputs, buffer = _quantize(model), io.BytesIO()
    torch.onnx.export(model.half(), inputs.half(), buffer, opset_version=20, dynamo=False)
    graph = onnx.load_model_from_string(buffer.getvalue())
    onnx.checker.check_model(graph)
    return graph


def _attention_qdq(graph):
    producers = {output: node for node in graph.graph.node for output in node.output}

    def source(name):
        node = producers.get(name)
        if node is not None and node.op_type == "Cast":
            node = producers.get(node.input[0])
        if node is None or node.op_type != "TRT_FP8DequantizeLinear":
            return None
        quantize = producers.get(node.input[0])
        if quantize is None or quantize.op_type != "TRT_FP8QuantizeLinear":
            return None
        return quantize, producers.get(quantize.input[0])

    return producers, [
        (node, inputs)
        for node in graph.graph.node
        if node.op_type == "MatMul" and all(inputs := [source(name) for name in node.input])
    ]


@pytest.mark.parametrize(("style", "calls"), [("keyword", 1), ("positional", 1), ("repeated", 2)])
def test_delegated_helper_call_styles(style, calls):
    model = DelegatedModel(style=style)
    inputs = _quantize(model)
    seen = []
    with model.attn.q_bmm_quantizer.register_forward_hook(lambda *_: seen.append(None)):
        model(inputs)
    assert len(seen) == calls
    assert all(getattr(model.attn, f"{name}_bmm_quantizer").amax is not None for name in "qkv")


@pytest.mark.parametrize("attention_cls", [DelegatedAttention, AliasedAttention])
def test_export_uses_fp8_sdpa_symbolic(attention_cls):
    graph = _export(attention_cls)
    producers, attention = _attention_qdq(graph)
    qk = next(x for _, x in attention if all(source.op_type == "Mul" for _, source in x))
    assert any(producers[source.input[0]].op_type == "Transpose" for _, source in qk)
    pv, inputs = next(
        x for x in attention if any(source.op_type == "Softmax" for _, source in x[1])
    )
    softmax_q = next(q for q, source in inputs if source.op_type == "Softmax")
    scale = producers[softmax_q.input[1]].attribute[0].t.raw_data
    assert scale == torch.tensor(1 / 448, dtype=torch.float16).numpy().tobytes()
    reachable = set(pv.output)
    for node in graph.graph.node:
        if reachable.intersection(node.input):
            reachable.update(node.output)
    qdq = {node.input[0] for node in graph.graph.node if node.op_type == "TRT_FP8QuantizeLinear"}
    assert reachable & qdq


def test_export_fails_when_helper_does_not_reach_sdpa():
    helper = _fake_attention
    with pytest.raises(RuntimeError, match="did not reach SDPA"):
        _export(NonSDPAAttention)
    assert _fake_attention is helper and F.scaled_dot_product_attention is _SDPA_ALIAS


@pytest.mark.parametrize("case", ["eligible", "fp32", "int8", "misaligned", "explicit"])
def test_fp8_mha_eligibility(case):
    model = DelegatedModel()
    _quantize(model)
    model.attn.q_bmm_quantizer.num_bits = 8 if case == "int8" else (4, 3)
    if case == "explicit":
        model.attn._disable_fp8_mha = True
    dtype = torch.float32 if case == "fp32" else torch.float16
    head_dim = 15 if case == "misaligned" else 16
    qkv = (torch.randn(2, 2, 4, head_dim, dtype=dtype) for _ in range(3))
    assert _fp8_mha_disabled(model.attn, *qkv) is (case != "eligible")


@pytest.mark.parametrize(("model_mixin", "enabled"), [(True, True), (True, False), (False, True)])
def test_registration_and_enablement_guards(model_mixin, enabled):
    model = DelegatedModel() if model_mixin else nn.Sequential(DelegatedAttention())
    _quantize(model, enabled)
    attention = model.attn if model_mixin else model[0]
    assert bool(getattr(attention, "_auto_fp8_mha", False)) is model_mixin
    if model_mixin:
        assert attention.q_bmm_quantizer.is_enabled is enabled
