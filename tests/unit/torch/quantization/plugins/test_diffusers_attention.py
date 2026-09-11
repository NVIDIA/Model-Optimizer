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

"""Tests for on-the-fly registration of delegated diffusion attention."""

import copy
import io
from collections import defaultdict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytest.importorskip("diffusers")
from diffusers import ModelMixin

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins.attention import register_attention_for_kv_quant

_QKV_QUANTIZER_NAMES = ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer")


def delegated_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


class DelegatedAttention(nn.Module):
    def __init__(self, hidden_size=16, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)

    def _reshape(self, hidden_states):
        batch_size, sequence_length, _ = hidden_states.shape
        return hidden_states.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _unused_attention(self, q, k, v):
        scores = torch.matmul(q, k)
        return torch.matmul(scores, v)

    def forward(self, hidden_states):
        q = self._reshape(self.q(hidden_states))
        k = self._reshape(self.k(hidden_states))
        v = self._reshape(self.v(hidden_states))
        output = delegated_attention(q=q * 0.5, k=k * 0.25, v=v)
        return self.o(output.transpose(1, 2).flatten(2))


class PositionalDelegatedAttention(DelegatedAttention):
    def forward(self, hidden_states):
        q = self._reshape(self.q(hidden_states))
        k = self._reshape(self.k(hidden_states))
        v = self._reshape(self.v(hidden_states))
        output = delegated_attention(q * 0.5, k * 0.25, v)
        return self.o(output.transpose(1, 2).flatten(2))


class RepeatedDelegatedAttention(DelegatedAttention):
    def forward(self, hidden_states):
        q = self._reshape(self.q(hidden_states))
        k = self._reshape(self.k(hidden_states))
        v = self._reshape(self.v(hidden_states))
        output = delegated_attention(q, k, v)
        output = delegated_attention(output, k, v)
        return self.o(output.transpose(1, 2).flatten(2))


class HelperWithMethodMatmulAttention(DelegatedAttention):
    def forward(self, hidden_states):
        q = self._reshape(self.q(hidden_states))
        k = self._reshape(self.k(hidden_states))
        v = self._reshape(self.v(hidden_states))
        if hidden_states.shape[0] == 0:
            scores = q.matmul(k.transpose(-2, -1))
            output = scores.matmul(v)
        else:
            output = delegated_attention(q, k, v)
        return self.o(output.transpose(1, 2).flatten(2))


class AuxiliaryHelperAttention(nn.Module):
    def forward(self, hidden_states):
        return hidden_states

    def _unused_attention(self, q, k, v):
        return delegated_attention(q, k, v)


class MethodMatmulAttention(nn.Module):
    def forward(self, q, k, v):
        scores = q.matmul(k.transpose(-2, -1))
        return scores.matmul(v)


class CustomSetupDelegatedAttention(DelegatedAttention):
    def _setup(self):
        pass

    def forward(self, hidden_states):
        q = self._reshape(self.q(hidden_states))
        k = self._reshape(self.k(hidden_states))
        v = self._reshape(self.v(hidden_states))
        output = delegated_attention(q, k, v)
        return self.o(output.transpose(1, 2).flatten(2))


class DelegatedDiffusionModel(ModelMixin):
    def __init__(self, attention_cls):
        super().__init__()
        self.attn = attention_cls()

    def forward(self, hidden_states):
        return self.attn(hidden_states)


def _get_fp8_attention_config():
    quant_config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    quant_config["quant_cfg"].append(
        {
            "quantizer_name": "*[qkv]_bmm_quantizer",
            "cfg": {"num_bits": (4, 3), "axis": None},
            "enable": True,
        }
    )
    return quant_config


@pytest.mark.parametrize(
    ("attention_cls", "expected_calls"),
    [
        (DelegatedAttention, 1),
        (PositionalDelegatedAttention, 1),
        (RepeatedDelegatedAttention, 2),
        (HelperWithMethodMatmulAttention, 1),
    ],
)
def test_quantize_registers_delegated_attention(attention_cls, expected_calls):
    model = DelegatedDiffusionModel(attention_cls)
    inputs = torch.randn(2, 4, 16)

    try:
        mtq.quantize(model, _get_fp8_attention_config(), lambda quant_model: quant_model(inputs))

        for name in _QKV_QUANTIZER_NAMES:
            quantizer = getattr(model.attn, name)
            assert quantizer.is_enabled
            assert quantizer.amax is not None

        call_counts = dict.fromkeys(_QKV_QUANTIZER_NAMES, 0)

        def make_count_hook(name):
            def count_call(_module, _args, _output):
                call_counts[name] += 1

            return count_call

        handles = [
            getattr(model.attn, name).register_forward_hook(make_count_hook(name))
            for name in _QKV_QUANTIZER_NAMES
        ]
        model(inputs)
        for handle in handles:
            handle.remove()
        assert call_counts == dict.fromkeys(_QKV_QUANTIZER_NAMES, expected_calls)
    finally:
        if attention_cls in mtq.QuantModuleRegistry:
            mtq.unregister(attention_cls)


def test_helper_registration_preserves_direct_attention_methods():
    model = DelegatedDiffusionModel(DelegatedAttention)
    inputs = torch.randn(2, 4, 16)

    try:
        mtq.quantize(model, _get_fp8_attention_config(), lambda quant_model: quant_model(inputs))
        call_counts = dict.fromkeys(_QKV_QUANTIZER_NAMES, 0)

        def make_count_hook(name):
            def count_call(_module, _args, _output):
                call_counts[name] += 1

            return count_call

        handles = [
            getattr(model.attn, name).register_forward_hook(make_count_hook(name))
            for name in _QKV_QUANTIZER_NAMES
        ]
        q, k, v = (torch.randn(1, 2, 2) for _ in range(3))
        model.attn._unused_attention(q, k, v)
        for handle in handles:
            handle.remove()
        assert call_counts == dict.fromkeys(_QKV_QUANTIZER_NAMES, 1)
    finally:
        if DelegatedAttention in mtq.QuantModuleRegistry:
            mtq.unregister(DelegatedAttention)


def test_default_fp8_config_disables_delegated_attention_quantizers():
    model = DelegatedDiffusionModel(DelegatedAttention)
    inputs = torch.randn(2, 4, 16)

    try:
        mtq.quantize(
            model, copy.deepcopy(mtq.FP8_DEFAULT_CFG), lambda quant_model: quant_model(inputs)
        )
        for name in _QKV_QUANTIZER_NAMES:
            assert not getattr(model.attn, name).is_enabled
    finally:
        if DelegatedAttention in mtq.QuantModuleRegistry:
            mtq.unregister(DelegatedAttention)


@pytest.mark.parametrize("attention_cls", [AuxiliaryHelperAttention, MethodMatmulAttention])
def test_register_attention_rejects_unsupported_patterns(attention_cls):
    try:
        assert not register_attention_for_kv_quant(attention_cls)
        assert attention_cls not in mtq.QuantModuleRegistry
    finally:
        if attention_cls in mtq.QuantModuleRegistry:
            mtq.unregister(attention_cls)


def test_discovery_skips_attention_with_custom_setup():
    model = DelegatedDiffusionModel(CustomSetupDelegatedAttention)
    inputs = torch.randn(2, 4, 16)

    try:
        mtq.quantize(model, _get_fp8_attention_config(), lambda quant_model: quant_model(inputs))
        assert not hasattr(model.attn, "q_bmm_quantizer")
    finally:
        if CustomSetupDelegatedAttention in mtq.QuantModuleRegistry:
            mtq.unregister(CustomSetupDelegatedAttention)


def test_exported_attention_matmuls_have_fp8_qdq():
    onnx = pytest.importorskip("onnx")
    from modelopt.onnx.export import FP8QuantExporter

    model = DelegatedDiffusionModel(DelegatedAttention).eval()
    inputs = torch.randn(2, 4, 16)

    try:
        mtq.quantize(model, _get_fp8_attention_config(), lambda quant_model: quant_model(inputs))
        assert model.attn.o.input_quantizer.is_enabled
        buffer = io.BytesIO()
        torch.onnx.export(model, inputs, buffer, opset_version=20, dynamo=False)
        exported_model = onnx.load_model_from_string(buffer.getvalue())
        processed_model = FP8QuantExporter.process_model(exported_model)
        onnx.checker.check_model(processed_model)

        producer_by_output = {
            output: node for node in processed_model.graph.node for output in node.output
        }
        consumers_by_input = defaultdict(list)
        for node in processed_model.graph.node:
            for tensor_name in node.input:
                consumers_by_input[tensor_name].append(node)

        def is_activation_dq(tensor_name):
            dq_node = producer_by_output.get(tensor_name)
            return (
                dq_node is not None
                and dq_node.op_type == "DequantizeLinear"
                and (q_node := producer_by_output.get(dq_node.input[0])) is not None
                and q_node.op_type == "QuantizeLinear"
            )

        def is_softmax_qdq_input(tensor_name):
            dq_node = producer_by_output[tensor_name]
            q_node = producer_by_output[dq_node.input[0]]
            source_node = producer_by_output.get(q_node.input[0])
            return source_node is not None and source_node.op_type == "Softmax"

        attention_matmuls = [
            node
            for node in processed_model.graph.node
            if node.op_type == "MatMul" and all(is_activation_dq(name) for name in node.input)
        ]
        assert len(attention_matmuls) == 2

        value_matmuls = [
            node
            for node in attention_matmuls
            if any(is_softmax_qdq_input(tensor_name) for tensor_name in node.input)
        ]
        assert len(value_matmuls) == 1

        def feeds_dequantized_matmul(q_node):
            return any(
                dq_node.op_type == "DequantizeLinear"
                and any(
                    consumer.op_type == "MatMul"
                    for consumer in consumers_by_input[dq_node.output[0]]
                )
                for dq_node in consumers_by_input[q_node.output[0]]
            )

        pending = list(value_matmuls[0].output)
        visited = set(pending)
        output_projection_qdq = False
        while pending:
            tensor_name = pending.pop()
            for node in consumers_by_input.get(tensor_name, []):
                if node.op_type == "QuantizeLinear" and feeds_dequantized_matmul(node):
                    output_projection_qdq = True
                for output_name in node.output:
                    if output_name not in visited:
                        visited.add(output_name)
                        pending.append(output_name)
        assert output_projection_qdq
    finally:
        if DelegatedAttention in mtq.QuantModuleRegistry:
            mtq.unregister(DelegatedAttention)
