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

import io

import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from _test_utils.onnx.lib_test_models import build_se_residual_model

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer

aten = torch.ops.aten


def test_quantize_graph_shares_quantizers():
    model, example = build_se_residual_model()

    graph_module = mtq.quantize_graph(
        model, mtq.INT8_DEFAULT_CFG, (example,), lambda m: m(example), log_level="WARNING"
    )

    quantizer_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_module"
        and isinstance(graph_module.get_submodule(node.target), TensorQuantizer)
    ]
    readers = [{user.target for user in node.users} for node in quantizer_nodes]
    # One quantizer feeds the residual add and the next conv, another the pooling and the SE multiply.
    assert {aten.add.Tensor, aten.conv2d.default} in readers
    assert {aten.adaptive_avg_pool2d.default, aten.mul.Tensor} in readers
    assert all(readers)
    # Eval-mode identities stay in front of the quantizer so they still apply in train mode.
    assert not any(aten.dropout.default in targets for targets in readers)
    for node in quantizer_nodes:
        quantizer = graph_module.get_submodule(node.target)
        assert quantizer.is_enabled and quantizer.amax is not None, node.target
    assert {"3.conv1.input_quantizer", "3.conv1.weight_quantizer"} <= {
        node.target for node in quantizer_nodes
    }
    torch.testing.assert_close(graph_module(example), model(example), rtol=0.1, atol=0.1)


def test_quantize_graph_attention():
    torch.manual_seed(0)
    model = torch.nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True).eval()
    example = torch.randn(4, 16, 64)

    graph_module = mtq.quantize_graph(
        model, mtq.INT8_DEFAULT_CFG, (example,), lambda m: m(example), log_level="WARNING"
    )

    # The ONNX path quantizes both inputs of both attention matmuls.
    matmul_inputs = [
        arg
        for node in graph_module.graph.nodes
        if node.target in (aten.matmul.default, aten.bmm.default)
        for arg in node.args[:2]
    ]
    assert len(matmul_inputs) == 4
    assert all(arg.op == "call_module" for arg in matmul_inputs)
    torch.testing.assert_close(graph_module(example), model(example), rtol=0.1, atol=0.1)


def test_quantize_graph_qat_save_restore():
    model, example = build_se_residual_model()
    example = example.repeat(2, 1, 1, 1)
    graph_module = mtq.quantize_graph(
        model,
        mtq.INT8_DEFAULT_CFG,
        (example,),
        lambda m: m(example),
        dynamic_batch=True,
        log_level="WARNING",
    )

    # Train mode uses batch statistics and trains the eager model's parameters in place.
    graph_module.train()
    batch_norm = model[2].bn1
    running_mean = batch_norm.running_mean.clone()
    graph_module(example).square().mean().backward()
    assert not torch.equal(batch_norm.running_mean, running_mean)
    assert all(param.grad is not None for param in model.parameters())
    graph_module.eval()
    example = example[:1]  # the batch is dynamic
    expected = graph_module(example)

    buffer = io.BytesIO()
    mto.save(graph_module, buffer)
    buffer.seek(0)
    restored_model, _ = build_se_residual_model()
    restored = mto.restore(restored_model, buffer)
    torch.testing.assert_close(restored(example), expected, rtol=0, atol=0)
