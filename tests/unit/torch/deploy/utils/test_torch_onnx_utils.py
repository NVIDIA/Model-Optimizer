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

import copy
import json
from contextlib import nullcontext

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
from _test_utils.torch.deploy.lib_test_models import BaseDeployModel, get_deploy_models

import modelopt.torch.quantization as mtq
from modelopt.onnx.utils import get_batch_size_from_bytes, validate_batch_size
from modelopt.torch._deploy.utils import (
    OnnxBytes,
    flatten_tree,
    generate_onnx_input,
    get_onnx_bytes_and_metadata,
)
from modelopt.torch._deploy.utils.torch_onnx import _to_expected_onnx_type
from modelopt.torch.utils import standardize_model_args, unflatten_tree

deploy_benchmark_all = get_deploy_models()
deploy_benchmark_dynamo = get_deploy_models(dynamic_control_flow=False)

# `torch.onnx.export(dynamo=True)` is expensive (~1.5s/export), so the dynamo matrix is
# trimmed to representatives that cover each distinct input/output container shape plus
# the compile-failure path. Full structural and numeric-type coverage still runs in the
# (cheap) non-dynamo ``test_onnx_export_and_inputs`` below.
_DYNAMO_REPRESENTATIVE_MODELS = {
    "TensorModel",  # plain single tensor
    "ListMultiModel",  # list of tensors (arg flattening)
    "ListDictModel",  # mixed list + dict nesting
    "NestedModel",  # deeply nested inputs
    "DictMultiModel",  # dict inputs
    "ArgsKwargsModel1",  # args + kwargs (success)
    "ArgsKwargsModel2",  # args + kwargs (compile_fail path)
    "TwoOutModel",  # multiple outputs
    "NestedOutModel",  # nested outputs
}
deploy_benchmark_dynamo = {
    k: v for k, v in deploy_benchmark_dynamo.items() if k in _DYNAMO_REPRESENTATIVE_MODELS
}


def _export_fp8_model(source_dtype, weights_dtype, conv=False):
    if conv:
        model = nn.Sequential(nn.Conv2d(1, 1, 1, bias=False))
        sample_input = torch.ones(1, 1, 2, 2, dtype=source_dtype)
    else:
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        sample_input = torch.arange(4, dtype=source_dtype).reshape(1, 4)
    model = model.eval().to(source_dtype)
    if conv:
        with torch.no_grad():
            model[0].weight.fill_(1e-38)
    model = mtq.quantize(
        model,
        mtq.FP8_DEFAULT_CFG,
        forward_loop=lambda quantized_model: quantized_model(sample_input),
    )
    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model,
        (sample_input,),
        weights_dtype=weights_dtype,
        onnx_opset=23,
    )
    onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
    return onnx.load_model_from_string(onnx_bytes_obj.get_onnx_model_file_bytes())


@pytest.mark.parametrize(
    "model", deploy_benchmark_dynamo.values(), ids=deploy_benchmark_dynamo.keys()
)
def test_onnx_dynamo_export(skip_on_windows, model: BaseDeployModel):
    # One numeric type is enough here — numeric-type coverage is in test_onnx_export_and_inputs.
    for active in range(1):
        # retrieve args
        model.get.active = active
        model.get.set_default_counter()
        args = model.get_args()

        with pytest.raises(AssertionError) if model.compile_fail else nullcontext():
            onnx_bytes, _ = get_onnx_bytes_and_metadata(model, args, dynamo_export=True)
            onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
            model_bytes = onnx_bytes_obj.get_onnx_model_file_bytes()

        if model.compile_fail:
            continue

        assert model_bytes != b""
        assert onnx.load_model_from_string(model_bytes)


@pytest.mark.parametrize("model", deploy_benchmark_all.values(), ids=deploy_benchmark_all.keys())
def test_onnx_export_and_inputs(model: BaseDeployModel):
    # try it for all potential numeric types
    for active in range(model.get.num_choices):
        # retrieve args
        model.get.active = active
        model.get.set_default_counter()
        args = model.get_args()

        with pytest.raises(AssertionError) if model.compile_fail else nullcontext():
            onnx_bytes, metadata = get_onnx_bytes_and_metadata(model, args)
            onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
            onnx_bytes = onnx_bytes_obj.onnx_model[f"{onnx_bytes_obj.model_name}.onnx"]

        if model.compile_fail:
            continue

        assert onnx_bytes != b""
        assert onnx.load_model_from_string(onnx_bytes)

        # check correct naming assignment of ops by running a onnx inference session
        ort_session = ort.InferenceSession(onnx_bytes)
        ort_inputs = [inp.name for inp in ort_session.get_inputs()]

        print(ort_session)

        # NOTE: for dict inputs the order is determined by the order of the keys!
        # So if we change the order of the keys in the input this check might fail
        assert ort_inputs == model.onnx_input_names()

        # check correct naming assignments of outputs
        ort_outputs = [out.name for out in ort_session.get_outputs()]
        assert ort_outputs == model.onnx_output_names()

        # check correct output structure
        assert json.dumps(metadata["output_tree_spec"].spec) == json.dumps(model.output_spec())

        if not model.check_input_option(active):
            continue

        # run inference in ORT session with hand-generated input
        model.get.set_default_counter()
        out_ort_flat = ort_session.run(None, {k: np.asarray(model.get()) for k in ort_inputs})

        # run inference with torch model and flatten them
        out_torch = model(*standardize_model_args(model, args))
        out_torch_flat, out_torch_tree_spec = flatten_tree(out_torch)

        # making sure we have pytorch output type as expected ...
        out_torch_flat = [_to_expected_onnx_type(x) for x in out_torch_flat]

        print(out_ort_flat, out_torch_flat)

        # compare flat ORT and torch results
        assert all(
            torch.allclose(ot, torch.from_numpy(oo).to(ot))
            for ot, oo in zip(out_torch_flat, out_ort_flat)
        )

        # run inference with properly generated onnx inputs and fill data structure
        inputs_generated = generate_onnx_input(metadata, args)

        if model.invalid_device_input:
            continue

        inputs_generated = {k: v.cpu().numpy() for k, v in inputs_generated.items()}
        out_ort2 = unflatten_tree(ort_session.run(None, inputs_generated), out_torch_tree_spec)

        # now flatten both and compare
        out_ort2_flat, _ = flatten_tree(out_ort2)
        print(out_torch_flat, out_ort2_flat)
        assert all(
            torch.allclose(ot, torch.from_numpy(oo).to(ot))
            for ot, oo in zip(out_torch_flat, out_ort2_flat)
        )


@pytest.mark.parametrize(
    ("source_dtype", "weights_dtype", "expected_onnx_dtype", "conv"),
    [
        (torch.bfloat16, "bf16", onnx.TensorProto.BFLOAT16, False),
        (torch.bfloat16, "bf16", onnx.TensorProto.BFLOAT16, True),
        (torch.float32, "fp16", onnx.TensorProto.FLOAT16, False),
    ],
)
def test_fp8_export_with_supported_weights_dtype(
    source_dtype, weights_dtype, expected_onnx_dtype, conv
):
    exported_model = _export_fp8_model(source_dtype, weights_dtype, conv)

    onnx.checker.check_model(exported_model, full_check=True)
    assert not any(
        node.op_type in {"TRT_FP8QuantizeLinear", "TRT_FP8DequantizeLinear"}
        for node in exported_model.graph.node
    )
    initializer_by_name = {
        initializer.name: initializer for initializer in exported_model.graph.initializer
    }
    fp8_weight_dq_nodes = [
        node
        for node in exported_model.graph.node
        if node.op_type == "DequantizeLinear"
        and node.input[0] in initializer_by_name
        and initializer_by_name[node.input[0]].data_type == onnx.TensorProto.FLOAT8E4M3FN
    ]
    assert fp8_weight_dq_nodes
    assert all(
        initializer_by_name[node.input[1]].data_type == expected_onnx_dtype
        for node in fp8_weight_dq_nodes
    )
    if conv:
        assert all(
            set(initializer_by_name[node.input[0]].raw_data).isdisjoint({0x7F, 0xFF})
            for node in fp8_weight_dq_nodes
        )
    graph_io = [*exported_model.graph.input, *exported_model.graph.output]
    assert all(value.type.tensor_type.elem_type == expected_onnx_dtype for value in graph_io)


def test_fp8_export_rejects_bf16_conversion_from_fp32():
    with pytest.raises(
        AssertionError,
        match="Converting a quantized ONNX graph to BF16 is not supported",
    ):
        _export_fp8_model(torch.float32, "bf16")


def test_fp8_bf16_noop_rejects_incompatible_mixed_format():
    model = nn.Sequential(*(nn.Linear(128, 128, bias=False) for _ in range(2)))
    model = model.eval().to(torch.bfloat16)
    sample_input = torch.ones(1, 128, dtype=torch.bfloat16)
    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    config["quant_cfg"].extend(
        [
            {
                "quantizer_name": "1.weight_quantizer",
                "cfg": {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}},
            },
            {"quantizer_name": "1.input_quantizer", "enable": False},
        ]
    )
    model = mtq.quantize(model, config, forward_loop=lambda model: model(sample_input))
    with pytest.raises(
        AssertionError,
        match="Converting a quantized ONNX graph to BF16 is not supported",
    ):
        get_onnx_bytes_and_metadata(model, (sample_input,), weights_dtype="bf16", onnx_opset=23)


def test_fp8_bf16_noop_rejects_mixed_source_dtypes():
    class MixedDtypeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bf16_layer = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
            self.fp32_layer = nn.Linear(4, 4, bias=False)

        def forward(self, inputs):
            hidden = self.bf16_layer(inputs)
            return self.fp32_layer(hidden.float())

    model = MixedDtypeModel().eval()
    sample_input = torch.ones(1, 4, dtype=torch.bfloat16)
    model = mtq.quantize(
        model,
        mtq.FP8_DEFAULT_CFG,
        forward_loop=lambda quantized_model: quantized_model(sample_input),
    )
    with pytest.raises(
        AssertionError, match="Converting a quantized ONNX graph to BF16 is not supported"
    ):
        get_onnx_bytes_and_metadata(model, (sample_input,), weights_dtype="bf16", onnx_opset=23)


def test_fp8_bf16_noop_rejects_mixed_buffer_dtype():
    class MixedBufferModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bf16_layer = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
            self.register_buffer("fp32_offset", torch.ones(4))

        def forward(self, inputs):
            return self.bf16_layer(inputs).float() + self.fp32_offset

    model = MixedBufferModel().eval()
    sample_input = torch.ones(1, 4, dtype=torch.bfloat16)
    model = mtq.quantize(
        model,
        mtq.FP8_DEFAULT_CFG,
        forward_loop=lambda quantized_model: quantized_model(sample_input),
    )
    with pytest.raises(
        AssertionError, match="Converting a quantized ONNX graph to BF16 is not supported"
    ):
        get_onnx_bytes_and_metadata(model, (sample_input,), weights_dtype="bf16", onnx_opset=23)


class SingleArgModel(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.add(x, x) - x


class DoubleArgModel(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.add(x, y) - x


@pytest.mark.parametrize(
    ("model", "n_args", "batch_size"),
    [
        (SingleArgModel(), 1, 1),
        (SingleArgModel(), 1, 2),
        (DoubleArgModel(), 2, 1),
        (DoubleArgModel(), 2, 2),
    ],
)
def test_get_and_validate_batch_size(model, n_args, batch_size):
    inputs = (torch.randn([batch_size, 3, 32, 32]),) * n_args
    onnx_bytes, _ = get_onnx_bytes_and_metadata(model, inputs)
    onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
    onnx_bytes = onnx_bytes_obj.onnx_model[f"{onnx_bytes_obj.model_name}.onnx"]

    assert validate_batch_size(onnx_bytes, batch_size)
    assert validate_batch_size(onnx_bytes, 3) is False

    assert batch_size == get_batch_size_from_bytes(onnx_bytes)
