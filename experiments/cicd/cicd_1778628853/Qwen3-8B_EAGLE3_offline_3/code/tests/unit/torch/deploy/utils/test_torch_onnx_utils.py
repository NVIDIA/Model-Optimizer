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

import json
from contextlib import nullcontext

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
from _test_utils.torch.deploy.lib_test_models import BaseDeployModel, get_deploy_models

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


@pytest.mark.parametrize(
    "model", deploy_benchmark_dynamo.values(), ids=deploy_benchmark_dynamo.keys()
)
def test_onnx_dynamo_export(skip_on_windows, model: BaseDeployModel):
    # try it for all potential numeric types
    for active in range(model.get.num_choices):
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
