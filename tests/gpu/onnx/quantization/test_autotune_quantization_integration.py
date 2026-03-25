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

import os
from unittest.mock import patch

import onnx
import onnx_graphsurgeon as gs
from _test_utils.import_helper import skip_if_no_tensorrt
from _test_utils.onnx.lib_test_models import export_as_onnx
from _test_utils.onnx.quantization.autotune.models import _create_simple_resnet18_model

from modelopt.onnx.quantization.autotune.workflows import (
    init_benchmark_instance,
    region_pattern_autotuning_workflow,
)
from modelopt.onnx.quantization.quantize import _preprocess_onnx, quantize

skip_if_no_tensorrt()


def _quantized_tensor_indices(model: onnx.ModelProto) -> set[tuple[str, int]]:
    """Return (node_name, input_index) for every DQ-fed input slot in the model."""
    graph = gs.import_onnx(model)
    return {
        (node.name, inp_idx)
        for node in graph.nodes
        for inp_idx, inp in enumerate(node.inputs)
        if inp.inputs and inp.inputs[0].op == "DequantizeLinear"
    }


def _collect_q_scales(model: onnx.ModelProto) -> dict[str, float]:
    """Return {scale_initializer_name: float_value} for every QuantizeLinear node.

    Works for both float32 and float16 scale initializers (the latter produced by
    the fp16-conversion pass that runs after ORT calibration).
    """
    initializers = {init.name: init for init in model.graph.initializer}
    scales = {}
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and len(node.input) >= 2:
            scale_name = node.input[1]
            if scale_name in initializers:
                raw = onnx.numpy_helper.to_array(initializers[scale_name])
                scales[scale_name] = float(raw.flat[0])
    return scales


def test_autotune_quantization_integration(tmp_path):
    """Ensure that the quantized tensors are the same for standalone Autotune and MOQ with Autotune.

    Also ensure that the scales in the Q/DQ nodes have been updated from standalone Autotune to MOQ with Autotune.

    Runs the autotuner once to obtain a fixed set of insertion points. The same autotuner instance is then injected
    into quantize() via patching so that both sides reflect identical placement decisions without a second TRT
    profiling run.

    Compares the set of (node_name, input_index) pairs where a DQ node feeds the input between:
    - the autotuner's own export (via export_onnx), and
    - the quantize(autotune=True) output model.
    """
    model_torch, input_tensor = _create_simple_resnet18_model()
    onnx_path = os.path.join(tmp_path, "model.onnx")
    output_path = onnx_path.replace(".onnx", ".quant.onnx")

    # Export torch model to ONNX
    export_as_onnx(model_torch, input_tensor, onnx_filename=onnx_path)

    # Load and pre-process ONNX
    onnx_path, onnx_model, *_ = _preprocess_onnx(
        onnx_path,
        use_external_data_format=False,
        output_path=output_path,
        enable_shared_constants_duplication=True,
        trt_plugins=None,
        trt_plugins_precision=None,
        override_shapes=None,  # type: ignore[arg-type]
        quantize_mode="int8",
    )

    # Run autotune once to get a determined set of placement decisions.
    init_benchmark_instance(use_trtexec=False)
    autotuner = region_pattern_autotuning_workflow(
        onnx_model,
        quant_type="int8",
        default_dq_dtype="float16",
    )

    # Autotune path: export the Q/DQ model directly and collect quantized tensor slots.
    autotune_model = onnx.load_from_string(autotuner.export_onnx(best=True))
    autotune_tensors = _quantized_tensor_indices(autotune_model)

    # MOQ + Autotune path: inject the same autotuner so placement decisions are identical,
    # then run the full quantize() pipeline and collect quantized tensor slots.
    with patch(
        "modelopt.onnx.quantization.quantize.region_pattern_autotuning_workflow",
        return_value=autotuner,
    ):
        quantize(onnx_path, autotune=True, output_path=output_path)

    # Check Q/DQ nodes placement
    moq_tensors = _quantized_tensor_indices(onnx.load(output_path))
    assert autotune_tensors == moq_tensors

    # Check Q/DQ scales
    scales_random = _collect_q_scales(autotune_model)
    scales_calib = _collect_q_scales(onnx.load(output_path))
    assert scales_random, "Expected at least one Q scale in the standalone Autotune model"
    assert scales_calib, "Expected at least one Q scale in the MOQ + Autotune integrated model"
    assert len(scales_random.keys()) == len(scales_calib.keys()), (
        "Both models must quantize the same number of tensor"
    )
    assert all(
        v != list(scales_calib.values())[idx] for idx, v in enumerate(scales_random.values())
    ), (
        "All or some Q/DQ scales are identical between the standalone Autotune and MOQ + Autotune integrated models. "
        "The integrated quantization appears to have had no effect on scale computation."
    )
