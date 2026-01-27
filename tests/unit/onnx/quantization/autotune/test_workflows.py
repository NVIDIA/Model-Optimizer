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
import sys
import tempfile
from pathlib import Path

import onnx
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _test_utils.import_helper import skip_if_no_tensorrt, skip_if_no_trtexec
from unit.onnx.quantization.autotune.test_autotuner import create_simple_conv_model

from modelopt.onnx.quantization.autotune.workflows import (
    init_benchmark_instance,
    region_pattern_autotuning_workflow,
)


@pytest.mark.parametrize("use_trtexec", [True, False])
def test_export_quantized_model(use_trtexec):
    """Test exporting quantized model with Q/DQ."""
    if use_trtexec:
        skip_if_no_trtexec()
    else:
        skip_if_no_tensorrt()

    model = create_simple_conv_model()

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        baseline_model_path = f.name

    # Save baseline model
    onnx.save(model, baseline_model_path)

    output_dir = baseline_model_path.strip(".onnx")
    output_path = output_dir + ".quant.onnx"

    try:
        init_benchmark_instance(use_trtexec=False)
        autotuner = region_pattern_autotuning_workflow(baseline_model_path, Path(output_dir))

        # Export model with Q/DQ insertion
        autotuner.export_onnx(output_path, insert_qdq=True)

        # Verify file was created
        assert os.path.exists(output_path)

        # Verify it's a valid ONNX model
        exported_model = onnx.load(output_path)
        assert exported_model is not None

        # Verify that it contains Q/DQ nodes
        qdq_nodes = [
            n
            for n in exported_model.graph.node
            if n.op_type in ["QuantizeLinear", "DequantizeLinear"]
        ]
        assert qdq_nodes, "Q/DQ nodes not found in quantized model"

        print("✓ QDQAutotuner export quantized model")
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)
