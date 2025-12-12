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

"""FP8 quantization exporter."""

import time

import onnx
import onnx_graphsurgeon as gs
import torch
from onnx_graphsurgeon.ir.tensor import LazyValues

from .base_exporter import ONNXQuantExporter


class FP8QuantExporter(ONNXQuantExporter):
    """Exporter for FP8 quantization."""

    @staticmethod
    def pre_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Pre-processes the ONNX model for FP8 quantization."""
        return onnx_model

    @staticmethod
    def compute_scales(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Computes the scales for the weights in the ONNX model for FP8 quantization."""
        return onnx_model

    @staticmethod
    def compress_weights(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Compresses FP32/FP16 weights to FP8 by folding QDQ nodes to DQ only.

        Even though modelopt supports FP8 onnx export, the weights are represented in fp32 + QDQ.
        The storage is therefore very bad. In this function,
        Q nodes will get removed from the weights and have only DQ nodes with those converted FP8
        weights in the output model.

        Parameters:
            onnx_model: ONNX model with FP32/FP16 weights and QDQ nodes.

        Returns:
            ONNX model with FP8 weights and only DQ nodes for weights (QDQ preserved for activations).
        """
        start_time = time.time()
        print("Replacing all (fp32 weights + fp8 QDQ) with (fp8 weights + DQ)...")

        graph = gs.import_onnx(onnx_model)
        # Fold constants is required since the scale is not constant yet.
        graph.cleanup().toposort().fold_constants().cleanup()

        for node in graph.nodes:
            if node.op == "TRT_FP8QuantizeLinear":
                # Should not remove input QDQ
                if not isinstance(node.inputs[0], gs.Constant):
                    continue

                weights = node.inputs[0]
                scale = node.inputs[1]
                torch_weights = torch.from_numpy(weights.values)
                torch_scale = torch.from_numpy(scale.values)
                quantizer_name = scale.name.rsplit("/", 1)[0]
                dq_op = node.outputs[0].outputs[0]
                assert dq_op.op == "TRT_FP8DequantizeLinear", (
                    f"QDQ does not occur in pairs. You reached {dq_op.op}"
                )

                # Replace it with Dequantize with FP8 weights. This is a WAR because numpy does not support fp8.
                numpy_weights = (
                    (torch_weights / torch_scale).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
                )
                tensor = onnx.TensorProto()
                tensor.data_type = onnx.TensorProto.FLOAT8E4M3FN
                tensor.dims.extend(numpy_weights.shape)
                tensor.raw_data = numpy_weights.tobytes()
                values = LazyValues(tensor)
                onnx_weights_fp8 = gs.Constant(quantizer_name + "/fp8_weights", values)

                node.outputs.clear()
                # DQ Op is separated out
                dq_op.inputs[0] = onnx_weights_fp8
                dq_op.op = "DequantizeLinear"
                dq_op.outputs[0].dtype = dq_op.inputs[1].dtype

        graph.cleanup().toposort()
        end_time = time.time()
        print(f"fp8 qdq replaced with only dq completed in {end_time - start_time}s.")

        return gs.export_onnx(graph)

    @staticmethod
    def post_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Post-processes the ONNX model for FP8 quantization."""
        return onnx_model
