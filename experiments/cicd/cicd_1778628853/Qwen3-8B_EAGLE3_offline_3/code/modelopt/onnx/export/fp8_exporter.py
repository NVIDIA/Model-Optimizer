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

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from onnx_graphsurgeon.ir.tensor import LazyValues

from modelopt.onnx.logging_config import logger

from .base_exporter import ONNXQuantExporter

# FP8 E4M3 max representable magnitude; softmax output in [0, 1] saturates exactly at 1.0
# when using 1/448 as the Q scale (single fixed value — softmax range is data-independent).
_FP8_E4M3_MAX = 448.0
_FP8_E4M3_SOFTMAX_SCALE = 1.0 / _FP8_E4M3_MAX


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
        weights in the output model. TRT custom ops are converted to native ONNX DequantizeLinear.

        Parameters:
            onnx_model: ONNX model with FP32/FP16 weights and TRT_FP8 QDQ nodes.

        Returns:
            ONNX model with FP8 weights and native ONNX DQ nodes for weights (QDQ preserved for activations).
        """
        start_time = time.time()
        print("Replacing all (fp32 weights + fp8 QDQ) with (fp8 weights + DQ)...")

        graph = gs.import_onnx(onnx_model)
        # Fold constants is required since the scale is not constant yet.
        graph.cleanup().toposort().fold_constants().cleanup()

        n_t_folded = 0

        for node in graph.nodes:
            if node.op == "TRT_FP8QuantizeLinear":
                # Should not remove input QDQ (only process weight quantization)
                if not isinstance(node.inputs[0], gs.Constant):
                    continue

                weights = node.inputs[0]
                scale = node.inputs[1]
                torch_weights = torch.from_numpy(weights.values)
                torch_scale = torch.from_numpy(scale.values)
                quantizer_name = scale.name.rsplit("/", 1)[0]
                dq_op = node.outputs[0].outputs[0]
                if dq_op.op != "TRT_FP8DequantizeLinear":
                    raise RuntimeError(f"QDQ does not occur in pairs. You reached {dq_op.op}")

                # Pre-transpose constant weights if DQ feeds ``Transpose → MatMul`` (or
                # ``Cast → Transpose → MatMul`` after fp16 conversion) so TRT sees DQ→MatMul.
                # Control flow: scan candidates; a Cast-wrapped candidate is accepted only if it
                # leads to a Transpose; a bare Transpose whose all consumers are MatMul wins and
                # breaks the loop. Any other shape defaults `cast_to_remove` back to None and
                # continues scanning.
                transpose_to_remove = None
                cast_to_remove = None
                for candidate in list(dq_op.outputs[0].outputs):
                    if candidate.op == "Cast":
                        cast_to_remove = candidate
                        candidate = next(
                            (c for c in candidate.outputs[0].outputs if c.op == "Transpose"),
                            None,
                        )
                        if candidate is None:
                            cast_to_remove = None
                            continue
                    if candidate.op != "Transpose":
                        cast_to_remove = None
                        continue
                    t_consumers = list(candidate.outputs[0].outputs)
                    # Only fold the transpose when every downstream consumer is MatMul; otherwise
                    # non-MatMul consumers would observe the un-transposed weights.
                    if t_consumers and all(c.op == "MatMul" for c in t_consumers):
                        perm = candidate.attrs.get("perm", None)
                        torch_weights = (
                            torch_weights.permute(*perm).contiguous()
                            if perm is not None
                            else torch_weights.T.contiguous()
                        )
                        transpose_to_remove = candidate
                    else:
                        cast_to_remove = None
                    break

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
                # Convert TRT DQ to native ONNX DequantizeLinear with FP8 weights
                dq_op.inputs[0] = onnx_weights_fp8
                dq_op.op = "DequantizeLinear"
                dq_op.outputs[0].dtype = dq_op.inputs[1].dtype
                dq_op.outputs[0].shape = list(numpy_weights.shape)

                if transpose_to_remove is not None:
                    t_out = transpose_to_remove.outputs[0]
                    for consumer in list(t_out.outputs):
                        for i, inp in enumerate(consumer.inputs):
                            if inp is t_out:
                                consumer.inputs[i] = dq_op.outputs[0]
                    transpose_to_remove.outputs.clear()
                    if cast_to_remove is not None:
                        cast_to_remove.outputs.clear()
                    n_t_folded += 1

        graph.cleanup().toposort()
        end_time = time.time()
        if n_t_folded > 0:
            logger.info(f"Folded {n_t_folded} weight Transpose nodes during weight compression")
        print(f"fp8 qdq replaced with only dq completed in {end_time - start_time}s.")

        return gs.export_onnx(graph)

    @staticmethod
    def _quantize_conv_weights_to_fp8(graph: gs.Graph) -> int:
        """Add FP8 weight DequantizeLinear for Conv layers with unquantized weights.

        Conv weight quantizers are disabled during TorchScript ONNX export because the
        TRT_FP8DequantizeLinear custom op produces outputs with unknown shapes, causing
        the _convolution symbolic to fail. This method restores FP8 weight quantization
        by inserting DQ nodes in the ONNX graph, mirroring the compress_weights logic.

        For each Conv node with an unquantized constant weight:
        1. Compute per-tensor scale = max(abs(weight)) / 448.0
        2. Quantize weights to FP8E4M3FN
        3. Insert a DequantizeLinear(fp8_weights, scale) before the Conv weight input

        Args:
            graph: The onnx-graphsurgeon graph to modify in-place.

        Returns:
            Number of Conv weight DQ nodes inserted.
        """
        count = 0

        for node in list(graph.nodes):
            if node.op != "Conv":
                continue
            if len(node.inputs) < 2:
                continue

            weight_input = node.inputs[1]
            if not isinstance(weight_input, gs.Constant):
                continue

            # Skip if weight already has a DQ producer
            if any(out.op == "DequantizeLinear" for out in weight_input.outputs):
                continue

            torch_weights = torch.from_numpy(weight_input.values.copy())
            amax = torch_weights.abs().max().float()
            if amax == 0:
                continue
            scale_val = (amax / _FP8_E4M3_MAX).item()

            # Quantize weights to FP8 (WAR: numpy doesn't support fp8)
            fp8_data = (torch_weights / scale_val).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
            fp8_tensor = onnx.TensorProto()
            fp8_tensor.data_type = onnx.TensorProto.FLOAT8E4M3FN
            fp8_tensor.dims.extend(fp8_data.shape)
            fp8_tensor.raw_data = fp8_data.tobytes()
            fp8_constant = gs.Constant(
                node.name + "/weight_quantizer/fp8_weights", LazyValues(fp8_tensor)
            )

            # Scale in FP16 — DQ output type matches scale dtype, must match activation type
            scale_constant = gs.Constant(
                node.name + "/weight_quantizer/scale",
                np.array(scale_val, dtype=np.float16),
            )

            dq_output = gs.Variable(node.name + "/weight_quantizer/dq_output")
            dq_node = gs.Node(
                op="DequantizeLinear",
                name=node.name + "/weight_quantizer/DequantizeLinear",
                inputs=[fp8_constant, scale_constant],
                outputs=[dq_output],
            )
            graph.nodes.append(dq_node)
            node.inputs[1] = dq_output
            count += 1

        return count

    @staticmethod
    def _move_mul_before_qdq(graph: gs.Graph) -> int:
        """Move attention-scaling Mul(const) from after DQ to before Q for TRT MatMul fusion.

        Handles both ``DQ → Mul → MatMul`` and ``DQ → Transpose → Mul → MatMul`` (K path).
        """
        count = 0
        for mul_node in list(graph.nodes):
            if mul_node.op != "Mul":
                continue

            const_input = next(
                (i for i in mul_node.inputs if isinstance(i, gs.Constant) and i.values.size == 1),
                None,
            )
            tensor_input = next(
                (i for i in mul_node.inputs if not isinstance(i, gs.Constant)), None
            )
            if const_input is None or tensor_input is None:
                continue
            if not (isinstance(tensor_input, gs.Variable) and len(tensor_input.inputs) == 1):
                continue

            producer = tensor_input.inputs[0]
            transpose_node = producer if producer.op == "Transpose" else None
            dq_node = producer if producer.op == "DequantizeLinear" else None
            if transpose_node is not None:
                t_input = transpose_node.inputs[0]
                if (
                    isinstance(t_input, gs.Variable)
                    and len(t_input.inputs) == 1
                    and t_input.inputs[0].op == "DequantizeLinear"
                ):
                    dq_node = t_input.inputs[0]
            if dq_node is None:
                continue

            q_output = dq_node.inputs[0]
            if (
                not isinstance(q_output, gs.Variable)
                or len(q_output.inputs) != 1
                or q_output.inputs[0].op != "QuantizeLinear"
            ):
                continue
            q_node = q_output.inputs[0]
            q_input = q_node.inputs[0]
            if not isinstance(q_input, gs.Variable):
                continue

            mul_output = mul_node.outputs[0]
            mul_consumers = list(mul_output.outputs)
            # Require every consumer to be MatMul: rewiring all consumers to bypass the Mul
            # would silently drop the scale for any non-MatMul branch.
            if not mul_consumers or not all(c.op == "MatMul" for c in mul_consumers):
                continue

            new_mul_output = gs.Variable(
                q_input.name + "_scaled", dtype=q_input.dtype, shape=q_input.shape
            )
            graph.nodes.append(
                gs.Node(
                    op="Mul",
                    name=mul_node.name + "_moved",
                    inputs=[q_input, const_input],
                    outputs=[new_mul_output],
                )
            )
            q_node.inputs[0] = new_mul_output

            replacement = (
                transpose_node.outputs[0] if transpose_node is not None else dq_node.outputs[0]
            )
            for consumer in mul_consumers:
                for i, inp in enumerate(consumer.inputs):
                    if inp is mul_output:
                        consumer.inputs[i] = replacement
            mul_node.outputs.clear()
            count += 1

        graph.cleanup().toposort()
        return count

    @staticmethod
    def _move_transpose_before_qdq(graph: gs.Graph) -> int:
        """Move Transpose from ``DQ → Transpose → MatMul`` to ``Transpose → Q → DQ → MatMul`` (K path)."""
        count = 0
        for transpose_node in list(graph.nodes):
            if transpose_node.op != "Transpose":
                continue

            t_input = transpose_node.inputs[0]
            if (
                not isinstance(t_input, gs.Variable)
                or len(t_input.inputs) != 1
                or t_input.inputs[0].op != "DequantizeLinear"
            ):
                continue
            dq_node = t_input.inputs[0]

            dq_input = dq_node.inputs[0]
            if (
                not isinstance(dq_input, gs.Variable)
                or len(dq_input.inputs) != 1
                or dq_input.inputs[0].op != "QuantizeLinear"
            ):
                continue
            q_node = dq_input.inputs[0]
            q_input = q_node.inputs[0]
            if not isinstance(q_input, gs.Variable):
                continue

            t_output = transpose_node.outputs[0]
            t_consumers = list(t_output.outputs)
            # Require every consumer to be MatMul: rewiring to dq_node.outputs[0] would drop
            # the transpose for any non-MatMul branch, producing a wrong-shape tensor.
            if not t_consumers or not all(c.op == "MatMul" for c in t_consumers):
                continue

            new_t_output = gs.Variable(q_input.name + "_transposed", dtype=q_input.dtype)
            graph.nodes.append(
                gs.Node(
                    op="Transpose",
                    name=transpose_node.name + "_moved",
                    inputs=[q_input],
                    outputs=[new_t_output],
                    attrs=transpose_node.attrs,
                )
            )
            q_node.inputs[0] = new_t_output

            for consumer in t_consumers:
                for i, inp in enumerate(consumer.inputs):
                    if inp is t_output:
                        consumer.inputs[i] = dq_node.outputs[0]
            transpose_node.outputs.clear()
            count += 1

        graph.cleanup().toposort()
        return count

    @staticmethod
    def _insert_qdq_after_softmax(graph: gs.Graph) -> int:
        """Insert FP8 Q→DQ on Softmax outputs feeding MatMul (required by TRT MHA fusion).

        Softmax output is data-independently bounded to [0, 1], so we use a fixed scale
        ``_FP8_E4M3_SOFTMAX_SCALE`` (1/448) that saturates exactly at 1.0 while covering
        the full FP8 E4M3 representable range. No calibration is required. Only applied
        when every Softmax consumer is a MatMul so we do not insert quantization error
        on unrelated branches.
        """
        count = 0
        for softmax_node in list(graph.nodes):
            if softmax_node.op != "Softmax":
                continue
            softmax_output = softmax_node.outputs[0]
            consumers = list(softmax_output.outputs)
            if not consumers or not all(c.op == "MatMul" for c in consumers):
                continue
            if any(c.op == "QuantizeLinear" for c in consumers):
                continue

            # Match scale dtype to the graph's current float dtype so TRT stronglyTyped
            # sees consistent Q/DQ types with the surrounding compute.
            scale_dtype = softmax_output.dtype if softmax_output.dtype is not None else np.float32
            scale_val = np.array(_FP8_E4M3_SOFTMAX_SCALE, dtype=scale_dtype)
            scale_constant = gs.Constant(softmax_node.name + "/softmax_q_scale", scale_val)
            dq_scale_constant = gs.Constant(
                softmax_node.name + "/softmax_dq_scale", scale_val.copy()
            )

            zp_tensor = onnx.TensorProto()
            zp_tensor.data_type = onnx.TensorProto.FLOAT8E4M3FN
            zp_tensor.dims.extend([1])
            zp_tensor.raw_data = b"\x00"
            zp_constant = gs.Constant(
                softmax_node.name + "/softmax_q_zero_point", LazyValues(zp_tensor)
            )

            q_output = gs.Variable(softmax_node.name + "/q_output")
            dq_output = gs.Variable(softmax_node.name + "/dq_output", dtype=softmax_output.dtype)
            q_node = gs.Node(
                op="QuantizeLinear",
                name=softmax_node.name + "/QuantizeLinear",
                inputs=[softmax_output, scale_constant, zp_constant],
                outputs=[q_output],
                attrs={"saturate": 1},
            )
            dq_node = gs.Node(
                op="DequantizeLinear",
                name=softmax_node.name + "/DequantizeLinear",
                inputs=[q_output, dq_scale_constant],
                outputs=[dq_output],
            )
            graph.nodes.extend([q_node, dq_node])

            for consumer in consumers:
                if consumer is q_node:
                    continue
                for i, inp in enumerate(consumer.inputs):
                    if inp is softmax_output:
                        consumer.inputs[i] = dq_output
            count += 1

        graph.cleanup().toposort()
        return count

    @staticmethod
    def post_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Post-processes the ONNX model for FP8 quantization.

        Converts TRT_FP8 QDQ ops to native ONNX QuantizeLinear/DequantizeLinear,
        adds FP8 weight DQ for Conv layers whose weight quantizers were disabled during
        TorchScript export, and rewrites attention scaling / K-transpose / softmax-output
        patterns so TRT can fuse DQ into the attention MatMul kernels.

        Args:
            onnx_model: The ONNX model containing TRT_FP8 quantization nodes.

        Returns:
            The post-processed ONNX model with native ONNX quantization ops.
        """
        logger.info("Post-processing FP8 quantized model")
        graph = gs.import_onnx(onnx_model)

        # Convert TRT_FP8QuantizeLinear to native QuantizeLinear
        for node in graph.nodes:
            if node.op == "TRT_FP8QuantizeLinear":
                node.op = "QuantizeLinear"
                # Add FP8 zero_point if not present
                if len(node.inputs) == 2:
                    # Create FP8 zero point constant
                    zp_tensor = onnx.TensorProto()
                    zp_tensor.data_type = onnx.TensorProto.FLOAT8E4M3FN
                    zp_tensor.dims.extend([1])  # 1-element tensor
                    zp_tensor.raw_data = b"\x00"  # Zero in FP8
                    zp_values = LazyValues(zp_tensor)
                    zero_point = gs.Constant(node.name + "_zero_point", zp_values)
                    node.inputs.append(zero_point)
                # Add saturate attribute for FP8
                node.attrs["saturate"] = 1
                logger.debug(f"Converted {node.name} from TRT_FP8QuantizeLinear to QuantizeLinear")

        # Convert TRT_FP8DequantizeLinear to native DequantizeLinear
        for node in graph.nodes:
            if node.op == "TRT_FP8DequantizeLinear":
                node.op = "DequantizeLinear"
                logger.debug(
                    f"Converted {node.name} from TRT_FP8DequantizeLinear to DequantizeLinear"
                )

        # Add FP8 weight DQ for Conv layers that had weight quantizers disabled during export
        count = FP8QuantExporter._quantize_conv_weights_to_fp8(graph)
        if count > 0:
            logger.info(f"Inserted FP8 weight DequantizeLinear for {count} Conv nodes")

        # Attention-aware rewrites so TRT can fuse DQ into the attention MatMuls.
        n_mul = FP8QuantExporter._move_mul_before_qdq(graph)
        n_t = FP8QuantExporter._move_transpose_before_qdq(graph)
        n_sm = FP8QuantExporter._insert_qdq_after_softmax(graph)
        if n_mul or n_t or n_sm:
            logger.info(
                f"Attention QDQ rewrites: moved {n_mul} Mul, {n_t} Transpose; "
                f"inserted QDQ on {n_sm} Softmax outputs"
            )

        graph.cleanup().toposort()
        return gs.export_onnx(graph)
