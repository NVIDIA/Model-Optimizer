# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Windows-optimized LLM export script for torch→ONNX pathway.

This script extends the base torch_onnx/llm_export.py with Windows/NVFP4-specific
post-processing:
  - Overrides _trt_high_precision_dtype to "Half" on all TensorQuantizers after
    quantization (so FP4 scale tensors are FP16 instead of FP32)
  - NVFP4 surgeon: converts TRT-domain DQ nodes to native ONNX, upgrades opset to 23,
    fixes Transpose output dtypes for projection weight paths
  - Sets ir_version = 10 for compatibility
"""

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager

import onnx
import onnx_graphsurgeon as gs
import torch
from packaging.version import Version
from transformers import AutoConfig, AutoTokenizer

import modelopt
from modelopt.onnx.export import INT4QuantExporter, NVFP4QuantExporter
from modelopt.onnx.graph_surgery import replace_attention_with_gqa
from modelopt.onnx.llm_export_utils.export_utils import (
    ModelLoader,
    WrapperModelForCausalLM,
    llm_to_onnx,
)
from modelopt.onnx.llm_export_utils.quantization_utils import quantize
from modelopt.onnx.llm_export_utils.surgeon_utils import fold_fp8_qdq_to_dq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils import is_quantized_linear


def compress_int8_weights(onnx_model):
    """Compress INT8 QDQ weights: fold QuantizeLinear+DequantizeLinear into DequantizeLinear with INT8 initializers.

    Finds patterns: initializer(FP16) -> QuantizeLinear -> DequantizeLinear -> consumer
    Replaces with:  initializer(INT8) -> DequantizeLinear -> consumer
    """
    import numpy as np
    from onnx import numpy_helper

    graph = onnx_model.graph
    init_map = {i.name: i for i in graph.initializer}
    nodes_to_remove = []

    producer = {}
    for node in graph.node:
        for out in node.output:
            producer[out] = node

    for dq_node in graph.node:
        if dq_node.op_type != "DequantizeLinear":
            continue

        q_input = dq_node.input[0]
        q_node = producer.get(q_input)
        if q_node is None or q_node.op_type != "QuantizeLinear":
            continue

        weight_name = q_node.input[0]
        if weight_name not in init_map:
            continue

        weight_arr = numpy_helper.to_array(init_map[weight_name])
        scale_name = q_node.input[1]

        if scale_name in init_map:
            scale_arr = numpy_helper.to_array(init_map[scale_name])
        else:
            scale_prod = producer.get(scale_name)
            if scale_prod and scale_prod.op_type == "Constant":
                for attr in scale_prod.attribute:
                    if attr.name == "value":
                        scale_arr = numpy_helper.to_array(attr.t)
            else:
                continue

        axis = None
        for attr in q_node.attribute:
            if attr.name == "axis":
                axis = attr.i

        if axis is not None and scale_arr.ndim == 1:
            shape = [1] * weight_arr.ndim
            shape[axis] = -1
            scale_broad = scale_arr.reshape(shape)
        else:
            scale_broad = scale_arr

        quantized = np.clip(np.round(weight_arr / scale_broad), -128, 127).astype(np.int8)

        int8_name = weight_name + "_int8"
        int8_tensor = numpy_helper.from_array(quantized, int8_name)
        graph.initializer.append(int8_tensor)

        dq_node.input[0] = int8_name
        dq_node.input[1] = q_node.input[1]
        if len(dq_node.input) > 2 and len(q_node.input) > 2:
            dq_node.input[2] = q_node.input[2]
        elif len(q_node.input) > 2:
            dq_node.input.append(q_node.input[2])

        dq_has_axis = any(a.name == "axis" for a in dq_node.attribute)
        if axis is not None and not dq_has_axis:
            axis_attr = dq_node.attribute.add()
            axis_attr.name = "axis"
            axis_attr.i = axis

        nodes_to_remove.append(q_node.name)

    if nodes_to_remove:
        new_nodes = [n for n in graph.node if n.name not in nodes_to_remove]
        del graph.node[:]
        graph.node.extend(new_nodes)

    used_inputs = set()
    for n in graph.node:
        for inp in n.input:
            used_inputs.add(inp)
    new_inits = [i for i in graph.initializer if i.name in used_inputs or i.name.endswith("_int8")]
    del graph.initializer[:]
    graph.initializer.extend(new_inits)

    print(f"  Compressed {len(nodes_to_remove)} weight Q+DQ pairs to DQ with INT8 weights")

    from onnx import numpy_helper as _nh

    vi_map = {vi.name: vi for vi in graph.value_info}
    init_map = {i.name: i for i in graph.initializer}

    cast_pattern = re.compile(
        r"/model/layers\.\d+/(self_attn/o_proj|mlp/down_proj)/input_quantizer/Cast$"
    )
    mul_pattern = re.compile(
        r"/model/layers\.\d+/(self_attn/o_proj|mlp/down_proj)/input_quantizer/Mul$"
    )

    casts_to_remove = []
    for node in graph.node:
        if node.op_type == "Mul" and mul_pattern.search(node.name):
            init_name = node.input[1]
            if init_name in init_map:
                init = init_map[init_name]
                if init.data_type != onnx.TensorProto.FLOAT16:
                    arr = _nh.to_array(init).astype("float16")
                    init.CopyFrom(_nh.from_array(arr, init_name))
            for out in node.output:
                if out in vi_map:
                    vi_map[out].type.tensor_type.elem_type = onnx.TensorProto.FLOAT16

        if node.op_type == "Cast" and cast_pattern.search(node.name):
            cast_input = node.input[0]
            cast_output = node.output[0]
            for other in graph.node:
                for i, inp in enumerate(other.input):
                    if inp == cast_output:
                        other.input[i] = cast_input
            casts_to_remove.append(node.name)

    if casts_to_remove:
        new_nodes = [n for n in graph.node if n.name not in casts_to_remove]
        del graph.node[:]
        graph.node.extend(new_nodes)
        print(f"  Removed {len(casts_to_remove)} input_quantizer Cast nodes")

    qkv_qdq_pattern = re.compile(
        r"/model/layers\.\d+/(self_attn/(q_proj|k_proj|v_proj)|mlp/(gate_proj|up_proj|down_proj))/input_quantizer/"
    )
    qkv_q_nodes = {}
    qkv_dq_nodes = {}

    for node in graph.node:
        if qkv_qdq_pattern.search(node.name):
            if node.op_type == "QuantizeLinear":
                qkv_q_nodes[node.output[0]] = node
            elif node.op_type == "DequantizeLinear":
                qkv_dq_nodes[node.output[0]] = (node, node.input[0])

    qkv_nodes_to_remove = set()
    for dq_out, (dq_node, q_out) in qkv_dq_nodes.items():
        if q_out in qkv_q_nodes:
            q_node = qkv_q_nodes[q_out]
            mul_output = q_node.input[0]
            for other in graph.node:
                for i, inp in enumerate(other.input):
                    if inp == dq_out:
                        other.input[i] = mul_output
            qkv_nodes_to_remove.add(q_node.name)
            qkv_nodes_to_remove.add(dq_node.name)

    if qkv_nodes_to_remove:
        new_nodes = [n for n in graph.node if n.name not in qkv_nodes_to_remove]
        del graph.node[:]
        graph.node.extend(new_nodes)
        print(
            f"  Removed {len(qkv_nodes_to_remove)} Q/DQ nodes from "
            f"q/k/v_proj + gate/up/down_proj activations"
        )

    return onnx_model


def llm_arguments():
    """Parse the arguments for the llm export script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model_path",
        type=str,
        help="The folder of HF PyTorch model ckpt or HuggingFace model name/path (e.g., 'Qwen/Qwen3-0.6B')",
        required=False,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp8", "int4_awq", "int8_sq", "nvfp4"],
        help="The precision of onnx export",
    )

    parser.add_argument(
        "--lm_head",
        type=str,
        default="fp16",
        choices=["fp16"],
        help="The precision of lm_head. Currently only fp16 is tested and supported",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory to store the generated ONNX model",
        required=True,
    )

    parser.add_argument(
        "--onnx_path",
        type=str,
        help="Pass this option when you have existing onnx to surgeon",
        required=False,
    )
    parser.add_argument(
        "--save_original",
        action="store_true",
        default=False,
        help="Save the original ONNX from torch.onnx.export without any modification",
    )
    parser.add_argument(
        "--dataset_dir", type=str, help="The path of dataset for quantization", required=False
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="The path of config.json, in case it is not with the PyTorch or ONNX file",
        default=None,
    )
    parser.add_argument(
        "--calib_size", type=int, help="The size of calibration dataset", default=512
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Trust remote code when loading model from HuggingFace Hub",
    )
    return parser


def get_config_path(args):
    """Get config.json file path from the arguments.

    The default priority is: config_path > hf_model_path/config.json > onnx_path/../config.json
    """
    if args.config_path and os.path.exists(args.config_path):
        return args.config_path
    if args.hf_model_path:
        if os.path.isdir(args.hf_model_path):
            torch_config = os.path.join(args.hf_model_path, "config.json")
            if os.path.exists(torch_config):
                return torch_config
        else:
            try:
                config = AutoConfig.from_pretrained(
                    args.hf_model_path, trust_remote_code=args.trust_remote_code
                )
                temp_config_path = os.path.join(
                    tempfile.gettempdir(), f"config_{args.hf_model_path.replace('/', '_')}.json"
                )
                with open(temp_config_path, "w") as f:
                    json.dump(config.to_dict(), f, indent=2)
                return temp_config_path
            except Exception as e:
                print(f"Warning: Could not download config for {args.hf_model_path}: {e}")

    if args.onnx_path:
        onnx_config = os.path.join(os.path.dirname(args.onnx_path), "config.json")
        if os.path.exists(onnx_config):
            return onnx_config
    print("Warning: cannot find config.json. Please pass in --config_path.")
    return None


def _override_trt_high_precision_dtype(model, dtype_str="Half"):
    """Override _trt_high_precision_dtype on all TensorQuantizers in the model.

    For the Windows NVFP4 pathway, we set this to "Half" so that all Q/DQ scale
    tensors are exported as FP16 instead of FP32, avoiding mixed-precision casts
    in the ONNX graph.

    Args:
        model: The quantized PyTorch model.
        dtype_str: The target dtype string ("Half", "Float", "BFloat16").
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            module._trt_high_precision_dtype = dtype_str
            count += 1
    print(f"Overrode _trt_high_precision_dtype to '{dtype_str}' on {count} TensorQuantizers.")


def export_raw_llm(
    model,
    output_dir,
    dtype,
    config_path,
    hf_model_path,
    lm_head_precision="fp16",
    dataset_dir="",
    wrapper_cls=WrapperModelForCausalLM,
    extra_inputs={},
    extra_dyn_axes={},
    calib_size=512,
    trust_remote_code=False,
):
    """Export raw llm model to ONNX and perform quantization.

    Args:
        model: torch.nn.module
        output_dir: str
        dtype: str
        config_path: str
        hf_model_path: str, Used for loading tokenizer for quantization
        dataset_dir: str, Used for quantization
        wrapper_cls: class, Used for wrapping the model
        extra_inputs: dict, Used for extra inputs
        extra_dyn_axes: dict, Used for extra dynamic axes
        calib_size: int, Used for quantization calibration size
        trust_remote_code: bool, Trust remote code when loading tokenizer
    """
    os.makedirs(output_dir, exist_ok=True)

    if dtype == "fp16":
        print("Loading fp16 ONNX model...")

        llm_to_onnx(
            wrapper_cls(model), output_dir, extra_inputs=extra_inputs, extra_dyn_axes=extra_dyn_axes
        )
        shutil.copy(config_path, os.path.join(output_dir, "config.json"))

    if dtype in ["fp8", "int4_awq", "int8_sq", "nvfp4"]:
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_path, trust_remote_code=trust_remote_code
        )
        if os.path.isdir(hf_model_path):
            modelopt_state = os.path.join(hf_model_path, "modelopt_state.pth")
            model_needs_quantization = not os.path.exists(modelopt_state)
        else:
            model_needs_quantization = True

        if model_needs_quantization:
            model = quantize(
                model, tokenizer, dtype, lm_head_precision, dataset_dir, calib_size=calib_size
            )

            _override_trt_high_precision_dtype(model, "Half")

            if dtype == "nvfp4":
                for module in model.modules():
                    assert not isinstance(module, torch.nn.Linear) or is_quantized_linear(module)
                    if isinstance(module, torch.nn.Linear):
                        module.input_quantizer._trt_high_precision_dtype = "Half"
                        module.input_quantizer._onnx_quantizer_type = "dynamic"
                        module.weight_quantizer._onnx_quantizer_type = "static"

            if dtype in {"fp8", "int4_awq", "int8_sq", "nvfp4"}:
                print(f"Exporting {dtype} ONNX model from quantized PyTorch model...")
                llm_to_onnx(
                    wrapper_cls(
                        model,
                    ),
                    output_dir,
                    extra_inputs=extra_inputs,
                    extra_dyn_axes=extra_dyn_axes,
                )
                shutil.copy(config_path, os.path.join(output_dir, "config.json"))

            quantized_model_dir = f"{output_dir}_{dtype}_quantized"
            os.makedirs(quantized_model_dir, exist_ok=True)
            with torch.inference_mode():
                export_hf_checkpoint(model, dtype=torch.float16, export_dir=quantized_model_dir)

    return model.state_dict()


def surgeon_llm(
    raw_onnx_path,
    output_dir,
    dtype,
    config_path,
    hf_model_path=None,
    lm_head_precision="fp16",
    trust_remote_code=False,
):
    """Surgeon raw llm onnx to fit TRT.

    For example, insert quantization q/dq nodes.
    Includes Windows-specific NVFP4 post-processing:
      - Convert DQ nodes from TRT domain to native ONNX
      - Upgrade opset to 23 (minimum for FP4 DequantizeLinear)
      - Remove trt opset import
      - Fix Transpose output dtype for projection weight paths
      - GQA surgery: replace attention with GroupQueryAttention

    Args:
        raw_onnx_path: str
        output_dir: str
        dtype: str
        config_path: str
        hf_model_path: str, HuggingFace model ID for GQA surgery (RoPE caches, config)
        lm_head_precision: str
        trust_remote_code: bool, Trust remote code when loading HF config
    """

    t0 = time.time()
    onnx.shape_inference.infer_shapes_path(raw_onnx_path)
    graph = gs.import_onnx(onnx.load(raw_onnx_path))
    t1 = time.time()
    print(f"Importing ONNX graph takes {t1 - t0}s.")
    graph.fold_constants().cleanup().toposort()

    if dtype == "fp8" or lm_head_precision == "fp8":
        graph = fold_fp8_qdq_to_dq(graph)

    os.makedirs(output_dir, exist_ok=True)
    t2 = time.time()

    onnx_model = gs.export_onnx(graph)

    @contextmanager
    def time_operation(operation_name):
        start_time = time.time()
        yield
        end_time = time.time()
        print(f"{operation_name} takes {end_time - start_time}s.")

    if dtype == "nvfp4":
        with time_operation("quantizing weights to nvfp4"):
            onnx_model = NVFP4QuantExporter.process_model(onnx_model)

        for node in onnx_model.graph.node:
            if node.op_type == "DequantizeLinear" and node.domain != "":
                node.domain = ""

        existing_vi = {vi.name: vi for vi in onnx_model.graph.value_info}
        fp4_count = 0
        for node in onnx_model.graph.node:
            if node.op_type == "TRT_FP4DynamicQuantize":
                for output_name in node.output:
                    if output_name in existing_vi:
                        existing_vi[
                            output_name
                        ].type.tensor_type.elem_type = onnx.TensorProto.FLOAT4E2M1
                    else:
                        vi = onnx.helper.make_empty_tensor_value_info(output_name)
                        vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT4E2M1
                        onnx_model.graph.value_info.append(vi)
                    fp4_count += 1
        print(f"  Set {fp4_count} TRT_FP4DynamicQuantize output(s) -> float4e2m1")

        has_trt_nodes = any(node.domain == "trt" for node in onnx_model.graph.node)

        new_opsets = []
        for opset in onnx_model.opset_import:
            if opset.domain == "trt":
                if has_trt_nodes:
                    new_opsets.append(opset)
                continue
            if opset.domain == "":
                opset.version = max(opset.version, 23)
            new_opsets.append(opset)
        if has_trt_nodes and not any(op.domain == "trt" for op in new_opsets):
            trt_opset = onnx.OperatorSetIdProto()
            trt_opset.domain = "trt"
            trt_opset.version = 1
            new_opsets.append(trt_opset)
            print("  Added missing opset_import: domain='trt', version=1")
        del onnx_model.opset_import[:]
        onnx_model.opset_import.extend(new_opsets)

        vi_map = {vi.name: vi for vi in onnx_model.graph.value_info}
        transpose_pattern = re.compile(
            r"layers\.\d+/"
            r"(mlp/(down_proj|up_proj|gate_proj)|self_attn/(q_proj|k_proj|v_proj|o_proj))"
            r"/Transpose"
        )
        for node in onnx_model.graph.node:
            if node.op_type == "Transpose" and transpose_pattern.search(node.name):
                for out in node.output:
                    if out in vi_map:
                        vi_map[out].type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    elif dtype == "int4_awq":
        with time_operation("quantizing weights to int4"):
            onnx_model = INT4QuantExporter.process_model(onnx_model)

    elif dtype == "int8_sq":
        with time_operation("compressing INT8 weights (Q+DQ -> DQ with INT8 weights)"):
            onnx_model = compress_int8_weights(onnx_model)

    if dtype in ("int4_awq", "int8_sq"):
        for node in onnx_model.graph.node:
            if node.op_type == "DequantizeLinear" and node.domain != "":
                node.domain = ""
        new_opsets = []
        min_opset = 23 if dtype == "int4_awq" else 19
        for opset in onnx_model.opset_import:
            if opset.domain == "trt":
                continue
            if opset.domain == "":
                opset.version = max(opset.version, min_opset)
            new_opsets.append(opset)
        del onnx_model.opset_import[:]
        onnx_model.opset_import.extend(new_opsets)
        print(f"  Converted DQ nodes to native ONNX domain, opset set to {min_opset}")

        from onnx import numpy_helper as _nh

        vi_map = {vi.name: vi for vi in onnx_model.graph.value_info}
        init_map = {i.name: i for i in onnx_model.graph.initializer}
        mul_pattern = re.compile(r"/input_quantizer/Mul$")
        mul_fixed = 0
        for node in onnx_model.graph.node:
            if node.op_type == "Mul" and mul_pattern.search(node.name):
                scale_name = node.input[1]
                if scale_name in init_map:
                    init = init_map[scale_name]
                    if init.data_type != onnx.TensorProto.FLOAT16:
                        arr = _nh.to_array(init).astype("float16")
                        init.CopyFrom(_nh.from_array(arr, scale_name))
                for out in node.output:
                    if out in vi_map:
                        vi_map[out].type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
                mul_fixed += 1
        if mul_fixed:
            print(f"  Fixed {mul_fixed} input_quantizer/Mul nodes to FP16 output")

        layer_cast_pattern = re.compile(r"^/model/layers\.\d+/Cast(_\d+)?$")
        casts_removed = 0
        for node in list(onnx_model.graph.node):
            if node.op_type == "Cast" and layer_cast_pattern.match(node.name):
                cast_input = node.input[0]
                cast_output = node.output[0]
                for other in onnx_model.graph.node:
                    for i, inp in enumerate(other.input):
                        if inp == cast_output:
                            other.input[i] = cast_input
                for out in onnx_model.graph.output:
                    if out.name == cast_output:
                        out.name = cast_input
                onnx_model.graph.node.remove(node)
                casts_removed += 1
        if casts_removed:
            print(f"  Removed {casts_removed} /model/layers.*/Cast nodes")

        vi_map = {vi.name: vi for vi in onnx_model.graph.value_info}
        add_pattern = re.compile(r"^/model/layers\.\d+/Add$")
        oproj_pattern = re.compile(r"^/model/layers\.\d+/self_attn/o_proj/MatMul$")
        dtype_fixed = 0
        for node in onnx_model.graph.node:
            if (node.op_type == "Add" and add_pattern.match(node.name)) or (
                node.op_type == "MatMul" and oproj_pattern.match(node.name)
            ):
                for tensor_name in list(node.input) + list(node.output):
                    if tensor_name in vi_map:
                        vi_map[tensor_name].type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
                        dtype_fixed += 1
        if dtype_fixed:
            print(f"  Fixed {dtype_fixed} value_info entries to FP16 for Add/o_proj nodes")

        last_add1_pattern = re.compile(r"^/model/layers\.(\d+)/Add_1$")
        last_add1_node = None
        last_layer_idx = -1
        for node in onnx_model.graph.node:
            if node.op_type == "Add":
                m = last_add1_pattern.match(node.name)
                if m and int(m.group(1)) > last_layer_idx:
                    last_layer_idx = int(m.group(1))
                    last_add1_node = node

        if last_add1_node is not None:
            add1_output = last_add1_node.output[0]
            if add1_output in vi_map:
                vi_map[add1_output].type.tensor_type.elem_type = onnx.TensorProto.FLOAT16

            new_add1_output = add1_output + "_fp16"
            cast_output = add1_output

            last_add1_node.output[0] = new_add1_output

            cast_node = onnx.helper.make_node(
                "Cast",
                inputs=[new_add1_output],
                outputs=[cast_output],
                name=f"/model/layers.{last_layer_idx}/Add_1/Cast_to_fp32",
                to=onnx.TensorProto.FLOAT,
            )
            onnx_model.graph.node.append(cast_node)

            fp16_vi = onnx.helper.make_empty_tensor_value_info(new_add1_output)
            fp16_vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
            onnx_model.graph.value_info.append(fp16_vi)

            if cast_output in vi_map:
                vi_map[cast_output].type.tensor_type.elem_type = onnx.TensorProto.FLOAT

            print(
                f"  Fixed layers.{last_layer_idx}/Add_1: output→FP16, "
                f"inserted Cast→FP32 for norm/lm_head"
            )

    # Fix logits output shape for all dtypes
    vocab_size = None
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            model_config = json.load(f)
        vocab_size = model_config.get("vocab_size")

    for output in onnx_model.graph.output:
        if output.name == "logits":
            shape = output.type.tensor_type.shape
            if shape and len(shape.dim) == 3:
                old_dims = [d.dim_param if d.dim_param else str(d.dim_value) for d in shape.dim]
                shape.dim[0].ClearField("dim_value")
                shape.dim[0].dim_param = "batch_size"
                shape.dim[1].ClearField("dim_value")
                shape.dim[1].dim_param = "sequence_length"
                if vocab_size is not None:
                    shape.dim[2].ClearField("dim_param")
                    shape.dim[2].dim_value = vocab_size
                print(
                    f"  Fixed logits shape: {old_dims} -> "
                    f"[batch_size, sequence_length, {vocab_size or 'unchanged'}]"
                )
            break

    print(
        f"Saving ONNX files in {output_dir}. All existing ONNX in the folder will be overwritten."
    )
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if (
                os.path.isfile(file_path) or os.path.islink(file_path)
            ) and ".json" not in file_path:
                os.unlink(file_path)

        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    onnx_model.ir_version = 10

    pre_gqa_onnx = os.path.join(output_dir, "_pre_gqa_model.onnx")
    pre_gqa_data = "_pre_gqa_model.data"
    onnx.save_model(
        onnx_model,
        pre_gqa_onnx,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=pre_gqa_data,
        convert_attribute=True,
    )

    if os.path.exists(config_path):
        if os.path.isfile(config_path) and config_path.endswith("config.json"):
            shutil.copy(config_path, os.path.join(output_dir, "config.json"))
        elif os.path.isdir(config_path):
            shutil.copy(
                os.path.join(config_path, "config.json"), os.path.join(output_dir, "config.json")
            )
        else:
            print(f"Warning: Unexpected config_path format: {config_path}")

    t3 = time.time()
    print(f"Surgeon LLM completed in {t3 - t2}s.")

    final_onnx = os.path.join(output_dir, "model.onnx")
    if hf_model_path:
        print("\n" + "=" * 60)
        print("Running GQA surgery: replacing attention with GroupQueryAttention...")
        print("=" * 60)
        t_gqa_start = time.time()

        replace_attention_with_gqa(
            model_path=pre_gqa_onnx,
            output_path=final_onnx,
            hf_model_id=hf_model_path,
            max_seq_len=4096,
            io_dtype="float16",
            use_external_data=True,
            external_data_name="model.onnx_data",
            ir_version=10,
            trust_remote_code=trust_remote_code,
        )

        t_gqa_end = time.time()
        print(f"GQA surgery completed in {t_gqa_end - t_gqa_start:.1f}s.")
    else:
        print("Warning: hf_model_path not provided, skipping GQA surgery.")
        os.rename(pre_gqa_onnx, final_onnx)
        os.rename(
            os.path.join(output_dir, pre_gqa_data),
            os.path.join(output_dir, "model.onnx_data"),
        )

    for temp_file in [pre_gqa_onnx, os.path.join(output_dir, pre_gqa_data)]:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
            print(f"  Removed intermediate: {os.path.basename(temp_file)}")


def check_dtype_support(args):
    """Check whether the dtype is supported by DriveOS LLM SDK.

    Returns False if it is not supported because of:
        1. Modelopt < 0.23.0 does not support nvfp4
    """

    def get_modelopt_version():
        try:
            return Version(modelopt.__version__)
        except Exception as e:
            print(f"Modelopt version cannot be parsed. Reason: {e!s}")

    if (args.dtype == "nvfp4") and get_modelopt_version() < Version("0.23.0"):
        print(
            "nvfp4 is not supported by installed modelopt version. Please upgrade to 0.23.0 or above for nvfp4 export."
        )
        return False

    return True


def main(args):
    """Main function to export the LLM model to ONNX."""
    assert args.hf_model_path or args.onnx_path, (
        "You need to provide either --hf_model_path or --onnx_path to process the export script."
    )
    start_time = time.time()

    if not check_dtype_support(args):
        return

    if args.onnx_path:
        raw_onnx_path = args.onnx_path

    model_loader = ModelLoader(args.hf_model_path, args.config_path)

    if args.hf_model_path:
        model = model_loader.load_model(trust_remote_code=args.trust_remote_code)
        onnx_dir = args.output_dir + "_raw" if args.save_original else args.output_dir
        raw_onnx_path = f"{onnx_dir}/model.onnx"
        extra_inputs, extra_dyn_axes = {}, {}
        export_raw_llm(
            model=model,
            output_dir=onnx_dir,
            dtype=args.dtype,
            config_path=args.config_path,
            hf_model_path=args.hf_model_path,
            lm_head_precision=args.lm_head,
            dataset_dir=args.dataset_dir,
            wrapper_cls=WrapperModelForCausalLM,
            extra_inputs=extra_inputs,
            extra_dyn_axes=extra_dyn_axes,
            calib_size=args.calib_size,
            trust_remote_code=args.trust_remote_code,
        )

    surgeon_llm(
        raw_onnx_path=raw_onnx_path,
        output_dir=args.output_dir,
        dtype=args.dtype,
        config_path=args.config_path,
        hf_model_path=args.hf_model_path,
        lm_head_precision=args.lm_head,
        trust_remote_code=args.trust_remote_code,
    )

    end_time = time.time()
    print(
        f"LLM ONNX saved to {args.output_dir} with {args.dtype} precision in {end_time - start_time}s."
    )

    if args.dtype == "int8_sq":
        print(
            "\nNOTE: INT8 SmoothQuant models currently work only with the CUDA EP. "
            "There are some decoding issues with fp16 max precision when running with NvTensorRtRtx."
        )


if __name__ == "__main__":
    parser = llm_arguments()
    args = parser.parse_args()
    args.config_path = get_config_path(args)
    main(args)
