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

"""Replace standard attention subgraph with GroupQueryAttention (GQA).

This module provides functionality to transform ONNX models exported from
HuggingFace/Optimum to use Microsoft's GroupQueryAttention operator,
which is optimized for inference with ONNX Runtime.

The transformation includes:
1. Converting weights to target dtype (FP16/BF16)
2. Removing unnecessary Cast nodes in layers
3. Adding Gemma-specific casts if needed
4. Computing and adding RoPE cos/sin caches
5. Adding attention mask reformatting subgraph
6. Replacing attention pattern with GQA for all layers
7. Fusing Q/K/V projections into single MatMul
8. Adding past/present KV cache inputs/outputs
"""

import contextlib
import os
import re

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.external_data_helper import convert_model_to_external_data

from ..logging_config import logger
from .utils.graph_utils import (
    add_initializer,
    array_to_initializer,
    cleanup_unused_ios,
    convert_initializers_to_dtype,
    convert_model_fp16_to_bf16,
    find_initializer,
    get_onnx_dtype,
    initializer_to_array,
)
from .utils.rope_cache import get_rope_caches


def _remove_layer_cast_nodes(graph: onnx.GraphProto, verbose: bool = True) -> int:
    """Remove unnecessary /model/layers.{i}/Cast and /Cast_1 nodes.

    Args:
        graph: ONNX graph to modify.
        verbose: Whether to print progress.

    Returns:
        Number of Cast nodes removed.
    """
    cast_pattern = re.compile(r"^/model/layers\.(\d+)/Cast(_1)?$")
    cast_nodes_removed = 0

    cast_nodes_to_remove = [
        node for node in graph.node if node.op_type == "Cast" and cast_pattern.match(node.name)
    ]

    for cast_node in cast_nodes_to_remove:
        if len(cast_node.input) == 1 and len(cast_node.output) == 1:
            cast_input = cast_node.input[0]
            cast_output = cast_node.output[0]

            # Rewire: replace all uses of cast_output with cast_input
            for node in graph.node:
                for i, inp in enumerate(node.input):
                    if inp == cast_output:
                        node.input[i] = cast_input

            # Also check graph outputs
            for out in graph.output:
                if out.name == cast_output:
                    out.name = cast_input

            graph.node.remove(cast_node)
            cast_nodes_removed += 1
            if verbose:
                logger.info(f"  Removed: {cast_node.name}")

    return cast_nodes_removed


def _add_gemma_cast_nodes(
    graph: onnx.GraphProto,
    hf_model_id: str,
    io_dtype: str,
    onnx_dtype: int,
    verbose: bool = True,
    trust_remote_code: bool = False,
) -> int:
    """Add Cast to target dtype after layernorm Mul nodes for Gemma models.

    Args:
        graph: ONNX graph to modify.
        hf_model_id: HuggingFace model ID.
        io_dtype: Target IO dtype string.
        onnx_dtype: ONNX dtype constant.
        verbose: Whether to print progress.
        trust_remote_code: Whether to trust remote code in HuggingFace model config.

    Returns:
        Number of Cast nodes added.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)
    num_layers = config.num_hidden_layers
    cast_nodes_added = 0

    # Build list of target Mul nodes for all layers
    gemma_cast_targets = []
    for layer_id in range(num_layers):
        gemma_cast_targets.append(f"/model/layers.{layer_id}/input_layernorm/Mul")
        gemma_cast_targets.append(f"/model/layers.{layer_id}/post_attention_layernorm/Mul")

    # Also add the final norm/Mul
    gemma_cast_targets.append("/model/norm/Mul")

    for target_mul_name in gemma_cast_targets:
        # Find the target Mul node
        target_mul_node = None
        for node in graph.node:
            if node.name == target_mul_name:
                target_mul_node = node
                break

        if target_mul_node and len(target_mul_node.output) > 0:
            mul_output = target_mul_node.output[0]

            # Check if output is a graph output
            is_graph_output = False
            original_output_name = None
            for out in graph.output:
                if out.name == mul_output:
                    is_graph_output = True
                    original_output_name = mul_output
                    break

            if is_graph_output:
                new_mul_output = f"{target_mul_name}_output_before_cast"
                target_mul_node.output[0] = new_mul_output
                cast_output = original_output_name
                cast_input = new_mul_output
            else:
                cast_output = f"{target_mul_name}/Cast_to_fp16/output_0"
                cast_input = mul_output

            # Create Cast node
            dtype_suffix = (
                "fp16" if io_dtype == "float16" else ("bf16" if io_dtype == "bfloat16" else "fp32")
            )
            cast_node = helper.make_node(
                "Cast",
                inputs=[cast_input],
                outputs=[cast_output],
                name=f"{target_mul_name}/Cast_to_{dtype_suffix}",
                to=onnx_dtype,
            )
            graph.node.append(cast_node)

            if not is_graph_output:
                # Rewire consumers
                for node in graph.node:
                    if node.name == cast_node.name:
                        continue
                    for i, inp in enumerate(node.input):
                        if inp == mul_output:
                            node.input[i] = cast_output

            # Update graph output type if applicable
            for out in graph.output:
                if out.name == cast_output:
                    out.type.tensor_type.elem_type = onnx_dtype

            # Add value_info
            cast_info = helper.make_tensor_value_info(
                cast_output, onnx_dtype, ["batch_size", "sequence_length", "hidden_size"]
            )
            graph.value_info.append(cast_info)

            cast_nodes_added += 1

    if verbose:
        logger.info(f"  Added {cast_nodes_added} Cast nodes for Gemma model")

    return cast_nodes_added


def _add_bf16_lm_head_cast(graph: onnx.GraphProto, verbose: bool = True) -> bool:
    """Add Cast to FP32 after /lm_head/MatMul for bfloat16 TensorRT compatibility.

    Args:
        graph: ONNX graph to modify.
        verbose: Whether to print progress.

    Returns:
        True if cast was added, False otherwise.
    """
    lm_head_matmul_name = "/lm_head/MatMul"
    lm_head_node = None
    for node in graph.node:
        if node.name == lm_head_matmul_name:
            lm_head_node = node
            break

    if lm_head_node and len(lm_head_node.output) > 0:
        # Create new output name for MatMul
        new_matmul_output = f"{lm_head_matmul_name}_output_bf16"
        lm_head_node.output[0] = new_matmul_output

        # Create Cast node to FP32
        lm_head_cast_node = helper.make_node(
            "Cast",
            inputs=[new_matmul_output],
            outputs=["logits"],
            name=f"{lm_head_matmul_name}/Cast_to_fp32",
            to=TensorProto.FLOAT,
        )
        graph.node.append(lm_head_cast_node)

        # Add value_info for bf16 intermediate
        lm_head_bf16_info = helper.make_tensor_value_info(
            new_matmul_output, TensorProto.BFLOAT16, ["batch_size", "sequence_length", "vocab_size"]
        )
        graph.value_info.append(lm_head_bf16_info)

        # Update logits output to FP32
        for out in graph.output:
            if out.name == "logits":
                out.type.tensor_type.elem_type = TensorProto.FLOAT

        if verbose:
            logger.info(f"  Added Cast to FP32 after {lm_head_matmul_name}")
        return True
    else:
        if verbose:
            logger.info(f"  Warning: Could not find {lm_head_matmul_name} node")
        return False


def _create_attention_mask_subgraph(
    graph: onnx.GraphProto,
    onnx_dtype: int,
) -> tuple[str, str]:
    """Create attention mask reformatting subgraph for GQA.

    Args:
        graph: ONNX graph to modify.
        onnx_dtype: ONNX dtype constant.

    Returns:
        Tuple of (seqlens_k_output, total_seq_len_output) tensor names.
    """
    attn_mask_basename = "/model/attn_mask_reformat/attn_mask_subgraph"

    # ReduceSum: sum attention_mask along axis 1
    reduce_sum_node = helper.make_node(
        "ReduceSum",
        inputs=["attention_mask", "/model/constants/INT64/[1]"],
        outputs=[f"{attn_mask_basename}/ReduceSum/output_0"],
        name=f"{attn_mask_basename}/ReduceSum",
        keepdims=0,
    )
    graph.node.append(reduce_sum_node)

    # Sub: seqlens_k = ReduceSum - 1
    sub_node = helper.make_node(
        "Sub",
        inputs=[f"{attn_mask_basename}/ReduceSum/output_0", "/model/constants/INT64/1"],
        outputs=[f"{attn_mask_basename}/Sub/output_0"],
        name=f"{attn_mask_basename}/Sub",
    )
    graph.node.append(sub_node)

    # Cast seqlens_k to int32
    cast_seqlens_node = helper.make_node(
        "Cast",
        inputs=[f"{attn_mask_basename}/Sub/output_0"],
        outputs=[f"{attn_mask_basename}/Sub/Cast/output_0"],
        name=f"{attn_mask_basename}/Sub/Cast",
        to=TensorProto.INT32,
    )
    graph.node.append(cast_seqlens_node)

    # Shape of attention_mask
    shape_node = helper.make_node(
        "Shape",
        inputs=["attention_mask"],
        outputs=[f"{attn_mask_basename}/Shape/output_0"],
        name=f"{attn_mask_basename}/Shape",
    )
    graph.node.append(shape_node)

    # Gather index 1 (sequence length dimension)
    gather_node = helper.make_node(
        "Gather",
        inputs=[f"{attn_mask_basename}/Shape/output_0", "/model/constants/INT64/1"],
        outputs=[f"{attn_mask_basename}/Gather/output_0"],
        name=f"{attn_mask_basename}/Gather",
        axis=0,
    )
    graph.node.append(gather_node)

    # Cast total_seq_len to int32
    cast_total_node = helper.make_node(
        "Cast",
        inputs=[f"{attn_mask_basename}/Gather/output_0"],
        outputs=[f"{attn_mask_basename}/Gather/Cast/output_0"],
        name=f"{attn_mask_basename}/Gather/Cast",
        to=TensorProto.INT32,
    )
    graph.node.append(cast_total_node)

    seqlens_k_output = f"{attn_mask_basename}/Sub/Cast/output_0"
    total_seq_len_output = f"{attn_mask_basename}/Gather/Cast/output_0"

    # Add value_info for mask subgraph outputs
    seqlens_k_info = helper.make_tensor_value_info(
        seqlens_k_output, TensorProto.INT32, ["batch_size"]
    )
    total_seq_len_info = helper.make_tensor_value_info(total_seq_len_output, TensorProto.INT32, [])
    graph.value_info.extend([seqlens_k_info, total_seq_len_info])

    return seqlens_k_output, total_seq_len_output


def _fuse_qkv_and_create_gqa(
    graph: onnx.GraphProto,
    layer_id: int,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    seqlens_k_output: str,
    total_seq_len_output: str,
    onnx_dtype: int,
    attn_prefix: str = "self_attn",
    pack_qkv: bool = False,
    verbose: bool = True,
) -> tuple[list[onnx.NodeProto], list[onnx.NodeProto], list[onnx.NodeProto]]:
    """Fuse Q/K/V MatMuls and create GQA node for a single layer.

    Args:
        graph: ONNX graph to modify.
        layer_id: Layer index.
        num_attention_heads: Number of attention heads.
        num_kv_heads: Number of key-value heads.
        head_dim: Dimension per head.
        hidden_size: Hidden dimension.
        seqlens_k_output: Name of seqlens_k tensor.
        total_seq_len_output: Name of total_seq_len tensor.
        onnx_dtype: ONNX dtype constant.
        attn_prefix: Attention namespace prefix ("self_attn" or "attn").
        verbose: Whether to print progress.

    Returns:
        Tuple of (qkv_matmul_nodes, gqa_nodes, nodes_to_remove).
    """
    qkv_matmul_nodes = []
    gqa_nodes = []
    qkv_nodes_to_remove = []

    # Helper to find node by name in graph
    def _find_node(name: str) -> onnx.NodeProto | None:
        for node in graph.node:
            if node.name == name:
                return node
        return None

    # Helper to find node by name pattern (partial match), optionally filtered by op_type
    def _find_node_by_pattern(pattern: str, op_type: str | None = None) -> onnx.NodeProto | None:
        for node in graph.node:
            if pattern in node.name:
                if op_type is None or node.op_type == op_type:
                    return node
        return None

    # Try separate Q, K, V MatMul nodes first
    q_matmul_name = f"/model/layers.{layer_id}/{attn_prefix}/q_proj/MatMul"
    k_matmul_name = f"/model/layers.{layer_id}/{attn_prefix}/k_proj/MatMul"
    v_matmul_name = f"/model/layers.{layer_id}/{attn_prefix}/v_proj/MatMul"

    q_matmul = _find_node(q_matmul_name)
    k_matmul = _find_node(k_matmul_name)
    v_matmul = _find_node(v_matmul_name)

    if not q_matmul or not k_matmul or not v_matmul:
        # Check for combined qkv_proj pattern (e.g. TinyLlama)
        qkv_proj_pattern = f"/model/layers.{layer_id}/{attn_prefix}/qkv_proj/MatMul"
        qkv_matmul = _find_node_by_pattern(qkv_proj_pattern, op_type="MatMul")

        if qkv_matmul is not None:
            if verbose:
                logger.info(f"  Layer {layer_id}: Combined qkv_proj detected: {qkv_matmul.name}")
                logger.info("    Using packed GQA mode (QKV already combined)")

            qkv_output = qkv_matmul.output[0]

            # Check for bias Add after combined qkv_proj MatMul
            qkv_add = _find_node(f"/model/layers.{layer_id}/{attn_prefix}/qkv_proj/Add")
            if qkv_add is not None:
                qkv_output = qkv_add.output[0]
                if verbose:
                    logger.info(f"    Found qkv_proj/Add, using Add output: {qkv_output}")

            # Create GQA node with packed QKV input
            past_key = f"past_key_values.{layer_id}.key"
            past_value = f"past_key_values.{layer_id}.value"
            present_key = f"present.{layer_id}.key"
            present_value = f"present.{layer_id}.value"
            gqa_output = f"/model/layers.{layer_id}/{attn_prefix}/GQA/output_0"

            gqa_node = helper.make_node(
                "GroupQueryAttention",
                inputs=[
                    qkv_output,  # packed QKV
                    "",  # key (empty for packed mode)
                    "",  # value (empty for packed mode)
                    past_key,
                    past_value,
                    seqlens_k_output,
                    total_seq_len_output,
                    "cos_cache",
                    "sin_cache",
                    "",  # position_ids
                    "",  # attention_bias
                ],
                outputs=[
                    gqa_output,
                    present_key,
                    present_value,
                ],
                name=f"/model/layers.{layer_id}/{attn_prefix}/GQA",
                domain="com.microsoft",
                num_heads=num_attention_heads,
                kv_num_heads=num_kv_heads,
                scale=1.0 / (head_dim**0.5),
                do_rotary=1,
                rotary_interleaved=0,
                local_window_size=-1,
            )
            gqa_nodes.append(gqa_node)

            # Add value_info for GQA output
            gqa_output_info = helper.make_tensor_value_info(
                gqa_output, onnx_dtype, ["batch_size", "sequence_length", hidden_size]
            )
            graph.value_info.append(gqa_output_info)

            if verbose:
                logger.info(f"    Created GQA with packed input: {qkv_output}")

            return qkv_matmul_nodes, gqa_nodes, qkv_nodes_to_remove

        if verbose:
            logger.info(f"  Warning: Could not find Q/K/V MatMul nodes for layer {layer_id}")
        return [], [], []

    assert q_matmul is not None
    assert k_matmul is not None
    assert v_matmul is not None

    # Get weight initializer names
    q_weight_name = q_matmul.input[1]
    k_weight_name = k_matmul.input[1]
    v_weight_name = v_matmul.input[1]

    # Get the input to Q MatMul
    qkv_input = q_matmul.input[0]

    # Find weight initializers
    q_weight = find_initializer(graph, q_weight_name)
    k_weight = find_initializer(graph, k_weight_name)
    v_weight = find_initializer(graph, v_weight_name)

    if not all([q_weight, k_weight, v_weight]):
        # Quantized model path: keep separate Q/K/V projections
        if verbose:
            logger.info(
                f"  Layer {layer_id}: Quantized model detected (weights behind DequantizeLinear)"
            )

        q_output = q_matmul.output[0]
        k_output = k_matmul.output[0]
        v_output = v_matmul.output[0]

        # Check for bias Add nodes after Q/K/V MatMuls
        q_add = _find_node(f"/model/layers.{layer_id}/{attn_prefix}/q_proj/Add")
        k_add = _find_node(f"/model/layers.{layer_id}/{attn_prefix}/k_proj/Add")
        v_add = _find_node(f"/model/layers.{layer_id}/{attn_prefix}/v_proj/Add")
        if q_add and k_add and v_add:
            # Use Add outputs (after bias) instead of raw MatMul outputs
            q_output = q_add.output[0]
            k_output = k_add.output[0]
            v_output = v_add.output[0]
            if verbose:
                logger.info("    Found bias Add nodes, using Add outputs for Q/K/V")

        past_key = f"past_key_values.{layer_id}.key"
        past_value = f"past_key_values.{layer_id}.value"
        present_key = f"present.{layer_id}.key"
        present_value = f"present.{layer_id}.value"
        gqa_output = f"/model/layers.{layer_id}/{attn_prefix}/GQA/output_0"

        if pack_qkv:
            # Packed QKV mode: Concat Q, K, V horizontally (axis=-1) then feed as single input
            if verbose:
                logger.info("    Packing Q/K/V with Concat (axis=-1) for packed GQA mode")

            concat_output = f"/model/layers.{layer_id}/{attn_prefix}/qkv_concat/output_0"

            concat_node = helper.make_node(
                "Concat",
                inputs=[q_output, k_output, v_output],
                outputs=[concat_output],
                name=f"/model/layers.{layer_id}/{attn_prefix}/qkv_concat",
                axis=-1,
            )
            qkv_matmul_nodes.append(concat_node)

            # Add value_info for concat output
            # Q dim = num_heads * head_dim, K dim = kv_heads * head_dim, V dim = kv_heads * head_dim
            qkv_dim = (num_attention_heads + 2 * num_kv_heads) * head_dim
            concat_info = helper.make_tensor_value_info(
                concat_output, onnx_dtype, ["batch_size", "sequence_length", qkv_dim]
            )
            graph.value_info.append(concat_info)

            gqa_node = helper.make_node(
                "GroupQueryAttention",
                inputs=[
                    concat_output,  # packed QKV
                    "",  # key (empty for packed mode)
                    "",  # value (empty for packed mode)
                    past_key,
                    past_value,
                    seqlens_k_output,
                    total_seq_len_output,
                    "cos_cache",
                    "sin_cache",
                    "",  # position_ids
                    "",  # attention_bias
                ],
                outputs=[
                    gqa_output,
                    present_key,
                    present_value,
                ],
                name=f"/model/layers.{layer_id}/{attn_prefix}/GQA",
                domain="com.microsoft",
                num_heads=num_attention_heads,
                kv_num_heads=num_kv_heads,
                scale=1.0 / (head_dim**0.5),
                do_rotary=1,
                rotary_interleaved=0,
                local_window_size=-1,
            )
        else:
            # Unpacked mode: separate Q/K/V inputs to GQA
            if verbose:
                logger.info(
                    "    Keeping separate Q/K/V projections, creating GQA with unpacked inputs"
                )

            gqa_node = helper.make_node(
                "GroupQueryAttention",
                inputs=[
                    q_output,  # query
                    k_output,  # key
                    v_output,  # value
                    past_key,
                    past_value,
                    seqlens_k_output,
                    total_seq_len_output,
                    "cos_cache",
                    "sin_cache",
                    "",  # position_ids
                    "",  # attention_bias
                ],
                outputs=[
                    gqa_output,
                    present_key,
                    present_value,
                ],
                name=f"/model/layers.{layer_id}/{attn_prefix}/GQA",
                domain="com.microsoft",
                num_heads=num_attention_heads,
                kv_num_heads=num_kv_heads,
                scale=1.0 / (head_dim**0.5),
                do_rotary=1,
                rotary_interleaved=0,
                local_window_size=-1,
            )

        gqa_nodes.append(gqa_node)

        # Add value_info for GQA output
        gqa_output_info = helper.make_tensor_value_info(
            gqa_output, onnx_dtype, ["batch_size", "sequence_length", hidden_size]
        )
        graph.value_info.append(gqa_output_info)

        # Don't remove any projection nodes, don't create fused MatMul
        return qkv_matmul_nodes, gqa_nodes, qkv_nodes_to_remove

    # Convert weights to numpy arrays
    q_arr, q_bf16 = initializer_to_array(q_weight)
    k_arr, _k_bf16 = initializer_to_array(k_weight)
    v_arr, _v_bf16 = initializer_to_array(v_weight)
    is_bfloat16 = q_bf16 == "bfloat16"

    # Concatenate weights
    qkv_weight_arr = np.concatenate([q_arr, k_arr, v_arr], axis=1)

    # Create fused QKV weight initializer
    qkv_weight_name = f"/model/layers.{layer_id}/{attn_prefix}/qkv_proj/weight"
    qkv_weight_tensor = array_to_initializer(qkv_weight_arr, qkv_weight_name, is_bfloat16)
    graph.initializer.append(qkv_weight_tensor)

    # Create fused QKV MatMul node
    qkv_matmul_name = f"/model/layers.{layer_id}/{attn_prefix}/qkv_proj/MatMul"
    qkv_matmul_output = f"{qkv_matmul_name}_output_0"

    qkv_matmul_node = helper.make_node(
        "MatMul",
        inputs=[qkv_input, qkv_weight_name],
        outputs=[qkv_matmul_output],
        name=qkv_matmul_name,
    )
    qkv_matmul_nodes.append(qkv_matmul_node)

    # Add value_info for fused QKV output
    packed_qkv_dim = (num_attention_heads + 2 * num_kv_heads) * head_dim
    qkv_output_info = helper.make_tensor_value_info(
        qkv_matmul_output, onnx_dtype, ["batch_size", "sequence_length", packed_qkv_dim]
    )
    graph.value_info.append(qkv_output_info)

    # Mark old Q/K/V MatMul nodes for removal
    qkv_nodes_to_remove.extend([q_matmul, k_matmul, v_matmul])

    # Remove old weight initializers
    graph.initializer.remove(q_weight)
    graph.initializer.remove(k_weight)
    graph.initializer.remove(v_weight)

    # Check for bias Add nodes (e.g. Qwen models)
    q_add_name = f"/model/layers.{layer_id}/{attn_prefix}/q_proj/Add"
    k_add_name = f"/model/layers.{layer_id}/{attn_prefix}/k_proj/Add"
    v_add_name = f"/model/layers.{layer_id}/{attn_prefix}/v_proj/Add"

    q_add = _find_node(q_add_name)
    k_add = _find_node(k_add_name)
    v_add = _find_node(v_add_name)

    gqa_input = qkv_matmul_output

    if q_add and k_add and v_add:
        # Fuse bias Add operations
        if verbose:
            logger.info(f"  Layer {layer_id}: Found bias Add nodes, fusing biases...")

        # Get bias initializer names
        q_bias_name = q_add.input[1] if find_initializer(graph, q_add.input[1]) else q_add.input[0]
        k_bias_name = k_add.input[1] if find_initializer(graph, k_add.input[1]) else k_add.input[0]
        v_bias_name = v_add.input[1] if find_initializer(graph, v_add.input[1]) else v_add.input[0]

        q_bias = find_initializer(graph, q_bias_name)
        k_bias = find_initializer(graph, k_bias_name)
        v_bias = find_initializer(graph, v_bias_name)

        if all([q_bias, k_bias, v_bias]):
            # Concatenate biases
            q_bias_arr, qb_bf16 = initializer_to_array(q_bias)
            k_bias_arr, _ = initializer_to_array(k_bias)
            v_bias_arr, _ = initializer_to_array(v_bias)
            bias_is_bfloat16 = qb_bf16 == "bfloat16"

            qkv_bias_arr = np.concatenate([q_bias_arr, k_bias_arr, v_bias_arr], axis=0)

            # Create fused bias initializer
            qkv_bias_name = f"/model/layers.{layer_id}/{attn_prefix}/qkv_proj/bias"
            qkv_bias_tensor = array_to_initializer(qkv_bias_arr, qkv_bias_name, bias_is_bfloat16)
            graph.initializer.append(qkv_bias_tensor)

            # Create fused Add node
            qkv_add_name = f"/model/layers.{layer_id}/{attn_prefix}/qkv_proj/Add"
            qkv_add_output = f"{qkv_add_name}_output_0"

            qkv_add_node = helper.make_node(
                "Add",
                inputs=[qkv_matmul_output, qkv_bias_name],
                outputs=[qkv_add_output],
                name=qkv_add_name,
            )
            graph.node.append(qkv_add_node)

            # Add value_info
            qkv_add_info = helper.make_tensor_value_info(
                qkv_add_output, onnx_dtype, ["batch_size", "sequence_length", packed_qkv_dim]
            )
            graph.value_info.append(qkv_add_info)

            # Update GQA input
            gqa_input = qkv_add_output

            # Mark old Add nodes for removal
            qkv_nodes_to_remove.extend([q_add, k_add, v_add])

            # Remove old bias initializers
            graph.initializer.remove(q_bias)
            graph.initializer.remove(k_bias)
            graph.initializer.remove(v_bias)

            if verbose:
                logger.info(
                    f"  Layer {layer_id}: Fused biases {q_bias_arr.shape} + "
                    f"{k_bias_arr.shape} + {v_bias_arr.shape} -> {qkv_bias_arr.shape}"
                )

    # Create GQA node
    past_key = f"past_key_values.{layer_id}.key"
    past_value = f"past_key_values.{layer_id}.value"
    present_key = f"present.{layer_id}.key"
    present_value = f"present.{layer_id}.value"
    gqa_output = f"/model/layers.{layer_id}/{attn_prefix}/GQA/output_0"

    gqa_node = helper.make_node(
        "GroupQueryAttention",
        inputs=[
            gqa_input,
            "",  # key (empty for packed mode)
            "",  # value (empty for packed mode)
            past_key,
            past_value,
            seqlens_k_output,
            total_seq_len_output,
            "cos_cache",
            "sin_cache",
            "",  # position_ids
            "",  # attention_bias
        ],
        outputs=[
            gqa_output,
            present_key,
            present_value,
        ],
        name=f"/model/layers.{layer_id}/{attn_prefix}/GQA",
        domain="com.microsoft",
        num_heads=num_attention_heads,
        kv_num_heads=num_kv_heads,
        scale=1.0 / (head_dim**0.5),
        do_rotary=1,
        rotary_interleaved=0,
        local_window_size=-1,
    )
    gqa_nodes.append(gqa_node)

    # Add value_info for GQA output
    gqa_output_info = helper.make_tensor_value_info(
        gqa_output, onnx_dtype, ["batch_size", "sequence_length", hidden_size]
    )
    graph.value_info.append(gqa_output_info)

    if verbose:
        logger.info(
            f"  Layer {layer_id}: Fused Q/K/V weights {q_arr.shape} + "
            f"{k_arr.shape} + {v_arr.shape} -> {qkv_weight_arr.shape}"
        )

    return qkv_matmul_nodes, gqa_nodes, qkv_nodes_to_remove


def replace_attention_with_gqa(
    model_path: str,
    output_path: str,
    hf_model_id: str,
    max_seq_len: int = 4096,
    io_dtype: str = "float16",
    use_external_data: bool = True,
    external_data_name: str | None = None,
    ir_version: int | None = None,
    pack_qkv: bool = False,
    verbose: bool = True,
    trust_remote_code: bool = False,
) -> onnx.ModelProto:
    """Replace attention subgraphs with GroupQueryAttention (GQA) in an ONNX model.

    This function transforms an ONNX model exported from HuggingFace/Optimum
    to use Microsoft's GroupQueryAttention operator, which is optimized for
    inference with ONNX Runtime.

    The transformation includes:
    - Converting weights to target dtype (FP16/BF16) [non-quantized models only]
    - Adding RoPE cos/sin caches
    - Replacing attention patterns with GQA for all layers
    - Fusing Q/K/V projections into single MatMul [non-quantized models only]
    - Concatenating Q/K/V outputs for GQA [quantized models only]
    - Adding past/present KV cache inputs/outputs

    Args:
        model_path: Path to input ONNX model.
        output_path: Path to save modified model.
        hf_model_id: HuggingFace model ID for config.
        max_seq_len: Maximum sequence length for caches.
        io_dtype: Data type for I/O tensors ("float16", "float32", or "bfloat16").
            If the model has FP16 initializers and "bfloat16" is specified,
            they are automatically converted to BF16.
        use_external_data: Save weights as external data file.
        external_data_name: Name for external data file (default: model.onnx_data).
        ir_version: If specified, set the ONNX IR version to this value. Useful for
            compatibility with older ONNX Runtime versions (e.g., set to 9 for ORT 1.16).
        verbose: Whether to print progress messages.
        trust_remote_code: Whether to trust remote code in HuggingFace model config.

    Returns:
        Modified ONNX model.

    Example:
        >>> from modelopt.onnx.graph_surgery import replace_attention_with_gqa
        >>> model = replace_attention_with_gqa(
        ...     model_path="model_fp16.onnx",
        ...     output_path="model_gqa.onnx",
        ...     hf_model_id="meta-llama/Llama-2-7b-hf",
        ...     max_seq_len=4096,
        ...     io_dtype="float16",
        ... )
    """
    if verbose:
        logger.info(f"Loading model from: {model_path}")
    model = onnx.load(model_path)
    graph = model.graph

    onnx_dtype = get_onnx_dtype(io_dtype)

    # Early detection: check if model is quantized (has DequantizeLinear nodes)
    has_dequantize = any(n.op_type == "DequantizeLinear" for n in graph.node)
    if has_dequantize and verbose:
        logger.info("Quantized model detected (DequantizeLinear nodes found)")
        logger.info("  Skipping dtype conversion and Cast removal to preserve quantization graph")

    if not has_dequantize:
        # Step 0: Convert float32 weights to target dtype (non-quantized models only)
        if verbose:
            logger.info(f"\nConverting float32 initializers to {io_dtype}...")
        converted_count = convert_initializers_to_dtype(graph, io_dtype)
        if verbose:
            logger.info(f"  Converted {converted_count} initializers to {io_dtype}")

        # Step 0.1: If target is bfloat16, also convert all FP16 elements to BF16
        if io_dtype == "bfloat16":
            if verbose:
                logger.info("\nConverting FP16 elements to BF16 (io_dtype=bfloat16)...")
            convert_model_fp16_to_bf16(graph, verbose=verbose)

        # Step 0.5: Remove unnecessary Cast nodes in layers
        if verbose:
            logger.info("\nRemoving unnecessary /model/layers.{i}/Cast and /Cast_1 nodes...")
        cast_nodes_removed = _remove_layer_cast_nodes(graph, verbose)
        if verbose:
            logger.info(f"  Total Cast nodes removed: {cast_nodes_removed}")

    if not has_dequantize:
        # Step 0.6: Gemma-specific casts
        is_gemma = "gemma" in hf_model_id.lower()
        if is_gemma:
            if verbose:
                logger.info(
                    "\nGemma model detected - adding Cast to fp16 after layernorm Mul nodes..."
                )
            _add_gemma_cast_nodes(
                graph, hf_model_id, io_dtype, onnx_dtype, verbose, trust_remote_code
            )

        # Step 0.7: BF16 lm_head cast for TensorRT compatibility
        if io_dtype == "bfloat16":
            if verbose:
                logger.info(
                    "\nAdding Cast to FP32 after /lm_head/MatMul for bfloat16 TensorRT compatibility..."
                )
            _add_bf16_lm_head_cast(graph, verbose)

    # Get config and compute caches
    if verbose:
        logger.info(f"\nComputing RoPE caches from: {hf_model_id}")
    cos_cache, sin_cache, config = get_rope_caches(
        hf_model_id, max_seq_len, io_dtype, trust_remote_code=trust_remote_code
    )

    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    head_dim = config.hidden_size // num_attention_heads
    hidden_size = config.hidden_size

    # Auto-detect attention namespace: "self_attn" vs "attn"
    attn_prefix = "self_attn"
    for node in graph.node:
        if "/layers.0/attn/" in node.name and "/self_attn/" not in node.name:
            attn_prefix = "attn"
            break

    # Auto-detect combined QKV pattern
    has_combined_qkv = any(f"/layers.0/{attn_prefix}/qkv_proj/" in node.name for node in graph.node)

    if verbose:
        logger.info("Model config:")
        logger.info(f"  num_layers: {num_layers}")
        logger.info(f"  num_attention_heads: {num_attention_heads}")
        logger.info(f"  num_kv_heads: {num_kv_heads}")
        logger.info(f"  head_dim: {head_dim}")
        logger.info(f"  hidden_size: {hidden_size}")
        logger.info(f"  cos_cache shape: {cos_cache.shape}")
        logger.info(f"  sin_cache shape: {sin_cache.shape}")
        logger.info(f"  attn_prefix: {attn_prefix}")
        logger.info(f"  has_combined_qkv: {has_combined_qkv}")

    # Step 1: Add cos/sin cache initializers
    if verbose:
        logger.info("\nAdding cos/sin cache initializers...")
    add_initializer(graph, "cos_cache", cos_cache, onnx_dtype)
    add_initializer(graph, "sin_cache", sin_cache, onnx_dtype)

    # Add value_info for cos/sin caches
    cos_cache_info = helper.make_tensor_value_info("cos_cache", onnx_dtype, list(cos_cache.shape))
    sin_cache_info = helper.make_tensor_value_info("sin_cache", onnx_dtype, list(sin_cache.shape))
    graph.value_info.extend([cos_cache_info, sin_cache_info])

    # Step 2: Add constant initializers
    if verbose:
        logger.info("Adding constant initializers...")
    add_initializer(
        graph, "/model/constants/INT64/1", np.array(1, dtype=np.int64), TensorProto.INT64
    )
    add_initializer(
        graph, "/model/constants/INT64/[1]", np.array([1], dtype=np.int64), TensorProto.INT64
    )

    # Step 2.5: Rename suffixed I/O names to standard names
    # Some models have suffixed names like input_ids_318, attention_mask_337 etc.
    # Rename them to standard names for compatibility with genai_config.
    io_renames = {}
    for inp in graph.input:
        if inp.name.startswith("input_ids") and inp.name != "input_ids":
            io_renames[inp.name] = "input_ids"
        elif inp.name.startswith("attention_mask") and inp.name != "attention_mask":
            io_renames[inp.name] = "attention_mask"
    for out in graph.output:
        if out.name.startswith("logits") and out.name != "logits":
            io_renames[out.name] = "logits"

    if io_renames and verbose:
        logger.info("Renaming suffixed I/O names to standard names...")

    for old_name, new_name in io_renames.items():
        # Rename graph input or output
        for inp in graph.input:
            if inp.name == old_name:
                inp.name = new_name
                if verbose:
                    logger.info(f"  Renamed input: {old_name} -> {new_name}")
                break
        for out in graph.output:
            if out.name == old_name:
                out.name = new_name
                if verbose:
                    logger.info(f"  Renamed output: {old_name} -> {new_name}")
                break
        # Rename in all node inputs/outputs
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp == old_name:
                    node.input[i] = new_name
            for i, out in enumerate(node.output):
                if out == old_name:
                    node.output[i] = new_name
        # Rename in value_info
        for vi in graph.value_info:
            if vi.name == old_name:
                vi.name = new_name

    # Step 3: Ensure attention_mask input exists with dynamic shape
    existing_inputs = [inp.name for inp in graph.input]
    if "attention_mask" not in existing_inputs:
        if verbose:
            logger.info("Adding attention_mask input...")
        attn_mask_input = helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]
        )
        graph.input.append(attn_mask_input)
    else:
        # Update existing attention_mask to have dynamic shape
        for inp in graph.input:
            if inp.name == "attention_mask":
                inp.CopyFrom(
                    helper.make_tensor_value_info(
                        "attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]
                    )
                )
                if verbose:
                    logger.info("Updated attention_mask input to dynamic shape")
                break

    # Also ensure input_ids has dynamic shape
    for inp in graph.input:
        if inp.name == "input_ids":
            inp.CopyFrom(
                helper.make_tensor_value_info(
                    "input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
                )
            )
            if verbose:
                logger.info("Updated input_ids input to dynamic shape")
            break

    # Step 4: Remove existing past_key_values inputs and present outputs
    if verbose:
        logger.info("Handling past_key_values inputs and present outputs...")

    # Remove existing past_key_values inputs
    inputs_to_remove = []
    for inp in graph.input:
        if "past_key_values" in inp.name or "past_key" in inp.name or "past_value" in inp.name:
            inputs_to_remove.append(inp)
            if verbose:
                logger.info(f"  Removing existing input: {inp.name}")
    for inp in inputs_to_remove:
        graph.input.remove(inp)

    # Remove existing present outputs
    outputs_to_remove = []
    for out in graph.output:
        if "present" in out.name:
            outputs_to_remove.append(out)
            if verbose:
                logger.info(f"  Removing existing output: {out.name}")
    for out in outputs_to_remove:
        graph.output.remove(out)

    # Clean up value_info and initializers
    value_info_to_remove = [
        vi for vi in graph.value_info if "past_key_values" in vi.name or "present" in vi.name
    ]
    for vi in value_info_to_remove:
        graph.value_info.remove(vi)

    initializers_to_remove = [
        init
        for init in graph.initializer
        if "past_key_values" in init.name or "present" in init.name
    ]
    for init in initializers_to_remove:
        graph.initializer.remove(init)

    if verbose:
        logger.info(f"  Removed {len(inputs_to_remove)} existing past_key_values inputs")
        logger.info(f"  Removed {len(outputs_to_remove)} existing present outputs")
        logger.info("  Adding new past_key_values inputs and present outputs...")

    # Add new past_key_values inputs and present outputs
    kv_cache_shape = ["batch_size", num_kv_heads, "past_sequence_length", head_dim]
    present_shape = ["batch_size", num_kv_heads, "total_sequence_length", head_dim]

    for layer_id in range(num_layers):
        # Past key/value inputs
        past_key_name = f"past_key_values.{layer_id}.key"
        past_value_name = f"past_key_values.{layer_id}.value"
        graph.input.append(helper.make_tensor_value_info(past_key_name, onnx_dtype, kv_cache_shape))
        graph.input.append(
            helper.make_tensor_value_info(past_value_name, onnx_dtype, kv_cache_shape)
        )

        # Present key/value outputs
        present_key_name = f"present.{layer_id}.key"
        present_value_name = f"present.{layer_id}.value"
        graph.output.append(
            helper.make_tensor_value_info(present_key_name, onnx_dtype, present_shape)
        )
        graph.output.append(
            helper.make_tensor_value_info(present_value_name, onnx_dtype, present_shape)
        )

    # Step 5: Create attention mask reformatting subgraph
    if verbose:
        logger.info("Creating attention mask reformatting subgraph...")
    seqlens_k_output, total_seq_len_output = _create_attention_mask_subgraph(graph, onnx_dtype)

    # Step 6: Process Q/K/V projections and create GQA nodes
    all_qkv_matmul_nodes = []
    all_gqa_nodes = []
    all_qkv_nodes_to_remove = []

    # Fuse Q/K/V weights (or keep separate for quantized models)
    if verbose:
        logger.info("Processing Q/K/V projections and creating GQA nodes for each layer...")

    for layer_id in range(num_layers):
        qkv_matmul_nodes, gqa_nodes, qkv_nodes_to_remove = _fuse_qkv_and_create_gqa(
            graph,
            layer_id,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
            seqlens_k_output,
            total_seq_len_output,
            onnx_dtype,
            attn_prefix=attn_prefix,
            pack_qkv=pack_qkv,
            verbose=verbose,
        )
        all_qkv_matmul_nodes.extend(qkv_matmul_nodes)
        all_gqa_nodes.extend(gqa_nodes)
        all_qkv_nodes_to_remove.extend(qkv_nodes_to_remove)

    # Detect if model is quantized: use the early detection flag (has_dequantize)
    # We can't rely on all_qkv_matmul_nodes being empty because pack_qkv adds Concat nodes there
    is_quantized = has_dequantize

    # Step 7: Identify attention nodes to remove
    if verbose:
        logger.info(
            f"\nIdentifying attention subgraphs to replace "
            f"(quantized={is_quantized}, combined_qkv={has_combined_qkv}, "
            f"attn_prefix={attn_prefix})..."
        )

    nodes_to_remove = []
    for layer_id in range(num_layers):
        layer_prefix = f"/model/layers.{layer_id}/{attn_prefix}"

        for node in graph.node:
            if layer_prefix in node.name:
                if has_combined_qkv:
                    # Combined QKV model: keep qkv_proj/ and o_proj/ chains
                    if any(x in node.name for x in ["/qkv_proj/", "/o_proj/"]):
                        continue
                elif is_quantized:
                    # Quantized model: keep entire q_proj/, k_proj/, v_proj/, o_proj/ chains
                    # Also keep AWQ pre_quant_scale nodes (activation scaling before o_proj)
                    if any(
                        x in node.name
                        for x in ["/q_proj/", "/k_proj/", "/v_proj/", "/o_proj/", "pre_quant_scale"]
                    ):
                        continue
                # Non-quantized: only keep the 4 projection MatMul nodes
                elif any(
                    x in node.name
                    for x in [
                        "/q_proj/MatMul",
                        "/k_proj/MatMul",
                        "/v_proj/MatMul",
                        "/o_proj/MatMul",
                    ]
                ):
                    continue
                nodes_to_remove.append(node)

    if verbose:
        logger.info(f"Found {len(nodes_to_remove)} nodes to remove")

    # Remove old Q/K/V nodes
    if verbose:
        logger.info(f"Removing {len(all_qkv_nodes_to_remove)} old Q/K/V MatMul nodes...")
    for node in all_qkv_nodes_to_remove:
        with contextlib.suppress(ValueError):
            graph.node.remove(node)

    # Add fused QKV MatMul nodes
    if verbose:
        logger.info(f"Adding {len(all_qkv_matmul_nodes)} fused QKV MatMul nodes...")
    for node in all_qkv_matmul_nodes:
        graph.node.append(node)

    # Step 8: Remove old attention nodes
    if verbose:
        logger.info("Removing old attention nodes...")
    for node in nodes_to_remove:
        with contextlib.suppress(ValueError):
            graph.node.remove(node)

    # Step 9: Add GQA nodes
    if verbose:
        logger.info("Adding GQA nodes...")
    for gqa_node in all_gqa_nodes:
        graph.node.append(gqa_node)

    # Step 10: Reconnect o_proj to GQA output
    if verbose:
        logger.info(
            f"Reconnecting o_proj inputs to GQA outputs "
            f"(quantized={is_quantized}, combined_qkv={has_combined_qkv})..."
        )

    for layer_id in range(num_layers):
        gqa_output = f"/model/layers.{layer_id}/{attn_prefix}/GQA/output_0"

        if has_combined_qkv:
            # Combined QKV model: find o_proj MatMul by pattern match
            o_proj_pattern = f"/model/layers.{layer_id}/{attn_prefix}/o_proj/MatMul"
            connected = False
            for node in graph.node:
                if o_proj_pattern in node.name and node.op_type == "MatMul":
                    node.input[0] = gqa_output
                    if verbose:
                        logger.info(
                            f"  Layer {layer_id}: Connected {node.name} input[0] to {gqa_output}"
                        )
                    connected = True
                    break
            if not connected and verbose:
                logger.info(f"  Warning: Could not find o_proj MatMul for layer {layer_id}")
        elif is_quantized:
            # Quantized model: connect GQA output to the first node in the o_proj quantization chain.
            # AWQ pattern:  [attn output] -> pre_quant_scale_mul -> o_proj/MatMul
            # INT8 pattern: [attn output] -> o_proj/input_quantizer/Mul -> Cast -> o_proj/MatMul
            # FP4 pattern:  [attn output] -> o_proj/input_quantizer/Cast ->
            #   TRT_FP4DynamicQuantize -> DQL -> DQL_1 -> Cast(_f16) -> o_proj/MatMul
            connected = False

            # Try AWQ pattern first: pre_quant_scale_mul before o_proj
            layer_prefix_local = f"/model/layers.{layer_id}/{attn_prefix}"
            for node in graph.node:
                if (
                    layer_prefix_local in node.name
                    and "pre_quant_scale" in node.name
                    and node.op_type == "Mul"
                ):
                    node.input[0] = gqa_output
                    if verbose:
                        logger.info(
                            f"  Layer {layer_id}: Connected {node.name} input[0] to {gqa_output} (AWQ)"
                        )
                    connected = True
                    break

            # Try INT8 pattern: o_proj/input_quantizer/Mul
            if not connected:
                o_proj_quant_mul = (
                    f"/model/layers.{layer_id}/{attn_prefix}/o_proj/input_quantizer/Mul"
                )
                for node in graph.node:
                    if node.name == o_proj_quant_mul:
                        node.input[0] = gqa_output
                        if verbose:
                            logger.info(
                                f"  Layer {layer_id}: Connected {o_proj_quant_mul} input[0] to {gqa_output}"
                            )
                        connected = True
                        break

            # Try FP4 pattern: o_proj/input_quantizer/Cast
            if not connected:
                o_proj_quant_cast = (
                    f"/model/layers.{layer_id}/{attn_prefix}/o_proj/input_quantizer/Cast"
                )
                for node in graph.node:
                    if node.name == o_proj_quant_cast:
                        node.input[0] = gqa_output
                        if verbose:
                            logger.info(
                                f"  Layer {layer_id}: Connected {o_proj_quant_cast} input[0] to {gqa_output}"
                            )
                        connected = True
                        break

            # Fallback: connect directly to o_proj/MatMul
            if not connected:
                o_proj_name = f"/model/layers.{layer_id}/{attn_prefix}/o_proj/MatMul"
                for node in graph.node:
                    if node.name == o_proj_name:
                        node.input[0] = gqa_output
                        if verbose:
                            logger.info(
                                f"  Layer {layer_id}: Connected {o_proj_name} to {gqa_output} (fallback)"
                            )
                        connected = True
                        break
            if not connected and verbose:
                logger.info(f"  Warning: Could not connect o_proj for layer {layer_id}")
        else:
            # Non-quantized: connect directly to o_proj/MatMul
            o_proj_name = f"/model/layers.{layer_id}/{attn_prefix}/o_proj/MatMul"
            for node in graph.node:
                if node.name == o_proj_name:
                    node.input[0] = gqa_output
                    if verbose:
                        logger.info(f"  Layer {layer_id}: Connected {o_proj_name} to {gqa_output}")
                    break

    # Step 10.5: Fix o_proj input_quantizer dtype for quantized models
    # GQA outputs float16, but the input_quantizer/Mul scale is float32 and there's
    # a redundant Cast node. Convert scale to fp16, remove Cast, rewire directly.
    if is_quantized and not has_combined_qkv:
        if verbose:
            logger.info("Fixing o_proj input_quantizer dtypes for quantized model...")
        for layer_id in range(num_layers):
            quant_mul_name = f"/model/layers.{layer_id}/{attn_prefix}/o_proj/input_quantizer/Mul"
            cast_name = f"/model/layers.{layer_id}/{attn_prefix}/o_proj/MatMul_act_cast_fp16"

            # 1) Convert scale initializer to float16
            for node in graph.node:
                if node.name == quant_mul_name:
                    scale_name = node.input[1]
                    for init in graph.initializer:
                        if init.name == scale_name and init.data_type == TensorProto.FLOAT:
                            from onnx import numpy_helper

                            arr_fp16 = numpy_helper.to_array(init).astype(np.float16)
                            converted_init = numpy_helper.from_array(arr_fp16, name=init.name)
                            init.CopyFrom(converted_init)
                            if verbose:
                                logger.info(
                                    f"  Layer {layer_id}: Converted {scale_name} to float16"
                                )
                            break
                    break

            # 2) Remove Cast node and rewire: Mul output goes directly to MatMul
            cast_node = None
            for node in graph.node:
                if node.name == cast_name:
                    cast_node = node
                    break

            if cast_node is not None:
                cast_input = cast_node.input[0]  # input_quantizer/Mul_output_0
                cast_output = cast_node.output[0]  # input_quantizer/Mul_output_0_cast_fp16
                # Rewire all consumers of cast_output to use cast_input
                for node in graph.node:
                    for i, inp in enumerate(node.input):
                        if inp == cast_output:
                            node.input[i] = cast_input
                graph.node.remove(cast_node)
                # Update value_info for the Mul output to float16
                for vi in graph.value_info:
                    if vi.name == cast_input:
                        vi.type.tensor_type.elem_type = TensorProto.FLOAT16
                        if verbose:
                            logger.info(
                                f"  Layer {layer_id}: Updated {cast_input} value_info to float16"
                            )
                        break
                if verbose:
                    logger.info(f"  Layer {layer_id}: Removed {cast_name}")

    # Step 11: Add opset import for com.microsoft domain
    if verbose:
        logger.info("Adding com.microsoft opset import...")
    has_ms_domain = any(opset.domain == "com.microsoft" for opset in model.opset_import)

    if not has_ms_domain:
        ms_opset = helper.make_opsetid("com.microsoft", 1)
        model.opset_import.append(ms_opset)

    # Step 12: Clean up unused I/Os
    if verbose:
        logger.info("\nCleaning up unused I/Os and orphaned nodes...")
    cleanup_stats = cleanup_unused_ios(graph)
    if verbose:
        logger.info(f"  Removed {cleanup_stats['nodes_removed']} orphaned nodes")
        logger.info(f"  Removed {cleanup_stats['inputs_removed']} unused inputs")
        logger.info(f"  Removed {cleanup_stats['outputs_removed']} unused outputs")
        logger.info(f"  Removed {cleanup_stats['initializers_removed']} unused initializers")
        logger.info(f"  Removed {cleanup_stats['value_info_removed']} unused value_info entries")

    # Step 12.5: Adjust IR version if specified
    if ir_version is not None:
        # Check if opset 21 is being used (requires IR version >= 10)
        current_opset = 0
        for opset in model.opset_import:
            if opset.domain in {"", "ai.onnx"}:
                current_opset = opset.version
                break

        if current_opset >= 21 and ir_version < 10:
            if verbose:
                logger.info(
                    f"\nWarning: opset {current_opset} requires IR version >= 10,"
                    f" but --ir-version {ir_version} was requested."
                )
                logger.info("  Setting IR version to 10 (minimum for opset 21)")
            ir_version = 10

        old_ir = model.ir_version
        model.ir_version = ir_version
        if verbose:
            logger.info(f"\nSetting IR version: {old_ir} -> {ir_version}")

    # Step 13: Save the modified model
    if verbose:
        logger.info(f"\nSaving modified model to: {output_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if external_data_name is None:
        external_data_name = os.path.basename(output_path) + "_data"

    if use_external_data:
        if verbose:
            logger.info(f"  Saving weights to external file: {external_data_name}")

        convert_model_to_external_data(
            model,
            all_tensors_to_one_file=True,
            location=external_data_name,
            size_threshold=1024,
            convert_attribute=False,
        )

        onnx.save(
            model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_name,
            size_threshold=1024,
        )
    else:
        onnx.save(model, output_path)

    # Run shape inference (file-to-file, works with external data)
    if verbose:
        logger.info("\nRunning shape inference (file-to-file)...")
    try:
        onnx.shape_inference.infer_shapes_path(
            output_path, output_path, check_type=False, strict_mode=False, data_prop=False
        )
        if verbose:
            logger.info("  Shape inference completed")
    except Exception as e:
        if verbose:
            logger.info(f"  Shape inference failed (non-fatal, model already saved): {e}")

    if verbose:
        logger.info("\n" + "=" * 60)
        logger.info("DONE! Model has been modified with GQA attention.")
        logger.info("=" * 60)
        logger.info("\nSummary:")
        if is_quantized:
            logger.info("  - Mode: Quantized (INT4/INT8) - separate Q/K/V projections")
        else:
            logger.info("  - Mode: Standard (FP16/BF16) - fused Q/K/V weights")
            logger.info(f"  - Added {len(all_qkv_matmul_nodes)} fused QKV MatMul nodes")
        logger.info(f"  - Replaced {num_layers} attention subgraphs with GQA")
        logger.info(f"  - Added cos_cache shape: {cos_cache.shape}")
        logger.info(f"  - Added sin_cache shape: {sin_cache.shape}")
        logger.info(f"  - Added {num_layers * 2} past_key_values inputs")
        logger.info(f"  - Added {num_layers * 2} present outputs")
        logger.info("  - Added attention mask reformatting subgraph")
        logger.info(f"  - Cleaned up {sum(cleanup_stats.values())} unused graph elements")
        if use_external_data:
            logger.info(f"  - Weights saved to: {external_data_name}")

    return model


def analyze_attention_pattern(model_path: str, layer_id: int = 0) -> list[onnx.NodeProto]:
    """Analyze the attention pattern in an existing model.

    This is useful for debugging before running the full replacement.

    Args:
        model_path: Path to ONNX model.
        layer_id: Layer to analyze.

    Returns:
        List of attention nodes in the specified layer.
    """
    logger.info(f"Analyzing attention pattern for layer {layer_id}...")
    model = onnx.load(model_path)
    graph = model.graph

    layer_prefix = f"/model/layers.{layer_id}/self_attn"

    logger.info(f"\nNodes in {layer_prefix}:")
    logger.info("-" * 80)

    attn_nodes = []
    for node in graph.node:
        if layer_prefix in node.name:
            attn_nodes.append(node)
            logger.info(f"  {node.op_type:20} | {node.name}")
            logger.info(f"    inputs:  {list(node.input)}")
            logger.info(f"    outputs: {list(node.output)}")

    logger.info(f"Total attention nodes in layer {layer_id}: {len(attn_nodes)}")

    # Find layer norm output
    layernorm_pattern = f"/model/layers.{layer_id}/input_layernorm"
    logger.info(f"\nLayer norm nodes ({layernorm_pattern}):")
    for node in graph.node:
        if layernorm_pattern in node.name:
            logger.info(f"  {node.op_type:20} | {node.name}")
            logger.info(f"    outputs: {list(node.output)}")

    return attn_nodes
