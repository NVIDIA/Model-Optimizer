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

"""Add cross-attention KV cache outputs to encoder model.

This module provides functionality to transform Optimum-exported encoder models
(e.g., Whisper encoder) by adding cross-attention Key/Value projection outputs.
This is required for ONNX Runtime GenAI compatibility where the decoder expects
pre-computed encoder K/V caches.

The transformation:
1. Loads cross-attention K/V projection weights from HuggingFace model
2. Adds MatMul -> Reshape -> Transpose nodes to encoder graph
3. Adds new outputs: present_key_cross_0, present_value_cross_0, etc.
"""

import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..logging_config import logger
from .utils.graph_utils import detect_model_dtype


def _get_cross_attn_weights_from_hf(
    model_id: str, trust_remote_code: bool = False
) -> tuple[dict, int, int, int]:
    """Extract cross-attention K and V projection weights from HuggingFace model.

    Args:
        model_id: HuggingFace model ID (e.g., "openai/whisper-large-v3-turbo").
        trust_remote_code: Whether to trust remote code in HuggingFace model.

    Returns:
        Tuple of (weights_dict, num_heads, head_size, num_layers).
    """
    from transformers import WhisperForConditionalGeneration

    logger.info(f"Loading PyTorch model: {model_id}")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )

    weights = {}
    num_layers = model.config.decoder_layers
    num_heads = model.config.decoder_attention_heads
    hidden_size = model.config.d_model
    head_size = hidden_size // num_heads

    logger.info(f"  num_layers: {num_layers}")
    logger.info(f"  num_heads: {num_heads}")
    logger.info(f"  hidden_size: {hidden_size}")
    logger.info(f"  head_size: {head_size}")

    for i in range(num_layers):
        layer = model.model.decoder.layers[i]

        # encoder_attn is the cross-attention layer
        k_proj = layer.encoder_attn.k_proj
        v_proj = layer.encoder_attn.v_proj

        weights[i] = {
            "k_weight": k_proj.weight.detach().cpu().numpy(),  # (out_features, in_features)
            "v_weight": v_proj.weight.detach().cpu().numpy(),
            "k_bias": k_proj.bias.detach().cpu().numpy() if k_proj.bias is not None else None,
            "v_bias": v_proj.bias.detach().cpu().numpy() if v_proj.bias is not None else None,
        }

        logger.debug(
            f"  Layer {i}: K weight {weights[i]['k_weight'].shape}, "
            f"V weight {weights[i]['v_weight'].shape}"
        )

    return weights, num_heads, head_size, num_layers


def _rename_input(graph: onnx.GraphProto, old_name: str, new_name: str) -> None:
    """Rename an input in the graph."""
    for inp in graph.input:
        if inp.name == old_name:
            inp.name = new_name
            logger.debug(f"  Renamed input: {old_name} -> {new_name}")
            break

    # Rename in all nodes that use this input
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == old_name:
                node.input[i] = new_name


def _add_cross_kv_outputs(
    encoder_model: onnx.ModelProto,
    cross_attn_weights: dict,
    hidden_state_output_name: str,
    num_heads: int,
    head_size: int,
    num_layers: int,
    rename_input_features: bool = True,
    onnx_dtype: int = TensorProto.FLOAT,
    np_dtype: np.dtype = np.float32,
) -> onnx.ModelProto:
    """Add cross-attention KV cache computation to encoder model.

    Args:
        encoder_model: ONNX encoder model to modify.
        cross_attn_weights: Dictionary of cross-attention K/V weights per layer.
        hidden_state_output_name: Name of the encoder hidden state output.
        num_heads: Number of attention heads.
        head_size: Size of each attention head.
        num_layers: Number of decoder layers.
        rename_input_features: Whether to rename input_features to audio_features.
        onnx_dtype: ONNX tensor data type (TensorProto.FLOAT or TensorProto.FLOAT16).
        np_dtype: NumPy dtype for weight conversion.

    Returns:
        Modified ONNX model with cross-attention KV outputs.
    """
    logger.info(f"Adding cross KV outputs with dtype: {np_dtype}")
    graph = encoder_model.graph

    # Rename input_features to audio_features if requested
    if rename_input_features:
        _rename_input(graph, "input_features", "audio_features")

    # Rename output to encoder_hidden_states
    encoder_hidden_states_name = "encoder_hidden_states"

    # Find the node that produces this output and rename it directly
    for node in graph.node:
        for i, out in enumerate(node.output):
            if out == hidden_state_output_name:
                node.output[i] = encoder_hidden_states_name
                logger.debug(
                    f"  Renamed output: {hidden_state_output_name} -> {encoder_hidden_states_name}"
                )
                break

    # Update the graph output
    for output in list(graph.output):
        if output.name == hidden_state_output_name:
            dims = [
                d.dim_param if d.dim_param else d.dim_value
                for d in output.type.tensor_type.shape.dim
            ]
            new_output = helper.make_tensor_value_info(
                encoder_hidden_states_name,
                output.type.tensor_type.elem_type,
                dims,
            )
            graph.output.remove(output)
            graph.output.append(new_output)
            break

    logger.info(f"Adding cross KV cache outputs for {num_layers} layers")

    new_nodes = []
    new_outputs = []
    new_initializers = []

    # Shape constant for reshape: (batch, seq, num_heads, head_size)
    reshape_shape_name = "cross_kv_reshape_shape"
    reshape_shape = np.array([0, -1, num_heads, head_size], dtype=np.int64)
    new_initializers.append(numpy_helper.from_array(reshape_shape, name=reshape_shape_name))

    for layer_idx in range(num_layers):
        layer_weights = cross_attn_weights[layer_idx]

        k_weight = layer_weights["k_weight"]
        v_weight = layer_weights["v_weight"]
        k_bias = layer_weights["k_bias"]
        v_bias = layer_weights["v_bias"]

        # Transpose weights for ONNX MatMul
        # Use detected model dtype for weight conversion
        k_weight_t = k_weight.T.astype(np_dtype)
        v_weight_t = v_weight.T.astype(np_dtype)

        # Add weight initializers
        k_weight_name = f"encoder.cross_attn_k_weight.{layer_idx}"
        v_weight_name = f"encoder.cross_attn_v_weight.{layer_idx}"

        new_initializers.append(numpy_helper.from_array(k_weight_t, name=k_weight_name))
        new_initializers.append(numpy_helper.from_array(v_weight_t, name=v_weight_name))

        # MatMul: encoder_hidden_states @ k_weight
        k_matmul_out = f"cross_k_matmul_{layer_idx}"
        new_nodes.append(
            helper.make_node(
                "MatMul",
                inputs=[encoder_hidden_states_name, k_weight_name],
                outputs=[k_matmul_out],
                name=f"CrossK_MatMul_{layer_idx}",
            )
        )

        v_matmul_out = f"cross_v_matmul_{layer_idx}"
        new_nodes.append(
            helper.make_node(
                "MatMul",
                inputs=[encoder_hidden_states_name, v_weight_name],
                outputs=[v_matmul_out],
                name=f"CrossV_MatMul_{layer_idx}",
            )
        )

        # Add bias if present
        if k_bias is not None:
            k_bias_name = f"encoder.cross_attn_k_bias.{layer_idx}"
            new_initializers.append(
                numpy_helper.from_array(k_bias.astype(np_dtype), name=k_bias_name)
            )
            k_add_out = f"cross_k_add_{layer_idx}"
            new_nodes.append(
                helper.make_node(
                    "Add",
                    inputs=[k_matmul_out, k_bias_name],
                    outputs=[k_add_out],
                    name=f"CrossK_Add_{layer_idx}",
                )
            )
            k_matmul_out = k_add_out

        if v_bias is not None:
            v_bias_name = f"encoder.cross_attn_v_bias.{layer_idx}"
            new_initializers.append(
                numpy_helper.from_array(v_bias.astype(np_dtype), name=v_bias_name)
            )
            v_add_out = f"cross_v_add_{layer_idx}"
            new_nodes.append(
                helper.make_node(
                    "Add",
                    inputs=[v_matmul_out, v_bias_name],
                    outputs=[v_add_out],
                    name=f"CrossV_Add_{layer_idx}",
                )
            )
            v_matmul_out = v_add_out

        # Reshape: (batch, seq, hidden) -> (batch, seq, num_heads, head_size)
        k_reshape_out = f"cross_k_reshape_{layer_idx}"
        new_nodes.append(
            helper.make_node(
                "Reshape",
                inputs=[k_matmul_out, reshape_shape_name],
                outputs=[k_reshape_out],
                name=f"CrossK_Reshape_{layer_idx}",
            )
        )

        v_reshape_out = f"cross_v_reshape_{layer_idx}"
        new_nodes.append(
            helper.make_node(
                "Reshape",
                inputs=[v_matmul_out, reshape_shape_name],
                outputs=[v_reshape_out],
                name=f"CrossV_Reshape_{layer_idx}",
            )
        )

        # Transpose: (batch, seq, num_heads, head_size) -> (batch, num_heads, seq, head_size)
        k_output_name = f"present_key_cross_{layer_idx}"
        new_nodes.append(
            helper.make_node(
                "Transpose",
                inputs=[k_reshape_out],
                outputs=[k_output_name],
                perm=[0, 2, 1, 3],
                name=f"CrossK_Transpose_{layer_idx}",
            )
        )

        v_output_name = f"present_value_cross_{layer_idx}"
        new_nodes.append(
            helper.make_node(
                "Transpose",
                inputs=[v_reshape_out],
                outputs=[v_output_name],
                perm=[0, 2, 1, 3],
                name=f"CrossV_Transpose_{layer_idx}",
            )
        )

        # Add outputs with shape: (batch_size, num_heads, seq_len, head_size)
        # Use detected model dtype for output tensor type
        k_output = helper.make_tensor_value_info(
            k_output_name,
            onnx_dtype,
            ["batch_size", num_heads, "encoder_sequence_length", head_size],
        )
        v_output = helper.make_tensor_value_info(
            v_output_name,
            onnx_dtype,
            ["batch_size", num_heads, "encoder_sequence_length", head_size],
        )
        new_outputs.append(k_output)
        new_outputs.append(v_output)

    # Add new nodes, initializers, and outputs
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_initializers)
    graph.output.extend(new_outputs)

    return encoder_model


def add_cross_kv_to_encoder(
    encoder_path: str,
    output_path: str,
    hf_model_id: str,
    hidden_state_output_name: str = "last_hidden_state",
    rename_input_features: bool = True,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    decoder_filename: str = "decoder_with_past_model.onnx",
    generate_genai_config: bool = True,
    provider: str = "cuda",
    verbose: bool = True,
    trust_remote_code: bool = False,
) -> onnx.ModelProto:
    """Add cross-attention KV cache outputs to encoder model.

    This function transforms an Optimum-exported encoder model by adding
    cross-attention Key/Value projection outputs. This is required for
    ONNX Runtime GenAI compatibility where the decoder expects pre-computed
    encoder K/V caches.

    The transformation:
    1. Renames input_features -> audio_features (optional)
    2. Renames last_hidden_state -> encoder_hidden_states
    3. Adds K/V projection weights from HuggingFace model
    4. Adds MatMul -> Reshape -> Transpose subgraph for each layer
    5. Adds outputs: present_key_cross_X, present_value_cross_X
    6. Generates genai_config.json and audio_processor_config.json (optional)

    Args:
        encoder_path: Path to encoder ONNX model.
        output_path: Path to save modified encoder.
        hf_model_id: HuggingFace model ID for loading cross-attention weights.
        hidden_state_output_name: Name of encoder hidden state output.
        rename_input_features: Whether to rename input_features to audio_features.
        use_external_data: Whether to save weights as external data.
        external_data_name: Name for external data file.
        decoder_filename: Filename for decoder model in genai_config.json.
            Default is "decoder_with_past_model.onnx".
        generate_genai_config: Whether to generate genai_config.json.
        provider: Execution provider for genai_config.json ("cuda", "cpu", "NvTensorRtRtx").
        verbose: Whether to print progress messages.
        trust_remote_code: Whether to trust remote code in HuggingFace model.

    Returns:
        Modified encoder model with cross-attention KV cache outputs.

    Example:
        >>> from modelopt.onnx.graph_surgery import add_cross_kv_to_encoder
        >>> model = add_cross_kv_to_encoder(
        ...     encoder_path="encoder_model.onnx",
        ...     output_path="encoder_model_with_kv.onnx",
        ...     hf_model_id="openai/whisper-large-v3-turbo",
        ... )
    """
    # Load cross-attention weights from HuggingFace model
    cross_attn_weights, num_heads, head_size, num_layers = _get_cross_attn_weights_from_hf(
        hf_model_id, trust_remote_code=trust_remote_code
    )

    if verbose:
        logger.info(f"Loading encoder model from: {encoder_path}")

    encoder_model = onnx.load(encoder_path, load_external_data=True)

    # Detect model dtype
    onnx_dtype, np_dtype = detect_model_dtype(encoder_model)
    if verbose:
        dtype_names = {
            TensorProto.FLOAT: "FP32",
            TensorProto.FLOAT16: "FP16",
            TensorProto.BFLOAT16: "BF16",
        }
        logger.info(f"Detected model dtype: {dtype_names.get(onnx_dtype, 'unknown')}")

    if verbose:
        logger.info("Adding cross KV cache outputs to encoder...")

    modified_encoder = _add_cross_kv_outputs(
        encoder_model,
        cross_attn_weights,
        hidden_state_output_name,
        num_heads,
        head_size,
        num_layers,
        rename_input_features,
        onnx_dtype=onnx_dtype,
        np_dtype=np_dtype,
    )

    # Save model
    if verbose:
        logger.info(f"Saving modified encoder to: {output_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if use_external_data:
        if external_data_name is None:
            external_data_name = Path(output_path).name.replace(".onnx", ".onnx_data")

        if verbose:
            logger.info(f"  Saving weights to external file: {external_data_name}")

        onnx.save_model(
            modified_encoder,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_name,
            size_threshold=1024,
            convert_attribute=False,
        )
    else:
        onnx.save(modified_encoder, output_path)

    if verbose:
        logger.info("Done!")
        logger.info("\nEncoder inputs:")
        for inp in modified_encoder.graph.input:
            logger.info(f"  {inp.name}")
        logger.info("\nEncoder outputs:")
        for output in modified_encoder.graph.output:
            logger.info(f"  {output.name}")

        logger.info("\n" + "=" * 60)
        logger.info("UPDATE genai_config.json with:")
        logger.info("=" * 60)
        logger.info(
            """
"encoder": {
    "filename": "<your_encoder_filename>.onnx",
    "inputs": {
        "audio_features": "audio_features"
    },
    "outputs": {
        "encoder_hidden_states": "encoder_hidden_states",
        "cross_present_key_names": "present_key_cross_%d",
        "cross_present_value_names": "present_value_cross_%d"
    }
}
"""
        )

    # Generate config files if output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        # Save audio processor config
        from .utils.whisper_utils import save_audio_processor_config

        save_audio_processor_config(
            output_dir,
            hf_model_id=hf_model_id,
            overwrite=False,
            trust_remote_code=trust_remote_code,
        )

        # Generate genai_config.json with encoder pointing to this output
        if generate_genai_config:
            from .utils.whisper_utils import save_genai_config as _save_genai_config

            encoder_filename = os.path.basename(output_path)
            _save_genai_config(
                output_dir=output_dir,
                encoder_filename=encoder_filename,
                decoder_filename=decoder_filename,
                hf_model_id=hf_model_id,
                provider=provider,
                trust_remote_code=trust_remote_code,
                overwrite=False,  # Don't overwrite if exists
            )

    return modified_encoder
