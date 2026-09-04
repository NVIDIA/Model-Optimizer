# Adapted https://github.com/huggingface/optimum/blob/15a162824d0c5d8aa7a3d14ab6e9bb07e5732fb6/optimum/exporters/onnx/convert.py#L573-L614

# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

import os
import shutil
import tempfile
import uuid
from contextlib import nullcontext, suppress
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from diffusers.models.transformers import (
    FluxTransformer2DModel,
    SD3Transformer2DModel,
    WanTransformer3DModel,
)
from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel
from diffusers.models.unets import UNet2DConditionModel
from torch.onnx import export as onnx_export

from modelopt.onnx.export import NVFP4QuantExporter
from modelopt.onnx.quantization.graph_utils import get_tensor_consumer_nodes
from modelopt.torch.quantization.export_onnx import configure_linear_module_onnx_quantizers
from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear
from modelopt.torch.utils import torch_to

from .fp8_onnx_graphsurgeon import convert_zp_fp8

MODEL_ID_TO_DYNAMIC_AXES = {
    "sdxl-1.0": {
        "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        "timestep": {0: "steps"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        "text_embeds": {0: "batch_size"},
        "time_ids": {0: "batch_size"},
        "latent": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
    },
    "sdxl-turbo": {
        "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        "timestep": {0: "steps"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        "text_embeds": {0: "batch_size"},
        "time_ids": {0: "batch_size"},
        "latent": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
    },
    "sd3-medium": {
        "hidden_states": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        "timestep": {0: "steps"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        "pooled_projections": {0: "batch_size"},
        "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
    },
    "sd3.5-medium": {
        "hidden_states": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        "timestep": {0: "steps"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        "pooled_projections": {0: "batch_size"},
        "out_hidden_states": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
    },
    "flux-dev": {
        "hidden_states": {0: "batch_size", 1: "latent_dim"},
        "encoder_hidden_states": {0: "batch_size"},
        "pooled_projections": {0: "batch_size"},
        "timestep": {0: "batch_size"},
        "img_ids": {0: "latent_dim"},
        "guidance": {0: "batch_size"},
        "latent": {0: "batch_size"},
    },
    "flux-schnell": {
        "hidden_states": {0: "batch_size", 1: "latent_dim"},
        "encoder_hidden_states": {0: "batch_size"},
        "pooled_projections": {0: "batch_size"},
        "timestep": {0: "batch_size"},
        "img_ids": {0: "latent_dim"},
        "latent": {0: "batch_size"},
    },
    "ltx-video-dev": {
        "hidden_states": {0: "batch_size", 1: "latent_dim"},
        "encoder_hidden_states": {0: "batch_size"},
        "timestep": {0: "batch_size"},
        "encoder_attention_mask": {0: "batch_size"},
        "video_coords": {0: "batch_size", 2: "latent_dim"},
    },
    "wan2.2-t2v-14b": {
        "hidden_states": {0: "batch_size", 2: "frame_num", 3: "height", 4: "width"},
        "encoder_hidden_states": {0: "batch_size"},
        "timestep": {0: "batch_size"},
    },
}


def flux_convert_rope_weight_type(onnx_graph):
    graph = gs.import_onnx(onnx_graph)
    for node in graph.nodes:
        if node.op == "Einsum":
            node.inputs[1].dtype = "float32"
    return gs.export_onnx(graph)


def generate_fp8_scales(backbone, *, conv_only=False):
    # temporary solution due to a known bug in torch.onnx._dynamo_export
    module_types = (torch.nn.Conv2d,) if conv_only else (torch.nn.Linear, torch.nn.Conv2d)
    quantizer_states = []
    try:
        for _, module in backbone.named_modules():
            if not isinstance(module, module_types):
                continue
            for quantizer_name in ("input_quantizer", "weight_quantizer"):
                quantizer = getattr(module, quantizer_name, None)
                if (
                    quantizer is None
                    or not quantizer.is_enabled
                    or quantizer.num_bits != (4, 3)
                    or getattr(quantizer, "_amax", None) is None
                ):
                    continue
                quantizer_states.append((quantizer, quantizer._num_bits, quantizer._amax))
                quantizer._num_bits = 8
                quantizer._amax = quantizer._amax * (127 / 448.0)
    except BaseException:
        restore_fp8_scales(quantizer_states)
        raise
    return quantizer_states


def restore_fp8_scales(quantizer_states):
    for quantizer, num_bits, amax in reversed(quantizer_states):
        quantizer._num_bits = num_bits
        quantizer._amax = amax


def _gen_dummy_inp_and_dyn_shapes_sdxl(backbone, min_bs=1, opt_bs=1):
    assert isinstance(backbone, UNet2DConditionModel) or isinstance(
        backbone._orig_mod, UNet2DConditionModel
    )
    cfg = backbone.config
    assert cfg.addition_embed_type == "text_time"

    dynamic_shapes = {
        "sample": {
            "min": [min_bs, cfg.in_channels, cfg.sample_size, cfg.sample_size],
            "opt": [opt_bs, cfg.in_channels, cfg.sample_size, cfg.sample_size],
        },
        "timestep": {"min": [1], "opt": [1]},
        "encoder_hidden_states": {
            "min": [min_bs, 77, cfg.cross_attention_dim],
            "opt": [opt_bs, 77, cfg.cross_attention_dim],
        },
        "added_cond_kwargs.text_embeds": {
            "min": [
                min_bs,
                backbone.add_embedding.linear_1.in_features
                - 6 * backbone.add_time_proj.num_channels,
            ],
            "opt": [
                opt_bs,
                backbone.add_embedding.linear_1.in_features
                - 6 * backbone.add_time_proj.num_channels,
            ],
        },
        "added_cond_kwargs.time_ids": {"min": [min_bs, 6], "opt": [opt_bs, 6]},
    }

    dummy_kwargs = {
        "sample": torch.randn(*dynamic_shapes["sample"]["min"]),
        "timestep": torch.ones(1),
        "encoder_hidden_states": torch.randn(*dynamic_shapes["encoder_hidden_states"]["min"]),
        "added_cond_kwargs": {
            "text_embeds": torch.randn(*dynamic_shapes["added_cond_kwargs.text_embeds"]["min"]),
            "time_ids": torch.randn(*dynamic_shapes["added_cond_kwargs.time_ids"]["min"]),
        },
        "return_dict": False,
    }
    dummy_kwargs = torch_to(dummy_kwargs, dtype=backbone.dtype)

    return dummy_kwargs, dynamic_shapes


def _gen_dummy_inp_and_dyn_shapes_sd3(backbone, min_bs=1, opt_bs=1):
    assert isinstance(backbone, SD3Transformer2DModel) or isinstance(
        backbone._orig_mod, SD3Transformer2DModel
    )
    cfg = backbone.config

    dynamic_shapes = {
        "hidden_states": {
            "min": [min_bs, cfg.in_channels, cfg.sample_size, cfg.sample_size],
            "opt": [opt_bs, cfg.in_channels, cfg.sample_size, cfg.sample_size],
        },
        "timestep": {"min": [2], "opt": [16]},
        "encoder_hidden_states": {
            "min": [min_bs, 333, cfg.joint_attention_dim],
            "opt": [opt_bs, 333, cfg.joint_attention_dim],
        },
        "pooled_projections": {
            "min": [min_bs, cfg.pooled_projection_dim],
            "opt": [opt_bs, cfg.pooled_projection_dim],
        },
    }

    dummy_kwargs = {
        "hidden_states": torch.randn(*dynamic_shapes["hidden_states"]["min"]),
        "timestep": torch.ones(1),
        "encoder_hidden_states": torch.randn(*dynamic_shapes["encoder_hidden_states"]["min"]),
        "pooled_projections": torch.randn(*dynamic_shapes["pooled_projections"]["min"]),
        "return_dict": False,
    }
    dummy_kwargs = torch_to(dummy_kwargs, dtype=backbone.dtype)

    return dummy_kwargs, dynamic_shapes


def _gen_dummy_inp_and_dyn_shapes_flux(backbone, min_bs=1, opt_bs=1):
    assert isinstance(backbone, FluxTransformer2DModel) or isinstance(
        backbone._orig_mod, FluxTransformer2DModel
    )
    cfg = backbone.config
    text_maxlen = 512
    img_dim = 4096

    dynamic_shapes = {
        "hidden_states": {
            "min": [min_bs, img_dim, cfg.in_channels],
            "opt": [opt_bs, img_dim, cfg.in_channels],
        },
        "encoder_hidden_states": {
            "min": [min_bs, text_maxlen, cfg.joint_attention_dim],
            "opt": [opt_bs, text_maxlen, cfg.joint_attention_dim],
        },
        "pooled_projections": {
            "min": [min_bs, cfg.pooled_projection_dim],
            "opt": [opt_bs, cfg.pooled_projection_dim],
        },
        "timestep": {"min": [1], "opt": [1]},
        "img_ids": {"min": [img_dim, 3], "opt": [img_dim, 3]},
        "txt_ids": {"min": [text_maxlen, 3], "opt": [text_maxlen, 3]},
    }
    if cfg.guidance_embeds:  # flux-dev
        dynamic_shapes["guidance"] = {"min": [1], "opt": [1]}

    dtype = backbone.dtype
    dummy_kwargs = {
        "hidden_states": torch.randn(*dynamic_shapes["hidden_states"]["min"], dtype=dtype),
        "encoder_hidden_states": torch.randn(
            *dynamic_shapes["encoder_hidden_states"]["min"], dtype=dtype
        ),
        "pooled_projections": torch.randn(
            *dynamic_shapes["pooled_projections"]["min"], dtype=dtype
        ),
        "timestep": torch.ones(1, dtype=dtype),
        "img_ids": torch.randn(*dynamic_shapes["img_ids"]["min"], dtype=torch.float32),
        "txt_ids": torch.randn(*dynamic_shapes["txt_ids"]["min"], dtype=torch.float32),
        "return_dict": False,
    }
    if cfg.guidance_embeds:  # flux-dev
        dummy_kwargs["guidance"] = torch.full((1,), 3.5, dtype=torch.float32)

    return dummy_kwargs, dynamic_shapes


def _gen_dummy_inp_and_dyn_shapes_ltx(backbone, min_bs=2, opt_bs=2):
    assert isinstance(backbone, LTXVideoTransformer3DModel) or isinstance(
        backbone._orig_mod, LTXVideoTransformer3DModel
    )
    cfg = backbone.config
    dtype = backbone.dtype
    video_dim = 2240
    dynamic_shapes = {
        "hidden_states": {
            "min": [min_bs, 720, cfg.in_channels],
            "opt": [opt_bs, video_dim, cfg.in_channels],
        },
        "encoder_hidden_states": {
            "min": [min_bs, 256, cfg.cross_attention_dim],
            "opt": [opt_bs, 256, cfg.cross_attention_dim],
        },
        "timestep": {"min": [min_bs, 1], "opt": [opt_bs, 1]},
        "encoder_attention_mask": {
            "min": [min_bs, 256],
            "opt": [opt_bs, 256],
        },
        "video_coords": {
            "min": [min_bs, 3, 720],
            "opt": [opt_bs, 3, video_dim],
        },
    }
    dummy_kwargs = {
        "hidden_states": torch.randn(*dynamic_shapes["hidden_states"]["min"], dtype=dtype),
        "encoder_hidden_states": torch.randn(
            *dynamic_shapes["encoder_hidden_states"]["min"], dtype=dtype
        ),
        "timestep": torch.ones(*dynamic_shapes["timestep"]["min"], dtype=dtype),
        "encoder_attention_mask": torch.randn(
            *dynamic_shapes["encoder_attention_mask"]["min"], dtype=dtype
        ),
        "video_coords": torch.randn(*dynamic_shapes["video_coords"]["min"], dtype=dtype),
    }

    return dummy_kwargs, dynamic_shapes


def _gen_dummy_inp_and_dyn_shapes_wan(backbone, min_bs=1, opt_bs=2):
    assert isinstance(backbone, WanTransformer3DModel)
    dtype = backbone.dtype

    channels = 16  # latent channels from VAE
    hidden_size = 4096  # text encoder hidden size (UMT5-XXL)

    # num of frames for wan is 4*n+1, as from the official codebase:
    # https://github.com/Wan-Video/Wan2.2/blob/e9783574ef77be11fcab9aa5607905402538c08d/generate.py#L126
    # picking n == 1 as min, n = 20 as opt as 81 is the default num of frames in their code base
    min_num_frames = 4 * 1 + 1
    opt_num_frames = 4 * 20 + 1

    # height and width configs are from their codebase:
    # https://github.com/Wan-Video/Wan2.2/blob/e9783574ef77be11fcab9aa5607905402538c08d/wan/configs/__init__.py#L21
    min_height = 480
    min_width = 480

    # height max can be 1280, but opt setting is 1280x720, so use 720 here
    opt_height = 720
    opt_width = 1280

    min_latent_height = min_height // 8
    min_latent_width = min_width // 8
    opt_latent_height = opt_height // 8
    opt_latent_width = opt_width // 8

    dynamic_shapes = {
        "hidden_states": {
            "min": [min_bs, channels, min_num_frames, min_latent_height, min_latent_width],
            "opt": [opt_bs, channels, opt_num_frames, opt_latent_height, opt_latent_width],
        },
        "encoder_hidden_states": {
            "min": [min_bs, 512, hidden_size],
            "opt": [opt_bs, 512, hidden_size],
        },
        "timestep": {"min": [min_bs], "opt": [opt_bs]},
    }

    dummy_kwargs = {
        "hidden_states": torch.randn(*dynamic_shapes["hidden_states"]["min"], dtype=dtype),
        "encoder_hidden_states": torch.randn(
            *dynamic_shapes["encoder_hidden_states"]["min"], dtype=dtype
        ),
        "timestep": torch.ones(*dynamic_shapes["timestep"]["min"], dtype=dtype),
    }
    return dummy_kwargs, dynamic_shapes


def update_dynamic_axes(model_id, dynamic_axes):
    if model_id in ["flux-dev", "flux-schnell"]:
        dynamic_axes["out.0"] = dynamic_axes.pop("latent")
    elif model_id in ["sdxl-1.0", "sdxl-turbo"]:
        dynamic_axes["added_cond_kwargs.text_embeds"] = dynamic_axes.pop("text_embeds")
        dynamic_axes["added_cond_kwargs.time_ids"] = dynamic_axes.pop("time_ids")
        dynamic_axes["out.0"] = dynamic_axes.pop("latent")
    elif model_id == "sd3-medium":
        dynamic_axes["out.0"] = dynamic_axes.pop("sample")
    elif model_id == "sd3.5-medium":
        dynamic_axes["out.0"] = dynamic_axes.pop("out_hidden_states")


def _create_trt_dynamic_shapes(dynamic_shapes):
    min_shapes = {}
    opt_shapes = {}
    for key, value in dynamic_shapes.items():
        min_shapes[key] = value["min"]
        opt_shapes[key] = value["opt"]
    return {
        "minShapes": min_shapes,
        "optShapes": opt_shapes,
        "maxShapes": opt_shapes,
    }


def generate_dummy_kwargs_and_dynamic_axes_and_shapes(model_id, backbone):
    """Generate dummy inputs, dynamic axes, and dynamic shapes for the given model."""
    if model_id in ["sdxl-1.0", "sdxl-turbo"]:
        dummy_kwargs, dynamic_shapes = _gen_dummy_inp_and_dyn_shapes_sdxl(
            backbone, min_bs=2, opt_bs=16
        )
    elif model_id in ["sd3-medium", "sd3.5-medium"]:
        dummy_kwargs, dynamic_shapes = _gen_dummy_inp_and_dyn_shapes_sd3(
            backbone, min_bs=2, opt_bs=16
        )
    elif model_id in ["flux-dev", "flux-schnell"]:
        dummy_kwargs, dynamic_shapes = _gen_dummy_inp_and_dyn_shapes_flux(
            backbone, min_bs=1, opt_bs=1
        )
    elif model_id == "ltx-video-dev":
        dummy_kwargs, dynamic_shapes = _gen_dummy_inp_and_dyn_shapes_ltx(
            backbone, min_bs=2, opt_bs=2
        )
    elif model_id == "wan2.2-t2v-14b":
        dummy_kwargs, dynamic_shapes = _gen_dummy_inp_and_dyn_shapes_wan(
            backbone, min_bs=1, opt_bs=2
        )
    else:
        raise NotImplementedError(f"Unsupported model_id: {model_id}")

    dummy_kwargs = torch_to(dummy_kwargs, device=backbone.device)
    dynamic_axes = MODEL_ID_TO_DYNAMIC_AXES[model_id]

    return dummy_kwargs, dynamic_axes, dynamic_shapes


def get_io_shapes(model_id, onnx_load_path, trt_dynamic_shapes):
    output_name = "out.0"
    if onnx_load_path != "":
        if model_id in ["sdxl-1.0", "sdxl-turbo"]:
            output_name = "latent"
        elif model_id == "sd3-medium":
            output_name = "sample"
        elif model_id == "sd3.5-medium":
            output_name = "out_hidden_states"
        elif model_id in ["flux-dev", "flux-schnell"]:
            output_name = "output"
        else:
            raise NotImplementedError(f"Unsupported model_id: {model_id}")

    if model_id in ["sdxl-1.0", "sdxl-turbo"]:
        io_shapes = {output_name: trt_dynamic_shapes["minShapes"]["sample"]}
    elif model_id in ["sd3-medium", "sd3.5-medium"]:
        io_shapes = {output_name: trt_dynamic_shapes["minShapes"]["hidden_states"]}
    elif model_id in ["flux-dev", "flux-schnell"]:
        io_shapes = {}

    return io_shapes


def remove_nesting(trt_dynamic_shapes):
    trt_dynamic_shapes["minShapes"]["text_embeds"] = trt_dynamic_shapes["minShapes"].pop(
        "added_cond_kwargs.text_embeds"
    )
    trt_dynamic_shapes["minShapes"]["time_ids"] = trt_dynamic_shapes["minShapes"].pop(
        "added_cond_kwargs.time_ids"
    )
    trt_dynamic_shapes["optShapes"]["text_embeds"] = trt_dynamic_shapes["optShapes"].pop(
        "added_cond_kwargs.text_embeds"
    )
    trt_dynamic_shapes["optShapes"]["time_ids"] = trt_dynamic_shapes["optShapes"].pop(
        "added_cond_kwargs.time_ids"
    )


def _get_int_attribute(node, name):
    for attribute in node.attribute:
        if attribute.name == name and attribute.type == onnx.AttributeProto.INT:
            return attribute.i
    return None


_WEIGHT_PASSTHROUGH_OPS = {"Cast", "Flatten", "Identity", "Reshape", "Transpose"}


def _trace_initializer_source(tensor_name, producers, initializer_names):
    visited = set()
    while tensor_name not in initializer_names:
        if tensor_name in visited:
            return None
        visited.add(tensor_name)
        producer = producers.get(tensor_name)
        if (
            producer is None
            or producer.op_type not in _WEIGHT_PASSTHROUGH_OPS
            or not producer.input
        ):
            return None
        tensor_name = producer.input[0]
    return tensor_name


def _trace_tensor_consumers(
    tensor_names,
    consumers,
    graph_outputs,
    terminal_ops,
    terminal_input_index,
    passthrough_ops=_WEIGHT_PASSTHROUGH_OPS,
    allowed_cast_dtypes=None,
):
    pending = list(tensor_names)
    visited = set()
    terminal_consumers = []
    invalid_consumers = set()

    while pending:
        tensor_name = pending.pop()
        if tensor_name in visited:
            continue
        visited.add(tensor_name)

        if tensor_name in graph_outputs:
            invalid_consumers.add(f"graph output {tensor_name}")
        tensor_consumers = consumers.get(tensor_name, [])
        if not tensor_consumers:
            invalid_consumers.add(f"unused tensor {tensor_name}")
            continue

        for consumer in tensor_consumers:
            input_indices = [
                index
                for index, input_name in enumerate(consumer.input)
                if input_name == tensor_name
            ]
            if consumer.op_type in passthrough_ops and input_indices == [0]:
                if (
                    consumer.op_type == "Cast"
                    and allowed_cast_dtypes is not None
                    and _get_int_attribute(consumer, "to") not in allowed_cast_dtypes
                ):
                    invalid_consumers.add(consumer.name or consumer.op_type)
                    continue
                if consumer.output:
                    pending.extend(consumer.output)
                else:
                    invalid_consumers.add(consumer.name or consumer.op_type)
            elif consumer.op_type in terminal_ops and input_indices == [terminal_input_index]:
                terminal_consumers.append(consumer)
            else:
                invalid_consumers.add(
                    consumer.name or (consumer.output[0] if consumer.output else consumer.op_type)
                )

    return terminal_consumers, sorted(invalid_consumers)


def _trace_source_node(tensor_name, producers):
    visited = set()
    while tensor_name not in visited:
        visited.add(tensor_name)
        producer = producers.get(tensor_name)
        if producer is None or producer.op_type not in _WEIGHT_PASSTHROUGH_OPS:
            return producer
        if not producer.input:
            return None
        tensor_name = producer.input[0]
    return None


def _find_initializer_backed_qdq_weights(onnx_model, allow_fp8_conv):
    initializer_names = {initializer.name for initializer in onnx_model.graph.initializer}
    producers = {output: node for node in onnx_model.graph.node for output in node.output if output}
    consumers = get_tensor_consumer_nodes(onnx_model.graph)
    graph_outputs = {output.name for output in onnx_model.graph.output}
    disallowed_consumers = set()
    allowed_pairs = []

    for node in onnx_model.graph.node:
        if node.op_type != "DequantizeLinear" or not node.input:
            continue
        quantize_node = producers.get(node.input[0])
        if (
            quantize_node is None
            or quantize_node.op_type != "QuantizeLinear"
            or not quantize_node.input
            or _trace_initializer_source(quantize_node.input[0], producers, initializer_names)
            is None
        ):
            continue

        terminal_consumers, invalid_consumers = _trace_tensor_consumers(
            node.output, consumers, graph_outputs, {"Conv"}, 1
        )
        if allow_fp8_conv and terminal_consumers and not invalid_consumers:
            allowed_pairs.append((quantize_node, node, terminal_consumers))
        else:
            disallowed_consumers.update(invalid_consumers)
            disallowed_consumers.update(
                consumer.name or (consumer.output[0] if consumer.output else consumer.op_type)
                for consumer in terminal_consumers
            )
            if not terminal_consumers and not invalid_consumers:
                disallowed_consumers.add(node.name or node.output[0])

    return allowed_pairs, sorted(disallowed_consumers)


def _get_tensor_dtype(tensor_name, initializers, producers):
    initializer = initializers.get(tensor_name)
    if initializer is not None:
        return initializer.data_type
    producer = producers.get(tensor_name)
    if producer is None or producer.op_type != "Constant":
        return None
    for attribute in producer.attribute:
        if attribute.name == "value" and attribute.type == onnx.AttributeProto.TENSOR:
            return attribute.t.data_type
    return None


def _get_effective_tensor_dtype(tensor_name, initializers, producers, declared_dtypes):
    visited = set()
    while tensor_name not in visited:
        visited.add(tensor_name)
        producer = producers.get(tensor_name)
        if producer is not None and producer.op_type == "Cast":
            return _get_int_attribute(producer, "to")
        if tensor_name in declared_dtypes:
            return declared_dtypes[tensor_name]
        tensor_dtype = _get_tensor_dtype(tensor_name, initializers, producers)
        if tensor_dtype is not None:
            return tensor_dtype
        if producer is None or producer.op_type not in _WEIGHT_PASSTHROUGH_OPS:
            return None
        if not producer.input:
            return None
        tensor_name = producer.input[0]
    return None


def _get_constant_array(tensor_name, initializers, producers):
    tensor = initializers.get(tensor_name)
    if tensor is None:
        producer = producers.get(tensor_name)
        if producer is not None and producer.op_type == "Constant":
            tensor = next(
                (
                    attribute.t
                    for attribute in producer.attribute
                    if attribute.name == "value" and attribute.type == onnx.AttributeProto.TENSOR
                ),
                None,
            )
    if tensor is None:
        return None
    try:
        return onnx.numpy_helper.to_array(tensor)
    except (TypeError, ValueError):
        return None


def _validate_positive_scalar(tensor_name, role, initializers, producers):
    value = _get_constant_array(tensor_name, initializers, producers)
    if value is None or value.size != 1 or not np.isfinite(value).all() or not (value > 0).all():
        return [f"{role} must be a finite positive scalar constant"]
    return []


def _validate_normalized_fp8_qdq_pair(
    quantize_node,
    dequantize_node,
    pair_name,
    initializers,
    producers,
    consumers,
    expected_scale_dtype=None,
):
    errors = []
    if len(quantize_node.input) != 3 or len(dequantize_node.input) != 3:
        return [f"{pair_name} must use three-input FP8 Q/DQ nodes"]
    if len(quantize_node.output) != 1 or dequantize_node.input[0] != quantize_node.output[0]:
        errors.append(f"{pair_name} does not form a direct Q/DQ pair")
    elif consumers.get(quantize_node.output[0], []) != [dequantize_node]:
        errors.append(f"{pair_name} quantized tensor must be consumed only by its DQ")

    if quantize_node.input[1:] != dequantize_node.input[1:]:
        errors.append(f"{pair_name} Q/DQ nodes do not share scale and zero point")
    if any(
        attribute.name == "axis"
        for node in (quantize_node, dequantize_node)
        for attribute in node.attribute
    ):
        errors.append(f"{pair_name} must use per-tensor FP8 Q/DQ without an axis")

    errors.extend(
        _validate_positive_scalar(
            quantize_node.input[1], f"{pair_name} FP8 scale", initializers, producers
        )
    )
    scale_dtype = _get_tensor_dtype(quantize_node.input[1], initializers, producers)
    if scale_dtype not in {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.BFLOAT16,
    }:
        errors.append(f"{pair_name} FP8 scale must use a floating-point dtype")
    elif expected_scale_dtype is not None and scale_dtype != expected_scale_dtype:
        errors.append(f"{pair_name} FP8 scale dtype does not match its quantized tensor")
    for role, node in (
        ("QuantizeLinear", quantize_node),
        ("DequantizeLinear", dequantize_node),
    ):
        zero_name = node.input[2]
        if _get_tensor_dtype(zero_name, initializers, producers) != onnx.TensorProto.FLOAT8E4M3FN:
            errors.append(f"{pair_name} {role} zero point is not FLOAT8E4M3FN")
            continue
        zero = _get_constant_array(zero_name, initializers, producers)
        if zero is None or zero.size != 1 or not (zero == 0).all():
            errors.append(f"{pair_name} {role} zero point must be a scalar zero")
    return errors


def _validate_normalized_fp8_qdq(onnx_model, qdq_records):
    initializers = {initializer.name: initializer for initializer in onnx_model.graph.initializer}
    producers = {output: node for node in onnx_model.graph.node for output in node.output if output}
    consumers = get_tensor_consumer_nodes(onnx_model.graph)
    graph_outputs = {output.name for output in onnx_model.graph.output}
    initializer_names = set(initializers)
    declared_dtypes = {
        value.name: value.type.tensor_type.elem_type
        for values in (
            onnx_model.graph.input,
            onnx_model.graph.value_info,
            onnx_model.graph.output,
        )
        for value in values
        if value.type.HasField("tensor_type")
    }
    errors = []
    fp8_conv_ids = {
        id(consumer) for _, _, terminal_consumers in qdq_records for consumer in terminal_consumers
    }
    validated_activation_dq_ids = set()

    for quantize_node, dequantize_node, terminal_consumers in qdq_records:
        pair_name = dequantize_node.name or dequantize_node.output[0]
        if len(terminal_consumers) != 1:
            errors.append(f"{pair_name} weight DQ must feed exactly one FP8 Conv input 1")
        errors.extend(
            _validate_normalized_fp8_qdq_pair(
                quantize_node,
                dequantize_node,
                pair_name,
                initializers,
                producers,
                consumers,
                _get_effective_tensor_dtype(
                    quantize_node.input[0], initializers, producers, declared_dtypes
                ),
            )
        )

        for conv_node in terminal_consumers:
            conv_name = conv_node.name or conv_node.output[0]
            activation_dq = _trace_source_node(conv_node.input[0], producers)
            if activation_dq is None or activation_dq.op_type != "DequantizeLinear":
                errors.append(f"{conv_name} has no FP8 activation Q/DQ on input 0")
                continue
            activation_q = producers.get(activation_dq.input[0]) if activation_dq.input else None
            if activation_q is None or activation_q.op_type != "QuantizeLinear":
                errors.append(f"{conv_name} has no FP8 activation QuantizeLinear on input 0")
                continue
            if (
                not activation_q.input
                or _trace_initializer_source(activation_q.input[0], producers, initializer_names)
                is not None
            ):
                errors.append(f"{conv_name} activation Q/DQ is initializer-backed")
                continue

            if id(activation_dq) not in validated_activation_dq_ids:
                activation_name = activation_dq.name or activation_dq.output[0]
                errors.extend(
                    _validate_normalized_fp8_qdq_pair(
                        activation_q,
                        activation_dq,
                        activation_name,
                        initializers,
                        producers,
                        consumers,
                        _get_effective_tensor_dtype(
                            activation_q.input[0], initializers, producers, declared_dtypes
                        ),
                    )
                )
                activation_consumers, invalid_consumers = _trace_tensor_consumers(
                    activation_dq.output, consumers, graph_outputs, {"Conv"}, 0
                )
                if invalid_consumers:
                    errors.append(
                        f"{activation_name} has non-Conv activation consumers: "
                        + ", ".join(invalid_consumers[:5])
                    )
                if len(activation_consumers) != 1 or id(conv_node) not in {
                    id(consumer) for consumer in activation_consumers
                }:
                    errors.append(
                        f"{activation_name} must feed exactly one validated FP8 Conv input 0"
                    )
                unexpected_conv_ids = {
                    id(consumer) for consumer in activation_consumers
                } - fp8_conv_ids
                if unexpected_conv_ids:
                    errors.append(
                        f"{activation_name} reaches a Conv without a validated FP8 weight Q/DQ"
                    )
                validated_activation_dq_ids.add(id(activation_dq))
    return errors


def _validate_dynamic_fp4_activations(
    onnx_model, expected_block_size, fp4_weight_terminal_consumers
):
    initializers = {initializer.name: initializer for initializer in onnx_model.graph.initializer}
    producers = {output: node for node in onnx_model.graph.node for output in node.output if output}
    consumers = get_tensor_consumer_nodes(onnx_model.graph)
    graph_outputs = {output.name for output in onnx_model.graph.output}
    expected_terminal_ids = {id(node) for node in fp4_weight_terminal_consumers}
    dynamic_terminal_counts = {}
    initializer_or_constant_names = set(initializers) | {
        output
        for node in onnx_model.graph.node
        if node.op_type == "Constant"
        for output in node.output
    }
    errors = []
    dynamic_nodes = [
        node for node in onnx_model.graph.node if node.op_type == "TRT_FP4DynamicQuantize"
    ]

    if not dynamic_nodes:
        errors.append("no TRT_FP4DynamicQuantize activation nodes were exported")

    for node in dynamic_nodes:
        node_name = node.name or (node.output[0] if node.output else "<unnamed>")
        if len(node.input) != 2 or len(node.output) != 2:
            errors.append(f"{node_name} must have two inputs and two outputs")
            continue
        if node.domain != "trt":
            errors.append(f"{node_name} must use the trt domain")
        if (
            _trace_initializer_source(node.input[0], producers, initializer_or_constant_names)
            is not None
        ):
            errors.append(f"{node_name} input 0 must be a dynamic activation")
        if _get_int_attribute(node, "block_size") != expected_block_size:
            errors.append(f"{node_name} does not use block_size={expected_block_size}")
        if _get_int_attribute(node, "axis") != -1:
            errors.append(f"{node_name} does not use axis=-1")
        if _get_int_attribute(node, "scale_type") != onnx.TensorProto.FLOAT8E4M3FN:
            errors.append(f"{node_name} does not produce FLOAT8E4M3FN block scales")
        if _get_tensor_dtype(node.input[1], initializers, producers) != onnx.TensorProto.FLOAT:
            errors.append(f"{node_name} global scale is not FLOAT")
        errors.extend(
            _validate_positive_scalar(
                node.input[1], f"{node_name} global scale", initializers, producers
            )
        )

        quantized_consumers = consumers.get(node.output[0], [])
        if (
            len(quantized_consumers) != 1
            or quantized_consumers[0].op_type != "DequantizeLinear"
            or not quantized_consumers[0].input
            or quantized_consumers[0].input[0] != node.output[0]
        ):
            errors.append(f"{node_name} FP4 output must feed exactly one DequantizeLinear")
            continue
        activation_dq = quantized_consumers[0]
        activation_dq_name = activation_dq.name or activation_dq.output[0]
        if len(activation_dq.input) != 2:
            errors.append(f"{activation_dq_name} must be a two-input DequantizeLinear")
            continue
        if _get_int_attribute(activation_dq, "block_size") != expected_block_size:
            errors.append(f"{activation_dq_name} does not use block_size={expected_block_size}")
        if _get_int_attribute(activation_dq, "axis") != -1:
            errors.append(f"{activation_dq_name} does not use axis=-1")

        scale_consumers = consumers.get(node.output[1], [])
        if (
            len(scale_consumers) != 1
            or scale_consumers[0].op_type != "DequantizeLinear"
            or not scale_consumers[0].input
            or scale_consumers[0].input[0] != node.output[1]
        ):
            errors.append(f"{node_name} FP8 scale output must feed exactly one DequantizeLinear")
            continue
        scale_dq = scale_consumers[0]
        scale_dq_name = scale_dq.name or scale_dq.output[0]
        if len(scale_dq.input) != 2:
            errors.append(f"{scale_dq_name} must be a two-input DequantizeLinear")
            continue
        if any(attribute.name in {"axis", "block_size"} for attribute in scale_dq.attribute):
            errors.append(f"{scale_dq_name} must not use axis or block_size")
        if _get_tensor_dtype(scale_dq.input[1], initializers, producers) != onnx.TensorProto.FLOAT:
            errors.append(f"{scale_dq_name} global scale is not FLOAT")
        errors.extend(
            _validate_positive_scalar(
                scale_dq.input[1], f"{scale_dq_name} global scale", initializers, producers
            )
        )
        quantize_scale = _get_constant_array(node.input[1], initializers, producers)
        dequantize_scale = _get_constant_array(scale_dq.input[1], initializers, producers)
        if (
            quantize_scale is not None
            and dequantize_scale is not None
            and not np.array_equal(quantize_scale, dequantize_scale)
        ):
            errors.append(f"{node_name} quantize and dequantize global scales do not match")
        if not scale_dq.output or activation_dq.input[1] != scale_dq.output[0]:
            errors.append(f"{activation_dq_name} is not scaled by {scale_dq_name}")
        elif consumers.get(scale_dq.output[0], []) != [activation_dq]:
            errors.append(f"{scale_dq_name} output must be consumed only by {activation_dq_name}")

        terminal_consumers, invalid_consumers = _trace_tensor_consumers(
            activation_dq.output,
            consumers,
            graph_outputs,
            {"Gemm", "MatMul"},
            0,
            {"Cast", "Identity"},
            {onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16},
        )
        for consumer in terminal_consumers:
            terminal_id = id(consumer)
            dynamic_terminal_counts[terminal_id] = dynamic_terminal_counts.get(terminal_id, 0) + 1
        if not terminal_consumers:
            errors.append(f"{activation_dq_name} does not reach a Gemm/MatMul activation input")
        elif len(terminal_consumers) != 1:
            errors.append(
                f"{activation_dq_name} must feed exactly one Gemm/MatMul activation input"
            )
        if invalid_consumers:
            errors.append(
                f"{activation_dq_name} has non-activation consumers: "
                + ", ".join(invalid_consumers[:5])
            )

    dynamic_terminal_ids = set(dynamic_terminal_counts)
    duplicate = sum(count != 1 for count in dynamic_terminal_counts.values())
    if dynamic_terminal_ids != expected_terminal_ids or duplicate:
        missing = len(expected_terminal_ids - dynamic_terminal_ids)
        extra = len(dynamic_terminal_ids - expected_terminal_ids)
        errors.append(
            "dynamic NVFP4 activation paths do not match FLOAT4 weight consumers "
            f"(missing={missing}, extra={extra}, duplicate={duplicate})"
        )
    return errors


def _validate_raw_fp4_graph(
    onnx_model,
    expected_block_size=16,
    *,
    allow_fp8_conv=False,
    expected_linear_count=None,
    expected_fp8_conv_count=None,
):
    initializer_names = {initializer.name for initializer in onnx_model.graph.initializer}
    consumers = get_tensor_consumer_nodes(onnx_model.graph)
    graph_outputs = {output.name for output in onnx_model.graph.output}
    fp4_nodes = [node for node in onnx_model.graph.node if node.op_type == "TRT_FP4QDQ"]
    errors = []

    if not fp4_nodes:
        errors.append("no TRT_FP4QDQ weight markers were exported")
    if expected_linear_count is not None and len(fp4_nodes) != expected_linear_count:
        errors.append(
            f"found {len(fp4_nodes)} TRT_FP4QDQ weight markers, expected "
            f"{expected_linear_count} enabled Linear pairs"
        )

    for node in fp4_nodes:
        node_name = node.name or (node.output[0] if node.output else "<unnamed>")
        if not node.input or node.input[0] not in initializer_names:
            errors.append(f"{node_name} is not backed by a weight initializer")
        block_size = _get_int_attribute(node, "block_size")
        if block_size != expected_block_size:
            errors.append(
                f"{node_name} has block_size={block_size}, expected {expected_block_size}"
            )
        terminal_consumers, invalid_consumers = _trace_tensor_consumers(
            node.output, consumers, graph_outputs, {"Gemm", "MatMul"}, 1
        )
        if not terminal_consumers:
            errors.append(f"{node_name} does not reach a Gemm/MatMul weight input")
        if invalid_consumers:
            errors.append(
                f"{node_name} has non-weight consumers: " + ", ".join(invalid_consumers[:5])
            )

    fp8_conv_pairs, disallowed_qdq_weights = _find_initializer_backed_qdq_weights(
        onnx_model, allow_fp8_conv
    )
    if allow_fp8_conv and not fp8_conv_pairs:
        errors.append("no initializer-backed FP8 Conv weight Q/DQ was exported")
    if expected_fp8_conv_count is not None and len(fp8_conv_pairs) != expected_fp8_conv_count:
        errors.append(
            f"found {len(fp8_conv_pairs)} initializer-backed FP8 Conv weight Q/DQ pairs, "
            f"expected {expected_fp8_conv_count} enabled Conv2d pairs"
        )
    if disallowed_qdq_weights:
        errors.append(
            "disallowed initializer-backed Q/DQ weight consumers: "
            + ", ".join(disallowed_qdq_weights[:5])
        )

    if errors:
        raise ValueError("Invalid raw FP4 ONNX graph: " + "; ".join(errors))
    return len(fp4_nodes)


def _validate_final_fp4_graph(
    onnx_model,
    expected_weight_count,
    expected_block_size=16,
    *,
    allow_fp8_conv=False,
    expected_fp8_conv_count=None,
):
    initializers = {initializer.name: initializer for initializer in onnx_model.graph.initializer}
    producers = {output: node for node in onnx_model.graph.node for output in node.output if output}
    consumers = get_tensor_consumer_nodes(onnx_model.graph)
    graph_outputs = {output.name for output in onnx_model.graph.output}
    fp4_initializer_names = {
        name
        for name, initializer in initializers.items()
        if initializer.data_type == onnx.TensorProto.FLOAT4E2M1
    }
    weight_dq_nodes = [
        node
        for node in onnx_model.graph.node
        if node.op_type == "DequantizeLinear"
        and node.input
        and node.input[0] in fp4_initializer_names
    ]
    errors = []
    weight_dq_ids = {id(node) for node in weight_dq_nodes}
    fp4_weight_terminal_consumers = []

    remaining_markers = [node for node in onnx_model.graph.node if node.op_type == "TRT_FP4QDQ"]
    if remaining_markers:
        errors.append(f"{len(remaining_markers)} TRT_FP4QDQ weight markers remain")
    if len(fp4_initializer_names) != expected_weight_count:
        errors.append(
            f"found {len(fp4_initializer_names)} FLOAT4 weights, expected {expected_weight_count}"
        )
    if len(weight_dq_nodes) != expected_weight_count:
        errors.append(
            f"found {len(weight_dq_nodes)} FLOAT4 weight DQ nodes, expected {expected_weight_count}"
        )

    for initializer_name in fp4_initializer_names:
        direct_consumers = consumers.get(initializer_name, [])
        if (
            len(direct_consumers) != 1
            or id(direct_consumers[0]) not in weight_dq_ids
            or [
                index
                for index, input_name in enumerate(direct_consumers[0].input)
                if input_name == initializer_name
            ]
            != [0]
        ):
            errors.append(
                f"FLOAT4 weight {initializer_name} must feed exactly one weight DequantizeLinear"
            )

    fp4_weight_names = set()
    fp8_scale_names = set()
    for node in weight_dq_nodes:
        node_name = node.name or (node.output[0] if node.output else "<unnamed>")
        fp4_weight_names.add(node.input[0])
        if len(node.input) != 2:
            errors.append(f"{node_name} is not a two-input FLOAT4 DequantizeLinear")
            continue
        if _get_int_attribute(node, "block_size") != expected_block_size:
            errors.append(f"{node_name} does not use block_size={expected_block_size}")
        if _get_int_attribute(node, "axis") != -1:
            errors.append(f"{node_name} does not use axis=-1")

        scale_dq = producers.get(node.input[1])
        if scale_dq is None or scale_dq.op_type != "DequantizeLinear":
            errors.append(f"{node_name} is not scaled by a preceding DequantizeLinear")
            continue
        scale_dq_name = scale_dq.name or (scale_dq.output[0] if scale_dq.output else "<unnamed>")
        if len(scale_dq.input) != 2:
            errors.append(f"{scale_dq_name} must be a two-input DequantizeLinear")
            continue
        if any(attribute.name in {"axis", "block_size"} for attribute in scale_dq.attribute):
            errors.append(f"{scale_dq_name} must not use axis or block_size")

        fp8_scale = initializers.get(scale_dq.input[0])
        global_scale = initializers.get(scale_dq.input[1])
        if fp8_scale is None or fp8_scale.data_type != onnx.TensorProto.FLOAT8E4M3FN:
            errors.append(f"{node_name} does not use a FLOAT8E4M3FN block-scale initializer")
        else:
            fp8_scale_names.add(fp8_scale.name)
            fp8_scale_consumers = consumers.get(fp8_scale.name, [])
            if (
                len(fp8_scale_consumers) != 1
                or id(fp8_scale_consumers[0]) != id(scale_dq)
                or [
                    index
                    for index, input_name in enumerate(scale_dq.input)
                    if input_name == fp8_scale.name
                ]
                != [0]
            ):
                errors.append(f"{fp8_scale.name} must be consumed only by {scale_dq_name} input 0")
        if global_scale is None or global_scale.data_type != onnx.TensorProto.FLOAT:
            errors.append(f"{node_name} does not use a FLOAT global-scale initializer")
        else:
            errors.extend(
                _validate_positive_scalar(
                    global_scale.name,
                    f"{scale_dq_name} global scale",
                    initializers,
                    producers,
                )
            )
        if len(scale_dq.output) != 1 or node.input[1] != scale_dq.output[0]:
            errors.append(f"{node_name} is not scaled by {scale_dq_name}")
        else:
            scale_output_consumers = consumers.get(scale_dq.output[0], [])
            if (
                len(scale_output_consumers) != 1
                or id(scale_output_consumers[0]) != id(node)
                or [
                    index
                    for index, input_name in enumerate(node.input)
                    if input_name == scale_dq.output[0]
                ]
                != [1]
            ):
                errors.append(
                    f"{scale_dq_name} output must be consumed only by {node_name} input 1"
                )

        terminal_consumers, invalid_consumers = _trace_tensor_consumers(
            node.output, consumers, graph_outputs, {"Gemm", "MatMul"}, 1
        )
        fp4_weight_terminal_consumers.extend(terminal_consumers)
        if not terminal_consumers:
            errors.append(f"{node_name} does not reach a Gemm/MatMul weight input")
        if invalid_consumers:
            errors.append(
                f"{node_name} has non-weight consumers: " + ", ".join(invalid_consumers[:5])
            )

    if len(fp4_weight_names) != expected_weight_count:
        errors.append(
            f"found {len(fp4_weight_names)} referenced FLOAT4 weights, "
            f"expected {expected_weight_count}"
        )

    if len(fp8_scale_names) != expected_weight_count:
        errors.append(
            f"found {len(fp8_scale_names)} FLOAT8 block scales, expected {expected_weight_count}"
        )

    errors.extend(
        _validate_dynamic_fp4_activations(
            onnx_model, expected_block_size, fp4_weight_terminal_consumers
        )
    )

    fp8_conv_pairs, disallowed_qdq_weights = _find_initializer_backed_qdq_weights(
        onnx_model, allow_fp8_conv
    )
    if allow_fp8_conv:
        if not fp8_conv_pairs:
            errors.append("no initializer-backed FP8 Conv weight Q/DQ remains")
        errors.extend(_validate_normalized_fp8_qdq(onnx_model, fp8_conv_pairs))
    if expected_fp8_conv_count is not None and len(fp8_conv_pairs) != expected_fp8_conv_count:
        errors.append(
            f"found {len(fp8_conv_pairs)} initializer-backed FP8 Conv weight Q/DQ pairs, "
            f"expected {expected_fp8_conv_count} enabled Conv2d pairs"
        )
    if disallowed_qdq_weights:
        errors.append(
            "disallowed initializer-backed Q/DQ weight consumers: "
            + ", ".join(disallowed_qdq_weights[:5])
        )

    if errors:
        raise ValueError("Invalid final FP4 ONNX graph: " + "; ".join(errors))


def _normalize_fp8_qdq(onnx_model):
    graph = gs.import_onnx(onnx_model)
    graph.cleanup().toposort()
    onnx_model = convert_zp_fp8(gs.export_onnx(graph))
    graph = gs.import_onnx(onnx_model)
    return gs.export_onnx(graph.cleanup().toposort())


def _ensure_default_opset(onnx_model, minimum_version):
    for opset_import in onnx_model.opset_import:
        if opset_import.domain in {"", "ai.onnx"}:
            opset_import.version = max(opset_import.version, minimum_version)
            return
    opset_import = onnx_model.opset_import.add()
    opset_import.domain = ""
    opset_import.version = minimum_version


def _process_fp4_onnx_graph(
    onnx_model,
    model_name,
    expected_block_size=16,
    *,
    expected_linear_count=None,
    expected_fp8_conv_count=None,
):
    allow_fp8_conv = model_name in {"sdxl-1.0", "sdxl-turbo"}
    expected_weight_count = _validate_raw_fp4_graph(
        onnx_model,
        expected_block_size,
        allow_fp8_conv=allow_fp8_conv,
        expected_linear_count=expected_linear_count,
        expected_fp8_conv_count=expected_fp8_conv_count,
    )
    if allow_fp8_conv:
        onnx_model = _normalize_fp8_qdq(onnx_model)
    onnx_model = NVFP4QuantExporter.process_model(onnx_model)
    _ensure_default_opset(onnx_model, 23)
    _validate_final_fp4_graph(
        onnx_model,
        expected_weight_count,
        expected_block_size,
        allow_fp8_conv=allow_fp8_conv,
        expected_fp8_conv_count=expected_fp8_conv_count,
    )
    return onnx_model


def _get_sdxl_fp4_expected_counts(backbone):
    linear_count = 0
    conv_count = 0
    for module in backbone.modules():
        input_quantizer = getattr(module, "input_quantizer", None)
        weight_quantizer = getattr(module, "weight_quantizer", None)
        pair_enabled = all(
            quantizer is not None and getattr(quantizer, "is_enabled", False)
            for quantizer in (input_quantizer, weight_quantizer)
        )
        if not pair_enabled:
            continue
        if isinstance(module, (torch.nn.Linear, RealQuantLinear)):
            linear_count += 1
        elif isinstance(module, torch.nn.Conv2d):
            conv_count += 1
    return linear_count, conv_count


def save_onnx(onnx_model, output, external_data_name=None):
    onnx.save(
        onnx_model,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_name or output.name + "_data",
        size_threshold=1024,
    )
    print(f"ONNX model saved to {output}")


def _get_external_data_paths(output):
    fallback = output.with_name(output.name + "_data")
    if not output.exists():
        return set()
    try:
        onnx_model = onnx.load(str(output), load_external_data=False)
    except Exception:
        return {fallback} if fallback.exists() else set()

    output_parent = output.parent.resolve()
    paths = set()
    for initializer in onnx_model.graph.initializer:
        for entry in initializer.external_data:
            if entry.key != "location":
                continue
            path = (output.parent / entry.value).resolve()
            if path.parent == output_parent and path != output.resolve():
                paths.add(path)
    return paths


def _save_onnx_atomically(onnx_model, output):
    staging_dir = Path(tempfile.mkdtemp(prefix=".modelopt-export-", dir=output.parent))
    staged_output = staging_dir / output.name
    external_data_name = f"{output.name}_data.{uuid.uuid4().hex}"
    staged_data = staging_dir / external_data_name
    published_data = output.parent / external_data_name
    old_data_paths = _get_external_data_paths(output)
    had_previous_output = output.exists()
    previous_output = staging_dir / "previous-model.onnx"
    try:
        if had_previous_output:
            shutil.copy2(output, previous_output)
        save_onnx(onnx_model, staged_output, external_data_name=external_data_name)
        onnx.checker.check_model(str(staged_output))
        has_external_data = staged_data.exists()
        try:
            if has_external_data:
                os.replace(staged_data, published_data)
            os.replace(staged_output, output)
        except BaseException:
            if previous_output.exists():
                os.replace(previous_output, output)
            elif not had_previous_output:
                output.unlink(missing_ok=True)
            published_data.unlink(missing_ok=True)
            raise

        for old_data_path in old_data_paths - {published_data.resolve()}:
            with suppress(OSError):
                old_data_path.unlink(missing_ok=True)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def modelopt_export_sd(backbone, onnx_dir, model_name, precision, expected_fp4_block_size=16):
    model_file_name = "model.onnx"
    os.makedirs(f"{onnx_dir}", exist_ok=True)
    tmp_subfolder = tempfile.mkdtemp(prefix=".modelopt-raw-", dir=onnx_dir)
    tmp_output = Path(f"{tmp_subfolder}/{model_file_name}")
    q_output = Path(f"{onnx_dir}/{model_file_name}")
    strict_sdxl_fp4 = precision == "fp4" and model_name in {"sdxl-1.0", "sdxl-turbo"}
    expected_linear_count = None
    expected_fp8_conv_count = None
    if strict_sdxl_fp4:
        expected_linear_count, expected_fp8_conv_count = _get_sdxl_fp4_expected_counts(backbone)

    try:
        quantizer_context = (
            configure_linear_module_onnx_quantizers(backbone)
            if precision == "fp4"
            else nullcontext()
        )

        dummy_kwargs, dynamic_axes, _ = generate_dummy_kwargs_and_dynamic_axes_and_shapes(
            model_name, backbone
        )

        if model_name in ["sdxl-1.0", "sdxl-turbo"]:
            input_names = [
                "sample",
                "timestep",
                "encoder_hidden_states",
                "text_embeds",
                "time_ids",
            ]
            output_names = ["latent"]
        elif model_name == "sd3-medium":
            input_names = [
                "hidden_states",
                "encoder_hidden_states",
                "pooled_projections",
                "timestep",
            ]
            output_names = ["sample"]
        elif model_name == "sd3.5-medium":
            input_names = [
                "hidden_states",
                "encoder_hidden_states",
                "pooled_projections",
                "timestep",
            ]
            output_names = ["out_hidden_states"]
        elif model_name in ["flux-dev", "flux-schnell"]:
            input_names = [
                "hidden_states",
                "encoder_hidden_states",
                "pooled_projections",
                "timestep",
                "img_ids",
                "txt_ids",
            ]
            if model_name == "flux-dev":
                input_names.append("guidance")
            output_names = ["latent"]
        elif model_name == "ltx-video-dev":
            input_names = [
                "hidden_states",
                "encoder_hidden_states",
                "timestep",
                "encoder_attention_mask",
                "video_coords",
            ]
            output_names = ["latent"]
        elif model_name == "wan2.2-t2v-14b":
            input_names = [
                "hidden_states",
                "timestep",
                "encoder_hidden_states",
            ]
            output_names = ["latent"]
        else:
            raise NotImplementedError(f"Unsupported model_id: {model_name}")

        with quantizer_context, torch.inference_mode():
            onnx_export(
                backbone,
                (),
                f=tmp_output.as_posix(),
                kwargs=dummy_kwargs,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                opset_version=20,
                dynamo=False,
            )
        print(f"Saved at {tmp_output}")
        onnx_model = onnx.load(str(tmp_output), load_external_data=True)
        if precision == "fp8":
            if not model_name.startswith("flux"):
                onnx_model = _normalize_fp8_qdq(onnx_model)
            else:
                flux_convert_rope_weight_type(onnx_model)
        if precision == "fp4":
            if strict_sdxl_fp4:
                onnx_model = _process_fp4_onnx_graph(
                    onnx_model,
                    model_name,
                    expected_fp4_block_size,
                    expected_linear_count=expected_linear_count,
                    expected_fp8_conv_count=expected_fp8_conv_count,
                )
            else:
                onnx_model = NVFP4QuantExporter.process_model(onnx_model)
        if strict_sdxl_fp4:
            _save_onnx_atomically(onnx_model, q_output)
        else:
            save_onnx(onnx_model, q_output)
    finally:
        shutil.rmtree(tmp_subfolder, ignore_errors=True)
