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
from contextlib import nullcontext
from pathlib import Path

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
from modelopt.torch.quantization.export_onnx import configure_linear_module_onnx_quantizers
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


def generate_fp8_scales(backbone):
    # temporary solution due to a known bug in torch.onnx._dynamo_export
    for _, module in backbone.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)) and (
            hasattr(module.input_quantizer, "_amax") and module.input_quantizer is not None
        ):
            module.input_quantizer._num_bits = 8
            module.weight_quantizer._num_bits = 8
            module.input_quantizer._amax = module.input_quantizer._amax * (127 / 448.0)
            module.weight_quantizer._amax = module.weight_quantizer._amax * (127 / 448.0)


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
        elif model_id in ["sd3-medium"]:
            output_name = "sample"
        elif model_id in ["sd3.5-medium"]:
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


def save_onnx(onnx_model, output):
    onnx.save(
        onnx_model,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output.name + "_data",
        size_threshold=1024,
    )
    print(f"ONNX model saved to {output}")


def modelopt_export_sd(backbone, onnx_dir, model_name, precision):
    model_file_name = "model.onnx"
    os.makedirs(f"{onnx_dir}", exist_ok=True)
    tmp_subfolder = tempfile.mkdtemp(prefix="myapp_")
    tmp_output = Path(f"{tmp_subfolder}/{model_file_name}")
    q_output = Path(f"{onnx_dir}/{model_file_name}")

    quantizer_context = (
        configure_linear_module_onnx_quantizers(backbone) if precision == "fp4" else nullcontext()
    )

    dummy_kwargs, dynamic_axes, _ = generate_dummy_kwargs_and_dynamic_axes_and_shapes(
        model_name, backbone
    )

    if model_name in ["sdxl-1.0", "sdxl-turbo"]:
        input_names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]
        output_names = ["latent"]
    elif model_name == "sd3-medium":
        input_names = ["hidden_states", "encoder_hidden_states", "pooled_projections", "timestep"]
        output_names = ["sample"]
    elif model_name == "sd3.5-medium":
        input_names = ["hidden_states", "encoder_hidden_states", "pooled_projections", "timestep"]
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
    elif model_name in ["ltx-video-dev"]:
        input_names = [
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "encoder_attention_mask",
            "video_coords",
        ]
        output_names = ["latent"]
    elif model_name in ["wan2.2-t2v-14b"]:
        input_names = [
            "hidden_states",
            "timestep",
            "encoder_hidden_states",
        ]
        output_names = ["latent"]
    else:
        raise NotImplementedError(f"Unsupported model_id: {model_name}")

    do_constant_folding = True
    opset_version = 20

    with quantizer_context, torch.inference_mode():
        onnx_export(
            backbone,
            (),
            f=tmp_output.as_posix(),
            kwargs=dummy_kwargs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=do_constant_folding,
            opset_version=opset_version,
            dynamo=False,
        )
    print(f"Saved at {tmp_output}")
    onnx_model = onnx.load(str(tmp_output), load_external_data=True)
    if precision == "fp8":
        if not model_name.startswith("flux"):
            graph = gs.import_onnx(onnx_model)
            graph.cleanup().toposort()
            onnx_model = gs.export_onnx(graph)
            onnx_model = convert_zp_fp8(onnx_model)
            graph = gs.import_onnx(onnx_model)
            onnx_model = gs.export_onnx(graph.cleanup())
        else:
            flux_convert_rope_weight_type(onnx_model)
    if precision == "fp4":
        onnx_model = NVFP4QuantExporter.process_model(onnx_model)
    save_onnx(onnx_model, q_output)
    shutil.rmtree(tmp_subfolder, ignore_errors=True)
