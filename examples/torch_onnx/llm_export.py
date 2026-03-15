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

"""This script is used to export a LLM model to ONNX and perform quantization."""

import argparse
import json
import os
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
from modelopt.onnx.llm_export_utils.export_utils import (
    ModelLoader,
    WrapperModelForCausalLM,
    llm_to_onnx,
)
from modelopt.onnx.llm_export_utils.quantization_utils import quantize
from modelopt.onnx.llm_export_utils.surgeon_utils import fold_fp8_qdq_to_dq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.utils import is_quantized_linear


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
        choices=["fp16", "fp8", "int4_awq", "nvfp4"],
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
    """
    Get config.json file path from the arguments.
    The default priority is: config_path > hf_model_path/config.json > onnx_path/../config.json
    """
    if args.config_path and os.path.exists(args.config_path):
        return args.config_path
    if args.hf_model_path:
        # Check if hf_model_path is a local directory
        if os.path.isdir(args.hf_model_path):
            torch_config = os.path.join(args.hf_model_path, "config.json")
            if os.path.exists(torch_config):
                return torch_config
        else:
            # For HuggingFace model names, download config temporarily
            try:
                # Download config from HuggingFace
                config = AutoConfig.from_pretrained(
                    args.hf_model_path, trust_remote_code=args.trust_remote_code
                )

                # Save to temporary file
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
    """
    os.makedirs(output_dir, exist_ok=True)

    if dtype == "fp16":
        print("Loading fp16 ONNX model...")

        llm_to_onnx(
            wrapper_cls(model), output_dir, extra_inputs=extra_inputs, extra_dyn_axes=extra_dyn_axes
        )
        shutil.copy(config_path, os.path.join(output_dir, "config.json"))

    # Need to quantize model to fp8, int4_awq or nvfp4
    if dtype in ["fp8", "int4_awq", "nvfp4"]:
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_path, trust_remote_code=args.trust_remote_code
        )
        # Only check for local modelopt_state if hf_model_path is a local directory
        if os.path.isdir(hf_model_path):
            modelopt_state = os.path.join(hf_model_path, "modelopt_state.pth")
            model_needs_quantization = not os.path.exists(modelopt_state)
        else:
            # For HuggingFace model names, always quantize as we can't have local state files
            model_needs_quantization = True

        if model_needs_quantization:
            model = quantize(
                model, tokenizer, dtype, lm_head_precision, dataset_dir, calib_size=calib_size
            )

            if dtype == "nvfp4":
                # This is required for nvfp4 ONNX export
                for module in model.modules():
                    assert not isinstance(module, torch.nn.Linear) or is_quantized_linear(module)
                    if isinstance(module, torch.nn.Linear):
                        module.input_quantizer._trt_high_precision_dtype = "Half"
                        module.input_quantizer._onnx_quantizer_type = "dynamic"
                        module.weight_quantizer._onnx_quantizer_type = "static"

            if dtype in {"fp8", "int4_awq", "nvfp4"}:
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

            # Compress weights
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
    lm_head_precision="fp16",
):
    """Surgeon raw llm onnx to fit TRT.

    For example, insert quantization q/dq nodes.

    Args:
        raw_onnx_path: str
        output_dir: str
        dtype: str
        config_path: str
        lm_head_precision: str
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

    elif dtype == "int4_awq":
        with time_operation("quantizing weights to int4"):
            onnx_model = INT4QuantExporter.process_model(onnx_model)

    output_onnx_name = f"{output_dir}/model.onnx"
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

    onnx.save_model(
        onnx_model,
        output_onnx_name,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="onnx_model.data",
        convert_attribute=True,
    )

    if os.path.exists(config_path):
        if os.path.isfile(config_path) and config_path.endswith("config.json"):
            # config_path is already a config.json file
            shutil.copy(config_path, os.path.join(output_dir, "config.json"))
        elif os.path.isdir(config_path):
            # config_path is a directory containing config.json
            shutil.copy(
                os.path.join(config_path, "config.json"), os.path.join(output_dir, "config.json")
            )
        else:
            print(f"Warning: Unexpected config_path format: {config_path}")

    t3 = time.time()
    print(f"Surgeon LLM completed in {t3 - t2}s.")


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
        # Exporting ONNX from PyTorch model
        model = model_loader.load_model(trust_remote_code=args.trust_remote_code)
        onnx_dir = args.output_dir + "_raw" if args.save_original else args.output_dir
        # Surgeon graph based on precision
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
        )

    # Providing the config path to config.json results in a hf validation error for internvl_chat.
    surgeon_llm(
        raw_onnx_path=raw_onnx_path,
        output_dir=args.output_dir,
        dtype=args.dtype,
        config_path=args.config_path,
        lm_head_precision=args.lm_head,
    )

    end_time = time.time()
    print(
        f"LLM ONNX saved to {args.output_dir} with {args.dtype} precision in {end_time - start_time}s."
    )


if __name__ == "__main__":
    parser = llm_arguments()
    args = parser.parse_args()
    args.config_path = get_config_path(args)
    main(args)
