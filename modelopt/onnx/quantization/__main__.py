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

"""Command-line entrypoint for ONNX PTQ."""

import argparse

import numpy as np

from modelopt.onnx.quantization.quantize import quantize

__all__ = ["main"]


def get_parser() -> argparse.ArgumentParser:
    """Get the argument parser for ONNX PTQ."""
    argparser = argparse.ArgumentParser("python -m modelopt.onnx.quantization")
    group = argparser.add_mutually_exclusive_group(required=False)
    argparser.add_argument(
        "--onnx_path", required=True, type=str, help="Input onnx model without Q/DQ nodes."
    )
    argparser.add_argument(
        "--quantize_mode",
        type=str,
        choices=["fp8", "int8", "int4"],
        default="int8",
        help="Quantization mode for the given ONNX model.",
    )
    argparser.add_argument(
        "--calibration_method",
        type=str,
        choices=["max", "entropy", "awq_clip", "rtn_dq"],
        help=(
            "Calibration method choices for int8/fp8: {entropy (default), max}, "
            "int4: {awq_clip (default), rtn_dq}."
        ),
    )
    group.add_argument(
        "--calibration_data_path",
        type=str,
        help="Calibration data in npz/npy format. If None, random data for calibration will be used.",
    )
    group.add_argument(
        "--calibration_cache_path",
        type=str,
        help="Pre-calculated activation tensor scaling factors aka calibration cache path.",
    )
    argparser.add_argument(
        "--calibration_shapes",
        type=str,
        required=False,
        help=(
            "Optional model input shapes for calibration."
            "Users should provide the shapes specifically if the model has non-batch dynamic dimensions."
            "Example input shapes spec: input0:1x3x256x256,input1:1x3x128x128"
        ),
    )
    argparser.add_argument(
        "--calibration_eps",
        type=str,
        default=["cpu", "cuda:0", "trt"],
        nargs="+",
        help=(
            "Priority order for the execution providers (EP) to calibrate the model. "
            "Any subset of ['trt', 'cuda:x', dml:x, 'cpu'], where 'x' is the device id."
            "If a custom op is detected in the model, 'trt' will automatically be added to the EP list."
        ),
    )
    argparser.add_argument(
        "--override_shapes",
        type=str,
        required=False,
        help=(
            "Override model input shapes with static shapes."
            "Example input shapes spec: input0:1x3x256x256,input1:1x3x128x128"
        ),
    )
    argparser.add_argument(
        "--op_types_to_quantize",
        type=str,
        nargs="+",
        help="A space-separated list of node types to quantize.",
    )
    argparser.add_argument(
        "--op_types_to_exclude",
        type=str,
        nargs="+",
        help="A space-separated list of node types to exclude from quantization.",
    )
    argparser.add_argument(
        "--op_types_to_exclude_fp16",
        type=str,
        nargs="+",
        help=(
            "A space-separated list of node types to exclude from FP16/BF16 conversion. "
            "Relevant when --high_precision_dtype is 'fp16' or 'bf16'."
        ),
    )
    argparser.add_argument(
        "--nodes_to_quantize",
        type=str,
        nargs="+",
        help="A space-separated list of node names to quantize. Regular expressions are supported.",
    )
    argparser.add_argument(
        "--nodes_to_exclude",
        type=str,
        nargs="+",
        help="A space-separated list of node names to exclude from quantization. Regular expressions are supported.",
    )
    argparser.add_argument(
        "--use_external_data_format",
        action="store_true",
        help=(
            "If True or model size is larger than 2GB, "
            "<MODEL_NAME>.onnx_data will be used to write weights and constants."
        ),
    )
    argparser.add_argument(
        "--keep_intermediate_files",
        action="store_true",
        help=(
            "If True, keep the files generated during the ONNX models' conversion/calibration. "
            "Otherwise, only the converted ONNX file is kept for the user."
        ),
    )
    argparser.add_argument(
        "--output_path",
        type=str,
        help=(
            "Output filename to save the converted ONNX model. If None, save it in the same dir as "
            "the original ONNX model with an appropriate suffix."
        ),
    )
    argparser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "debug", "info", "warning", "error"],
        default="INFO",
        help="Set the logging level for the quantization process.",
    )
    argparser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to the log file for the quantization process.",
    )
    argparser.add_argument(
        "--trt_plugins",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A space-separated list with the custom TensorRT plugin library paths in .so format (compiled shared "
            "library). If this is not None, the TensorrtExecutionProvider is invoked, so make sure that the TensorRT "
            "libraries are in the PATH or LD_LIBRARY_PATH variables."
        ),
    )
    argparser.add_argument(
        "--trt_plugins_precision",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A space-separated list indicating the precision for each custom op. "
            "Each item should have the format <op_type>:<precision> (all inputs and outputs have the same precision) "
            "or <op_type>:[<inp1_precision>,<inp2_precision>,...]:[<out1_precision>,<out2_precision>,...] "
            "(inputs and outputs can have different precisions), where precision can be fp32 (default), "
            "fp16, int8, or fp8. Note that int8/fp8 should be the same as the quantization mode. "
            "For example: op_type_1:fp16 op_type_2:[int8,fp32]:[int8]."
        ),
    )
    argparser.add_argument(
        "--high_precision_dtype",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help=(
            "High precision data type of the output model. If the input model is of dtype fp32, "
            "it will be converted to fp16 dtype by default."
        ),
    )
    argparser.add_argument(
        "--mha_accumulation_dtype",
        type=str,
        default="fp16",
        help=(
            "Accumulation dtype of MHA. This flag will only take effect when mha_accumulation_dtype == 'fp32' "
            "and quantize_mode == 'fp8'. One of ['fp32', 'fp16']"
        ),
    )
    argparser.add_argument(
        "--disable_mha_qdq",
        action="store_true",
        help="If True, Q/DQ will not be added to MatMuls in MHA pattern.",
    )
    argparser.add_argument(
        "--dq_only",
        action="store_true",
        help=(
            "If True, FP32/FP16 weights will be converted to INT8/FP8 weights. Q nodes will get removed from the "
            "weights and have only DQ nodes with those converted INT8/FP8 weights in the output model."
        ),
    )
    argparser.add_argument(
        "--use_zero_point",
        type=bool,
        default=False,
        help=(
            "If True, zero-point based quantization will be used - currently, applicable for awq_lite algorithm."
        ),
    )
    argparser.add_argument(
        "--passes",
        type=str,
        choices=["concat_elimination"],
        default=["concat_elimination"],
        nargs="+",
        help=(
            "A space-separated list of optimization passes name, if set, appropriate pre/post-processing passes will "
            "be invoked."
        ),
    )
    argparser.add_argument(
        "--simplify",
        action="store_true",
        help="If True, the given ONNX model will be simplified before quantization is performed.",
    )
    argparser.add_argument(
        "--calibrate_per_node",
        action="store_true",
        help=(
            "If set, performs calibration per node instead of running inference over the entire network. "
            "Useful for reducing memory consumption during large model inference."
        ),
    )
    argparser.add_argument(
        "--direct_io_types",
        action="store_true",
        help=(
            "If True, the I/O types in the quantized ONNX model will be modified to be lower precision whenever "
            "possible. Else, they will match the I/O types in the given ONNX model. "
            "The currently supported precisions are {fp16, int8, fp8}."
        ),
    )
    return argparser


def main():
    """Command-line entrypoint for ONNX PTQ."""
    args = get_parser().parse_args()
    calibration_data = None
    if args.calibration_data_path:
        calibration_data = np.load(args.calibration_data_path, allow_pickle=True)
        if args.calibration_data_path.endswith(".npz"):
            # Convert the NpzFile object to a Python dictionary
            calibration_data = {key: calibration_data[key] for key in calibration_data.files}

    quantize(
        args.onnx_path,
        quantize_mode=args.quantize_mode,
        calibration_data=calibration_data,
        calibration_method=args.calibration_method,
        calibration_cache_path=args.calibration_cache_path,
        calibration_shapes=args.calibration_shapes,
        calibration_eps=args.calibration_eps,
        override_shapes=args.override_shapes,
        op_types_to_quantize=args.op_types_to_quantize,
        op_types_to_exclude=args.op_types_to_exclude,
        op_types_to_exclude_fp16=args.op_types_to_exclude_fp16,
        nodes_to_quantize=args.nodes_to_quantize,
        nodes_to_exclude=args.nodes_to_exclude,
        use_external_data_format=args.use_external_data_format,
        keep_intermediate_files=args.keep_intermediate_files,
        output_path=args.output_path,
        log_level=args.log_level,
        log_file=args.log_file,
        trt_plugins=args.trt_plugins,
        trt_plugins_precision=args.trt_plugins_precision,
        high_precision_dtype=args.high_precision_dtype,
        mha_accumulation_dtype=args.mha_accumulation_dtype,
        disable_mha_qdq=args.disable_mha_qdq,
        dq_only=args.dq_only,
        use_zero_point=args.use_zero_point,
        passes=args.passes,
        simplify=args.simplify,
        calibrate_per_node=args.calibrate_per_node,
        direct_io_types=args.direct_io_types,
    )


if __name__ == "__main__":
    main()
