# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import logging
import os
import time

import cv2
import numpy as np

from modelopt.onnx.quantization.quantize import quantize as quantize_top_level_api

logging.getLogger().setLevel(logging.INFO)


def prepare_input(image: np.ndarray, np_dtype, image_input_width, image_input_height) -> np.ndarray:
    img_height, img_width = image.shape[:2]

    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (image_input_width, image_input_height))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_img = (input_img / 255.0 - mean) / std
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np_dtype)

    return input_tensor


def parse_calibration_eps(value):
    """Parse and validate the calibration_eps input."""
    valid_choices = {"cuda", "cpu", "dml"}
    # Split the input by commas and remove any surrounding whitespace
    eps = [item.strip() for item in value.split(",")]
    # Validate each calibration endpoint
    for ep in eps:
        if ep not in valid_choices:
            raise argparse.ArgumentTypeError(
                f"Invalid calibration endpoint: '{ep}'. Choose from 'cuda', 'cpu', 'dml'."
            )
    return eps


def parse_op_types_to_quantize(value):
    """Parse and validate the op_types_to_quantize input."""
    valid_choices = {"MatMul", "Conv"}
    # Split the input by commas and remove any surrounding whitespace
    op_types = [item.strip() for item in value.split(",")]
    # Validate each calibration endpoint
    for op in op_types:
        if op not in valid_choices:
            raise argparse.ArgumentTypeError(
                f"Invalid op-type: '{op}'. Choose from 'MatMul', 'Conv'."
            )
    return op_types


def get_calib_data_for_encoder(
    image_directory, calib_size, data_type, file_extension, image_input_dimension
):
    np_dtype = np.float16 if data_type == "fp16" else np.float32

    calib_data = {}

    image_files = [
        os.path.join(image_directory, f)
        for f in os.listdir(image_directory)
        if (os.path.isfile(os.path.join(image_directory, f)) and f.endswith(file_extension))
    ]

    assert len(image_files) > 0, "no image files found for encoder's calibration"

    print(
        f"\nPreparing calibration data for encoder. Number of images in image-directory = {len(image_files)}\n"
    )

    image_input_width, image_input_height = image_input_dimension.split(",")
    image_input_width, image_input_height = int(image_input_width), int(image_input_height)

    for i, image in enumerate(image_files):
        cv2_image = cv2.imread(image)
        assert cv2_image is not None, "cv2-image is none"
        tensor = prepare_input(cv2_image, np_dtype, image_input_width, image_input_height)
        tensor = tensor.astype(np_dtype)
        x = calib_data.get("image")
        if x is None:
            calib_data["image"] = tensor
        else:
            calib_data["image"] = np.concatenate((x, tensor), axis=0)
        if i == calib_size:
            break

    print(f"\nCalibration data for ENCODER is created. calib_size={calib_size}\n")
    return calib_data


def main(args):
    start_time = time.time()

    # args.qdq_for_weights = True

    print("\n\n######### SAM2's 8-bit Quantization:  Settings...\n\n")

    print(
        f"  quantization_mode={args.quant_mode},\n  calibrartion_method={args.calib_method},"
        f"\n  calib_size={args.calib_size},\n  use-random-calib-data={args.use_random_calib},"
        f"\n  op_types_to_quantize={args.op_types_to_quantize},\n  calibration-EPs={args.calibration_eps},"
        f"\n  qdq_for_weights={args.qdq_for_weights},\n  dq_only_for_weights={not args.qdq_for_weights},"
        f"\n  dtype={args.dtype},\n  image_input_dimension={args.image_input_dimension}\n"
    )
    print(
        f"\n  input-onnx-path={args.onnx_path},\n  image_dir={args.image_dir}"
        f"\n  image_file_extension={args.image_file_extension},\n  output-path={args.output_path},\n"
    )

    print("\n=========================================================\n\n")

    if args.use_random_calib:
        calib_data = None
    else:
        calib_data = get_calib_data_for_encoder(
            args.image_dir,
            args.calib_size,
            args.dtype,
            args.image_file_extension,
            args.image_input_dimension,
        )

    assert args.use_random_calib or calib_data is not None, "calibration data not prepared"

    logging.info("\nQuantizing the model....\n")
    quantize_top_level_api(
        onnx_path=args.onnx_path,
        quantize_mode=args.quant_mode,
        calibration_method=args.calib_method,
        calibration_data=None if args.use_random_calib else calib_data,
        calibration_eps=args.calibration_eps,
        use_external_data_format=True,
        output_path=args.output_path,
        op_types_to_quantize=args.op_types_to_quantize,
        nodes_to_exclude=[r"/lm_head", r"/Shape"],
        dq_only=not args.qdq_for_weights,
        verbose=True,
        high_precision_dtype="fp16" if args.dtype == "fp16" else "fp32",
        mha_accumulation_dtype="fp16" if args.dtype == "fp16" else "fp32",
        enable_gemv_detection_for_trt=False,
        enable_shared_constants_duplication=False,
    )
    logging.info(
        f"\nQuantization process (along with saving) took {time.time() - start_time} seconds\n"
    )

    print("\n\nDone\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize SAM2 ONNX model with INT8/FP8.")
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Input ONNX model path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output quantized model path.",
    )
    parser.add_argument(
        "--calib_method",
        type=str,
        default="max",
        help="calibration method for quantization (max or entropy)",
    )
    parser.add_argument(
        "--quant_mode",
        type=str,
        default="int8",
        help="quantization mode to be used (int8 or fp8)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        help="precision of the model tensors. Choose from 'fp32', 'fp16'.",
    )
    parser.add_argument(
        "--calib_size",
        type=int,
        default=32,
        help="Number of input calibration samples, should be no more than 256",
    )
    parser.add_argument(
        "--use_random_calib",
        type=bool,
        default=False,
        help="True when we want to use one randomly generated calibration sample",
    )
    parser.add_argument(
        "--qdq_for_weights",
        default=False,
        action="store_true",
        help="If True, Q->DQ nodes will be added for weights, otherwise only DQ nodes will be added.",
    )
    parser.add_argument(
        "--calibration_eps",
        type=parse_calibration_eps,  # Use the custom parser
        default=["cuda", "cpu"],  # Default as a list
        help="Comma-separated list of calibration endpoints. Choose from 'cuda', 'cpu', 'dml'.",
    )
    parser.add_argument(
        "--op_types_to_quantize",
        type=parse_op_types_to_quantize,  # Use the custom parser
        default=["MatMul"],  # Default as a list
        help="Comma-separated list of op-types that need to be quantized. Choose from 'MatMul', 'Conv'.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing image files to be used for calibration of sam2's encoder.",
    )
    parser.add_argument(
        "--image_file_extension",
        type=str,
        default="jpg",
        help="Extension of image files to be used for calibration of sam2's encoder. E.g. jpg, png.",
    )
    parser.add_argument(
        "--image_input_dimension",
        type=str,
        default="1024,1024",
        help="Last 2 dimensions of the image input to encoder, in comma-separated fashion",
    )
    args = parser.parse_args()
    main(args)
