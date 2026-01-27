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

"""Performs kv cache quantization, and returns the ONNX ModelProto."""

import copy
import os
import pickle
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization.calibrate import CalibrationDataReader
from tqdm import tqdm

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.ort_utils import create_inference_session
from modelopt.onnx.utils import save_onnx


# using fp8 as default quantization mode
def kv_cache_quantize(
    onnx_model: onnx.ModelProto,
    kv_cache_dtype: str = "fp8",  # only support fp8 and int8 for now, will support fp16, uint8 later
    kv_quant_mode: str = "NONE",  # NONE / PER_TENSOR / PER_CHANNEL
    kv_cache_bit_width: int = 0,  # only used for uint8, available options: 2, 4, 8
    intermediate_generated_files: list[str] = [],
) -> onnx.ModelProto:
    """Perform kv cache quantization on the given ONNX model."""
    if kv_cache_dtype == "NONE":
        kv_cache_dtype = "fp8"

    logger.info(
        f"Start kv cache quantization with kv_cache_dtype {kv_cache_dtype}, "
        f"kv_quant_mode {kv_quant_mode}"
    )

    # TODO: add a validation for dtype, model, input model(name)
    validate_kv_cache_config()

    logger.info(f"intermediate_generated_files: {intermediate_generated_files}")

    kv_tensor_names_list = []
    output_type_map = {"int8": onnx.TensorProto.INT8, "fp8": onnx.TensorProto.FLOAT8E4M3FN}

    # replace each tensor starting with past_key_values
    for input in onnx_model.graph.input:
        if "past_key_values" in input.name:
            input.type.tensor_type.elem_type = output_type_map.get(kv_cache_dtype)

    # Update graph output similarly, at the sametime add all output names,
    # it will be used to collect calibration data later

    for output in onnx_model.graph.output:
        if "present" in output.name:
            kv_tensor_names_list.append(output.name)
            output.type.tensor_type.elem_type = output_type_map.get(kv_cache_dtype)

    kv_tensor_names_list.sort()

    # loop through all nodes and find GQA node
    group_query_attention_nodes = [
        node for node in onnx_model.graph.node if node.op_type == "GroupQueryAttention"
    ]

    tensor_range = []
    # both scale is a TensorData, scale's shape depends on kv_quant_mode
    k_scale = None
    v_scale = None

    # look for file named tmp_calib_data.json in intermediate_generated_files
    for intermediate_file in intermediate_generated_files:
        # if a file end with .calib_data, use it as calibration data
        if intermediate_file.endswith("tmp_calib_data.json"):
            # load calibration data from file
            with open(intermediate_file, "rb") as f:
                tensor_range = pickle.load(f)
            logger.info(
                f"Using calibration data from {intermediate_file} for kv cache quantization"
            )
            break

    # parse tensor_range
    if kv_quant_mode == "PER_TENSOR":
        for node in group_query_attention_nodes:
            # calculate k_scale based on input and output range
            k_max = 0
            v_max = 0
            for output in node.output:
                if "key" not in output and "value" not in output:
                    continue
                index = kv_tensor_names_list.index(output)
                abs_max = 0
                for data_range in tensor_range:
                    abs_max = max(abs_max, np.abs(np.asarray(data_range[index])).max())
                if "key" in output:
                    k_max = max(k_max, abs_max)
                if "value" in output:
                    v_max = max(v_max, abs_max)

            tmax = 0
            if kv_cache_dtype == "fp8":
                tmax = 448  # max fp value for E4M3
            elif kv_cache_dtype == "int8":
                tmax = 127  # max int8 value
            else:
                raise ValueError(
                    f"Unsupported kv_cache_dtype {kv_cache_dtype} for kv cache quantization"
                )
            # create onnx tensor data as fp16 and assign to k_scale and v_scale
            k_scale_value = k_max / tmax if k_max != 0 else 1.0
            v_scale_value = v_max / tmax if v_max != 0 else 1.0

            k_scale = onnx.helper.make_tensor(
                name=node.name + "_k_scale",
                data_type=onnx.TensorProto.FLOAT16,
                dims=[1],
                vals=[k_scale_value],
            )
            v_scale = onnx.helper.make_tensor(
                name=node.name + "_v_scale",
                data_type=onnx.TensorProto.FLOAT16,
                dims=[1],
                vals=[v_scale_value],
            )
            onnx_model.graph.initializer.append(k_scale)
            onnx_model.graph.initializer.append(v_scale)
            # add scale to input, use empty string to pad the input to 12,
            # insert k_scale at index 12 and v_scale at index 13
            while len(node.input) < 12:
                node.input.append("")
            node.input.append(k_scale.name)
            node.input.append(v_scale.name)
    elif kv_quant_mode == "PER_CHANNEL":
        raise NotImplementedError("PER_CHANNEL kv cache quantization is not implemented yet")
    else:
        raise ValueError(f"Unsupported kv_quant_mode {kv_quant_mode} for kv cache quantization")

    # add attributes to GQA node
    for node in group_query_attention_nodes:
        # add attribute for quantization type
        node.attribute.append(onnx.helper.make_attribute("k_quant_type", kv_quant_mode))
        node.attribute.append(onnx.helper.make_attribute("v_quant_type", kv_quant_mode))
        # set bit width attribute, only used for uint8, not supported currently
        if kv_cache_dtype == "uint8":
            node.attribute.append(
                onnx.helper.make_attribute("kv_cache_bit_width", kv_cache_bit_width)
            )
    logger.info("kv cache quantization done")

    return onnx_model


def validate_kv_cache_config():
    """Validate kv cache quantization configuration.
    The GQA node needs to have present and past_key_values tensors.
    """
    # TODO: this feature only works for model generated by model builder
    return True


def save_kv_cache_calib_data(
    onnx_model: str | Path | onnx.ModelProto,
    session: ort.InferenceSession | None = None,
    inputs: list[dict] = [],
    intermediate_generated_files: list[str] = [],
    **kwargs: Any,
):
    """Save kv cache calibration data for quantization."""
    if not isinstance(onnx_model, onnx.ModelProto):
        onnx_model = onnx.load(onnx_model)

    kv_tensor_names_list = [
        output.name for output in onnx_model.graph.output if "present" in output.name
    ]
    kv_tensor_input_names_list = [
        input.name for input in onnx_model.graph.input if "past_key_values" in input.name
    ]

    kv_tensor_input_names_list.sort()
    kv_tensor_names_list.sort()

    custom_calib_input = kwargs.get("custom_calib_input", False)
    if custom_calib_input:
        # create a custom calibration input, only apply to llama GQA model
        np_inp_d = {}
        seq_len = 100
        # each element of input_ids is random int64 between 0 and 128256
        np_inp_d["input_ids"] = np.random.randint(0, 128000, size=(1, seq_len), dtype=np.int64)
        np_inp_d["attention_mask"] = np.zeros((1, seq_len), dtype=np.int64)
        # initilize past_key_values with zeros, each shape is (1,8,10,128)
        for name in kv_tensor_input_names_list:
            np_inp_d[name] = np.zeros((1, 8, seq_len, 128), dtype=np.float16)

        logger.info(f"Input for kv cache calibration: {list(np_inp_d.keys())}")
        logger.info(f"Input_id shape: {np_inp_d['input_ids'].shape}")
        logger.info(f"attention mask shape: {np_inp_d['attention_mask'].shape}")
        inputs = [np_inp_d]

    # get the para from kwargs
    use_real_calib_input = kwargs.get("use_real_calib_input", True)
    if use_real_calib_input:
        import sys

        sys.path.append("C:\\repos\\TensorRT-Model-Optimizer")
        from examples.windows.onnx_ptq.genai_llm.quantize import get_calib_inputs

        inputs = get_calib_inputs(
            dataset_name="cnn_dailymail",
            model_name="C:\\repos\\models\\Llama-3.2-3B-Instruct-ONNX\\cuda\\cuda-fp16",
            cache_dir="C:\\tmp",
            calib_size=4,  # should no longer than 128
            batch_size=1,
            block_size=512,
            device=torch.device("cuda:0"),  # "cuda:0", "cpu"
            use_fp32=False,
            use_buffer_share=False,
            add_past_kv_inputs=True,
            max_calib_rows_to_load=128,
            add_position_ids=False,
            trust_remote_code=False,
        )

    # inputs = inputs[:10]  # use only first 10 inputs for calibration to save time

    kv_tensor_data = []

    enable_auto_regressive = kwargs.get("enable_auto_regressive", False)
    if enable_auto_regressive:
        for i in tqdm(range(len(inputs)), desc="Caching kv cache key values..."):
            inp_d = inputs[i]
            np_inp_d = {name: np.asarray(tensor) for name, tensor in inp_d.items()}
            # set attention mask to all zeros
            np_inp_d["attention_mask"][0, :] = 0
            auto_regressive_steps = len(inp_d["input_ids"][0]) // 2
            for j in range(auto_regressive_steps):
                # mark attention mask
                np_inp_d["attention_mask"][0, j] = 1
                output = session.run(kv_tensor_names_list, np_inp_d)
                kv_tensor_data.append(output)
                # update past_key_values with output
                for k, name in enumerate(kv_tensor_names_list):
                    # replace 'present' with 'past_key_values' in name
                    inp_name = name.replace("present", "past_key_values")
                    np_inp_d[inp_name] = output[k]
    else:
        for i in tqdm(range(len(inputs)), desc="Caching kv cache key values..."):
            inp_d = inputs[i]
            np_inp_d = {name: np.asarray(tensor) for name, tensor in inp_d.items()}
            output = session.run(kv_tensor_names_list, np_inp_d)
            kv_tensor_data.append(output)

    # save to tmp file named tmp_calib_data.json
    tmp_dir = tempfile.mkdtemp()
    calib_data_path = Path(tmp_dir).joinpath("tmp_calib_data.json").as_posix()
    # call to_dict and save to json
    with open(calib_data_path, "wb") as f:
        pickle.dump(kv_tensor_data, f)
    intermediate_generated_files.append(calib_data_path)


def save_kv_cache_calib_data_rtn(
    onnx_model: onnx.ModelProto,
    data_reader: CalibrationDataReader | None = None,
    calibration_eps: list[str] = [],
    input_shapes_profile: Sequence[dict[str, str]] | None = None,
    intermediate_generated_files: list[str] = [],
    use_external_data_format: bool = False,
    **kwargs: Any,
):
    """Save kv cache calibration data for RTN quantization. Create inference session internally."""
    augmented_model = copy.deepcopy(onnx_model)

    # save model in augmented_onnx_path for creating inference session
    augmented_onnx_file, augmented_onnx_path = tempfile.mkstemp(suffix=".onnx")
    os.close(augmented_onnx_file)

    save_onnx(augmented_model, augmented_onnx_path, use_external_data_format)

    # Creating inference session and preparing inputs for calibration
    session = create_inference_session(augmented_onnx_path, calibration_eps, input_shapes_profile)
    inputs = []
    for inp_d in data_reader:
        inputs.append(inp_d)
        assert isinstance(inp_d, dict)
    save_kv_cache_calib_data(
        onnx_model,
        session=session,
        inputs=inputs,
        intermediate_generated_files=intermediate_generated_files,
        **kwargs,
    )

    logger.info("Saved kv-cache calibration data for RTN quantization")

    del session

    try:
        os.remove(augmented_onnx_path)
        if use_external_data_format:
            os.remove(augmented_onnx_path + "_data")
    except OSError:
        logger.warn("Augmented ONNX model or external data file was not found")
