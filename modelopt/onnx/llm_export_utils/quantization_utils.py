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

"""Quantization utilities for LLM models."""

import copy
import time

import modelopt.torch.quantization as mtq
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader


def _quantize_model(model, quant_config, calib_dataloader=None):
    """The calibration loop for the model can be setup using the modelopt API.

    Example usage:
    from modelopt.torch.utils.dataset_utils import create_forward_loop
    model = ...  # Initialize the model
    tokenizer = ...  # Initialize the tokenizer
    quant_cfg = ...  # Setup quantization configuration
    forward_loop = create_forward_loop(model=model, dataset_name="cnn_dailymail", tokenizer=tokenizer)
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    """

    def calibrate_loop(model):
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in enumerate(calib_dataloader):
            if idx % 10 == 0:
                print(f"Calibrating batch {idx}...")
            if isinstance(data, dict):
                data = {k: v.to(model.device) for k, v in data.items()}
                model(**data)
            else:
                data = data.to(model.device)
                model(data)

    print("Starting quantization...")
    start_time = time.time()
    mtq.quantize(model, quant_config, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization finishes in {end_time - start_time}s.")

    return model


def get_quant_config(precision, lm_head_precision="fp16"):
    """Get the quantization configuration."""
    if precision == "fp8":
        quant_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)

    elif precision == "nvfp4":
        quant_cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)

    elif precision == "int4_awq":
        quant_cfg = copy.deepcopy(mtq.INT4_AWQ_CFG)  # type: ignore[arg-type]

    else:
        raise ValueError(f"Unsupported precision: {precision}")

    quant_cfg_list: list = [
        e for e in quant_cfg["quant_cfg"] if isinstance(e, dict) and "quantizer_name" in e
    ]

    if lm_head_precision == "fp8":
        quant_cfg_list.append(
            {
                "quantizer_name": "*lm_head.input_quantizer",
                "cfg": {"num_bits": (4, 3), "axis": None},
            }
        )
        quant_cfg_list.append(
            {
                "quantizer_name": "*lm_head.weight_quantizer",
                "cfg": {"num_bits": (4, 3), "axis": None},
            }
        )
    elif lm_head_precision == "nvfp4":
        quant_cfg_list.append(
            {
                "quantizer_name": "*lm_head.input_quantizer",
                "cfg": {
                    "num_bits": (2, 1),
                    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                    "axis": None,
                },
                "enable": True,
            }
        )
        quant_cfg_list.append(
            {
                "quantizer_name": "*lm_head.weight_quantizer",
                "cfg": {
                    "num_bits": (2, 1),
                    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                    "axis": None,
                },
                "enable": True,
            }
        )
    quant_cfg["quant_cfg"] = quant_cfg_list
    return quant_cfg


def quantize(
    model, tokenizer, precision, lm_head_precision="fp16", dataset_dir=None, calib_size=512
):
    """Quantize the PyTorch model to fp8 or int4_awq."""
    assert precision in [
        "fp8",
        "int4_awq",
        "nvfp4",
    ], (
        f"Only fp8(W8A8), int4_awq(W4A16), nvfp4(W4A4) is supported. You passed an unsupported precision: {precision}."
    )

    assert lm_head_precision in ["fp16"], (
        f"Only fp16(unquantized) is supported for lm_head. You passed an unsupported precision: {lm_head_precision}."
    )

    if tokenizer.pad_token != "<unk>":  # nosec B105
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not dataset_dir:
        dataset_dir = "cnn_dailymail"

    batch_size = 1
    data_loader = get_dataset_dataloader(
        dataset_name=dataset_dir, tokenizer=tokenizer, batch_size=batch_size, num_samples=calib_size
    )
    quant_config = get_quant_config(precision, lm_head_precision)
    quantized_model = _quantize_model(model, quant_config, data_loader)
    mtq.print_quant_summary(quantized_model)
    return quantized_model
