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


import pytest
import torch
from _test_utils.examples.run_command import run_example_command


# fmt: off
def _run_command(extra_cmd_args: list[str]):
    run_example_command(
        [
            "./launch.sh",
            "--fsdp_transformer_layer_cls_to_wrap", "LlamaDecoderLayer",
            "--num_epochs", "0.3",
            "--lr", "1e-5",
            "--save_steps", "5",
            "--calib_size", "64",
            "--train_size", "256",
            "--eval_size", "64",
            *extra_cmd_args,
        ],
        "llm_qat",
        setup_free_port=True,
    )

@pytest.mark.parametrize("backend", [
    pytest.param("fsdp1", marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="need 2 GPUs!")),
    "fsdp2",
    "deepspeed",
    "ddp",
])
def test_llama_qat_int4w_int8a(tiny_llama_path, tmp_path, backend):
    ptq_output_dir = tmp_path / "ptq"
    qat_output_dir = tmp_path / "qat"

    # Run PTQ
    _run_command(
        [
            "--model", tiny_llama_path,
            "--do_train", "False",
            "--quant_cfg", "INT4_WEIGHT_INT8_ACTIVATIONS",
            "--output_dir", ptq_output_dir,
            "--backend", backend,
        ]
    )

    # Run QAT on PTQ checkpoint
    _run_command(
        [
            "--model", ptq_output_dir,
            "--do_train", "True",
            "--output_dir", qat_output_dir,
            "--backend", backend,
        ]
    )

@pytest.mark.parametrize("backend", [
    pytest.param("fsdp1", marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="need 2 GPUs!")),
    "fsdp2",
    "deepspeed",
    "ddp",
])
def test_llama_qat_int4w_int8a_direct_qat(tiny_llama_path, tmp_path, backend):
    # Run PTQ + QAT together
    _run_command(
        [
            "--model", tiny_llama_path,
            "--do_train", "True",
            "--quant_cfg", "INT4_WEIGHT_INT8_ACTIVATIONS",
            "--output_dir", tmp_path,
            "--backend", backend,
        ]
    )

def test_llama_lora_qat_nvfp4(tiny_llama_path, tmp_path):
    _run_command(
        [
            "--model", tiny_llama_path,
            "--do_train", "True",
            "--quant_cfg", "NVFP4_DEFAULT_CFG",
            "--lora", "True",
            "--output_dir", tmp_path / "lora_qat",
            "--backend", "fsdp2",
        ]
    )


def test_llama_qlora_nvfp4(tiny_llama_path, tmp_path):
    _run_command(
        [
            "--model", tiny_llama_path,
            "--do_train", "True",
            "--quant_cfg", "NVFP4_DEFAULT_CFG",
            "--lora", "True",
            "--compress", "True",
            "--output_dir", tmp_path / "qlora",
        ]
    )
