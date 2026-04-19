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
from _test_utils.examples.run_command import run_example_command

# Mapping from backend name to accelerate config file
BACKEND_CONFIGS = {
    "fsdp2": "configs/accelerate/fsdp2.yaml",
    "ddp": "configs/accelerate/ddp.yaml",
    "deepspeed": "configs/accelerate/deepspeed.yaml",
}

# Backends that need gradient checkpointing
GRADIENT_CHECKPOINTING_BACKENDS = {"ddp", "deepspeed"}


# fmt: off
def _fast_data_args(cache_dir: str) -> list[str]:
    """Fast dataset overrides for all tests (small samples, no shuffle, temp cache)."""
    return [
        "--dataset_config", "configs/dataset/blend_test.yaml",
        "--train_samples", "64",
        "--eval_samples", "16",
        "--shuffle", "False",
        "--dataset_cache_dir", cache_dir,
    ]


def _run_quantize(extra_cmd_args: list[str], cache_dir: str = ""):
    run_example_command(
        [
            "python", "quantize.py",
            *_fast_data_args(cache_dir),
            *extra_cmd_args,
        ],
        "llm_qat",
    )


def _run_train(extra_cmd_args: list[str], backend: str = "fsdp2", cache_dir: str = ""):
    config_file = BACKEND_CONFIGS[backend]
    gradient_args = (
        ["--gradient_checkpointing", "True"]
        if backend in GRADIENT_CHECKPOINTING_BACKENDS
        else []
    )
    run_example_command(
        [
            "accelerate", "launch",
            "--config-file", config_file,
            "train.py",
            *_fast_data_args(cache_dir),
            "--num_train_epochs", "0.3",
            "--learning_rate", "1e-5",
            "--per_device_train_batch_size", "2",
            "--per_device_eval_batch_size", "2",
            "--save_steps", "5",
            "--eval_steps", "5",
            *gradient_args,
            *extra_cmd_args,
        ],
        "llm_qat",
        setup_free_port=True,
    )

def test_dataset_utils_pretokenize(tiny_qwen3_path, tmp_path):
    """Test dataset_utils.py standalone CLI pre-tokenization."""
    cache_dir = tmp_path / "dataset_cache"
    run_example_command(
        [
            "python", "dataset_utils.py",
            *_fast_data_args(str(cache_dir)),
            "--model_name_or_path", tiny_qwen3_path,
        ],
        "llm_qat",
    )
    assert cache_dir.exists(), "Cache directory should be created"
    assert any(cache_dir.iterdir()), "Cache directory should contain tokenized data"


@pytest.mark.parametrize("backend", [
    "fsdp2",
    "deepspeed",
    "ddp",
])
def test_qwen3_qat_nvfp4(tiny_qwen3_path, tmp_path, backend):
    ptq_output_dir = tmp_path / "ptq"
    qat_output_dir = tmp_path / "qat"
    cache_dir = str(tmp_path / "dataset_cache")

    # Step 1: Quantize
    _run_quantize(
        [
            "--model_name_or_path", tiny_qwen3_path,
            "--recipe", "general/ptq/nvfp4_default-fp8_kv",
            "--calib_size", "64",
            "--quantize_output_dir", str(ptq_output_dir),
        ],
        cache_dir=cache_dir,
    )

    # Step 2: QAT
    _run_train(
        [
            "--model_name_or_path", str(ptq_output_dir),
            "--do_train", "True",
            "--output_dir", str(qat_output_dir),
        ],
        backend=backend,
        cache_dir=cache_dir,
    )

def test_qwen3_lora_qat_nvfp4(tiny_qwen3_path, tmp_path):
    ptq_output_dir = tmp_path / "ptq"
    cache_dir = str(tmp_path / "dataset_cache")

    # Step 1: Quantize
    _run_quantize(
        [
            "--model_name_or_path", tiny_qwen3_path,
            "--recipe", "general/ptq/nvfp4_default-fp8_kv",
            "--calib_size", "64",
            "--quantize_output_dir", str(ptq_output_dir),
        ],
        cache_dir=cache_dir,
    )

    # Step 2: LoRA QAT
    _run_train(
        [
            "--model_name_or_path", str(ptq_output_dir),
            "--do_train", "True",
            "--lora", "True",
            "--output_dir", str(tmp_path / "lora_qat"),
        ],
        backend="fsdp2",
        cache_dir=cache_dir,
    )


def test_qwen3_qlora_nvfp4(tiny_qwen3_path, tmp_path):
    ptq_output_dir = tmp_path / "ptq"
    cache_dir = str(tmp_path / "dataset_cache")

    # Step 1: Quantize with compression for QLoRA
    _run_quantize(
        [
            "--model_name_or_path", tiny_qwen3_path,
            "--recipe", "general/ptq/nvfp4_default-fp8_kv",
            "--calib_size", "64",
            "--compress", "True",
            "--quantize_output_dir", str(ptq_output_dir),
        ],
        cache_dir=cache_dir,
    )

    # Step 2: QLoRA training
    _run_train(
        [
            "--model_name_or_path", str(ptq_output_dir),
            "--do_train", "True",
            "--lora", "True",
            "--output_dir", str(tmp_path / "qlora"),
        ],
        backend="ddp",
        cache_dir=cache_dir,
    )
