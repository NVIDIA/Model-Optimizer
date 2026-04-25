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

# Fast training overrides (short runs with frequent eval)
FAST_TRAIN_ARGS = [
    "--model_max_length",
    "128",
    "--num_train_epochs",
    "1.0",
    "--save_steps",
    "5",
    "--eval_steps",
    "5",
]


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


def _run_quantize(config: str, extra_cmd_args: list[str], cache_dir: str = ""):
    run_example_command(
        [
            "python", "quantize.py",
            "--config", config,
            *_fast_data_args(cache_dir),
            *extra_cmd_args,
        ],
        "llm_qat",
    )


def _run_train(config: str, extra_cmd_args: list[str], backend: str = "fsdp2", cache_dir: str = ""):
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
            "--config", config,
            *_fast_data_args(cache_dir),
            *FAST_TRAIN_ARGS,
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
        "configs/train/qat_nvfp4.yaml",
        [
            "--model_name_or_path", tiny_qwen3_path,
            "--calib_size", "64",
            "--quantize_output_dir", str(ptq_output_dir),
        ],
        cache_dir=cache_dir,
    )

    # Step 2: QAT
    _run_train(
        "configs/train/qat_nvfp4.yaml",
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
        "configs/train/qat_nvfp4.yaml",
        [
            "--model_name_or_path", tiny_qwen3_path,
            "--calib_size", "64",
            "--quantize_output_dir", str(ptq_output_dir),
        ],
        cache_dir=cache_dir,
    )

    # Step 2: LoRA QAT
    _run_train(
        "configs/train/qat_nvfp4.yaml",
        [
            "--model_name_or_path", str(ptq_output_dir),
            "--do_train", "True",
            "--lora", "True",
            "--output_dir", str(tmp_path / "lora_qat"),
        ],
        backend="fsdp2",
        cache_dir=cache_dir,
    )


@pytest.mark.parametrize("backend", [
    "fsdp2",
    "deepspeed",
])
def test_qwen3_qad_nvfp4(tiny_qwen3_path, tmp_path, backend):
    ptq_output_dir = tmp_path / "ptq"
    qad_output_dir = tmp_path / "qad"
    cache_dir = str(tmp_path / "dataset_cache")

    # Step 1: Quantize student
    _run_quantize(
        "configs/train/qad_nvfp4.yaml",
        [
            "--model_name_or_path", tiny_qwen3_path,
            "--calib_size", "64",
            "--quantize_output_dir", str(ptq_output_dir),
        ],
        cache_dir=cache_dir,
    )

    # Step 2: QAD (quantization-aware distillation)
    _run_train(
        "configs/train/qad_nvfp4.yaml",
        [
            "--model_name_or_path", str(ptq_output_dir),
            "--do_train", "True",
            "--output_dir", str(qad_output_dir),
            "--distill", "True",
            "--teacher_model", tiny_qwen3_path,
        ],
        backend=backend,
        cache_dir=cache_dir,
    )


def test_qwen3_qlora_nvfp4(tiny_qwen3_path, tmp_path):
    ptq_output_dir = tmp_path / "ptq"
    cache_dir = str(tmp_path / "dataset_cache")

    # Step 1: Quantize with compression for QLoRA
    _run_quantize(
        "configs/train/qlora_nvfp4.yaml",
        [
            "--model_name_or_path", tiny_qwen3_path,
            "--calib_size", "64",
            "--compress", "True",
            "--quantize_output_dir", str(ptq_output_dir),
        ],
        cache_dir=cache_dir,
    )

    # Step 2: QLoRA training
    _run_train(
        "configs/train/qlora_nvfp4.yaml",
        [
            "--model_name_or_path", str(ptq_output_dir),
            "--do_train", "True",
            "--lora", "True",
            "--output_dir", str(tmp_path / "qlora"),
        ],
        backend="ddp",
        cache_dir=cache_dir,
    )
