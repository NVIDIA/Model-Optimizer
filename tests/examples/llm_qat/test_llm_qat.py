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


import json

import pytest
import torch
from _test_utils.examples.run_command import run_example_command
from safetensors.torch import load_file

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


def _run_export(ckpt_dir: str, export_dir: str):
    run_example_command(
        [
            "python", "export.py",
            "--pyt_ckpt_path", ckpt_dir,
            "--export_path", export_dir,
        ],
        "llm_qat",
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
            "--recipe", "general/ptq/nvfp4_default-kv_fp8",
            "--calib_size", "64",
            "--output_dir", str(ptq_output_dir),
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
            "--recipe", "general/ptq/nvfp4_default-kv_fp8",
            "--calib_size", "64",
            "--output_dir", str(ptq_output_dir),
        ],
        cache_dir=cache_dir,
    )

    # Step 2: LoRA QAT
    lora_qat_output_dir = tmp_path / "lora_qat"
    _run_train(
        "configs/train/qat_nvfp4.yaml",
        [
            "--model_name_or_path", str(ptq_output_dir),
            "--do_train", "True",
            "--lora", "True",
            "--output_dir", str(lora_qat_output_dir),
        ],
        backend="fsdp2",
        cache_dir=cache_dir,
    )

    # Step 3: Export. This checkpoint is fake-quantized, so the calibrated amaxes rather than
    # packed weights are what must survive the load.
    export_dir = tmp_path / "lora_qat_export"
    _run_export(str(lora_qat_output_dir), str(export_dir))

    base_model_dir = export_dir / "base_model"
    with open(base_model_dir / "hf_quant_config.json") as f:
        assert json.load(f)["quantization"]["quant_algo"] == "NVFP4"

    base_weights = load_file(base_model_dir / "model.safetensors")
    assert not any("base_layer" in k or k.endswith("_amax") for k in base_weights)

    # LoRA freezes the base model, so a direct PTQ export is a trusted oracle for every calibrated
    # value. This catches scales that keep their key but were reset to defaults.
    ptq_export_dir = tmp_path / "ptq_export"
    _run_export(str(ptq_output_dir), str(ptq_export_dir))
    reference = load_file(ptq_export_dir / "model.safetensors")

    scales = [k for k in reference if k.endswith(("_scale", "_scale_2"))]
    assert scales, "no NVFP4 scales in the reference PTQ export"
    for key in scales:
        assert key in base_weights, f"{key} missing from the LoRA-QAT export"
        assert torch.equal(base_weights[key], reference[key]), f"{key} does not match PTQ export"


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
            "--recipe", "general/ptq/nvfp4_default-kv_fp8",
            "--calib_size", "64",
            "--output_dir", str(ptq_output_dir),
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
            "--recipe", "general/ptq/nvfp4_default-kv_fp8",
            "--calib_size", "64",
            "--compress", "True",
            "--output_dir", str(ptq_output_dir),
        ],
        cache_dir=cache_dir,
    )

    # Step 2: QLoRA training
    qlora_output_dir = tmp_path / "qlora"
    _run_train(
        "configs/train/qlora_nvfp4.yaml",
        [
            "--model_name_or_path", str(ptq_output_dir),
            "--do_train", "True",
            "--lora", "True",
            "--output_dir", str(qlora_output_dir),
        ],
        backend="ddp",
        cache_dir=cache_dir,
    )

    # Step 3: Export the QLoRA checkpoint for deployment
    export_dir = tmp_path / "qlora_export"
    _run_export(str(qlora_output_dir), str(export_dir))

    # The base model is exported compressed; the adapters stay at the top level.
    base_model_dir = export_dir / "base_model"
    assert (export_dir / "adapter_model.safetensors").is_file()
    assert (base_model_dir / "hf_quant_config.json").is_file()

    with open(base_model_dir / "hf_quant_config.json") as f:
        assert json.load(f)["quantization"]["quant_algo"] == "NVFP4"

    # NVFP4 needs the packed weight and *both* scales to be dequantizable downstream.
    base_weights = load_file(base_model_dir / "model.safetensors")
    packed_weights = [k for k, v in base_weights.items() if k.endswith(".weight") and v.dtype == torch.uint8]
    assert packed_weights, "no NVFP4-packed weights found in the exported base model"
    for key in packed_weights:
        prefix = key.removesuffix(".weight")
        assert f"{prefix}.weight_scale" in base_weights
        assert f"{prefix}.weight_scale_2" in base_weights
    assert not any("base_layer" in k or "lora" in k for k in base_weights)
