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
import math
from pathlib import Path

import pytest
import torch
from _test_utils.examples.readme import readme_command
from _test_utils.examples.run_command import run_example_command
from _test_utils.fs_utils import assert_unmodified_tree
from _test_utils.torch.transformers_models import create_tiny_qwen3_dir
from datasets import load_from_disk
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
README = REPO / "examples/llm_qat/README.md"
README_SCENARIOS = {
    "configs/train/qat_nvfp4.yaml": "llm-qat-quickstart",
    "configs/train/qad_nvfp4.yaml": "llm-qad-quickstart",
    "configs/train/qlora_nvfp4.yaml": "llm-qlora-quickstart",
}

# Mapping from backend name to accelerate config file
BACKEND_CONFIGS = {
    "fsdp2": "configs/accelerate/fsdp2.yaml",
    "ddp": "configs/accelerate/ddp.yaml",
    "deepspeed": "configs/accelerate/deepspeed.yaml",
}

# Backends that need gradient checkpointing
GRADIENT_CHECKPOINTING_BACKENDS = {"ddp", "deepspeed"}

# PR runners have one GPU, so a backend case only exercises the wrapper and ModelOpt's patches, not
# sharding. PRs run fsdp2 (the default, with the most ModelOpt-specific code); the other backends are
# marked nightly. QLoRA still trains on ddp on PRs.

# Fast training overrides (short runs with frequent eval)
FAST_TRAIN_ARGS = {
    "model_max_length": "128",
    "num_train_epochs": "1.0",
    "max_steps": "2",
    "save_steps": "1",
    "eval_steps": "1",
}


def _assert_training_progress(output):
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 2
    losses = [
        entry[key]
        for entry in state["log_history"]
        for key in ("loss", "eval_loss", "train_loss")
        if key in entry
    ]
    assert losses and all(math.isfinite(value) for value in losses)
    assert any("train_loss" in entry for entry in state["log_history"])


def _assert_qat_export(ptq, trained, exported):
    _assert_training_progress(trained)
    assert (trained / "modelopt_state.pth").stat().st_size > 0
    before = load_file(str(Path(ptq) / "model.safetensors"))
    after = load_file(str(trained / "model.safetensors"))
    keys = [key for key in before.keys() & after.keys() if key.endswith("weight")]
    assert keys and any(not torch.equal(before[key], after[key]) for key in keys)
    config = json.loads((exported / "hf_quant_config.json").read_text())
    assert config["quantization"]["quant_algo"] == "NVFP4"
    assert any(
        value.dtype == torch.uint8
        for value in load_file(str(exported / "model.safetensors")).values()
    )


def _fast_data_args(cache_dir: str):
    """Fast dataset overrides for all tests (small samples, no shuffle, temp cache)."""
    return {
        "dataset_config": "configs/dataset/blend_test.yaml",
        "train_samples": "64",
        "eval_samples": "16",
        "shuffle": "False",
        "dataset_cache_dir": cache_dir,
    }


def _readme_command(config, script, overrides):
    return readme_command(
        README,
        README_SCENARIOS[config],
        script,
        overrides=overrides,
        cwd_overrides={"llm-qat-quickstart": REPO},
    )


def _run_quantize(config: str, overrides, cache_dir: str = ""):
    run_example_command(
        _readme_command(config, "quantize.py", {**_fast_data_args(cache_dir), **overrides}),
        "llm_qat",
    )


def _run_train(config: str, overrides, backend: str = "fsdp2", cache_dir: str = ""):
    args = {
        "config-file": BACKEND_CONFIGS[backend],
        **_fast_data_args(cache_dir),
        **FAST_TRAIN_ARGS,
    }
    if backend in GRADIENT_CHECKPOINTING_BACKENDS:
        args["gradient_checkpointing"] = "True"
    run_example_command(
        _readme_command(config, "train.py", {**args, **overrides}),
        "llm_qat",
        setup_free_port=True,
    )


def _run_export(ckpt_dir: str, export_dir: str, config="configs/train/qat_nvfp4.yaml"):
    run_example_command(
        _readme_command(
            config,
            "export.py",
            {
                "pyt_ckpt_path": ckpt_dir,
                "export_path": export_dir,
            },
        ),
        "llm_qat",
    )


@pytest.fixture(scope="module")
def dataset_cache_dir(tmp_path_factory):
    """Tokenized-dataset cache for every quantize/train run below.

    The cache key covers the tokenizer, max length and blend (not the command), so the dataset is
    built once for quantize.py (8192) and once for train.py (128) instead of once per command.
    """
    return str(tmp_path_factory.mktemp("dataset_cache"))


@pytest.fixture(scope="module")
def tied_qwen3_path(tmp_path_factory):
    """Tiny Qwen3 with tied embeddings, which accelerate 1.14 could not shard with FSDP2."""
    model_dir = create_tiny_qwen3_dir(
        tmp_path_factory.mktemp("tied_qwen3"),
        with_tokenizer=True,
        hidden_size=512,
        intermediate_size=512,
        tie_word_embeddings=True,
    )
    with assert_unmodified_tree(model_dir) as path:
        yield str(path)


@pytest.fixture(scope="module")
def nvfp4_ptq_dir(tied_qwen3_path, tmp_path_factory, dataset_cache_dir):
    """NVFP4 PTQ checkpoint shared by the QAT and QAD tests, which only read it.

    The QAT and QAD README quickstarts use the same PTQ command, so one run serves both.
    """
    ptq_dir = tmp_path_factory.mktemp("ptq")
    _run_quantize(
        "configs/train/qat_nvfp4.yaml",
        {"model_name_or_path": tied_qwen3_path, "calib_size": "64", "output_dir": str(ptq_dir)},
        cache_dir=dataset_cache_dir,
    )
    with assert_unmodified_tree(ptq_dir) as path:
        yield str(path)


def test_dataset_utils_pretokenize(tiny_qwen3_path, tmp_path):
    """Test dataset_utils.py standalone CLI pre-tokenization."""
    cache_dir = tmp_path / "dataset_cache"
    run_example_command(
        readme_command(
            README,
            "llm-qat-dataset",
            "dataset_utils.py",
            overrides={**_fast_data_args(str(cache_dir)), "model_name_or_path": tiny_qwen3_path},
        ),
        "llm_qat",
    )
    assert cache_dir.exists(), "Cache directory should be created"
    assert any(cache_dir.iterdir()), "Cache directory should contain tokenized data"
    caches = list(cache_dir.glob("*/dataset_dict.json"))
    assert len(caches) == 1
    cached = load_from_disk(str(caches[0].parent))
    assert len(cached["train"]) > 0 and len(cached["eval"]) > 0
    assert {"input_ids", "attention_mask", "labels"} <= set(cached["train"].column_names)


@pytest.mark.parametrize(
    "backend",
    [
        "fsdp2",
        pytest.param("deepspeed", marks=pytest.mark.nightly),
        pytest.param("ddp", marks=pytest.mark.nightly),
    ],
)
def test_qwen3_qat_nvfp4(nvfp4_ptq_dir, dataset_cache_dir, tmp_path, backend):
    qat_output_dir = tmp_path / "qat"

    _run_train(
        "configs/train/qat_nvfp4.yaml",
        {"model_name_or_path": nvfp4_ptq_dir, "output_dir": str(qat_output_dir)},
        backend=backend,
        cache_dir=dataset_cache_dir,
    )
    export_dir = tmp_path / "qat_export"
    _run_export(str(qat_output_dir), str(export_dir))
    _assert_qat_export(nvfp4_ptq_dir, qat_output_dir, export_dir)


@pytest.mark.skip(reason="FSDP2 LoRA checkpoint save omits adapter_model.safetensors")
def test_qwen3_lora_qat_nvfp4(tiny_qwen3_path, tmp_path):
    ptq_output_dir = tmp_path / "ptq"
    cache_dir = str(tmp_path / "dataset_cache")

    # Step 1: Quantize
    _run_quantize(
        "configs/train/qat_nvfp4.yaml",
        {
            "model_name_or_path": tiny_qwen3_path,
            "calib_size": "64",
            "output_dir": str(ptq_output_dir),
        },
        cache_dir=cache_dir,
    )

    # Step 2: LoRA QAT
    lora_qat_output_dir = tmp_path / "lora_qat"
    _run_train(
        "configs/train/qat_nvfp4.yaml",
        {
            "model_name_or_path": str(ptq_output_dir),
            "do_train": "True",
            "lora": "True",
            "output_dir": str(lora_qat_output_dir),
        },
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


@pytest.mark.parametrize(
    "backend",
    [
        "fsdp2",
        pytest.param("deepspeed", marks=pytest.mark.nightly),
    ],
)
def test_qwen3_qad_nvfp4(tied_qwen3_path, nvfp4_ptq_dir, dataset_cache_dir, tmp_path, backend):
    qad_output_dir = tmp_path / "qad"

    # QAD (quantization-aware distillation) of the shared quantized student
    _run_train(
        "configs/train/qad_nvfp4.yaml",
        {
            "model_name_or_path": nvfp4_ptq_dir,
            "output_dir": str(qad_output_dir),
            "teacher_model": tied_qwen3_path,
        },
        backend=backend,
        cache_dir=dataset_cache_dir,
    )
    export_dir = tmp_path / "qad_export"
    _run_export(str(qad_output_dir), str(export_dir), config="configs/train/qad_nvfp4.yaml")
    _assert_qat_export(nvfp4_ptq_dir, qad_output_dir, export_dir)


def test_qwen3_qlora_nvfp4(tiny_qwen3_path, dataset_cache_dir, tmp_path):
    ptq_output_dir = tmp_path / "ptq"

    # Step 1: Quantize with compression for QLoRA (its own checkpoint: the others are not compressed)
    _run_quantize(
        "configs/train/qlora_nvfp4.yaml",
        {
            "model_name_or_path": tiny_qwen3_path,
            "calib_size": "64",
            "output_dir": str(ptq_output_dir),
        },
        cache_dir=dataset_cache_dir,
    )

    # Step 2: QLoRA training
    qlora_output_dir = tmp_path / "qlora"
    _run_train(
        "configs/train/qlora_nvfp4.yaml",
        {"model_name_or_path": str(ptq_output_dir), "output_dir": str(qlora_output_dir)},
        backend="ddp",
        cache_dir=dataset_cache_dir,
    )

    _assert_training_progress(qlora_output_dir)

    # Step 3: Export the QLoRA checkpoint for deployment
    export_dir = tmp_path / "qlora_export"
    _run_export(str(qlora_output_dir), str(export_dir), config="configs/train/qlora_nvfp4.yaml")

    # The base model is exported compressed; the adapters stay at the top level.
    base_model_dir = export_dir / "base_model"
    assert (export_dir / "adapter_model.safetensors").is_file()
    assert (base_model_dir / "hf_quant_config.json").is_file()

    with open(base_model_dir / "hf_quant_config.json") as f:
        assert json.load(f)["quantization"]["quant_algo"] == "NVFP4"

    # NVFP4 needs the packed weight and *both* scales to be dequantizable downstream.
    base_weights = load_file(base_model_dir / "model.safetensors")
    packed_weights = [
        k for k, v in base_weights.items() if k.endswith(".weight") and v.dtype == torch.uint8
    ]
    assert packed_weights, "no NVFP4-packed weights found in the exported base model"
    for key in packed_weights:
        prefix = key.removesuffix(".weight")
        assert f"{prefix}.weight_scale" in base_weights
        assert f"{prefix}.weight_scale_2" in base_weights
    assert not any("base_layer" in k or "lora" in k for k in base_weights)
