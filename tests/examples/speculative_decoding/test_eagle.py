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
import os

import pytest
import safetensors.torch
import torch
from _test_utils.examples.run_command import run_example_command
from packaging.version import Version

from modelopt.torch.export.plugins.hf_spec_export import LLAMA_EAGLE_SINGLE_LAYER


@pytest.fixture(scope="module")
def eagle_output_dir(tmp_path_factory):
    """Eagle output directory shared in this module."""
    return tmp_path_factory.mktemp("eagle_output_dir")


@pytest.fixture(scope="module")
def draft_vocab_cache_dir(tmp_path_factory):
    """Eagle output directory shared in this module."""
    return tmp_path_factory.mktemp("eagle_output_dir")


def test_calibrate_draft_vocab(tiny_llama_path, tiny_daring_anteater_path, draft_vocab_cache_dir):
    """Test calibration of draft vocabulary."""
    run_example_command(
        [
            "python",
            "./scripts/calibrate_draft_vocab.py",
            "--model",
            tiny_llama_path,
            "--data",
            tiny_daring_anteater_path,
            "--draft_vocab_size",
            "100",
            "--save_dir",
            draft_vocab_cache_dir,
        ],
        "speculative_decoding",
    )

    model_name = os.path.basename(os.path.normpath(tiny_llama_path))
    d2t = torch.load(os.path.join(draft_vocab_cache_dir, model_name, "d2t.pt"))
    assert d2t.shape[0] == 100, f"Expected draft vocab size 100, got {d2t.shape[0]}"


# fmt: off
@pytest.mark.parametrize("cp_size", [1, 2])
def test_llama_eagle3(tiny_llama_path, tiny_daring_anteater_path, tmp_path, eagle_output_dir, cp_size):
    """Test Eagle3 training with a tiny llama model, using different cp_size values."""
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if cp_size == 2 and available_gpus < 2:
        pytest.skip("cp_size=2 requires at least 2 GPUs, but only {} found.".format(available_gpus))
    if cp_size == 2 and not Version(torch.__version__) >= Version("2.10.0"):
        pytest.skip("cp_size=2 requires torch 2.10.0")
    # Create an ultra-tiny EAGLE config for testing to reduce memory usage
    tiny_eagle_config = {
        "max_position_embeddings": 128,
        "num_hidden_layers": 1,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 64,
    }

    # Write the tiny config to a temporary file
    config_file = tmp_path / f"tiny_eagle_config_cp{cp_size}.json"
    with open(config_file, "w") as f:
        json.dump(tiny_eagle_config, f)

    run_example_command(
        [
            "./launch_train.sh",
            "--model", tiny_llama_path,
            "--data", tiny_daring_anteater_path,
            "--num_epochs", "0.25",
            "--lr", "1e-5",
            "--mode", "eagle3",
            "--eagle_config", str(config_file),
            "--output_dir", eagle_output_dir / f"eagle-tinyllama-cp{cp_size}",
            "--training_seq_len", "128", # Match max_position_embeddings
            "--cp_size", str(cp_size),
        ],
        "speculative_decoding",
    )


def test_resume_training(tiny_daring_anteater_path, eagle_output_dir):
    """Test resume training of Eagle3."""
    run_example_command(
        [
            "./launch_train.sh",
            "--model",  eagle_output_dir / "eagle-tinyllama-cp1",
            "--data", tiny_daring_anteater_path,
            "--num_epochs", "0.5",
            "--lr", "1e-5",
            "--mode", "eagle3",
            "--output_dir", eagle_output_dir / "eagle-tinyllama-cp1",
            "--training_seq_len", "128", # Match max_position_embeddings
        ],
        "speculative_decoding",
    )


def test_ar_validate(eagle_output_dir):
    """Test in-framework AR evaluation."""
    run_example_command(
        [
            "python", "./scripts/ar_validate.py",
            "--model_path", eagle_output_dir / "eagle-tinyllama-cp1",
            "--osl", "10",
            "--num_samples", "5",
            "--steps", "3"
        ],
        "speculative_decoding",
    )


def test_export_hf_checkpoint(eagle_output_dir):
    """Test export of Eagle3 checkpoint."""
    run_example_command(
        [
            "python", "./scripts/export_hf_checkpoint.py",
            "--model_path", eagle_output_dir / "eagle-tinyllama-cp1",
            "--export_path", eagle_output_dir / "eagle-tinyllama-export",
        ],
        "speculative_decoding",
    )
    # Check the exported checkpoints have required keys
    state_dict = safetensors.torch.load_file(eagle_output_dir / "eagle-tinyllama-export" / "model.safetensors")
    for required_key in LLAMA_EAGLE_SINGLE_LAYER["required"]:
        assert required_key in state_dict, f"Missing key '{required_key}' in state_dict"


def test_convert_to_vllm_ckpt(tiny_llama_path, eagle_output_dir):
    """Test conversion of Eagle3 checkpoint to VLLM one-model checkpoint."""
    run_example_command(
        [
            "python", "./scripts/convert_to_vllm_ckpt.py",
            "--input", eagle_output_dir / "eagle-tinyllama-export",
            "--verifier", tiny_llama_path,
            "--output", eagle_output_dir / "eagle-tinyllama-export-vllm-one-ckpt",
        ],
        "speculative_decoding",
    )
