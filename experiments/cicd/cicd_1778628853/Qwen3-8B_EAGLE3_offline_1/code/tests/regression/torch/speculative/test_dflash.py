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

"""DFlash E2E regression tests.

Tests the full DFlash pipeline using Qwen3-0.6B and the synthetic dataset
(examples/dataset/synthetic_conversations_1k.jsonl). Matches the configuration
in tools/launcher/examples/Qwen/Qwen3-0.6B/hf_online_dflash.yaml.

Convergence baseline (from L40 run):
  Step  100 (epoch 0.2): loss=6.59  acc=0.079
  Step  500 (epoch 1.0): loss=1.78  acc=0.525
  Step 1500 (epoch 3.0): loss=1.11  acc=0.595
"""

import json
import os

import pytest
from _test_utils.examples.run_command import MODELOPT_ROOT, run_example_command

DFLASH_YAML = str(
    MODELOPT_ROOT / "modelopt_recipes" / "general" / "speculative_decoding" / "dflash.yaml"
)

CHAT_TEMPLATE = str(
    MODELOPT_ROOT
    / "tools"
    / "launcher"
    / "examples"
    / "Qwen"
    / "Qwen3-0.6B"
    / "chat_template_train.jinja"
)

SYNTH_DATA = str(MODELOPT_ROOT / "examples" / "dataset" / "synthetic_conversations_1k.jsonl")

# Match tools/launcher/examples/Qwen/Qwen3-0.6B/hf_online_dflash.yaml
_DFLASH_OVERRIDES = [
    f"data.data_path={SYNTH_DATA}",
    f"data.chat_template={CHAT_TEMPLATE}",
    "training.training_seq_len=512",
    "training.per_device_train_batch_size=2",
    "training.logging_steps=100",
    "training.answer_only_loss=true",
    "dflash.dflash_block_size=8",
    "dflash.dflash_mask_token_id=151669",
    "dflash.dflash_use_torch_compile=False",
    "dflash.dflash_architecture_config.num_hidden_layers=2",
]


@pytest.fixture(scope="session")
def qwen3_model_name():
    """Qwen3-0.6B model name (downloaded from HF on first use)."""
    return "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="session")
def dflash_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("dflash_output")


def test_dflash_training(qwen3_model_name, dflash_output_dir):
    """Train DFlash on Qwen3-0.6B and validate loss convergence."""
    output_dir = str(dflash_output_dir / "dflash-qwen3-0.6b")
    overrides = [
        f"model.model_name_or_path={qwen3_model_name}",
        f"training.output_dir={output_dir}",
        "training.num_train_epochs=3",
        "training.save_steps=500",
        *_DFLASH_OVERRIDES,
    ]

    run_example_command(
        ["./launch_train.sh", "--config", DFLASH_YAML, *overrides],
        "speculative_decoding",
    )

    # Verify checkpoint was saved
    assert os.path.exists(os.path.join(output_dir, "modelopt_state.pth")) or any(
        "checkpoint-" in d
        for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    )

    # Regression: verify loss decreased
    trainer_state = os.path.join(output_dir, "trainer_state.json")
    assert os.path.exists(trainer_state), "trainer_state.json not found"
    with open(trainer_state) as f:
        state = json.load(f)
    logs = [h for h in state.get("log_history", []) if "loss" in h]
    assert len(logs) >= 2, f"Expected at least 2 log entries, got {len(logs)}"

    first_loss = float(logs[0]["loss"])
    final_loss = float(logs[-1]["loss"])
    assert final_loss < first_loss, f"Loss did not decrease: {first_loss:.3f} -> {final_loss:.3f}"
    # Sanity: final loss should be reasonable (baseline: ~1.1 on L40)
    assert final_loss < 3.0, f"Final loss {final_loss:.3f} too high (expected < 3.0)"


def test_dflash_resume(qwen3_model_name, dflash_output_dir):
    """Resume DFlash training from checkpoint."""
    output_dir = str(dflash_output_dir / "dflash-qwen3-0.6b")
    overrides = [
        f"model.model_name_or_path={qwen3_model_name}",
        f"training.output_dir={output_dir}",
        "training.num_train_epochs=4",
        "training.save_steps=5000",
        *_DFLASH_OVERRIDES,
    ]

    run_example_command(
        ["./launch_train.sh", "--config", DFLASH_YAML, *overrides],
        "speculative_decoding",
    )


def test_dflash_export(dflash_output_dir):
    """Export DFlash checkpoint to deployment format."""
    output_dir = str(dflash_output_dir / "dflash-qwen3-0.6b")
    export_dir = str(dflash_output_dir / "dflash-export")

    run_example_command(
        [
            "python",
            "./scripts/export_hf_checkpoint.py",
            "--model_path",
            output_dir,
            "--export_path",
            export_dir,
        ],
        "speculative_decoding",
    )

    assert os.path.exists(os.path.join(export_dir, "model.safetensors"))
    assert os.path.exists(os.path.join(export_dir, "config.json"))

    with open(os.path.join(export_dir, "config.json")) as f:
        config = json.load(f)
    assert config["architectures"] == ["DFlashDraftModel"]
    assert config["model_type"] == "qwen3"
    assert "dflash_config" in config
    assert "block_size" in config


def test_dflash_ar_validate(dflash_output_dir):
    """AR validation on trained DFlash checkpoint."""
    output_dir = str(dflash_output_dir / "dflash-qwen3-0.6b")

    run_example_command(
        [
            "python",
            "./scripts/ar_validate.py",
            "--model_path",
            output_dir,
            "--osl",
            "10",
            "--num_samples",
            "3",
            "--steps",
            "7",
        ],
        "speculative_decoding",
    )
