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

"""DFlash offline E2E regression tests.

Mirrors test_dflash.py but exercises the offline pipeline:
  1. Dump base-model hidden states from a slice of synthetic_conversations_1k.jsonl
     via examples/speculative_decoding/collect_hidden_states/compute_hidden_states_hf.py.
  2. Train DFlash with data.offline_data_path set (triggers _derive_dflash_offline,
     which deletes base-model layers post-convert to save memory).
  3. Verify loss decreases on the offline path.

Aux-layer ids 1,25 match build_target_layer_ids(num_orig_hidden_layers=28,
num_draft_layers=2) for Qwen3-0.6B (28 hidden layers); changing the base model
or draft layer count requires updating --aux-layers accordingly so the dumped
aux_hidden_states dim matches the draft module input.
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

# Match _DFLASH_OVERRIDES in test_dflash.py so the offline run is comparable to online.
_DFLASH_OVERRIDES = [
    f"data.chat_template={CHAT_TEMPLATE}",
    "training.training_seq_len=512",
    "training.per_device_train_batch_size=2",
    "training.logging_steps=50",
    "training.answer_only_loss=true",
    "dflash.dflash_block_size=8",
    "dflash.dflash_mask_token_id=151669",
    "dflash.dflash_use_torch_compile=False",
    "dflash.dflash_architecture_config.num_hidden_layers=2",
]

# Number of conversations to dump. Smaller than the full 1K to keep dump time
# bounded; large enough that loss-decrease becomes visible across logging_steps.
_DUMP_NUM_CONVERSATIONS = 200


@pytest.fixture(scope="session")
def qwen3_model_name():
    """Qwen3-0.6B model name (downloaded from HF on first use)."""
    return "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="session")
def dflash_offline_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("dflash_offline_output")


@pytest.fixture(scope="session")
def tagged_synth_data_path(dflash_offline_output_dir):
    """Tag each row of synthetic_conversations_1k.jsonl with a stable conversation_id.

    compute_hidden_states_hf.py uses conversation_id as the dump filename and
    resume/skip key, asserting it is non-null. The shared synthetic dataset
    only ships a `messages` field, so we materialize a tagged copy here.
    """
    tagged_path = dflash_offline_output_dir / "tagged_synth.jsonl"
    with open(SYNTH_DATA) as src, open(tagged_path, "w") as dst:
        for i, line in enumerate(src):
            entry = json.loads(line)
            entry.setdefault("conversation_id", f"{i:04d}")
            dst.write(json.dumps(entry) + "\n")
    return tagged_path


@pytest.fixture(scope="session")
def offline_hidden_states_dir(qwen3_model_name, dflash_offline_output_dir, tagged_synth_data_path):
    """Dump base-model hidden states once for the whole test module."""
    dump_dir = dflash_offline_output_dir / "hidden_states"
    run_example_command(
        [
            "python",
            "collect_hidden_states/compute_hidden_states_hf.py",
            "--model",
            qwen3_model_name,
            "--input-data",
            str(tagged_synth_data_path),
            "--output-dir",
            str(dump_dir),
            "--debug-max-num-conversations",
            str(_DUMP_NUM_CONVERSATIONS),
            # Two draft layers — matches build_target_layer_ids(28, 2) for Qwen3-0.6B.
            "--aux-layers",
            "1,25",
            "--answer-only-loss",
            "--chat-template",
            CHAT_TEMPLATE,
        ],
        "speculative_decoding",
    )
    pt_files = list(dump_dir.rglob("*.pt"))
    assert pt_files, f"No .pt files dumped under {dump_dir}"
    return dump_dir


def test_dflash_offline_training(
    qwen3_model_name, dflash_offline_output_dir, offline_hidden_states_dir
):
    """Train DFlash from dumped hidden states and validate loss decreases."""
    output_dir = str(dflash_offline_output_dir / "dflash-qwen3-0.6b-offline")
    overrides = [
        f"model.model_name_or_path={qwen3_model_name}",
        f"data.offline_data_path={offline_hidden_states_dir}",
        f"training.output_dir={output_dir}",
        # Two epochs over the dumped slice gives enough steps for two log entries
        # at logging_steps=50 with batch=2 (200/2 * 2 = 200 steps → 4 entries).
        "training.num_train_epochs=2",
        "training.save_steps=500",
        *_DFLASH_OVERRIDES,
    ]

    run_example_command(
        ["./launch_train.sh", "--config", DFLASH_YAML, *overrides],
        "speculative_decoding",
    )

    trainer_state = os.path.join(output_dir, "trainer_state.json")
    assert os.path.exists(trainer_state), "trainer_state.json not found"
    with open(trainer_state) as f:
        state = json.load(f)
    logs = [h for h in state.get("log_history", []) if "loss" in h]
    assert len(logs) >= 2, f"Expected at least 2 log entries, got {len(logs)}"

    first_loss = float(logs[0]["loss"])
    final_loss = float(logs[-1]["loss"])
    assert final_loss < first_loss, f"Loss did not decrease: {first_loss:.3f} -> {final_loss:.3f}"
    # Sanity ceiling — same threshold as the online regression. Offline trains
    # on fewer samples so we don't tighten it further here.
    assert final_loss < 5.0, f"Final loss {final_loss:.3f} too high (expected < 5.0)"
