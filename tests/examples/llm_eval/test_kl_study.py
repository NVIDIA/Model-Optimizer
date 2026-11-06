# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Exercise frozen study inputs and result artifacts through the offline CLI."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "examples/llm_eval/kl_study.py"


def test_frozen_bank_preserves_eos_and_flags_context_limits(tmp_path):
    bank = tmp_path / "bank.jsonl"
    records = [
        {"id": "short", "source": "a", "prompt_ids": [1, 2], "generated_ids": [3, 0]},
        {"id": "long", "source": "b", "prompt_ids": [1] * 4, "generated_ids": [3, 4]},
    ]
    bank.write_text("".join(json.dumps(row) + "\n" for row in records))
    output = tmp_path / "validation.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "validate",
        "--input",
        str(bank),
        "--output",
        str(output),
        "--context_limit",
        "6",
        "--max_new_tokens",
        "4",
        "--eos_token_ids",
        "0",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["examples"][0]["generated_ids"] == [3, 0]
    assert result["examples"][0]["stop_reason"] == "eos"
    assert result["examples"][1]["context_limited"] is True
    assert result["examples"][1]["prompt_ids"] == [1] * 4
    original_identity = result["input_sha256"]
    records[0]["generated_ids"] = [3, 0, 4]
    bank.write_text("".join(json.dumps(row) + "\n" for row in records))
    invalid = subprocess.run(command, capture_output=True, text=True)
    assert invalid.returncode != 0
    assert "after EOS" in invalid.stderr
    records[0]["generated_ids"] = [4, 0]
    bank.write_text("".join(json.dumps(row) + "\n" for row in records))
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(output.read_text())["input_sha256"] != original_identity


def test_score_resume_is_bound_to_inputs_settings_and_model_bytes(tmp_path):
    bank = tmp_path / "bank.jsonl"
    bank.write_text(json.dumps({"id": "a", "prompt_ids": [1], "generated_ids": [2]}) + "\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"gpt2","eos_token_id":0}')
    output = tmp_path / "results.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "score",
        "--dry_run",
        "--input",
        str(bank),
        "--output",
        str(output),
        "--work_dir",
        str(tmp_path / "work"),
        "--context_limit",
        "8",
        "--max_new_tokens",
        "4",
        "--model",
        str(model),
        "--quantized_model",
        str(model),
        "--runner_sha256",
        "a" * 64,
        "--evaluator_commit",
        "b" * 40,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    first = output.read_text()
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert output.read_text() == first
    overlapping = subprocess.run(
        [*command, "--reference_devices", "0,1", "--candidate_devices", "1,2"],
        capture_output=True,
        text=True,
    )
    assert overlapping.returncode != 0
    assert "disjoint" in overlapping.stderr.lower()
    bank.write_text(
        json.dumps(
            {"id": "a", "prompt_ids": [1], "generated_ids": [2], "reference_sha256": "f" * 64}
        )
        + "\n"
    )
    wrong_reference = subprocess.run(
        [*command, "--work_dir", str(tmp_path / "other")], capture_output=True, text=True
    )
    assert wrong_reference.returncode != 0
    assert "reference checkpoint" in wrong_reference.stderr.lower()
    bank.write_text(json.dumps({"id": "a", "prompt_ids": [1], "generated_ids": [2]}) + "\n")
    (model / "config.json").write_text('{"model_type":"llama","eos_token_id":0}')
    invalid = subprocess.run(command, capture_output=True, text=True)
    assert invalid.returncode != 0
    assert "incompatible" in invalid.stderr.lower()


def test_score_derives_complete_reference_eos_and_records_commit(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "generation_config.json").write_text('{"eos_token_id":[0,3]}')
    bank = tmp_path / "bank.jsonl"
    bank.write_text(json.dumps({"id": "a", "prompt_ids": [1], "generated_ids": [2, 3]}) + "\n")
    output = tmp_path / "result.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "score",
        "--dry_run",
        "--model",
        str(model),
        "--quantized_model",
        str(model),
        "--input",
        str(bank),
        "--output",
        str(output),
        "--work_dir",
        str(tmp_path / "work"),
        "--context_limit",
        "8",
        "--max_new_tokens",
        "4",
        "--runner_sha256",
        "a" * 64,
        "--evaluator_commit",
        "b" * 40,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert report["examples"][0]["stop_reason"] == "eos"
    assert report["manifest"]["settings"]["eos_token_ids"] == [0, 3]
    assert report["manifest"]["settings"]["evaluator_commit"] == "b" * 40
    assert report["manifest"]["settings"]["evaluator_dirty"] is False
    incomplete = subprocess.run([*command, "--eos_token_ids", "0"], capture_output=True, text=True)
    assert incomplete.returncode != 0
    assert "complete" in incomplete.stderr.lower()
    bank.write_text(json.dumps({"id": "a", "prompt_ids": [1], "generated_ids": [3, 2]}) + "\n")
    after_eos = subprocess.run(command, capture_output=True, text=True)
    assert after_eos.returncode != 0
    assert "after eos" in after_eos.stderr.lower()
