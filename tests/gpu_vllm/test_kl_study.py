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

"""Real-runtime control; point KL_STUDY_MODEL at an unquantized local checkpoint."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.environ.get("KL_STUDY_MODEL"), reason="Requires a local GPU checkpoint")
@pytest.mark.timeout(1200)
def test_bounded_capture_matches_dense_scores_and_resumes(tmp_path):
    script = Path(__file__).resolve().parents[2] / "examples/llm_eval/kl_study.py"
    bank = tmp_path / "input.jsonl"
    bank.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"id": "eos", "prompt_ids": [1, 2, 3], "generated_ids": [0]},
                {"id": "chunked", "prompt_ids": [1] * 19, "generated_ids": [4] * 8 + [0]},
            ]
        )
    )
    output = tmp_path / "output.json"
    command = [
        sys.executable,
        str(script),
        "score",
        "--input",
        str(bank),
        "--output",
        str(output),
        "--model",
        os.environ["KL_STUDY_MODEL"],
        "--quantized_model",
        os.environ["KL_STUDY_MODEL"],
        "--context_limit",
        "64",
        "--max_new_tokens",
        "16",
        "--eos_token_ids",
        "0",
        "--prefill_chunk_size",
        "8",
        "--work_dir",
        str(tmp_path / "work"),
        "--top_ks",
        "1",
        "2",
        "4",
        "--audit",
        "--runner_sha256",
        os.environ["KL_STUDY_RUNNER_SHA256"],
    ]
    subprocess.run(command, check=True)
    report = json.loads(output.read_text())
    assert report["examples"][0]["metrics"]["full_vocab_kl"] == pytest.approx([0], abs=1e-6)
    assert report["examples"][0]["dense_max_error"] < 1e-5
    assert report["examples"][0]["stop_reason"] == "eos"
    assert report["examples"][1]["metrics"]["full_vocab_kl"] == pytest.approx([0] * 9, abs=1e-6)
    original = output.read_text()
    subprocess.run(command, check=True)
    assert output.read_text() == original
