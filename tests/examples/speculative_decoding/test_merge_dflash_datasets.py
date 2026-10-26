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

"""Tests for the DFlash JSONL merger recipe."""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
MERGER = REPO_ROOT / "examples/speculative_decoding/recipes/merge_dflash_datasets.py"


def write_record(path: Path, record_id: str, prompt: str, answer: str) -> None:
    """Write a one-turn DFlash-training conversation."""

    path.write_text(
        json.dumps(
            {
                "id": record_id,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_merge_keeps_synthetic_variants_together(tmp_path):
    """Single and parallel merges deduplicate matching variants from one synthetic shard."""

    synthetic_output = tmp_path / "synthetic-output"
    synthetic_output.mkdir()
    write_record(
        synthetic_output / "output-00000-00000-temp-0.0.jsonl",
        "duplicate-first",
        "Describe the image.",
        "The image shows a blue square.",
    )
    write_record(
        synthetic_output / "output-00000-00000-temp-0.1.jsonl",
        "duplicate-second",
        "Describe the image.",
        "The image shows a blue square.",
    )
    write_record(
        synthetic_output / "output-00001-00001-temp-0.0.jsonl",
        "synthetic-unique",
        "Describe the video.",
        "The video shows a red circle.",
    )
    curated_text = tmp_path / "curated-text.jsonl"
    write_record(curated_text, "curated-unique", "What is two plus two?", "Four.")
    expected_ids = {"duplicate-first", "synthetic-unique", "curated-unique"}
    for jobs in (1, 2):
        output = tmp_path / f"merged-{jobs}.jsonl"
        subprocess.run(
            [
                sys.executable,
                str(MERGER),
                "--source",
                f"synthetic={synthetic_output}",
                "--source",
                f"curated={curated_text}",
                "--output",
                str(output),
                "--jobs",
                str(jobs),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        assert {record["id"] for record in records} == expected_ids
        assert not list(tmp_path.glob(f".{output.name}.parallel-*"))


def test_parallel_merge_failure_preserves_work_dir_with_stale_output(tmp_path):
    """A failed overwrite retains worker artifacts even when prior output exists."""

    invalid_source = tmp_path / "invalid.jsonl"
    invalid_source.write_text("not valid JSON\n", encoding="utf-8")
    output = tmp_path / "merged.jsonl"
    output.write_text("stale output\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(MERGER),
            "--source",
            f"invalid={invalid_source}",
            "--output",
            str(output),
            "--jobs",
            "2",
            "--overwrite",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert output.read_text(encoding="utf-8") == "stale output\n"
    work_dirs = list(tmp_path.glob(f".{output.name}.parallel-*"))
    assert len(work_dirs) == 1
    assert list((work_dirs[0] / "manifests").glob("*.txt"))
