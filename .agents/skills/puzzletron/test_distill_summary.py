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

# Generated with Codex

"""Tests for the Puzzletron distillation-run summary."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).with_name("distill_summary.py")
SPEC = importlib.util.spec_from_file_location("distill_summary", SCRIPT_PATH)
assert SPEC and SPEC.loader
distill_summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(distill_summary)


def _write_run(run_dir: Path, data_paths: list[str], accuracy: float):
    run_dir.mkdir(parents=True)
    arguments = {
        "data_paths": repr(data_paths),
        "eval_interval": "10",
        "eval_iters": "10",
        "gbs": "8",
        "kd_loss_scale": "1.0",
        "log_interval": "1",
        "lr": "0.0001",
        "lr_warmup_iters": "10",
        "mbs": "1",
        "min_lr": "1e-05",
        "no_skip_lm_loss": "False",
        "pp_size": "1",
        "cp_size": "1",
        "ep_size": "1",
        "seed": "1234",
        "seq_length": "4096",
        "student_hf_path": "/models/student",
        "teacher_hf_path": "/models/teacher",
        "tp_size": "8",
        "train_iters": "100",
    }
    body = "\n".join(f"{key:<36}{value}" for key, value in arguments.items())
    (run_dir / "log.txt").write_text(
        f"==================== Arguments ====================\n{body}\n"
        "===================================================\n"
    )
    for iteration in (60, 70, 80, 90, 100):
        (run_dir / "checkpoints" / f"iter_{iteration:07d}").mkdir(parents=True)
    result_dir = run_dir / "hf/eval_results/mmlu/model"
    result_dir.mkdir(parents=True)
    (run_dir / "hf/model.safetensors").write_bytes(b"weights")
    (result_dir / "results_full.json").write_text(
        json.dumps(
            {
                "config": {"limit": None},
                "results": {"mmlu": {"acc,none": accuracy}},
            }
        )
    )
    (result_dir / "results_limited.json").write_text(
        json.dumps(
            {
                "config": {"limit": 10},
                "results": {"mmlu": {"acc,none": 0.99}},
            }
        )
    )


def test_summary_reports_recipes_results_and_matched_groups(tmp_path, capsys):
    """Report datasets, full MMLU results, and matched recipe groups."""
    puzzle_dir = tmp_path / "puzzle_dir_model"
    distillation_dir = puzzle_dir / "distillation"
    _write_run(
        distillation_dir / "wiki",
        ["1.0", "/data/Salesforce--wikitext_wikitext-103-v1_train_text"],
        0.61,
    )
    _write_run(
        distillation_dir / "nemotron",
        [
            "1.0",
            "/data/nvidia--Nemotron-Post-Training-Dataset-v2_default_math_messages_max100000",
            "1.0",
            "/data/nvidia--Nemotron-Post-Training-Dataset-v2_default_stem_messages_max100000",
        ],
        0.58,
    )

    distill_summary.print_summary(puzzle_dir)

    output = capsys.readouterr().out
    assert "WikiText-103" in output
    assert "Nemotron v2: math+stem" in output
    assert "60-100/10" in output
    assert "0.6100" in output
    assert "0.5800" in output
    assert "0.9900" not in output
    assert "Matched-recipe groups" in output
    assert "nemotron, wiki" in output


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("None", []),
        ("['1.0', '/data/wiki']", ["/data/wiki"]),
    ],
)
def test_parse_data_paths(value, expected):
    """Parse serialized data paths and empty argument values."""
    assert distill_summary.parse_data_paths(value) == expected
