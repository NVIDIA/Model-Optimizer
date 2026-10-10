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

"""Tests for the vLLM skip-softmax calibration driver."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "vllm_serve"


@pytest.fixture
def calibration_driver(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("calibrate_sparse_attn")


@pytest.mark.parametrize(
    "flags",
    [
        ["--target_sparse_ratio", "-0.1"],
        ["--target_sparse_ratio", "1.1"],
        ["--target_sparse_ratio", "nan"],
        ["--decode_tokens", "-1"],
        ["--engine_kwargs", "[]"],
        ["--engine_kwargs", "not-json"],
        ["--engine_kwargs", '{"model": "/other"}'],
        ["--engine_kwargs", '{"worker_cls": "other.Worker"}'],
        ["--engine_kwargs", '{"enforce_eager": false}'],
        ["--engine_kwargs", '{"enable_prefix_caching": true}'],
        ["--engine_kwargs", '{"enable_chunked_prefill": true}'],
        ["--engine_kwargs", '{"max_num_seqs": 8}'],
        ["--engine_kwargs", '{"seed": 1}'],
        ["--engine_kwargs", '{"pipeline_parallel_size": 2}'],
        ["--engine_kwargs", '{"data_parallel_size": 2}'],
    ],
)
def test_parser_rejects_invalid_inputs_before_engine_start(calibration_driver, flags):
    with pytest.raises(SystemExit):
        calibration_driver._build_parser().parse_args(
            ["/checkpoint", "--validation_prompts_file", "held_out.txt", *flags]
        )


def test_parser_accepts_safe_engine_kwargs(calibration_driver):
    args = calibration_driver._build_parser().parse_args(
        [
            "/checkpoint",
            "--validation_prompts_file",
            "held_out.txt",
            "--engine_kwargs",
            '{"enable_expert_parallel": true}',
        ]
    )
    assert args.engine_kwargs == {"enable_expert_parallel": True}


def test_load_prompts_reads_nonempty_lines(calibration_driver, tmp_path):
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text(" first prompt\n\nsecond prompt \n")
    args = SimpleNamespace(prompts_file=str(prompts_file))

    assert calibration_driver._load_prompts(None, args) == ["first prompt", "second prompt"]


@pytest.mark.parametrize("prompts_contents", [None, "\n\n"])
def test_preflight_rejects_invalid_prompts_before_engine_start(
    calibration_driver, tmp_path, prompts_contents
):
    prompts_file = tmp_path / "prompts.txt"
    if prompts_contents is not None:
        prompts_file.write_text(prompts_contents)
    parser = calibration_driver._build_parser()
    args = parser.parse_args(
        [
            "/checkpoint",
            "--validation_prompts_file",
            "held_out.txt",
            "--prompts_file",
            str(prompts_file),
        ]
    )

    with pytest.raises(SystemExit):
        calibration_driver._preflight_prompt_inputs(args, parser)


def test_preflight_requires_ruler_data_before_engine_start(calibration_driver):
    parser = calibration_driver._build_parser()
    args = parser.parse_args(["/checkpoint", "--validation_prompts_file", "held_out.txt"])

    with pytest.raises(SystemExit):
        calibration_driver._preflight_prompt_inputs(args, parser)


def test_preflight_validates_ruler_essay_files_before_engine_start(calibration_driver, tmp_path):
    parser = calibration_driver._build_parser()
    args = parser.parse_args(
        [
            "/checkpoint",
            "--validation_prompts_file",
            "held_out.txt",
            "--calib_data_dir",
            str(tmp_path),
        ]
    )
    with pytest.raises(SystemExit):
        calibration_driver._preflight_prompt_inputs(args, parser)

    essays = tmp_path / "essays"
    essays.mkdir()
    (essays / "sample.txt").write_text("essay")
    assert calibration_driver._preflight_prompt_inputs(args, parser) is None


def test_existing_sparse_config_reads_only_dict(calibration_driver, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    config_path.write_text(json.dumps({"sparse_attention_config": {"config_groups": {}}}))
    assert calibration_driver._existing_sparse_config(str(checkpoint)) == {"config_groups": {}}

    config_path.write_text(json.dumps({"sparse_attention_config": ["invalid"]}))
    assert calibration_driver._existing_sparse_config(str(checkpoint)) is None


@pytest.mark.parametrize("update_checkpoint", [False, True])
def test_write_config_emits_artifact_and_optionally_updates_checkpoint(
    calibration_driver, tmp_path, monkeypatch, update_checkpoint
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    config_path.write_text(json.dumps({"model_type": "test"}))
    sparse_config = {"config_groups": {"group_0": {"algorithm": "skip_softmax"}}}
    monkeypatch.chdir(tmp_path)

    calibration_driver._write_config(str(checkpoint), sparse_config, update_checkpoint)

    assert json.loads((tmp_path / "sparse_attention_config.json").read_text()) == sparse_config
    checkpoint_config = json.loads(config_path.read_text())
    assert checkpoint_config["model_type"] == "test"
    if update_checkpoint:
        assert checkpoint_config["sparse_attention_config"] == sparse_config
    else:
        assert "sparse_attention_config" not in checkpoint_config


@pytest.mark.parametrize("held_out_passes", [False, True])
def test_main_freezes_candidate_before_held_out_and_gates_export(
    calibration_driver, monkeypatch, tmp_path, held_out_passes
):
    prompts_file = tmp_path / "calibration.txt"
    prompts_file.write_text("calibration prompt")
    held_out_file = tmp_path / "held_out.txt"
    held_out_file.write_text("held-out prompt")
    output_dir = tmp_path / "output"
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            assert kwargs["max_num_seqs"] == 1
            assert not kwargs["enable_prefix_caching"]
            assert not kwargs["enable_chunked_prefill"]
            assert kwargs["enforce_eager"]

        def generate(self, prompts, sampling):
            assert sampling.ignore_eos
            assert sampling.max_tokens == 33
            calls.append(prompts)

        def collective_rpc(self, name, args=()):
            if name == "sparse_calib_enable":
                return [1]
            if name == "sparse_calib_status":
                return [{"impl_types": {"FakeImpl": 1}}]
            if name == "sparse_calib_counts":
                return []
            if name == "sparse_validation_enable":
                return [1]
            assert name == "sparse_validation_collect"
            skipped = 500 if calls[-1] == ["calibration prompt"] or held_out_passes else 400
            return [
                {
                    phase: {
                        "total": 1000,
                        "skipped": skipped,
                        "launches": 1,
                        "unmeasured_launches": 0,
                        "min_seq_len": 1024,
                    }
                    for phase in ("prefill", "decode")
                }
            ]

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM, SamplingParams=SimpleNamespace)
    )
    monkeypatch.setattr(calibration_driver, "merge_phase_counts", lambda counts: counts)
    monkeypatch.setattr(
        calibration_driver,
        "fit_from_counts",
        lambda *a, **kw: {phase: {"a": 4.0, "b": 0.0} for phase in ("prefill", "decode")},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_sparse_attn.py",
            "model",
            "--prompts_file",
            str(prompts_file),
            "--validation_prompts_file",
            str(held_out_file),
            "--output_dir",
            str(output_dir),
        ],
    )
    if held_out_passes:
        calibration_driver.main()
    else:
        with pytest.raises(RuntimeError, match="no config exported"):
            calibration_driver.main()
    assert calls == [["calibration prompt"], ["calibration prompt"], ["held-out prompt"]]
    report = json.loads((output_dir / "sparse_attention_validation.json").read_text())
    assert report["passed"] is held_out_passes
    assert (output_dir / "sparse_attention_config.json").exists() is held_out_passes
