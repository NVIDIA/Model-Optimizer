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
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, PreTrainedTokenizerFast

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
        ["--target_sparse_ratio", "0"],
        ["--target_sparse_ratio", "1"],
        ["--sparsity_tolerance", "0"],
        ["--sparsity_tolerance", "1"],
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


@pytest.mark.parametrize("source", ["file", "ruler", "legacy", "missing_api", "no_tokenizer"])
def test_main_rejects_normalized_overlap_before_engine_or_output(
    calibration_driver, tmp_path, monkeypatch, capsys, source
):
    held_out = tmp_path / "held_out.txt"
    held_out.write_text("same prompt")
    output_dir = tmp_path / "output"
    flags = ["--validation_prompts_file", str(held_out), "--output_dir", str(output_dir)]

    def unexpected_engine(**kwargs):
        pytest.fail("Prompt overlap must be rejected before engine creation")

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=unexpected_engine, SamplingParams=SimpleNamespace)
    )
    if source == "file":
        calibration = tmp_path / "calibration.txt"
        calibration.write_text("same\t prompt")
        flags += ["--prompts_file", str(calibration)]
    else:
        essays = tmp_path / "essays"
        essays.mkdir()
        (essays / "sample.txt").write_text("essay")
        flags += [
            "--calib_data_dir",
            str(tmp_path),
            "--trust_remote_code",
            "--engine_kwargs",
            json.dumps(
                {
                    "tokenizer": "custom",
                    "tokenizer_mode": "slow",
                    "tokenizer_revision": "tokenizer-rev",
                    "revision": "model-rev",
                }
            ),
        ]
        tokenizer = object()

        model_config = object()

        class EngineArgs:
            def __init__(self, **kwargs):
                assert kwargs["tokenizer"] == "custom"
                assert kwargs["tokenizer_mode"] == "slow"
                assert kwargs["trust_remote_code"]
                assert kwargs["tokenizer_revision"] == "tokenizer-rev"
                assert kwargs["revision"] == "model-rev"
                assert kwargs["worker_cls"] == "sparse_attn_worker.SkipSoftmaxCalibWorker"

            def create_model_config(self):
                return model_config

        def tokenizer_from_config(config):
            assert config is model_config
            return None if source == "no_tokenizer" else tokenizer

        class Builder:
            def __init__(self, **kwargs):
                assert kwargs["tokenizer_name_or_path"] is tokenizer

            def build_calibration_dataset(self):
                return [{"input": "same\n\tprompt", "length": 2}]

        monkeypatch.setitem(
            sys.modules, "vllm.engine.arg_utils", SimpleNamespace(EngineArgs=EngineArgs)
        )
        for module in ("vllm.tokenizers", "vllm.transformers_utils.tokenizer"):
            monkeypatch.setitem(sys.modules, module, None)
        if source != "missing_api":
            module = (
                "vllm.transformers_utils.tokenizer" if source == "legacy" else "vllm.tokenizers"
            )
            monkeypatch.setitem(
                sys.modules,
                module,
                SimpleNamespace(cached_tokenizer_from_config=tokenizer_from_config),
            )
        monkeypatch.setattr(calibration_driver, "RulerDatasetBuilder", Builder)

    monkeypatch.setattr(sys, "argv", ["calibrate_sparse_attn.py", "model", *flags])
    with pytest.raises(SystemExit) as exc:
        calibration_driver.main()
    assert exc.value.code == 2
    assert not output_dir.exists()
    error = capsys.readouterr().err
    if source in ("missing_api", "no_tokenizer"):
        assert "Cannot build RULER prompts with the vLLM tokenizer" in error
    else:
        assert "must not overlap" in error


def test_ruler_preflight_tokenizer_matches_serving_configuration(
    calibration_driver, tmp_path, monkeypatch
):
    # vLLM is optional for the other driver tests, which exercise preflight errors.
    engine_args = pytest.importorskip("vllm.engine.arg_utils").EngineArgs
    vllm_config = pytest.importorskip("vllm.config").VllmConfig
    renderer_from_config = pytest.importorskip("vllm.renderers").renderer_from_config
    model_dir, tokenizer_dir = tmp_path / "model", tmp_path / "tokenizer"
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        torch_dtype="float16",
    ).save_pretrained(model_dir)
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "test": 1, "prompt": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(
        tokenizer_dir
    )
    kwargs = {
        "model": str(model_dir),
        "tokenizer": str(tokenizer_dir),
        "tokenizer_mode": "AUTO",
        "dtype": "float16",
        "max_model_len": 128,
    }
    config = engine_args(**kwargs).create_model_config()
    assert config.tokenizer_mode == "auto"
    serving = renderer_from_config(vllm_config(model_config=config)).tokenizer

    class Builder:
        def __init__(self, tokenizer_name_or_path, **kwargs):
            assert (
                tokenizer_name_or_path.encode("test prompt")
                == serving.encode("test prompt")
                == [1, 2]
            )

        def build_calibration_dataset(self):
            return [{"input": "test prompt", "length": 2}]

    monkeypatch.setattr(calibration_driver, "RulerDatasetBuilder", Builder)
    args = calibration_driver._build_parser().parse_args(
        [str(model_dir), "--validation_prompts_file", "unused.txt"]
    )
    assert calibration_driver._load_ruler_prompts(args, kwargs) == ["test prompt"]


@pytest.mark.parametrize(
    ("outcome", "decode_tokens", "update_checkpoint"),
    [
        ("passed", 32, True),
        ("passed", 0, False),
        ("held_out_failed", 32, True),
        ("calibration_failed", 32, True),
    ],
)
def test_main_freezes_candidate_before_held_out_and_gates_export(
    calibration_driver, monkeypatch, tmp_path, outcome, decode_tokens, update_checkpoint
):
    prompts_file = tmp_path / "calibration.txt"
    prompts_file.write_text("calibration prompt")
    held_out_file = tmp_path / "held_out.txt"
    held_out_file.write_text("held-out prompt")
    output_dir = tmp_path / "output"
    calls = []
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    original_config = '{"sentinel": true}'
    config_path.write_text(original_config)
    phases = ("prefill", "decode") if decode_tokens else ("prefill",)
    resolved_metadata = {
        "config_groups": {
            "group_0": {"algorithm": "skip_softmax", "ignore": ["ignored_attn"]},
            "group_1": {"algorithm": "sparse_softmax", "sparsity_n": 1, "sparsity_m": 4},
        }
    }
    engine_kwargs = {
        "dtype": "bfloat16",
        "hf_overrides": {"sparse_attention_config": resolved_metadata},
    }

    class FakeLLM:
        def __init__(self, **kwargs):
            assert all(kwargs[key] == value for key, value in engine_kwargs.items())
            assert kwargs["max_num_seqs"] == 1
            assert not kwargs["enable_prefix_caching"]
            assert not kwargs["enable_chunked_prefill"]
            assert kwargs["enforce_eager"]

        def generate(self, prompts, sampling):
            assert sampling.ignore_eos
            assert sampling.max_tokens == decode_tokens + 1
            calls.append(prompts)

        def collective_rpc(self, name, args=()):
            if name == "sparse_calib_enable":
                return [1]
            if name == "sparse_calib_status":
                return [
                    {"impl_types": {"FakeImpl": 1}, "sparse_attention_config": resolved_metadata}
                ]
            if name == "sparse_calib_counts":
                return []
            if name == "sparse_validation_enable":
                groups = args[0]["config_groups"]
                assert groups["group_0"]["ignore"] == ["ignored_attn"]
                assert groups["group_1"] == resolved_metadata["config_groups"]["group_1"]
                return [1]
            assert name == "sparse_validation_collect"
            failed_phase = None
            if outcome == "calibration_failed":
                failed_phase = "prefill"
            elif outcome == "held_out_failed" and calls[-1] == ["held-out prompt"]:
                failed_phase = "decode"
            return [
                {
                    phase: {
                        "total": 1000,
                        "skipped": 400 if phase == failed_phase else 500,
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
            str(checkpoint),
            "--engine_kwargs",
            json.dumps(engine_kwargs),
            "--decode_tokens",
            str(decode_tokens),
            *(["--update_checkpoint_config"] if update_checkpoint else []),
            "--prompts_file",
            str(prompts_file),
            "--validation_prompts_file",
            str(held_out_file),
            "--output_dir",
            str(output_dir),
            "--max_refinement_steps",
            "0",
        ],
    )
    passed = outcome == "passed"
    if passed:
        calibration_driver.main()
    else:
        with pytest.raises(RuntimeError, match="no config exported"):
            calibration_driver.main()
    expected_calls = [["calibration prompt"], ["calibration prompt"]]
    if outcome != "calibration_failed":
        expected_calls.append(["held-out prompt"])
    assert calls == expected_calls
    report = json.loads((output_dir / "sparse_attention_validation.json").read_text())
    assert report["phases"] == list(phases)
    assert report["passed"] is passed
    if not (passed and update_checkpoint):
        assert config_path.read_text() == original_config
    assert (output_dir / "sparse_attention_config.json").exists() is passed
    if outcome == "calibration_failed":
        assert report["held_out_status"] == "not_run_calibration_failed"
        assert "held_out" not in report
        assert all(row["split"] == "calibration" for row in report["history"])
    if passed:
        exported = json.loads((output_dir / "sparse_attention_config.json").read_text())
        assert exported == report["candidate_config"] == report["history"][-1]["config"]
        assert exported["config_groups"]["group_0"]["ignore"] == ["ignored_attn"]
        assert exported["config_groups"]["group_1"] == resolved_metadata["config_groups"]["group_1"]
        group = exported["config_groups"]["group_0"]
        assert group["target_sparsity"] == dict.fromkeys(phases, 0.5)
        assert set(group["threshold_scale_factor"]) == {*phases, "formula"}
        if update_checkpoint:
            assert json.loads(config_path.read_text()) == {
                "sentinel": True,
                "sparse_attention_config": exported,
            }
        validation = group["validation"]
        assert validation == {
            "target_sparsity": dict.fromkeys(phases, 0.5),
            "tolerance": 0.02,
            "metric": report["metric"],
            "scope": report["scope"],
        }
