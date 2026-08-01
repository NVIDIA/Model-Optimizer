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

"""Focused launch/RPC test for the vLLM mask-reuse collection driver."""

import importlib.util
import json
import os
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    create_checkpoint_manifest,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.mask_reuse_capture import (
    canonical_json_sha256,
)

_SCRIPT_PATH = Path(__file__).parents[5] / "examples/vllm_serve/collect_mask_reuse.py"
_SPEC = importlib.util.spec_from_file_location("collect_mask_reuse_cli", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
collect_mask_reuse_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collect_mask_reuse_cli)


class _Tokenizer:
    def encode(self, prompt, *, add_special_tokens):
        assert add_special_tokens is True
        offset = ord(prompt[0])
        return [offset + index for index in range(256)]


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _LLM:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.invocation = None
        self.events = []
        self.__class__.instances.append(self)

    def get_tokenizer(self):
        return _Tokenizer()

    def collective_rpc(self, method, args=()):
        self.events.append(method)
        if method == "mask_reuse_capture_status":
            return [
                {
                    "capture_schema_version": 2,
                    "available": True,
                    "rank": 0,
                    "world_size": 1,
                    "reason": None,
                }
            ]
        if method == "mask_reuse_capture_begin":
            self.invocation = args[0]
            return [
                {
                    "capture_schema_version": 2,
                    "armed": True,
                    "rank": 0,
                    "world_size": 1,
                    "invocation_sha256": canonical_json_sha256(self.invocation),
                }
            ]
        assert method == "mask_reuse_capture_drain"
        invocation = self.invocation
        return [
            {
                "capture_schema_version": 2,
                "rank": 0,
                "world_size": 1,
                "invocation": invocation,
                "invocation_sha256": canonical_json_sha256(invocation),
                "geometry": invocation["expected_geometry"],
                "global_num_heads": 2,
                "eligible_tiles": 3,
                "anchor_stats_by_layer": {
                    "0": {"retained_tiles": [2, 3], "dropped_mass": [0.01, 0.02]}
                },
                "consumer_layers": {
                    "1": {
                        "anchor_layer": 0,
                        "consumer_head_start": 0,
                        "dropped_mass": [[0.01, 0.02], [0.03, 0.04]],
                    }
                },
                "attention_call_counts": {"prefill": 2, "decode": 0},
                "tp_head_order_evidence": {
                    "sentinel_device_type": "cuda",
                    "gather_dim": 0,
                    "local_rank": 0,
                    "local_num_heads": 2,
                    "gathered_rank_local_head": [[0, 0], [0, 1]],
                },
                "dense_shadow_evidence": {
                    "enabled": True,
                    "atol_hex": (0.0).hex(),
                    "rtol_hex": (0.0).hex(),
                    "validated_layer_indices": [0, 1],
                },
            }
        ]

    def generate(self, token_ids, sampling, *, use_tqdm):
        assert len(token_ids) == 256
        assert sampling.kwargs == {"temperature": 0.0, "max_tokens": 1, "ignore_eos": True}
        assert use_tqdm is False
        self.events.append("generate")
        return []


def test_main_bootstraps_policy_free_backend_and_writes_normalized_evidence(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join(
            json.dumps(
                {
                    "split": split,
                    "partition": ("development" if split == "calibration" else "outer_test"),
                    "inner_fold": 0 if split == "calibration" else None,
                    "prompt_id": prompt_id,
                    "source": source,
                    "source_group_sha256": sha256(source.encode()).hexdigest(),
                    "prompt": text,
                    "min_kv_tokens": 129,
                    "max_kv_tokens": 512,
                }
            )
            for split, prompt_id, source, text in (
                ("calibration", "cal-0", "ruler/niah", "alpha"),
                ("heldout", "held-0", "longbench/qasper", "beta"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    vanilla = tmp_path / "config.json"
    vanilla.write_text(
        json.dumps(
            {
                "threshold_scale_factor": {
                    "formula": "a * exp(b * target_sparsity)",
                    "prefill": {
                        "a": 1.0,
                        "b": 1.0,
                        "min_observed_sparsity": 0.5,
                        "max_observed_sparsity": 0.8,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "observations.jsonl"
    manifest = tmp_path / "manifest.json"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"toy-weights")
    checkpoint_manifest = create_checkpoint_manifest(checkpoint, model="test-model")
    fa4_source = tmp_path / "flash-attention"
    cute = fa4_source / "flash_attn/cute"
    cute.mkdir(parents=True)
    (cute / "interface.py").write_text("# pinned\n", encoding="utf-8")
    (cute / "block_sparsity.py").write_text("# pinned\n", encoding="utf-8")
    fake_vllm = SimpleNamespace(LLM=_LLM, SamplingParams=_SamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(
        collect_mask_reuse_cli.importlib.metadata,
        "entry_points",
        lambda **kwargs: [
            SimpleNamespace(name="mask_reuse_fa4", value="mask_reuse_vllm.plugin:register")
        ],
    )
    monkeypatch.setattr(
        collect_mask_reuse_cli.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            stdout="a" * 40 + "\n" if command[-2:] == ["rev-parse", "HEAD"] else ""
        ),
    )
    monkeypatch.setenv("MASK_REUSE_FA4_POLICY", "stale-policy.json")
    monkeypatch.setenv("MASK_REUSE_FA4_POLICY_SHA256", "stale")
    parsed_payloads = {}
    real_parse_prompts = collect_mask_reuse_cli.parse_prompt_specs_jsonl
    real_parse_vanilla = collect_mask_reuse_cli.parse_vanilla_prefill_fit

    def parse_prompts(payload):
        parsed_payloads["prompts"] = payload
        return real_parse_prompts(payload)

    def parse_vanilla(payload):
        parsed_payloads["vanilla"] = payload
        return real_parse_vanilla(payload)

    monkeypatch.setattr(collect_mask_reuse_cli, "parse_prompt_specs_jsonl", parse_prompts)
    monkeypatch.setattr(collect_mask_reuse_cli, "parse_vanilla_prefill_fit", parse_vanilla)
    _LLM.instances.clear()

    result = collect_mask_reuse_cli.main(
        [
            str(checkpoint),
            "--model-id",
            "test-model",
            "--plan",
            "test_stride2",
            "--fa4-source",
            str(fa4_source),
            "--prompts-jsonl",
            str(prompts),
            "--vanilla-config",
            str(vanilla),
            "--target-sparsities",
            "0.7",
            "--output",
            str(output),
            "--output-manifest",
            str(manifest),
            "--max-model-len",
            "512",
            "--validate-dense-output",
        ]
    )

    assert result == 0
    engine = _LLM.instances[0]
    assert engine.kwargs["attention_backend"] == "CUSTOM"
    assert engine.kwargs["dtype"] == "bfloat16"
    assert engine.kwargs["worker_cls"].endswith("MaskReuseCaptureWorker")
    assert engine.kwargs["max_num_batched_tokens"] == 8192
    assert "quantization" not in engine.kwargs
    assert engine.events == [
        "mask_reuse_capture_status",
        "mask_reuse_capture_begin",
        "generate",
        "mask_reuse_capture_drain",
        "mask_reuse_capture_begin",
        "generate",
        "mask_reuse_capture_drain",
    ]
    assert collect_mask_reuse_cli.os.environ["MASK_REUSE_FA4_CALIBRATION_CAPTURE"] == "1"
    assert (
        collect_mask_reuse_cli.os.environ["MASK_REUSE_FA4_CHECKPOINT_MANIFEST_SHA256"]
        == checkpoint_manifest.sha256
    )
    assert collect_mask_reuse_cli.os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    assert collect_mask_reuse_cli.os.environ["MASK_REUSE_FA4_CAPTURE_DENSE_SHADOW"] == "1"
    assert "MASK_REUSE_FA4_POLICY" not in collect_mask_reuse_cli.os.environ
    assert "MASK_REUSE_FA4_POLICY_SHA256" not in collect_mask_reuse_cli.os.environ

    captures = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(captures) == 2
    assert all(capture["compact_capture_schema_version"] == 1 for capture in captures)
    assert all("observations" not in capture for capture in captures)
    assert {capture["invocation"]["split"] for capture in captures} == {
        "calibration",
        "heldout",
    }
    assert {capture["invocation"]["target_sparsity_hex"] for capture in captures} == {(0.7).hex()}
    report = json.loads(manifest.read_text())
    assert report["capture_manifest_schema_version"] == 3
    assert report["capture_protocol"] == "modelopt_vllm_mask_reuse_target_sparsity_v3"
    assert report["checkpoint_manifest_sha256"] == checkpoint_manifest.sha256
    assert report["prompt_plan_file_sha256"] == sha256(parsed_payloads["prompts"]).hexdigest()
    assert report["vanilla_config_file_sha256"] == sha256(parsed_payloads["vanilla"]).hexdigest()
    assert report["fa4_source_commit"] == "a" * 40
    assert report["dense_shadow_validation_requested"] is True
    assert report["engine_kwargs"]["tensor_parallel_size"] == 1
    assert report["capture_count"] == len(captures)
    assert "observation_count" not in report
    assert len(report["captures"]) == 2
    assert {capture["invocation"]["source"] for capture in report["captures"]} == {
        "ruler/niah",
        "longbench/qasper",
    }


def test_publish_no_clobber_preserves_destination_created_by_racer(tmp_path, monkeypatch):
    temporary = tmp_path / "capture.tmp"
    destination = tmp_path / "capture.jsonl"
    temporary.write_text("ours", encoding="utf-8")
    real_link = os.link

    def racing_link(source, target, **kwargs):
        Path(target).write_text("racer", encoding="utf-8")
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(collect_mask_reuse_cli.os, "link", racing_link)

    with pytest.raises(FileExistsError):
        collect_mask_reuse_cli._publish_no_clobber(temporary, destination)

    assert destination.read_text(encoding="utf-8") == "racer"
    assert temporary.read_text(encoding="utf-8") == "ours"


def test_publish_no_clobber_rolls_back_capture_and_manifest_on_fsync_failure(tmp_path, monkeypatch):
    def fail_fsync(path):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(collect_mask_reuse_cli, "_fsync_directory", fail_fsync)

    for name in ("capture.jsonl", "capture.manifest.json"):
        temporary = tmp_path / f".{name}.tmp"
        destination = tmp_path / name
        temporary.write_text("complete", encoding="utf-8")

        with pytest.raises(OSError, match="injected fsync failure"):
            collect_mask_reuse_cli._publish_no_clobber(temporary, destination)

        assert not destination.exists()
        assert not temporary.exists()


def test_capture_environment_rejects_untracked_fa4_source(tmp_path, monkeypatch):
    fa4_source = tmp_path / "flash-attention"
    cute = fa4_source / "flash_attn/cute"
    cute.mkdir(parents=True)
    (cute / "interface.py").write_text("# pinned\n", encoding="utf-8")
    (cute / "block_sparsity.py").write_text("# pinned\n", encoding="utf-8")
    monkeypatch.setattr(
        collect_mask_reuse_cli.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            stdout=(
                "a" * 40 + "\n"
                if command[-2:] == ["rev-parse", "HEAD"]
                else "?? flash_attn/cute/local_override.py\n"
            )
        ),
    )

    with pytest.raises(
        collect_mask_reuse_cli.CaptureContractError,
        match="tracked or untracked modifications",
    ):
        collect_mask_reuse_cli._configure_capture_environment(
            "test_stride2",
            str(fa4_source),
            "0" * 64,
            validate_dense_output=True,
        )
