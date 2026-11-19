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
import runpy
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelopt.torch.sparsity.attention_sparsity.calibration import source_manifest
from modelopt.torch.sparsity.attention_sparsity.calibration.checkpoint_manifest import (
    create_checkpoint_manifest,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.mask_reuse_capture import (
    _eligible_tiles,
    canonical_json_sha256,
    source_capture_sha256,
)

_SCRIPT_PATH = Path(__file__).parents[5] / "examples/vllm_serve/collect_mask_reuse.py"
_SPEC = importlib.util.spec_from_file_location("collect_mask_reuse_cli", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
collect_mask_reuse_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collect_mask_reuse_cli)

_FA4_COMMIT = "a" * 40
_FA4_TREE = "b" * 40
_FA4_ARCHIVE_SHA256 = "c" * 64


def _write_source_witness(source: Path, destination: Path, *, provider="fa4") -> str:
    snapshot = source_manifest._source_tree_snapshot(source)
    raw = {
        "source_manifest_schema_version": 1,
        "source_kind": "mask-reuse-reference" if provider == "reference" else "flash-attention-4",
        "git_commit": _FA4_COMMIT,
        "git_tree": _FA4_TREE,
        "git_archive_sha256": _FA4_ARCHIVE_SHA256,
        "archive_scope": "src/attention_backend" if provider == "reference" else "flash_attn",
        "directories": list(snapshot.directories),
        "files": list(snapshot.files),
    }
    payload = (
        json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()
    destination.write_bytes(payload)
    return sha256(payload).hexdigest()


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
    observed_page = 16

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.invocation = None
        self.events = []
        self.__class__.instances.append(self)

    def get_tokenizer(self):
        return _Tokenizer()

    def collective_rpc(self, method, args=()):
        self.events.append(method)
        if method == "mask_reuse_capture_runtime_geometry":
            return [{"kv_page_tokens": self.observed_page}]
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
                    "enabled": os.environ.get(collect_mask_reuse_cli.DENSE_SHADOW_ENV) == "1",
                    "atol_hex": (0.0).hex(),
                    "rtol_hex": (0.0).hex(),
                    "validated_layer_indices": (
                        [0, 1]
                        if os.environ.get(collect_mask_reuse_cli.DENSE_SHADOW_ENV) == "1"
                        else []
                    ),
                },
            }
        ]

    def generate(self, prompt, sampling, *, use_tqdm):
        assert set(prompt) == {"prompt_token_ids"}
        assert len(prompt["prompt_token_ids"]) == 256
        assert sampling.kwargs == {"temperature": 0.0, "max_tokens": 1, "ignore_eos": True}
        assert use_tqdm is False
        self.events.append("generate")
        return []


@pytest.mark.parametrize("use_exact_tokens", [False, True])
@pytest.mark.parametrize(
    ("provider", "page"), [("fa4", 16), ("reference", 16), ("reference", 4160)]
)
@pytest.mark.parametrize("quantized_checkpoint", [False, True])
def test_main_bootstraps_policy_free_backend_and_writes_normalized_evidence(
    tmp_path, monkeypatch, use_exact_tokens, provider, page, quantized_checkpoint
):
    monkeypatch.setattr(_LLM, "observed_page", page)
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
                    **(
                        {
                            "prompt_token_ids": [ord(text[0]) + i for i in range(256)],
                            "prompt_token_ids_sha256": source_capture_sha256(
                                [ord(text[0]) + i for i in range(256)]
                            ),
                        }
                        if use_exact_tokens
                        else {}
                    ),
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
    checkpoint_config = (
        {
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
            }
        }
        if quantized_checkpoint
        else {}
    )
    checkpoint_payload = json.dumps(checkpoint_config) + "\n"
    (checkpoint / "config.json").write_text(checkpoint_payload, encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"toy-weights")
    checkpoint_manifest = create_checkpoint_manifest(
        checkpoint, model="test-model", manifest_path=tmp_path / "checkpoint-external.json"
    )
    fa4_source = tmp_path / "flash-attention"
    if provider == "fa4":
        cute = fa4_source / "flash_attn/cute"
        cute.mkdir(parents=True)
        (cute / "interface.py").write_text("# pinned\n", encoding="utf-8")
        (cute / "block_sparsity.py").write_text("# pinned\n", encoding="utf-8")
    else:
        reference = fa4_source / "src/attention_backend/src/mask_reuse_vllm/reference"
        (reference / "vendor").mkdir(parents=True)
        (reference / "capture.py").write_text("# pinned\n")
        (reference / "vendor/triton_fa.py").write_text("# pinned\n")
    fa4_source_manifest = tmp_path / "fa4-source-manifest.json"
    fa4_source_manifest_sha256 = _write_source_witness(
        fa4_source, fa4_source_manifest, provider=provider
    )
    fake_vllm = SimpleNamespace(LLM=_LLM, SamplingParams=_SamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    if use_exact_tokens:

        def forbid_retokenization(*args, **kwargs):
            raise AssertionError("exact chat-template tokens must not be re-tokenized")

        monkeypatch.setattr(_Tokenizer, "encode", forbid_retokenization)
    monkeypatch.setattr(
        collect_mask_reuse_cli.importlib.metadata,
        "entry_points",
        lambda **kwargs: [
            SimpleNamespace(name="mask_reuse_fa4", value="mask_reuse_vllm.plugin:register")
        ],
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
    source_flag = "--provider" if provider == "reference" else "--fa4"
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: _Tokenizer())

    result = collect_mask_reuse_cli.main(
        [
            str(checkpoint),
            "--checkpoint-manifest",
            str(checkpoint_manifest.manifest_path),
            "--model-id",
            "test-model",
            "--checkpoint-manifest-sha256",
            checkpoint_manifest.sha256,
            "--plan",
            "test_stride2",
            "--capture-provider",
            provider,
            "--capture-scope",
            "final-chunk",
            source_flag + "-source",
            str(fa4_source),
            source_flag + "-source-manifest",
            str(fa4_source_manifest),
            source_flag + "-source-manifest-sha256",
            fa4_source_manifest_sha256,
            source_flag + "-commit",
            _FA4_COMMIT,
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
            *(["--validate-dense-output"] if provider == "fa4" else []),
        ]
    )

    assert result == 0
    engine = _LLM.instances[0]
    assert engine.kwargs["attention_backend"] == ("CUSTOM" if provider == "fa4" else "FLASH_ATTN")
    assert engine.kwargs["dtype"] == "bfloat16"
    assert engine.kwargs["kv_cache_dtype"] == "bfloat16"
    assert engine.kwargs["worker_cls"].endswith("MaskReuseCaptureWorker")
    assert engine.kwargs["max_num_batched_tokens"] == 8192
    assert "quantization" not in engine.kwargs
    assert (checkpoint / "config.json").read_text(encoding="utf-8") == checkpoint_payload
    assert engine.events == [
        "mask_reuse_capture_status",
        "mask_reuse_capture_runtime_geometry",
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
    assert collect_mask_reuse_cli.os.environ["MASK_REUSE_FA4_CAPTURE_DENSE_SHADOW"] == (
        "1" if provider == "fa4" else "0"
    )
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
    assert report["capture_manifest_schema_version"] == (
        4 if provider == "fa4" else 8 if page == 4160 else 6
    )
    if page == 4160:
        assert report["observed_kv_page_tokens"] == 4160
    else:
        assert "observed_kv_page_tokens" not in report
    assert report["capture_protocol"] == (
        "modelopt_vllm_mask_reuse_target_sparsity_v4"
        if provider == "fa4"
        else "modelopt_vllm_mask_reuse_reference_target_sparsity_v1"
    )
    assert report["checkpoint_manifest_sha256"] == checkpoint_manifest.sha256
    assert collect_mask_reuse_cli.os.environ["MASK_REUSE_CHECKPOINT_MANIFEST"] == str(
        checkpoint_manifest.manifest_path
    )
    assert not (checkpoint / "checkpoint_manifest.json").exists()
    assert report["prompt_plan_file_sha256"] == sha256(parsed_payloads["prompts"]).hexdigest()
    assert report["vanilla_config_file_sha256"] == sha256(parsed_payloads["vanilla"]).hexdigest()
    prefix = "fa4_source" if provider == "fa4" else "provider_source"
    assert report[prefix + "_commit"] == _FA4_COMMIT
    assert report[prefix + "_git_tree"] == _FA4_TREE
    assert report[prefix + "_git_archive_sha256"] == _FA4_ARCHIVE_SHA256
    assert report[prefix + "_manifest_sha256"] == fa4_source_manifest_sha256
    assert report[prefix + "_file_count"] == 2
    assert report["dense_shadow_validation_requested"] is (provider == "fa4")
    if provider == "reference":
        assert report["provider_kind"] == "triton-reference"
        assert report["evidence_level"] == "accuracy-reference-only"
        assert not any(field.startswith("fa4_source") for field in report)
        assert os.environ["VLLM_PLUGINS"] == ""
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


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8", "bfloat16"])
def test_engine_kwargs_cannot_override_pinned_kv_cache_precision(cache_dtype):
    args = SimpleNamespace(engine_kwargs=json.dumps({"kv_cache_dtype": cache_dtype}))
    with pytest.raises(collect_mask_reuse_cli.CaptureContractError, match="unsupported keys"):
        collect_mask_reuse_cli._engine_kwargs(args)


def test_capture_environment_rejects_extra_fa4_source_path(tmp_path, monkeypatch):
    fa4_source = tmp_path / "flash-attention"
    cute = fa4_source / "flash_attn/cute"
    cute.mkdir(parents=True)
    (cute / "interface.py").write_text("# pinned\n", encoding="utf-8")
    (cute / "block_sparsity.py").write_text("# pinned\n", encoding="utf-8")
    manifest = tmp_path / "fa4-source-manifest.json"
    digest = _write_source_witness(fa4_source, manifest)
    (cute / "local_override.py").write_text("# extra\n", encoding="utf-8")

    with pytest.raises(
        collect_mask_reuse_cli.CaptureContractError,
        match="does not exactly match",
    ):
        collect_mask_reuse_cli._configure_capture_environment(
            "test_stride2",
            str(fa4_source),
            str(manifest),
            digest,
            _FA4_COMMIT,
            "0" * 64,
            validate_dense_output=True,
        )


@pytest.fixture
def resume_case(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"num_hidden_layers":2,"num_attention_heads":2}\n')
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    witness = create_checkpoint_manifest(checkpoint, model="toy")
    source = tmp_path / "source"
    cute = source / "flash_attn/cute"
    cute.mkdir(parents=True)
    (cute / "interface.py").write_text("# pinned\n")
    (cute / "block_sparsity.py").write_text("# pinned\n")
    source_witness = tmp_path / "source.json"
    source_sha = _write_source_witness(source, source_witness)
    rows = []
    for index, split in enumerate(("calibration", "heldout")):
        tokens = list(range(index, index + 256))
        rows.append(
            {
                "split": split,
                "partition": "development" if index == 0 else "outer_test",
                "inner_fold": 0 if index == 0 else None,
                "prompt_id": split,
                "source": split,
                "source_group_sha256": sha256(split.encode()).hexdigest(),
                "prompt": split,
                "min_kv_tokens": 129,
                "max_kv_tokens": 512,
                "prompt_token_ids": tokens,
                "prompt_token_ids_sha256": source_capture_sha256(tokens),
            }
        )
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    fit = tmp_path / "fit.json"
    fit.write_text(
        json.dumps(
            {
                "prefill": {
                    "a": 1.0,
                    "b": 1.0,
                    "min_observed_sparsity": 0.5,
                    "max_observed_sparsity": 0.8,
                }
            }
        )
    )
    args = collect_mask_reuse_cli._parser().parse_args(
        [
            str(checkpoint),
            "--capture-provider",
            "fa4",
            "--capture-scope",
            "final-chunk",
            "--model-id",
            "toy",
            "--checkpoint-manifest-sha256",
            witness.sha256,
            "--plan",
            "test_stride2",
            "--fa4-source",
            str(source),
            "--fa4-source-manifest",
            str(source_witness),
            "--fa4-source-manifest-sha256",
            source_sha,
            "--fa4-commit",
            _FA4_COMMIT,
            "--prompts-jsonl",
            str(prompts),
            "--vanilla-config",
            str(fit),
            "--target-sparsities",
            "0.6",
            "0.7",
            "--max-model-len",
            "512",
            "--output",
            str(tmp_path / "first/captures.jsonl"),
        ]
    )
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=_LLM, SamplingParams=_SamplingParams)
    )
    monkeypatch.setattr(
        collect_mask_reuse_cli.importlib.metadata,
        "entry_points",
        lambda **kwargs: [
            SimpleNamespace(name="mask_reuse_fa4", value="mask_reuse_vllm.plugin:register")
        ],
    )
    _LLM.instances.clear()
    return args


def test_publication_failure_can_resume_completed_capture(resume_case, tmp_path, monkeypatch):
    args = resume_case
    args.max_wall_seconds = 3600
    publish = collect_mask_reuse_cli._publish_no_clobber

    def fail_publication(*args):
        raise OSError("output filesystem does not support hard links")

    monkeypatch.setattr(collect_mask_reuse_cli, "_publish_no_clobber", fail_publication)
    with pytest.raises(OSError):
        collect_mask_reuse_cli.run(args)
    prior = Path(args.output).parent
    assert json.loads((prior / "capture.failure.json").read_text())["resumable_prefix"]
    monkeypatch.setattr(collect_mask_reuse_cli, "_publish_no_clobber", publish)
    args.resume_from = prior
    args.output = str(tmp_path / "new-destination/captures.jsonl")
    capture, manifest = collect_mask_reuse_cli.run(args)
    assert (
        len(capture.read_bytes().splitlines())
        == json.loads(manifest.read_text())["capture_count"]
        == 4
    )
    assert _LLM.instances[-1].events.count("generate") == 0


@pytest.mark.parametrize("torn_tail", [False, True])
def test_partial_resume_matches_uninterrupted_capture_and_manifest_bytes(
    resume_case, tmp_path, torn_tail
):
    args = resume_case
    args.max_wall_seconds = 1e-12
    assert collect_mask_reuse_cli.run(args) is None
    prior = Path(args.output).parent
    receipt = json.loads((prior / "capture.partial.json").read_bytes())
    assert receipt["completed_capture_count"] == 1
    assert receipt["eligible_as_capture_manifest"] is False
    assert not Path(args.output).exists()
    assert not Path(args.output + ".manifest.json").exists()
    if torn_tail:
        (prior / "capture.partial.json").unlink()
        with (prior / "capture.progress.jsonl").open("ab") as handle:
            handle.write(b'{"uncommitted":')
    before = (prior / "capture.progress.jsonl").read_bytes()
    args.resume_from = prior
    args.max_wall_seconds = None
    args.output = str(tmp_path / "resumed/captures.jsonl")
    resumed, resumed_manifest = collect_mask_reuse_cli.run(args)
    assert _LLM.instances[-1].events.count("generate") == 3
    assert (prior / "capture.progress.jsonl").read_bytes() == before
    args.resume_from = None
    args.output = str(tmp_path / "uninterrupted/captures.jsonl")
    complete, complete_manifest = collect_mask_reuse_cli.run(args)
    assert _LLM.instances[-1].events.count("generate") == 4
    assert resumed.read_bytes() == complete.read_bytes()
    assert resumed_manifest.read_bytes() == complete_manifest.read_bytes()


@pytest.mark.parametrize(
    "corruption",
    [
        "partial_line",
        "missing",
        "reorder",
        "tamper",
        "duplicate_key",
        "fit",
        "prompt",
        "engine",
        "controller",
    ],
)
def test_resume_rejects_changed_identity_or_corrupted_prefix_before_engine(
    resume_case, tmp_path, monkeypatch, corruption
):
    args = resume_case
    args.max_wall_seconds = 3600
    collect_mask_reuse_cli.run(args)
    prior = Path(args.output).parent
    journal = prior / "capture.progress.jsonl"
    lines = journal.read_bytes().splitlines(keepends=True)
    if corruption == "partial_line":
        journal.write_bytes(b"".join(lines) + b'{"incomplete":')
    elif corruption == "missing":
        journal.write_bytes(b"".join(lines[:-1]))
    elif corruption == "reorder":
        journal.write_bytes(b"".join([lines[1], lines[0], *lines[2:]]))
    elif corruption == "tamper":
        journal.write_bytes(
            b"".join(lines).replace(b'"eligible_tiles":3', b'"eligible_tiles":4', 1)
        )
    elif corruption == "duplicate_key":
        journal.write_bytes(
            b"".join(lines).replace(
                b'"eligible_tiles":3', b'"eligible_tiles":3,"eligible_tiles":3', 1
            )
        )
    elif corruption == "fit":
        fit = json.loads(Path(args.vanilla_config).read_bytes())
        fit["prefill"]["a"] = 2.0
        Path(args.vanilla_config).write_text(json.dumps(fit))
    elif corruption == "prompt":
        Path(args.prompts_jsonl).write_bytes(Path(args.prompts_jsonl).read_bytes() + b"\n")
    elif corruption == "engine":
        args.max_model_len = 1024
    else:
        monkeypatch.setattr(
            collect_mask_reuse_cli, "_controller_identity", lambda: {"changed": True}
        )
    args.resume_from = prior
    args.output = str(tmp_path / "rejected/captures.jsonl")
    with pytest.raises((ValueError, KeyError)):
        collect_mask_reuse_cli.run(args)
    assert len(_LLM.instances) == 1
    assert not Path(args.output).exists()
    assert not (Path(args.output).parent / "capture.request.json").exists()


def test_known_identity_failure_is_sticky_after_checkpoint_restore(
    resume_case, tmp_path, monkeypatch
):
    args = resume_case
    config = Path(args.model) / "config.json"
    original = config.read_bytes()

    class MutatingEngine(_LLM):
        def generate(self, *a, **kw):
            result = super().generate(*a, **kw)
            config.write_bytes(b'{"changed": true}\n')
            return result

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=MutatingEngine, SamplingParams=_SamplingParams)
    )
    args.max_wall_seconds = 3600
    with pytest.raises(ValueError):
        collect_mask_reuse_cli.run(args)
    prior = Path(args.output).parent
    failure = json.loads((prior / "capture.failure.json").read_bytes())
    assert failure["completed_capture_count"] == 4
    assert failure["identity_verification"] == "failed"
    assert failure["resumable_prefix"] is False
    config.write_bytes(original)
    args.resume_from = prior
    args.output = str(tmp_path / "restored/captures.jsonl")
    with pytest.raises(ValueError, match="invalidated"):
        collect_mask_reuse_cli.run(args)
    assert len(_LLM.instances) == 1
    assert not Path(args.output).exists()


def test_runtime_interruption_preserves_only_completed_requests(resume_case, tmp_path, monkeypatch):
    args = resume_case

    class InterruptedEngine(_LLM):
        def generate(self, *a, **kw):
            result = super().generate(*a, **kw)
            if self.events.count("generate") == 2:
                raise RuntimeError("simulated interrupted request")
            return result

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=InterruptedEngine, SamplingParams=_SamplingParams)
    )
    args.max_wall_seconds = 3600
    with pytest.raises(RuntimeError, match="interrupted"):
        collect_mask_reuse_cli.run(args)
    prior = Path(args.output).parent
    failure = json.loads((prior / "capture.failure.json").read_bytes())
    assert failure["completed_capture_count"] == 1 and failure["resumable_prefix"] is True
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=_LLM, SamplingParams=_SamplingParams)
    )
    args.resume_from = prior
    args.output = str(tmp_path / "resumed/captures.jsonl")
    result, manifest = collect_mask_reuse_cli.run(args)
    assert len(result.read_bytes().splitlines()) == 4
    assert json.loads(manifest.read_bytes())["capture_count"] == 4
    assert _LLM.instances[-1].events.count("generate") == 3


class _StreamLLM(_LLM):
    discovery = False
    fail_after_chunk = None
    tamper_completion = False
    wrong_output_tokens = False
    malformed_completion = False

    def collective_rpc(self, method, args=()):
        if method == "mask_reuse_capture_stream_begin":
            self.request, spool, self.completed = args
            self.spool = Path(spool)
            self.events.append((method, self.completed))
            return [
                {
                    "stream_schema_version": 1,
                    "rank": 0,
                    "world_size": 1,
                    "armed": True,
                    "request_sha256": canonical_json_sha256(self.request),
                }
            ]
        if method == "mask_reuse_capture_stream_drain":
            if self.malformed_completion:
                return [{"complete": True}]
            return [
                {
                    "stream_schema_version": 1,
                    "rank": 0,
                    "world_size": 1,
                    "request_sha256": canonical_json_sha256(self.request),
                    "complete": True,
                    "captured_chunk_indices": [
                        chunk["chunk_index"]
                        for chunk in self.request["chunks"][self.completed :]
                        if chunk["invocation"] is not None
                    ],
                    "spool_sha256": "0" * 64
                    if self.tamper_completion
                    else sha256((self.spool / "rank-0.jsonl").read_bytes()).hexdigest(),
                    "completed_chunk_count": len(self.request["chunks"]),
                    "observed_chunk_count": len(self.request["chunks"]),
                    "native_dense_chunk_indices": [
                        chunk["chunk_index"]
                        for chunk in self.request["chunks"]
                        if chunk["invocation"] is None
                    ],
                }
            ]
        return super().collective_rpc(method, args)

    def generate(self, prompt, sampling, *, use_tqdm):
        assert len(prompt["prompt_token_ids"]) == self.request["parent_invocation"]["sample_length"]
        self.events.append("generate")
        with (self.spool / "rank-0.jsonl").open("xb") as handle:
            for chunk in self.request["chunks"][self.completed :]:
                self.invocation = chunk["invocation"]
                payload = None
                if self.invocation is not None:
                    payload = super().collective_rpc("mask_reuse_capture_drain")[0]
                    eligible = _eligible_tiles(chunk["geometry"])
                    payload["eligible_tiles"] = eligible
                    payload["anchor_stats_by_layer"]["0"]["retained_tiles"] = [
                        eligible - 1,
                        eligible,
                    ]
                    if self.discovery:
                        consumer = payload.pop("consumer_layers")["1"]
                        consumer.pop("anchor_layer")
                        payload.update(
                            capture_mode="topology_discovery",
                            attention_layers=[0, 1],
                            max_reuse_span=1,
                            consumer_candidates_by_layer={"1": {"0": consumer}},
                        )
                        payload["anchor_stats_by_layer"]["1"] = dict(
                            payload["anchor_stats_by_layer"]["0"]
                        )

                handle.write(
                    collect_mask_reuse_cli._canonical_capture_line(
                        {
                            "stream_schema_version": 1,
                            "request_sha256": canonical_json_sha256(self.request),
                            "chunk_index": chunk["chunk_index"],
                            "payload": payload,
                        }
                    )
                )
                handle.flush()
                os.fsync(handle.fileno())
                if chunk["chunk_index"] == self.fail_after_chunk:
                    raise RuntimeError("interrupted chunk stream")
        return [
            SimpleNamespace(
                prompt_token_ids=[] if self.wrong_output_tokens else prompt["prompt_token_ids"]
            )
        ]


@pytest.fixture
def whole_resume_case(resume_case, tmp_path, monkeypatch):
    args = resume_case
    source = tmp_path / "reference-source"
    reference = source / "src/attention_backend/src/mask_reuse_vllm/reference"
    (reference / "vendor").mkdir(parents=True)
    (reference / "capture.py").write_text("# pinned\n")
    (reference / "vendor/triton_fa.py").write_text("# pinned\n")
    witness = tmp_path / "reference-source.json"
    args.capture_provider = "reference"
    args.capture_scope = "whole-prefill"
    args.provider_source = str(source)
    args.provider_source_manifest = str(witness)
    args.provider_source_manifest_sha256 = _write_source_witness(
        source, witness, provider="reference"
    )
    args.provider_commit = _FA4_COMMIT
    for field in ("source", "source_manifest", "source_manifest_sha256", "commit"):
        setattr(args, f"fa4_{field}", None)
    rows = [json.loads(line) for line in Path(args.prompts_jsonl).read_bytes().splitlines()]
    for index, row in enumerate(rows):
        tokens = list(range(index, index + 16400))
        row.update(
            prompt_token_ids=tokens,
            prompt_token_ids_sha256=source_capture_sha256(tokens),
            max_kv_tokens=65536,
        )
    Path(args.prompts_jsonl).write_text("".join(json.dumps(row) + "\n" for row in rows))
    args.max_model_len = 20000
    args.max_wall_seconds = 3600
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=_StreamLLM, SamplingParams=_SamplingParams)
    )
    monkeypatch.setattr(_StreamLLM, "fail_after_chunk", None)
    monkeypatch.setattr(_StreamLLM, "tamper_completion", False)
    monkeypatch.setattr(_StreamLLM, "wrong_output_tokens", False)
    monkeypatch.setattr(_StreamLLM, "malformed_completion", False)
    return args


@pytest.mark.parametrize("discovery", [False, True])
def test_whole_prefill_collector_output_runs_through_refit(
    whole_resume_case, tmp_path, monkeypatch, discovery
):
    args = whole_resume_case
    monkeypatch.setattr(_StreamLLM, "discovery", discovery)
    if discovery:
        args.plan = "test_topology_discovery"
        args.max_reuse_span = 1
    fit = json.loads(Path(args.vanilla_config).read_text())
    fit["prefill"].update(a=82.07903190934026, b=8.239507888257323)
    Path(args.vanilla_config).write_text(json.dumps(fit))
    args.target_sparsities = [0.5, 0.6, 0.7]
    args.max_model_len = 40961
    prompts = Path(args.prompts_jsonl)
    rows = [json.loads(line) for line in prompts.read_text().splitlines()]
    for i, row in enumerate(rows):
        tokens = list(range(i, i + 40960))
        row.update(prompt_token_ids=tokens, prompt_token_ids_sha256=source_capture_sha256(tokens))
    # Different-length parents deliberately share their early chunks. Preserve
    # every planned observation without admitting duplicate full prompts.
    for row in list(rows):
        tokens = row["prompt_token_ids"][:32768]
        rows.append(
            {
                **row,
                "prompt_id": row["prompt_id"] + "-short",
                "prompt_token_ids": tokens,
                "prompt_token_ids_sha256": source_capture_sha256(tokens),
            }
        )
    prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    capture, manifest = collect_mask_reuse_cli.run(args)
    evidence = json.loads(manifest.read_text())
    assert evidence["capture_count"] == 54
    assert evidence["whole_prefill"]["analytical_dense_capture_count"] == 16
    # Old whole-prefill manifests could silently omit all-dense prefixes.
    from modelopt.torch.sparsity.attention_sparsity.plugins.mask_reuse_stream import (
        validate_whole_prefill_evidence,
    )

    omitted = json.loads(manifest.read_text())
    omitted["whole_prefill"]["analytical_dense_capture_count"] = 0
    omitted["capture_count"] -= 16
    with pytest.raises(collect_mask_reuse_cli.CaptureContractError):
        validate_whole_prefill_evidence(omitted)
    topology = tmp_path / "topology.json"
    if discovery:
        discover = runpy.run_path(str(_SCRIPT_PATH.with_name("calibrate_mask_reuse_topology.py")))
        discovery_args = [
            "--checkpoint",
            args.model,
            "--captures",
            str(capture),
            "--capture-manifest",
            str(manifest),
            "--vanilla-config",
            args.vanilla_config,
            "--prompt-plan",
            args.prompts_jsonl,
            "--target-bmm1-skip-ratio",
            "1e-8",
            "--reuse-dropped-mass-report-threshold",
            "0.02",
            "--output",
            str(topology),
        ]
        assert discover["main"]([*discovery_args, "--benchmark-candidate"]) == 0
        assert (
            json.loads(topology.read_bytes())["provenance"]["exact_joint_optimum_proved"] is False
        )
        for name, target, budget, expected in (
            ("unmet", "1", "60", 2),
            ("unresolved", "1e-8", "1e-9", 3),
        ):
            attempt = list(discovery_args)
            attempt[attempt.index("--target-bmm1-skip-ratio") + 1] = target
            output = tmp_path / (name + ".json")
            attempt[attempt.index("--output") + 1] = str(output)
            attempt += ["--solver-time-limit-seconds", budget]
            if expected == 2:
                with pytest.raises(SystemExit) as exc:
                    discover["main"](attempt)
                assert exc.value.code == 2
            else:
                assert discover["main"](attempt) == expected
            assert not output.exists()
            diagnostic = json.loads(Path(str(output) + ".report.json").read_bytes())
            if expected == 3:
                assert diagnostic["selection_status"] == "unresolved"
    else:
        topology.write_text('{"anchors":[0],"nearest":{"0":0,"1":0}}')
    policy, report = tmp_path / "policy.json", tmp_path / "report.json"
    cli = runpy.run_path(str(_SCRIPT_PATH.with_name("calibrate_mask_reuse.py")))
    refit_args = [
        "--checkpoint",
        args.model,
        "--compact-captures",
        str(capture),
        "--capture-manifest",
        str(manifest),
        "--vanilla-config",
        args.vanilla_config,
        "--topology",
        str(topology),
        "--target-bmm1-skip-ratio",
        "1e-8",
        "--reuse-dropped-mass-report-threshold",
        "0.02",
        "--output-policy",
        str(policy),
        "--output-report",
        str(report),
    ]
    assert cli["main"]([*refit_args, "--benchmark-candidate"]) == 0
    result = json.loads(policy.read_bytes())
    assert result["provenance"]["input_capture_count"] == 54
    for row in result["calibration_report"]["by_bucket"]:
        assert row["benchmark_proof"]["status"] == "proved_optimal"
        assert row["bmm1_skip_objective"]["skipped_tiles"] > 0
    assert (
        json.loads(report.read_bytes())["candidate_policy_sha256"]
        == sha256(policy.read_bytes()).hexdigest()
    )
    next_policy = tmp_path / "retried.json"
    refit_args[refit_args.index("--output-policy") + 1] = str(next_policy)
    refit_args[refit_args.index("--output-report") + 1] = str(tmp_path / "retried-report.json")
    assert (
        cli["main"]([*refit_args, "--benchmark-candidate", "--previous-candidate", str(policy)])
        == 0
    )
    assert json.loads(next_policy.read_bytes())["context_policies"] == result["context_policies"]


@pytest.mark.parametrize("hard_kill", [False, True])
def test_whole_stream_resume_recovers_durable_chunks_and_matches_complete(
    whole_resume_case, tmp_path, monkeypatch, hard_kill
):
    args = whole_resume_case
    monkeypatch.setattr(_StreamLLM, "fail_after_chunk", 0)
    with pytest.raises(RuntimeError, match="interrupted chunk"):
        collect_mask_reuse_cli.run(args)
    prior = Path(args.output).parent
    assert len((prior / "capture.progress.jsonl").read_bytes().splitlines()) == 1
    if hard_kill:
        # SIGKILL can leave fsynced rank chunks but no controller journal/receipt.
        (prior / "capture.progress.jsonl").write_bytes(b"")
        (prior / "capture.failure.json").unlink()
        spool = next((prior / "capture-spools").glob("*/rank-0.jsonl"))
        with spool.open("ab") as handle:
            handle.write(b'{"killed_mid_write":')
    before = {path: path.read_bytes() for path in prior.rglob("*") if path.is_file()}
    monkeypatch.setattr(_StreamLLM, "fail_after_chunk", None)
    args.resume_from = prior
    args.output = str(tmp_path / "resumed/captures.jsonl")
    resumed, resumed_manifest = collect_mask_reuse_cli.run(args)
    assert ("mask_reuse_capture_stream_begin", 1) in _LLM.instances[-1].events
    assert all(path.read_bytes() == payload for path, payload in before.items())
    if hard_kill:
        evidence = json.loads((resumed.parent / "capture.complete.json").read_bytes())
        assert evidence["uncommitted_spool_suffix_preserved"] is True
        assert (
            evidence["recovered_rank_spools"][0]["sha256"] == sha256(spool.read_bytes()).hexdigest()
        )
    assert len(resumed.read_bytes().splitlines()) == 8
    manifest = json.loads(resumed_manifest.read_bytes())
    assert manifest["whole_prefill"]["dense_exclusion_count"] == 4
    assert manifest["whole_prefill"]["measured_capture_count"] == 8
    args.resume_from = None
    args.output = str(tmp_path / "complete/captures.jsonl")
    complete, complete_manifest = collect_mask_reuse_cli.run(args)
    assert resumed.read_bytes() == complete.read_bytes()
    assert resumed_manifest.read_bytes() == complete_manifest.read_bytes()


@pytest.mark.parametrize(
    "corruption", ["tamper_completion", "malformed_completion", "wrong_output_tokens"]
)
def test_whole_stream_invalid_completion_cannot_be_recovered_from_spools(
    whole_resume_case, tmp_path, monkeypatch, corruption
):
    args = whole_resume_case
    monkeypatch.setattr(_StreamLLM, corruption, True)
    with pytest.raises(ValueError, match=r"completion|exact input token IDs"):
        collect_mask_reuse_cli.run(args)
    prior = Path(args.output).parent
    assert json.loads((prior / "capture.failure.json").read_bytes())["resumable_prefix"] is False
    args.resume_from = prior
    args.output = str(tmp_path / "rejected/captures.jsonl")
    with pytest.raises(ValueError, match="invalidated"):
        collect_mask_reuse_cli.run(args)
