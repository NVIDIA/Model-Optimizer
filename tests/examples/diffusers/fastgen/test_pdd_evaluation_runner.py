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

"""Tests for the Qwen-Image PDD evaluation runner."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
for path in (_REPO_ROOT, _FASTGEN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pdd.artifacts import canonical_json_bytes, load_canonical_json, write_canonical_json
from pdd.evaluate_qwen_image import (
    _prompt_pairs,
    _publish_staging,
    _resolve_output_paths,
    _run_repetition,
    _summary,
    main,
)
from pdd.export import PDD_INFERENCE_SCHEDULES
from pdd.inference_qwen_image import main as inference_main
from pdd.inference_runtime import QwenPDDInferenceRuntime, pdd_tensor_sha256


class _Scheduler:
    def step(self, value=0):
        return value


class _Student(nn.Module):
    def forward(self, value):
        time.sleep(0.0001)
        return value + 1


class _FakeRuntime:
    def __init__(self, export_root: pathlib.Path, blocks=(1, 1)) -> None:
        self.student = _Student()
        self.scheduler = _Scheduler()
        self.device = torch.device("cpu")
        self.dtype = torch.bfloat16
        self.config = SimpleNamespace(
            inference_blocks=list(blocks),
            grid_size=sum(blocks),
            grid_max_t=0.999,
            flow_shift=5.0,
        )
        self.descriptor = SimpleNamespace(root=export_root)
        self.model_identity = {
            "id": "Qwen/Qwen-Image",
            "revision": "f" * 40,
            "dtype": "bfloat16",
        }
        self.encode_calls = 0
        self.sample_calls = 0

    def encode_prompt(self, prompt, max_sequence_length):
        assert prompt and max_sequence_length > 0
        self.encode_calls += 1
        return torch.tensor(0)

    def make_raw_noise(self, *, seed, height, width):
        return torch.randn((1, 1, height, width), generator=torch.Generator().manual_seed(seed))

    def sample_decode(self, condition, raw_noise):
        del condition, raw_noise
        self.sample_calls += 1
        value = torch.tensor(0)
        for _ in self.config.inference_blocks:
            value = self.student(value)
        return [Image.new("RGB", (2, 2), color=(int(value), 0, 0))]

    def trajectory_identity(self, raw_noise):
        full = torch.linspace(0.999, 0.0, self.config.grid_size + 1, dtype=torch.float32)
        boundaries = [0]
        for block in self.config.inference_blocks:
            boundaries.append(boundaries[-1] + block)
        boundary = full[boundaries]
        initial = (raw_noise.to(torch.float64) * self.config.grid_max_t).to(torch.float32)
        return {
            "raw_noise_sha256": pdd_tensor_sha256(raw_noise, "raw_noise"),
            "initial_state_sha256": pdd_tensor_sha256(initial, "initial_state"),
            "full_time_nodes": full.tolist(),
            "full_time_nodes_sha256": pdd_tensor_sha256(full, "full_time_nodes"),
            "boundary_indices": boundaries,
            "boundary_time_nodes": boundary.tolist(),
            "boundary_time_nodes_sha256": pdd_tensor_sha256(boundary, "boundary_time_nodes"),
            "first_sigma": float(full[0]),
        }


def test_prompt_manifest_expands_exact_order_and_rejects_unsafe_data() -> None:
    value = {
        "schema_version": 1,
        "prompts": [
            {"prompt_id": "a", "prompt": "alpha", "seeds": [1, 2]},
            {"prompt_id": "b-2", "prompt": "beta", "seeds": [3]},
        ],
    }
    assert [(pair.prompt_id, pair.seed) for pair in _prompt_pairs(value)] == [
        ("a", 1),
        ("a", 2),
        ("b-2", 3),
    ]
    value["prompts"][1]["prompt_id"] = "../escape"
    with pytest.raises(ValueError, match="safe path component"):
        _prompt_pairs(value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"schema_version": 1, "prompts": [], "extra": 1}, "exactly"),
        ({"schema_version": 2, "prompts": []}, "schema_version"),
        ({"schema_version": True, "prompts": []}, "schema_version"),
        (
            {
                "schema_version": 1,
                "prompts": [
                    {"prompt_id": "b", "prompt": "one", "seeds": [1]},
                    {"prompt_id": "a", "prompt": "two", "seeds": [2]},
                ],
            },
            "lexicographically",
        ),
        (
            {
                "schema_version": 1,
                "prompts": [{"prompt_id": "a", "prompt": "one", "seeds": [2, 1]}],
            },
            "sorted and unique",
        ),
        (
            {
                "schema_version": 1,
                "prompts": [{"prompt_id": "a", "prompt": "one", "seeds": [True]}],
            },
            "invalid seed",
        ),
    ],
)
def test_prompt_manifest_expected_red_matrix(value, message) -> None:
    with pytest.raises(ValueError, match=message):
        _prompt_pairs(value)


def test_tensor_hash_uses_exact_header_little_endian_payload_and_domain() -> None:
    tensor = torch.tensor([[1.0, -2.5]], dtype=torch.float32)
    header = {
        "schema_version": 1,
        "domain": "raw_noise",
        "dtype": "float32",
        "shape": [1, 2],
        "byte_order": "little",
        "order": "C",
    }
    payload = np.ascontiguousarray(tensor.numpy(), dtype="<f4").tobytes(order="C")
    expected = hashlib.sha256(canonical_json_bytes(header) + b"\0" + payload).hexdigest()
    assert pdd_tensor_sha256(tensor, "raw_noise") == expected
    assert pdd_tensor_sha256(tensor, "initial_state") != expected
    with pytest.raises(TypeError, match="float32"):
        pdd_tensor_sha256(tensor.to(torch.float64), "raw_noise")
    with pytest.raises(ValueError, match="unknown"):
        pdd_tensor_sha256(tensor, "other")
    with pytest.raises(FloatingPointError, match="non-finite"):
        pdd_tensor_sha256(torch.tensor([float("nan")]), "raw_noise")


def test_runtime_trajectory_identity_uses_the_raw_noise_device(monkeypatch) -> None:
    requested_devices = []

    class Sampler:
        def time_grid(self, device):
            requested_devices.append(device)
            return torch.tensor([0.999, 0.5, 0.0], dtype=torch.float32)

    class RawNoise:
        device = torch.device("cuda:7")

        def to(self, dtype):
            return torch.ones((1, 1), dtype=dtype)

    runtime = QwenPDDInferenceRuntime(
        student=None,
        scheduler=None,
        descriptor=None,
        model_identity={},
        dtype=torch.bfloat16,
        device=torch.device("cuda:7"),
        config=SimpleNamespace(inference_blocks=(1, 1), grid_max_t=0.999),
        pipe=None,
        sampler=Sampler(),
    )
    monkeypatch.setattr("pdd.inference_runtime.pdd_tensor_sha256", lambda _tensor, domain: domain)
    identity = runtime.trajectory_identity(RawNoise())
    assert requested_devices == [torch.device("cuda:7")]
    assert identity["full_time_nodes"] == pytest.approx([0.999, 0.5, 0.0])


def test_source_owned_schedules_and_summary_contract() -> None:
    assert PDD_INFERENCE_SCHEDULES == {
        "pdd-2": (64, 64),
        "pdd-4": (32, 32, 32, 32),
        "pdd-8": (16, 16, 16, 16, 16, 16, 16, 16),
    }
    assert _summary([4.0, 1.0, 3.0, 2.0]) == {"median": 2.5, "p95": 4.0}


def test_repetition_counts_calls_times_cpu_and_restores_scheduler(tmp_path) -> None:
    runtime = _FakeRuntime(tmp_path)
    original = runtime.scheduler.step
    observation = _run_repetition(runtime, "prompt", torch.zeros(1), 8)
    assert observation.scheduler_calls == 0
    assert observation.transformer_calls == 2
    assert observation.peak_device_memory_bytes is None
    assert observation.transformer_seconds > 0
    assert observation.end_to_end_seconds >= observation.transformer_seconds
    assert runtime.scheduler.step == original
    assert not runtime.student._forward_pre_hooks
    assert not runtime.student._forward_hooks


@pytest.mark.parametrize("blocks", [(64, 64), (32, 32, 32, 32), (16,) * 8])
def test_repetition_counts_each_supported_schedule(tmp_path, blocks) -> None:
    runtime = _FakeRuntime(tmp_path, blocks=blocks)
    observation = _run_repetition(runtime, "prompt", torch.zeros(1), 8)
    assert observation.transformer_calls == len(blocks)


def test_repetition_rejects_scheduler_and_transformer_count_collapse(tmp_path, monkeypatch) -> None:
    runtime = _FakeRuntime(tmp_path)

    def scheduler_call(_condition, _noise):
        runtime.scheduler.step()
        value = runtime.student(torch.tensor(0))
        value = runtime.student(value)
        return [Image.new("RGB", (2, 2))]

    monkeypatch.setattr(runtime, "sample_decode", scheduler_call)
    with pytest.raises(RuntimeError, match=r"scheduler\.step"):
        _run_repetition(runtime, "prompt", torch.zeros(1), 8)
    assert "step" not in vars(runtime.scheduler)

    def missing_transformer(_condition, _noise):
        return [Image.new("RGB", (2, 2))]

    monkeypatch.setattr(runtime, "sample_decode", missing_transformer)
    with pytest.raises(RuntimeError, match="transformer calls"):
        _run_repetition(runtime, "prompt", torch.zeros(1), 8)


def test_mocked_cuda_instrumentation_orders_sync_reset_events_and_memory(
    tmp_path, monkeypatch
) -> None:
    runtime = _FakeRuntime(tmp_path)
    runtime.device = torch.device("cuda")
    actions = []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing is True

        def record(self):
            actions.append("event")

        def elapsed_time(self, _other):
            actions.append("elapsed")
            return 0.01

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: actions.append("sync"))
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda _device: actions.append("reset")
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 123)
    observation = _run_repetition(runtime, "prompt", torch.zeros(1), 8)
    assert actions[:2] == ["sync", "reset"]
    assert actions[-3:] == ["sync", "elapsed", "elapsed"]
    assert observation.peak_device_memory_bytes == 123


def test_repetition_restores_instrumentation_on_failure(tmp_path, monkeypatch) -> None:
    runtime = _FakeRuntime(tmp_path)
    original = runtime.scheduler.step

    def fail(_condition, _noise):
        runtime.scheduler.step()
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime, "sample_decode", fail)
    with pytest.raises(RuntimeError, match="boom"):
        _run_repetition(runtime, "prompt", torch.zeros(1), 8)
    assert runtime.scheduler.step == original
    assert not runtime.student._forward_pre_hooks
    assert not runtime.student._forward_hooks


def test_repetition_restores_after_initial_sync_failure(tmp_path, monkeypatch) -> None:
    runtime = _FakeRuntime(tmp_path)
    runtime.device = torch.device("cuda")
    original = runtime.scheduler.step

    def fail_sync(_device):
        raise RuntimeError("sync failed")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_sync)
    with pytest.raises(RuntimeError, match="sync failed"):
        _run_repetition(runtime, "prompt", torch.zeros(1), 8)
    assert runtime.scheduler.step == original
    assert not runtime.student._forward_pre_hooks
    assert not runtime.student._forward_hooks


def test_repetition_restores_after_partial_hook_install_failure(tmp_path, monkeypatch) -> None:
    runtime = _FakeRuntime(tmp_path)
    original = runtime.scheduler.step

    def fail_post_hook(*_args, **_kwargs):
        raise RuntimeError("post-hook failed")

    monkeypatch.setattr(runtime.student, "register_forward_hook", fail_post_hook)
    with pytest.raises(RuntimeError, match="post-hook failed"):
        _run_repetition(runtime, "prompt", torch.zeros(1), 8)
    assert runtime.scheduler.step == original
    assert not runtime.student._forward_pre_hooks
    assert not runtime.student._forward_hooks


def test_atomic_publish_rolls_back_failed_parent_fsync(tmp_path, monkeypatch) -> None:
    staging = tmp_path / ".result.staging"
    staging.mkdir()
    (staging / "complete").write_text("complete")
    output = tmp_path / "result"
    module = sys.modules["pdd.evaluate_qwen_image"]
    original = module._fsync_directory
    failed = False

    def fail_once(path):
        nonlocal failed
        if path == tmp_path and output.exists() and not failed:
            failed = True
            raise OSError("fsync failed")
        original(path)

    monkeypatch.setattr(module, "_fsync_directory", fail_once)
    with pytest.raises(OSError, match="fsync failed"):
        _publish_staging(staging, output)
    assert failed
    assert not output.exists()
    assert (staging / "complete").is_file()


def test_output_transaction_rejects_escape_collision_and_symlink(tmp_path) -> None:
    with pytest.raises(ValueError, match="strictly beneath"):
        _resolve_output_paths(tmp_path / "output", tmp_path / "outside.json")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        _resolve_output_paths(existing, existing / "result.json")
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _resolve_output_paths(link / "output", link / "output" / "result.json")
    output = real / "output"
    unresolved_then_link = tmp_path / "missing" / ".." / "link" / "output"
    with pytest.raises(ValueError, match="parent traversal"):
        _resolve_output_paths(unresolved_then_link, output / "result.json")
    unresolved_result = unresolved_then_link / "result.json"
    with pytest.raises(ValueError, match="parent traversal"):
        _resolve_output_paths(output, unresolved_result)


def test_main_publishes_complete_atomic_cpu_result(tmp_path, monkeypatch) -> None:
    export = tmp_path / "export"
    export.mkdir()
    (export / "manifest.json").write_bytes(b"manifest")
    prompts = tmp_path / "prompts.json"
    write_canonical_json(
        prompts,
        {
            "schema_version": 1,
            "prompts": [{"prompt_id": "sample", "prompt": "text", "seeds": [7]}],
        },
    )
    runtime = _FakeRuntime(export, blocks=(1, 1, 1, 1))
    monkeypatch.setattr(
        "pdd.evaluate_qwen_image.load_qwen_pdd_runtime",
        lambda _export, _schedule, _device: runtime,
    )
    output = tmp_path / "evaluation"
    result = output / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_qwen_image.py",
            "--export-dir",
            str(export),
            "--prompts",
            str(prompts),
            "--schedule",
            "pdd-4",
            "--output-dir",
            str(output),
            "--result-json",
            str(result),
            "--warmup-runs",
            "1",
            "--measured-runs",
            "2",
            "--height",
            "2",
            "--width",
            "2",
            "--device",
            "cpu",
        ],
    )
    main()
    value = load_canonical_json(result)
    assert value["record_type"] == "pdd_qwen_evaluation"
    assert value["identity"]["schedule"] == "pdd-4"
    record = value["records"][0]
    assert record["observed_scheduler_step_calls"] == [0, 0]
    assert record["actual_transformer_invocations"] == [4, 4]
    assert record["peak_device_memory_bytes"] == [None, None]
    assert record["summaries"]["peak_device_memory_bytes"] is None
    assert runtime.encode_calls == runtime.sample_calls == 3
    assert (output / record["output"]["path"]).is_file()
    assert not list(tmp_path.glob(".evaluation.*.staging"))
    assert json.loads(result.read_text())["schema_version"] == 1


def test_main_second_prompt_failure_publishes_nothing(tmp_path, monkeypatch) -> None:
    export = tmp_path / "export"
    export.mkdir()
    (export / "manifest.json").write_bytes(b"manifest")
    prompts = tmp_path / "prompts.json"
    write_canonical_json(
        prompts,
        {
            "schema_version": 1,
            "prompts": [
                {"prompt_id": "a", "prompt": "first", "seeds": [1]},
                {"prompt_id": "b", "prompt": "second", "seeds": [2]},
            ],
        },
    )
    runtime = _FakeRuntime(export, blocks=(1, 1, 1, 1))
    original_encode = runtime.encode_prompt

    def fail_second(prompt, max_sequence_length):
        if prompt == "second":
            raise RuntimeError("second prompt failed")
        return original_encode(prompt, max_sequence_length)

    monkeypatch.setattr(runtime, "encode_prompt", fail_second)
    monkeypatch.setattr(
        "pdd.evaluate_qwen_image.load_qwen_pdd_runtime",
        lambda _export, _schedule, _device: runtime,
    )
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_qwen_image.py",
            "--export-dir",
            str(export),
            "--prompts",
            str(prompts),
            "--schedule",
            "pdd-4",
            "--output-dir",
            str(output),
            "--result-json",
            str(output / "result.json"),
            "--warmup-runs",
            "1",
            "--measured-runs",
            "1",
            "--height",
            "2",
            "--width",
            "2",
            "--device",
            "cpu",
        ],
    )
    with pytest.raises(RuntimeError, match="second prompt failed"):
        main()
    assert not output.exists()
    assert not list(tmp_path.glob(".evaluation.*.staging"))


def test_legacy_inference_preserves_scope_and_adds_observed_scheduler_count(
    tmp_path, monkeypatch
) -> None:
    export = tmp_path / "export"
    export.mkdir()
    (export / "manifest.json").write_bytes(b"manifest")
    runtime = _FakeRuntime(export, blocks=(1, 1, 1, 1))
    monkeypatch.setattr(
        "pdd.inference_qwen_image.load_qwen_pdd_runtime",
        lambda _export, _schedule, _device: runtime,
    )
    output = tmp_path / "image.png"
    result = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference_qwen_image.py",
            "--export-dir",
            str(export),
            "--prompt",
            "text",
            "--prompt-id",
            "sample",
            "--schedule",
            "pdd-4",
            "--seed",
            "7",
            "--height",
            "2",
            "--width",
            "2",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--result-json",
            str(result),
        ],
    )
    inference_main()
    value = load_canonical_json(result)
    assert value["schema_version"] == 2
    assert value["scheduler_steps"] == 4
    assert value["observed_scheduler_step_calls"] == 0
    assert value["actual_transformer_invocations"] == 4
    assert value["latency_seconds"] > 0
    assert runtime.encode_calls == runtime.sample_calls == 1
    assert output.is_file()


def test_legacy_inference_restores_instrumentation_on_initial_sync_failure(
    tmp_path, monkeypatch
) -> None:
    export = tmp_path / "export"
    export.mkdir()
    (export / "manifest.json").write_bytes(b"manifest")
    runtime = _FakeRuntime(export, blocks=(1, 1, 1, 1))
    runtime.device = torch.device("cuda")
    original = runtime.scheduler.step
    monkeypatch.setattr(
        "pdd.inference_qwen_image.load_qwen_pdd_runtime",
        lambda _export, _schedule, _device: runtime,
    )

    def fail_sync(_device):
        raise RuntimeError("sync failed")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_sync)
    output = tmp_path / "image.png"
    result = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference_qwen_image.py",
            "--export-dir",
            str(export),
            "--prompt",
            "text",
            "--prompt-id",
            "sample",
            "--schedule",
            "pdd-4",
            "--seed",
            "7",
            "--height",
            "2",
            "--width",
            "2",
            "--device",
            "cuda",
            "--output",
            str(output),
            "--result-json",
            str(result),
        ],
    )
    with pytest.raises(RuntimeError, match="sync failed"):
        inference_main()
    assert runtime.scheduler.step == original
    assert not runtime.student._forward_pre_hooks
    assert not output.exists()
    assert not result.exists()
