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

"""Tests for Puzzletron AIPerf context-capacity handling."""

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelopt.torch.puzzletron.benchmarks.aiperf import (
    _aiperf_subprocess_environment,
    _canonical_topology,
    _clean_subprocess_environment,
    _exact_length_extra_inputs,
    _parse_export,
    _PeakGpuMemorySampler,
    _prepare_vllm_checkpoint,
    _profile_command,
    _server_max_model_len,
    _topology_vllm_args,
    _vllm_server_command,
    run_aiperf_sweep,
)
from modelopt.torch.puzzletron.benchmarks.provenance import (
    artifact_sha256,
    benchmark_result_fingerprint,
)

# Subprocess environment and request sizing


def test_aiperf_server_environment_installs_vllm_torch_compatibility(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/existing")

    env = _clean_subprocess_environment(
        "0,1", architecture_id="architecture", topology_id="topology"
    )

    paths = env["PYTHONPATH"].split(":")
    assert env["VLLM_USE_LAYERNAME"] == "0"
    assert Path(paths[0]).name == "vllm_compat"
    assert (Path(paths[0]) / "sitecustomize.py").is_file()


def test_aiperf_server_environment_uses_the_active_vllm_package_source(monkeypatch):
    active_source = Path("/compatible/vllm_new")
    monkeypatch.setenv("PYTHONPATH", "/existing")
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf.importlib.util.find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=[str(active_source / "vllm")]),
    )

    env = _clean_subprocess_environment(
        "0,1", architecture_id="architecture", topology_id="topology"
    )

    paths = env["PYTHONPATH"].split(":")
    assert paths[1] == str(active_source)


@pytest.mark.parametrize(
    ("extension_names", "expect_stub"),
    [
        pytest.param(("_C_stable_libtorch",), True, id="stable-only"),
        pytest.param(("_C", "_C_stable_libtorch"), False, id="native-plus-stable"),
        pytest.param((), False, id="neither"),
    ],
)
def test_vllm_compat_stubs_c_only_for_stable_libtorch_layout(
    monkeypatch, tmp_path, extension_names, expect_stub
):
    """Install an empty ``vllm._C`` companion only for the stable-only layout."""

    extension_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    vllm_package = tmp_path / "vllm"
    vllm_package.mkdir()
    for extension_name in extension_names:
        (vllm_package / f"{extension_name}{extension_suffix}").touch()

    real_find_spec = importlib.util.find_spec

    def find_spec(name):
        if name == "vllm":
            return SimpleNamespace(submodule_search_locations=[str(vllm_package)])
        if name == "torch._opaque_base":
            return SimpleNamespace()
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    missing_module = object()
    previous_module = sys.modules.pop("vllm._C", missing_module)
    compatibility_path = (
        Path(__file__).parents[4]
        / "modelopt/torch/puzzletron/benchmarks/vllm_compat/sitecustomize.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        f"_test_vllm_compat_{len(extension_names)}", compatibility_path
    )
    assert module_spec is not None and module_spec.loader is not None
    compatibility = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(compatibility)

        assert ("vllm._C" in sys.modules) is expect_stub
        if expect_stub:
            assert sys.modules["vllm._C"].__package__ == "vllm"
    finally:
        sys.modules.pop("vllm._C", None)
        if previous_module is not missing_module:
            sys.modules["vllm._C"] = previous_module


def test_exact_length_defaults_to_ignore_eos_without_overriding_policy():
    assert _exact_length_extra_inputs(None, 32) == {"ignore_eos": True}
    assert _exact_length_extra_inputs({"temperature": 0.0}, 32) == {
        "temperature": 0.0,
        "ignore_eos": True,
    }
    assert _exact_length_extra_inputs({"min_tokens": 32}, 32) == {"min_tokens": 32}
    assert _exact_length_extra_inputs({"ignore_eos": False}, 32) == {"ignore_eos": False}


def test_server_context_includes_chat_template_headroom():
    assert _server_max_model_len(256, 32, {}) == 352
    assert _server_max_model_len(256, 32, {"server_context_overhead_tokens": 8}) == 296


def test_server_context_headroom_cannot_be_negative():
    with pytest.raises(ValueError, match="nonnegative"):
        _server_max_model_len(256, 32, {"server_context_overhead_tokens": -1})


# Checkpoint preparation and tokenizer policy


def test_prepare_vllm_checkpoint_refreshes_heterogeneous_metadata(monkeypatch, tmp_path):
    config = {
        "architectures": ["BaseModel"],
        "text_config": {"per_layer_config": {"0": {"intermediate_size": 8}}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    observed = []
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.utils.vllm_adapter.refresh_realized_checkpoint_config",
        lambda path, **kwargs: observed.append((path, kwargs)),
    )

    assert _prepare_vllm_checkpoint(tmp_path) is True
    assert observed == [(tmp_path, {"trust_remote_code": False})]


def test_prepare_vllm_checkpoint_preserves_explicit_remote_code_trust(monkeypatch, tmp_path):
    config = {
        "architectures": ["BaseModel"],
        "text_config": {"per_layer_config": {"0": {"intermediate_size": 8}}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    observed = []
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.utils.vllm_adapter.refresh_realized_checkpoint_config",
        lambda path, **kwargs: observed.append((path, kwargs)),
    )

    assert _prepare_vllm_checkpoint(tmp_path, trust_remote_code=True) is True
    assert observed == [(tmp_path, {"trust_remote_code": True})]


def test_prepare_vllm_checkpoint_leaves_native_teacher_unchanged(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["BaseModel"], "text_config": {}})
    )
    assert _prepare_vllm_checkpoint(tmp_path) is False


def test_aiperf_online_tokenizer_relaxation_is_explicit_and_non_mutating():
    expected_source = _offline_environment()
    source = dict(expected_source)

    default_environment = _aiperf_subprocess_environment(source)
    resolved = _aiperf_subprocess_environment(
        source,
        allow_aiperf_v011_online_tokenizer_resolution=True,
    )

    assert default_environment == source
    assert default_environment is not source
    assert "HF_HUB_OFFLINE" not in resolved
    assert "TRANSFORMERS_OFFLINE" not in resolved
    assert resolved["HF_DATASETS_OFFLINE"] == "1"
    assert resolved["HF_HOME"] == "/cache/huggingface"
    assert resolved["UNCHANGED"] == "value"
    assert source == expected_source


# vLLM topology and policy-owned arguments


def test_canonical_topology_covers_tp_pp_dp_effective_ep_and_context_parallel():
    topology = _canonical_topology(
        {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 2,
            "enable_expert_parallel": True,
            "prefill_context_parallel_size": 2,
            "decode_context_parallel_size": 2,
            "gpu_group_size": 8,
        }
    )
    assert topology == {
        "tp": 2,
        "pp": 1,
        "dp": 2,
        "prefill_cp": 2,
        "decode_cp": 2,
        "enable_expert_parallel": True,
        "effective_ep": 4,
        "gpu_count": 8,
        "distributed_executor_backend": "mp",
    }


def test_vllm_topology_args_enable_dp_and_expert_parallel_only_when_requested():
    dp_args = _topology_vllm_args(
        {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 2,
            "enable_expert_parallel": False,
            "gpu_group_size": 2,
        }
    )
    assert dp_args[dp_args.index("--data-parallel-size") + 1] == "2"
    assert dp_args[dp_args.index("--data-parallel-size-local") + 1] == "2"
    assert "--enable-expert-parallel" not in dp_args

    ep_args = _topology_vllm_args(
        {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 4,
            "enable_expert_parallel": True,
            "gpu_group_size": 8,
        }
    )
    assert "--enable-expert-parallel" in ep_args
    assert "--expert-parallel-size" not in ep_args


def test_vllm_server_command_applies_explicit_remote_code_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf._descriptor_vllm_args",
        lambda _checkpoint, **_kwargs: [],
    )

    default_command = _vllm_server_command(
        checkpoint_dir=tmp_path,
        port=8000,
        model_name="served-model",
        input_tokens=32,
        output_tokens=8,
        topology={"gpu_group_size": 1},
        trust_remote_code=False,
    )
    trusted_command = _vllm_server_command(
        checkpoint_dir=tmp_path,
        port=8000,
        model_name="served-model",
        input_tokens=32,
        output_tokens=8,
        topology={"gpu_group_size": 1},
        trust_remote_code=True,
    )

    assert "--trust-remote-code" not in default_command
    assert "--trust-remote-code" in trusted_command


def test_vllm_server_command_loads_full_multimodal_model(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["AnyModel"],
                "base_architecture": "Qwen3_5ForConditionalGeneration",
                "vision_config": {"model_type": "qwen3_5_vision"},
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "layer_types": ["linear_attention"],
                },
            }
        )
    )
    (tmp_path / "preprocessor_config.json").write_text("{}")

    command = _vllm_server_command(
        checkpoint_dir=tmp_path,
        port=8000,
        model_name="served-model",
        input_tokens=100,
        output_tokens=80,
        topology={"gpu_group_size": 1, "server_context_overhead_tokens": 12288},
        trust_remote_code=False,
        image_batch_size=12,
        image_width_mean=1280,
        image_height_mean=720,
    )

    assert "--language-model-only" not in command
    assert command[command.index("--max-model-len") + 1] == "12468"
    assert json.loads(command[command.index("--limit-mm-per-prompt") + 1]) == {
        "image": 12,
        "video": 0,
    }
    assert json.loads(command[command.index("--mm-processor-kwargs") + 1]) == {
        "max_num_frames": 1,
        "max_pixels": 1280 * 720,
        "min_pixels": 262144,
    }
    assert "--mamba-cache-mode" in command
    assert "--enable-prefix-caching" in command


def test_vllm_server_command_requires_visual_context_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf._descriptor_vllm_args",
        lambda _checkpoint, **_kwargs: [],
    )

    with pytest.raises(ValueError, match="visual-token budget"):
        _vllm_server_command(
            checkpoint_dir=tmp_path,
            port=8000,
            model_name="served-model",
            input_tokens=100,
            output_tokens=80,
            topology={"gpu_group_size": 1},
            trust_remote_code=False,
            image_batch_size=1,
            image_width_mean=1280,
            image_height_mean=720,
        )


def test_vllm_server_command_bounds_processor_pixels_for_small_images(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "vision_config": {},
            }
        )
    )
    (tmp_path / "preprocessor_config.json").write_text("{}")
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf._descriptor_vllm_args",
        lambda _checkpoint, **_kwargs: [],
    )

    command = _vllm_server_command(
        checkpoint_dir=tmp_path,
        port=8000,
        model_name="served-model",
        input_tokens=100,
        output_tokens=80,
        topology={"gpu_group_size": 1, "server_context_overhead_tokens": 256},
        trust_remote_code=False,
        image_batch_size=1,
        image_width_mean=128,
        image_height_mean=128,
    )

    processor = json.loads(command[command.index("--mm-processor-kwargs") + 1])
    assert processor["min_pixels"] == processor["max_pixels"] == 128 * 128


@pytest.mark.parametrize(
    "extra_arg",
    [
        "--trust-remote-code",
        "--trust-remote-code=true",
        "--trust_remote_code",
        "--trust_remote_code=true",
        "--trust-rem",
        "--trust_rem",
        "--config",
        "--config=policy.yaml",
        "--conf",
    ],
)
def test_vllm_server_command_rejects_remote_code_policy_override(monkeypatch, tmp_path, extra_arg):
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf._descriptor_vllm_args",
        lambda _checkpoint, **_kwargs: [],
    )

    with pytest.raises(ValueError, match="cannot set policy-owned vLLM options"):
        _vllm_server_command(
            checkpoint_dir=tmp_path,
            port=8000,
            model_name="served-model",
            input_tokens=32,
            output_tokens=8,
            topology={"gpu_group_size": 1, "extra_vllm_args": [extra_arg]},
            trust_remote_code=False,
        )


# AIPerf command and result mapping


def test_profile_command_maps_each_workload_answer_to_aiperf_cli(tmp_path):
    command = _profile_command(
        executable=Path("/opt/aiperf/bin/aiperf"),
        model_name="served-model",
        port=8000,
        endpoint_type="chat",
        concurrency=8,
        request_count=23,
        input_tokens=1024,
        output_tokens=128,
        tokenizer_dir=tmp_path / "tokenizer",
        artifact_dir=tmp_path / "artifacts",
        seed=7,
        extra_inputs=None,
        use_server_token_count=True,
        gpu_telemetry=None,
        image_batch_size=0,
        image_width_mean=0,
        image_height_mean=0,
    )

    assert command[command.index("--concurrency") + 1] == "8"
    assert command[command.index("--request-count") + 1] == "23"
    assert command[command.index("--synthetic-input-tokens-mean") + 1] == "1024"
    assert command[command.index("--synthetic-input-tokens-stddev") + 1] == "0"
    assert command[command.index("--output-tokens-mean") + 1] == "128"
    assert command[command.index("--output-tokens-stddev") + 1] == "0"
    assert command[command.index("--tokenizer") + 1] == str(tmp_path / "tokenizer")
    assert command[command.index("--image-batch-size") + 1] == "0"
    assert "--image-width-mean" not in command
    assert "--image-height-mean" not in command
    assert "--use-server-token-count" in command


def test_profile_command_maps_multimodal_workload_to_aiperf_cli(tmp_path):
    command = _profile_command(
        executable=Path("/opt/aiperf/bin/aiperf"),
        model_name="served-model",
        port=8000,
        endpoint_type="chat",
        concurrency=1,
        request_count=4,
        input_tokens=100,
        output_tokens=80,
        tokenizer_dir=tmp_path / "tokenizer",
        artifact_dir=tmp_path / "artifacts",
        seed=7,
        extra_inputs={"min_tokens": 80},
        use_server_token_count=True,
        gpu_telemetry=None,
        image_batch_size=12,
        image_width_mean=1280,
        image_height_mean=720,
    )

    assert command[command.index("--image-batch-size") + 1] == "12"
    assert command[command.index("--image-width-mean") + 1] == "1280"
    assert command[command.index("--image-width-stddev") + 1] == "0"
    assert command[command.index("--image-height-mean") + 1] == "720"
    assert command[command.index("--image-height-stddev") + 1] == "0"


@pytest.mark.parametrize(
    ("endpoint_type", "image_batch_size", "image_width_mean", "image_height_mean", "match"),
    [
        ("completions", 1, 1280, 720, "require endpoint_type='chat'"),
        ("chat", -1, 0, 0, "must be nonnegative"),
        ("chat", 1, 0, 720, "require positive"),
        ("chat", 0, 1280, 720, "require image_batch_size > 0"),
    ],
)
def test_profile_command_rejects_invalid_multimodal_workloads(
    tmp_path, endpoint_type, image_batch_size, image_width_mean, image_height_mean, match
):
    with pytest.raises(ValueError, match=match):
        _profile_command(
            executable=Path("/opt/aiperf/bin/aiperf"),
            model_name="served-model",
            port=8000,
            endpoint_type=endpoint_type,
            concurrency=1,
            request_count=1,
            input_tokens=100,
            output_tokens=80,
            tokenizer_dir=tmp_path / "tokenizer",
            artifact_dir=tmp_path / "artifacts",
            seed=7,
            extra_inputs=None,
            use_server_token_count=True,
            gpu_telemetry=None,
            image_batch_size=image_batch_size,
            image_width_mean=image_width_mean,
            image_height_mean=image_height_mean,
        )


def test_parse_export_preserves_interactivity_and_energy_metrics(tmp_path):
    export = tmp_path / "profile_export_aiperf.json"
    export.write_text(
        json.dumps(
            {
                "request_throughput": {"avg": 4.0},
                "output_token_throughput": {"avg": 512.0},
                "output_token_throughput_per_user": {"avg": 128.0, "p95": 130.0},
                "time_to_first_token": {"avg": 10.0, "p95": 12.0, "p99": 14.0},
                "inter_token_latency": {"avg": 2.0, "p95": 3.0, "p99": 4.0},
                "request_latency": {"avg": 50.0, "p95": 60.0, "p99": 70.0},
                "input_sequence_length": {"avg": 1024.0},
                "output_sequence_length": {"avg": 128.0},
                "total_gpu_power": {"avg": 900.0},
                "total_gpu_energy": {"avg": 4500.0},
                "output_tokens_per_joule": {"avg": 64.0},
                "energy_per_user": {"avg": 70.0},
                "num_images": {"avg": 12.0},
                "image_throughput": {"avg": 24.0},
                "image_latency": {"avg": 40.0, "p95": 50.0, "p99": 60.0},
                "error_request_count": {"avg": 0.0},
            }
        )
    )

    metrics, failures = _parse_export(export)

    assert failures == 0
    assert metrics["output_token_throughput_per_user_mean"] == 128.0
    assert metrics["output_token_throughput_per_user_p95"] == 130.0
    assert metrics["total_gpu_power_w"] == 900.0
    assert metrics["total_gpu_energy_j"] == 4500.0
    assert metrics["output_tokens_per_joule"] == 64.0
    assert metrics["num_images"] == 12.0
    assert metrics["image_throughput"] == 24.0
    assert metrics["image_latency_p95_ms"] == 50.0


def test_peak_gpu_memory_sampler_tracks_device_peaks_and_releases_nvml(monkeypatch):
    class PollOnce:
        def __init__(self):
            self.calls = 0

        def wait(self, _timeout):
            self.calls += 1
            return self.calls > 1

        def set(self):
            pass

    class ImmediateThread:
        def __init__(self, *, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

        def join(self):
            pass

    samples = {"index-0": iter((100, 300, 150)), "uuid-GPU-test": iter((200, 400, 180))}
    shutdowns = []
    pynvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: shutdowns.append(True),
        nvmlDeviceGetHandleByIndex=lambda index: f"index-{index}",
        nvmlDeviceGetHandleByUUID=lambda uuid: f"uuid-{uuid}",
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=next(samples[handle])),
    )
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    monkeypatch.setattr("modelopt.torch.puzzletron.benchmarks.aiperf.Thread", ImmediateThread)

    sampler = _PeakGpuMemorySampler("0,GPU-test", interval_seconds=3600)
    sampler._stop = PollOnce()

    with sampler:
        assert sampler.metrics == {
            "peak_gpu_memory_bytes": 700.0,
            "peak_gpu_memory_bytes_per_gpu_max": 400.0,
        }

    assert sampler.metrics == {
        "peak_gpu_memory_bytes": 700.0,
        "peak_gpu_memory_bytes_per_gpu_max": 400.0,
    }
    assert shutdowns == [True]


def test_peak_gpu_memory_sampler_propagates_polling_failure_during_join(monkeypatch):
    class PollOnce:
        def __init__(self):
            self.calls = 0

        def wait(self, _timeout):
            self.calls += 1
            return self.calls > 1

        def set(self):
            pass

    class FailingOnJoinThread:
        def __init__(self, *, target, **_kwargs):
            self.target = target

        def start(self):
            pass

        def join(self):
            self.target()

    memory_calls = 0
    shutdowns = []

    def sample_memory(_handle):
        nonlocal memory_calls
        memory_calls += 1
        if memory_calls == 2:
            raise RuntimeError("NVML polling failed")
        return SimpleNamespace(used=100)

    pynvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: shutdowns.append(True),
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetMemoryInfo=sample_memory,
    )
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    monkeypatch.setattr("modelopt.torch.puzzletron.benchmarks.aiperf.Thread", FailingOnJoinThread)
    sampler = _PeakGpuMemorySampler("0", interval_seconds=3600)
    sampler._stop = PollOnce()

    with pytest.raises(RuntimeError, match="peak GPU memory sampling failed"), sampler:
        pass

    assert shutdowns == [True]


def test_peak_gpu_memory_sampler_releases_nvml_when_sampling_fails(monkeypatch):
    shutdowns = []

    def fail_memory_info(_handle):
        raise RuntimeError("NVML failed")

    pynvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: shutdowns.append(True),
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetMemoryInfo=fail_memory_info,
    )
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    with (
        pytest.raises(RuntimeError, match="NVML failed"),
        _PeakGpuMemorySampler("0", interval_seconds=3600),
    ):
        pass

    assert shutdowns == [True]


def test_peak_gpu_memory_sampler_preserves_workload_error_when_final_sample_fails(monkeypatch):
    samples = iter((100, RuntimeError("final sample failed")))
    shutdowns = []

    def sample_memory(_handle):
        sample = next(samples)
        if isinstance(sample, BaseException):
            raise sample
        return SimpleNamespace(used=sample)

    pynvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: shutdowns.append(True),
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetMemoryInfo=sample_memory,
    )
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    with (
        pytest.raises(ValueError, match="workload failed"),
        _PeakGpuMemorySampler("0", interval_seconds=3600),
    ):
        raise ValueError("workload failed")

    assert shutdowns == [True]


def test_multimodal_sweep_keeps_image_workloads_and_cache_paths_distinct(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["AnyModel"],
                "base_architecture": "Qwen3_5ForConditionalGeneration",
                "vision_config": {"model_type": "qwen3_5_vision"},
            }
        )
    )
    (checkpoint / "preprocessor_config.json").write_text("{}")
    aiperf_executable = tmp_path / "aiperf"
    vllm_executable = tmp_path / "vllm"
    aiperf_executable.write_text("#!/bin/sh\n")
    vllm_executable.write_text("#!/bin/sh\n")
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf.shutil.which",
        lambda name, **_kwargs: str(vllm_executable) if name == "vllm" else None,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf._descriptor_vllm_args",
        lambda _checkpoint, **_kwargs: [],
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf._free_port",
        lambda: 8000,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf._wait_for_health",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf._stop_process_group",
        lambda *_args, **_kwargs: None,
    )
    server_starts = []

    def fake_popen(*_args, **_kwargs):
        server_starts.append(True)
        return SimpleNamespace()

    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf.subprocess.Popen",
        fake_popen,
    )

    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        artifact_dir = Path(command[command.index("--artifact-dir") + 1])
        image_batch_size = int(command[command.index("--image-batch-size") + 1])
        (artifact_dir / "profile_export_aiperf.json").write_text(
            json.dumps(
                {
                    "input_sequence_length": {"avg": 100.0},
                    "output_sequence_length": {"avg": 80.0},
                    "num_images": {"avg": float(image_batch_size)},
                    "image_throughput": {"avg": float(image_batch_size)},
                    "error_request_count": {"avg": 0.0},
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf.checkpoint_identity",
        lambda _checkpoint: {"serialized_size_bytes": 1024, "parameter_count": 128},
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf.hardware_identity",
        lambda _gpu_ids: {"gpus": [{"name": "test-gpu"}]},
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf.software_identity",
        lambda: {
            "packages": {"vllm": "test"},
            "source_manifests": {"modelopt_benchmarks": "test", "vllm": "test"},
        },
    )

    class FakePeakMemorySampler:
        def __init__(self, _gpu_ids):
            self.metrics = {
                "peak_gpu_memory_bytes": 768.0,
                "peak_gpu_memory_bytes_per_gpu_max": 768.0,
            }

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return None

    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf._PeakGpuMemorySampler",
        FakePeakMemorySampler,
    )

    def run_sweep():
        return run_aiperf_sweep(
            checkpoint,
            artifact_dir=tmp_path / "artifacts",
            concurrencies=(1,),
            input_tokens=100,
            output_tokens=80,
            gpu_ids="0",
            topology={
                "gpu_group_size": 1,
                "server_context_overhead_tokens": 16384,
                "env": {"VLLM_USE_V1": "0"},
            },
            request_counts={1: 1},
            warmup_request_count=2,
            warmup_seed=99,
            repetitions=2,
            collect_peak_gpu_memory=True,
            executable=aiperf_executable,
            endpoint_type="chat",
            extra_inputs={"min_tokens": 80},
            image_batch_sizes=(1, 6, 12),
            image_width_mean=1280,
            image_height_mean=720,
            gpu_telemetry=None,
        )

    results = run_sweep()

    assert [result.workload["image_batch_size"] for result in results] == [1, 1, 6, 6, 12, 12]
    assert [result.repetition for result in results] == [0, 1, 0, 1, 0, 1]
    assert len({result.workload_id for result in results}) == 3
    assert len({result.cache_identity for result in results}) == 6
    assert [command[command.index("--request-count") + 1] for command in commands].count("2") == 3
    warmup_commands = [
        command for command in commands if command[command.index("--request-count") + 1] == "2"
    ]
    assert all(command[command.index("--random-seed") + 1] == "99" for command in warmup_commands)
    assert all(result.metrics["peak_gpu_memory_bytes"] == 768.0 for result in results)
    assert all(result.checkpoint_identity["serialized_size_bytes"] == 1024 for result in results)
    assert all(result.hardware_identity["gpus"][0]["name"] == "test-gpu" for result in results)
    assert all(result.measurement_contract["warmup_request_count"] == 2 for result in results)
    assert all(result.measurement_contract["warmup_seed"] == 99 for result in results)
    assert all(
        result.measurement_contract["extra_inputs"] == {"min_tokens": 80} for result in results
    )
    assert all(
        result.measurement_contract["server_contract"]["env"] == {"VLLM_USE_V1": "0"}
        for result in results
    )
    assert all(
        result.result_fingerprint == benchmark_result_fingerprint(result.model_dump(mode="json"))
        for result in results
    )
    assert all(
        f"images_{result.workload['image_batch_size']}" in result.raw_artifacts["profile"]
        and f"repetition_{result.repetition:02d}" in result.raw_artifacts["profile"]
        for result in results
    )

    command_count = len(commands)
    cached_results = run_sweep()
    assert len(commands) == command_count
    assert len(server_starts) == 1
    assert [result.result_fingerprint for result in cached_results] == [
        result.result_fingerprint for result in results
    ]

    stale_export = Path(results[0].raw_artifacts["profile"])
    stale_export.write_text("{}")
    refreshed_export_results = run_sweep()
    assert len(server_starts) == 2
    assert len(commands) == command_count + 2
    assert refreshed_export_results[0].raw_artifact_sha256["profile"] == artifact_sha256(
        refreshed_export_results[0].raw_artifacts["profile"]
    )

    stale_metadata = stale_export.with_name("puzzletron_aiperf_result.json")
    stale_payload = json.loads(stale_metadata.read_text())
    stale_payload["result_fingerprint"] = "corrupt"
    stale_metadata.write_text(json.dumps(stale_payload))

    refreshed_results = run_sweep()
    assert len(server_starts) == 3
    assert len(commands) == command_count + 4
    assert refreshed_results[0].result_fingerprint != "corrupt"
    assert all(
        result.result_fingerprint == benchmark_result_fingerprint(result.model_dump(mode="json"))
        for result in refreshed_results
    )

    stale_metadata.write_text("{")
    recovered_results = run_sweep()
    assert len(server_starts) == 4
    assert len(commands) == command_count + 6
    assert all(
        result.result_fingerprint == benchmark_result_fingerprint(result.model_dump(mode="json"))
        for result in recovered_results
    )


def test_multimodal_sweep_rejects_an_empty_image_workload_axis(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    artifact_dir = tmp_path / "artifacts"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf._prepare_vllm_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("checkpoint preparation must follow validation"),
    )

    with pytest.raises(ValueError, match="must be non-empty"):
        run_aiperf_sweep(
            checkpoint,
            artifact_dir=artifact_dir,
            concurrencies=(1,),
            input_tokens=100,
            output_tokens=80,
            gpu_ids="0",
            topology={"gpu_group_size": 1},
            image_batch_sizes=(),
        )
    assert not artifact_dir.exists()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"warmup_request_count": -1}, "nonnegative"),
        ({"concurrencies": (2,), "warmup_request_count": 1}, "largest measured concurrency"),
        ({"repetitions": 0}, "positive"),
    ],
)
def test_aiperf_sweep_rejects_invalid_warmup_and_repetition_policy(
    monkeypatch, tmp_path, overrides, match
):
    checkpoint = tmp_path / "checkpoint"
    artifact_dir = tmp_path / "artifacts"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.aiperf._prepare_vllm_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("checkpoint preparation must follow validation"),
    )
    settings = {
        "artifact_dir": artifact_dir,
        "concurrencies": (1,),
        "input_tokens": 100,
        "output_tokens": 80,
        "gpu_ids": "0",
        "topology": {"gpu_group_size": 1},
        **overrides,
    }

    with pytest.raises(ValueError, match=match):
        run_aiperf_sweep(checkpoint, **settings)

    assert not artifact_dir.exists()


def _offline_environment() -> dict[str, str]:
    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": "/cache/huggingface",
        "UNCHANGED": "value",
    }
