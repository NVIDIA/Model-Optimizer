import json
from pathlib import Path
from types import SimpleNamespace

from modelopt.torch.puzzletron.benchmarks.aiperf import (
    _canonical_topology,
    _clean_subprocess_environment,
    _exact_length_extra_inputs,
    _parse_export,
    _prepare_vllm_checkpoint,
    _server_max_model_len,
    _topology_vllm_args,
)


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
        lambda name: SimpleNamespace(
            submodule_search_locations=[str(active_source / "vllm")]
        ),
    )

    env = _clean_subprocess_environment(
        "0,1", architecture_id="architecture", topology_id="topology"
    )

    paths = env["PYTHONPATH"].split(":")
    assert paths[1] == str(active_source)


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
    try:
        _server_max_model_len(256, 32, {"server_context_overhead_tokens": -1})
    except ValueError as error:
        assert "nonnegative" in str(error)
    else:
        raise AssertionError("negative server context overhead must fail")


def test_prepare_vllm_checkpoint_refreshes_heterogeneous_metadata(monkeypatch, tmp_path):
    config = {
        "architectures": ["BaseModel"],
        "text_config": {"per_layer_config": {"0": {"intermediate_size": 8}}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    observed = []
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.utils.vllm_adapter.refresh_realized_checkpoint_config",
        lambda path: observed.append(path),
    )

    assert _prepare_vllm_checkpoint(tmp_path) is True
    assert observed == [tmp_path]


def test_prepare_vllm_checkpoint_leaves_native_teacher_unchanged(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["BaseModel"], "text_config": {}})
    )
    assert _prepare_vllm_checkpoint(tmp_path) is False


def test_canonical_topology_covers_tp_pp_dp_ep_and_context_parallel():
    topology = _canonical_topology(
        {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "expert_parallel_size": 1,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 1,
        }
    )
    assert topology == {
        "tp": 2,
        "pp": 1,
        "dp": 1,
        "ep": 1,
        "prefill_cp": 1,
        "decode_cp": 1,
        "gpu_count": 2,
        "distributed_executor_backend": "mp",
    }


def test_vllm_topology_args_enable_dp_and_ep_only_when_requested():
    dp_args = _topology_vllm_args(
        {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 2,
            "expert_parallel_size": 1,
        }
    )
    assert dp_args[dp_args.index("--data-parallel-size") + 1] == "2"
    assert dp_args[dp_args.index("--data-parallel-size-local") + 1] == "2"
    assert "--enable-expert-parallel" not in dp_args

    ep_args = _topology_vllm_args(
        {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 2,
            "expert_parallel_size": 2,
        }
    )
    assert "--enable-expert-parallel" in ep_args


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
