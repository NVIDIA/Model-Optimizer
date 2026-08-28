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

"""Tests for Puzzletron sparse runtime statistics and library-building stages."""

import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import modelopt.torch.puzzletron.stages.pipeline as pipeline_stages
import modelopt.torch.puzzletron.subblock_stats.calc_runtime_stats as runtime_stats_module
from modelopt.torch.puzzletron.anymodel.capabilities import (
    CapabilityValidationError,
    default_capabilities,
)
from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MoEConfig,
)
from modelopt.torch.puzzletron.manifest import StageManifest
from modelopt.torch.puzzletron.pipeline_config import (
    load_runtime_hydra_config,
    pipeline_config_from_path,
)
from modelopt.torch.puzzletron.stage_runner import _preflight
from modelopt.torch.puzzletron.stages.pipeline import build_library_stage
from modelopt.torch.puzzletron.subblock_stats.calc_runtime_stats import (
    _runtime_shard_results_complete,
    calc_runtime_for_blocks,
    calc_runtime_for_subblocks,
)
from modelopt.torch.puzzletron.subblock_stats.measurements import apply_vllm_measurement
from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import RuntimeMeasurement


def test_runtime_hydra_config_preserves_named_vllm_measurement_overlay(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        json.dumps(
            {
                "puzzle_dir": str(tmp_path / "run"),
                "vllm_stats": {
                    "enabled": True,
                    "subblock_stats_filename": "subblock_stats.json",
                    "measurements": {
                        "isl_heavy": {
                            "prefill_seq_len": 32,
                            "generation_seq_len": 8,
                            "max_num_seqs": 2,
                            "batch_size": 2,
                            "runtime_stats": {
                                "topology": {
                                    "tensor_parallel_size": 1,
                                    "pipeline_parallel_size": 1,
                                    "data_parallel_size": 1,
                                    "prefill_context_parallel_size": 1,
                                    "decode_context_parallel_size": 1,
                                    "gpu_group_size": 1,
                                }
                            },
                        }
                    },
                },
            }
        )
    )
    selected = apply_vllm_measurement(
        pipeline_config_from_path(config_path),
        "isl_heavy",
    )

    hydra_cfg = load_runtime_hydra_config(selected)

    expected = "artifacts/vllm_stats/measurements/isl_heavy/subblock_stats.json"
    assert selected["vllm_stats"]["subblock_stats_filename"] == expected
    assert hydra_cfg.calc_subblock_stats.subblock_stats_filename == expected


def test_explicit_vllm_stage_requires_descriptor_vllm_capability():
    capabilities = default_capabilities(descriptor_name="unsupported")
    resolution = SimpleNamespace(
        capabilities=replace(capabilities, export=replace(capabilities.export, vllm=False))
    )

    with pytest.raises(CapabilityValidationError, match="does not support vLLM export"):
        _preflight({"vllm_stats": {"enabled": True}}, resolution, stage="vllm_stats")


def test_vllm_stage_prepares_sparse_subblock_selection_from_teacher(tmp_path):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    teacher_block = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
            FFNConfig(intermediate_size=16),
        )
    )
    (teacher_dir / "config.json").write_text(
        json.dumps({"block_configs": [teacher_block.to_dict()]})
    )
    runtime_cfg = OmegaConf.create(
        {
            "sparse_sampling": {
                "enabled": True,
                "max_pairwise_per_family": 0,
                "seed": 17,
            }
        }
    )
    config = {
        "search_space": {
            "axes": {
                "kv_groups": {"enabled": True, "values": [1]},
                "ffn_intermediate": {"enabled": True, "values": [8]},
            }
        },
        "build_library": {"include_noops": False},
    }

    summary = pipeline_stages._prepare_sparse_runtime_selection(
        config,
        runtime_cfg=runtime_cfg,
        teacher_dir=teacher_dir,
        puzzle_dir=tmp_path,
    )

    manifest_path = tmp_path / "artifacts" / "vllm_stats" / "sparse_subblock_samples.json"
    manifest = json.loads(manifest_path.read_text())
    assert runtime_cfg.selection_manifest == str(manifest_path)
    assert manifest["mode"] == "subblock_runtime"
    assert {tuple(row["changed_axes"]) for row in manifest["selected"]} == {
        (),
        ("ffn_intermediate",),
        ("kv_groups",),
    }
    assert summary == {
        "path": str(manifest_path),
        "identity": manifest["identity"],
        "candidate_count": 4,
        "selected_count": 4,
        "excluded_count": 2,
    }


def test_subblock_runtime_uses_homogeneous_n_and_2n_layouts(monkeypatch):
    attention = AttentionConfig(num_query_heads=8, num_kv_heads=2)
    base_block = BlockConfig(subblock_configs=(attention, FFNConfig(no_op=True)))

    class Descriptor:
        @staticmethod
        def runtime_benchmark_config_fields(_config):
            return {}

        @staticmethod
        def runtime_benchmark_base_block_config(_runtime_config):
            return base_block

        @staticmethod
        def runtime_benchmark_sublayers_are_exclusive():
            return True

    captured_layouts = []

    def run_benchmarks(specs, _gpu_ids, _cache_dir, _measurement_pairs):
        from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import RuntimeMeasurement

        results = {}
        for key, (_runtime, layout) in specs.items():
            captured_layouts.append(layout)
            results[key] = RuntimeMeasurement(
                total_ms=6.0 + 2.0 * len(layout),
                prefill_ms=2.0 + 0.5 * len(layout),
            )
        return results

    monkeypatch.setattr(runtime_stats_module, "_run_benchmarks", run_benchmarks)
    monkeypatch.setattr(runtime_stats_module, "_resolve_gpu_ids", lambda _size: ["0,1"])

    runtime, non_block = calc_runtime_for_subblocks(
        {attention},
        OmegaConf.create(
            {
                "repeat_block_n_times": 3,
                "num_iters": 3,
                "num_warmup_iters": 2,
                "topology": {
                    "pipeline_parallel_size": 2,
                    "gpu_group_size": 2,
                },
            }
        ),
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=8,
        num_key_value_heads=2,
        descriptor=Descriptor,
        lm_config=SimpleNamespace(),
        tokenizer_path="tokenizer",
        prefill_seq_len=16,
        generation_seq_len=4,
        batch_size=1,
    )

    assert sorted(len(layout) for layout in captured_layouts) == [4, 8]
    assert all(
        block == attention.to_blockconfig() for layout in captured_layouts for block in layout
    )
    assert runtime[attention] == RuntimeMeasurement(total_ms=2.0, prefill_ms=0.5)
    assert non_block == RuntimeMeasurement(total_ms=6.0, prefill_ms=2.0)


def test_block_runtime_deduplicates_base_scaffold_measurement_pair(monkeypatch):
    attention = AttentionConfig(num_query_heads=8, num_kv_heads=2)
    base_block = BlockConfig(subblock_configs=(attention, FFNConfig(no_op=True)))
    scaffolded_block = BlockConfig(subblock_configs=(FFNConfig(intermediate_size=16),))

    class Descriptor:
        @staticmethod
        def runtime_benchmark_config_fields(_config):
            return {}

        @staticmethod
        def runtime_benchmark_base_block_config(_runtime_config):
            return base_block

        @staticmethod
        def runtime_benchmark_scaffold_policy(candidate):
            if candidate == scaffolded_block:
                return "attention_scaffold_per_pp_stage"
            return "none"

    captured = {}

    def run_benchmarks(specs, _gpu_ids, _cache_dir, measurement_pairs):
        captured["specs"] = specs
        captured["measurement_pairs"] = measurement_pairs
        results = {}
        for key, (_runtime, block_layout) in specs.items():
            base_count = sum(block == base_block for block in block_layout)
            scaffolded_count = sum(block == scaffolded_block for block in block_layout)
            results[key] = RuntimeMeasurement(
                total_ms=10.0 + 5.0 * base_count + 2.0 * scaffolded_count,
                prefill_ms=3.0 + 1.0 * base_count + 0.5 * scaffolded_count,
            )
        return results

    monkeypatch.setattr(runtime_stats_module, "_run_benchmarks", run_benchmarks)
    monkeypatch.setattr(runtime_stats_module, "_resolve_gpu_ids", lambda _size: ["0"])

    runtime, non_block = calc_runtime_for_blocks(
        {base_block, scaffolded_block},
        OmegaConf.create({"repeat_block_n_times": 2}),
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=8,
        num_key_value_heads=2,
        descriptor=Descriptor,
        lm_config=SimpleNamespace(),
        tokenizer_path="tokenizer",
        prefill_seq_len=16,
        generation_seq_len=4,
        batch_size=1,
    )

    assert len(captured["specs"]) == 4
    assert len(captured["measurement_pairs"]) == 3
    assert len(set(captured["measurement_pairs"])) == 2
    assert {key for pair in captured["measurement_pairs"] for key in pair} == set(captured["specs"])
    assert runtime[base_block] == RuntimeMeasurement(total_ms=5.0, prefill_ms=1.0)
    assert runtime[scaffolded_block] == RuntimeMeasurement(total_ms=2.0, prefill_ms=0.5)
    assert non_block == RuntimeMeasurement(total_ms=10.0, prefill_ms=3.0)


@pytest.mark.parametrize(
    "failure_source",
    ["candidate_module_import", "candidate_library_build"],
    ids=("candidate-module-import", "candidate-library-build"),
)
def test_build_library_propagates_candidate_import_errors_without_success_manifest(
    tmp_path, monkeypatch, failure_source
):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    config = {
        "experiment": {"dir": str(tmp_path)},
        "build_replacement_library": {},
        "search_space": {},
    }
    hydra_cfg = OmegaConf.create(
        {
            "puzzle_dir": str(tmp_path),
            "build_replacement_library": {},
            "calc_subblock_stats": {"subblock_stats_filename": "subblock_stats.json"},
        }
    )

    class ScoringParent:
        path = teacher_dir

        @staticmethod
        def to_dict():
            return {"path": str(teacher_dir)}

    monkeypatch.setattr(pipeline_stages, "load_runtime_hydra_config", lambda _: hydra_cfg)
    monkeypatch.setattr(
        pipeline_stages,
        "ensure_scoring_parent",
        lambda *_args, **_kwargs: ScoringParent(),
    )
    monkeypatch.setattr(pipeline_stages, "_distributed", lambda _: nullcontext())
    monkeypatch.setattr(pipeline_stages.dist, "is_master", lambda: True)
    monkeypatch.setattr(pipeline_stages.dist, "barrier", lambda: None)
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.replacement_library.build_replacement_library.launch_build_replacement_library",
        lambda _: None,
    )
    monkeypatch.setattr(pipeline_stages, "_calculate_static_workload_stats", lambda *_args: None)
    if failure_source == "candidate_module_import":
        monkeypatch.setitem(sys.modules, "modelopt.torch.puzzletron.candidates", None)
        error_match = "candidates"
    else:
        error_match = "candidate library build failed"

        def raise_candidate_library_import_error(*_args, **_kwargs):
            raise ImportError(error_match)

        monkeypatch.setattr(
            "modelopt.torch.puzzletron.candidates.build_candidate_library_from_checkpoint",
            raise_candidate_library_import_error,
        )

    with pytest.raises(ImportError, match=error_match):
        build_library_stage(config, StageManifest(stage="build_library", config=config))

    assert not (tmp_path / "manifests" / "build_library.json").exists()


def test_runtime_shards_are_complete_only_with_result_and_marker(tmp_path):
    for index in range(2):
        (tmp_path / f"shard_{index:04d}.json").write_text("{}\n")
        (tmp_path / f"shard_{index:04d}.done").write_text("{}\n")

    assert _runtime_shard_results_complete(tmp_path, shard_count=2)

    (tmp_path / "shard_0001.json").unlink()
    assert not _runtime_shard_results_complete(tmp_path, shard_count=2)


def test_runtime_measurement_arithmetic_and_json_round_trip():
    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import RuntimeMeasurement

    total = RuntimeMeasurement(total_ms=10.0, prefill_ms=4.0)
    baseline = RuntimeMeasurement(total_ms=2.0, prefill_ms=1.0)
    marginal = (total - baseline) / 2

    assert marginal == RuntimeMeasurement(total_ms=4.0, prefill_ms=1.5)
    assert marginal.decode_ms == 2.5
    assert marginal.decode_ms_per_token(6) == 0.5
    assert RuntimeMeasurement.from_dict(marginal.to_dict()) == marginal


def test_subblock_runtime_rejects_negative_marginal_phase(monkeypatch):
    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import RuntimeMeasurement

    base = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
            FFNConfig(no_op=True),
        )
    )
    moe = MoEConfig(num_experts=8, expert_intermediate_size=16, top_k=2)

    class Descriptor:
        @staticmethod
        def runtime_benchmark_config_fields(_config):
            return {}

        @staticmethod
        def runtime_benchmark_base_block_config(_runtime_config):
            return base

    def run_benchmarks(specs, _gpu_ids, _cache_dir, _measurement_pairs):
        results = {}
        for key, (_runtime, layout) in specs.items():
            if len(layout) == 3:
                value = RuntimeMeasurement(total_ms=10.0, prefill_ms=4.0)
            else:
                value = RuntimeMeasurement(total_ms=18.3, prefill_ms=14.0)
            results[key] = value
        return results

    monkeypatch.setattr(runtime_stats_module, "_run_benchmarks", run_benchmarks)
    monkeypatch.setattr(runtime_stats_module, "_resolve_gpu_ids", lambda _size: ["0"])

    with pytest.raises(ValueError, match="negative marginal phase"):
        calc_runtime_for_subblocks(
            {moe},
            OmegaConf.create({"repeat_block_n_times": 3}),
            vocab_size=32,
            hidden_size=16,
            num_attention_heads=8,
            num_key_value_heads=2,
            descriptor=Descriptor,
            lm_config=SimpleNamespace(),
            tokenizer_path="tokenizer",
            prefill_seq_len=16,
            generation_seq_len=4,
            batch_size=1,
        )

    with pytest.warns(RuntimeWarning, match="Ignoring negative marginal phase"):
        runtime_by_subblock, _ = calc_runtime_for_subblocks(
            {moe},
            OmegaConf.create({"repeat_block_n_times": 3, "ignore_negatives": True}),
            vocab_size=32,
            hidden_size=16,
            num_attention_heads=8,
            num_key_value_heads=2,
            descriptor=Descriptor,
            lm_config=SimpleNamespace(),
            tokenizer_path="tokenizer",
            prefill_seq_len=16,
            generation_seq_len=4,
            batch_size=1,
        )

    assert runtime_by_subblock[moe].decode_ms < 0


def test_vllm_runtime_topology_rejects_data_parallel_latency_measurement():
    from modelopt.torch.puzzletron.subblock_stats.topology import RuntimeTopology

    with pytest.raises(ValueError, match="data_parallel_size=1"):
        RuntimeTopology.from_config(
            {
                "tensor_parallel_size": 2,
                "data_parallel_size": 2,
                "gpu_group_size": 4,
            }
        )


def test_vllm_subprocess_env_assigns_fresh_rendezvous_port_for_mp(monkeypatch):
    from modelopt.torch.puzzletron.subblock_stats import runtime_vllm
    from modelopt.torch.puzzletron.subblock_stats.runtime_utils import RuntimeConfig

    runtime = RuntimeConfig(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        descriptor=object,
        model_config_fields=(),
        tokenizer_path="tokenizer",
        repeat_block_n_times=4,
        prefill_seq_len=8,
        generation_seq_len=4,
        batch_size=1,
        num_iters=1,
        num_warmup_iters=0,
    )
    assert runtime.topology.distributed_executor_backend == "mp"
    monkeypatch.setenv("MASTER_PORT", "28561")
    monkeypatch.setattr(runtime_vllm, "_free_tcp_port", lambda: 31001)

    env = runtime_vllm._build_subprocess_env("3", runtime.topology)

    assert env["VLLM_PORT"] == "31001"
    assert "MASTER_PORT" not in env


def test_vllm_environment_override_changes_runtime_cache_identity():
    from modelopt.torch.puzzletron.subblock_stats.runtime_utils import RuntimeConfig
    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import (
        _benchmark_cache_key,
        _runtime_environment_metadata,
    )

    common = {
        "vocab_size": 32,
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "descriptor": object,
        "model_config_fields": (),
        "tokenizer_path": "tokenizer",
        "repeat_block_n_times": 4,
        "prefill_seq_len": 8,
        "generation_seq_len": 4,
        "batch_size": 1,
        "num_iters": 1,
        "num_warmup_iters": 0,
    }
    fused = RuntimeConfig(
        **common,
        vllm_env=(("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "1"),),
    )
    fallback = RuntimeConfig(
        **common,
        vllm_env=(("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0"),),
    )

    model_config = {"model_type": "test"}
    fused_key = _benchmark_cache_key(
        model_config,
        {"vllm_env": _runtime_environment_metadata(fused)},
    )
    fallback_key = _benchmark_cache_key(
        model_config,
        {"vllm_env": _runtime_environment_metadata(fallback)},
    )

    assert fused_key != fallback_key


def test_vllm_failure_output_preserves_stdout_root_cause_and_stderr_warnings():
    from types import SimpleNamespace

    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import _called_process_failure_output

    output = _called_process_failure_output(
        SimpleNamespace(stdout="ROOT CAUSE ON STDOUT", stderr="CUDA WARNING ON STDERR")
    )

    assert "ROOT CAUSE ON STDOUT" in output
    assert "CUDA WARNING ON STDERR" in output


def test_vllm_editable_install_import_race_is_retryable_startup_failure():
    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import (
        _transient_distributed_startup_failure,
    )

    assert _transient_distributed_startup_failure(
        "ModuleNotFoundError: No module named 'vllm.v1.engine'"
    )
    assert not _transient_distributed_startup_failure(
        "ModuleNotFoundError: No module named 'modelopt.missing'"
    )


def test_short_mamba_smoke_uses_safe_aligned_cache_token_budget():
    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import _mamba_max_num_batched_tokens

    assert _mamba_max_num_batched_tokens(32) == 2048
    assert _mamba_max_num_batched_tokens(4096) == 8192
