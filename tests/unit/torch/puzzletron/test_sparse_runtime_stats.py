import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
from immutabledict import immutabledict
from omegaconf import OmegaConf

import modelopt.torch.puzzletron.stages.pipeline as pipeline_stages
import modelopt.torch.puzzletron.subblock_stats.calc_runtime_stats as runtime_stats_module
from examples.puzzletron import run_runtime_stats_packed as packed_runtime_stats
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
    adapt_runtime_hydra_config,
    load_runtime_hydra_config,
    normalize_pipeline_config,
    pipeline_config_from_path,
)
from modelopt.torch.puzzletron.stage_runner import _preflight
from modelopt.torch.puzzletron.stages import DEFAULT_HANDLERS
from modelopt.torch.puzzletron.stages.pipeline import (
    _vllm_stats_is_explicit,
    build_library_stage,
    vllm_stats_stage,
)
from modelopt.torch.puzzletron.subblock_stats.calc_runtime_stats import (
    _merge_runtime_shard_results,
    _runtime_shard_results_complete,
    _runtime_shard_spec_identity,
    calc_runtime_for_blocks,
    calc_runtime_for_subblocks,
    enumerate_runtime_block_configs,
)
from modelopt.torch.puzzletron.subblock_stats.calc_subblock_stats import (
    _load_parameter_inventory_cache,
    _parameter_inventory_progress,
    _reuse_runtime_stats,
    _runtime_measurement_fields,
    _select_runtime_subblock_configs,
    _unique_hidden_sizes,
    _validate_sparse_runtime_settings,
)
from modelopt.torch.puzzletron.subblock_stats.measurements import apply_vllm_measurement
from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import RuntimeMeasurement


def _indexed(config, layer):
    return immutabledict(
        {"subblock_config": config, "parent_layer_indices": (layer,)}
    )


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


def test_parameter_stats_hidden_widths_are_stably_deduplicated():
    assert _unique_hidden_sizes([2688, 2560, 2432], 2688) == (2688, 2560, 2432)


def test_parameter_inventory_cache_resumes_only_matching_identity(tmp_path):
    cache = tmp_path / "width-2560.json"
    cache.write_text(
        json.dumps(
            {
                "identity": "matching",
                "total": 2,
                "status": "running",
                "rows": [{"inventory_key": "first", "num_params": 11}],
            }
        )
    )

    assert _load_parameter_inventory_cache(cache, identity="matching", total=2)["rows"] == [
        {"inventory_key": "first", "num_params": 11}
    ]
    assert _load_parameter_inventory_cache(cache, identity="stale", total=2) is None
    assert _load_parameter_inventory_cache(cache, identity="matching", total=3) is None


def test_parameter_inventory_progress_reports_rate_and_eta():
    progress = _parameter_inventory_progress(
        width=2560,
        completed=25,
        total=100,
        elapsed_seconds=10.0,
        status="running",
    )

    assert progress["rate_per_second"] == pytest.approx(2.5)
    assert progress["eta_seconds"] == pytest.approx(30.0)
    assert progress["fraction_complete"] == pytest.approx(0.25)


def test_runtime_stats_can_be_reused_while_static_metrics_are_refreshed():
    config = AttentionConfig(num_query_heads=8, num_kv_heads=2)
    target = {
        "args": {"runtime_stats": False, "n_embd": 32},
        "subblocks": [
            {
                "subblock_config": config.to_dict(),
                "parent_layer_index": 0,
                "num_params": 123,
                "runtime_ms": None,
                "additive_metric_provenance": {"num_params": "refreshed"},
            },
            {
                "subblock_config": config.to_dict(),
                "parent_layer_index": 7,
                "num_params": 123,
                "runtime_ms": None,
                "additive_metric_provenance": {},
            },
        ],
        "non_block": {"num_params": 45, "runtime_ms": None},
    }
    source = {
        "args": {
            "runtime_stats": True,
            "runtime_granularity": "subblock",
            "runtime_backend": "vllm",
        },
        "subblocks": [
            {
                "subblock_config": {
                    **config.to_dict(),
                    # Historical robust JSON kept explicit nullable fields.
                    "qk_head_dim": None,
                },
                "parent_layer_index": 0,
                "runtime_ms": 7.0,
                "prefill_runtime_ms": 3.0,
                "decode_runtime_ms": 4.0,
                "decode_runtime_ms_per_token": 2.0,
                "latency_difference_negative": False,
                "additive_metric_provenance": {"runtime_ms": "vllm_measured"},
            }
        ],
        "non_block": {"runtime_ms": 1.0},
    }

    result = _reuse_runtime_stats(target, source, source_path="existing.json")

    assert result["args"]["runtime_stats"] is True
    assert result["args"]["runtime_reuse_source"] == "existing.json"
    assert result["subblocks"][0]["runtime_ms"] == 7.0
    assert result["subblocks"][1]["runtime_ms"] == 7.0
    assert result["subblocks"][0]["num_params"] == 123
    assert result["subblocks"][0]["additive_metric_provenance"] == {
        "num_params": "refreshed",
        "runtime_ms": "vllm_measured",
    }
    assert result["non_block"]["runtime_ms"] == 1.0


def test_runtime_block_candidates_load_converted_teacher_without_replacement_library(tmp_path):
    attention = AttentionConfig(num_query_heads=8, num_kv_heads=2)
    teacher_block = {
        "subblock_configs": [
            attention.to_dict(),
            FFNConfig(intermediate_size=16).to_dict(),
        ]
    }
    candidate_block = {
        "subblock_configs": [
            attention.to_dict(),
            FFNConfig(intermediate_size=8).to_dict(),
        ]
    }
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    (teacher_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "num_hidden_layers": 3,
                "vocab_size": 32,
                "block_configs": [teacher_block, candidate_block, teacher_block],
            }
        )
    )

    class CapabilityDescriptor:
        @staticmethod
        def requires_trust_remote_code():
            return False

        @staticmethod
        def get_language_model_config(model_config):
            return model_config

    candidates = enumerate_runtime_block_configs(
        teacher_dir,
        CapabilityDescriptor,
        search_space={"ffn_intermediate": {"values": [4]}},
    )

    assert candidates == (
        BlockConfig(
            subblock_configs=(attention, FFNConfig(intermediate_size=16))
        ),
        BlockConfig(
            subblock_configs=(attention, FFNConfig(intermediate_size=4)),
        ),
        BlockConfig(
            subblock_configs=(attention, FFNConfig(intermediate_size=8)),
        ),
    )


def test_explicit_vllm_stage_defaults_to_block_granularity():
    config = normalize_pipeline_config({"vllm_stats": {"enabled": True}})
    runtime_config = adapt_runtime_hydra_config(
        OmegaConf.create({"calc_subblock_stats": {"runtime_stats": {}}}), config
    )

    assert config["vllm_stats"] == {"enabled": True}
    assert runtime_config.calc_subblock_stats.runtime_stats.enabled is True
    assert runtime_config.calc_subblock_stats.runtime_stats.granularity == "block"


def test_disabled_vllm_stage_preserves_legacy_runtime_collection():
    config = normalize_pipeline_config({"vllm_stats": {"enabled": False}})
    runtime_config = adapt_runtime_hydra_config(
        OmegaConf.create({"calc_subblock_stats": {"runtime_stats": {"enabled": True}}}),
        config,
    )

    assert runtime_config.calc_subblock_stats.runtime_stats.enabled is True


def test_runtime_candidate_enumeration_honors_library_noop_policy(tmp_path):
    attention = AttentionConfig(num_query_heads=8, num_kv_heads=2)
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    (teacher_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "num_hidden_layers": 1,
                "vocab_size": 32,
                "block_configs": [
                    {
                        "subblock_configs": [
                            attention.to_dict(),
                            FFNConfig(intermediate_size=16).to_dict(),
                        ]
                    }
                ],
            }
        )
    )

    class CapabilityDescriptor:
        @staticmethod
        def requires_trust_remote_code():
            return False

        @staticmethod
        def get_language_model_config(model_config):
            return model_config

    candidates = enumerate_runtime_block_configs(
        teacher_dir,
        CapabilityDescriptor,
        search_space={"no_op": {"subblocks": ["ffn"]}},
        include_noops=False,
    )

    assert all(not subblock.no_op for candidate in candidates for subblock in candidate.subblock_configs)


def test_explicit_vllm_stage_requires_descriptor_vllm_capability():
    capabilities = default_capabilities(descriptor_name="unsupported")
    resolution = SimpleNamespace(
        capabilities=replace(capabilities, export=replace(capabilities.export, vllm=False))
    )

    with pytest.raises(CapabilityValidationError, match="does not support vLLM export"):
        _preflight({"vllm_stats": {"enabled": True}}, resolution, stage="vllm_stats")


def test_explicit_vllm_stage_is_registered_and_disables_inline_collection():
    assert DEFAULT_HANDLERS["vllm_stats"] is vllm_stats_stage
    assert _vllm_stats_is_explicit({"vllm_stats": {"enabled": True}})
    assert not _vllm_stats_is_explicit({})


def test_vllm_width_configuration_merges_embedding_search_widths():
    hydra_cfg = OmegaConf.create(
        {"calc_subblock_stats": {"model_hidden_sizes": [1024]}}
    )

    requested = pipeline_stages.configure_vllm_stats_widths(
        {"embedding_pruning": {"widths": [768, 1024]}}, hydra_cfg
    )

    assert requested == (1024, 768)
    assert hydra_cfg.calc_subblock_stats.model_hidden_sizes == [1024, 768]


def test_vllm_stage_collects_from_converted_teacher_without_replacement_library(
    tmp_path, monkeypatch
):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    config = {
        "experiment": {"dir": str(tmp_path)},
        "convert": {"teacher_dir": str(teacher_dir)},
        "search_space": {"ffn_intermediate": {"values": [8]}},
        "vllm_stats": {"enabled": True},
    }
    hydra_cfg = OmegaConf.create(
        {
            "puzzle_dir": str(tmp_path),
            "teacher_dir": str(teacher_dir),
            "descriptor": "fixture",
            "calc_subblock_stats": {
                "runtime_stats": {"granularity": "block"},
                "subblock_stats_filename": "subblock_stats.json",
            },
        }
    )
    block_config = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
            FFNConfig(intermediate_size=8),
        )
    )
    monkeypatch.setattr(pipeline_stages, "load_runtime_hydra_config", lambda _: hydra_cfg)
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.anymodel.model_descriptor.ModelDescriptorFactory.get",
        lambda _: object(),
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.subblock_stats.calc_runtime_stats.enumerate_runtime_block_configs",
        lambda teacher, descriptor, *, search_space, include_noops: (
            (block_config,) if include_noops else ()
        ),
    )
    # Convert is the sole writer of the runtime candidate library.
    (tmp_path / "subblock_library.json").write_text(
        json.dumps(
            [
                {
                    "attention_config": block_config.subblock_configs[0].to_dict(),
                    "ffn_config": block_config.subblock_configs[1].to_dict(),
                }
            ]
        )
    )

    def launch(cfg):
        assert cfg.calc_subblock_stats.runtime_stats.enabled is True
        assert not (tmp_path / "replacement_library.json").exists()
        rows = json.loads((tmp_path / "subblock_library.json").read_text())
        assert rows == [
            {
                "attention_config": block_config.subblock_configs[0].to_dict(),
                "ffn_config": block_config.subblock_configs[1].to_dict(),
            }
        ]
        (tmp_path / "subblock_stats.json").write_text(
            json.dumps(
                [
                    {
                        "args": {"runtime_stats": True},
                        "block_runtime_records": [
                            {
                                "block_config": block_config.to_dict(),
                                "runtime_ms": 1.0,
                            },
                            {
                                "block_config": BlockConfig(
                                    subblock_configs=(
                                        block_config.subblock_configs[0],
                                        FFNConfig(intermediate_size=4),
                                    )
                                ).to_dict(),
                                "runtime_ms": 0.8,
                            },
                        ],
                    }
                ]
            )
        )

    monkeypatch.setattr(
        "modelopt.torch.puzzletron.subblock_stats.calc_subblock_stats.launch_calc_subblock_stats",
        launch,
    )

    result = vllm_stats_stage(config, StageManifest(stage="vllm_stats", config=config))

    assert result.status == "success"
    assert not (tmp_path / "replacement_library.json").exists()
    assert (tmp_path / "artifacts/vllm_stats/summary.json").is_file()


def test_finalize_vllm_stats_report_uses_configured_aggregate(tmp_path, monkeypatch):
    stats_path = tmp_path / "runtime/custom_stats.json"
    stats_path.parent.mkdir(parents=True)
    stats_path.write_text("[{}]\n")
    config = {"experiment": {"dir": str(tmp_path)}}
    hydra_cfg = OmegaConf.create(
        {
            "puzzle_dir": str(tmp_path),
            "calc_subblock_stats": {
                "runtime_stats": {"granularity": "subblock"},
                "subblock_stats_filename": "runtime/custom_stats.json",
            },
        }
    )
    calls = []

    def generate(puzzle_dir, **kwargs):
        calls.append((puzzle_dir, kwargs))
        return {"kind": "vllm_stats", "record_count": 1}

    monkeypatch.setattr(
        "modelopt.torch.puzzletron.diagnostics.generate_vllm_stats_report",
        generate,
    )

    summary = pipeline_stages.finalize_vllm_stats_report(config, hydra_cfg)

    assert summary == {"kind": "vllm_stats", "record_count": 1}
    assert calls == [
        (
            tmp_path,
            {
                "stats_path": stats_path,
                "output_dir": tmp_path / "artifacts/vllm_stats",
                "granularity": "subblock",
            },
        )
    ]


@pytest.mark.parametrize(
    ("shard_indices", "worker_exit_code", "expected_finalizations"),
    (("0,1", 0, 1), ("2,3", 0, 0), ("0,1", 7, 0)),
)
def test_packed_runtime_stats_finalizes_only_successful_shard_zero_pack(
    monkeypatch,
    shard_indices,
    worker_exit_code,
    expected_finalizations,
):
    class Process:
        def wait(self):
            return worker_exit_code

    monkeypatch.setattr(packed_runtime_stats.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(
        packed_runtime_stats,
        "pipeline_config_from_path",
        lambda path, overrides: {"config_path": path, "overrides": overrides},
        raising=False,
    )
    finalizations = []
    monkeypatch.setattr(
        packed_runtime_stats,
        "finalize_vllm_stats_report",
        lambda config: finalizations.append(config),
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_runtime_stats_packed.py",
            "--config",
            "experiment.yaml",
            "--shard-indices",
            shard_indices,
            "--shard-count",
            "4",
            "--override",
            "vllm_stats.enabled=true",
        ],
    )

    assert packed_runtime_stats.main() == worker_exit_code
    assert len(finalizations) == expected_finalizations


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
    base_block = BlockConfig(
        subblock_configs=(attention, FFNConfig(no_op=True))
    )

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

    def run_benchmarks(specs, _gpu_ids, _cache_dir):
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
    assert all(block == attention.to_blockconfig() for layout in captured_layouts for block in layout)
    assert runtime[attention] == RuntimeMeasurement(total_ms=2.0, prefill_ms=0.5)
    assert non_block == RuntimeMeasurement(total_ms=6.0, prefill_ms=2.0)


def test_scaffolded_ffn_slope_cancels_one_attention_per_pp_stage(monkeypatch):
    attention = AttentionConfig(num_query_heads=8, num_kv_heads=2)
    ffn = FFNConfig(intermediate_size=16)
    scaffold = BlockConfig(
        subblock_configs=(attention, FFNConfig(no_op=True))
    )

    class Descriptor:
        @staticmethod
        def runtime_benchmark_config_fields(_config):
            return {}

        @staticmethod
        def runtime_benchmark_base_block_config(_runtime_config):
            return scaffold

        @staticmethod
        def runtime_benchmark_scaffold_policy(candidate):
            candidate_attention = candidate.get_subblock("attention")
            if candidate_attention is None or candidate_attention.no_op:
                return "attention_scaffold_per_pp_stage"
            return "none"

    captured_layouts = []

    def run_benchmarks(specs, _gpu_ids, _cache_dir):
        from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import RuntimeMeasurement

        results = {}
        for key, (_runtime, layout) in specs.items():
            captured_layouts.append(layout)
            scaffold_count = sum(block == scaffold for block in layout)
            candidate_count = len(layout) - scaffold_count
            results[key] = RuntimeMeasurement(
                total_ms=6.0 + 5.0 * scaffold_count + 2.0 * candidate_count,
                prefill_ms=2.0 + 1.0 * scaffold_count + 0.5 * candidate_count,
            )
        return results

    monkeypatch.setattr(runtime_stats_module, "_run_benchmarks", run_benchmarks)
    monkeypatch.setattr(runtime_stats_module, "_resolve_gpu_ids", lambda _size: ["0,1"])

    runtime, non_block = calc_runtime_for_subblocks(
        {ffn},
        OmegaConf.create(
            {
                "repeat_block_n_times": 3,
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

    candidate_layouts = [
        layout
        for layout in captured_layouts
        if any(block.get_subblock("ffn") == ffn for block in layout)
    ]
    assert sorted(len(layout) for layout in candidate_layouts) == [6, 10]
    assert [sum(block == scaffold for block in layout) for layout in candidate_layouts] == [2, 2]
    assert runtime[ffn] == RuntimeMeasurement(total_ms=2.0, prefill_ms=0.5)
    assert non_block == RuntimeMeasurement(total_ms=6.0, prefill_ms=2.0)


def test_block_runtime_uses_homogeneous_n_and_2n_layouts(monkeypatch):
    block = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
            FFNConfig(intermediate_size=16),
        )
    )

    class Descriptor:
        @staticmethod
        def runtime_benchmark_config_fields(_config):
            return {}

        @staticmethod
        def runtime_benchmark_base_block_config(_runtime_config):
            return block

    captured_layouts = []

    def run_benchmarks(specs, _gpu_ids, _cache_dir):
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

    runtime, non_block = calc_runtime_for_blocks(
        {block},
        OmegaConf.create(
            {
                "repeat_block_n_times": 3,
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
    assert all(candidate == block for layout in captured_layouts for candidate in layout)
    assert runtime[block] == RuntimeMeasurement(total_ms=2.0, prefill_ms=0.5)
    assert non_block == RuntimeMeasurement(total_ms=6.0, prefill_ms=2.0)


@pytest.mark.parametrize(
    ("calculator", "candidate"),
    (
        (
            calc_runtime_for_blocks,
            BlockConfig(
                subblock_configs=(AttentionConfig(num_query_heads=8, num_kv_heads=2),)
            ),
        ),
        (
            calc_runtime_for_subblocks,
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
        ),
    ),
    ids=("block", "subblock"),
)
def test_block_and_subblock_runtime_default_to_four_repeats(
    monkeypatch, calculator, candidate
):
    base_block = BlockConfig(
        subblock_configs=(AttentionConfig(num_query_heads=8, num_kv_heads=2),)
    )

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

    def run_benchmarks(specs, _gpu_ids, _cache_dir):
        results = {}
        for key, (_runtime, layout) in specs.items():
            captured_layouts.append(layout)
            results[key] = RuntimeMeasurement(
                total_ms=6.0 + 2.0 * len(layout),
                prefill_ms=2.0 + 0.5 * len(layout),
            )
        return results

    monkeypatch.setattr(runtime_stats_module, "_run_benchmarks", run_benchmarks)
    monkeypatch.setattr(runtime_stats_module, "_resolve_gpu_ids", lambda _size: ["0"])

    calculator(
        {candidate},
        OmegaConf.create({}),
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


def test_exclusive_sublayer_runtime_candidate_does_not_keep_attention_active():
    base_block = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
            FFNConfig(no_op=True),
        )
    )

    class Descriptor:
        @staticmethod
        def runtime_benchmark_base_block_config(_runtime_config):
            return base_block

        @staticmethod
        def runtime_benchmark_sublayers_are_exclusive():
            return True

    runtime_config = SimpleNamespace(descriptor=Descriptor)
    candidate = runtime_stats_module._block_config_for_subblock(
        runtime_config,
        MoEConfig(num_experts=8, expert_intermediate_size=16, top_k=2),
    )

    active = [
        subblock.kind for subblock in candidate.subblock_configs if not subblock.no_op
    ]
    assert active == ["moe"]


def test_runtime_subblock_builder_isolates_generic_ffn():
    base_block = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
            FFNConfig(intermediate_size=32),
        )
    )

    class Descriptor:
        @staticmethod
        def runtime_benchmark_base_block_config(_runtime_config):
            return base_block

    runtime = SimpleNamespace(descriptor=Descriptor)
    candidate = runtime_stats_module._block_config_for_subblock(
        runtime, FFNConfig(intermediate_size=16)
    )

    assert candidate.require_subblock("ffn").no_op is False
    assert candidate.require_subblock("attention").no_op is True


def test_runtime_subblock_builder_isolates_generic_attention():
    base_block = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
            FFNConfig(intermediate_size=32),
        )
    )

    class Descriptor:
        @staticmethod
        def runtime_benchmark_base_block_config(_runtime_config):
            return base_block

    runtime = SimpleNamespace(descriptor=Descriptor)
    candidate = runtime_stats_module._block_config_for_subblock(
        runtime, AttentionConfig(num_query_heads=8, num_kv_heads=2)
    )

    assert candidate.require_subblock("attention").no_op is False
    assert candidate.require_subblock("ffn").no_op is True


def test_vllm_stage_rejects_missing_runtime_aggregate(tmp_path, monkeypatch):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    config = {
        "experiment": {"dir": str(tmp_path)},
        "convert": {"teacher_dir": str(teacher_dir)},
        "vllm_stats": {"enabled": True},
    }
    hydra_cfg = OmegaConf.create(
        {
            "puzzle_dir": str(tmp_path),
            "teacher_dir": str(teacher_dir),
            "descriptor": "fixture",
            "calc_subblock_stats": {
                "runtime_stats": {"granularity": "block"},
                "subblock_stats_filename": "subblock_stats.json",
            },
        }
    )
    monkeypatch.setattr(pipeline_stages, "load_runtime_hydra_config", lambda _: hydra_cfg)
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.anymodel.model_descriptor.ModelDescriptorFactory.get",
        lambda _: object(),
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.subblock_stats.calc_runtime_stats.enumerate_runtime_block_configs",
        lambda *args, **kwargs: (
            BlockConfig(
                subblock_configs=(
                    AttentionConfig(num_query_heads=8, num_kv_heads=2),
                    FFNConfig(intermediate_size=8),
                )
            ),
        ),
    )
    (tmp_path / "subblock_library.json").write_text("[]\n")
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.subblock_stats.calc_subblock_stats.launch_calc_subblock_stats",
        lambda _: None,
    )

    with pytest.raises(RuntimeError, match="aggregate"):
        vllm_stats_stage(config, StageManifest(stage="vllm_stats", config=config))


@pytest.mark.parametrize("explicit_vllm", (False, True))
def test_build_library_refreshes_static_stats_for_each_vllm_mode(
    tmp_path, monkeypatch, explicit_vllm
):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    stats_path = tmp_path / "subblock_stats.json"
    if explicit_vllm:
        stats_path.write_text("[]\n")
    config = {
        "experiment": {"dir": str(tmp_path)},
        "vllm_stats": {"enabled": explicit_vllm},
        "build_replacement_library": {},
        "search_space": {},
    }
    hydra_cfg = OmegaConf.create(
        {
            "puzzle_dir": str(tmp_path),
            "build_replacement_library": {},
            "calc_subblock_stats": {
                "runtime_stats": {"enabled": True, "execution": "inline"},
                "subblock_stats_filename": "subblock_stats.json",
                "batch_sizes": [2],
                "prefill_seq_len": 32,
                "generation_seq_len": 8,
            },
        }
    )
    calls = []

    def record_static_stats(cfg):
        assert cfg.calc_subblock_stats.runtime_stats.enabled is False
        assert cfg.calc_subblock_stats.merge_with_existing_stats is True
        assert cfg.calc_subblock_stats.batch_sizes == [2]
        assert cfg.calc_subblock_stats.prefill_seq_len == 32
        assert cfg.calc_subblock_stats.generation_seq_len == 8
        calls.append("static")

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
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.replacement_library.build_replacement_library.launch_build_replacement_library",
        lambda _: calls.append("library"),
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.subblock_stats.calc_subblock_stats.launch_calc_subblock_stats",
        record_static_stats,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.candidates.build_candidate_library_from_checkpoint",
        lambda *args, **kwargs: calls.append("candidates"),
    )

    build_library_stage(config, StageManifest(stage="build_library", config=config))

    assert calls == ["library", "static", "candidates"]


def test_sparse_runtime_selection_is_unique_and_layer_independent():
    teacher_ffn = FFNConfig(intermediate_size=16)
    reduced_ffn = FFNConfig(intermediate_size=8)
    attention = AttentionConfig(num_query_heads=8, num_kv_heads=2)
    available = [
        _indexed(teacher_ffn, 0),
        _indexed(teacher_ffn, 7),
        _indexed(reduced_ffn, 0),
        _indexed(attention, 3),
    ]
    manifest = {
        "identity": "sparse_samples_123",
        "selected": [
            {"subblock_config": teacher_ffn.to_dict()},
            {"subblock_config": reduced_ffn.to_dict()},
        ],
    }

    selected = _select_runtime_subblock_configs(available, manifest)

    assert {row["subblock_config"] for row in selected} == {reduced_ffn, teacher_ffn}
    assert selected == _select_runtime_subblock_configs(available, manifest)
    assert all(row["parent_layer_indices"] == (-1,) for row in selected)


def test_sparse_runtime_selection_rejects_unknown_config():
    manifest = {
        "identity": "sparse_samples_123",
        "selected": [{"subblock_config": FFNConfig(intermediate_size=4).to_dict()}],
    }
    with pytest.raises(ValueError, match="not present in the canonical library"):
        _select_runtime_subblock_configs(
            [_indexed(FFNConfig(intermediate_size=16), 0)], manifest
        )


def test_sparse_runtime_requires_subblock_two_warmup_three_measured_iterations():
    _validate_sparse_runtime_settings(
        {"granularity": "subblock", "num_warmup_iters": 2, "num_iters": 3}
    )
    with pytest.raises(ValueError, match="num_warmup_iters=2 and num_iters=3"):
        _validate_sparse_runtime_settings(
            {"granularity": "subblock", "num_warmup_iters": 10, "num_iters": 30}
        )
    with pytest.raises(ValueError, match="subblock granularity"):
        _validate_sparse_runtime_settings(
            {"granularity": "block", "num_warmup_iters": 2, "num_iters": 3}
        )


def test_runtime_shards_merge_serialized_results_without_replaying_models(tmp_path):
    ordered_items = [(('spec-a',), None), (('spec-b',), None), (('spec-c',), None)]
    (tmp_path / "shard_0000.json").write_text(
        '{"spec_identity":"identity","results":{'
        '"0":{"total_ms":1.25,"prefill_ms":0.5},'
        '"2":{"total_ms":3.75,"prefill_ms":1.5}}}\n'
    )
    (tmp_path / "shard_0001.json").write_text(
        '{"spec_identity":"identity","results":{'
        '"1":{"total_ms":2.5,"prefill_ms":1.0}}}\n'
    )

    merged = _merge_runtime_shard_results(
        ordered_items,
        status_dir=tmp_path,
        shard_count=2,
        spec_identity="identity",
    )

    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import RuntimeMeasurement

    assert merged == {
        ('spec-a',): RuntimeMeasurement(total_ms=1.25, prefill_ms=0.5),
        ('spec-b',): RuntimeMeasurement(total_ms=2.5, prefill_ms=1.0),
        ('spec-c',): RuntimeMeasurement(total_ms=3.75, prefill_ms=1.5),
    }


def test_runtime_shards_are_complete_only_with_result_and_marker(tmp_path):
    for index in range(2):
        (tmp_path / f"shard_{index:04d}.json").write_text("{}\n")
        (tmp_path / f"shard_{index:04d}.done").write_text("{}\n")

    assert _runtime_shard_results_complete(tmp_path, shard_count=2)

    (tmp_path / "shard_0001.json").unlink()
    assert not _runtime_shard_results_complete(tmp_path, shard_count=2)


def test_runtime_shard_identity_includes_result_schema_version():
    ordered_items = [(("spec-a",), None), (("spec-b",), None)]

    assert _runtime_shard_spec_identity(
        ordered_items, result_schema_version=2
    ) != _runtime_shard_spec_identity(ordered_items, result_schema_version=3)


def test_runtime_measurement_arithmetic_and_json_round_trip():
    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import RuntimeMeasurement

    total = RuntimeMeasurement(total_ms=10.0, prefill_ms=4.0)
    baseline = RuntimeMeasurement(total_ms=2.0, prefill_ms=1.0)
    marginal = (total - baseline) / 2

    assert marginal == RuntimeMeasurement(total_ms=4.0, prefill_ms=1.5)
    assert marginal.decode_ms == 2.5
    assert marginal.decode_ms_per_token(6) == 0.5
    assert RuntimeMeasurement.from_dict(marginal.to_dict()) == marginal


def test_runtime_measurement_mean_preserves_both_latency_phases():
    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import RuntimeMeasurement

    result = RuntimeMeasurement.mean(
        [
            RuntimeMeasurement(total_ms=8.0, prefill_ms=2.0),
            RuntimeMeasurement(total_ms=12.0, prefill_ms=4.0),
        ]
    )

    assert result == RuntimeMeasurement(total_ms=10.0, prefill_ms=3.0)
    with pytest.raises(ValueError, match="at least one"):
        RuntimeMeasurement.mean([])


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

    def run_benchmarks(specs, _gpu_ids, _cache_dir):
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
            OmegaConf.create(
                {"repeat_block_n_times": 3, "ignore_negatives": True}
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

    assert runtime_by_subblock[moe].decode_ms < 0


def test_runtime_measurement_fields_save_phase_breakdown_and_noise_flag():
    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import RuntimeMeasurement

    fields = _runtime_measurement_fields(
        RuntimeMeasurement(total_ms=8.0, prefill_ms=3.0),
        generation_seq_len=6,
    )

    assert fields["runtime_ms"] == 8.0
    assert fields["prefill_runtime_ms"] == 3.0
    assert fields["decode_runtime_ms"] == 5.0
    assert fields["decode_runtime_ms_per_token"] == 1.0
    assert fields["latency_difference_negative"] is False
    assert fields["additive_metric_provenance"] == {
        "runtime_ms": "vllm_measured",
        "prefill_runtime_ms": "vllm_measured_prompt_plus_one_output",
        "decode_runtime_ms": "combined_minus_prefill",
        "decode_runtime_ms_per_token": "combined_minus_prefill_per_remaining_output",
    }


def test_runtime_config_restores_frozen_mapping_values():
    from modelopt.torch.puzzletron.subblock_stats.runtime_utils import RuntimeConfig

    runtime = RuntimeConfig(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        descriptor=object,
        model_config_fields=(("rope_parameters", (("factor", 32.0), ("type", "yarn"))),),
        tokenizer_path="tokenizer",
        repeat_block_n_times=3,
        prefill_seq_len=8,
        generation_seq_len=4,
        batch_size=1,
        num_iters=1,
        num_warmup_iters=0,
    )

    assert runtime.model_config_value("rope_parameters") == {
        "factor": 32.0,
        "type": "yarn",
    }


def test_runtime_cache_schema_separates_candidate_slope_from_legacy_cache():
    from modelopt.torch.puzzletron.subblock_stats import runtime_vllm

    assert runtime_vllm._RUNTIME_CACHE_SCHEMA_VERSION == 5


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


def test_runtime_config_records_candidate_slope_estimator_metadata():
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

    assert runtime.estimator_schema == "candidate_slope_v1"
    assert runtime.estimator_mode == "homogeneous"
    assert runtime.effective_repeat_count is None
    assert runtime.scaffold_policy == "none"


def test_vllm_subprocess_env_applies_explicit_runtime_overrides(monkeypatch):
    from modelopt.torch.puzzletron.subblock_stats.runtime_utils import RuntimeConfig
    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import _build_subprocess_env

    monkeypatch.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "1")
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
        vllm_env=(("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0"),),
    )

    env = _build_subprocess_env(
        "3", runtime.topology, runtime.vllm_env
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["VLLM_USE_FUSED_MOE_GROUPED_TOPK"] == "0"


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


def test_vllm_environment_config_is_frozen_in_stable_string_order():
    from modelopt.torch.puzzletron.subblock_stats.calc_runtime_stats import _vllm_env

    runtime_stats = OmegaConf.create(
        {
            "vllm_env": {
                "VLLM_USE_FUSED_MOE_GROUPED_TOPK": 0,
                "VLLM_LOGGING_LEVEL": "WARNING",
            }
        }
    )

    assert _vllm_env(runtime_stats) == (
        ("VLLM_LOGGING_LEVEL", "WARNING"),
        ("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0"),
    )


def test_runtime_cache_metadata_records_vllm_environment_overrides():
    from modelopt.torch.puzzletron.subblock_stats.runtime_utils import RuntimeConfig
    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import _runtime_environment_metadata

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
        vllm_env=(("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0"),),
    )

    assert _runtime_environment_metadata(runtime) == {
        "VLLM_USE_FUSED_MOE_GROUPED_TOPK": "0"
    }


def test_vllm_environment_override_changes_runtime_cache_identity():
    from modelopt.torch.puzzletron.subblock_stats.runtime_utils import RuntimeConfig
    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import (
        _benchmark_cache_key,
        _runtime_environment_metadata,
    )

    common = dict(
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


def test_runtime_cache_identity_records_estimator_metadata():
    from modelopt.torch.puzzletron.subblock_stats.runtime_utils import RuntimeConfig
    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import _runtime_estimator_metadata

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
        effective_repeat_count=4,
    )

    assert _runtime_estimator_metadata(runtime) == {
        "schema": "candidate_slope_v1",
        "mode": "homogeneous",
        "effective_repeat_count": 4,
        "scaffold_policy": "none",
    }


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
