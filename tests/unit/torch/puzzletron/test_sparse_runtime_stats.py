from immutabledict import immutabledict
import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import modelopt.torch.puzzletron.stages.pipeline as pipeline_stages
import modelopt.torch.puzzletron.subblock_stats.calc_runtime_stats as runtime_stats_module
from modelopt.torch.puzzletron.block_config import AttentionConfig, BlockConfig, FFNConfig, MoEConfig
from modelopt.torch.puzzletron.anymodel.capabilities import (
    CapabilityValidationError,
    default_capabilities,
)
from modelopt.torch.puzzletron.manifest import StageManifest
from modelopt.torch.puzzletron.stage_runner import _preflight
from modelopt.torch.puzzletron.subblock_stats.calc_subblock_stats import (
    _runtime_measurement_fields,
    _select_runtime_subblock_configs,
    _validate_sparse_runtime_settings,
)
from modelopt.torch.puzzletron.subblock_stats.calc_runtime_stats import (
    _merge_runtime_shard_results,
    _runtime_shard_results_complete,
    _runtime_shard_spec_identity,
    calc_runtime_for_subblocks,
    enumerate_runtime_block_configs,
)
from modelopt.torch.puzzletron.pipeline_config import (
    adapt_runtime_hydra_config,
    normalize_pipeline_config,
)
from modelopt.torch.puzzletron.stages import DEFAULT_HANDLERS
from modelopt.torch.puzzletron.stages.pipeline import (
    _vllm_stats_is_explicit,
    build_library_stage,
    vllm_stats_stage,
)


def _indexed(config, layer):
    return immutabledict(
        {"subblock_config": config, "parent_layer_indices": (layer,)}
    )


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


def test_subblock_runtime_uses_two_cache_bearing_base_layers_for_pp2(monkeypatch):
    attention = AttentionConfig(num_query_heads=8, num_kv_heads=2)
    base_block = BlockConfig(
        subblock_configs=(attention, FFNConfig(intermediate_size=16))
    )

    class Descriptor:
        @staticmethod
        def runtime_benchmark_config_fields(_config):
            return {}

        @staticmethod
        def runtime_benchmark_base_block_config(_runtime_config):
            return base_block

    captured_specs = []

    def run_benchmarks(specs, _gpu_ids, _cache_dir):
        from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import RuntimeMeasurement

        captured_specs.extend(specs.values())
        return {
            key: RuntimeMeasurement(total_ms=float(index + 10), prefill_ms=0.0)
            for index, key in enumerate(specs)
        }

    monkeypatch.setattr(runtime_stats_module, "_run_benchmarks", run_benchmarks)
    monkeypatch.setattr(runtime_stats_module, "_resolve_gpu_ids", lambda _size: ["0,1"])

    calc_runtime_for_subblocks(
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

    assert any(block is None and trailing for _, block, trailing in captured_specs)
    assert not any(block is None and not trailing for _, block, trailing in captured_specs)


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
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.subblock_stats.calc_subblock_stats.launch_calc_subblock_stats",
        lambda _: None,
    )

    with pytest.raises(RuntimeError, match="aggregate"):
        vllm_stats_stage(config, StageManifest(stage="vllm_stats", config=config))


@pytest.mark.parametrize("explicit_vllm", (False, True))
def test_build_library_uses_selected_vllm_producer(tmp_path, monkeypatch, explicit_vllm):
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
            },
        }
    )
    calls = []

    def assert_runtime_enabled(cfg):
        assert cfg.calc_subblock_stats.runtime_stats.enabled is True

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
        "modelopt.torch.puzzletron.build_library_and_stats.launch_build_library_and_stats",
        lambda cfg: (assert_runtime_enabled(cfg), calls.append("inline")),
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.candidates.build_candidate_library_from_checkpoint",
        lambda *args, **kwargs: calls.append("candidates"),
    )

    build_library_stage(config, StageManifest(stage="build_library", config=config))

    assert calls == (["library", "candidates"] if explicit_vllm else ["inline", "candidates"])


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
        for key, (runtime, block, trailing) in specs.items():
            if block is None and trailing:
                value = RuntimeMeasurement(total_ms=10.0, prefill_ms=4.0)
            elif runtime.repeat_block_n_times == 2:
                value = RuntimeMeasurement(total_ms=14.0, prefill_ms=6.0)
            elif block is not None and block.get_subblock("moe") is not None:
                value = RuntimeMeasurement(total_ms=18.3, prefill_ms=6.6)
            else:
                value = RuntimeMeasurement(total_ms=18.0, prefill_ms=6.0)
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


def test_vllm_failure_output_preserves_stdout_root_cause_and_stderr_warnings():
    from types import SimpleNamespace

    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import (
        _called_process_failure_output,
    )

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
    from modelopt.torch.puzzletron.subblock_stats.runtime_vllm import (
        _mamba_max_num_batched_tokens,
    )

    assert _mamba_max_num_batched_tokens(32) == 2048
    assert _mamba_max_num_batched_tokens(4096) == 8192
