import json
from types import SimpleNamespace

import pytest
import torch
from immutabledict import immutabledict
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
)
from modelopt.torch.puzzletron.diagnostics.html_report import generate_vllm_stats_report
from modelopt.torch.puzzletron.diagnostics.sweep_data import load_vllm_records
from modelopt.torch.puzzletron.mip.run_puzzle import _get_block_stats
from modelopt.torch.puzzletron.subblock_stats import calc_subblock_params_and_memory
from modelopt.torch.puzzletron.subblock_stats import calc_subblock_stats as stats_module
from modelopt.torch.puzzletron.subblock_stats.calc_subblock_params_and_memory import (
    calc_subblock_active_params,
    calculate_additive_metrics,
    calculate_attention_memory,
)
from modelopt.torch.puzzletron.subblock_stats.calc_subblock_stats import (
    _parameter_inventory_key,
    _subblock_stats_already_complete,
    calculate_subblock_stats,
)


def _metrics(config, *, num_params, active_params, prefill=4, generation=3):
    return calculate_additive_metrics(
        config,
        model_config=SimpleNamespace(),
        descriptor=object,
        batch_size=2,
        prefill_seq_len=prefill,
        generation_seq_len=generation,
        n_embd=32,
        n_head=8,
        weights_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        num_params=num_params,
        active_params=active_params,
    )


def test_checkpoint_tensor_count_includes_persistent_buffers():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.empty(2, 3))
    module.register_buffer("router_correction", torch.empty(5))
    module.register_buffer("scratch", torch.empty(7), persistent=False)

    assert calc_subblock_params_and_memory._checkpoint_tensor_count(module) == 11


def test_attention_memory_uses_precomputed_parameter_count(monkeypatch):
    monkeypatch.setattr(
        calc_subblock_params_and_memory,
        "calculate_subblock_params",
        lambda *_args, **_kwargs: pytest.fail("parameter count was recomputed"),
    )
    monkeypatch.setattr(
        calc_subblock_params_and_memory,
        "calculate_kv_dim",
        lambda *_args, **_kwargs: 8,
    )

    result = calculate_attention_memory(
        AttentionConfig(num_query_heads=8, num_kv_heads=2),
        SimpleNamespace(),
        object,
        batch_size=2,
        prefill_seq_len=4,
        generation_seq_len=1,
        n_embd=32,
        n_head=8,
        weights_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        num_params=123,
    )

    assert result["memory_mib"] == pytest.approx((2 * 5 * 8 * 2 + 123 * 2) / 2**20)


def test_dense_active_params_use_precomputed_total(monkeypatch):
    monkeypatch.setattr(
        calc_subblock_params_and_memory,
        "calculate_subblock_params",
        lambda *_args, **_kwargs: pytest.fail("parameter count was recomputed"),
    )

    assert (
        calc_subblock_active_params(
            FFNConfig(intermediate_size=16),
            SimpleNamespace(),
            object,
            n_embd=32,
            num_params=123,
        )
        == 123
    )


def test_stats_assembly_uses_parameter_inventory_without_model_construction(
    tmp_path, monkeypatch
):
    subblock = FFNConfig(intermediate_size=16)
    indexed = immutabledict(
        {"subblock_config": subblock, "parent_layer_indices": (0,)}
    )

    class Descriptor:
        @staticmethod
        def runtime_benchmark_supported():
            return False

        @staticmethod
        def get_language_model_config(config):
            return config

        @staticmethod
        def truncate_pattern_for_subblock(_config, _layer):
            pytest.fail("width config was copied/truncated during inventory assembly")

    monkeypatch.setattr(
        stats_module,
        "calculate_subblock_params",
        lambda *_args, **_kwargs: pytest.fail("parameter count was recomputed"),
    )
    inventory = {
        "identity": "inventory-v1",
        "rows": [
            {
                "inventory_key": _parameter_inventory_key(subblock, 0),
                "num_params": 123,
                "active_params": 123,
            }
        ],
        "non_block_params": 45,
        "non_block_parameter_count_source": "checkpoint_tensor_inventory",
    }

    result = calculate_subblock_stats(
        OmegaConf.create({}),
        teacher_dir=tmp_path,
        model_config=SimpleNamespace(num_hidden_layers=1),
        descriptor=Descriptor,
        master_puzzle_dir=tmp_path,
        subblock_configs=[indexed],
        batch_size=1,
        prefill_seq_len=4,
        generation_seq_len=1,
        n_embd=32,
        n_head=8,
        vocab_size=64,
        runtime_stats_enabled=False,
        use_cuda_graph=False,
        weights_dtype=torch.bfloat16,
        activations_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        parameter_inventory=inventory,
    )

    assert result["subblocks"][0]["num_params"] == 123
    assert result["subblocks"][0]["active_params"] == 123
    assert result["non_block"]["num_params"] == 45
    assert result["args"]["parameter_inventory_identity"] == "inventory-v1"


def test_attention_additive_metrics_include_cache_weight_and_phase_flops():
    result = _metrics(
        AttentionConfig(num_query_heads=8, num_kv_heads=2, qk_head_dim=4),
        num_params=100,
        active_params=100,
    )

    assert result["weight_memory_mib"] == pytest.approx(200 / 2**20)
    assert result["kv_cache_bytes_per_token"] == 32
    assert result["state_cache_bytes_per_sequence"] == 0
    # Linear work uses five prefill-phase tokens (prompt + first output),
    # while attention adds causal prompt pairs plus the first decode context.
    assert result["prefill_flops"] == 2 * 100 * 2 * 5 + 4 * 2 * 8 * 4 * (10 + 5)
    assert result["decode_flops"] == 2 * 100 * 2 * 2 + 4 * 2 * 8 * 4 * (6 + 7)
    assert result["additive_metric_provenance"]["prefill_flops"] == "typed_formula"


def test_ffn_additive_flops_use_active_parameters():
    result = _metrics(
        FFNConfig(intermediate_size=16),
        num_params=1000,
        active_params=300,
    )

    assert result["prefill_flops"] == 2 * 300 * 2 * 5
    assert result["decode_flops"] == 2 * 300 * 2 * 2
    assert result["kv_cache_bytes_per_token"] == 0


def test_mamba_additive_metrics_save_state_bytes_per_sequence():
    result = _metrics(
        MambaConfig(
            num_heads=4,
            head_dim=8,
            state_dim=16,
            num_groups=2,
            conv_kernel_size=4,
        ),
        num_params=500,
        active_params=500,
    )

    # conv state: (32 + 2*2*16) * 4; SSM state: 4*8*16; bf16=2 bytes.
    assert result["state_cache_bytes_per_sequence"] == ((96 * 4) + 512) * 2
    assert result["kv_cache_bytes_per_token"] == 0


def test_sparse_vllm_report_exposes_complete_additive_metric_bundle(tmp_path):
    metric_values = {
        "runtime_ms": 8.0,
        "prefill_runtime_ms": 3.0,
        "decode_runtime_ms": 5.0,
        "decode_runtime_ms_per_token": 2.5,
        "weight_memory_mib": 0.25,
        "kv_cache_bytes_per_token": 64,
        "state_cache_bytes_per_sequence": 0,
        "prefill_flops": 4096,
        "decode_flops": 2048,
    }
    rows = []
    for size, scale in ((8, 1), (16, 2)):
        rows.append(
            {
                "subblock_config": FFNConfig(intermediate_size=size).to_dict(),
                **{name: value * scale for name, value in metric_values.items()},
                "additive_metric_provenance": dict.fromkeys(metric_values, "test_formula"),
            }
        )
    stats_path = tmp_path / "sparse_subblock_stats.json"
    stats_path.write_text(
        json.dumps(
            [
                {
                    "args": {
                        "runtime_stats": True,
                        "runtime_granularity": "subblock",
                        "batch_size": 1,
                        "prefill_seq_len": 4,
                        "generation_seq_len": 3,
                    },
                    "subblocks": rows,
                }
            ]
        )
    )

    records = load_vllm_records(stats_path, puzzle_dir=tmp_path)
    summary = generate_vllm_stats_report(
        tmp_path,
        stats_path=stats_path,
        output_dir=tmp_path / "report",
    )

    assert len(records) == 2
    assert records[0].metrics == metric_values
    assert set(summary["metrics"]) == set(metric_values)
    assert (tmp_path / "report" / "vllm_stats_sanity.html").is_file()


def test_legacy_runtime_row_does_not_satisfy_additive_resume_contract():
    config = FFNConfig(intermediate_size=16)
    existing = [
        {
            "args": {
                "batch_size": 1,
                "weights_dtype": str(torch.bfloat16),
                "activations_dtype": str(torch.bfloat16),
                "kv_cache_dtype": str(torch.bfloat16),
                "n_embd": 32,
                "runtime_selection_identity": "sparse-v1",
                "runtime_stats": True,
                "runtime_granularity": "subblock",
            },
            "subblocks": [
                {
                    "subblock_config": config.to_dict(),
                    "parent_layer_index": 0,
                    "runtime_ms": 8.0,
                }
            ],
        }
    ]

    assert not _subblock_stats_already_complete(
        existing,
        [{"subblock_config": config, "parent_layer_indices": [0]}],
        batch_sizes=[1],
        data_types=[(torch.bfloat16, torch.bfloat16, torch.bfloat16)],
        model_hidden_sizes=[32],
        runtime_stats_enabled=True,
        runtime_selection_identity="sparse-v1",
    )


def test_mip_block_aggregation_ignores_metric_provenance_mappings():
    attention = AttentionConfig(num_query_heads=8, num_kv_heads=2, qk_head_dim=4)
    ffn = FFNConfig(intermediate_size=16)
    block = BlockConfig(subblock_configs=(attention, ffn))
    stats = {
        "subblocks": {
            (attention, None): {
                "runtime_ms": 3.0,
                "num_params": 10,
                "additive_metric_provenance": {"runtime_ms": "measured"},
            },
            (ffn, None): {
                "runtime_ms": 5.0,
                "num_params": 20,
                "additive_metric_provenance": {"runtime_ms": "measured"},
            },
        }
    }

    result = _get_block_stats(stats, block)

    assert result["runtime_ms"] == 8.0
    assert result["num_params"] == 30
    assert "additive_metric_provenance" not in result


def test_mip_noop_stats_do_not_coerce_provenance_to_numeric_cost():
    attention = AttentionConfig(num_query_heads=8, num_kv_heads=2, qk_head_dim=4)
    block = BlockConfig(
        subblock_configs=(attention, FFNConfig(intermediate_size=16, no_op=True))
    )
    stats = {
        "subblocks": {
            (attention, None): {
                "runtime_ms": 3.0,
                "num_params": 10,
                "additive_metric_provenance": {"runtime_ms": "measured"},
            }
        }
    }

    result = _get_block_stats(stats, block)

    assert result["runtime_ms"] == 3.0
    assert result["num_params"] == 10
    assert "additive_metric_provenance" not in result
