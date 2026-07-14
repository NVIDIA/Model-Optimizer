import json
from pathlib import Path

import pytest

from examples.puzzletron.verify_cross_model_libraries import verify_model_library
from modelopt.torch.puzzletron.candidates import build_candidate_library
from modelopt.torch.puzzletron.block_config import BlockConfig


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "model"
    teacher = root / "ckpts" / "elastic_sorted_teacher"
    blocks = [
        {
            "subblock_configs": [
                {
                    "kind": "attention",
                    "name": "attention",
                    "num_query_heads": 8,
                    "num_kv_heads": 2,
                    "no_op": False,
                },
                {
                    "kind": "ffn",
                    "name": "ffn",
                    "intermediate_size": 16,
                    "no_op": False,
                },
            ]
        },
        {
            "subblock_configs": [
                {
                    "kind": "ffn",
                    "name": "ffn",
                    "intermediate_size": 16,
                    "no_op": False,
                }
            ]
        },
    ]
    _write_json(
        teacher / "config.json",
        {"hidden_size": 32, "num_hidden_layers": 2, "block_configs": blocks},
    )
    search_space = {
        "no_op": {"subblocks": [], "whole_block": False, "cartesian": False},
        "axes": {
            "ffn_intermediate": {"enabled": True, "sizes": [8, 16]},
            "kv_groups": {"enabled": True, "sizes": [1, 2]},
            "q_heads_per_group": {"enabled": True, "sizes": [2, 4]},
        },
    }
    expected = build_candidate_library(
        [BlockConfig(**block) for block in blocks],
        search_space=search_space,
        parent_checkpoint_identity="parent",
        include_self=True,
        include_noops=False,
        hidden_width=32,
    )
    entries = [
        {
            "weight_paths": [],
            "parent_layer_indices": [candidate.layer_idx],
            "child_block_configs": [candidate.block_config.to_dict()],
        }
        for candidate in expected
    ]
    _write_json(
        root / "replacement_library.json",
        {
            "version": 2,
            "sorted_teacher_dir": str(teacher),
            "hidden_width": 32,
            "teacher_hidden_width": 32,
            "entries": entries,
        },
    )
    _write_json(
        root / "candidate_library.json",
        {
            "version": 1,
            "parent_checkpoint_identity": "parent",
            "settings_identity": "settings",
            "candidates": [candidate.to_dict() for candidate in expected],
            "metadata": {"num_layers": 2, "hidden_width": 32},
        },
    )
    teacher_keys = {(idx, json.dumps(block, sort_keys=True)) for idx, block in enumerate(blocks)}
    solutions = []
    for entry in entries:
        key = (
            entry["parent_layer_indices"][0],
            json.dumps(entry["child_block_configs"][0], sort_keys=True),
        )
        if key not in teacher_keys:
            solutions.append({"single_sequence_replacement": entry})
    _write_json(root / "single_sequence_replacement_solutions.json", solutions)
    _write_json(root / "subblock_stats.json", [{"args": {"n_embd": 32, "runtime_stats": False}}])
    config = {
        "search_space": search_space,
        "build_replacement_library": {"include_noops": False, "hidden_width": None},
    }
    return root, config


def test_verify_model_library_derives_exact_cartesian_cardinality(tmp_path):
    root, config = _fixture(tmp_path)

    summary = verify_model_library("fixture", root, config, require_runtime=False)

    assert summary["num_layers"] == 2
    assert summary["entries"] == 10
    assert summary["teacher_entries"] == 2
    assert summary["solutions"] == 8
    assert summary["per_layer"] == {"0": 8, "1": 2}


def test_verify_model_library_rejects_duplicate_layer_config(tmp_path):
    root, config = _fixture(tmp_path)
    path = root / "replacement_library.json"
    payload = json.loads(path.read_text())
    payload["entries"].append(payload["entries"][0])
    _write_json(path, payload)

    with pytest.raises(ValueError, match="duplicate replacement"):
        verify_model_library("fixture", root, config, require_runtime=False)


def test_verify_model_library_requires_runtime_measurements_after_vllm(tmp_path):
    root, config = _fixture(tmp_path)

    with pytest.raises(ValueError, match="runtime measurements"):
        verify_model_library("fixture", root, config, require_runtime=True)


def test_verify_model_library_requires_complete_additive_runtime_bundle(tmp_path):
    root, config = _fixture(tmp_path)
    _write_json(
        root / "subblock_stats.json",
        [
            {
                "args": {"runtime_stats": True, "runtime_granularity": "subblock"},
                "subblocks": [{"runtime_ms": 1.0}],
            }
        ],
    )

    with pytest.raises(ValueError, match="additive runtime metrics"):
        verify_model_library("fixture", root, config, require_runtime=True)

    metric_values = {
        "runtime_ms": 8.0,
        "prefill_runtime_ms": 3.0,
        "decode_runtime_ms": 5.0,
        "decode_runtime_ms_per_token": 2.5,
        "weight_memory_mib": 0.25,
        "kv_cache_bytes_per_token": 0,
        "state_cache_bytes_per_sequence": 0,
        "prefill_flops": 4096,
        "decode_flops": 2048,
    }
    _write_json(
        root / "subblock_stats.json",
        [
            {
                "args": {"runtime_stats": True, "runtime_granularity": "subblock"},
                "subblocks": [
                    {
                        **metric_values,
                        "additive_metric_provenance": {
                            name: "test" for name in metric_values
                        },
                    }
                ],
            }
        ],
    )

    summary = verify_model_library("fixture", root, config, require_runtime=True)

    assert summary["runtime_entries"] == 1


def test_verify_model_library_uses_configured_sparse_stats_filename(tmp_path):
    root, config = _fixture(tmp_path)
    sparse_path = root / "sparse_subblock_stats.json"
    (root / "subblock_stats.json").rename(sparse_path)
    config["calc_subblock_stats"] = {
        "subblock_stats_filename": sparse_path.name,
    }

    with pytest.raises(ValueError) as error:
        verify_model_library("fixture", root, config, require_runtime=True)

    assert str(sparse_path) in str(error.value)
