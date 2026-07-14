# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json

import pytest


def _observations_module():
    return importlib.import_module(
        "modelopt.torch.puzzletron.bypass_distillation.observations"
    )


def _selection(candidate_id, *, active=75, teacher=100):
    return {
        "step": 1,
        "hidden_width": 4096,
        "ple_width": None,
        "layers": [
            {
                "layer_idx": 0,
                "candidate_id": candidate_id,
                "block_config": {
                    "subblock_configs": [
                        {"kind": "attention", "name": "attention", "num_query_heads": 8},
                        {"kind": "ffn", "name": "ffn", "intermediate_size": 16},
                    ]
                },
                "changed_axes": {"ffn_intermediate": 16},
                "parameter_count": active,
                "teacher_parameter_count": teacher,
                "subblocks": [
                    {
                        "kind": "attention",
                        "name": "attention",
                        "config": {"kind": "attention", "name": "attention", "num_query_heads": 8},
                        "parameter_count": 40,
                        "teacher_parameter_count": 40,
                    },
                    {
                        "kind": "ffn",
                        "name": "ffn",
                        "config": {"kind": "ffn", "name": "ffn", "intermediate_size": 16},
                        "parameter_count": active - 40,
                        "teacher_parameter_count": teacher - 40,
                    },
                ],
            }
        ],
    }


def test_merge_rank_observations_averages_model_parallel_copies_per_lane():
    observations = _observations_module()
    rank_payloads = [
        {
            "dp_lane": 0,
            "selection": _selection("lane-0"),
            "per_layer_loss": {"0": 1.0},
            "per_subblock_loss": {"0:ffn:ffn": 2.0},
        },
        {
            "dp_lane": 0,
            "selection": _selection("lane-0"),
            "per_layer_loss": {"0": 3.0},
            "per_subblock_loss": {"0:ffn:ffn": 4.0},
        },
        {
            "dp_lane": 1,
            "selection": _selection("lane-1", active=50),
            "per_layer_loss": {"0": 5.0},
            "per_subblock_loss": {"0:ffn:ffn": 6.0},
        },
    ]

    block_points, block_catalog = observations.merge_rank_observations(
        rank_payloads,
        step=1,
        granularity="block",
        learning_rate=1.0e-5,
        grad_norm=0.5,
        elapsed_seconds=4.0,
    )
    subblock_points, subblock_catalog = observations.merge_rank_observations(
        rank_payloads,
        step=1,
        granularity="subblock",
        learning_rate=1.0e-5,
        grad_norm=0.5,
        elapsed_seconds=4.0,
    )

    assert [(point.dp_lane, point.loss) for point in block_points] == [(0, 2.0), (1, 5.0)]
    assert [point.parameter_ratio for point in block_points] == [0.75, 0.5]
    assert [(point.dp_lane, point.loss) for point in subblock_points] == [(0, 3.0), (1, 6.0)]
    assert [point.parameter_ratio for point in subblock_points] == [35 / 60, 10 / 60]
    assert len(block_catalog) == 2
    assert len(subblock_catalog) == 2


def test_candidate_catalog_rejects_one_id_with_conflicting_configs():
    observations = _observations_module()
    catalog = observations.CandidateCatalog()
    catalog.register("candidate", {"width": 8})

    with pytest.raises(RuntimeError, match="candidate.*conflicting"):
        catalog.register("candidate", {"width": 4})


def test_parameter_ratio_rejects_zero_teacher_size():
    observations = _observations_module()

    with pytest.raises(ValueError, match="teacher parameter count"):
        observations.normalized_parameter_ratio(0, 0)


def test_observation_writer_recovers_partial_line_and_truncates_after_step(tmp_path):
    observations = _observations_module()
    path = tmp_path / "observations.jsonl"
    path.write_text(
        json.dumps({"step": 1, "observations": []})
        + "\n"
        + json.dumps({"step": 2, "observations": []})
        + "\n"
        + '{"step": 3'
    )
    writer = observations.ObservationWriter(path)

    assert writer.steps() == [1, 2]
    writer.append_step(3, [])
    writer.truncate_after_step(2)

    assert writer.steps() == [1, 2]
    assert path.read_text().endswith("\n")
