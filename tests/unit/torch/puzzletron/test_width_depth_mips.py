import json
from pathlib import Path

import pytest

import examples.puzzletron.run_width_depth_mips as width_depth_mips
from examples.puzzletron.run_width_depth_mips import _replacement_score_paths, _stats_profile
from modelopt.torch.puzzletron.mip.profiles import DepthSelection


def test_depth_stage_config_prefers_canonical_depth_importance_section():
    config = {
        "depth": {"max_subblocks_to_remove": 1, "granularity": "block"},
        "depth_importance": {
            "max_subblocks_to_remove": 5,
            "granularity": "subblock",
        },
    }

    assert width_depth_mips._depth_stage_config(config) == config["depth_importance"]


def test_depth_stage_config_keeps_legacy_depth_fallback():
    config = {"depth": {"max_subblocks_to_remove": 3, "granularity": "subblock"}}

    assert width_depth_mips._depth_stage_config(config) == config["depth"]


@pytest.mark.parametrize(
    ("granularity", "validation_name"),
    [
        ("block", "single_sequence_replacement_solutions--validation"),
        ("subblock", "single_subblock_replacement_solutions--validation"),
    ],
)
def test_replacement_score_paths_follow_configured_granularity(
    tmp_path: Path,
    granularity: str,
    validation_name: str,
) -> None:
    scoring_dir, canonical_path = _replacement_score_paths(tmp_path, granularity)

    assert scoring_dir == tmp_path / validation_name
    assert canonical_path == tmp_path / "single_sequence_replacement_solutions.json"


def test_replacement_score_paths_reject_unknown_granularity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported MIP score_granularity"):
        _replacement_score_paths(tmp_path, "layer")


def test_stats_profile_accepts_equivalent_static_dtype_profiles(tmp_path: Path) -> None:
    profiles = [
        {
            "args": {
                "runtime_stats": False,
                "weights_dtype": dtype,
                "n_embd": 2048,
            },
            "non_block": {"num_params": 11, "memory_mib": memory},
            "subblocks": [
                {"parent_layer_index": 0, "num_params": 13, "memory_mib": memory},
                {"parent_layer_index": 1, "num_params": 17, "memory_mib": memory},
            ],
        }
        for dtype, memory in (("nvfp4", 1.0), ("torch.int8", 2.0))
    ]
    path = tmp_path / "subblock_stats.json"
    path.write_text(json.dumps(profiles))

    assert _stats_profile(path, runtime_stats=False, hidden_width=2048) == profiles[0]


def test_stats_profile_selects_requested_width_before_comparing_inventories(
    tmp_path: Path,
) -> None:
    profiles = [
        {
            "args": {
                "runtime_stats": False,
                "weights_dtype": "nvfp4",
                "n_embd": width,
            },
            "non_block": {"num_params": non_block_params},
            "subblocks": [
                {"parent_layer_index": 0, "num_params": subblock_params},
            ],
        }
        for width, non_block_params, subblock_params in (
            (1792, 17, 19),
            (2048, 23, 29),
        )
    ]
    path = tmp_path / "subblock_stats.json"
    path.write_text(json.dumps(profiles))

    assert _stats_profile(path, runtime_stats=False, hidden_width=2048) == profiles[1]


def test_stats_profile_rejects_conflicting_static_parameter_inventories(
    tmp_path: Path,
) -> None:
    profiles = [
        {
            "args": {
                "runtime_stats": False,
                "weights_dtype": "nvfp4",
                "n_embd": 2048,
            },
            "non_block": {"num_params": 11},
            "subblocks": [{"parent_layer_index": 0, "num_params": 13}],
        },
        {
            "args": {
                "runtime_stats": False,
                "weights_dtype": "torch.int8",
                "n_embd": 2048,
            },
            "non_block": {"num_params": 11},
            "subblocks": [{"parent_layer_index": 0, "num_params": 19}],
        },
    ]
    path = tmp_path / "subblock_stats.json"
    path.write_text(json.dumps(profiles))

    with pytest.raises(RuntimeError, match="conflicting parameter inventories"):
        _stats_profile(path, runtime_stats=False, hidden_width=2048)


def test_teacher_summary_costs_include_named_workload_denominators():
    costs = width_depth_mips._teacher_summary_costs(
        {"stats.num_params": 100.0, "stats.memory_mib": 200.0},
        {
            "serving-8k": {
                "stats.memory_mib": 300.0,
                "stats.runtime_ms": 400.0,
            }
        },
    )

    assert costs == {
        "stats.num_params": 100.0,
        "stats.memory_mib": 200.0,
        "stats.memory_mib@serving-8k": 300.0,
        "stats.runtime_ms@serving-8k": 400.0,
    }


def test_forced_removals_support_total_and_typed_prefixes_in_global_order():
    selected = [
        {"layer_idx": 0, "kind": "attention"},
        {"layer_idx": 1, "kind": "mamba"},
        {"layer_idx": 2, "kind": "moe"},
        {"layer_idx": 3, "kind": "attention"},
        {"layer_idx": 4, "kind": "moe"},
    ]

    assert width_depth_mips._forced_removals_for_depth(
        selected, DepthSelection.total_prefix(3)
    ) == selected[:3]
    assert width_depth_mips._forced_removals_for_depth(
        selected,
        DepthSelection((("attention", 2), ("moe", 1))),
    ) == [selected[0], selected[2], selected[3]]


def test_forced_removals_reject_unavailable_typed_count():
    selected = [{"layer_idx": 0, "kind": "attention"}]

    with pytest.raises(ValueError, match="attention.*2.*1"):
        width_depth_mips._forced_removals_for_depth(
            selected,
            DepthSelection((("attention", 2),)),
        )


def test_completed_scenario_resume_uses_full_depth_selection_identity(
    tmp_path: Path,
) -> None:
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    solution_path = scenario_root / "solutions.json"
    solution_path.write_text("[]")
    solve_identity = "solve-v1"
    manifest = {
        "profile_id": "params-075",
        "hidden_width": 2688,
        "removed_sublayers": 3,
        "constraint_type": "named_profile",
        "status": "infeasible",
        "solve_identity": solve_identity,
        "solution_path": str(solution_path),
        "solution_count": 0,
        "solutions": [],
    }
    (scenario_root / "scenario_manifest.json").write_text(json.dumps(manifest))

    resumed = width_depth_mips._load_completed_scenario(
        scenario_root,
        profile_id="params-075",
        width=2688,
        depth_selection=DepthSelection.total_prefix(3),
        constraint_type="named_profile",
        solve_only=True,
        solve_identity=solve_identity,
    )
    assert resumed == manifest

    manifest["depth_selection"] = {"attention": 2, "moe": 1}
    (scenario_root / "scenario_manifest.json").write_text(json.dumps(manifest))
    matching = DepthSelection((("attention", 2), ("moe", 1)))
    same_total_but_different = DepthSelection((("attention", 1), ("moe", 2)))

    assert (
        width_depth_mips._load_completed_scenario(
            scenario_root,
            profile_id="params-075",
            width=2688,
            depth_selection=matching,
            constraint_type="named_profile",
            solve_only=True,
            solve_identity=solve_identity,
        )
        == manifest
    )
    assert (
        width_depth_mips._load_completed_scenario(
            scenario_root,
            profile_id="params-075",
            width=2688,
            depth_selection=same_total_but_different,
            constraint_type="named_profile",
            solve_only=True,
            solve_identity=solve_identity,
        )
        is None
    )
