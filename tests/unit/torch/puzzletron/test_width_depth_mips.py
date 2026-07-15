import json
from pathlib import Path

import pytest

from examples.puzzletron.run_width_depth_mips import _replacement_score_paths, _stats_profile


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
            "args": {"runtime_stats": False, "weights_dtype": dtype},
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

    assert _stats_profile(path, runtime_stats=False) == profiles[0]


def test_stats_profile_rejects_conflicting_static_parameter_inventories(
    tmp_path: Path,
) -> None:
    profiles = [
        {
            "args": {"runtime_stats": False, "weights_dtype": "nvfp4"},
            "non_block": {"num_params": 11},
            "subblocks": [{"parent_layer_index": 0, "num_params": 13}],
        },
        {
            "args": {"runtime_stats": False, "weights_dtype": "torch.int8"},
            "non_block": {"num_params": 11},
            "subblocks": [{"parent_layer_index": 0, "num_params": 19}],
        },
    ]
    path = tmp_path / "subblock_stats.json"
    path.write_text(json.dumps(profiles))

    with pytest.raises(RuntimeError, match="conflicting parameter inventories"):
        _stats_profile(path, runtime_stats=False)
