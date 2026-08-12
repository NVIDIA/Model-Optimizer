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

import pytest

from modelopt.torch.puzzletron.block_config import FFNConfig
from modelopt.torch.puzzletron.mip.profiles import (
    compile_profile_constraints,
    normalize_mip_profiles,
)
from modelopt.torch.puzzletron.mip.run_puzzle import (
    _merge_namespaced_workload_stats,
    filter_subblock_stats_by_args,
)


def test_named_workloads_and_repeated_workload_constraints_are_normalized():
    profiles = normalize_mip_profiles(
        {
            "workloads": {
                "isl-heavy": {"isl": 8192, "osl": 128, "batch_size": 4},
                "osl-heavy": {"isl": 1024, "osl": 8192, "batch_size": 4},
                "serving": {"isl": 8192, "osl": 1024, "concurrency": 4},
            },
            "defaults": {"objectives": "metrics.cosine_embedding_loss_hidden_states"},
            "runs": {
                "multi-workload": {
                    "homogeneous": {"enabled": True, "keep": 3},
                    "constraints": {
                        "params": "75%",
                        "active_params": {"min": "70%", "max": "90%"},
                        "memory": {
                            "at": {
                                "isl-heavy": "80%",
                                "osl-heavy": {"max": "70%"},
                            }
                        },
                        "runtime": {"at": {"serving": {"max": "75%"}}},
                    },
                    "search_space": {
                        "depth": {"range": [0, 4]},
                        "embedding": [2688, 2432],
                        "axes_default": "teacher",
                        "axes": {"n_routed_experts": "all"},
                    },
                }
            },
        },
        available_depths=range(6),
        available_embeddings=(2688, 2560, 2432),
    )

    profile = profiles[0]
    assert profile.profile_id == "multi-workload"
    assert profile.num_homogeneous_solutions == 3
    assert profile.depths == (0, 1, 2, 3, 4)
    assert profile.embedding_widths == (2688, 2432)
    assert profile.axes_default == "teacher"
    assert profile.axis_options == {"n_routed_experts": "all"}
    assert profile.workloads == {
        "isl-heavy": {
            "prefill_seq_len": 8192,
            "generation_seq_len": 128,
            "batch_size": 4,
        },
        "osl-heavy": {
            "prefill_seq_len": 1024,
            "generation_seq_len": 8192,
            "batch_size": 4,
        },
        "serving": {
            "prefill_seq_len": 8192,
            "generation_seq_len": 1024,
            "max_num_seqs": 4,
        },
    }
    assert [(item.metric, item.workload) for item in profile.constraints] == [
        ("params", None),
        ("active_params", None),
        ("memory", "isl-heavy"),
        ("memory", "osl-heavy"),
        ("runtime", "serving"),
    ]


def test_percentage_and_absolute_constraints_compile_against_correct_teachers():
    (profile,) = normalize_mip_profiles(
        {
            "workloads": {
                "isl": {"isl": 8192, "osl": 128, "batch_size": 4},
                "osl": {"isl": 128, "osl": 8192, "batch_size": 4},
            },
            "defaults": {"objectives": "metrics.cosine_embedding_loss_hidden_states"},
            "runs": {
                "budgets": {
                    "constraints": {
                        "params": 22.5e9,
                        "active_params": {"min": "2B", "max": "90%"},
                        "memory": {
                            "at": {
                                "isl": "80%",
                                "osl": {"max": "120GiB"},
                            }
                        },
                    }
                }
            },
        },
        available_depths=(0,),
        available_embeddings=(2688,),
    )

    constraints = compile_profile_constraints(
        profile,
        teacher_totals={
            None: {"num_params": 30e9, "active_params": 3e9},
            "isl": {"memory_mib": 200_000},
            "osl": {"memory_mib": 180_000},
        },
    )

    assert constraints == {
        "stats.num_params": 22.5e9,
        "stats.active_params": (2e9, 2.7e9),
        "stats.memory_mib@isl": 160_000,
        "stats.memory_mib@osl": 120 * 1024,
    }


def test_matrix_expands_profiles_but_search_lists_do_not():
    profiles = normalize_mip_profiles(
        {
            "defaults": {"objectives": "metrics.cosine_embedding_loss_hidden_states"},
            "runs": {
                "grid": {
                    "variants": {
                        "constraint-grid": {
                            "matrix": {
                                "constraints.params": ["70%", "75%"],
                                "constraints.kv_heads": [64, 32],
                            }
                        }
                    },
                    "search_space": {"depth": [0, 1, 4], "embedding": [2688, 2560]},
                }
            }
        },
        available_depths=range(5),
        available_embeddings=(2688, 2560, 2432),
    )

    assert len(profiles) == 4
    assert len({profile.profile_id for profile in profiles}) == 4
    assert all(profile.depths == (0, 1, 4) for profile in profiles)
    assert all(profile.embedding_widths == (2688, 2560) for profile in profiles)
    compiled = [
        compile_profile_constraints(
            profile,
            teacher_totals={None: {"num_params": 100, "num_kv_heads": 128}},
        )
        for profile in profiles
    ]
    assert {
        (constraints["stats.num_params"], constraints["stats.num_kv_heads"])
        for constraints in compiled
    } == {
        (70.0, 64.0),
        (70.0, 32.0),
        (75.0, 64.0),
        (75.0, 32.0),
    }


def test_explicit_total_depth_selector_matches_list_depth_selector():
    def normalize(depth):
        (profile,) = normalize_mip_profiles(
            {
                "defaults": {
                    "objectives": "metrics.cosine_embedding_loss_hidden_states"
                },
                "runs": {
                    "depth": {
                        "constraints": {"params": "75%"},
                        "search_space": {"depth": depth},
                    }
                }
            },
            available_depths=range(5),
            available_embeddings=(2688,),
        )
        return profile

    listed = normalize([2, 3])
    explicit = normalize({"total": [2, 3]})

    assert explicit.depths == listed.depths == (2, 3)
    assert [selection.as_dict() for selection in explicit.depth_selections] == [
        {"total": 2},
        {"total": 3},
    ]
    assert [selection.slug for selection in explicit.depth_selections] == [
        "depth-02",
        "depth-03",
    ]


def test_typed_depth_selectors_form_a_cartesian_product_with_distinct_identities():
    (profile,) = normalize_mip_profiles(
        {
            "defaults": {"objectives": "metrics.cosine_embedding_loss_hidden_states"},
            "runs": {
                "typed": {
                    "constraints": {"params": "75%"},
                    "search_space": {
                        "depth": {"attention": [1, 2], "moe": [1, 2]}
                    },
                }
            }
        },
        available_depths=range(7),
        available_embeddings=(2688,),
        available_depth_counts={"attention": 2, "mamba": 1, "moe": 2},
        depth_granularity="subblock",
    )

    assert [selection.as_dict() for selection in profile.depth_selections] == [
        {"attention": 1, "moe": 1},
        {"attention": 1, "moe": 2},
        {"attention": 2, "moe": 1},
        {"attention": 2, "moe": 2},
    ]
    assert profile.depths == (2, 3, 3, 4)
    assert len({selection.slug for selection in profile.depth_selections}) == 4


def test_typed_depth_selector_validation_is_eager():
    base = {
        "defaults": {"objectives": "metrics.cosine_embedding_loss_hidden_states"},
        "runs": {
            "typed": {
                "constraints": {"params": "75%"},
                "search_space": {"depth": {"attention": 1}},
            }
        }
    }
    kwargs = {
        "available_depths": range(5),
        "available_embeddings": (2688,),
        "available_depth_counts": {"attention": 2, "moe": 1},
    }

    with pytest.raises(ValueError, match="subblock"):
        normalize_mip_profiles(base, depth_granularity="block", **kwargs)

    base["runs"]["typed"]["search_space"]["depth"] = {"ffn": 1}
    with pytest.raises(ValueError, match="unknown.*ffn"):
        normalize_mip_profiles(base, depth_granularity="subblock", **kwargs)

    base["runs"]["typed"]["search_space"]["depth"] = {"attention": 3}
    with pytest.raises(ValueError, match="unavailable.*3"):
        normalize_mip_profiles(base, depth_granularity="subblock", **kwargs)

    base["runs"]["typed"]["search_space"]["depth"] = {
        "total": 2,
        "attention": 1,
    }
    with pytest.raises(ValueError, match="total.*typed"):
        normalize_mip_profiles(base, depth_granularity="subblock", **kwargs)


@pytest.mark.parametrize("value", [-2, -1.5, True])
def test_invalid_homogeneous_keep_is_rejected(value):
    with pytest.raises(ValueError, match="homogeneous"):
        normalize_mip_profiles(
            {
                "defaults": {
                    "objectives": "metrics.cosine_embedding_loss_hidden_states"
                },
                "runs": {
                    "bad": {
                        "homogeneous": {"enabled": True, "keep": value},
                        "constraints": {"params": "75%"},
                    }
                }
            },
            available_depths=(0,),
            available_embeddings=(2688,),
        )


def test_workload_constraint_requires_declared_workload():
    with pytest.raises(ValueError, match="unknown workload"):
        normalize_mip_profiles(
            {
                "workloads": {},
                "defaults": {
                    "objectives": "metrics.cosine_embedding_loss_hidden_states"
                },
                "runs": {
                    "bad": {"constraints": {"runtime": {"at": {"missing": "75%"}}}}
                },
            },
            available_depths=(0,),
            available_embeddings=(2688,),
        )


def test_named_workload_stats_are_kept_as_distinct_additive_fields():
    subblock = FFNConfig(intermediate_size=16)

    def stats(name, memory):
        return {
            "args": {"name": name},
            "non_block": {"memory_mib": memory},
            "subblocks": [
                {
                    "subblock_config_class": "FFNConfig",
                    "subblock_config": subblock.to_dict(),
                    "parent_layer_index": 0,
                    "memory_mib": memory * 2,
                }
            ],
        }

    all_stats = [stats("base", 10), stats("isl", 20), stats("osl", 30)]
    base = filter_subblock_stats_by_args(all_stats, {"name": "base"})
    _merge_namespaced_workload_stats(
        base,
        all_stats,
        {"isl-heavy": {"name": "isl"}, "osl-heavy": {"name": "osl"}},
    )

    assert base["non_block"] == {
        "memory_mib": 10,
        "memory_mib@isl-heavy": 20,
        "memory_mib@osl-heavy": 30,
    }
    assert base["subblocks"][(subblock, 0)] == {
        "parent_layer_index": 0,
        "memory_mib": 20,
        "memory_mib@isl-heavy": 40,
        "memory_mib@osl-heavy": 60,
    }
