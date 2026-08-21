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

"""Tests for scheduler-neutral configs emitted by the Puzzletron setup wizard."""

import pytest

from puzzletron_setup import SetupError
from puzzletron_setup.bundle import (
    _align_model_stage_batches,
    _serving_parallel,
    render_execution,
    render_experiment,
    render_runner,
)
from puzzletron_setup.state import AnswerState
from puzzletron_setup.wizard import (
    _ask_aiperf_config,
    _ask_mesh,
    _ask_mip,
    _default_flow,
    _downstream_evaluation_metric_suggestions,
    _resource_rows,
)


class _NormalMipPrompts:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def begin(self, state, section: str) -> None:
        self.messages.append(f"begin:{section}")

    def checkpoint(self) -> int:
        return len(self.messages)

    def rewind(self, checkpoint: int) -> None:
        self.messages = self.messages[:checkpoint]

    def select(self, message: str, choices, *, default=None, description=None):
        self.messages.append(message)
        return default

    def integer(self, message: str, *, default: int, **kwargs) -> int:
        self.messages.append(message)
        return default

    def confirm(self, message: str, *, default: bool, **kwargs) -> bool:
        self.messages.append(message)
        return False if message == "Add another independent MIP run?" else default

    def text(self, message: str, *, default=None, **kwargs) -> str:
        self.messages.append(message)
        if message == "Embedding widths for this MIP run (all or YAML list):":
            return "[1024, 768]"
        return str(default or "")


class _MeshPrompts:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def integer(self, message: str, *, default: int, **kwargs) -> int:
        self.messages.append(message)
        return default


class _ServingPrompts:
    def __init__(self, values: dict[str, int]) -> None:
        self.values = values
        self.messages: list[str] = []

    def checkpoint(self) -> int:
        return len(self.messages)

    def rewind(self, checkpoint: int) -> None:
        self.messages = self.messages[:checkpoint]

    def integer(self, message: str, *, default: int, **kwargs) -> int:
        self.messages.append(message)
        return self.values.get(message, default)


def test_serving_parallel_treats_vllm_expert_parallelism_as_boolean_mode() -> None:
    topology = {
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "data_parallel_size": 4,
        "expert_parallel_size": 8,
        "gpu_group_size": 8,
    }

    assert _serving_parallel(topology) == {
        "tp": 2,
        "cp": 1,
        "pp": 1,
        "ep": 1,
        "dp_shard": 1,
        "dp_replicate": 4,
        "sequence_parallel": False,
    }

    topology["expert_parallel_size"] = 4
    with pytest.raises(SetupError, match=r"expected 1 or TP \* DP=8"):
        _serving_parallel(topology)


def _nemotron_render_state(*, latent_moe: bool) -> dict:
    axes = {
        "hidden_width": {
            "enabled": True,
            "teacher_value": 2688,
            "values": [2688, 2560],
            "alignment": 128,
        }
    }
    if latent_moe:
        axes["moe_latent_dim"] = {
            "enabled": True,
            "teacher_value": 1024,
            "values": [1024, 896],
            "alignment": 128,
        }
    return {
        "model": {
            "source": "nvidia/NVIDIA-Nemotron-3",
            "resolved_revision": "revision",
            "config": {"moe_latent_size": 1024 if latent_moe else None},
        },
        "inventory": {
            "family": "nemotron3",
            "descriptor": "nemotron_h",
            "family_config": "examples/puzzletron/configs/families/nemotron3/family.yaml",
            "model_type": "nemotron_h",
            "architectures": ["NemotronHForCausalLM"],
            "num_layers": 4,
            "num_sublayers": 4,
            "facts": {"hidden_size": 2688},
        },
        "answers": {
            "data": {
                "source": "/dataset",
                "modality": "text",
                "layout": "fixed",
                "sequence_length": 2048,
            },
            "pruning": {
                "width_importance_samples": 128,
                "replacement_samples": 16,
                "depth_remove": 0,
                "axes": axes,
                "bypass": {"enabled": False},
            },
            "runtime": {
                "vllm_enabled": False,
                "isl": 2048,
                "osl": 256,
                "concurrency": 1,
            },
            "mip": {"runs": {}},
            "post_mip": {"flows": {}},
            "infrastructure": {
                "meshes": {
                    "common": {},
                    "bypass": {},
                    "global_kd": {},
                }
            },
            "output": {"result_root": "/results"},
        },
    }


def test_batch_alignment_preserves_hydra_references() -> None:
    config = {
        "automodel": {
            "parallel": {
                "pp": 2,
                "dp_shard": 2,
                "dp_replicate": 1,
            }
        },
        "micro_batch_size": "${replacement_scoring.micro_batch_size}",
        "nested": {"micro_batch_size": 3},
    }

    _align_model_stage_batches(config)

    assert config["micro_batch_size"] == "${replacement_scoring.micro_batch_size}"
    assert config["nested"]["micro_batch_size"] == 4


def test_nemotron_nano_omits_latent_moe_activation_pass() -> None:
    experiment = render_experiment(
        _nemotron_render_state(latent_moe=False),
        "production",
    )

    pass_names = {item["name"] for item in experiment["pruning"]["activation_passes"]}
    assert "moe_latent" not in pass_names


def test_nemotron_super_keeps_latent_moe_activation_pass() -> None:
    experiment = render_experiment(
        _nemotron_render_state(latent_moe=True),
        "production",
    )

    pass_names = {item["name"] for item in experiment["pruning"]["activation_passes"]}
    assert "moe_latent" in pass_names


def test_first_class_text_acquisition_renders_local_path_and_keeps_tokenization() -> None:
    state = _nemotron_render_state(latent_moe=False)
    state["answers"]["data"].update(
        {
            "source": "/datasets/puzzle-kd",
            "adapter": "puzzle_kd_v2",
            "acquisition": {
                "adapter": "puzzle_kd_v2",
                "source": "nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2",
                "output": "/datasets/puzzle-kd",
                "seed": 408,
                "train_samples": 8192,
                "validation_samples": 1024,
            },
        }
    )

    experiment = render_experiment(state, "production")

    assert experiment["dataset_path"] == "/datasets/puzzle-kd"
    assert experiment["data"]["path"] == "/datasets/puzzle-kd"
    assert experiment["data"]["acquisition"]["adapter"] == "puzzle_kd_v2"
    assert experiment["tokenize_data"]["enabled"] is True


def test_custom_dataset_rendering_does_not_add_acquisition_fields() -> None:
    experiment = render_experiment(_nemotron_render_state(latent_moe=False), "production")

    assert "acquisition" not in experiment["data"]


def test_rendered_data_keeps_controller_and_worker_sequence_length_in_sync() -> None:
    experiment = render_experiment(_nemotron_render_state(latent_moe=False), "production")

    assert experiment["data"]["sequence_length"] == 2048
    assert experiment["data"]["sequence_length"] == experiment["data"]["max_sample_length"]


def test_packed_text_uses_native_automodel_data_instead_of_fixed_token_memmaps() -> None:
    state = _nemotron_render_state(latent_moe=False)
    state["answers"]["data"]["layout"] = "packed_varlen"

    experiment = render_experiment(state, "production")

    assert experiment["tokenize_data"]["enabled"] is False
    assert experiment["tokenize_data"]["caches"] == []
    assert "packed_token_cache_path" not in experiment["depth_importance"]
    assert "packed_token_cache_path" not in experiment["replacement_scoring"]
    assert "packed_token_cache_path" not in experiment["bypass"]["data"]


def test_first_class_vlm_acquisition_disables_text_token_memmaps() -> None:
    state = _nemotron_render_state(latent_moe=False)
    state["answers"]["data"].update(
        {
            "source": "/datasets/nemotron-vlm",
            "adapter": "nemotron_vlm_v2",
            "modality": "multimodal",
            "layout": "packed_varlen",
            "acquisition": {
                "adapter": "nemotron_vlm_v2",
                "source": "nvidia/Nemotron-VLM-Dataset-v2",
                "output": "/datasets/nemotron-vlm",
                "seed": 42,
                "subsets": ["sparsetables"],
                "num_samples": 512,
                "max_shards_per_subset": 1,
            },
        }
    )

    experiment = render_experiment(state, "production")

    assert experiment["data"]["acquisition"]["subsets"] == ["sparsetables"]
    assert experiment["tokenize_data"]["enabled"] is False
    assert experiment["tokenize_data"]["caches"] == []
    assert "packed_token_cache_path" not in experiment["depth_importance"]
    assert "packed_token_cache_path" not in experiment["replacement_scoring"]
    assert "packed_token_cache_path" not in experiment["bypass"]["data"]


def test_hugging_face_subset_selection_is_emitted_in_catalog_order() -> None:
    state = _nemotron_render_state(latent_moe=False)
    state["answers"]["data"].update(
        {
            "subsets": ["small", "large"],
            "subset_revision": "sha",
            "subset_weights": {"small": 0.25, "large": 0.75},
        }
    )

    experiment = render_experiment(state, "production")

    assert experiment["data"]["subsets"] == ["small", "large"]
    assert experiment["data"]["subset_revision"] == "sha"
    assert experiment["data"]["subset_weights"] == {
        "small": 0.25,
        "large": 0.75,
    }


def test_render_experiment_uses_global_runtime_repeat_default() -> None:
    experiment = render_experiment(
        _nemotron_render_state(latent_moe=False),
        "production",
    )

    assert experiment["vllm_stats"]["runtime_stats"]["repeat_block_n_times"] == 4


def test_normal_aiperf_config_asks_shared_cp_and_maps_moe_ep_to_dp() -> None:
    prompts = _ServingPrompts(
        {
            "Serving tensor parallel (TP):": 2,
            "Serving pipeline parallel (PP):": 1,
            "Serving context parallel (CP):": 2,
            "Serving expert parallel (EP):": 4,
        }
    )

    config = _ask_aiperf_config(
        prompts,
        detailed=False,
        moe=True,
        runtime={"isl": 4096, "osl": 1024, "concurrency": 1},
    )

    assert config["topology"] == {
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 2,
        "decode_context_parallel_size": 2,
        "data_parallel_size": 4,
        "expert_parallel_size": 4,
        "distributed_executor_backend": "mp",
        "gpu_group_size": 16,
    }
    assert config["input_tokens"] == 4096
    assert config["output_tokens"] == 1024
    assert config["concurrency"] == [1]
    assert config["benchmark_timeout"] == 900
    assert "AIPerf ISL:" not in prompts.messages


def test_advanced_aiperf_config_asks_split_cp_and_workload() -> None:
    prompts = _ServingPrompts(
        {
            "Serving tensor parallel (TP):": 4,
            "Serving pipeline parallel (PP):": 2,
            "Serving prefill context parallel (CP):": 2,
            "Serving decode context parallel (CP):": 2,
            "AIPerf ISL:": 8192,
            "AIPerf OSL:": 2048,
            "AIPerf concurrency:": 8,
        }
    )

    config = _ask_aiperf_config(
        prompts,
        detailed=True,
        moe=False,
        runtime={"isl": 4096, "osl": 1024, "concurrency": 1},
    )

    assert config["topology"] == {
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 2,
        "prefill_context_parallel_size": 2,
        "decode_context_parallel_size": 2,
        "data_parallel_size": 1,
        "expert_parallel_size": 1,
        "distributed_executor_backend": "mp",
        "gpu_group_size": 16,
    }
    assert config["input_tokens"] == 8192
    assert config["output_tokens"] == 2048
    assert config["concurrency"] == [8]


def test_default_post_mip_flow_uses_fifteen_minute_aiperf_timeout() -> None:
    flow = _default_flow(
        "memory",
        {"objectives": [{"metric": "metrics.lm_loss", "direction": "minimize"}]},
        {"isl": 4096, "osl": 1024, "concurrency": 1},
        {"sequence_length": 4096},
        prefix="",
        include_initial_filter=False,
    )

    assert flow["nodes"]["serving"]["config"]["benchmark_timeout"] == 900
    assert flow["nodes"]["best_lm"]["metric"] == "online_eval.lm_loss"
    assert flow["nodes"]["best"]["metric"] == "final_eval.lm_loss"


def test_render_execution_uses_vllm_runtime_topology() -> None:
    state = {
        "answers": {
            "infrastructure": {
                "gpus_per_node": 8,
                "workers": {"pool": 8, "sharded": 8},
                "meshes": {
                    "common": {
                        "tp": 1,
                        "cp": 1,
                        "pp": 1,
                        "ep": 2,
                        "dp_shard": 2,
                        "dp_replicate": 1,
                    },
                    "bypass": {},
                    "global_kd": {},
                },
            }
        }
    }

    experiment = {
        "vllm_stats": {
            "runtime_stats": {
                "topology": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                    "data_parallel_size": 2,
                    "prefill_context_parallel_size": 1,
                    "decode_context_parallel_size": 1,
                    "gpu_group_size": 4,
                }
            }
        }
    }

    execution = render_execution(state, experiment, "production")

    assert execution["execution"]["stages"]["vllm_stats"]["parallel"] == {
        "tp": 2,
        "cp": 1,
        "pp": 1,
        "ep": 1,
        "dp_shard": 1,
        "dp_replicate": 2,
        "sequence_parallel": False,
    }
    assert execution["execution"]["stages"]["bypass"]["strategy"] == "single"
    assert execution["execution"]["stages"]["bypass"]["instances"] == 1
    for stage_id in ("sort", "build_library"):
        assert execution["execution"]["stages"][stage_id]["parallel"] == {
            "tp": 1,
            "cp": 1,
            "pp": 1,
            "ep": 1,
            "dp_shard": 1,
            "dp_replicate": 1,
            "sequence_parallel": False,
        }


def test_render_execution_uses_common_mesh_for_post_mip_evaluation_only() -> None:
    common = {
        "tp": 1,
        "cp": 1,
        "pp": 1,
        "ep": 2,
        "dp_shard": 2,
        "dp_replicate": 1,
    }
    state = {
        "answers": {
            "infrastructure": {
                "gpus_per_node": 8,
                "workers": {"pool": 8, "sharded": 8},
                "meshes": {"common": common, "bypass": {}, "global_kd": {}},
            }
        }
    }
    experiment = {
        "post_mip": {
            "flows": {
                "run": {
                    "nodes": {
                        "eval": {"type": "evaluation"},
                        "materialized": {"type": "materialize", "input": "eval"},
                        "serve": {
                            "type": "aiperf",
                            "input": "materialized",
                            "config": {
                                "topology": {
                                    "tensor_parallel_size": 2,
                                    "pipeline_parallel_size": 2,
                                    "prefill_context_parallel_size": 1,
                                    "decode_context_parallel_size": 1,
                                    "data_parallel_size": 1,
                                    "expert_parallel_size": 1,
                                    "distributed_executor_backend": "mp",
                                    "gpu_group_size": 4,
                                }
                            },
                        },
                    }
                }
            }
        }
    }

    execution = render_execution(state, experiment, "production")["execution"]["stages"]

    assert execution["post.run.eval"]["parallel"] == {
        **common,
        "sequence_parallel": False,
    }
    assert execution["post.run.serve"]["parallel"] == {
        "tp": 2,
        "cp": 1,
        "pp": 2,
        "ep": 1,
        "dp_shard": 1,
        "dp_replicate": 1,
        "sequence_parallel": False,
    }
    assert execution["post.run.materialized"]["parallel"] == {
        "tp": 1,
        "cp": 1,
        "pp": 1,
        "ep": 1,
        "dp_shard": 1,
        "dp_replicate": 1,
        "sequence_parallel": False,
    }
    assert execution["post.run.materialized"]["instances"] == 1


def test_downstream_evaluation_metric_suggestions_match_runner_keys() -> None:
    assert _downstream_evaluation_metric_suggestions(
        "lmms_eval",
        {"tasks": ["ifeval", "gsm8k", "custom_task"]},
    ) == [
        "lmms_eval.ifeval.prompt_level_strict_acc_none",
        "lmms_eval.gsm8k.exact_match_strict-match",
    ]
    assert _downstream_evaluation_metric_suggestions(
        "lmms_eval",
        {"tasks": "gsm8k,ifeval"},
    ) == [
        "lmms_eval.gsm8k.exact_match_strict-match",
        "lmms_eval.ifeval.prompt_level_strict_acc_none",
    ]


def test_render_execution_uses_vllm_mesh_for_post_mip_downstream_evaluation() -> None:
    state = {
        "answers": {
            "infrastructure": {
                "gpus_per_node": 8,
                "workers": {"pool": 8, "sharded": 8},
                "runner": {"slurm": {}},
                "meshes": {
                    "common": {"tp": 1, "cp": 1, "pp": 1, "dp_shard": 2, "ep": 1},
                    "bypass": {"tp": 1, "cp": 1, "pp": 1, "dp_shard": 1, "ep": 1},
                    "global_kd": {"tp": 1, "cp": 1, "pp": 1, "dp_shard": 1, "ep": 1},
                },
            }
        },
    }
    experiment = {
        "embedding_pruning": {"widths": []},
        "vllm_stats": {"runtime_stats": {"topology": {"gpu_group_size": 1}}},
        "post_mip": {
            "flows": {
                "run": {
                    "nodes": {
                        "materialized": {"type": "materialize"},
                        "lmms_eval": {
                            "type": "downstream_evaluation",
                            "input": "materialized",
                            "config": {
                                "topology": {
                                    "tensor_parallel_size": 4,
                                    "pipeline_parallel_size": 2,
                                    "data_parallel_size": 1,
                                    "prefill_context_parallel_size": 1,
                                    "decode_context_parallel_size": 1,
                                    "enable_expert_parallel": False,
                                    "gpu_group_size": 8,
                                }
                            },
                        },
                    }
                }
            }
        },
    }

    stages = render_execution(state, experiment, "production")["execution"]["stages"]

    assert stages["post.run.lmms_eval"]["strategy"] == "sharded"
    assert stages["post.run.lmms_eval"]["instances"] == 8
    assert stages["post.run.lmms_eval"]["parallel"] == {
        "tp": 4,
        "cp": 1,
        "pp": 2,
        "ep": 1,
        "dp_shard": 1,
        "dp_replicate": 1,
        "sequence_parallel": False,
    }


def test_render_execution_caps_post_mip_workers_at_upstream_top_k() -> None:
    common = {
        "tp": 1,
        "cp": 1,
        "pp": 1,
        "ep": 1,
        "dp_shard": 2,
        "dp_replicate": 1,
    }
    state = {
        "answers": {
            "infrastructure": {
                "gpus_per_node": 8,
                "workers": {"pool": 8, "sharded": 8, "aiperf": 16},
                "meshes": {"common": common, "bypass": common, "global_kd": common},
            }
        }
    }
    experiment = {
        "post_mip": {
            "flows": {
                "run": {
                    "nodes": {
                        "fastest": {
                            "type": "filter",
                            "mode": "top_k",
                            "metric": "serving.request_throughput",
                            "direction": "maximize",
                            "top_k": 4,
                        },
                        "short_kd": {
                            "type": "global_kd",
                            "input": "fastest",
                        },
                        "final_eval": {
                            "type": "evaluation",
                            "input": "short_kd",
                        },
                    }
                }
            }
        }
    }

    execution = render_execution(state, experiment, "production")["execution"]["stages"]

    assert execution["post.run.short_kd"]["instances"] == 4
    assert execution["post.run.final_eval"]["instances"] == 4


@pytest.mark.parametrize(
    ("best_selection_mode", "expected_instances"),
    [("individual_best", 2), ("best_per_concurrency", 6)],
)
def test_render_execution_accounts_for_per_concurrency_top_k_union(
    best_selection_mode: str,
    expected_instances: int,
) -> None:
    common = {
        "tp": 1,
        "cp": 1,
        "pp": 1,
        "ep": 1,
        "dp_shard": 2,
        "dp_replicate": 1,
    }
    state = {
        "answers": {
            "infrastructure": {
                "gpus_per_node": 8,
                "workers": {"pool": 8, "sharded": 8, "aiperf": 16},
                "meshes": {"common": common, "bypass": common, "global_kd": common},
            }
        }
    }
    experiment = {
        "post_mip": {
            "flows": {
                "run": {
                    "nodes": {
                        "serving": {
                            "type": "aiperf",
                            "config": {"concurrency": [1, 2, 4]},
                        },
                        "fastest": {
                            "type": "filter",
                            "input": "serving",
                            "mode": "top_k",
                            "metric": "serving.output_token_throughput",
                            "direction": "maximize",
                            "top_k": 2,
                            "best_selection_mode": best_selection_mode,
                        },
                        "short_kd": {
                            "type": "global_kd",
                            "input": "fastest",
                        },
                    }
                }
            }
        }
    }

    execution = render_execution(state, experiment, "production")["execution"]["stages"]

    assert execution["post.run.short_kd"]["instances"] == expected_instances


def test_render_execution_ignores_legacy_aiperf_worker_override() -> None:
    state = {
        "answers": {
            "infrastructure": {
                "gpus_per_node": 8,
                "workers": {"pool": 4, "sharded": 4, "aiperf": 16},
                "meshes": {
                    "common": {},
                    "bypass": {},
                    "global_kd": {},
                },
            }
        }
    }
    experiment = {
        "post_mip": {
            "flows": {
                "run": {
                    "nodes": {
                        "serving": {
                            "type": "aiperf",
                            "config": {
                                "topology": {
                                    "tensor_parallel_size": 1,
                                    "pipeline_parallel_size": 1,
                                    "prefill_context_parallel_size": 1,
                                    "decode_context_parallel_size": 1,
                                    "data_parallel_size": 2,
                                    "expert_parallel_size": 2,
                                    "gpu_group_size": 2,
                                }
                            },
                        }
                    }
                }
            }
        }
    }

    stages = render_execution(state, experiment, "production")["execution"]["stages"]

    assert stages["post.run.serving"]["instances"] == 4


def test_render_execution_uses_cpu_partition_for_io_bound_stages() -> None:
    state = {
        "answers": {
            "infrastructure": {
                "runner": {
                    "kind": "slurm",
                    "slurm": {
                        "account": "acct",
                        "partition_batch": "batch",
                        "partition_cpu": "cpu",
                    },
                },
                "execution_contract": {"repository": "/repo", "venv": "/repo/.venv"},
                "gpus_per_node": 8,
                "workers": {"pool": 8, "sharded": 8},
                "meshes": {"common": {}, "bypass": {}, "global_kd": {}},
            }
        }
    }
    experiment = {
        "post_mip": {
            "flows": {
                "run": {
                    "nodes": {
                        "filter": {"type": "filter"},
                        "materialized": {"type": "materialize", "input": "filter"},
                        "eval": {"type": "evaluation", "input": "materialized"},
                        "serving": {"type": "aiperf", "input": "materialized"},
                        "short_kd": {"type": "global_kd", "input": "materialized"},
                    }
                }
            }
        }
    }

    runner = render_runner(state, "production")
    stages = render_execution(state, experiment, "production")["execution"]["stages"]

    assert runner["runner"]["slurm"]["partition_cpu"] == "cpu"
    for stage_id in (
        "convert",
        "tokenize_data",
        "build_library",
        "mip",
        "post.run.filter",
        "post.run.materialized",
    ):
        assert stages[stage_id]["resource"] == "cpu"
        assert stages[stage_id]["partition"] == "cpu"
        assert stages[stage_id]["instances"] == 1
    assert "resource" not in stages["sort"]
    assert "partition" not in stages["sort"]
    assert stages["sort"]["strategy"] == "single"
    assert stages["sort"]["parallel"] == {
        "tp": 1,
        "cp": 1,
        "pp": 1,
        "ep": 1,
        "dp_shard": 1,
        "dp_replicate": 1,
        "sequence_parallel": False,
    }
    assert stages["post.run.eval"]["instances"] == 8
    assert stages["post.run.serving"]["instances"] == 8
    assert stages["post.run.short_kd"]["instances"] == 8


def test_render_execution_falls_back_to_one_gpu_for_cpu_stages() -> None:
    state = {
        "answers": {
            "infrastructure": {
                "runner": {
                    "kind": "slurm",
                    "slurm": {
                        "account": "acct",
                        "partition_batch": "batch",
                        "partition_cpu": None,
                    },
                },
                "gpus_per_node": 8,
                "workers": {"pool": 8, "sharded": 8},
                "meshes": {"common": {}, "bypass": {}, "global_kd": {}},
            }
        }
    }
    experiment = {
        "post_mip": {
            "flows": {
                "run": {
                    "nodes": {
                        "materialized": {"type": "materialize"},
                    }
                }
            }
        }
    }

    stages = render_execution(state, experiment, "production")["execution"]["stages"]

    assert stages["mip"].get("resource", "gpu") == "gpu"
    assert stages["post.run.materialized"].get("resource", "gpu") == "gpu"
    assert stages["post.run.materialized"]["instances"] == 1


def test_render_experiment_defaults_to_single_gpu_subblock_vllm() -> None:
    state = {
        "model": {"source": "Qwen/test", "resolved_revision": "revision"},
        "inventory": {
            "family": "qwen3_5",
            "descriptor": "qwen3_5",
            "family_config": "examples/puzzletron/configs/families/qwen3_5/family.yaml",
            "num_layers": 4,
            "num_sublayers": 8,
        },
        "answers": {
            "data": {
                "source": "/dataset",
                "modality": "text",
                "layout": "fixed",
                "sequence_length": 2048,
            },
            "pruning": {
                "width_importance_samples": 128,
                "replacement_samples": 16,
                "depth_remove": 0,
                "axes": {
                    "hidden_width": {
                        "enabled": True,
                        "teacher_value": 1024,
                        "values": [1024, 768],
                        "alignment": 256,
                    },
                    "ffn_intermediate": {
                        "enabled": True,
                        "teacher_value": 3584,
                        "values": [3584, 3072],
                        "alignment": 256,
                    },
                },
                "bypass": {"enabled": True, "batch_size": 3},
            },
            "runtime": {
                "vllm_enabled": True,
                "isl": 2048,
                "osl": 256,
                "concurrency": 4,
            },
            "mip": {"runs": {}},
            "post_mip": {
                "flows": {
                    "memory": {
                        "source": {"run": "memory"},
                        "nodes": {
                            "serving": {
                                "type": "aiperf",
                                "config": {
                                    "concurrency": [1],
                                    "topology": {
                                        "tensor_parallel_size": 4,
                                        "pipeline_parallel_size": 1,
                                        "prefill_context_parallel_size": 2,
                                        "decode_context_parallel_size": 2,
                                        "data_parallel_size": 1,
                                        "expert_parallel_size": 1,
                                        "distributed_executor_backend": "mp",
                                        "gpu_group_size": 8,
                                    },
                                },
                            },
                            "short_kd": {
                                "type": "global_kd",
                                "input": "serving",
                                "config": {"local_batch_size": 1},
                            },
                        },
                    }
                }
            },
            "infrastructure": {
                "meshes": {
                    "common": {"tp": 2, "pp": 2, "dp_shard": 2, "ep": 1},
                    "bypass": {"pp": 1, "dp_shard": 2, "dp_replicate": 1},
                    "global_kd": {
                        "pp": 2,
                        "dp_shard": 2,
                        "dp_replicate": 2,
                    },
                }
            },
            "output": {"result_root": "/results"},
        },
    }

    experiment = render_experiment(state, "production")
    runtime = experiment["vllm_stats"]["runtime_stats"]
    serving = experiment["post_mip"]["flows"]["memory"]["nodes"]["serving"]["config"]

    assert runtime["granularity"] == "subblock"
    assert experiment["embedding_pruning"]["widths"] == [1024, 768]
    assert experiment["vllm_stats"]["model_hidden_sizes"] == [1024, 768]
    assert experiment["search_space"]["axes"]["hidden_width"]["values"] == [1024, 768]
    assert experiment["search_space"]["axes"]["ffn_intermediate"]["values"] == [3584, 3072]
    assert {
        key: runtime["topology"][key]
        for key in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
            "gpu_group_size",
        )
    } == {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "gpu_group_size": 1,
    }
    assert serving["topology"] == {
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 2,
        "decode_context_parallel_size": 2,
        "data_parallel_size": 1,
        "expert_parallel_size": 1,
        "distributed_executor_backend": "mp",
        "gpu_group_size": 8,
    }
    assert serving["topology"] != runtime["topology"]
    assert experiment["data"]["calibration"]["micro_batch_size"] == 4
    assert experiment["data"]["replacement_scoring"]["micro_batch_size"] == 4
    assert experiment["pruning"]["micro_batch_size"] == 4
    assert experiment["sort_sanity"]["micro_batch_size"] == 4
    assert experiment["depth_importance"]["micro_batch_size"] == 4
    assert experiment["replacement_scoring"]["micro_batch_size"] == 4
    assert experiment["bypass"]["training"]["micro_batch_size"] == 4
    assert experiment["bypass"]["training"]["val_micro_batch_size"] == 2
    assert experiment["bypass"]["iter_num"] == 1
    assert experiment["bypass"]["step_num"] == 1
    assert experiment["bypass"]["training"]["training_tokens"] == 4096 * 2048
    assert experiment["global_distillation"]["local_batch_size"] == 8
    assert (
        experiment["post_mip"]["flows"]["memory"]["nodes"]["short_kd"]["config"]["local_batch_size"]
        == 8
    )


def test_normal_mip_flow_asks_for_embedding_widths(tmp_path) -> None:
    state = AnswerState.start(tmp_path / "campaign", detailed=False)
    state.record_many(
        "runtime",
        {"vllm_enabled": False, "workload_id": "serving-default"},
    )
    prompts = _NormalMipPrompts()

    _ask_mip(prompts, state)

    run = next(iter(state.section("mip")["runs"].values()))
    assert "Embedding widths for this MIP run (all or YAML list):" in prompts.messages
    assert run["search_space"]["embedding"] == [1024, 768]
    assert run["solver"]["num_solutions"] == 3
    assert run["homogeneous"] == {
        "enabled": True,
        "keep": 8,
        "rank_by": "objective",
    }


def test_resource_summary_separates_serving_and_evaluation_meshes(tmp_path) -> None:
    state = AnswerState.start(tmp_path / "campaign", detailed=False)
    state.record_many(
        "post_mip",
        {
            "flows": {
                "run": {
                    "nodes": {
                        "serving": {
                            "type": "aiperf",
                            "config": {
                                "topology": {
                                    "tensor_parallel_size": 2,
                                    "pipeline_parallel_size": 2,
                                    "prefill_context_parallel_size": 1,
                                    "decode_context_parallel_size": 1,
                                    "data_parallel_size": 1,
                                    "expert_parallel_size": 1,
                                    "gpu_group_size": 4,
                                }
                            },
                        }
                    }
                }
            }
        },
    )

    rows = _resource_rows(
        state,
        common={"tp": 1, "cp": 1, "pp": 1, "dp_shard": 2, "dp_replicate": 1, "ep": 2},
        bypass={"tp": 1, "cp": 1, "pp": 1, "dp_shard": 2, "dp_replicate": 1, "ep": 2},
        global_kd={
            "tp": 1,
            "cp": 1,
            "pp": 1,
            "dp_shard": 2,
            "dp_replicate": 1,
            "ep": 2,
        },
        gpus_per_node=8,
        workers={"pool": 8, "sharded": 8},
    )

    serving = next(row for row in rows if row["stage"] == "AIPerf")
    assert serving == {
        "stage": "AIPerf",
        "instances": 8,
        "gpus_per_instance": 4,
        "nodes": 4,
    }
    evaluation = next(row for row in rows if row["stage"] == "evaluation")
    assert evaluation == {
        "stage": "evaluation",
        "instances": 8,
        "gpus_per_instance": 2,
        "nodes": 2,
    }


def test_resource_summary_caps_global_kd_at_upstream_top_k(tmp_path) -> None:
    state = AnswerState.start(tmp_path / "campaign", detailed=False)
    state.record_many(
        "post_mip",
        {
            "flows": {
                "run": {
                    "nodes": {
                        "fastest": {
                            "type": "filter",
                            "mode": "top_k",
                            "metric": "serving.request_throughput",
                            "direction": "maximize",
                            "top_k": 4,
                        },
                        "short_kd": {"type": "global_kd", "input": "fastest"},
                    }
                }
            }
        },
    )

    rows = _resource_rows(
        state,
        common={"tp": 1, "cp": 1, "pp": 1, "dp_shard": 2, "dp_replicate": 1, "ep": 1},
        bypass={"tp": 1, "cp": 1, "pp": 1, "dp_shard": 2, "dp_replicate": 1, "ep": 1},
        global_kd={
            "tp": 1,
            "cp": 1,
            "pp": 1,
            "dp_shard": 2,
            "dp_replicate": 1,
            "ep": 1,
        },
        gpus_per_node=8,
        workers={"pool": 8, "sharded": 8, "aiperf": 16},
    )

    global_kd = next(row for row in rows if row["stage"] == "global KD")
    assert global_kd == {
        "stage": "global KD",
        "instances": 4,
        "gpus_per_instance": 2,
        "nodes": 1,
    }


def test_dense_mesh_explains_that_ep_is_fixed(capsys) -> None:
    prompts = _MeshPrompts()

    mesh = _ask_mesh(prompts, "Common", moe=False)

    assert mesh["ep"] == 1
    assert not any("Expert parallel" in message for message in prompts.messages)
    assert "Expert parallel (EP): 1 (not applicable to dense models)." in capsys.readouterr().out
