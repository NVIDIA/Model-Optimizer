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

"""CPU contracts for the Qwen 3.5 0.8B model example."""

from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = REPOSITORY_ROOT / "examples/puzzletron/configs"
MODEL_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/model.yaml"
)
ADVANCED_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/advanced.yaml"
)


def test_qwen3p5_0p8b_model_identity_and_geometry_are_pinned() -> None:
    model = yaml.safe_load(MODEL_PATH.read_text())

    assert model["input_hf_model_path"] == "Qwen/Qwen3.5-0.8B"
    assert model["model_info"] == {
        "hf_repo": model["input_hf_model_path"],
        "hf_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "num_hidden_layers": 24,
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "tie_word_embeddings": True,
        "max_position_embeddings": 262144,
        "mtp_num_hidden_layers": 1,
        "layer_counts": {"linear_attention": 18, "full_attention": 6},
        "mamba": {
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        },
    }


def test_qwen3p5_0p8b_default_search_matches_tracked_runtime_campaign() -> None:
    model = yaml.safe_load(MODEL_PATH.read_text())

    assert model["pruning"] == {"intermediate_size_list": [3072, 2048]}
    assert model["search_space"]["axes"] == {
        "ffn_intermediate": {
            "enabled": True,
            "teacher_value": 3584,
            "values": [3072, 2048],
        }
    }


def test_qwen3p5_0p8b_advanced_search_keeps_broad_domains_explicit() -> None:
    advanced = yaml.safe_load(ADVANCED_PATH.read_text())
    axes = advanced["search_space"]["axes"]

    expected_enabled_domains = {
        "hidden_width": (1024, [768]),
        "kv_groups": (2, [1]),
        "q_heads_per_group": (4, [2]),
        "ffn_intermediate": (3584, [3072, 2560, 2048, 1792, 1536]),
        "gdn_key_groups": (16, [12, 8]),
        "gdn_value_head_dim": (128, [96]),
    }
    enabled_domains = {
        axis_id: (axis["teacher_value"], axis["values"])
        for axis_id, axis in axes.items()
        if axis["enabled"]
    }

    assert advanced["pruning"] == {
        "intermediate_size_list": [3072, 2560, 2048, 1792, 1536],
        "attn_heads_list": [[2, 1], [4, 1], [4, 2], [8, 2]],
    }
    assert enabled_domains == expected_enabled_domains
    assert {
        axis_id: axes[axis_id] for axis_id in ("gdn_value_heads_per_group", "gdn_key_head_dim")
    } == {
        "gdn_value_heads_per_group": {
            "enabled": False,
            "teacher_value": 1,
            "values": [],
        },
        "gdn_key_head_dim": {
            "enabled": False,
            "teacher_value": 128,
            "values": [],
        },
    }
    assert set(axes) == {
        *expected_enabled_domains,
        "gdn_value_heads_per_group",
        "gdn_key_head_dim",
    }


def test_qwen3p5_0p8b_advanced_search_composes_the_pinned_model() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        config = compose(config_name="families/qwen3_5/qwen3p5_0p8b/advanced")
    config = OmegaConf.to_container(config, resolve=True)

    assert config["input_hf_model_path"] == "Qwen/Qwen3.5-0.8B"
    assert config["model_info"]["hf_revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    assert config["model"]["revision"] == config["model_info"]["hf_revision"]
    assert config["search_space"]["axes"]["ffn_intermediate"]["values"] == [
        3072,
        2560,
        2048,
        1792,
        1536,
    ]
