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

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/model.yaml"
)


def test_qwen3p5_0p8b_model_identity_and_geometry_are_pinned() -> None:
    model = yaml.safe_load(MODEL_PATH.read_text())
    info = model["model_info"]

    assert model["input_hf_model_path"] == "Qwen/Qwen3.5-0.8B"
    assert info["hf_repo"] == model["input_hf_model_path"]
    assert info["hf_revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    assert info["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert info["num_hidden_layers"] == 24
    assert info["layer_counts"] == {"linear_attention": 18, "full_attention": 6}
    assert info["hidden_size"] == 1024
    assert info["intermediate_size"] == 3584
    assert info["num_attention_heads"] == 8
    assert info["num_key_value_heads"] == 2


def test_qwen3p5_0p8b_axis_domains_keep_teacher_and_reduced_values_distinct() -> None:
    model = yaml.safe_load(MODEL_PATH.read_text())
    axes = model["search_space"]["axes"]

    assert axes["hidden_width"] == {
        "enabled": True,
        "teacher_value": 1024,
        "values": [768],
    }
    assert axes["kv_groups"]["teacher_value"] == 2
    assert axes["kv_groups"]["values"] == [1]
    assert axes["q_heads_per_group"]["teacher_value"] == 4
    assert axes["q_heads_per_group"]["values"] == [2]
    assert axes["gdn_key_groups"]["teacher_value"] == 16
    assert axes["gdn_key_groups"]["values"] == [12, 8]
    assert axes["gdn_value_heads_per_group"]["enabled"] is False
