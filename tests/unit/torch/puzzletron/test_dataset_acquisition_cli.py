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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from examples.puzzletron.materialize_dataset import build_parser


def test_text_cli_parses_bounded_split_sizes(tmp_path):
    args = build_parser().parse_args(
        [
            "puzzle_kd_v2",
            "--output",
            str(tmp_path),
            "--train-samples",
            "100",
            "--validation-samples",
            "20",
        ]
    )

    assert args.adapter == "puzzle_kd_v2"
    assert args.train_samples == 100
    assert args.validation_samples == 20


def test_vlm_cli_parses_subsets_and_shard_bound(tmp_path):
    args = build_parser().parse_args(
        [
            "nemotron_vlm_v2",
            "--output",
            str(tmp_path),
            "--subsets",
            "sparsetables",
            "plotqa_cot",
            "--subset-rows",
            "sparsetables=100",
            "plotqa_cot=300",
            "--num-samples",
            "64",
            "--max-shards-per-subset",
            "2",
        ]
    )

    assert args.adapter == "nemotron_vlm_v2"
    assert args.subsets == ["sparsetables", "plotqa_cot"]
    assert args.subset_rows == [
        ("sparsetables", 100),
        ("plotqa_cot", 300),
    ]
    assert args.num_samples == 64
    assert args.max_shards_per_subset == 2
