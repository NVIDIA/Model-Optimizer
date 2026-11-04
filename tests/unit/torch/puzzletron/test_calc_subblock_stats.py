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

from omegaconf import OmegaConf

from modelopt.torch.puzzletron.subblock_stats.calc_subblock_stats import (
    calculate_subblock_stats_for_puzzle_dir,
)


def test_existing_subblock_stats_are_reused_on_resume(tmp_path):
    stats_file = tmp_path / "subblock_stats.json"
    stats_file.write_text('[{"args": {"batch_size": 1}}]')
    teacher_dir = tmp_path / "ckpts" / "teacher"  # absent: resuming must not reload the teacher

    calculate_subblock_stats_for_puzzle_dir(
        OmegaConf.create({}),
        master_puzzle_dir=tmp_path,
        teacher_dir=teacher_dir,
        descriptor=None,
        model_hidden_sizes=OmegaConf.create([]),
        ffn_hidden_sizes=OmegaConf.create([]),
    )

    assert stats_file.read_text() == '[{"args": {"batch_size": 1}}]'
