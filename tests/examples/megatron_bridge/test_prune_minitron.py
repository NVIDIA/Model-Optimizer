# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for prune_minitron.py and distill.py scripts."""

from pathlib import Path

from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.transformers_models import create_tiny_qwen3_dir


def test_prune_minitron(tmp_path: Path, num_gpus):
    teacher_hf_path, teacher_model = create_tiny_qwen3_dir(
        tmp_path, with_tokenizer=True, return_model=True, num_hidden_layers=num_gpus
    )
    teacher_params = sum(p.numel() for p in teacher_model.parameters())

    pruned_model_path = tmp_path / "pruned"
    prune_command_parts = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "prune_minitron.py"],
        hf_model_name_or_path=teacher_hf_path,
        output_hf_path=pruned_model_path,
        pp_size=num_gpus,
        calib_dataset_name="cnn_dailymail",
        calib_num_samples=16,
        seq_length=32,
        prune_target_params=teacher_params * 0.8,
        prune_score_func="mmlu_1pct",
        ss_channel_divisor=4,
        hparams_to_skip="num_attention_heads",
        top_k=1,
    )
    run_example_command(prune_command_parts, example_path="megatron_bridge")
    assert (pruned_model_path / "config.json").exists()
