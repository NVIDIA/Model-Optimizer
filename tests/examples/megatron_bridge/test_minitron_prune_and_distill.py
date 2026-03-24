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

import subprocess

import pytest
import torch
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.transformers_models import create_tiny_qwen3_dir


class TestMinitronPruneAndDistill:
    @pytest.fixture(autouse=True, scope="class")
    def setup(self, tmp_path_factory):
        cls = type(self)
        cls.tmp_path = tmp_path_factory.mktemp("minitron")
        cls.n_gpus = torch.cuda.device_count()

        teacher_hf_path, teacher_model = create_tiny_qwen3_dir(
            cls.tmp_path, with_tokenizer=True, return_model=True, num_hidden_layers=cls.n_gpus
        )
        cls.teacher_hf_path = teacher_hf_path
        cls.teacher_params = sum(p.numel() for p in teacher_model.parameters())
        cls.pruned_model_path = cls.tmp_path / "pruned"
        cls.distill_output_dir = cls.tmp_path / "distill_output"

    def test_prune(self):
        prune_command_parts = extend_cmd_parts(
            ["torchrun", f"--nproc_per_node={self.n_gpus}", "prune_minitron.py"],
            hf_model_name_or_path=self.teacher_hf_path,
            output_hf_path=self.pruned_model_path,
            pp_size=self.n_gpus,
            calib_dataset_name="cnn_dailymail",
            calib_num_samples=16,
            seq_length=32,
            prune_target_params=self.teacher_params * 0.8,
            prune_score_func="mmlu_1pct",
            ss_channel_divisor=4,
            hparams_to_skip="num_attention_heads",
            top_k=1,
        )
        run_example_command(prune_command_parts, example_path="megatron_bridge")
        assert (self.pruned_model_path / "config.json").exists()

    def test_distill_and_convert(self):
        assert self.pruned_model_path.exists(), "Pruned model not found; test_prune must run first."
        train_iters = 5
        distill_cmd_parts = extend_cmd_parts(
            ["torchrun", f"--nproc_per_node={self.n_gpus}", "distill.py", "--use_mock_data"],
            student_hf_path=self.pruned_model_path,
            teacher_hf_path=self.teacher_hf_path,
            output_dir=self.distill_output_dir,
            tp_size=self.n_gpus,
            seq_length=32,
            mbs=1,
            gbs=4,
            train_iters=train_iters,
            lr_warmup_iters=2,
            eval_interval=5,
            eval_iters=1,
            log_interval=1,
        )
        run_example_command(distill_cmd_parts, example_path="megatron_bridge")

        megatron_ckpt_path = self.distill_output_dir / f"checkpoints/iter_{train_iters:07d}"
        assert megatron_ckpt_path.exists()

        # Convert distilled Megatron checkpoint back to HF format
        distilled_hf_path = self.tmp_path / "distilled_hf"
        subprocess.run(
            [
                "python",
                "/opt/Megatron-Bridge/examples/conversion/convert_checkpoints.py",
                "export",
                "--hf-model",
                str(self.pruned_model_path),
                "--megatron-path",
                str(megatron_ckpt_path),
                "--hf-path",
                str(distilled_hf_path),
            ],
            check=True,
        )
        assert (distilled_hf_path / "config.json").exists()
