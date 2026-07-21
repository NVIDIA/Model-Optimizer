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
"""Tests for export_distilled_megatron_to_hf.py."""

from pathlib import Path

from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.transformers_models import create_tiny_qwen3_dir


def test_export_distilled_megatron_iterations(tmp_path: Path, num_gpus):
    teacher_hf_path = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)
    train_iters = 2
    distill_output_dir = tmp_path / "distill_output"
    selected_exports = tmp_path / "selected_exports"
    all_exports = tmp_path / "all_exports"
    distill_cmd_parts = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "distill.py", "--use_mock_data"],
        student_hf_path=teacher_hf_path,
        teacher_hf_path=teacher_hf_path,
        output_dir=distill_output_dir,
        tp_size=num_gpus,
        pp_size=1,
        seq_length=16,
        mbs=1,
        gbs=4,
        train_iters=train_iters,
        lr_warmup_iters=1,
        eval_interval=1,
        eval_iters=1,
        log_interval=1,
        checkpoint_keep_last=-1,
    )
    run_example_command(distill_cmd_parts, example_path="megatron_bridge")

    checkpoint_root = distill_output_dir / "checkpoints"
    assert (checkpoint_root / "iter_0000001").exists()
    assert (checkpoint_root / f"iter_{train_iters:07d}").exists()

    selected_export_cmd_parts = [
        "torchrun",
        "--nproc_per_node=1",
        "export_distilled_megatron_to_hf.py",
        "--student_hf_path",
        str(teacher_hf_path),
        "--megatron_path",
        str(checkpoint_root),
        "--hf_export_path",
        str(selected_exports),
        "--export_iterations",
        "1",
        "2",
    ]
    run_example_command(selected_export_cmd_parts, example_path="megatron_bridge")

    assert (selected_exports / "iter_0000001/config.json").exists()
    assert (selected_exports / "iter_0000002/config.json").exists()

    all_export_cmd_parts = [
        "torchrun",
        "--nproc_per_node=1",
        "export_distilled_megatron_to_hf.py",
        "--student_hf_path",
        str(teacher_hf_path),
        "--megatron_path",
        str(checkpoint_root),
        "--hf_export_path",
        str(all_exports),
        "--export_iterations",
        "all",
    ]
    run_example_command(all_export_cmd_parts, example_path="megatron_bridge")

    assert (all_exports / "iter_0000001/config.json").exists()
    assert (all_exports / "iter_0000002/config.json").exists()
