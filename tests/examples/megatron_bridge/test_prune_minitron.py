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

import pytest
import torch
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.transformers_models import (
    create_tiny_gemma3vl_dir,
    create_tiny_nemotron_h_dir,
    create_tiny_qwen3_5_moe_vl_dir,
    create_tiny_qwen3_dir,
)
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText


@pytest.mark.parametrize(
    ("create_teacher", "megatron_format"),
    [
        # Dense Qwen3 LM, exported back to HF (reloadable to verify the pruned param count).
        pytest.param(
            lambda tmp_path, num_gpus: create_tiny_qwen3_dir(
                tmp_path, with_tokenizer=True, return_model=True, num_hidden_layers=num_gpus
            ),
            False,
            id="qwen3",
        ),
        # NemotronH (nemotron-3-nano): Mamba + attention + MoE hybrid. Saved in Megatron checkpoint
        # format because HF export of a pruned NemotronH requires transformers<5.
        pytest.param(
            lambda tmp_path, num_gpus: create_tiny_nemotron_h_dir(
                tmp_path, with_tokenizer=True, return_model=True
            ),
            True,
            id="nemotron_h",
        ),
    ],
)
def test_prune_minitron(tmp_path, num_gpus, create_teacher, megatron_format):
    teacher_hf_path, teacher_model = create_teacher(tmp_path, num_gpus)
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    prune_target_params = int(teacher_params * 0.8)

    pruned_path = tmp_path / "pruned"
    output_kwarg = (
        {"output_megatron_path": pruned_path}
        if megatron_format
        else {"output_hf_path": pruned_path}
    )
    prune_command_parts = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "prune_minitron.py"],
        hf_model_name_or_path=teacher_hf_path,
        pp_size=num_gpus,
        calib_dataset_name="cnn_dailymail",
        calib_num_samples=8,
        seq_length=16,
        prune_target_params=prune_target_params,
        prune_score_func="mmlu_1pct_bs32",
        ss_channel_divisor=4,
        hparams_to_skip="num_attention_heads",
        top_k=1,
        **output_kwarg,
    )
    run_example_command(prune_command_parts, example_path="megatron_bridge")

    if megatron_format:
        # HF reload of a pruned NemotronH needs transformers<5; just verify the Megatron checkpoint.
        assert (pruned_path / "latest_checkpointed_iteration.txt").exists()
    else:
        assert (pruned_path / "config.json").exists()
        pruned_model = AutoModelForCausalLM.from_pretrained(pruned_path)
        assert sum(p.numel() for p in pruned_model.parameters()) <= prune_target_params


@pytest.mark.parametrize(
    "create_teacher",
    [
        # gemma3vl: sliding/full attention dense LM with a multimodal projector.
        pytest.param(
            lambda tmp_path, num_gpus: create_tiny_gemma3vl_dir(
                tmp_path,
                with_processor=True,
                return_model=True,
                num_hidden_layers=4 * num_gpus,
                intermediate_size=128,
                max_position_embeddings=1024,
            ),
            id="gemma3vl",
        ),
        # qwen3.5-VL MoE: hybrid GatedDeltaNet + gated-attention MoE LM (prunes depth + MoE dims).
        pytest.param(
            lambda tmp_path, num_gpus: create_tiny_qwen3_5_moe_vl_dir(
                tmp_path,
                with_processor=True,
                return_model=True,
                num_hidden_layers=4 * num_gpus,
                max_position_embeddings=1024,
            ),
            id="qwen3_5_moe_vl",
        ),
    ],
)
def test_prune_minitron_vlm(tmp_path, num_gpus, create_teacher):
    # >= 4 layers per PP stage so depth is prunable and stays balanced; max_position_embeddings
    # >= image tokens + text so the image-text calibration sequence is not truncated.
    teacher_hf_path, teacher_model = create_teacher(tmp_path, num_gpus)
    language_model_params = sum(p.numel() for p in teacher_model.model.language_model.parameters())
    prune_target_params = int(language_model_params * 0.7)

    pruned_model_path = tmp_path / "pruned"
    prune_command_parts = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "prune_minitron.py"],
        hf_model_name_or_path=teacher_hf_path,
        output_hf_path=pruned_model_path,
        pp_size=num_gpus,
        calib_dataset_name="scienceqa",  # image-text calibration runs the full VLM forward
        calib_num_samples=8,
        seq_length=1024,
        prune_target_params=prune_target_params,
        prune_score_func="mmlu_1pct_bs32",
        ss_channel_divisor=4,
        # Allow depth pruning (the primary param lever once hidden_size is fixed for VLMs).
        max_depth_pruning=0.6,
        hparams_to_skip="num_attention_heads",
        top_k=1,
    )
    run_example_command(prune_command_parts, example_path="megatron_bridge")
    assert (pruned_model_path / "config.json").exists()

    # from_pretrained (default strict load) verifies the saved weights match the pruned config.
    pruned_model = AutoModelForImageTextToText.from_pretrained(pruned_model_path)
    pruned_lm_params = sum(p.numel() for p in pruned_model.model.language_model.parameters())
    # Language model is pruned to the param target.
    assert pruned_lm_params <= prune_target_params
    # Everything outside the language model (vision tower, projector, lm_head) is byte-identical
    assert hasattr(pruned_model.config, "vision_config")
    teacher_non_lm = {
        n: p for n, p in teacher_model.named_parameters() if "language_model." not in n
    }
    pruned_non_lm = {n: p for n, p in pruned_model.named_parameters() if "language_model." not in n}
    assert pruned_non_lm.keys() == teacher_non_lm.keys()
    for name, expected in teacher_non_lm.items():
        torch.testing.assert_close(
            pruned_non_lm[name].detach().cpu(), expected.detach().cpu(), rtol=0, atol=0
        )
