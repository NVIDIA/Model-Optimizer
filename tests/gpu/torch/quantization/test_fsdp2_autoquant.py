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

"""Exercise real gradient AutoQuant on a root-and-decoder FSDP2 model."""

import math
from functools import partial

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from transformers import AutoConfig, AutoModelForCausalLM

import modelopt.torch.quantization as mtq
from modelopt.torch.utils.plugins.model_load_utils import parallel_load_and_prepare_fsdp2

pytestmark = [pytest.mark.usefixtures("need_2_gpus"), pytest.mark.timeout(300)]


def _run_gradient_autoquant(rank, size, checkpoint_dir, search_dir):
    device = torch.device(f"cuda:{rank}")
    model = parallel_load_and_prepare_fsdp2(checkpoint_dir, device, rank, size)
    torch.manual_seed(rank)
    batches = [torch.randint(0, 128, (1, 16), device=device) for _ in range(2)]
    _, state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG"],
        data_loader=batches,
        forward_step=lambda m, x: m(input_ids=x, labels=x, use_cache=False),
        loss_func=lambda output, _: output.loss,
        num_calib_steps=2,
        num_score_steps=2,
        method="gradient",
        checkpoint=search_dir,
    )
    scores = [v for stat in state["candidate_stats"].values() for v in stat["raw_scores"]]
    assert all(math.isfinite(value) and value >= 0 for value in scores)
    assert any(value > 0 for value in scores)
    assert state["best"]["is_satisfied"]
    assert state["best"]["constraints"]["effective_bits"] <= 8.0 + 1e-6
    results = [None] * size
    dist.all_gather_object(
        results, (scores, {k: str(v) for k, v in state["best"]["recipe"].items()})
    )
    assert all(result == results[0] for result in results)


def test_fsdp2_gradient_autoquant(dist_workers, tmp_path):
    checkpoint = create_tiny_llama_dir(tmp_path, vocab_size=128, num_hidden_layers=2)
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    dist_workers.run(
        partial(_run_gradient_autoquant, checkpoint_dir=str(checkpoint), search_dir=str(search_dir))
    )


def test_fsdp2_gradient_qwen_moe_autoquant(dist_workers, tmp_path):
    try:
        AutoConfig.for_model("qwen3_5_moe_text")
    except ValueError:
        pytest.skip("Qwen3.5 requires a recent Transformers")
    config = AutoConfig.for_model(
        "qwen3_5_moe_text",
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "full_attention"],
        vocab_size=128,
        max_position_embeddings=128,
    )
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    checkpoint = tmp_path / "qwen_moe"
    model.save_pretrained(checkpoint)
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    dist_workers.run(
        partial(
            _run_gradient_autoquant,
            checkpoint_dir=str(checkpoint),
            search_dir=str(search_dir),
        )
    )
