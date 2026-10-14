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

"""Expert-parallel loading and the all-to-all expert transport on real GPUs.

``load_and_prepare_ep`` shards an MoE's routed experts over the ranks, and the ``torch_a2a`` experts
forward sends each token to the rank that holds its expert. Every rank runs its own batch through the
sharded model, and the result must match the whole model run on that batch alone.
"""

import pytest
import torch
from packaging.version import Version

transformers = pytest.importorskip("transformers")
if Version(transformers.__version__) < Version("5.7"):
    pytest.skip("qwen3_moe's base_model_ep_plan ships in transformers 5.7", allow_module_level=True)

from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3MoeConfig

from modelopt.torch.utils.distributed import DistributionPlan, distribution_plan
from modelopt.torch.utils.moe import load_and_prepare_ep

NUM_EXPERTS = 8


def _save_tiny_qwen3_moe(ckpt_dir):
    config = Qwen3MoeConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).to(torch.bfloat16).save_pretrained(ckpt_dir)


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _sharded_model_matches_the_whole_one(rank, size, ckpt_dir, ep_degree):
    device = torch.device(f"cuda:{rank}")
    config = AutoConfig.from_pretrained(ckpt_dir)  # carries qwen3_moe's own base_model_ep_plan
    plan = DistributionPlan(size, ep_size=ep_degree)
    model = load_and_prepare_ep(ckpt_dir, plan, hf_config=config).eval()
    whole = AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=torch.bfloat16).to(device).eval()

    # Each rank holds its contiguous share of every layer's experts within its EP group, whose
    # ranks are consecutive.
    per_rank = NUM_EXPERTS // ep_degree
    ep_rank = rank % ep_degree
    owned = slice(ep_rank * per_rank, (ep_rank + 1) * per_rank)
    for layer, whole_layer in zip(model.model.layers, whole.model.layers):
        for name in ("gate_up_proj", "down_proj"):
            local = _local(getattr(layer.mlp.experts, name))
            assert local.shape[0] == per_rank
            torch.testing.assert_close(local, getattr(whole_layer.mlp.experts, name)[owned])
    # The load recorded what it could not place, for the export to carry: here, nothing. And the
    # plan it loaded by, on every module.
    assert model._modelopt_unplaced_source_keys == []
    assert model._modelopt_source_checkpoint == ckpt_dir
    assert all(distribution_plan(module) is plan for module in model.modules())

    # Every rank calibrates on different tokens, so each sends its own batch through the experts.
    generator = torch.Generator().manual_seed(rank)
    input_ids = torch.randint(0, 128, (2, 16), generator=generator).to(device)
    with torch.no_grad():
        sharded = model(input_ids).logits
        reference = whole(input_ids).logits
    torch.testing.assert_close(sharded.float(), reference.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("ep_share", [1, 2], ids=["ep-world", "ep-half"])
def test_expert_parallel_model_matches_the_whole_model(
    need_2_gpus, dist_workers, tmp_path, ep_share
):
    """With the experts sharded over every rank, or over half of them and replicated across the halves."""
    ep_degree = dist_workers.world_size // ep_share
    if ep_degree < 2 or dist_workers.world_size % ep_share or NUM_EXPERTS % ep_degree:
        pytest.skip(f"{NUM_EXPERTS} experts do not shard over EP {ep_degree} on these GPUs")
    _save_tiny_qwen3_moe(tmp_path)
    dist_workers.run(_sharded_model_matches_the_whole_one, str(tmp_path), ep_degree)
