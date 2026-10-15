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

"""``load_pretrained`` in a distributed run, on real GPUs.

Left to decide, it shards an expert-parallel MoE's experts, FSDP2-shards a dense model when asked to,
and otherwise gives every rank the whole model on its own GPU. Whichever load runs, every rank ends
up with the checkpoint's weights and the load's record of what it could not place.
"""

import os

import pytest
import torch
from packaging.version import Version

transformers = pytest.importorskip("transformers")

from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3MoeConfig

from modelopt.torch.utils.distributed import DistributionPlan, distribution_plan
from modelopt.torch.utils.plugins.model_load_utils import load_pretrained

NUM_EXPERTS = 8
_TINY = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 128,
    "max_position_embeddings": 64,
}


def _save(config, ckpt_dir):
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).to(torch.bfloat16).save_pretrained(ckpt_dir)


def _full(tensor):
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _load_and_check(rank, size, ckpt_dir, knobs, expected):
    """Load on every rank of the run with the plan left to decide, and check what each rank holds."""
    device = torch.device(f"cuda:{rank}")
    os.environ.update(knobs)
    try:
        model = load_pretrained(ckpt_dir)
        whole = AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=torch.bfloat16)

        params = dict(model.named_parameters())
        whole_params = dict(whole.named_parameters())
        assert params.keys() == whole_params.keys()
        sharded = {name for name, param in params.items() if isinstance(param, DTensor)}
        if expected == "replicated":
            assert not sharded
            assert all(param.device == device for param in params.values())
        elif expected == "fsdp2":
            assert {name for name in params if ".layers." in name} <= sharded
        else:
            # Each rank holds its contiguous share of the routed experts, and the rest whole.
            experts = {name for name in params if ".mlp.experts." in name}
            assert experts and not sharded - experts
            per_rank = NUM_EXPERTS // size
            for name in experts:
                local = params.pop(name)
                local = local.to_local() if isinstance(local, DTensor) else local
                assert local.shape[0] == per_rank
                owned = whole_params[name][rank * per_rank : (rank + 1) * per_rank]
                torch.testing.assert_close(local.cpu(), owned)
        for name, param in params.items():  # the same order on every rank: full_tensor gathers
            torch.testing.assert_close(_full(param).cpu(), whole_params[name])
        assert model._modelopt_unplaced_source_keys == []
        assert model._modelopt_source_checkpoint == ckpt_dir
        dense = "replicate" if expected == "replicated" else "fsdp"
        assert distribution_plan(model) == DistributionPlan(size, dense=dense)
    finally:
        for knob in knobs:
            del os.environ[knob]


def test_a_dense_model_loads_whole_on_every_rank(need_2_gpus, dist_workers, tmp_path):
    _save(LlamaConfig(**_TINY), tmp_path)
    dist_workers.run(_load_and_check, str(tmp_path), {}, "replicated")


def test_a_dense_model_asked_to_shard_loads_fsdp2_sharded(need_2_gpus, dist_workers, tmp_path):
    _save(LlamaConfig(**_TINY), tmp_path)
    dist_workers.run(_load_and_check, str(tmp_path), {"MODELOPT_FSDP_SIZE": "world"}, "fsdp2")


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.7"),
    reason="qwen3_moe's base_model_ep_plan ships in transformers 5.7",
)
def test_an_expert_parallel_moe_loads_with_its_experts_sharded(need_2_gpus, dist_workers, tmp_path):
    if NUM_EXPERTS % dist_workers.world_size:
        pytest.skip(f"{NUM_EXPERTS} experts do not shard over {dist_workers.world_size} GPUs")
    config = Qwen3MoeConfig(moe_intermediate_size=32, num_experts=NUM_EXPERTS, **_TINY)
    config.num_experts_per_tok = 2
    _save(config, tmp_path)
    dist_workers.run(_load_and_check, str(tmp_path), {}, "expert-parallel")
