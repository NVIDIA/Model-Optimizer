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

"""``load_pretrained`` in a distributed session, on real GPUs.

Left to decide, it FSDP2-shards a dense model when asked to, and otherwise gives every rank the whole
model on its own GPU. Whichever load runs, every rank ends up with the checkpoint's weights and the
load's record of what it could not place.
"""

import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, LlamaConfig

from modelopt.torch.utils.distributed import get_active_mo_session
from modelopt.torch.utils.plugins.model_load_utils import load_pretrained

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
    """Enter a distributed session as torchrun would, load with the plan left to decide, check it."""
    device = torch.device(f"cuda:{rank}")
    session = get_active_mo_session(required=False)
    session.initialize(
        distributed=True,
        rank=rank,
        world_size=size,
        local_rank=rank,
        device=device,
        is_main=rank == 0,
    )
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
        else:
            assert {name for name in params if ".layers." in name} <= sharded
        for name, param in params.items():  # the same order on every rank: full_tensor gathers
            torch.testing.assert_close(_full(param).cpu(), whole_params[name])
        assert model._modelopt_unplaced_source_keys == []
        assert model._modelopt_source_checkpoint == ckpt_dir
    finally:
        for knob in knobs:
            del os.environ[knob]
        session._reset()


def test_a_dense_model_loads_whole_on_every_rank(need_2_gpus, dist_workers, tmp_path):
    _save(LlamaConfig(**_TINY), tmp_path)
    dist_workers.run(_load_and_check, str(tmp_path), {}, "replicated")


def test_a_dense_model_asked_to_shard_loads_fsdp2_sharded(need_2_gpus, dist_workers, tmp_path):
    _save(LlamaConfig(**_TINY), tmp_path)
    dist_workers.run(_load_and_check, str(tmp_path), {"MODELOPT_FSDP_SIZE": "world"}, "fsdp2")
