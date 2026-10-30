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
sharded model, and the result must match the whole model run on that batch alone. A checkpoint whose
experts are compressed-tensors compressed loads the same way, each rank decompressing its own.
"""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import bounded_collectives
from packaging.version import Version
from safetensors.torch import load_file, save_file

transformers = pytest.importorskip("transformers")
if Version(transformers.__version__) < Version("5.7"):
    pytest.skip("qwen3_moe's base_model_ep_plan ships in transformers 5.7", allow_module_level=True)

import transformers.core_model_loading as cml
import transformers.modeling_utils as mu
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3MoeConfig

from modelopt.torch.models import _onfly_decompress
from modelopt.torch.utils.distributed import (
    DistributionPlan,
    distribution_plan,
    set_distribution_plan,
)
from modelopt.torch.utils.moe import expert_parallel_mesh
from modelopt.torch.utils.plugins.model_load_utils import load_and_prepare_ep

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
    # The loader swaps the plan on its own copy: the caller's config keeps the checkpoint's.
    assert config.base_model_ep_plan == AutoConfig.from_pretrained(ckpt_dir).base_model_ep_plan
    # The loader records no plan; loading through it directly, the caller records one.
    with pytest.raises(RuntimeError, match="no distribution plan recorded"):
        distribution_plan(model)
    set_distribution_plan(model, plan)
    # Calibration re-installs the all-to-all over the mesh the loader keeps on the model.
    assert expert_parallel_mesh(model).size() == ep_degree
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
    # The load recorded what it could not place, for the export to carry: here, nothing.
    assert model._modelopt_unplaced_source_keys == []
    assert model._modelopt_source_checkpoint == ckpt_dir

    # Every rank calibrates on different tokens, so each sends its own batch through the experts.
    generator = torch.Generator().manual_seed(rank)
    input_ids = torch.randint(0, 128, (2, 16), generator=generator).to(device)
    with torch.no_grad():
        sharded = model(input_ids).logits
        reference = whole(input_ids).logits
    torch.testing.assert_close(sharded.float(), reference.float(), atol=2e-2, rtol=2e-2)


# The compressions the published Kimi checkpoints use, INT4 (Kimi-K2.5) and MXFP4 (Kimi-K3), through
# compressed-tensors (quant_method "compressed-tensors"), which the on-the-fly decompressor reads; not
# transformers' own MXFP4 (quant_method "mxfp4"). The experts draw values each format holds exactly,
# at a power-of-two scale bf16 holds too, so they decompress to the dense model's weights exactly.
_INT4 = {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 32}
_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
_COMPRESSIONS = {
    "int4": ("pack-quantized", _INT4, torch.arange(-8, 8) / 64, 1 / 64),
    "mxfp4": (
        "mxfp4-pack-quantized",
        {**_INT4, "type": "float", "scale_dtype": "torch.uint8"},
        torch.cat([_E2M1, -_E2M1[1:]]) / 16,
        1 / 16,
    ),
}


def _save_compressed_tiny_qwen3_moe(dense_dir, packed_dir, compression):
    """The tiny Qwen3-MoE twice: whole in bf16, and with its experts compressed by ``compression``."""
    fmt, weights, values, scale = _COMPRESSIONS[compression]
    _save_tiny_qwen3_moe(dense_dir)
    state = load_file(str(dense_dir / "model.safetensors"))
    gen = torch.Generator().manual_seed(0)
    for key in [k for k in state if ".experts." in k]:  # ...mlp.experts.<e>.gate_proj.weight
        draw = torch.randint(0, len(values), state[key].shape, generator=gen)
        state[key] = values[draw].to(torch.bfloat16)
    save_file(state, str(dense_dir / "model.safetensors"), metadata={"format": "pt"})

    quantization_config = {
        "quant_method": "compressed-tensors",
        "format": fmt,
        "config_groups": {"experts": {"targets": ["Linear"], "weights": weights, "format": fmt}},
    }
    compressor, scheme, _, _ = _onfly_decompress._read_compression(
        SimpleNamespace(quantization_config=quantization_config)
    )
    packed = {}
    for key, tensor in state.items():
        if ".experts." not in key:
            packed[key] = tensor
            continue
        rows, cols = tensor.shape
        scales = torch.full((rows, cols // 32), scale)
        stored = compressor.compress({"weight": tensor.float(), "weight_scale": scales}, scheme)
        for name, value in stored.items():  # floats as bf16, as the Kimi checkpoints store them
            packed[key.removesuffix("weight") + name] = (
                value.bfloat16() if value.is_floating_point() else value
            )
    packed_dir.mkdir()
    save_file(packed, str(packed_dir / "model.safetensors"), metadata={"format": "pt"})
    config = json.loads((dense_dir / "config.json").read_text())
    config["quantization_config"] = quantization_config
    (packed_dir / "config.json").write_text(json.dumps(config))


def _compressed_experts_load_on_their_own_ranks(rank, size, dense_dir, packed_dir):
    device = torch.device(f"cuda:{rank}")
    problems = []
    # The decompressor's hooks are process-wide, and the workers outlive the job.
    hooks = (mu.get_model_conversion_mapping, cml.spawn_materialize)
    try:
        with bounded_collectives(seconds=120):
            config = AutoConfig.from_pretrained(packed_dir)
            _onfly_decompress.install_onfly_decompress(config)  # as a model spec's prepare does
            model = load_and_prepare_ep(
                packed_dir, DistributionPlan(size, ep_size=size), hf_config=config
            ).eval()
            whole = AutoModelForCausalLM.from_pretrained(dense_dir, dtype=torch.bfloat16)
            whole = whole.to(device).eval()

            # This rank's experts, decompressed as they loaded, are the dense model's, exactly.
            per_rank = NUM_EXPERTS // size
            owned = slice(rank * per_rank, (rank + 1) * per_rank)
            for i, (layer, whole_layer) in enumerate(zip(model.model.layers, whole.model.layers)):
                for name in ("gate_up_proj", "down_proj"):
                    local = _local(getattr(layer.mlp.experts, name))
                    if local.shape[0] != per_rank:
                        problems.append(f"layer {i} {name}: {local.shape[0]} experts held")
                    elif not torch.equal(local, getattr(whole_layer.mlp.experts, name)[owned]):
                        problems.append(f"layer {i} {name}: rank {rank}'s experts differ")
            # Every stored tensor of the experts it built went into them.
            if model._modelopt_unplaced_source_keys:
                problems.append(f"unplaced: {model._modelopt_unplaced_source_keys[:3]}")
            set_distribution_plan(model, DistributionPlan(size, ep_size=size))

            generator = torch.Generator().manual_seed(rank)
            input_ids = torch.randint(0, 128, (2, 16), generator=generator).to(device)
            with torch.no_grad():
                sharded = model(input_ids).logits.float()
                reference = whole(input_ids).logits.float()
            if not torch.allclose(sharded, reference, atol=2e-2, rtol=2e-2):
                problems.append(f"logits differ by {(sharded - reference).abs().max().item():.3g}")

            gathered = [None] * size
            dist.all_gather_object(gathered, problems)
    finally:
        mu.get_model_conversion_mapping, cml.spawn_materialize = hooks
    found = [problem for rank_problems in gathered for problem in rank_problems]
    assert not found, "\n".join(found[:20])


@pytest.mark.parametrize("compression", sorted(_COMPRESSIONS))
def test_compressed_experts_load_expert_parallel_each_rank_decompressing_its_own(
    need_2_gpus, dist_workers, tmp_path, compression
):
    """A checkpoint whose experts are compressed (INT4, MXFP4) loads expert-parallel through the
    on-the-fly decompressor: each rank holds exactly the dense model's weights for its own experts,
    every stored tensor is placed, and the sharded model matches the dense one loaded whole. Each
    rank's checks are gathered before any rank fails, so a regression fails instead of hanging."""
    pytest.importorskip(
        "compressed_tensors",
        reason="compressed-tensors is needed: the checkpoint's experts are compressed",
    )
    if Version(transformers.__version__) < Version("5.8"):
        pytest.skip("the on-the-fly decompression needs transformers 5.8")
    if NUM_EXPERTS % dist_workers.world_size:
        pytest.skip(f"{NUM_EXPERTS} experts do not shard over {dist_workers.world_size} GPUs")
    dense_dir, packed_dir = tmp_path / "dense", tmp_path / "packed"
    _save_compressed_tiny_qwen3_moe(dense_dir, packed_dir, compression)
    dist_workers.run(_compressed_experts_load_on_their_own_ranks, str(dense_dir), str(packed_dir))


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
