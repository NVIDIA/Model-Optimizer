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

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.utils.distributed import (
    DistributionPlan,
    distributed_forward_loop,
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


# (dense part sharded, share of the GPUs the experts span). A dense part FSDP2-sharded within several
# expert-parallel groups (sharded with half the GPUs per group) is refused by the loader: the export
# cannot write it.
_LAYOUTS = [
    pytest.param(False, 1, id="dense-replicated-ep-world"),
    pytest.param(False, 2, id="dense-replicated-ep-half"),
    pytest.param(True, 1, id="dense-sharded-ep-world"),
]


def _sharded_model_matches_the_whole_one(rank, size, ckpt_dir, fsdp_dense, ep_degree):
    device = torch.device(f"cuda:{rank}")
    config = AutoConfig.from_pretrained(ckpt_dir)  # carries qwen3_moe's own base_model_ep_plan
    plan = DistributionPlan(size, ep_size=ep_degree, dense="fsdp" if fsdp_dense else "replicate")
    model = load_and_prepare_ep(ckpt_dir, plan, hf_config=config).eval()
    # The loader swaps the plan on its own copy: the caller's config keeps the checkpoint's.
    assert config.base_model_ep_plan == AutoConfig.from_pretrained(ckpt_dir).base_model_ep_plan
    # The loader records no plan; loading through it directly, the caller records one.
    with pytest.raises(RuntimeError, match="no distribution plan recorded"):
        distribution_plan(model)
    set_distribution_plan(model, plan)
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


@pytest.mark.parametrize(("fsdp_dense", "ep_share"), _LAYOUTS)
def test_expert_parallel_model_matches_the_whole_model(
    need_2_gpus, dist_workers, tmp_path, fsdp_dense, ep_share
):
    """With the dense part replicated, or FSDP2-sharded and loaded shard by shard through DCP; and
    with the experts sharded over every rank, or over half of them and replicated across the halves."""
    ep_degree = dist_workers.world_size // ep_share
    if ep_degree < 2 or dist_workers.world_size % ep_share or NUM_EXPERTS % ep_degree:
        pytest.skip(f"{NUM_EXPERTS} experts do not shard over EP {ep_degree} on these GPUs")
    _save_tiny_qwen3_moe(tmp_path)
    dist_workers.run(_sharded_model_matches_the_whole_one, str(tmp_path), fsdp_dense, ep_degree)


def _batches(size):
    """Every rank's batch, the same on all ranks: each trains on its own, the reference on all."""
    return [
        torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(r))
        for r in range(size)
    ]


def _whole_model_grads_per_batch(whole, batches, device):
    """The whole model's gradient on each batch alone, as ``{parameter name: gradient}``."""
    grads = []
    for batch in batches:
        whole.zero_grad(set_to_none=True)
        batch = batch.to(device)
        whole(batch, labels=batch).loss.backward()
        grads.append({n: p.grad.detach().clone() for n, p in whole.named_parameters()})
    return grads


def _fail_together(problems):
    """Raise on every rank if any rank found a problem, so that no rank walks on alone into a collective
    the failed ones never reach (which would hang until the test timeout instead of failing)."""
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, problems)
    found = [
        f"rank {r}: {problem}"
        for r, rank_problems in enumerate(gathered)
        for problem in rank_problems
    ]
    assert not found, "\n".join(found[:20])


def _relative_error(actual, expected):
    # bf16 gradients summed in a different order on each side: compare by relative norm.
    return (
        (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    ).item()


def _assert_grads_match_the_whole_model(model, per_batch, rank, size, ep_degree):
    """Routed experts collect the gradient of every token routed to them, their EP group's batches;
    FSDP2-sharded dense parameters hold the gradient averaged over every rank; replicated dense
    parameters see only this rank's batch."""
    ep_rank = rank % ep_degree
    group = range(rank - ep_rank, rank - ep_rank + ep_degree)
    per_rank = NUM_EXPERTS // ep_degree
    owned = slice(ep_rank * per_rank, (ep_rank + 1) * per_rank)
    problems, names = [], set()
    for name, param in model.named_parameters():
        names.add(name)
        if param.grad is None:
            problems.append(f"{name} received no gradient")
            continue
        if ".mlp.experts." in name:
            actual = _local(param.grad)
            expected = sum(per_batch[r][name] for r in group)[owned]
        elif isinstance(param.grad, DTensor):
            actual = param.grad.full_tensor()
            expected = sum(per_batch[r][name] for r in range(size)) / size
        else:
            actual = param.grad
            expected = per_batch[rank][name]
        error = _relative_error(actual, expected)
        if not error < 3e-2:
            problems.append(f"{name}: gradient off by {error:.3g} (relative norm)")
    if names != set(per_batch[0]):
        problems.append(
            f"parameters differ from the whole model's: {sorted(names ^ set(per_batch[0]))[:5]}"
        )
    _fail_together(problems)


def _sharded_model_backward_matches_the_whole_one(rank, size, ckpt_dir, fsdp_dense, ep_degree):
    device = torch.device(f"cuda:{rank}")
    config = AutoConfig.from_pretrained(ckpt_dir)
    plan = DistributionPlan(size, ep_size=ep_degree, dense="fsdp" if fsdp_dense else "replicate")
    model = load_and_prepare_ep(ckpt_dir, plan, hf_config=config, freeze=False)
    set_distribution_plan(
        model, plan
    )  # loaded directly, so recorded here, as load_pretrained would
    whole = AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=torch.bfloat16).to(device)

    batches = _batches(size)
    mine = batches[rank].to(device)
    model(mine, labels=mine).loss.backward()
    _assert_grads_match_the_whole_model(
        model, _whole_model_grads_per_batch(whole, batches, device), rank, size, ep_degree
    )


@pytest.mark.parametrize(("fsdp_dense", "ep_share"), _LAYOUTS)
def test_expert_parallel_backward_matches_the_whole_model(
    need_2_gpus, dist_workers, tmp_path, fsdp_dense, ep_share
):
    """Loaded trainable, the sharded model's gradients are the whole model's: tokens' gradients travel
    back through the all-to-all to the experts that saw them, and to the router that chose them."""
    ep_degree = dist_workers.world_size // ep_share
    if ep_degree < 2 or dist_workers.world_size % ep_share or NUM_EXPERTS % ep_degree:
        pytest.skip(f"{NUM_EXPERTS} experts do not shard over EP {ep_degree} on these GPUs")
    _save_tiny_qwen3_moe(tmp_path)
    dist_workers.run(
        _sharded_model_backward_matches_the_whole_one, str(tmp_path), fsdp_dense, ep_degree
    )


def _calibrated_quantizers(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, TensorQuantizer) and module.is_enabled and module.amax is not None
    }


def _quantized_sharded_model_trains_like_the_whole_one(
    rank, size, ckpt_dir, fsdp_dense, fmt, ep_degree
):
    cfg = {"fp8": mtq.FP8_DEFAULT_CFG, "nvfp4": mtq.NVFP4_DEFAULT_CFG}[fmt]
    device = torch.device(f"cuda:{rank}")
    config = AutoConfig.from_pretrained(ckpt_dir)
    plan = DistributionPlan(size, ep_size=ep_degree, dense="fsdp" if fsdp_dense else "replicate")
    model = load_and_prepare_ep(ckpt_dir, plan, hf_config=config, freeze=False)
    set_distribution_plan(
        model, plan
    )  # loaded directly, so recorded here, as load_pretrained would
    whole = AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=torch.bfloat16).to(device)

    # Calibrate as hf_ptq does: each rank on its own batch, the reference on all of them.
    batches = _batches(size)
    mine = batches[rank].to(device)
    forward_loop = distributed_forward_loop(
        model, [{"input_ids": mine}], device=device, ep_mesh=expert_parallel_mesh(model)
    )
    mtq.quantize(model, cfg, forward_loop)
    mtq.quantize(whole, cfg, lambda m: [m(b.to(device)) for b in batches])

    # Expert-parallel calibration reaches the whole model's scales: weight scales from this rank's
    # experts, activation scales synced over the ranks (equal to bf16 rounding of the activations).
    per_rank = NUM_EXPERTS // ep_degree
    ep_rank = rank % ep_degree
    whole_quantizers = _calibrated_quantizers(whole)
    sharded_quantizers = _calibrated_quantizers(model)
    problems = [] if sharded_quantizers else ["no quantizer was calibrated"]
    for name, quantizer in sharded_quantizers.items():
        if "_weight_quantizers." in name:
            prefix, index = name.rsplit(".", 1)
            name = f"{prefix}.{ep_rank * per_rank + int(index)}"
        reference = whole_quantizers.get(name)
        if reference is None:
            problems.append(f"{name} is calibrated here but not in the whole model")
            continue
        error = _relative_error(quantizer.amax, reference.amax)
        if not error < 2e-2:
            problems.append(f"{name}: amax off by {error:.3g} (relative)")
        # Train on identical scales, so that the gradients compare the transport and not calibration.
        quantizer.amax = reference.amax
    _fail_together(problems)

    model(mine, labels=mine).loss.backward()
    _assert_grads_match_the_whole_model(
        model, _whole_model_grads_per_batch(whole, batches, device), rank, size, ep_degree
    )


@pytest.mark.parametrize("fmt", ["fp8", "nvfp4"])
@pytest.mark.parametrize(("fsdp_dense", "ep_share"), _LAYOUTS)
def test_quantized_expert_parallel_model_trains_like_the_whole_model(
    need_2_gpus, dist_workers, tmp_path, fsdp_dense, fmt, ep_share
):
    """QAT under expert parallelism: quantized and calibrated expert-parallel -- experts over every
    rank or half of them, dense replicated or FSDP2-sharded -- the model's scales and its gradients
    through the fake quantizers match the whole model calibrated on every rank's data."""
    ep_degree = dist_workers.world_size // ep_share
    if ep_degree < 2 or dist_workers.world_size % ep_share or NUM_EXPERTS % ep_degree:
        pytest.skip(f"{NUM_EXPERTS} experts do not shard over EP {ep_degree} on these GPUs")
    _save_tiny_qwen3_moe(tmp_path)
    dist_workers.run(
        _quantized_sharded_model_trains_like_the_whole_one,
        str(tmp_path),
        fsdp_dense,
        fmt,
        ep_degree,
    )
