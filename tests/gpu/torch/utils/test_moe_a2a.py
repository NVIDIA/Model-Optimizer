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

"""GPU tests for the ``torch_a2a`` expert-parallel MoE transport (``a2a_moe``)."""

import copy
import dataclasses
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import bounded_collectives
from packaging.version import Version
from torch.distributed.device_mesh import init_device_mesh

import modelopt.torch.quantization as mtq
import modelopt.torch.utils.moe as moe
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.utils.distributed import (
    DistributionPlan,
    ParallelState,
    distributed_forward_loop,
    set_distribution_plan,
)

transformers = pytest.importorskip("transformers")
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3MoeConfig
from transformers.distributed import DistributedConfig

# One MoE layer's expert stage, big enough that the order of each token's top-k adds shows in bf16.
TOKENS, HIDDEN, EXPERTS, TOP_K = 4096, 1024, 16, 8


def _grouped_mm_bit_for_bit_job(rank, size):
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    moe.register_torch_a2a()
    group, _ = dist.new_subgroups(1)
    try:
        config = Qwen3MoeConfig(
            hidden_size=HIDDEN,
            moe_intermediate_size=HIDDEN // 2,
            num_experts=EXPERTS,
            num_experts_per_tok=TOP_K,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
        )
        torch.manual_seed(0)
        grouped = Qwen3MoeExperts(config).to("cuda", torch.bfloat16)
        with torch.no_grad():
            grouped.gate_up_proj.normal_(std=0.05)
            grouped.down_proj.normal_(std=0.05)
        a2a = copy.deepcopy(grouped)
        grouped.config._experts_implementation = "grouped_mm"
        a2a.config._experts_implementation = "torch_a2a"
        x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
        weights, index = torch.randn(TOKENS, EXPERTS, device="cuda").softmax(-1).topk(TOP_K, dim=-1)
        with torch.no_grad():
            want = grouped(x, index, weights)
            got = a2a(x, index, weights, group)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    finally:
        dist.destroy_process_group(group)


def test_torch_a2a_gives_grouped_mms_result_bit_for_bit(dist_workers):
    """fp32 routing weights, as Mixtral's and DeepSeek-V3's routers return, make each token's top-k
    sum an fp32 one, whose bf16 result depends on the order of its adds. At EP=1 the torch_a2a
    forward adds them in grouped_mm's fixed order, not in the order of CUDA atomics."""
    transformers = pytest.importorskip("transformers")
    if Version(transformers.__version__) < Version("5.6"):
        pytest.skip("expert parallelism needs transformers >= 5.6")
    dist_workers.run(_grouped_mm_bit_for_bit_job)


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


def _load_expert_parallel(ckpt_dir, size):
    """The model with its experts sharded over every rank by transformers' own expert-parallel load,
    on the ``torch_a2a`` styles and experts forward, and the mesh they are sharded over."""
    moe.register_torch_a2a()
    config = AutoConfig.from_pretrained(ckpt_dir)
    swap = {"ep_router": "torch_a2a_router", "moe_tp_experts": "torch_a2a_experts"}
    config.base_model_ep_plan = {k: swap.get(v, v) for k, v in config.base_model_ep_plan.items()}
    mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("ep",))
    # Each option where this release takes it: on DistributedConfig when it has a field for it.
    fields = {field.name for field in dataclasses.fields(DistributedConfig)}
    options = {"tp_plan": "auto", "tp_size": size}
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        config=config,
        dtype=torch.bfloat16,
        device_mesh=mesh,
        distributed_config=DistributedConfig(
            enable_expert_parallel=True, **{k: v for k, v in options.items() if k in fields}
        ),
        experts_implementation="torch_a2a",
        **({} if "tp_plan" in fields else {"tp_plan": "auto"}),
    )
    for module in model.modules():  # the experts forward reads it at forward time
        if hasattr(getattr(module, "config", None), "_experts_implementation"):
            module.config._experts_implementation = "torch_a2a"
    return model, mesh


def _calibrated(model):
    return {
        name: module.amax.float().cpu()
        for name, module in model.named_modules()
        if isinstance(module, TensorQuantizer) and module.is_enabled and module.amax is not None
    }


def _expert_parallel_calibration_job(rank, size, ckpt_dir):
    device = torch.device("cuda", rank)
    batches = [
        torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(r))
        for r in range(size)
    ]
    # The scales' syncs give up in seconds on this group of every rank: the experts' expert-parallel
    # group (they span every rank), and the dense modules' data-parallel one.
    group = dist.new_group(timeout=timedelta(seconds=120))
    dist.barrier(group=group)
    try:
        with bounded_collectives(seconds=120):
            model, mesh = _load_expert_parallel(ckpt_dir, size)
            for name, module in model.named_modules():
                module._parallel_state = (
                    ParallelState(data_parallel_group=-1, expert_model_parallel_group=group)
                    if ".mlp.experts" in name
                    else ParallelState(data_parallel_group=group)
                )
            set_distribution_plan(model, DistributionPlan(size, ep_size=size))
            # Each rank its own batch, through the quantized model's re-planted torch_a2a forwards.
            forward_loop = distributed_forward_loop(
                model, [{"input_ids": batches[rank]}], device=device, ep_mesh=mesh
            )
            mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

            # The whole model, calibrated in one process on every rank's batch.
            whole = AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=torch.bfloat16).to(device)
            for module in whole.modules():
                module._parallel_state = ParallelState(data_parallel_group=-1)
            mtq.quantize(whole, mtq.FP8_DEFAULT_CFG, lambda m: [m(b.to(device)) for b in batches])

            sharded, reference = _calibrated(model), _calibrated(whole)
            problems, compared = [] if sharded else ["no quantizer was calibrated"], []
            per_rank = NUM_EXPERTS // size
            for name, amax in sharded.items():
                if "_weight_quantizers." in name:  # this rank's local expert -> the global one
                    prefix, index = name.rsplit(".", 1)
                    name = f"{prefix}.{rank * per_rank + int(index)}"
                compared.append(name)
                want = reference.get(name)
                if want is None:
                    problems.append(f"{name} is calibrated here but not in the whole model")
                    continue
                # bf16 activations, computed in another order: compare by relative norm.
                error = ((amax - want).norm() / want.norm().clamp_min(1e-12)).item()
                if not error < 2e-2:
                    problems.append(f"{name}: amax off by {error:.3g} (relative)")
            gathered = [None] * size
            dist.all_gather_object(gathered, (problems, compared))
    finally:
        dist.destroy_process_group(group)

    found = [problem for problems, _ in gathered for problem in problems]
    assert not found, "\n".join(found[:20])
    experts = {name for name in reference if "_weight_quantizers." in name}
    assert experts and experts <= {name for _, names in gathered for name in names}, experts


def test_expert_parallel_calibration_matches_the_whole_model(need_2_gpus, dist_workers, tmp_path):
    """transformers' own expert-parallel load on the torch_a2a transport, quantized and calibrated
    through distributed_forward_loop, each rank on its own batch: every expert's weight scales and
    every activation scale match the whole model calibrated on every rank's batch. Quantizer
    insertion drops the transport's forward wrappers on transformers >= 5.16, which the loop
    re-plants."""
    if Version(transformers.__version__) < Version("5.16"):
        pytest.skip("the torch_a2a forward wrappers are pre-hooks below transformers 5.16")
    if NUM_EXPERTS % torch.cuda.device_count():
        pytest.skip(f"{NUM_EXPERTS} experts do not shard over {torch.cuda.device_count()} GPUs")
    _save_tiny_qwen3_moe(tmp_path)
    dist_workers.run(_expert_parallel_calibration_job, ckpt_dir=str(tmp_path))
