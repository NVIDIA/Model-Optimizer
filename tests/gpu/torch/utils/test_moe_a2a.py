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

"""GPU tests for the ``torch_a2a`` expert-parallel MoE transport (``a2a_moe``), and for a model
loaded expert-parallel on it, calibrated and exported."""

import copy
import json

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import bounded_collectives
from packaging.version import Version
from safetensors.torch import load_file

import modelopt.torch.quantization as mtq
import modelopt.torch.utils.moe as moe
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.utils.distributed import (
    DistributionPlan,
    ParallelState,
    distributed_forward_loop,
)

transformers = pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM, Qwen3MoeConfig

from modelopt.torch.export.quant_aware_conversion import revert_weight_conversion_quant_aware
from modelopt.torch.export.unified_export_hf import (
    _export_transformers_checkpoint,
    _revert_quant_config_names_best_effort,
    export_hf_checkpoint,
)
from modelopt.torch.utils.plugins.model_load_utils import load_pretrained

# One MoE layer's expert stage, big enough that the order of each token's top-k adds shows in bf16.
TOKENS, HIDDEN, EXPERTS, TOP_K = 4096, 1024, 16, 8


def _grouped_mm_bit_for_bit_job(rank, size):
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


NUM_LAYERS, NUM_EXPERTS = 2, 8


def _save_tiny_qwen3_moe(ckpt_dir, dtype=torch.bfloat16):
    config = Qwen3MoeConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).to(dtype).save_pretrained(ckpt_dir)


def _calibrated(model):
    return {
        name: module.amax.float().cpu()
        for name, module in model.named_modules()
        if isinstance(module, TensorQuantizer) and module.is_enabled and module.amax is not None
    }


def _relative_error(got, want):
    got, want = got.float().cpu(), want.float().cpu()
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


def _raw_bytes(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8)


def _export_problems(export_dir, want):
    """How the checkpoint in ``export_dir`` differs from ``want``, the single-process export's.

    Each tensor sits in exactly one shard, the one the index names for it, under a name ``want``
    has, every expert of every layer among them. Each is byte-identical but the activation scales:
    those come from bf16 activations computed in another order, so they agree within 2e-2.
    """
    weight_map = json.loads((export_dir / "model.safetensors.index.json").read_text())["weight_map"]
    got, problems = {}, []
    for path in sorted(export_dir.glob("*.safetensors")):
        for key, tensor in load_file(path).items():
            if weight_map.get(key) != path.name:  # written twice, or not where the index says
                problems.append(f"{key} is in {path.name}; the index names {weight_map.get(key)}")
            got[key] = tensor
    problems += [f"the index names {key}, but no shard holds it" for key in weight_map.keys() - got]
    experts = [key for key in want if ".mlp.experts." in key and key.endswith("_proj.weight")]
    if len(experts) != NUM_LAYERS * NUM_EXPERTS * 3:  # gate, up and down
        problems.append(f"the single-process export has {len(experts)} expert weights, not all")
    problems += [f"{key} is not written" for key in want.keys() - got.keys()]
    problems += [f"{key} is not in the single-process export" for key in got.keys() - want.keys()]
    for key in want.keys() & got.keys():
        tensor, expected = got[key], want[key].cpu()
        if (tensor.dtype, tensor.shape) != (expected.dtype, expected.shape):
            problems.append(
                f"{key}: {tensor.dtype} {list(tensor.shape)}, not {expected.dtype} "
                f"{list(expected.shape)}"
            )
        elif key.endswith("input_scale"):
            if not (error := _relative_error(tensor, expected)) < 2e-2:
                problems.append(f"{key}: off by {error:.3g} (relative)")
        elif not torch.equal(_raw_bytes(tensor), _raw_bytes(expected)):
            problems.append(f"{key} is not byte-identical")
    return sorted(problems)


def _expert_parallel_calibration_and_export_job(rank, size, ckpt_dir, export_dir, ep_degree):
    device = torch.device("cuda", rank)
    batches = [
        torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(r))
        for r in range(size)
    ]
    with bounded_collectives(seconds=120):
        # The production load: each expert on one rank of every expert-parallel group of ep_degree
        # ranks, and every module's ParallelState seeded for the scales' syncs.
        model = load_pretrained(ckpt_dir, distribution=DistributionPlan(size, ep_size=ep_degree))
        # Each rank its own batch, through the quantized model's re-planted torch_a2a forwards.
        forward_loop = distributed_forward_loop(
            model,
            [{"input_ids": batches[rank]}],
            device=device,
            ep_mesh=moe.expert_parallel_mesh(model),
        )
        mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

        # The whole model, calibrated in one process on every rank's batch.
        whole = AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=torch.bfloat16).to(device)
        for module in whole.modules():
            module._parallel_state = ParallelState(data_parallel_group=-1)
        mtq.quantize(whole, mtq.FP8_DEFAULT_CFG, lambda m: [m(b.to(device)) for b in batches])

        sharded, reference = _calibrated(model), _calibrated(whole)
        problems, compared = [] if sharded else ["no quantizer was calibrated"], []
        per_rank = NUM_EXPERTS // ep_degree
        for name, amax in sharded.items():
            if "_weight_quantizers." in name:  # this rank's local expert -> the global one
                prefix, index = name.rsplit(".", 1)
                # An expert-parallel group's ranks are consecutive.
                name = f"{prefix}.{rank % ep_degree * per_rank + int(index)}"
            compared.append(name)
            want = reference.get(name)
            if want is None:
                problems.append(f"{name} is calibrated here but not in the whole model")
                continue
            # bf16 activations, computed in another order: compare by relative norm.
            error = _relative_error(amax, want)
            if not error < 2e-2:
                problems.append(f"{name}: amax off by {error:.3g} (relative)")

        # The no-gather writer's checkpoint, against what the single-process export writes for the
        # whole model.
        export_hf_checkpoint(model, dtype=torch.bfloat16, export_dir=export_dir)
        if rank == 0:
            written, _ = _export_transformers_checkpoint(whole, torch.bfloat16)
            written = revert_weight_conversion_quant_aware(whole, dict(written))
            problems += _export_problems(export_dir, written)
        gathered = [None] * size
        dist.all_gather_object(gathered, (problems, compared))

    found = [problem for problems, _ in gathered for problem in problems]
    assert not found, "\n".join(found[:20])
    experts = {name for name in reference if "_weight_quantizers." in name}
    assert experts and experts <= {name for _, names in gathered for name in names}, experts


@pytest.mark.parametrize("ep_share", [1, 2], ids=["ep-world", "ep-half"])
def test_expert_parallel_calibration_and_export_match_the_whole_model(
    need_2_gpus, dist_workers, tmp_path, ep_share
):
    """The production expert-parallel load on the torch_a2a transport, with the experts sharded over
    every rank, or over half of them and replicated across the halves, quantized and calibrated
    through distributed_forward_loop, each rank on its own batch, then exported by the no-gather
    writer. Every expert's weight scales and every activation scale match the whole model
    calibrated on every rank's batch, and the checkpoint matches the whole model's single-process
    export (:func:`_export_problems`). Quantizer insertion drops the transport's forward wrappers on
    transformers >= 5.16, which the loop re-plants."""
    if Version(transformers.__version__) < Version("5.16"):
        pytest.skip("the torch_a2a forward wrappers are pre-hooks below transformers 5.16")
    ep_degree = dist_workers.world_size // ep_share
    if ep_degree < 2 or dist_workers.world_size % ep_share or NUM_EXPERTS % ep_degree:
        pytest.skip(f"{NUM_EXPERTS} experts do not shard over EP {ep_degree} on these GPUs")
    _save_tiny_qwen3_moe(tmp_path / "ckpt")
    dist_workers.run(
        _expert_parallel_calibration_and_export_job,
        ckpt_dir=str(tmp_path / "ckpt"),
        export_dir=tmp_path / "export",
        ep_degree=ep_degree,
    )


def _expert_parallel_auto_quantize_and_export_job(rank, size, ckpt_dir, export_dir, plan):
    device = torch.device("cuda", rank)
    batches = [
        torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(r))
        for r in range(size)
    ]

    def search(model, batches):
        return mtq.auto_quantize(
            model,
            constraints={"effective_bits": 6.0},
            quantization_formats=[mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG],
            data_loader=[{"input_ids": b.to(device), "labels": b.to(device)} for b in batches],
            forward_step=lambda model, batch: model(**batch),
            loss_func=lambda output, _: output.loss,
            num_calib_steps=len(batches),
            num_score_steps=len(batches),
        )[1]

    with bounded_collectives(seconds=300):
        # The production load, each rank searching on its own batch; and the whole model in one
        # process searching on every rank's batch, one per step as the ranks do.
        model = load_pretrained(ckpt_dir, distribution=plan)
        state = search(model, batches[rank : rank + 1])
        whole = AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=torch.float32).to(device)
        for module in whole.modules():
            module._parallel_state = ParallelState(data_parallel_group=-1)
        whole_state = search(whole, batches)

        problems = []
        if state["best"]["recipe"] != whole_state["best"]["recipe"]:
            problems.append(
                f"picked {state['best']['recipe']}, not {whole_state['best']['recipe']}"
            )
        for name, stat in whole_state["candidate_stats"].items():
            mine = state["candidate_stats"][name]
            if (mine["formats"], mine["costs"]) != (stat["formats"], stat["costs"]):
                problems.append(f"{name}: costs {mine['costs']}, not {stat['costs']}")
            # float32: the scores differ only by their order of summation.
            error = _relative_error(torch.tensor(mine["scores"]), torch.tensor(stat["scores"]))
            if not error < 1e-2:
                problems.append(f"{name}: scores off by {error:.3g} (relative)")

        # The mixed-format checkpoint the no-gather writer writes, against the single-process one.
        export_hf_checkpoint(model, dtype=torch.bfloat16, export_dir=export_dir)
        if rank == 0:
            written, quant_config = _export_transformers_checkpoint(whole, torch.bfloat16)
            problems += _export_problems(
                export_dir, revert_weight_conversion_quant_aware(whole, dict(written))
            )
            want = _revert_quant_config_names_best_effort(whole, quant_config)["quantization"]
            got = json.loads((export_dir / "hf_quant_config.json").read_text())["quantization"]
            if want["quant_algo"] != "MIXED_PRECISION" or got != want:
                problems.append(f"hf_quant_config: {got}, not {want}")
        gathered = [None] * size
        dist.all_gather_object(gathered, problems)

    found = [problem for problems in gathered for problem in problems]
    assert not found, "\n".join(found[:20])


@pytest.mark.parametrize(
    ("dense", "ep_share"),
    [("replicate", 1), ("replicate", 2), ("fsdp", 1)],
    ids=["dense-replicated-ep-world", "dense-replicated-ep-half", "dense-sharded-ep-world"],
)
def test_expert_parallel_auto_quantize_and_export_match_the_whole_model(
    need_2_gpus, dist_workers, tmp_path, dense, ep_share
):
    """AutoQuantize of the production expert-parallel load -- experts over every rank or half of
    them, dense replicated or FSDP2-sharded -- each rank scoring its own batch, gives the costs,
    scores and formats of the whole model scoring every rank's batch, in float32, where only their
    order of summation differs. The no-gather writer's mixed-format checkpoint, weights and
    quantization config, then matches the whole model's single-process export."""
    if Version(transformers.__version__) < Version("5.16"):
        pytest.skip("the torch_a2a forward wrappers are pre-hooks below transformers 5.16")
    ep_degree = dist_workers.world_size // ep_share
    if ep_degree < 2 or dist_workers.world_size % ep_share or NUM_EXPERTS % ep_degree:
        pytest.skip(f"{NUM_EXPERTS} experts do not shard over EP {ep_degree} on these GPUs")
    _save_tiny_qwen3_moe(tmp_path / "ckpt", dtype=torch.float32)
    dist_workers.run(
        _expert_parallel_auto_quantize_and_export_job,
        ckpt_dir=str(tmp_path / "ckpt"),
        export_dir=tmp_path / "export",
        plan=DistributionPlan(dist_workers.world_size, ep_size=ep_degree, dense=dense),
    )
