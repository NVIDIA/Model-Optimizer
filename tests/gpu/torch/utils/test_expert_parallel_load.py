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
experts are compressed-tensors compressed loads the same way, each rank decompressing its own, and
so does a remote-code VLM through its model spec's adapter.
"""

import copy
import json
import shutil
from pathlib import Path
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
from _test_utils.torch import kimi_vl_tiny_modeling
from torch.distributed.tensor import DTensor
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    Qwen3MoeConfig,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.models import _onfly_decompress
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils.packed_experts import is_packed, unpacked
from modelopt.torch.utils.distributed import (
    DistributionPlan,
    distributed_forward_loop,
    distribution_plan,
    set_distribution_plan,
)
from modelopt.torch.utils.moe import apply_distributed_support, expert_parallel_mesh
from modelopt.torch.utils.plugins.model_load_utils import load_and_prepare_ep

NUM_EXPERTS = 8


def _decisive_router(model):
    """Route token ``t`` to expert ``t % 8`` and one more its id picks, by a code its embedding
    carries: every token's top two scores clear the third by over 2 in both layers. A random router
    leaves near ties, which bf16 rounding flips on some GPUs (an RTX 6000 Ada), and a token that
    takes other experts sends its gradient to other weights."""
    with torch.no_grad():
        embed = model.model.embed_tokens.weight
        embed[:, :NUM_EXPERTS] = 0
        for token in range(embed.shape[0]):
            first = token % NUM_EXPERTS
            second = (first + 1 + (token // NUM_EXPERTS) % (NUM_EXPERTS - 1)) % NUM_EXPERTS
            embed[token, first], embed[token, second] = 1.0, 0.8
        for layer in model.model.layers:
            layer.mlp.gate.weight.zero_()
            layer.mlp.gate.weight[:, :NUM_EXPERTS] = 0.5 * torch.eye(NUM_EXPERTS)


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
    model = AutoModelForCausalLM.from_config(config)
    _decisive_router(model)
    model.to(torch.bfloat16).save_pretrained(ckpt_dir)


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
    # Copied out of the file, which is written over below.
    state = {k: v.clone() for k, v in load_file(str(dense_dir / "model.safetensors")).items()}
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


def _checkpoint_tensors(directory):
    tensors = {}
    for file in sorted(Path(directory).glob("*.safetensors")):
        tensors.update(load_file(str(file)))
    return tensors


def _compressed_experts_load_on_their_own_ranks(
    rank, size, dense_dir, packed_dir, export_dir, dense
):
    device = torch.device(f"cuda:{rank}")
    plan = DistributionPlan(size, ep_size=size, dense=dense)
    calibrates = Version(transformers.__version__) >= Version("5.16")
    # Exported with the dense part replicated: packed experts unpack alike either way, and the
    # export of a sharded dense part is no part of keeping them packed.
    exports = calibrates and dense == "replicate"
    amaxes = {}
    problems = []
    # The decompressor's hooks are process-wide, and the workers outlive the job.
    hooks = (mu.get_model_conversion_mapping, cml.spawn_materialize)
    try:
        with bounded_collectives(seconds=120):
            whole = AutoModelForCausalLM.from_pretrained(dense_dir, dtype=torch.bfloat16)
            whole = whole.to(device).eval()
            generator = torch.Generator().manual_seed(rank)
            input_ids = torch.randint(0, 128, (2, 16), generator=generator).to(device)
            with torch.no_grad():
                reference = whole(input_ids).logits.float()
            per_rank = NUM_EXPERTS // size
            owned = slice(rank * per_rank, (rank + 1) * per_rank)
            # Kept packed, then decompressed as they load: the same model either way.
            for keep in (True, False):
                config = AutoConfig.from_pretrained(packed_dir)
                _onfly_decompress.install_onfly_decompress(config)  # as a model spec's prepare does
                model = load_and_prepare_ep(
                    packed_dir, plan, hf_config=config, keep_compressed=keep
                ).eval()

                # This rank's experts, decompressed, are the dense model's, exactly; kept packed,
                # they hold the stored values and scales of this rank's experts alone.
                for i, (layer, whole_layer) in enumerate(
                    zip(model.model.layers, whole.model.layers)
                ):
                    experts = layer.mlp.experts
                    if is_packed(experts) is not keep:
                        problems.append(f"keep={keep}: layer {i}'s experts are not as asked")
                        continue
                    stored = [p for n, p in experts.named_parameters() if n.endswith("proj_scales")]
                    if keep and [len(_local(p)) for p in stored] != [per_rank, per_rank]:
                        problems.append(f"layer {i}: scales of {[len(_local(p)) for p in stored]}")
                    with unpacked(experts):
                        for name in ("gate_up_proj", "down_proj"):
                            local = _local(getattr(experts, name))
                            if local.shape[0] != per_rank:
                                problems.append(f"layer {i} {name}: {local.shape[0]} experts held")
                            elif not torch.equal(
                                local, getattr(whole_layer.mlp.experts, name)[owned]
                            ):
                                problems.append(f"layer {i} {name}: rank {rank}'s experts differ")
                # Every stored tensor of the experts it built went into them.
                if model._modelopt_unplaced_source_keys:
                    problems.append(f"unplaced: {model._modelopt_unplaced_source_keys[:3]}")
                set_distribution_plan(model, plan)

                with torch.no_grad():
                    sharded = model(input_ids).logits.float()
                if not torch.allclose(sharded, reference, atol=2e-2, rtol=2e-2):
                    error = (sharded - reference).abs().max().item()
                    problems.append(f"keep={keep}: logits differ by {error:.3g}")

                if calibrates:
                    # Calibrated through the re-planted torch_a2a forwards, as hf_ptq does.
                    forward_loop = distributed_forward_loop(
                        model,
                        [{"input_ids": input_ids}],
                        device=device,
                        ep_mesh=expert_parallel_mesh(model),
                    )
                    mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)
                    amaxes[keep] = {
                        name: quantizer.amax.clone()
                        for name, quantizer in _calibrated_quantizers(model).items()
                    }
                if exports:
                    export_hf_checkpoint(
                        model, dtype=torch.bfloat16, export_dir=f"{export_dir}/{keep}"
                    )
                    dist.barrier()
                del model
                torch.cuda.empty_cache()

            if amaxes and (amaxes[True].keys() != amaxes[False].keys() or not amaxes[True]):
                problems.append(f"calibrated {len(amaxes[True])} and {len(amaxes[False])} scales")
            problems += [
                f"{name}: its scale differs, packed or not"
                for name, amax in amaxes.get(False, {}).items()
                if name in amaxes[True] and not torch.equal(amax, amaxes[True][name])
            ]
            if rank == 0 and exports:
                packed, decompressed = (
                    _checkpoint_tensors(f"{export_dir}/{k}") for k in (True, False)
                )
                if not packed or packed.keys() != decompressed.keys():
                    problems.append(
                        f"the exports hold {len(packed)} and {len(decompressed)} tensors"
                    )
                problems += [
                    f"{key} differs between the exports"
                    for key, tensor in decompressed.items()
                    if key in packed
                    and not torch.equal(
                        tensor.reshape(-1).view(torch.uint8),
                        packed[key].reshape(-1).view(torch.uint8),
                    )
                ]
                for name in ("config.json", "hf_quant_config.json"):
                    if (
                        Path(export_dir, "True", name).read_bytes()
                        != Path(export_dir, "False", name).read_bytes()
                    ):
                        problems.append(f"{name} differs between the exports")

            gathered = [None] * size
            dist.all_gather_object(gathered, problems)
    finally:
        mu.get_model_conversion_mapping, cml.spawn_materialize = hooks
    found = [problem for rank_problems in gathered for problem in rank_problems]
    assert not found, "\n".join(found[:20])


@pytest.mark.parametrize("dense", ["replicate", "fsdp"])
@pytest.mark.parametrize("compression", sorted(_COMPRESSIONS))
def test_compressed_experts_load_expert_parallel_each_rank_decompressing_its_own(
    need_2_gpus, dist_workers, tmp_path, compression, dense
):
    """A checkpoint whose experts are compressed (INT4, MXFP4) loads expert-parallel through the
    on-the-fly decompressor, decompressed as they load or kept packed: each rank holds exactly the
    dense model's weights for its own experts, every stored tensor is placed, and the sharded model
    matches the dense one loaded whole, with the dense part replicated or FSDP2-sharded. Kept
    packed, each rank holds its experts' stored values and scales alone, and the model calibrates
    to the decompressed one's scales exactly, and, its dense part replicated, exports the same
    checkpoint byte for byte.
    Each rank's checks are gathered before any rank fails, so a regression fails instead of
    hanging."""
    pytest.importorskip(
        "compressed_tensors",
        minversion="0.16",
        reason="compressed-tensors is needed: the checkpoint's experts are compressed",
    )
    if Version(transformers.__version__) < Version("5.8"):
        pytest.skip("the on-the-fly decompression needs transformers 5.8")
    if NUM_EXPERTS % dist_workers.world_size:
        pytest.skip(f"{NUM_EXPERTS} experts do not shard over {dist_workers.world_size} GPUs")
    dense_dir, packed_dir = tmp_path / "dense", tmp_path / "packed"
    _save_compressed_tiny_qwen3_moe(dense_dir, packed_dir, compression)
    dist_workers.run(
        _compressed_experts_load_on_their_own_ranks,
        str(dense_dir),
        str(packed_dir),
        str(tmp_path / "export"),
        dense,
    )


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


def _routes(model, batch):
    """Each layer's experts for every token of ``batch``, sorted: both routers emit global ids."""
    routes = []
    hooks = [
        layer.mlp.gate.register_forward_hook(
            lambda module, args, output: routes.append(output[2].sort(-1).values)
        )
        for layer in model.model.layers
    ]
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for hook in hooks:
            hook.remove()
    return routes


def _assert_routes_match_the_whole_model(model, whole, batch):
    """The gradients compare only if every token takes the same experts in both models: one routed
    elsewhere sends its gradient to other weights, which no tolerance should absorb. Said so, rather
    than as the gradient errors it would cause."""
    problems = []
    for layer, (sharded, reference) in enumerate(zip(_routes(model, batch), _routes(whole, batch))):
        if differ := (sharded != reference).any(-1).sum().item():
            problems.append(
                f"routing differs: layer {layer} sends {differ} of {sharded.shape[0]} tokens to "
                "other experts than the whole model"
            )
    _fail_together(problems)


def _assert_grads_match_the_whole_model(model, per_batch, rank, size, ep_degree):
    """Routed experts collect the gradient of every token routed to them, their EP group's batches;
    FSDP2-sharded dense parameters hold the gradient averaged over every rank; replicated dense
    parameters see only this rank's batch."""
    ep_rank = rank % ep_degree
    group = range(rank - ep_rank, rank - ep_rank + ep_degree)
    per_rank = NUM_EXPERTS // ep_degree
    owned = slice(ep_rank * per_rank, (ep_rank + 1) * per_rank)
    # Checked before the comparisons: full_tensor() below is a collective that a rank without the
    # gradient would skip, leaving the others waiting in it instead of failing.
    _fail_together(
        [f"{name} received no gradient" for name, p in model.named_parameters() if p.grad is None]
    )
    problems, names = [], set()
    for name, param in model.named_parameters():
        names.add(name)
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
    _assert_routes_match_the_whole_model(model, whole, mine)
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

    _assert_routes_match_the_whole_model(model, whole, mine)
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


# ------------------------------------------------------------------------------------- Kimi-VL

# A tiny Kimi-VL: a vision tower and a projector beside a DeepSeek-V3 language model with a dense first
# layer, then one MoE layer of 8 routed experts, as the checkpoint's own code builds it (remote code,
# ``kimi_vl_tiny_modeling``), its experts stored per expert.
KIMI_VL_TEXT = {
    "model_type": "deepseek_v3",
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 64,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 2,
    "first_k_dense_replace": 1,
    "n_routed_experts": NUM_EXPERTS,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "q_lora_rank": None,
    "kv_lora_rank": 16,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
}
KIMI_VL_MOE = "language_model.model.layers.1.mlp.experts"
KIMI_VL_OWN = ("vision_tower.weight", "multi_modal_projector.weight")


def _save_tiny_kimi_vl(ckpt_dir):
    """A tiny Kimi-VL-shaped remote checkpoint whose experts are stored per expert, as Kimi-VL's are."""
    ckpt_dir = Path(ckpt_dir)
    shutil.copy(kimi_vl_tiny_modeling.__file__, ckpt_dir / "modeling_kimi_vl.py")
    remote = "modeling_kimi_vl.KimiVLForConditionalGeneration"
    config = {
        "model_type": "kimi_vl",
        "architectures": ["KimiVLForConditionalGeneration"],
        "auto_map": {
            "AutoConfig": "modeling_kimi_vl.KimiVLConfig",
            "AutoModel": remote,
            "AutoModelForCausalLM": remote,
        },
        "torch_dtype": "bfloat16",
        "text_config": KIMI_VL_TEXT,
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    torch.manual_seed(0)
    language_model = DeepseekV3ForCausalLM(DeepseekV3Config(**KIMI_VL_TEXT)).to(torch.bfloat16)
    tensors = {}
    inter = KIMI_VL_TEXT["moe_intermediate_size"]
    for name, tensor in language_model.state_dict().items():
        name = f"language_model.{name}"
        if name == f"{KIMI_VL_MOE}.gate_up_proj":  # stored per expert, gate and up apart
            for e in range(NUM_EXPERTS):
                tensors[f"{KIMI_VL_MOE}.{e}.gate_proj.weight"] = tensor[e, :inter].contiguous()
                tensors[f"{KIMI_VL_MOE}.{e}.up_proj.weight"] = tensor[e, inter:].contiguous()
        elif name == f"{KIMI_VL_MOE}.down_proj":
            for e in range(NUM_EXPERTS):
                tensors[f"{KIMI_VL_MOE}.{e}.down_proj.weight"] = tensor[e].contiguous()
        else:
            tensors[name] = tensor.contiguous()
    width, hidden = 16, KIMI_VL_TEXT["hidden_size"]
    for name, shape in (
        ("vision_tower.weight", (width, width)),
        ("vision_tower.bias", (width,)),
        ("multi_modal_projector.weight", (hidden, width)),
        ("multi_modal_projector.bias", (hidden,)),
    ):
        tensors[name] = torch.randn(shape).to(torch.bfloat16)
    save_file(tensors, str(ckpt_dir / "model.safetensors"), metadata={"format": "pt"})


def _kimi_vl_expert_parallel_job(rank, size, ckpt_dir):
    device = torch.device("cuda", rank)
    stored = load_file(str(Path(ckpt_dir) / "model.safetensors"))
    per_rank = NUM_EXPERTS // size
    owned = range(rank * per_rank, (rank + 1) * per_rank)
    problems = []
    with bounded_collectives(seconds=120):
        config = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
        plan = DistributionPlan(size, ep_size=size)
        prepared = apply_distributed_support(
            "kimi_vl", copy.deepcopy(config), ckpt_path=ckpt_dir, trust_remote_code=True
        )
        model = load_and_prepare_ep(
            ckpt_dir, plan, trust_remote_code=True, hf_config=copy.deepcopy(prepared)
        ).eval()
        whole = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, config=copy.deepcopy(prepared), trust_remote_code=True, dtype=torch.bfloat16
        )
        whole = whole.to(device).eval()

        # The native language model the adapter swaps in fuses this rank's experts, and only those,
        # from the per-expert tensors as they load.
        experts = model.language_model.model.layers[1].mlp.experts
        gate_up = [
            torch.cat([stored[f"{KIMI_VL_MOE}.{e}.{part}_proj.weight"] for part in ("gate", "up")])
            for e in owned
        ]
        down = [stored[f"{KIMI_VL_MOE}.{e}.down_proj.weight"] for e in owned]
        for name, expected in (("gate_up_proj", gate_up), ("down_proj", down)):
            if not torch.equal(_local(getattr(experts, name)).cpu(), torch.stack(expected)):
                problems.append(f"{name}: rank {rank}'s experts {list(owned)} differ")
        # The vision tower and the projector, the checkpoint's own code's, load as stored.
        problems.extend(
            f"{name} differs from the checkpoint's"
            for name in KIMI_VL_OWN
            if not torch.equal(model.get_parameter(name).detach().cpu(), stored[name])
        )
        if model._modelopt_unplaced_source_keys:
            problems.append(f"unplaced: {model._modelopt_unplaced_source_keys[:3]}")

        # Each rank its own batch through the experts, against the whole model on that batch.
        generator = torch.Generator().manual_seed(rank)
        input_ids = torch.randint(0, KIMI_VL_TEXT["vocab_size"], (2, 16), generator=generator)
        with torch.no_grad():
            sharded = model(input_ids.to(device)).logits.float()
            reference = whole(input_ids.to(device)).logits.float()
        if not torch.allclose(sharded, reference, atol=2e-2, rtol=2e-2):
            problems.append(f"logits differ by {(sharded - reference).abs().max().item():.3g}")

        gathered = [None] * size
        dist.all_gather_object(gathered, problems)
    found = [problem for rank_problems in gathered for problem in rank_problems]
    assert not found, "\n".join(found[:20])


def test_kimi_vl_loads_expert_parallel_and_matches_the_whole_model(
    need_2_gpus, dist_workers, tmp_path
):
    """A Kimi-VL-shaped remote checkpoint loaded expert-parallel through its model spec: the native
    DeepSeek-V3 the adapter swaps in fuses each rank's own experts from the per-expert tensors, the
    vision tower and projector load as stored, nothing is left unplaced, and each rank's logits match
    the whole model's on one GPU. Each rank's checks are gathered before any rank fails, so a
    regression fails instead of hanging."""
    if NUM_EXPERTS % dist_workers.world_size:
        pytest.skip(f"{NUM_EXPERTS} experts do not shard over {dist_workers.world_size} GPUs")
    _save_tiny_kimi_vl(tmp_path)
    dist_workers.run(_kimi_vl_expert_parallel_job, ckpt_dir=str(tmp_path))
