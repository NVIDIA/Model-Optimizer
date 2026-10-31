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
from _test_utils.torch import kimi_k3_tiny_modeling, kimi_vl_tiny_modeling
from torch.distributed.tensor import DTensor
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    Qwen3MoeConfig,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_aware_conversion import build_reverse_name_mapper
from modelopt.torch.models import _onfly_decompress
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.utils import moe
from modelopt.torch.utils.distributed import (
    DistributionPlan,
    ParallelState,
    distributed_forward_loop,
    distribution_plan,
    set_distribution_plan,
)
from modelopt.torch.utils.moe import apply_distributed_support, expert_parallel_mesh
from modelopt.torch.utils.plugins.model_load_utils import load_and_prepare_ep, load_pretrained

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
        minversion="0.16",
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


# ------------------------------------------------------------------------------------- Kimi-K3

# A tiny Kimi-K3: a dense first layer, then one MoE layer of 8 routed experts, run in a 32-wide latent
# space, as Kimi-K3's own code builds it (remote code, ``kimi_k3_tiny_modeling``).
KIMI_K3_DIMS = {
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 64,
    "routed_expert_hidden_size": 32,
    "moe_intermediate_size": 32,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "num_hidden_layers": 2,
    "first_k_dense_replace": 1,
    "num_shared_experts": 1,
}
KIMI_K3_EXPERTS = KIMI_K3_DIMS["num_experts"]
KIMI_K3_MOE = "language_model.model.layers.1.block_sparse_moe.experts."
# Kimi-K3's own compression: MXFP4 routed experts, everything else bf16.
KIMI_K3_MXFP4 = {
    "quant_method": "compressed-tensors",
    "format": "mxfp4-pack-quantized",
    "quantization_status": "compressed",
    "config_groups": {
        "group_0": {
            "format": "mxfp4-pack-quantized",
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "float",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
                "scale_dtype": "torch.uint8",
                "dynamic": False,
                "observer": "minmax",
            },
        }
    },
    "ignore": [
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
        "re:.*lm_head.*",
    ],
}


def _kimi_k3_expert(expert):
    """Expert ``expert``'s ``w1``/``w3``/``w2``, the same on every rank."""
    gen = torch.Generator().manual_seed(expert)

    def draw(rows, cols):
        # MXFP4 (e2m1) values at a scale of 1/16: packed at that power-of-two scale, they decompress
        # exactly, and bf16 holds them exactly.
        values = _E2M1[torch.randint(0, len(_E2M1), (rows, cols), generator=gen)] / 16
        return values * torch.where(torch.rand(rows, cols, generator=gen) < 0.5, -1.0, 1.0)

    inter, latent = KIMI_K3_DIMS["moe_intermediate_size"], KIMI_K3_DIMS["routed_expert_hidden_size"]
    return {"w1": draw(inter, latent), "w3": draw(inter, latent), "w2": draw(latent, inter)}


def _save_tiny_kimi_k3(ckpt_dir):
    """A tiny Kimi-K3-shaped remote checkpoint whose routed experts are packed MXFP4, as Kimi-K3's are."""
    ckpt_dir = Path(ckpt_dir)
    shutil.copy(kimi_k3_tiny_modeling.__file__, ckpt_dir / "modeling_kimi_k3.py")
    remote = "modeling_kimi_k3.KimiK3ForConditionalGeneration"
    config = {
        "model_type": "kimi_k3",
        "architectures": ["KimiK3ForConditionalGeneration"],
        "auto_map": {
            "AutoConfig": "modeling_kimi_k3.KimiK3Config",
            "AutoModel": remote,
            "AutoModelForCausalLM": remote,
        },
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
        # As Kimi-K3's: the scheme sits on the language model's config only.
        "text_config": {
            "model_type": "kimi_linear",
            **KIMI_K3_DIMS,
            "quantization_config": KIMI_K3_MXFP4,
        },
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config))

    compressor, scheme, stored_as, _ = _onfly_decompress._read_compression(
        SimpleNamespace(quantization_config=KIMI_K3_MXFP4)
    )
    with torch.device("meta"):
        shapes = kimi_k3_tiny_modeling.KimiK3ForConditionalGeneration(
            kimi_k3_tiny_modeling.KimiK3Config(text_config=dict(KIMI_K3_DIMS))
        ).state_dict()
    gen = torch.Generator().manual_seed(0)
    tensors = {}
    for name, meta in shapes.items():
        if name.startswith(KIMI_K3_MOE):
            continue
        dense = torch.randn(meta.shape, generator=gen) * 0.1
        tensors[name] = (1 + dense if name.endswith("norm.weight") else dense).to(torch.bfloat16)
    for e in range(KIMI_K3_EXPERTS):
        for leaf, weight in _kimi_k3_expert(e).items():
            scale = torch.full((weight.shape[0], weight.shape[1] // 32), 1 / 16)
            packed = compressor.compress({"weight": weight, "weight_scale": scale}, scheme)
            for stored in stored_as:
                tensors[f"{KIMI_K3_MOE}{e}.{leaf}.{stored}"] = packed[stored]
    save_file(tensors, str(ckpt_dir / "model.safetensors"), metadata={"format": "pt"})


def _whole_kimi_k3(ckpt_dir, config, device):
    """The single-rank load: the whole model on one device, its experts decompressed as it loads."""
    prepared = apply_distributed_support(
        "kimi_k3", copy.deepcopy(config), ckpt_path=ckpt_dir, trust_remote_code=True
    )
    whole = AutoModelForCausalLM.from_pretrained(
        ckpt_dir, config=prepared, trust_remote_code=True, dtype=torch.bfloat16
    )
    for module in whole.modules():  # the experts forward reads it at forward time
        if hasattr(getattr(module, "config", None), "_experts_implementation"):
            module.config._experts_implementation = "eager"
    return whole.to(device).eval()


def _calibrated(model):
    return {
        name: module.amax.float().cpu()
        for name, module in model.named_modules()
        if isinstance(module, TensorQuantizer) and module.is_enabled and module.amax is not None
    }


def _kimi_k3_expert_parallel_job(rank, size, ckpt_dir):
    device = torch.device("cuda", rank)
    batches = [
        torch.randint(
            0, KIMI_K3_DIMS["vocab_size"], (2, 16), generator=torch.Generator().manual_seed(r)
        )
        for r in range(size)
    ]
    per_rank = KIMI_K3_EXPERTS // size
    owned = range(rank * per_rank, (rank + 1) * per_rank)
    problems = []
    # The decompressor's hooks are process-wide, and the workers outlive the job.
    hooks = (mu.get_model_conversion_mapping, cml.spawn_materialize)
    try:
        with bounded_collectives(seconds=120):
            config = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
            plan = DistributionPlan(size, ep_size=size)
            prepared = apply_distributed_support(
                "kimi_k3", copy.deepcopy(config), ckpt_path=ckpt_dir, trust_remote_code=True
            )
            model = load_and_prepare_ep(
                ckpt_dir, plan, trust_remote_code=True, hf_config=prepared
            ).eval()
            set_distribution_plan(model, plan)
            whole = _whole_kimi_k3(ckpt_dir, config, device)
            for module in whole.modules():
                module._parallel_state = ParallelState(data_parallel_group=-1)

            # This rank's experts, decompressed as they loaded, and only those.
            experts = model.language_model.model.layers[1].block_sparse_moe.experts
            want = [_kimi_k3_expert(e) for e in owned]
            gate_up = torch.stack([torch.cat([w["w1"], w["w3"]]) for w in want])
            down = torch.stack([w["w2"] for w in want])
            for name, expected in (("gate_up_proj", gate_up), ("down_proj", down)):
                local = _local(getattr(experts, name)).float().cpu()
                if not torch.equal(local, expected):
                    problems.append(f"{name}: rank {rank}'s experts {list(owned)} differ")

            # Each rank its own batch through the experts, against the whole model on that batch.
            with torch.no_grad():
                sharded = model(batches[rank].to(device)).logits.float()
                reference = whole(batches[rank].to(device)).logits.float()
            if not torch.allclose(sharded, reference, atol=2e-2, rtol=2e-2):
                problems.append(f"logits differ by {(sharded - reference).abs().max().item():.3g}")

            # Calibrated over every rank's batch: each rank on its own, the whole model on all.
            forward_loop = distributed_forward_loop(
                model,
                [{"input_ids": batches[rank]}],
                device=device,
                ep_mesh=expert_parallel_mesh(model),
            )
            mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)
            mtq.quantize(whole, mtq.FP8_DEFAULT_CFG, lambda m: [m(b.to(device)) for b in batches])
            sharded_amax, reference_amax = _calibrated(model), _calibrated(whole)
            if not any("_weight_quantizers." in name for name in sharded_amax):
                problems.append("no expert weight quantizer was calibrated")
            for name, amax in sharded_amax.items():
                if "_weight_quantizers." in name:  # this rank's local expert -> the global one
                    prefix, index = name.rsplit(".", 1)
                    name = f"{prefix}.{rank * per_rank + int(index)}"
                expected_amax = reference_amax.get(name)
                if expected_amax is None:
                    problems.append(f"{name} is calibrated here but not in the whole model")
                    continue
                # bf16 activations, summed in another order: compare by relative norm.
                error = (amax - expected_amax).norm() / expected_amax.norm().clamp_min(1e-12)
                if not error < 2e-2:
                    problems.append(f"{name}: amax off by {error.item():.3g} (relative)")

            # The export names each expert back to the checkpoint's w1/w3/w2.
            to_checkpoint = build_reverse_name_mapper(model, tensor_keys=True)
            for e in owned:
                parts = [
                    f"{KIMI_K3_MOE}{e}.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")
                ]
                leaves = [f"{KIMI_K3_MOE}{e}.{leaf}.weight" for leaf in ("w1", "w3", "w2")]
                if [to_checkpoint(part) for part in parts] != leaves:
                    problems.append(f"expert {e} exports as {[to_checkpoint(p) for p in parts]}")

            # The returned model's config has lost its scheme and the decompressor's mark. Passed
            # to another load of the checkpoint, it is refused, not read as bf16 experts.
            try:
                load_pretrained(
                    ckpt_dir, config=model.config, distribution=plan, trust_remote_code=True
                )
                problems.append("a load with the returned model.config was not refused")
            except ValueError as e:
                if "no compressed-tensors quantization_config" not in str(e):
                    problems.append(f"a load with the returned model.config failed otherwise: {e}")
            # Nor may a whole-model load of it in this process, through a class the sharded loads
            # refuse: prepare patched Kimi-K3's MoE block for every later load, so it is refused
            # before anything is built.
            try:
                load_pretrained(
                    ckpt_dir,
                    auto_class=AutoModel,
                    config=model.config,
                    distribution=DistributionPlan(size, 1, "replicate"),
                    trust_remote_code=True,
                )
                problems.append("a whole-model load after the expert-parallel one was not refused")
            except NotImplementedError as e:
                if "patched its modeling code" not in str(e):
                    problems.append(f"a whole-model load after it failed otherwise: {e}")

            gathered = [None] * size
            dist.all_gather_object(gathered, problems)
    finally:
        mu.get_model_conversion_mapping, cml.spawn_materialize = hooks
        moe._PATCHED_EVERY_LOAD.discard("kimi_k3")  # the workers outlive the job
    found = [problem for rank_problems in gathered for problem in rank_problems]
    assert not found, "\n".join(found[:20])


def test_kimi_k3_packed_experts_load_calibrate_and_name_back_expert_parallel(
    need_2_gpus, dist_workers, tmp_path
):
    """A Kimi-K3-shaped remote checkpoint with packed MXFP4 experts, loaded expert-parallel through
    its model spec: each rank decompresses only its own experts, the sharded model matches the whole
    one loaded on a single GPU, calibration over every rank's batch matches the whole model's, and
    the export names the experts back to the checkpoint's ``w1``/``w3``/``w2``. Each rank's
    checks are gathered before any rank fails, so a regression fails instead of hanging. Passed to
    another load, the returned model's config is refused rather than read as bf16 experts."""
    pytest.importorskip(
        "compressed_tensors",
        minversion="0.16",
        reason="compressed-tensors is needed: the checkpoint's experts are compressed",
    )
    if Version(transformers.__version__) < Version("5.16"):
        pytest.skip("the torch_a2a forward wrappers are pre-hooks below transformers 5.16")
    if KIMI_K3_EXPERTS % dist_workers.world_size:
        pytest.skip(f"{KIMI_K3_EXPERTS} experts do not shard over {dist_workers.world_size} GPUs")
    _save_tiny_kimi_k3(tmp_path)
    dist_workers.run(_kimi_k3_expert_parallel_job, ckpt_dir=str(tmp_path))
