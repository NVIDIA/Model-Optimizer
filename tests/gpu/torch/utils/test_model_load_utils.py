# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GPU/distributed tests for the loads: ``load_pretrained`` by its plan, the FSDP2 load, and their
helpers."""

import json
import os
from functools import partial

import pytest
import torch
import transformers
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from packaging.version import Version
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, DeepseekV3Config, LlamaConfig, Qwen3MoeConfig

from modelopt.torch.export.unified_export_hf import export_hf_checkpoint
from modelopt.torch.utils.distributed import (
    DistributionPlan,
    broadcast_state_dict,
    distribution_plan,
    set_distribution_plan,
)
from modelopt.torch.utils.moe import _EP_MIN_TRANSFORMERS
from modelopt.torch.utils.plugins.model_load_utils import (
    load_pretrained,
    parallel_load_and_prepare_fsdp2,
)

VOCAB_SIZE = 64


def _test_broadcast_state_dict_roundtrip(rank, size):
    """Round-trip from every rank as source, with a distinct payload per source rank."""
    device = torch.device(f"cuda:{rank}")
    for source in range(size):
        src_dict = {
            "w": torch.full((2, 4), float(source)),
            "b": torch.tensor([float(source), float(source) + 1.0]),
        }
        out = broadcast_state_dict(src_dict if rank == source else None, src=source, device=device)
        assert set(out.keys()) == {"w", "b"}
        assert out["w"].device == device
        assert torch.equal(out["w"].cpu(), src_dict["w"])
        assert torch.equal(out["b"].cpu(), src_dict["b"])


def test_broadcast_state_dict_roundtrip(dist_workers):
    dist_workers.run(_test_broadcast_state_dict_roundtrip)


def _test_parallel_load_and_export(rank, size, ckpt_dir, export_dir, cpu_offload):
    """Load a tiny Llama via the FSDP2 loader, forward, then export."""
    device = torch.device(f"cuda:{rank}")
    model = parallel_load_and_prepare_fsdp2(
        ckpt_dir,
        device,
        rank,
        size,
        cpu_offload=cpu_offload,
    )
    set_distribution_plan(model, DistributionPlan(size, dense="fsdp"))

    # Decoder layers AND root params (embed/lm_head) are sharded DTensors under shard_root=True.
    decoder_params = list(model.model.layers[0].parameters())
    assert any(isinstance(p, DTensor) for p in decoder_params)
    assert isinstance(model.model.embed_tokens.weight, DTensor)
    if not cpu_offload:
        # Non-offload: the root's local shard lives on GPU.
        assert model.model.embed_tokens.weight.to_local().device.type == "cuda"
    if cpu_offload:
        # Under cpu_offload the decoder shards live on CPU between forwards.
        decoder_dtensors = [p for p in decoder_params if isinstance(p, DTensor)]
        assert all(p.to_local().device.type == "cpu" for p in decoder_dtensors)

    # Forward exercises FSDP2 hooks + (under cpu_offload) the per-layer CPU↔GPU stream.
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8), device=device)
    out = model(input_ids=input_ids).logits
    assert out.shape == (1, 8, VOCAB_SIZE)

    # Export and verify the saved config.json retains the original architectures.
    export_hf_checkpoint(model, export_dir=export_dir, dtype=torch.bfloat16)

    if rank == 0:
        with open(os.path.join(export_dir, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["architectures"] == ["LlamaForCausalLM"]


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_parallel_load_and_export(dist_workers, tmp_path, cpu_offload):
    # Build the checkpoint once here (not inside the workers): every rank must see the same path.
    ckpt_dir = create_tiny_llama_dir(tmp_path, vocab_size=VOCAB_SIZE)
    dist_workers.run(
        partial(
            _test_parallel_load_and_export,
            ckpt_dir=str(ckpt_dir),
            export_dir=str(tmp_path / "export"),
            cpu_offload=cpu_offload,
        )
    )


def _test_carry_over_unplaced_weights(rank, size, ckpt_dir, export_dir, orphan_prefix):
    """Load, export, and require the weights the model could not place to survive."""
    device = torch.device(f"cuda:{rank}")
    model = parallel_load_and_prepare_fsdp2(ckpt_dir, device, rank, size)
    set_distribution_plan(model, DistributionPlan(size, dense="fsdp"))

    unplaced = getattr(model, "_modelopt_unplaced_source_keys", None)
    assert unplaced, (
        "the loader placed every checkpoint key; the fixture was supposed to leave an orphaned "
        "layer behind, so this test would pass vacuously"
    )
    assert any(k.startswith(orphan_prefix) for k in unplaced), (
        f"orphaned '{orphan_prefix}*' weights were not recorded as unplaced; got {sorted(unplaced)[:5]}"
    )

    export_hf_checkpoint(model, export_dir=export_dir, dtype=torch.bfloat16)

    if rank == 0:
        exported: dict = {}
        for shard in sorted(os.listdir(export_dir)):
            if shard.endswith(".safetensors"):
                exported.update(load_file(os.path.join(export_dir, shard)))
        source: dict = {}
        for shard in sorted(os.listdir(ckpt_dir)):
            if shard.endswith(".safetensors"):
                source.update(load_file(os.path.join(ckpt_dir, shard)))

        orphans = sorted(k for k in source if k.startswith(orphan_prefix))
        assert orphans, "fixture produced no orphaned weights"
        for k in orphans:
            assert k in exported, (
                f"'{k}' is in the source checkpoint and was never loaded into the model, so PTQ "
                f"could not touch it -- it must be copied into the export, but it is missing"
            )
            assert torch.equal(exported[k].cpu(), source[k].cpu()), (
                f"'{k}' was carried over but its value changed; it should be a verbatim copy"
            )


def test_carry_over_unplaced_weights(dist_workers, tmp_path):
    """Weights the built model has no home for must still reach the exported checkpoint.

    A checkpoint can carry parameters the model class does not build -- an MTP head is the common
    case (HF builds only ``num_hidden_layers`` decoders, leaving an inlined MTP tail orphaned), but
    an auxiliary tower or draft head behaves the same way. Quantization never sees them, so nothing
    downstream would notice their absence: the export just comes out quietly incomplete.

    The fixture reproduces that shape directly. Build a checkpoint with one more layer than the
    config admits, so the final layer's weights are present on disk with nowhere to load to.
    """
    ckpt_dir = create_tiny_llama_dir(tmp_path, vocab_size=VOCAB_SIZE, num_hidden_layers=3)

    # Truncate the config so the last layer becomes unplaceable -- the same situation an inlined
    # MTP tail creates, without needing an MTP-capable architecture in the test suite.
    cfg_path = os.path.join(str(ckpt_dir), "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    orphan_idx = cfg["num_hidden_layers"] - 1
    cfg["num_hidden_layers"] = orphan_idx
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    dist_workers.run(
        partial(
            _test_carry_over_unplaced_weights,
            ckpt_dir=str(ckpt_dir),
            export_dir=str(tmp_path / "export_carry"),
            orphan_prefix=f"model.layers.{orphan_idx}.",
        )
    )


# ------------------------------------------------------------------ load_pretrained, left to decide

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
    """Load on every rank of the run with the plan left to decide, and check what each rank holds.

    Left to decide, ``load_pretrained`` shards an expert-parallel MoE's experts, FSDP2-shards a dense
    model when asked to, and otherwise gives every rank the whole model on its own GPU. Whichever
    load runs, every rank ends up with the checkpoint's weights and the load's record of what it
    could not place. An expert-parallel model also computes what the whole model does: each rank's
    tokens reach the ranks holding their experts and come back.
    """
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
        assert (
            distribution_plan(model)
            == {
                "replicated": DistributionPlan(size),
                "fsdp2": DistributionPlan(size, dense="fsdp"),
                "expert-parallel": DistributionPlan(size, ep_size=size),
            }[expected]
        )
        if expected == "expert-parallel":
            # Every rank sends its own batch through the experts, as calibration does.
            whole.to(device)
            generator = torch.Generator().manual_seed(rank)
            input_ids = torch.randint(0, _TINY["vocab_size"], (2, 16), generator=generator)
            with torch.no_grad():
                sharded = model(input_ids.to(device)).logits
                reference = whole(input_ids.to(device)).logits
            torch.testing.assert_close(sharded.float(), reference.float(), atol=2e-2, rtol=2e-2)
    finally:
        for knob in knobs:
            del os.environ[knob]


def test_a_dense_model_loads_whole_on_every_rank(need_2_gpus, dist_workers, tmp_path):
    _save(LlamaConfig(**_TINY), tmp_path)
    dist_workers.run(_load_and_check, str(tmp_path), {}, "replicated")


def test_a_dense_model_asked_to_shard_loads_fsdp2_sharded(need_2_gpus, dist_workers, tmp_path):
    _save(LlamaConfig(**_TINY), tmp_path)
    dist_workers.run(_load_and_check, str(tmp_path), {"MODELOPT_FSDP_SIZE": "world"}, "fsdp2")


def _qwen3_moe():
    """Qwen3-MoE, whose config ships its own expert-parallel plan from transformers 5.7."""
    config = Qwen3MoeConfig(moe_intermediate_size=32, num_experts=NUM_EXPERTS, **_TINY)
    config.num_experts_per_tok = 2
    return config


def _deepseek_v3():
    """DeepSeek-V3 with no plan of its own, as before transformers 5.13, so only its model spec
    makes it expert-parallel. Every layer is an MoE layer, its attention shrunk to the tiny width."""
    config = DeepseekV3Config(
        n_routed_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        first_k_dense_replace=0,
        n_group=1,
        topk_group=1,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        **_TINY,
    )
    config.base_model_ep_plan = None  # saved in config.json: no plan of its own on any transformers
    return config


@pytest.mark.skipif(
    Version(transformers.__version__) < Version(_EP_MIN_TRANSFORMERS),
    reason=f"expert parallelism needs transformers >= {_EP_MIN_TRANSFORMERS}",
)
@pytest.mark.parametrize(
    "config", [_qwen3_moe, _deepseek_v3], ids=["its-own-plan", "its-spec-plan"]
)
def test_an_expert_parallel_moe_loads_with_its_experts_sharded(
    need_2_gpus, dist_workers, tmp_path, config
):
    """Whether the plan comes with the model's config or only from its model spec."""
    if NUM_EXPERTS % dist_workers.world_size:
        pytest.skip(f"{NUM_EXPERTS} experts do not shard over {dist_workers.world_size} GPUs")
    _save(config(), tmp_path)
    dist_workers.run(_load_and_check, str(tmp_path), {}, "expert-parallel")
