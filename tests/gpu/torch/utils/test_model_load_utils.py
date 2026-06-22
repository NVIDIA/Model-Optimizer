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

"""GPU/distributed tests for the FSDP2 load path and its helpers."""

import json
import os
import tempfile
from functools import partial

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def _test_broadcast_state_dict_roundtrip(rank, size):
    """Round-trip from every rank as source (matches the per-layer rotation in the loader)."""
    from modelopt.torch.utils.distributed import broadcast_state_dict

    device = torch.device(f"cuda:{rank}")
    # Distinct payload per source rank so a wrong-src result would fail content checks.
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


def _build_tiny_llama_checkpoint(path: str) -> None:
    """Write a tiny LlamaForCausalLM checkpoint (config + safetensors) to ``path``."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        torch_dtype="bfloat16",
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(path)


def _test_parallel_load_and_export(rank, size, cpu_offload):
    """Load a tiny Llama via the FSDP2 loader, forward, then export — config.architectures preserved.

    Parametrized over ``cpu_offload`` to cover both shard placements:
      - off: decoder DTensor shards on GPU, plain root on GPU.
      - on:  decoder DTensor shards on CPU (streamed per layer), root promoted to GPU
             via ``_promote_non_dtensor_to_gpu``.
    """
    from modelopt.torch.export.unified_export_hf import export_hf_checkpoint
    from modelopt.torch.utils.plugins.model_load_utils import parallel_load_and_prepare_fsdp2

    suffix = "offload" if cpu_offload else "noffload"
    ckpt_dir = os.path.join(tempfile.gettempdir(), f"_test_parallel_load_{suffix}_{os.getpid()}")
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        _build_tiny_llama_checkpoint(ckpt_dir)
    dist.barrier()

    device = torch.device(f"cuda:{rank}")
    model = parallel_load_and_prepare_fsdp2(
        ckpt_dir,
        device,
        rank,
        size,
        cpu_offload=cpu_offload,
        freeze=True,
    )

    # Decoder layers are sharded; root params (embed/lm_head) are plain on GPU.
    decoder_params = list(model.model.layers[0].parameters())
    assert any(isinstance(p, DTensor) for p in decoder_params)
    assert not isinstance(model.model.embed_tokens.weight, DTensor)
    assert model.model.embed_tokens.weight.device.type == "cuda"
    if cpu_offload:
        # Under cpu_offload the decoder shards live on CPU between forwards.
        decoder_dtensors = [p for p in decoder_params if isinstance(p, DTensor)]
        assert all(p.to_local().device.type == "cpu" for p in decoder_dtensors)

    # Forward exercises FSDP2 hooks + (under cpu_offload) the per-layer CPU↔GPU stream.
    input_ids = torch.randint(0, 64, (1, 8), device=device)
    out = model(input_ids=input_ids).logits
    assert out.shape == (1, 8, 64)

    # Export and verify the saved config.json retains the original architectures.
    export_dir = os.path.join(
        tempfile.gettempdir(), f"_test_parallel_export_{suffix}_{os.getpid()}"
    )
    if rank == 0:
        os.makedirs(export_dir, exist_ok=True)
    dist.barrier()
    export_hf_checkpoint(model, export_dir=export_dir, dtype=torch.bfloat16)

    if rank == 0:
        with open(os.path.join(export_dir, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["architectures"] == ["LlamaForCausalLM"]


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_parallel_load_and_export(dist_workers, cpu_offload):
    dist_workers.run(partial(_test_parallel_load_and_export, cpu_offload=cpu_offload))
