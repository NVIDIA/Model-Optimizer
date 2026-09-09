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
from functools import partial

import pytest
import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from torch.distributed.tensor import DTensor

from modelopt.torch.export.unified_export_hf import export_hf_checkpoint
from modelopt.torch.utils.distributed import broadcast_state_dict
from modelopt.torch.utils.plugins.model_load_utils import parallel_load_and_prepare_fsdp2

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
