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
from safetensors.torch import load_file
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


def _test_carry_over_unplaced_weights(rank, size, ckpt_dir, export_dir, orphan_prefix):
    """Load, export, and require the weights the model could not place to survive."""
    device = torch.device(f"cuda:{rank}")
    model = parallel_load_and_prepare_fsdp2(ckpt_dir, device, rank, size)

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
