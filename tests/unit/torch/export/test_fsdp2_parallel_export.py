# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""World=1 (single-process, CPU) tests for the FSDP2 streaming export.

The pipeline is world-agnostic: with ``torch.distributed`` uninitialized, ``rank/size`` are
``0/1``, the unshard window is a no-op, and rank 0 owns every unit. So these cover unit
enumeration, per-tensor export, the shard writer, index naming and tied-weight dropping without a
GPU. The multi-rank unshard collective needs real FSDP2 and is covered in ``tests/gpu``.
"""

import copy
import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_hf import _export_transformers_checkpoint
from modelopt.torch.export.unified_export_hf_streaming import _export_fsdp2_checkpoint_streaming

transformers = pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM, LlamaConfig


def _tiny_quantized_llama(quant_cfg=None, tie=True):
    mto.enable_huggingface_checkpointing()
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        tie_word_embeddings=tie,
        architectures=["LlamaForCausalLM"],
    )
    model = AutoModelForCausalLM.from_config(cfg).eval()
    calib = [torch.randint(0, 128, (1, 8)) for _ in range(4)]
    return mtq.quantize(model, quant_cfg or mtq.FP8_DEFAULT_CFG, lambda m: [m(x) for x in calib])


def _load_all(export_dir: Path) -> dict:
    index = export_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        out: dict = {}
        for fname in set(weight_map.values()):
            out.update(load_file(str(export_dir / fname)))
        return out
    return load_file(str(export_dir / "model.safetensors"))


# --------------------------------------------------------------------------- #
# End-to-end streaming export (world=1)
# --------------------------------------------------------------------------- #
def test_streaming_export_matches_resident_path():
    """Streaming export produces the same tensors as the whole-state-dict path."""
    model = _tiny_quantized_llama()
    ref_model = copy.deepcopy(model)  # export packs weights in place -> need two copies

    d = Path(tempfile.mkdtemp())
    _export_fsdp2_checkpoint_streaming(model, torch.bfloat16, export_dir=d)
    streamed = _load_all(d)

    ref, _ = _export_transformers_checkpoint(ref_model, torch.bfloat16)

    assert set(streamed) == set(ref)
    for key in ref:
        assert torch.equal(streamed[key].float(), ref[key].cpu().float()), key


def test_streaming_export_drops_tied_alias_and_writes_config():
    model = _tiny_quantized_llama()
    d = Path(tempfile.mkdtemp())
    _export_fsdp2_checkpoint_streaming(model, torch.bfloat16, export_dir=d)

    loaded = _load_all(d)
    assert "lm_head.weight" not in loaded  # tied alias dropped
    assert any("weight_scale" in key for key in loaded)  # quant scales exported
    assert (d / "config.json").exists()
    assert (d / "generation_config.json").exists()
    assert not list(d.glob("__shard_part*"))


def test_streaming_export_subsplits_by_max_shard_size():
    """A tiny max_shard_size forces several shard files; the index still maps every key."""
    model = _tiny_quantized_llama()
    d = Path(tempfile.mkdtemp())
    _export_fsdp2_checkpoint_streaming(model, torch.bfloat16, export_dir=d, max_shard_size=2048)

    index = json.loads((d / "model.safetensors.index.json").read_text())
    assert len(set(index["weight_map"].values())) > 1
    loaded = _load_all(d)
    assert set(loaded) == set(index["weight_map"])
    assert index["metadata"]["total_size"] > 0


def test_streaming_export_extra_state_dict_mtp():
    """extra_state_dict (MTP-style tensors the model never holds) is written by rank 0."""
    model = _tiny_quantized_llama()
    extra = {"model.layers.99.mtp.weight": torch.randn(8, 8, dtype=torch.bfloat16)}
    d = Path(tempfile.mkdtemp())
    _export_fsdp2_checkpoint_streaming(model, torch.bfloat16, export_dir=d, extra_state_dict=extra)

    loaded = _load_all(d)
    assert "model.layers.99.mtp.weight" in loaded
    assert torch.equal(loaded["model.layers.99.mtp.weight"], extra["model.layers.99.mtp.weight"])
