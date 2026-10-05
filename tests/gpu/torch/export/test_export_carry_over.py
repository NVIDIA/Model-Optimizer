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
"""Carry-over of source weights the model could not place, end to end under FSDP2.

A checkpoint can hold parameters the built model has no home for -- an MTP head, or a vision
tower that a language-model-only PTQ never touches. Quantization never sees those, so without an
explicit carry-over they are silently absent from the export: no error, plausible file sizes, a
checkpoint simply missing a component.

Asserted through the public ``export_hf_checkpoint`` API, so these hold for whichever FSDP2 write
path it selects:

1. every source key the model did not place is present in the export, with its source values;
2. a VLM whose language model alone was quantized keeps its vision tower -- unquantized, at full
   shape, and carrying no stray scales.
"""

import json
from functools import partial
from pathlib import Path

import pytest
import torch
from _test_utils.torch.transformers_models import create_tiny_qwen3_dir, create_tiny_qwen3vl_dir
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_hf import export_hf_checkpoint
from modelopt.torch.quantization.utils import patch_fsdp_mp_dtypes
from modelopt.torch.utils.distributed import fsdp2_wrap, is_fsdp2_model

# Small enough to force the tiny model across several shards, so the multi-file layout -- and the
# weight index that makes it loadable -- are exercised rather than collapsing to one model.safetensors.
# Not smaller: each extra shard is another write + consolidation round trip on a toy model.
MAX_SHARD_SIZE = "512KB"

# The default tiny Qwen3 is too degenerate to shard: head_dim = hidden_size / num_heads = 32/16 = 2,
# so q_norm/k_norm are 2-element tensors. Split over a 4-rank world most ranks get an empty chunk and
# the write planner rejects the coverage outright ("invalid fill tensor-volume"), taking the whole
# worker pool down with it. Size every FSDP2-sharded axis to stay comfortably divisible instead --
# still a toy model, but one that survives a multi-rank split. Last dims are multiples of 16 so NVFP4
# block quantization is also well defined.
TINY_KWARGS = {
    "hidden_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "intermediate_size": 128,
    "max_position_embeddings": 64,
    "num_hidden_layers": 2,
}
TINY_MOE_KWARGS = {"moe_intermediate_size": 128, "num_experts": 8, "num_experts_per_tok": 2}

# Suffixes the exporter ADDS to a checkpoint; every other exported key must correspond to a
# parameter of the source model.
_SCALE_SUFFIXES = (
    "weight_scale",
    "weight_scale_2",
    "weight_scale_inv",
    "input_scale",
    "pre_quant_scale",
    "k_scale",
    "v_scale",
)

# Exported dtypes that mean "this weight was quantized" (NVFP4 packs two fp4 per uint8).
_QUANTIZED_DTYPES = {"F8_E4M3", "U8", "I8"}


def _safetensors_meta(directory: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """``{tensor_name: (dtype, shape)}`` across every ``*.safetensors`` in ``directory``.

    Reads the safetensors header directly (8-byte little-endian length, then JSON) so the check
    depends only on what is on disk -- no loader, no dequantization, no GPU.
    """
    out: dict[str, tuple[str, tuple[int, ...]]] = {}
    for path in sorted(directory.glob("*.safetensors")):
        with open(path, "rb") as fh:
            header = json.loads(fh.read(int.from_bytes(fh.read(8), "little")))
        for name, spec in header.items():
            if name != "__metadata__":
                out[name] = (spec["dtype"], tuple(spec["shape"]))
    return out


def _is_scale(key: str) -> bool:
    return key.endswith(_SCALE_SUFFIXES)


def _ptq_and_export(rank, size, *, src_dir, export_dir, quant_cfg, **export_kwargs):
    """Load the tiny model on every rank, FSDP2-shard it, PTQ it, and export."""

    with patch_fsdp_mp_dtypes():
        model = AutoModelForCausalLM.from_pretrained(src_dir, dtype=torch.bfloat16).to("cuda")
        model.eval()

        fsdp2_wrap(model)
        # The FSDP2 export path is selected by exactly this predicate, so assert it here: without it
        # the export would silently fall back to the rank-0 gather and the test would pass while
        # never touching the code under test.
        assert is_fsdp2_model(model), "fsdp2_wrap did not shard the model"
        assert torch.distributed.is_initialized()
        torch.distributed.barrier()

        input_ids = torch.randint(0, model.config.vocab_size, (2, 8), device="cuda")
        mtq.quantize(model, quant_cfg, lambda m: m(input_ids))
        torch.distributed.barrier()

        export_hf_checkpoint(
            model, export_dir=export_dir, max_shard_size=MAX_SHARD_SIZE, **export_kwargs
        )
        torch.distributed.barrier()


@pytest.mark.timeout(600)
def test_fsdp2_distributed_export_carries_unplaced_weights_from_provenance(dist_workers, tmp_path):
    """Unplaced source weights are carried over even when nothing was recorded at load time.

    ``parallel_load_and_prepare_fsdp2`` records the checkpoint keys it could not place, but a plain
    ``from_pretrained`` -- which is how most callers build the model, and what this suite uses --
    drops unexpected keys silently and records nothing. The export then has to ask the question
    itself, from the model's own provenance (``config._name_or_path``), or the checkpoint comes out
    missing weights that PTQ never had the chance to touch.

    Same fixture shape as the loader-side test: a checkpoint holding one more layer than the config
    admits, so the last layer is present on disk with nowhere to load -- what an inlined MTP tail
    does, without needing an MTP-capable architecture here.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")

    src_dir = Path(
        create_tiny_qwen3_dir(
            tmp_path, with_tokenizer=True, **{**TINY_KWARGS, "num_hidden_layers": 3}
        )
    )
    cfg_path = src_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    orphan_idx = cfg["num_hidden_layers"] - 1
    cfg["num_hidden_layers"] = orphan_idx
    # Qwen3 validates that layer_types matches num_hidden_layers, so trim it alongside; leaving it
    # at the original length fails config construction before the model is ever built.
    if isinstance(cfg.get("layer_types"), list):
        cfg["layer_types"] = cfg["layer_types"][:orphan_idx]
    cfg_path.write_text(json.dumps(cfg))
    orphan_prefix = f"model.layers.{orphan_idx}."

    src_meta = _safetensors_meta(src_dir)
    orphans = sorted(k for k in src_meta if k.startswith(orphan_prefix))
    assert orphans, "fixture produced no orphaned weights, so this test would prove nothing"

    export_dir = tmp_path / "export_carry_provenance"
    dist_workers.run(
        partial(
            _ptq_and_export,
            src_dir=src_dir,
            export_dir=export_dir,
            quant_cfg=mtq.FP8_DEFAULT_CFG,
        )
    )

    exported = _safetensors_meta(export_dir)
    assert exported, "nothing exported"
    missing = [k for k in orphans if k not in exported]
    assert not missing, (
        f"{len(missing)} source weight(s) the model never loaded are absent from the export "
        f"(e.g. {missing[0]}); PTQ could not touch them, so they must be copied through"
    )

    merged: dict = {}
    for shard in sorted(export_dir.glob("*.safetensors")):
        merged.update(load_file(str(shard)))
    source: dict = {}
    for shard in sorted(src_dir.glob("*.safetensors")):
        source.update(load_file(str(shard)))
    for k in orphans:
        assert torch.equal(merged[k].cpu(), source[k].cpu()), (
            f"'{k}' was carried over but its value changed; it should be a verbatim copy"
        )


def _ptq_language_model_and_export(rank, size, *, src_dir, export_dir):
    """Quantize ONLY the VLM's language model, then export the whole model."""

    with patch_fsdp_mp_dtypes():
        model = AutoModelForImageTextToText.from_pretrained(src_dir, dtype=torch.bfloat16).to(
            "cuda"
        )
        model.eval()
        fsdp2_wrap(model)
        assert is_fsdp2_model(model), "fsdp2_wrap did not shard the VLM"
        torch.distributed.barrier()

        language_model = getattr(model.model, "language_model", None)
        assert language_model is not None, "fixture has no model.language_model to target"

        # A hardcoded token bound overruns a tiny fixture's embedding and surfaces as an opaque
        # device-side assert, so take the vocab from the config.
        text_config = getattr(model.config, "text_config", None)
        vocab = getattr(text_config, "vocab_size", None) or model.config.vocab_size
        input_ids = torch.randint(0, int(vocab), (1, 8), device="cuda")
        mtq.quantize(language_model, mtq.FP8_DEFAULT_CFG, lambda m: model(input_ids=input_ids))
        torch.distributed.barrier()

        export_hf_checkpoint(model, export_dir=export_dir, max_shard_size=MAX_SHARD_SIZE)
        torch.distributed.barrier()


@pytest.mark.timeout(600)
def test_fsdp2_distributed_export_of_vlm_keeps_vision_tower(dist_workers, tmp_path):
    """Quantizing only a VLM's language model must still export the COMPLETE model.

    This is the common VLM recipe: the vision tower is left alone and only the language model is
    quantized. The exported checkpoint is nonetheless expected to be the whole model -- a checkpoint
    holding just the language half loads as a broken VLM rather than failing outright, so nothing
    downstream would flag it.

    The writer builds its state dict from the model it is handed, so the risk is not that the vision
    tower is dropped but that the paths keyed on quantizer state -- the expert split, the scale
    postprocess, the reverse conversion -- mishandle a subtree that has none.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")

    src_dir = Path(create_tiny_qwen3vl_dir(tmp_path))
    export_dir = tmp_path / "export_vlm"
    dist_workers.run(
        partial(_ptq_language_model_and_export, src_dir=src_dir, export_dir=export_dir)
    )

    exported = _safetensors_meta(export_dir)
    assert exported, "nothing exported"

    src_meta = _safetensors_meta(src_dir)
    vision_src = sorted(k for k in src_meta if ".visual." in k or ".vision" in k)
    assert vision_src, "fixture has no vision tower, so this test would prove nothing"

    missing = [k for k in vision_src if k not in exported]
    assert not missing, (
        f"{len(missing)} vision-tower tensor(s) absent from the export of a VLM whose language "
        f"model was quantized (e.g. {missing[0]}) -- the checkpoint is not the complete model"
    )
    # The untouched tower must come through at its original precision, with no scales invented.
    for k in vision_src:
        dtype, shape = exported[k]
        assert dtype not in _QUANTIZED_DTYPES, f"unquantized '{k}' exported as {dtype}"
        assert shape == src_meta[k][1], f"'{k}': shape {shape}, source {src_meta[k][1]}"
    stray = sorted(k for k in exported if (".visual." in k or ".vision" in k) and _is_scale(k))
    assert not stray, f"vision tower was not quantized but carries scales: {stray}"

    # ...and the language model must actually be quantized, or the test passes vacuously.
    quantized = [k for k, (d, _) in exported.items() if d in _QUANTIZED_DTYPES]
    assert quantized, "no tensor was quantized -- the language-model PTQ did not take effect"
