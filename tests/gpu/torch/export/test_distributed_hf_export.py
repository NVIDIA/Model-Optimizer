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
"""End-to-end checks on the checkpoint written by the no-gather FSDP2 distributed export.

``export_hf_checkpoint`` routes an FSDP2-sharded model under ``torch.distributed`` to
``distributed_save_hf_checkpoint``: every rank writes its own DCP shards and the files are
consolidated in parallel, with no full-model host-RAM gather on rank 0. That writer reconstructs
the checkpoint from per-rank pieces, so the failure modes it can introduce are structural rather
than numerical -- a key silently dropped, a rank-local shard written where the full tensor belongs,
a sidecar or the weight index never emitted. Those all produce a checkpoint that *looks* fine (no
error, plausible file sizes) but is wrong or unloadable, so they are asserted here explicitly:

1. every source parameter survives the round trip, and nothing extra is left behind;
2. each quantized weight carries the scales its format needs;
3. every tensor has its full, unsharded shape;
4. the non-weight files are all present -- the ones the exporter writes (config, generation config,
   quant config, weight index) and the ones it must leave alone. The tokenizer/vocab/chat-template
   files are written into the export dir by ``hf_ptq.py`` (``tokenizer.save_pretrained`` +
   ``copy_custom_model_files``), not by ``export_hf_checkpoint``, so what is asserted of them here
   is survival: the writer creates and removes a ``sharded/`` staging dir inside ``export_dir``, and
   must not take the sidecars with it.

It also covers the two things the writer must not quietly drop on the way: a declared tied
weight still gets deduplicated, and options the path cannot honour are rejected rather than
ignored.

Run with >=2 GPUs so the expert axis and the FSDP2 shard axis are both actually split.
"""

import copy
import json
import shutil
from functools import partial
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from _test_utils.torch.transformers_models import (
    create_tiny_qwen3_dir,
    create_tiny_qwen3_moe_dir,
    create_tiny_qwen3vl_dir,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_hf import export_hf_checkpoint
from modelopt.torch.quantization.utils import patch_fsdp_mp_dtypes
from modelopt.torch.utils.distributed import fsdp2_wrap, is_fsdp2_model

# Small enough to force the tiny model across several shards, so the multi-file layout -- and the
# weight index that makes it loadable -- are exercised rather than collapsing to one model.safetensors.
# Not smaller: each extra shard is another DCP write + consolidation round trip on a toy model.
MAX_SHARD_SIZE = "512KB"

# The default tiny Qwen3 is too degenerate to shard: head_dim = hidden_size / num_heads = 32/16 = 2,
# so q_norm/k_norm are 2-element tensors. Split over a 4-rank world most ranks get an empty chunk and
# the DCP planner rejects the coverage outright ("invalid fill tensor-volume"), taking the whole
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
_PACKED_DTYPES = {"U8"}


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


def _expected_weights(src: dict[str, tuple[str, tuple[int, ...]]]) -> dict[str, tuple[int, ...]]:
    """Source ``{name: shape}`` rewritten into the keys the exporter is expected to emit.

    Everything maps 1:1 except a MoE's fused 3-D expert weights, which the exporter splits into
    per-expert 2-D weights: ``experts.gate_up_proj`` ``[E, 2I, H]`` becomes ``experts.<e>.gate_proj``
    and ``experts.<e>.up_proj`` (each ``[I, H]``), and ``experts.down_proj`` ``[E, H, I]`` becomes
    ``experts.<e>.down_proj`` ``[H, I]``.
    """
    expected: dict[str, tuple[int, ...]] = {}
    for name, (_dtype, shape) in src.items():
        if name.endswith("mlp.experts.gate_up_proj"):
            prefix, experts, two_i, hidden = name[: -len("gate_up_proj")], *shape
            assert two_i % 2 == 0, f"{name}: fused gate_up dim {two_i} is not even"
            for e in range(experts):
                expected[f"{prefix}{e}.gate_proj.weight"] = (two_i // 2, hidden)
                expected[f"{prefix}{e}.up_proj.weight"] = (two_i // 2, hidden)
        elif name.endswith("mlp.experts.down_proj"):
            prefix, experts, hidden, inter = name[: -len("down_proj")], *shape
            for e in range(experts):
                expected[f"{prefix}{e}.down_proj.weight"] = (hidden, inter)
        else:
            expected[name] = shape
    return expected


def _ptq_and_export(rank, size, *, src_dir, export_dir, quant_cfg, **export_kwargs):
    """Load the tiny model on every rank, FSDP2-shard it, PTQ it, and export."""
    from transformers import AutoModelForCausalLM

    with patch_fsdp_mp_dtypes():
        model = AutoModelForCausalLM.from_pretrained(src_dir, dtype=torch.bfloat16).to("cuda")
        model.eval()

        fsdp2_wrap(model)
        # The no-gather writer is selected by exactly this predicate, so assert it here: without it
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


# Four PTQ + distributed-export round trips; the tests/gpu default of 120s is not enough headroom.
@pytest.mark.timeout(600)
@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
@pytest.mark.parametrize(
    ("quant_cfg", "algo"),
    [(mtq.FP8_DEFAULT_CFG, "FP8"), (mtq.NVFP4_DEFAULT_CFG, "NVFP4")],
    ids=["fp8", "nvfp4"],
)
def test_fsdp2_distributed_export_is_complete(dist_workers, tmp_path, moe, quant_cfg, algo):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs to shard the expert and FSDP2 axes")

    make_dir = create_tiny_qwen3_moe_dir if moe else create_tiny_qwen3_dir
    kwargs = {**TINY_KWARGS, **(TINY_MOE_KWARGS if moe else {})}
    src_dir = Path(make_dir(tmp_path, with_tokenizer=True, **kwargs))
    export_dir = tmp_path / "export"

    # Seed the export dir with the source's non-weight files, standing in for the
    # tokenizer/sidecar save that hf_ptq.py performs around the export call. They must still be
    # there afterwards: the writer stages per-rank shards in export_dir/sharded/ and deletes that
    # tree when it consolidates, which is exactly the operation that could take them out.
    sidecars = {
        p.name
        for p in src_dir.iterdir()
        if p.is_file()
        and not p.name.endswith(".safetensors")
        and p.name != "model.safetensors.index.json"
    }
    export_dir.mkdir(parents=True, exist_ok=True)
    for name in sidecars:
        shutil.copy2(src_dir / name, export_dir / name)

    dist_workers.run(
        partial(_ptq_and_export, src_dir=src_dir, export_dir=export_dir, quant_cfg=quant_cfg)
    )

    src = _safetensors_meta(src_dir)
    exported = _safetensors_meta(export_dir)
    assert exported, f"no safetensors written to {export_dir}"
    expected = _expected_weights(src)

    # ---- 1. every source parameter survives, and nothing extra is left behind ----
    missing = sorted(set(expected) - set(exported))
    assert not missing, (
        f"{len(missing)} source parameter(s) missing from the export: {missing[:10]}"
    )

    unexpected = sorted(k for k in set(exported) - set(expected) if not _is_scale(k))
    # A fused expert weight left next to the per-expert weights it was split into would show up
    # here -- the same tensor exported twice, once unquantized.
    assert not unexpected, f"{len(unexpected)} unexpected non-scale key(s): {unexpected[:10]}"

    # ---- 2. + 3. scales present, and every tensor at its full unsharded shape ----
    quantized = 0
    for name, want_shape in expected.items():
        dtype, got_shape = exported[name]
        if dtype in _QUANTIZED_DTYPES and name.endswith(".weight"):
            quantized += 1
            prefix = name[: -len(".weight")]
            assert f"{prefix}.weight_scale" in exported, f"{name}: quantized but no weight_scale"
            assert f"{prefix}.input_scale" in exported, f"{name}: quantized but no input_scale"
            if algo == "NVFP4":
                assert f"{prefix}.weight_scale_2" in exported, f"{name}: NVFP4 needs weight_scale_2"
            if dtype in _PACKED_DTYPES:
                # Two 4-bit values per byte -> the packed last dim is half the logical one.
                want_shape = (*want_shape[:-1], want_shape[-1] // 2)
        assert got_shape == want_shape, f"{name}: shape {got_shape}, expected {want_shape}"
    assert quantized > 0, f"nothing was quantized -- {algo} config did not take effect"

    # ---- 4. the non-weight files came along ----
    exported_files = {p.name for p in export_dir.iterdir() if p.is_file()}
    for required in ("config.json", "generation_config.json", "hf_quant_config.json"):
        assert required in exported_files, f"{required} missing from the export"

    # Everything the source shipped alongside its weights (tokenizer, vocab/merges, chat template,
    # special-tokens map, ...) survived the export; only the weight files are rewritten.
    lost = sorted(sidecars - exported_files)
    assert not lost, f"sidecar file(s) lost: {lost}"
    assert not (export_dir / "sharded").exists(), "per-rank staging dir left behind"

    # A multi-file checkpoint is unloadable without the index, so require it whenever the writer
    # emitted sharded names, and require it to describe every tensor actually on disk.
    if any(p.name.startswith("model-") for p in export_dir.glob("*.safetensors")):
        index_path = export_dir / "model.safetensors.index.json"
        assert index_path.exists(), "sharded export without model.safetensors.index.json"
        weight_map = json.loads(index_path.read_text())["weight_map"]
        assert set(weight_map) == set(exported), (
            f"index/shard mismatch: {len(set(exported) - set(weight_map))} tensor(s) unindexed, "
            f"{len(set(weight_map) - set(exported))} indexed but absent"
        )
        for key, fname in weight_map.items():
            assert (export_dir / fname).exists(), f"index points at missing shard {fname} for {key}"

    assert json.loads((export_dir / "hf_quant_config.json").read_text())["quantization"][
        "quant_algo"
    ] == algo


def _export_with(rank, size, *, src_dir, export_dir, **export_kwargs):
    """Export once with ``export_kwargs``, for the rejection checks below."""
    from transformers import AutoModelForCausalLM

    with patch_fsdp_mp_dtypes():
        model = AutoModelForCausalLM.from_pretrained(src_dir, dtype=torch.bfloat16).to("cuda")
        model.eval()
        fsdp2_wrap(model)
        assert is_fsdp2_model(model)
        torch.distributed.barrier()
        export_hf_checkpoint(model, export_dir=export_dir, **export_kwargs)


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"save_modelopt_state": True}, "save_modelopt_state"),
    ],
    ids=["save_modelopt_state"],
)
def test_fsdp2_distributed_export_rejects_unsupported_options(
    dist_workers, tmp_path, kwargs, needle
):
    """Options the no-gather path cannot honour must raise, not be silently ignored.

    ``save_modelopt_state`` is forwarded to ``save_pretrained`` on the gather path; the distributed
    path writes the checkpoint itself and never calls it, so a caller passing it would otherwise get
    a checkpoint quietly missing the ModelOpt state. (``extra_state_dict`` used to be rejected here
    too and is now supported -- see the MTP test below.)
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")

    src_dir = Path(create_tiny_qwen3_dir(tmp_path, with_tokenizer=True, **TINY_KWARGS))
    with pytest.raises(Exception, match=needle):
        dist_workers.run(
            partial(
                _export_with, src_dir=src_dir, export_dir=tmp_path / "rejected", **kwargs
            )
        )


@pytest.mark.timeout(600)
def test_fsdp2_distributed_export_dedups_tied_weights(dist_workers, tmp_path):
    """A declared tie is deduplicated by name in the distributed write, as on the gather path.

    ``fully_shard`` splits the shared parameter into distinct per-module shards, so the tie survives
    only as matching names -- both sides reach the writer as independent DTensors and would both be
    written unless the writer applies the same ``TiedWeightMap`` the gather path uses.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")

    src_dir = Path(
        create_tiny_qwen3_dir(
            tmp_path, with_tokenizer=True, tie_word_embeddings=True, **TINY_KWARGS
        )
    )
    export_dir = tmp_path / "export_tied"
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
    assert "model.embed_tokens.weight" in exported, "canonical tied weight missing"
    assert "lm_head.weight" not in exported, (
        "tied alias 'lm_head.weight' was written alongside its canonical -- "
        "name-based dedup did not run in the distributed writer"
    )


@pytest.mark.timeout(600)
def test_fsdp2_distributed_export_writes_extra_state_dict(dist_workers, tmp_path):
    """Tensors handed in via ``extra_state_dict`` must reach the checkpoint.

    This is how MTP weights arrive: the FSDP2 loader drops them, so ``hf_ptq`` reads them straight
    off the source checkpoint (``load_mtp_weights``) and passes them here. They belong to no rank's
    shard and no rank's model, so unless the writer deliberately merges them in, a model with MTP
    exports silently incomplete -- valid-looking, right size, missing the MTP layer.

    They must also keep their original names: they were never part of the transformers conversion,
    so the reverse conversion must not touch them.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")

    src_dir = Path(create_tiny_qwen3_dir(tmp_path, with_tokenizer=True, **TINY_KWARGS))
    export_dir = tmp_path / "export_extra"
    # Shaped and named like an inlined MTP tail (DeepSeek-V3 / GLM-5.1 put it at layers.<N>).
    extra = {
        "model.layers.99.mtp.weight": torch.arange(64, dtype=torch.bfloat16).reshape(8, 8),
        "model.layers.99.mtp.bias": torch.full((8,), 3.0, dtype=torch.bfloat16),
    }
    dist_workers.run(
        partial(
            _ptq_and_export,
            src_dir=src_dir,
            export_dir=export_dir,
            quant_cfg=mtq.FP8_DEFAULT_CFG,
            extra_state_dict=extra,
        )
    )

    exported = _safetensors_meta(export_dir)
    assert exported, "nothing exported"
    for name, want in extra.items():
        assert name in exported, f"extra_state_dict tensor '{name}' never reached the checkpoint"
        assert exported[name][1] == tuple(want.shape), f"{name}: shape {exported[name][1]}"

    index = json.loads((export_dir / "model.safetensors.index.json").read_text())
    for name in extra:
        assert name in index["weight_map"], f"'{name}' written but missing from the weight index"

    # Values must survive intact, not just the names.
    merged: dict = {}
    for shard in sorted(export_dir.glob("*.safetensors")):
        merged.update(load_file(str(shard)))
    for name, want in extra.items():
        assert torch.equal(merged[name].cpu(), want), f"{name}: value changed on the way out"


@pytest.mark.timeout(600)
def test_fsdp2_distributed_export_keeps_unquantized_submodule(dist_workers, tmp_path):
    """A submodule left unquantized must still be exported, in full and unquantized.

    PTQ is routinely applied to part of a model -- the language model of a VLM, or a layer range
    excluded by the recipe -- and the checkpoint is expected to be the COMPLETE model, with the
    untouched part carried through at its original precision. The writer builds its state dict from
    the whole model, so the risk is not that it drops those tensors but that it mangles them: the
    expert split, the scale postprocess and the reverse conversion all walk keys that have no
    quantizer state here.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")

    # Disable every quantizer under layer 0; layer 1 stays quantized as the control.
    quant_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    quant_cfg["quant_cfg"].append({"quantizer_name": "*layers.0*", "enable": False})

    src_dir = Path(create_tiny_qwen3_dir(tmp_path, with_tokenizer=True, **TINY_KWARGS))
    export_dir = tmp_path / "export_partial"
    dist_workers.run(
        partial(
            _ptq_and_export, src_dir=src_dir, export_dir=export_dir, quant_cfg=quant_cfg
        )
    )

    exported = _safetensors_meta(export_dir)
    assert exported, "nothing exported"

    src_meta = _safetensors_meta(src_dir)
    layer0_src = {k for k in src_meta if k.startswith("model.layers.0.")}
    assert layer0_src, "fixture has no layer 0 weights to check"

    for name in sorted(layer0_src):
        assert name in exported, f"unquantized '{name}' missing from the checkpoint"
        dtype, shape = exported[name]
        assert dtype not in _QUANTIZED_DTYPES, (
            f"'{name}' was excluded from quantization but exported as {dtype}"
        )
        assert shape == src_meta[name][1], (
            f"'{name}': shape {shape}, source {src_meta[name][1]}"
        )

    # No scale companions should exist for the excluded layer...
    stray = sorted(k for k in exported if k.startswith("model.layers.0.") and _is_scale(k))
    assert not stray, f"excluded layer 0 carries quantization scales: {stray}"
    # ...while the control layer must actually be quantized, or the test proves nothing.
    quantized_l1 = [
        k for k, (d, _) in exported.items() if k.startswith("model.layers.1.") and d in _QUANTIZED_DTYPES
    ]
    assert quantized_l1, "layer 1 was not quantized -- the exclusion pattern disabled everything"


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
    from transformers import AutoModelForImageTextToText

    with patch_fsdp_mp_dtypes():
        model = AutoModelForImageTextToText.from_pretrained(src_dir, dtype=torch.bfloat16).to("cuda")
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
