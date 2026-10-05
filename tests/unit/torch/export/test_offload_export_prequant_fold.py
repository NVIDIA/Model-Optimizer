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

"""End-to-end: INT4_AWQ export of a CPU-offloaded model must fold only safe norms.

The unit tests next door drive ``_fuse_shared_input_modules`` and the two fold helpers
directly.  This one goes through the real entry point -- ``mtq.quantize`` then
``export_hf_checkpoint`` -- on a tiny Qwen3_5 replica whose decoder layers carry genuine
``accelerate`` offload hooks, so the exported safetensors can be checked directly.

Three properties are asserted, and each one is a real defect this change fixes:

1. **The out-of-group consumer guard survives the whole pipeline.**  ``v_proj`` is left
   unquantized, mirroring MiMo's ``in_proj_a`` / ``in_proj_b``, so ``input_layernorm``'s
   output feeds a module outside the fused group.  The exported norm weight must equal the
   pre-export weight and the fused members must still carry ``pre_quant_scale`` -- folding
   would rescale activations the excluded ``v_proj`` also consumes.

2. **The fold is written back to the offload holder.**  Offloaded norms hold their weight
   on meta between forwards.  ``fuse_prequant_layernorm`` folds
   ``(1 + w) * scale - 1`` for a zero-centered gamma (Qwen3_5) and ``w * scale`` otherwise;
   without materializing the weight and writing the result back, the exported norm is the
   *unfolded* weight while the members' ``pre_quant_scale`` has been deleted -- the scale is
   lost from the checkpoint entirely.

3. **The folded value is the formula's, not a re-derivation of it.**  ``post_attention_layernorm``
   has no out-of-group consumer, so it folds; the exported weight is compared against the
   formula applied to the pre-export weight.

Both norms are checked on the same export: one guarded, one folded, in a single pass.
"""

import copy
import warnings

import pytest
import torch

pytest.importorskip("accelerate")
pytest.importorskip("safetensors")

from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from accelerate.utils import set_module_tensor_to_device

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.utils.core_utils import (
    enable_weight_access_and_writeback,
    has_accelerate_offload,
)

# INT4 packing is block-quantized over the input dimension; every quantized Linear must
# have ``in_features % 128 == 0`` or ``pack_int4_in_uint8`` raises.
DIM = 128
INTERMEDIATE = 256
NUM_LAYERS = 2

# module name of the norm / member, relative to a decoder layer
GUARDED_NORM = "input_layernorm"
GUARDED_GROUP = ("self_attn.q_proj", "self_attn.k_proj")
GUARDED_OUTSIDER = "self_attn.v_proj"  # left unquantized: the out-of-group consumer
FOLDED_NORM = "post_attention_layernorm"
FOLDED_GROUP = ("mlp.gate_proj", "mlp.up_proj")


# ---------------------------------------------------------------------------
# Tiny replica + real accelerate offload
# ---------------------------------------------------------------------------


def _tiny_qwen35():
    """A tiny Qwen3_5 text model: the same zero-centered ``Qwen3_5RMSNorm`` MiMo uses."""
    config_cls = pytest.importorskip(
        "transformers.models.qwen3_5.configuration_qwen3_5"
    ).Qwen3_5TextConfig
    from transformers import AutoModelForCausalLM

    config = config_cls(
        vocab_size=128,
        hidden_size=DIM,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        layer_types=["full_attention"] * NUM_LAYERS,
        tie_word_embeddings=False,
        architectures=["Qwen3_5ForCausalLM"],
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    # Qwen3_5RMSNorm starts at weight == 0.  With the guard leaving input_layernorm
    # unfolded, an all-zero norm in the export is indistinguishable from a folded one
    # (both are zero) and the per-module pre_quant_scale is meaningless.  Seed every
    # 1-D weight so a non-folded norm's exported value is unambiguous.
    with torch.no_grad():
        for module in model.modules():
            w = getattr(module, "weight", None)
            if w is not None and w.dim() == 1:
                w.copy_(torch.randn_like(w) * 0.25)
    return model.eval()


def _cpu_offload_decoder_layers(model):
    """Attach real accelerate offload hooks to every decoder layer, CPU-only.

    Mirrors what ``dispatch_model`` produces for an offloaded ``device_map`` entry: the
    module holding weights gets an ``AlignDevicesHook(offload=True, weights_map=...)`` and
    its weight is moved to meta, while weightless containers get a plain hook.  The
    container hook matters -- ``enable_weight_access_and_writeback`` dispatches on the
    module it is handed, so a layer without one is never materialized.

    ``embed_tokens`` / ``lm_head`` deliberately keep resident weights: ``model.device``
    resolves from them, and a fully-meta model sends the exporter's dummy forward to meta.
    """
    for name, module in list(model.named_modules()):
        if ".layers." not in name:
            continue
        params = [n for n, _ in module.named_parameters(recurse=False)]
        if not params:
            add_hook_to_module(module, AlignDevicesHook(execution_device="cpu", offload=False))
            continue
        weights_map = {n: getattr(module, n).detach().clone() for n in params}
        add_hook_to_module(
            module,
            AlignDevicesHook(execution_device="cpu", offload=True, weights_map=weights_map),
        )
        for n in params:
            set_module_tensor_to_device(module, n, "meta")
    return model


def _materialized_weight(module, root_model):
    """Read a real weight out of an offloaded module (enters its materialization window)."""
    with enable_weight_access_and_writeback(module, root_model, None, writeback=False):
        return module.weight.detach().cpu().clone()


def _pre_quant_scale(module):
    quantizer = getattr(module, "input_quantizer", None)
    scale = getattr(quantizer, "_pre_quant_scale", None) if quantizer is not None else None
    return None if scale is None else scale.detach().cpu().clone()


# ---------------------------------------------------------------------------
# Export + read back
# ---------------------------------------------------------------------------


def _read_export(export_dir):
    """Merge every shard into one ``{key: tensor}`` dict, checking the index agrees."""
    import json

    from safetensors.torch import load_file

    merged = {}
    for shard in sorted(export_dir.glob("*.safetensors")):
        merged.update(load_file(str(shard)))
    index_path = export_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        assert set(weight_map) == set(merged), "index weight_map disagrees with shard contents"
        for shard_name in set(weight_map.values()):
            assert (export_dir / shard_name).exists(), f"indexed shard '{shard_name}' missing"
    return merged


def _awq_cfg_with_excluded_consumer():
    """INT4_AWQ plus the recipe rule that makes the guard fire.

    ``v_proj`` out of quantization is what makes ``input_layernorm`` a norm with a consumer
    outside its fused group -- the MiMo situation, where AWQ recipes leave GatedDeltaNet's
    ``in_proj_a`` / ``in_proj_b`` unquantized because quantizing them is not supported.
    """
    quant_cfg = copy.deepcopy(mtq.INT4_AWQ_CFG)
    quant_cfg["quant_cfg"].append({"quantizer_name": f"*{GUARDED_OUTSIDER}*", "enable": False})
    return quant_cfg


def _quantize_offloaded(model):
    def forward_loop(m):
        with torch.no_grad():
            m(torch.zeros(1, 8, dtype=torch.long))

    return mtq.quantize(model, _awq_cfg_with_excluded_consumer(), forward_loop)


@pytest.fixture(scope="module")
def offloaded_awq_export(tmp_path_factory):
    """Quantize + export the offloaded replica once; every assertion reads the result.

    Module-scoped because the whole point is to compare the *same* export against
    pre-export state, and INT4 AWQ calibration plus a streaming export is not cheap.
    """
    tmp_path = tmp_path_factory.mktemp("offloaded_awq")
    model = _cpu_offload_decoder_layers(_tiny_qwen35())
    assert has_accelerate_offload(model), "offload hooks were not attached"

    # A guard-only fix would still pass if the export took the resident path, so check
    # that the model really is in the state that path is for.
    norm = model.get_submodule(f"model.layers.0.{GUARDED_NORM}")
    assert norm.weight.is_meta, "offloaded norm must hold its weight on meta"

    model = _quantize_offloaded(model)

    # Snapshot pre-export state. Offloaded weights are meta between forwards, so norms are
    # read inside a materialization window.
    pre_export = {}
    for index in range(NUM_LAYERS):
        layer = model.get_submodule(f"model.layers.{index}")
        for suffix in (GUARDED_NORM, FOLDED_NORM):
            pre_export[(index, suffix)] = _materialized_weight(layer.get_submodule(suffix), model)
        for suffix in GUARDED_GROUP + FOLDED_GROUP:
            pre_export[(index, suffix)] = _pre_quant_scale(layer.get_submodule(suffix))

    export_dir = tmp_path / "hf_export"
    export_dir.mkdir()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_hf_checkpoint(model, export_dir=str(export_dir), max_shard_size="1GB")
    guard_warnings = [
        str(w.message) for w in caught if "Not folding pre_quant_scale" in str(w.message)
    ]
    return {
        "model": model,
        "pre_export": pre_export,
        "export": _read_export(export_dir),
        "guard_warnings": guard_warnings,
    }


def _layer_keys(exported, index, suffix):
    """Every exported key for ``suffix`` inside layer ``index`` (0, 1, or more)."""
    needle = f"layers.{index}.{suffix}"
    return [key for key in exported if key.endswith(needle) or f".{needle}." in key]


def _layer_key(exported, index, suffix):
    """The single exported key for ``suffix``; fails loudly if missing or ambiguous."""
    matches = _layer_keys(exported, index, suffix)
    assert len(matches) == 1, (
        f"expected exactly one exported key ending in {suffix!r}, got {matches}"
    )
    return matches[0]


# ---------------------------------------------------------------------------
# 1. The guard
# ---------------------------------------------------------------------------


def test_guarded_norm_is_not_folded_end_to_end(offloaded_awq_export):
    """A norm feeding an out-of-group consumer keeps its weight and every per-module scale."""
    model = offloaded_awq_export["model"]
    exported = offloaded_awq_export["export"]

    for index in range(NUM_LAYERS):
        key = _layer_key(exported, index, f"{GUARDED_NORM}.weight")
        before = offloaded_awq_export["pre_export"][(index, GUARDED_NORM)]
        got = exported[key]
        assert torch.equal(got, before), (
            f"{key} was folded even though {GUARDED_OUTSIDER} consumes the same norm output; "
            f"max|diff| = {float((got.float() - before.float()).abs().max())}"
        )

        # The members keep their own pre_quant_scale: folding would delete it.
        for suffix in GUARDED_GROUP:
            scale_key = _layer_key(exported, index, f"{suffix}.pre_quant_scale")
            expected = offloaded_awq_export["pre_export"][(index, suffix)]
            assert torch.allclose(
                exported[scale_key].float(), expected.float(), rtol=1e-5, atol=1e-6
            ), (
                f"{scale_key} was resmoothed to the group average, but each module keeps its "
                "own scale when the fold is skipped"
            )
            module = model.get_submodule(f"model.layers.{index}.{suffix}")
            assert not getattr(module, "fused_with_prequant", False), (
                f"{suffix} in layer {index} was marked as folded despite the guard"
            )


def test_guard_warns_for_every_out_of_group_norm(offloaded_awq_export):
    """One warning per guarded norm, naming the norm and the offending consumer."""
    warnings_ = offloaded_awq_export["guard_warnings"]

    assert len(warnings_) == NUM_LAYERS, (
        f"expected one guard warning per layer, got {len(warnings_)}: {warnings_}"
    )
    for index in range(NUM_LAYERS):
        matching = [
            message
            for message in warnings_
            if f"model.layers.{index}.{GUARDED_NORM}" in message and GUARDED_OUTSIDER in message
        ]
        assert matching, f"no guard warning for layer {index}'s {GUARDED_NORM}: {warnings_}"


# ---------------------------------------------------------------------------
# 2 + 3. The fold
# ---------------------------------------------------------------------------


def test_folded_norm_is_written_back_through_the_offload_holder(offloaded_awq_export):
    """The folded value must reach the exported checkpoint, with its scales consumed.

    The pre-fix failure mode is specific and silent: the fold ran on the norm's meta
    weight, so the export emitted the *unfolded* weight while every member's
    ``pre_quant_scale`` had already been deleted -- the scale vanishes from the checkpoint
    with nothing to indicate it.
    """
    exported = offloaded_awq_export["export"]

    for index in range(NUM_LAYERS):
        key = _layer_key(exported, index, f"{FOLDED_NORM}.weight")
        before = offloaded_awq_export["pre_export"][(index, FOLDED_NORM)]
        scale = torch.stack(
            [offloaded_awq_export["pre_export"][(index, s)] for s in FOLDED_GROUP]
        ).mean(dim=0)
        # Qwen3_5RMSNorm is zero-centered: its forward multiplies by (1 + weight), so the
        # fold has to invert that to make out_after == out_before * scale.
        expected = ((before + 1.0) * scale - 1.0).to(before.dtype)
        got = exported[key]

        assert got.dtype == before.dtype, f"{key} exported as {got.dtype}, model is {before.dtype}"
        assert not torch.allclose(got.float(), before.float(), rtol=1e-3, atol=1e-4), (
            f"{key} was exported unfolded: the fold was not written back to the offload "
            "holder, so the meta weight was folded and discarded"
        )
        assert torch.equal(got, expected), (
            f"{key} = {got[:4].tolist()}, expected (1 + w) * s_avg - 1 = {expected[:4].tolist()}"
        )

        for suffix in FOLDED_GROUP:
            leaked = _layer_keys(exported, index, f"{suffix}.pre_quant_scale")
            assert not leaked, (
                f"{leaked} was exported even though its norm was folded: "
                "the scale would be applied twice"
            )


def test_export_took_the_offloaded_streaming_path(offloaded_awq_export):
    """A second export of the same model must still work from a cold directory.

    Cheap end-to-end smoke check that the exported checkpoint itself re-imports and that
    the streamed tensors are all real (no meta or all-zero weight written out), which is
    what the writeback fix protects.
    """
    exported = offloaded_awq_export["export"]
    assert exported, "export produced no tensors"

    for key, tensor in exported.items():
        assert not tensor.is_meta, f"{key} is a meta tensor in the export"
        assert tensor.numel() > 0, f"{key} is empty in the export"
        if key.endswith("weight") and "quantizer" not in key and "scale" not in key:
            assert tensor.float().abs().sum() > 0, f"{key} is all zeros in the export"

    assert _layer_key(exported, 0, f"{GUARDED_NORM}.weight") in exported
    assert _layer_key(exported, 0, f"{FOLDED_NORM}.weight") in exported
