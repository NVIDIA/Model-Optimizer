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

"""Unit tests for quant-aware reverse weight conversion (CPU, no GPU needed).

Tensor shapes mirror a real NVFP4 linear from the MiniMax-M3 checkpoint: ``weight``
uint8 ``[out, in//2]``, ``weight_scale`` ``[out, in//16]``, ``weight_scale_2`` /
``input_scale`` 0-d scalars. The reverse logic is dtype-agnostic, so ``weight_scale``
uses float32 here (real checkpoints use float8_e4m3, whose CPU ops are not portable
across platforms) — only shapes and the scalar-vs-blocked distinction matter.
"""

import json
import types
import warnings
from fnmatch import fnmatchcase

import pytest
import torch
from safetensors.torch import load_file

import modelopt.torch.quantization as mtq
from modelopt.torch.export.layerwise_export import LayerwiseExporter
from modelopt.torch.export.quant_aware_conversion import (
    QuantConversionUnsupportedError,
    RenameRule,
    SplitRule,
    _assert_experts_pre_expanded,
    apply_reverse_rules,
    build_reverse_name_mapper,
    revert_quant_config_names,
    revert_weight_conversion_quant_aware,
)
from modelopt.torch.export.quant_utils import _prefix_wildcard_summarize_exclude_modules
from modelopt.torch.export.unified_export_hf import (
    _export_transformers_checkpoint,
    _revert_hf_quant_config_names,
    _revert_quant_config_names_best_effort,
    export_hf_checkpoint,
)
from modelopt.torch.export.unified_export_hf_streaming import (
    _assert_no_split_rules,
    _build_reverse_name_mapper_or_none,
    _make_tensor_sink,
    _StreamingShardWriter,
)

BLOCK = 16


def _set_scope_attr(transform, name, value):
    """Skip unsupported scope tests; callers fold an absent base prefix into the scope."""
    try:
        setattr(transform, name, value)
    except AttributeError:
        if name == "scope_prefix":
            pytest.skip("Transformers weight transforms do not support scope_prefix")
        if name != "base_model_prefix":
            raise


def _radio_qkv_conversion(base_model_prefix="", pattern_suffix=""):
    """Build the RADIO fused-QKV conversion with an optional parent-model scope."""
    pytest.importorskip("transformers.core_model_loading")
    # Local import: transformers is an optional dependency for ModelOpt.
    from transformers.core_model_loading import Chunk, WeightConverter

    qkv = WeightConverter(
        source_patterns="attn.qkv" + pattern_suffix,
        target_patterns=[f"attention.{part}_proj{pattern_suffix}" for part in ("q", "k", "v")],
        operations=[Chunk(dim=0)],
    )
    _set_scope_attr(qkv, "scope_prefix", "vision_model")
    if base_model_prefix and not hasattr(qkv, "base_model_prefix"):
        qkv.scope_prefix = f"{base_model_prefix}.{qkv.scope_prefix}"
    _set_scope_attr(qkv, "base_model_prefix", base_model_prefix)
    return qkv


# Tiny Mixtral shaped to match the synthetic expert tensors built by ``_nvfp4_linear`` below.
_MIXTRAL_KWARGS = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_local_experts": 2,
    "num_experts_per_tok": 2,
    "vocab_size": 64,
    "max_position_embeddings": 64,
}


def _nvfp4_linear(module: str, out: int, in_features: int) -> dict[str, torch.Tensor]:
    """Synthetic NVFP4 quantized-linear tensor group keyed under ``module``."""
    return {
        f"{module}.weight": torch.randint(0, 255, (out, in_features // 2), dtype=torch.uint8),
        f"{module}.weight_scale": torch.randn(out, in_features // BLOCK),
        f"{module}.weight_scale_2": torch.tensor(0.037, dtype=torch.float32),  # 0-d
        f"{module}.input_scale": torch.tensor(1.0, dtype=torch.float32),  # 0-d
    }


def _stream_tensors(model, state, export_dir):
    writer = _StreamingShardWriter(export_dir, max_shard_size=4096)
    sink = _make_tensor_sink(
        writer,
        _build_reverse_name_mapper_or_none(model),
        tied_alias_keys=set(),
        kv_cache_max_bound=448.0,
        kv_cache_format=None,
        is_modelopt_qlora=False,
    )
    for key, value in state.items():
        sink(key, value)
    writer.finalize()


def _load_shards(export_dir):
    return {
        key: value
        for shard in export_dir.glob("*.safetensors")
        for key, value in load_file(shard).items()
    }


def _fp8_llama(*quantizer_names):
    pytest.importorskip("transformers.core_model_loading")
    # Local import: transformers and its test fixtures are optional dependencies.
    from _test_utils.torch.transformers_models import get_tiny_llama

    model = get_tiny_llama(num_hidden_layers=1, num_key_value_heads=16)
    model.config.architectures = ["LlamaForCausalLM"]
    return mtq.quantize(
        model,
        {
            "quant_cfg": [{"quantizer_name": "*", "enable": False}]
            + [
                {
                    "quantizer_name": name,
                    "cfg": {"num_bits": (4, 3), "constant_amax": 1.0},
                    "enable": True,
                }
                for name in quantizer_names
            ],
            "algorithm": None,
        },
    )


def test_rename_carries_scale_siblings():
    """A module rename rewrites weight + all scale siblings with identical values."""
    sd = _nvfp4_linear("model.language_model.layers.10.mlp.experts.40.gate_proj", 8, 16)
    rules = [
        RenameRule(r"\.mlp\.experts\.", ".block_sparse_moe.experts."),
        RenameRule(r"(\.block_sparse_moe\.experts\.\d+\.)gate_proj", r"\1w1"),
        RenameRule(r"^model\.language_model\.", "language_model.model."),
    ]
    out = apply_reverse_rules(sd, [], rules)

    base = "language_model.model.layers.10.block_sparse_moe.experts.40.w1"
    assert set(out) == {
        f"{base}.weight",
        f"{base}.weight_scale",
        f"{base}.weight_scale_2",
        f"{base}.input_scale",
    }
    # values untouched: a rename rebinds the same tensor object (no copy)
    for leaf in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale"):
        old = sd[f"model.language_model.layers.10.mlp.experts.40.gate_proj{leaf}"]
        assert out[base + leaf] is old


@pytest.mark.parametrize(
    ("pattern", "replacement", "error"),
    [
        (r"^head\.weight$", "in_proj_weight", "cannot align scale"),
        (r"^head\.(weight|input_scale)$", "output.weight", "conflicting scale rename"),
    ],
)
def test_weight_specific_scale_rename_rejects_ambiguous_targets(pattern, replacement, error):
    """Unsupported or conflicting weight/scale mappings leave the input untouched."""
    state = _nvfp4_linear("head", 8, 16)
    before = dict(state)
    with pytest.raises(QuantConversionUnsupportedError, match=error):
        apply_reverse_rules(state, [], [RenameRule(pattern, replacement)])
    assert state.keys() == before.keys()
    assert all(state[key] is value for key, value in before.items())


def test_scale_specific_rename_without_weight_mapping():
    """An explicit scale-only mapping retains its original regex behavior."""
    scale = torch.tensor(0.5)
    assert apply_reverse_rules(
        {"head.input_scale": scale},
        [],
        [RenameRule(r"^head\.input_scale$", "head.activation_scale")],
    ) == {"head.activation_scale": scale}


def test_split_unfuses_dense_gate_up_with_scales():
    """gate_up_proj -> gate_proj + up_proj: weight/scale split on dim 0, scalars duplicated."""
    out_dim, in_dim = 8, 32  # fused output dim = 8 -> 4 per part
    sd = _nvfp4_linear("m.layers.0.mlp.gate_up_proj", out_dim, in_dim)
    rule = SplitRule(".gate_up_proj", (".gate_proj", ".up_proj"), dim=0)

    out = apply_reverse_rules(sd, [rule], [])

    g, u = "m.layers.0.mlp.gate_proj", "m.layers.0.mlp.up_proj"
    assert set(out) == {
        f"{g}.weight",
        f"{g}.weight_scale",
        f"{g}.weight_scale_2",
        f"{g}.input_scale",
        f"{u}.weight",
        f"{u}.weight_scale",
        f"{u}.weight_scale_2",
        f"{u}.input_scale",
    }
    # weight/scale halved on dim 0; concatenating the parts reconstructs the original
    assert out[f"{g}.weight"].shape == (out_dim // 2, in_dim // 2)
    assert out[f"{g}.weight_scale"].shape == (out_dim // 2, in_dim // BLOCK)
    assert torch.equal(
        torch.cat([out[f"{g}.weight"], out[f"{u}.weight"]], dim=0),
        sd["m.layers.0.mlp.gate_up_proj.weight"],
    )
    # 0-d scalars duplicated to both parts
    for part in (g, u):
        assert out[f"{part}.weight_scale_2"].dim() == 0
        assert torch.equal(
            out[f"{part}.weight_scale_2"], sd["m.layers.0.mlp.gate_up_proj.weight_scale_2"]
        )


def test_stacked_3d_expert_raises_unsupported():
    """A stacked [num_experts, out, in] weight must trigger the safe fallback path."""
    sd = {
        "m.layers.0.mlp.experts.gate_up_proj.weight": torch.zeros(4, 8, 16, dtype=torch.uint8),
    }
    rule = SplitRule(".gate_up_proj", (".gate_proj", ".up_proj"), dim=0)
    with pytest.raises(QuantConversionUnsupportedError):
        apply_reverse_rules(sd, [rule], [])


def test_non_divisible_split_raises():
    sd = {"m.mlp.gate_up_proj.weight": torch.zeros(7, 8, dtype=torch.uint8)}
    rule = SplitRule(".gate_up_proj", (".gate_proj", ".up_proj"), dim=0)
    with pytest.raises(QuantConversionUnsupportedError):
        apply_reverse_rules(sd, [rule], [])


def test_end_to_end_minimax_m3_like_reversal():
    """Reverse a v1-style (post-conversion) M3 state dict back to hub names."""
    sd = {}
    # dense MLP layer 0: fused gate_up + separate down
    sd.update(_nvfp4_linear("model.language_model.layers.0.mlp.gate_up_proj", 8, 16))
    sd.update(_nvfp4_linear("model.language_model.layers.0.mlp.down_proj", 16, 8))
    # MoE layer 10: per-expert (already unfused) + router
    sd.update(_nvfp4_linear("model.language_model.layers.10.mlp.experts.0.gate_proj", 8, 16))
    sd.update(_nvfp4_linear("model.language_model.layers.10.mlp.experts.0.up_proj", 8, 16))
    sd.update(_nvfp4_linear("model.language_model.layers.10.mlp.experts.0.down_proj", 16, 8))
    sd["model.language_model.layers.10.mlp.gate.weight"] = torch.randn(128, 6144)
    sd["lm_head.weight"] = torch.randn(32, 16)

    split_rules = [SplitRule(".gate_up_proj", (".gate_proj", ".up_proj"), dim=0)]
    rename_rules = [
        RenameRule(r"(\.experts\.\d+\.)gate_proj", r"\1w1"),
        RenameRule(r"(\.experts\.\d+\.)up_proj", r"\1w3"),
        RenameRule(r"(\.experts\.\d+\.)down_proj", r"\1w2"),
        RenameRule(r"\.mlp\.experts\.", ".block_sparse_moe.experts."),
        RenameRule(r"\.mlp\.gate\.", ".block_sparse_moe.gate."),
        RenameRule(r"^model\.language_model\.", "language_model.model."),
        RenameRule(r"^lm_head\.", "language_model.lm_head."),
    ]
    out = apply_reverse_rules(sd, split_rules, rename_rules)

    expected = {
        # dense un-fused, still under mlp
        "language_model.model.layers.0.mlp.gate_proj",
        "language_model.model.layers.0.mlp.up_proj",
        "language_model.model.layers.0.mlp.down_proj",
        # experts renamed to block_sparse_moe + w1/w3/w2
        "language_model.model.layers.10.block_sparse_moe.experts.0.w1",
        "language_model.model.layers.10.block_sparse_moe.experts.0.w3",
        "language_model.model.layers.10.block_sparse_moe.experts.0.w2",
    }
    got_modules = {k.rsplit(".", 1)[0] for k in out if ".experts." in k or ".mlp." in k}
    assert expected <= got_modules
    assert "language_model.model.layers.10.block_sparse_moe.gate.weight" in out
    assert "language_model.lm_head.weight" in out
    # no leftover in-memory names
    assert not any(k.startswith("model.language_model") for k in out)
    assert not any(".gate_up_proj" in k for k in out)


def test_build_reverse_rules_from_mixtral_conversion_mapping_cpu():
    """Derive rules from a real transformers conversion mapping (CPU, no quantize).

    Exercises ``revert_weight_conversion_quant_aware`` / ``_build_reverse_rules``:
    a ModelOpt-expanded per-expert state dict (in-memory ``mlp.experts.<i>.*`` names)
    must revert to the hub layout (``block_sparse_moe.experts.<i>.w{1,2,3}``).
    """
    # Imports stay function-local: unit tests must import without transformers installed.
    pytest.importorskip("transformers")
    from _test_utils.torch.transformers_models import get_tiny_mixtral

    try:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    except ImportError:
        pytest.skip("transformers build has no conversion_mapping API")
    if not get_checkpoint_conversion_mapping("mixtral"):
        pytest.skip("transformers build has no mixtral conversion_mapping")

    model = get_tiny_mixtral(**_MIXTRAL_KWARGS)

    p = "model.layers.0"
    sd = {f"{p}.mlp.gate.weight": torch.randn(2, 32)}
    for e in range(2):
        sd.update(_nvfp4_linear(f"{p}.mlp.experts.{e}.gate_proj", 64, 32))
        sd.update(_nvfp4_linear(f"{p}.mlp.experts.{e}.up_proj", 64, 32))
        sd.update(_nvfp4_linear(f"{p}.mlp.experts.{e}.down_proj", 32, 64))

    out = revert_weight_conversion_quant_aware(model, sd)

    # experts mapped to hub layout, with scale siblings carried along
    for e in range(2):
        base = f"{p}.block_sparse_moe.experts.{e}"
        assert f"{base}.w1.weight" in out  # gate_proj -> w1
        assert f"{base}.w3.weight" in out  # up_proj   -> w3
        assert f"{base}.w2.weight" in out  # down_proj -> w2
        assert f"{base}.w1.weight_scale" in out
        assert f"{base}.w1.weight_scale_2" in out
    assert f"{p}.block_sparse_moe.gate.weight" in out
    assert not any(".mlp.experts." in k for k in out)


def test_build_reverse_rules_orders_prefix_reorder_after_container():
    """WeightRenamings must reverse in reverse list order (M3 prefix-reorder bug).

    transformers *loads* by chaining renamings in list order: a component-reordering
    rename (``language_model.model`` -> ``model.language_model``) fires first, making
    ``language_model`` adjacent to ``layers`` so a later container rename anchored on
    that adjacency (``.language_model.layers.N.mlp.experts.`` ->
    ``.block_sparse_moe.experts.``) can match. On the save path the reorder must run
    *last*, else it moves ``language_model`` away from ``layers`` and the container
    rename silently no-ops -- exporting MiniMax-M3 experts as ``mlp.experts.*`` instead
    of the hub ``block_sparse_moe.experts.*``. Mixtral does not exercise this (no
    prefix reorder), so this reproduces it with a minimal two-renaming mapping.
    """
    pytest.importorskip("transformers.core_model_loading")
    from transformers.core_model_loading import WeightRenaming

    # Forward (hub -> in-memory) renamings; ``reverse_transform`` flips them on save.
    # Order matters: reorder is listed BEFORE the adjacency-anchored container rename,
    # exactly as a real M3 conversion mapping lists them.
    conversions = [
        WeightRenaming("^language_model.model.", "model.language_model."),
        WeightRenaming(
            ".language_model.layers.(\\d+).block_sparse_moe.experts.",
            ".language_model.layers.\\1.mlp.experts.",
        ),
    ]
    model = types.SimpleNamespace(_weight_conversions=conversions)

    # In-memory expert key (leaf already at ``w1``; isolates the container/prefix order).
    sd = _nvfp4_linear("model.language_model.layers.10.mlp.experts.0.w1", 8, 16)
    out = revert_weight_conversion_quant_aware(model, sd)

    base = "language_model.model.layers.10.block_sparse_moe.experts.0.w1"
    assert set(out) == {
        f"{base}.weight",
        f"{base}.weight_scale",
        f"{base}.weight_scale_2",
        f"{base}.input_scale",
    }
    # Regression guard: the buggy reorder-first order leaves these in-memory fragments.
    assert not any(k.startswith("model.language_model") for k in out)
    assert not any(".mlp.experts." in k for k in out)


def test_nested_text_prefix_reverse_does_not_capture_vlm_siblings():
    """A nested text-model conversion must not rewrite the full VLM namespace."""
    pytest.importorskip("transformers.core_model_loading")
    from transformers.core_model_loading import WeightRenaming

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.visual = torch.nn.Module()
    model.model.visual.patch_embed = torch.nn.Linear(2, 2, bias=False)
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
    model._weight_conversions = [
        WeightRenaming(
            source_patterns=r"^model.language_model.",
            target_patterns=r"^model.(?!language_model.)",
        )
    ]

    state_dict = {
        "model.visual.patch_embed.weight": torch.randn(2, 2),
        "model.language_model.layers.0.weight": torch.randn(2, 2),
    }
    reverted = revert_weight_conversion_quant_aware(model, state_dict)

    assert set(reverted) == set(state_dict)
    assert build_reverse_name_mapper(model) is None


def test_nested_text_prefix_reverse_still_applies_to_text_model():
    """The same conversion remains valid when the nested VLM namespace is absent."""
    pytest.importorskip("transformers.core_model_loading")
    from transformers.core_model_loading import WeightRenaming

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
    model._weight_conversions = [
        WeightRenaming(
            source_patterns=r"^model.language_model.",
            target_patterns=r"^model.(?!language_model.)",
        )
    ]

    state_dict = {"model.layers.0.weight": torch.randn(2, 2)}
    reverted = revert_weight_conversion_quant_aware(model, state_dict)

    assert set(reverted) == {"model.language_model.layers.0.weight"}
    mapper = build_reverse_name_mapper(model)
    assert mapper is not None
    assert mapper("model.layers.0") == "model.language_model.layers.0"


def test_scoped_submodel_prefix_change_does_not_capture_siblings():
    """A vision sub-model's ``PrefixChange`` must not prefix the whole VLM state dict.

    NVBug 6525511: ``LlavaForConditionalGeneration`` on transformers>=5.12 collects the
    vision tower's own "add ``vision_model.``" prefix change. transformers scopes it to
    ``model.vision_tower`` via ``scope_prefix`` and only matches keys under that prefix;
    applying the raw pattern instead prefixes *every* key, so the export writes
    ``vision_model.language_model.*`` / ``vision_model.lm_head.*`` and vLLM fails with
    "There is no module or parameter named 'vision_model'".
    """
    core = pytest.importorskip("transformers.core_model_loading")
    if not hasattr(core, "PrefixChange"):
        pytest.skip("Transformers does not expose PrefixChange")
    # Local import: optional dependency, guarded by the importorskip above.
    from transformers.core_model_loading import PrefixChange

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.vision_tower = torch.nn.Module()
    model.model.vision_tower.encoder = torch.nn.Linear(2, 2, bias=False)
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
    model.lm_head = torch.nn.Linear(2, 2, bias=False)

    prefix_change = PrefixChange(prefix_to_remove="vision_model")
    _set_scope_attr(prefix_change, "scope_prefix", "model.vision_tower")
    _set_scope_attr(prefix_change, "base_model_prefix", "model")
    model._weight_conversions = [prefix_change]

    state_dict = {
        "model.vision_tower.encoder.weight": torch.randn(2, 2),
        "model.language_model.layers.0.weight": torch.randn(2, 2),
        "lm_head.weight": torch.randn(2, 2),
    }
    reverted = revert_weight_conversion_quant_aware(model, state_dict)

    # Only the vision tower's own subtree gains the ``vision_model.`` segment.
    assert set(reverted) == {
        "model.vision_tower.vision_model.encoder.weight",
        "model.language_model.layers.0.weight",
        "lm_head.weight",
    }
    # Regression guard: nothing may be moved under a bogus top-level ``vision_model``.
    assert not any(k.startswith("vision_model.") for k in reverted)


def test_scoped_rule_maps_config_module_names_consistently():
    """``build_reverse_name_mapper`` must apply the same scoping as the weight rename.

    Otherwise ``exclude_modules`` (which lists the BF16 vision tower) lands in a
    different namespace than the weights and a deployment loader silently treats an
    excluded layer as quantized.
    """
    core = pytest.importorskip("transformers.core_model_loading")
    if not hasattr(core, "PrefixChange"):
        pytest.skip("Transformers does not expose PrefixChange")
    # Local import: optional dependency, guarded by the importorskip above.
    from transformers.core_model_loading import PrefixChange

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.vision_tower = torch.nn.Module()
    model.model.vision_tower.encoder = torch.nn.Linear(2, 2, bias=False)
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])

    prefix_change = PrefixChange(prefix_to_remove="vision_model")
    _set_scope_attr(prefix_change, "scope_prefix", "model.vision_tower")
    _set_scope_attr(prefix_change, "base_model_prefix", "model")
    model._weight_conversions = [prefix_change]

    mapper = build_reverse_name_mapper(model)
    assert mapper is not None
    assert mapper("model.vision_tower.encoder") == "model.vision_tower.vision_model.encoder"
    # Sibling namespaces are untouched.
    assert mapper("model.language_model.layers.0") == "model.language_model.layers.0"
    # A trailing-wildcard exclude pattern tracks the same rename its weights got, so the
    # excluded (BF16) vision tower still matches the exported tensor names.
    assert mapper("model.vision_tower*") == "model.vision_tower.vision_model*"


def test_root_scoped_rule_still_faces_shadowing_guard():
    """A ``scope_prefix == ""`` rule has whole-key-space reach and must not bypass #2032.

    ``_scope_prefixes`` keeps an empty candidate for the root scope, which
    ``_sub_scoped`` matches against every key -- so such a rule is as broad as an
    unscoped one. Skipping the shadowing heuristic merely because ``scope_prefixes`` is a
    non-empty *tuple* would reintroduce NVBug 6525534: the nested text model's
    ``^model.language_model.`` reverse would be kept and rewrite ``model.visual.*``.
    """
    pytest.importorskip("transformers.core_model_loading")
    # Local import: optional dependency, guarded by the importorskip above.
    from transformers.core_model_loading import WeightRenaming

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.visual = torch.nn.Module()
    model.model.visual.patch_embed = torch.nn.Linear(2, 2, bias=False)
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])

    renaming = WeightRenaming(
        source_patterns=r"^model.language_model.",
        target_patterns=r"^model.(?!language_model.)",
    )
    # Root scope: reaches every key, exactly like an unscoped rule.
    _set_scope_attr(renaming, "scope_prefix", "")
    _set_scope_attr(renaming, "base_model_prefix", "")
    model._weight_conversions = [renaming]

    state_dict = {
        "model.visual.patch_embed.weight": torch.randn(2, 2),
        "model.language_model.layers.0.weight": torch.randn(2, 2),
    }
    reverted = revert_weight_conversion_quant_aware(model, state_dict)

    # The sibling vision namespace must be untouched (the #2032 guarantee).
    assert set(reverted) == set(state_dict)
    assert not any("language_model.visual" in k for k in reverted)


@pytest.mark.parametrize(
    ("base_prefix", "pattern_suffix", "with_renames"),
    [("", "", False), ("model", ".weight", True), ("model", ".weight$", True)],
)
def test_scoped_vlm_conversion_keeps_tensors_and_config_aligned(
    base_prefix, pattern_suffix, with_renames
):
    """Scoped vision merges and expert renames preserve scales, exclusions, and siblings."""
    pytest.importorskip("transformers.core_model_loading")
    # Local import: optional dependency, guarded by the importorskip above.
    from transformers.core_model_loading import MergeModulelist, WeightConverter, WeightRenaming

    qkv = _radio_qkv_conversion(base_model_prefix=base_prefix, pattern_suffix=pattern_suffix)
    radio_blocks = WeightRenaming("radio_model.model.blocks", "encoder.layer")
    radio_blocks.scope_prefix = qkv.scope_prefix
    _set_scope_attr(radio_blocks, "base_model_prefix", base_prefix)
    projector = WeightRenaming("mlp1", "vision_projector.mlp1")
    expert = WeightConverter(
        source_patterns="mixer.experts.*.w1.weight",
        target_patterns="mixer.experts.up_proj",
        operations=[MergeModulelist(dim=0)],
    )
    _set_scope_attr(expert, "scope_prefix", "language_model")
    if base_prefix and not hasattr(expert, "base_model_prefix"):
        expert.scope_prefix = f"{base_prefix}.language_model"
    _set_scope_attr(expert, "base_model_prefix", base_prefix)
    conversions = [expert, qkv, radio_blocks, projector] if with_renames else [expert, qkv]
    model = types.SimpleNamespace(_weight_conversions=conversions)

    prefix = f"{base_prefix}." if base_prefix else ""
    language = prefix + "language_model.model.layers.0.mixer.experts.0.up_proj"
    sibling = prefix + "other_model.layers.0.mixer.experts.0.up_proj"
    parent = prefix + "vision_model.encoder.layer.0"
    excluded = [f"{parent}.attention.{part}_proj" for part in "qkv"]
    quantized = parent + ".mlp.fc1"
    state = {
        **_nvfp4_linear(language, 8, 16),
        **_nvfp4_linear(sibling, 8, 16),
        **_nvfp4_linear(quantized, 8, 16),
    }
    parts = [torch.full((2, 3), float(i), dtype=torch.bfloat16) for i in range(3)]
    state.update({name + ".weight": tensor for name, tensor in zip(excluded, parts)})
    state["vision_projector.mlp1.0.weight"] = torch.ones(3, 3)
    state[prefix + "language_model.attention.q_proj.weight"] = torch.ones(2, 3)
    out = revert_weight_conversion_quant_aware(model, state)
    mapper = build_reverse_name_mapper(model)
    fused = (
        prefix
        + "vision_model."
        + ("radio_model.model.blocks" if with_renames else "encoder.layer")
        + ".0.attn.qkv"
    )
    torch.testing.assert_close(out[fused + ".weight"], torch.cat(parts))
    assert len(out) == len(state) - 2
    assert not any(name + ".weight" in out for name in excluded)
    for key, value in state.items():
        if key not in {name + ".weight" for name in excluded}:
            expected = key.replace(language, language.replace(".up_proj", ".w1"))
            if with_renames:
                expected = expected.replace("vision_projector.mlp1", "mlp1").replace(
                    "vision_model.encoder.layer", "vision_model.radio_model.model.blocks"
                )
            assert out[expected] is value
    for suffix in ("*", ".*"):
        assert mapper(excluded[0] + suffix) == fused + suffix
    assert mapper(prefix + "vision_model.*") == prefix + "vision_model.*"

    patterns = sorted(_prefix_wildcard_summarize_exclude_modules(excluded, [quantized]))
    assert patterns == [parent + ".attention*"]
    quantized_out = fused.rsplit(".attn.qkv", 1)[0] + ".mlp.fc1"
    for excludes in (excluded, patterns):
        quant = {
            "quantized_layers": {name: {"quant_algo": "NVFP4"} for name in (language, quantized)},
            "exclude_modules": excludes,
            "kv_cache_quantized_layers": {sibling: {"quant_algo": "FP8"}},
        }
        mapped = _revert_hf_quant_config_names(
            {"quantization": quant}, mapper, module_names=[*excluded, language, sibling, quantized]
        )["quantization"]
        assert mapped["exclude_modules"] == [fused] * (3 if excludes == excluded else 1)
        assert not any(fnmatchcase(quantized_out, p) for p in mapped["exclude_modules"])
        assert mapped["quantized_layers"] == {
            language.replace(".up_proj", ".w1"): {"quant_algo": "NVFP4"},
            quantized_out: {"quant_algo": "NVFP4"},
        }
        assert all(name + ".weight" in out for name in mapped["quantized_layers"])
        assert mapped["kv_cache_quantized_layers"] == quant["kv_cache_quantized_layers"]
        assert quant["exclude_modules"] == excludes


def test_radio_merge_that_matches_no_keys_raises():
    """An unused merge rule must not let later renames create a mixed-name checkpoint."""
    model = types.SimpleNamespace(_weight_conversions=[_radio_qkv_conversion()])
    state_dict = {"language_model.layers.0.weight": torch.randn(2, 2)}

    with pytest.raises(
        QuantConversionUnsupportedError,
        match=r"matched no state-dict key .*scope_prefix=.*base_model_prefix=",
    ):
        revert_weight_conversion_quant_aware(model, state_dict)


@pytest.mark.parametrize(
    ("leaf", "tensors", "message"),
    [
        (
            "weight",
            (torch.ones(2, 3), torch.ones(2, 3, dtype=torch.float16), torch.ones(2, 3)),
            "mixed dtypes",
        ),
        (
            "input_scale",
            (torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)),
            "cannot merge quantization state",
        ),
    ],
    ids=["mixed-dtype", "scalar-quantization-state"],
)
def test_radio_merge_rejects_unsafe_tensor_groups(leaf, tensors, message):
    """Unsafe tensor groups must trigger the atomic in-memory-name fallback."""
    model = types.SimpleNamespace(_weight_conversions=[_radio_qkv_conversion()])
    state_dict = {
        f"vision_model.encoder.layer.0.attention.{part}_proj.{leaf}": tensor
        for part, tensor in zip(("q", "k", "v"), tensors)
    }

    with pytest.raises(QuantConversionUnsupportedError, match=message):
        revert_weight_conversion_quant_aware(model, state_dict)


@pytest.mark.parametrize(
    "leaf", ["input_scale", "weight_scale_2", "weight_scale", "weight_scale_inv"]
)
@pytest.mark.parametrize("shape", [(1,), (2, 1)])
@pytest.mark.parametrize("pattern_suffix", ["", ".weight$"])
def test_radio_merge_rejects_quantization_state_without_mutating_input(leaf, shape, pattern_suffix):
    """One-element and blocked scales must not be concatenated into a fused module."""
    model = types.SimpleNamespace(
        _weight_conversions=[_radio_qkv_conversion(pattern_suffix=pattern_suffix)]
    )
    state_dict = {
        f"vision_model.encoder.layer.0.attention.{part}_proj.weight": torch.ones(
            2, 3, dtype=torch.uint8
        )
        for part in ("q", "k", "v")
    }
    state_dict.update(
        {
            f"vision_model.encoder.layer.0.attention.{part}_proj.{leaf}": torch.ones(shape)
            for part in ("q", "k", "v")
        }
    )
    original = {key: tensor.clone() for key, tensor in state_dict.items()}

    with pytest.raises(QuantConversionUnsupportedError, match="cannot merge quantization state"):
        revert_weight_conversion_quant_aware(model, state_dict)

    assert state_dict.keys() == original.keys()
    for key, tensor in state_dict.items():
        torch.testing.assert_close(tensor, original[key])


def test_per_tensor_export_rejects_merge_rules():
    """Streaming and the guard shared with layerwise export reject cross-tensor merges."""
    model = types.SimpleNamespace(_weight_conversions=[_radio_qkv_conversion()])
    for guard in (_assert_no_split_rules, _build_reverse_name_mapper_or_none):
        with pytest.raises(NotImplementedError, match="tensor-level split or merge rules"):
            guard(model)


@pytest.mark.parametrize("export_mode", ["resident", "streaming"])
@pytest.mark.parametrize("scope", ["", "vision_model"])
def test_quantized_weight_specific_rename_keeps_scales_aligned(tmp_path, export_mode, scope):
    """Weight-only patterns carry every scale leaf through subsequent scoped renames."""
    pytest.importorskip("transformers.core_model_loading")
    # Local import: transformers is an optional dependency for ModelOpt.
    from transformers.core_model_loading import WeightRenaming

    weight_rename = WeightRenaming(r"^head\.weight$", "lm_head.weight")
    container_rename = WeightRenaming("output", "head")
    for rule in (weight_rename, container_rename):
        if scope:
            _set_scope_attr(rule, "scope_prefix", scope)
    model = types.SimpleNamespace(_weight_conversions=[container_rename, weight_rename])
    prefix = f"{scope}." if scope else ""
    state = _nvfp4_linear(prefix + "lm_head", 8, 16)
    state[prefix + "lm_head.weight_scale_inv"] = torch.ones(8, 1)
    state["unrelated.weight"] = torch.ones(8, 16)

    if export_mode == "resident":
        written = revert_weight_conversion_quant_aware(model, state)
    else:
        _stream_tensors(model, state, tmp_path)
        written = _load_shards(tmp_path)

    config = {"quantized_layers": {prefix + "lm_head": {"quant_algo": "NVFP4"}}}
    revert_quant_config_names(config, build_reverse_name_mapper(model))
    assert config["quantized_layers"] == {prefix + "output": {"quant_algo": "NVFP4"}}
    assert set(written) == {key.replace(prefix + "lm_head.", prefix + "output.") for key in state}
    for key, value in state.items():
        torch.testing.assert_close(
            written[key.replace(prefix + "lm_head.", prefix + "output.")], value
        )


@pytest.mark.parametrize("export_mode", ["resident", "streaming", "layerwise"])
@pytest.mark.parametrize("quantized_head", [False, True])
def test_export_weight_specific_rename_keeps_config_aligned(tmp_path, quantized_head, export_mode):
    """Every export path keeps renamed BF16/FP8 weights, scales, and config aligned."""
    names = ["model.layers.0.self_attn.q_proj.weight_quantizer"]
    model = _fp8_llama(*names, *(["lm_head.*quantizer"] if quantized_head else []))
    # Local import: _fp8_llama guards the optional dependency.
    from transformers.core_model_loading import WeightRenaming

    model._weight_conversions = [WeightRenaming(r"^head\.weight$", "lm_head.weight")]
    original_head = model.lm_head.weight.detach().clone()
    if export_mode == "layerwise":
        exporter = LayerwiseExporter(model, tmp_path)
        exporter.bind(list(model.model.layers))
        exporter.export_layer(0, model.model.layers[0])
        config = exporter.finalize()
    elif export_mode == "streaming":
        state, config = _export_transformers_checkpoint(model)
        _stream_tensors(model, state, tmp_path)
        config = _revert_quant_config_names_best_effort(model, config)
    else:
        export_hf_checkpoint(model, export_dir=tmp_path, save_modelopt_state=False)
        config = json.loads((tmp_path / "hf_quant_config.json").read_text())
    written = _load_shards(tmp_path)

    assert ("head" in config["quantization"]["exclude_modules"]) is not quantized_head
    assert "lm_head" not in config["quantization"]["exclude_modules"]
    assert not any(key.startswith("lm_head.") for key in written)
    assert "model.layers.0.self_attn.q_proj.weight" in written
    if quantized_head:
        assert written["head.weight"].dtype == torch.float8_e4m3fn
        assert "head.weight_scale" in written
        assert "head.input_scale" in written
    else:
        torch.testing.assert_close(written["head.weight"], original_head)


@pytest.mark.parametrize("quantized_merge", [False, True])
def test_resident_export_merge_fallback_warns_and_keeps_names_aligned(tmp_path, quantized_merge):
    """Export applies supported merges or warns and retains all weight/config namespaces."""
    attention = "model.layers.0.self_attn"
    names = ["lm_head.*quantizer"]
    model = _fp8_llama(*names, *([f"{attention}.q_proj.*quantizer"] if quantized_merge else []))
    # Local import: _fp8_llama guards the optional dependency.
    from transformers.core_model_loading import Chunk, WeightConverter, WeightRenaming

    # This unscoped converter also exercises merges on versions without scope support.
    model._weight_conversions = [
        WeightConverter(
            source_patterns=f"{attention}.qkv.weight",
            target_patterns=[f"{attention}.{part}_proj.weight" for part in "qkv"],
            operations=[Chunk(dim=0)],
        ),
        WeightRenaming(r"^head\.weight$", "lm_head.weight"),
    ]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        export_hf_checkpoint(model, export_dir=tmp_path, save_modelopt_state=False)
    messages = [
        str(w.message) for w in captured if "reverse weight conversion skipped" in str(w.message)
    ]
    written = _load_shards(tmp_path)
    config = json.loads((tmp_path / "hf_quant_config.json").read_text())["quantization"]
    if quantized_merge:
        assert len(messages) == 1
        assert attention + ".q_proj" in messages[0]
        assert "including unrelated submodels" in messages[0]
        assert "may fail to load or skip these weights" in messages[0]
        assert attention + ".qkv.weight" not in written
        assert all(f"{attention}.{part}_proj.weight" in written for part in "qkv")
        assert any(fnmatchcase(attention + ".k_proj", p) for p in config["exclude_modules"])
        assert not any(fnmatchcase(attention + ".q_proj", p) for p in config["exclude_modules"])
        head, other_head = "lm_head", "head"
    else:
        assert not messages
        assert written[attention + ".qkv.weight"].shape == (96, 32)
        assert not any(f"{attention}.{part}_proj.weight" in written for part in "qkv")
        assert any(fnmatchcase(attention + ".qkv", p) for p in config["exclude_modules"])
        head, other_head = "head", "lm_head"
    assert all(head + leaf in written for leaf in (".weight", ".weight_scale", ".input_scale"))
    assert not any(key.startswith(other_head + ".") for key in written)
    assert not any(fnmatchcase(head, p) for p in config["exclude_modules"])


def test_radio_merge_requires_converter_rename_source_key():
    """A transformers version without the bound rename helper must fall back cleanly."""
    pytest.importorskip("transformers.core_model_loading")
    # Local import: optional dependency, guarded by the importorskip above.
    from transformers.core_model_loading import Chunk, WeightConverter

    class ConverterWithoutRename(WeightConverter):
        rename_source_key = None

    converter = ConverterWithoutRename(
        source_patterns="attn.qkv",
        target_patterns=["attention.q_proj", "attention.k_proj", "attention.v_proj"],
        operations=[Chunk(dim=0)],
    )
    model = types.SimpleNamespace(_weight_conversions=[converter])

    with pytest.raises(QuantConversionUnsupportedError, match="rename_source_key is unavailable"):
        revert_weight_conversion_quant_aware(model, {})


def test_split_collision_raises():
    """A split whose target key already exists must fail instead of overwriting."""
    sd = _nvfp4_linear("m.gate_up_proj", 8, 16)
    sd["m.gate_proj.weight"] = torch.zeros(4, 16)  # pre-existing split target
    rule = SplitRule(".gate_up_proj", (".gate_proj", ".up_proj"), dim=0)
    with pytest.raises(QuantConversionUnsupportedError, match="split collision"):
        apply_reverse_rules(sd, [rule], [])


def test_stacked_experts_guard():
    """Experts not pre-expanded (stacked/fused 3-D leaf) must trigger the fallback.

    The per-expert-index leaf renames cannot rewrite a still-fused
    ``.experts.gate_up_proj`` tensor, so it would ship mis-named; guard by raising.
    """
    fused_leaves = ["gate_up_proj", "down_proj"]

    # Pre-expanded 2-D experts: no fused leaf present -> no raise.
    ok = _nvfp4_linear("model.language_model.layers.10.mlp.experts.0.gate_proj", 8, 16)
    _assert_experts_pre_expanded(ok, fused_leaves)

    # Still-fused stacked expert leaf (3-D) -> raise.
    bad = {"model.language_model.layers.10.mlp.experts.gate_up_proj.weight": torch.zeros(2, 8, 16)}
    with pytest.raises(QuantConversionUnsupportedError, match="not pre-expanded"):
        _assert_experts_pre_expanded(bad, fused_leaves)

    # No expert converters in the mapping -> guard is a no-op even for 3-D tensors.
    _assert_experts_pre_expanded(bad, [])


def test_revert_quant_config_names_mapper():
    """exclude_modules / quantized_layers keys revert to hub names, preserving wildcards.

    Regression for the bug where the reverse conversion renamed weight tensors to hub
    names but left the quant-config module references in the in-memory namespace, so a
    deployment loader matched none of the excludes and loaded an excluded BF16 layer as
    quantized. Uses Mixtral's real mapping (``mlp.experts`` <-> ``block_sparse_moe.experts``).
    """
    # Import stays function-local: the helper needs transformers, which unit tests run without.
    pytest.importorskip("transformers.core_model_loading")
    from _test_utils.torch.transformers_models import get_tiny_mixtral

    model = get_tiny_mixtral(**_MIXTRAL_KWARGS)
    mapper = build_reverse_name_mapper(model)
    assert mapper is not None

    quant = {
        "quant_algo": "NVFP4",
        "exclude_modules": [
            "model.layers.0.self_attn*",  # no container rename -> unchanged, wildcard kept
            "model.layers.0.mlp.experts.0*",  # in-memory -> block_sparse_moe.experts, wildcard kept
            "lm_head",
        ],
        "quantized_layers": {"model.layers.0.mlp.experts.0.w1": {"quant_algo": "NVFP4"}},
        "kv_cache_quantized_layers": {"model.layers.0.mlp.experts.0": {"quant_algo": "FP8"}},
    }
    revert_quant_config_names(
        quant, mapper, module_names=(name for name, _ in model.named_modules())
    )
    assert quant["exclude_modules"] == [
        "model.layers.0.self_attn*",
        "model.layers.0.block_sparse_moe.experts.0*",
        "lm_head",
    ]
    assert "model.layers.0.block_sparse_moe.experts.0.w1" in quant["quantized_layers"]
    assert "model.layers.0.block_sparse_moe.experts.0" in quant["kv_cache_quantized_layers"]
    # mapper(None) is a no-op
    q2 = {"exclude_modules": ["x*"]}
    revert_quant_config_names(q2, None)
    assert q2["exclude_modules"] == ["x*"]
