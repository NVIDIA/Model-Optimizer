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

import pytest
import torch

from modelopt.torch.export.quant_aware_conversion import (
    QuantConversionUnsupportedError,
    RenameRule,
    SplitRule,
    apply_reverse_rules,
)

BLOCK = 16


def _nvfp4_linear(module: str, out: int, in_features: int) -> dict[str, torch.Tensor]:
    """Synthetic NVFP4 quantized-linear tensor group keyed under ``module``."""
    return {
        f"{module}.weight": torch.randint(0, 255, (out, in_features // 2), dtype=torch.uint8),
        f"{module}.weight_scale": torch.randn(out, in_features // BLOCK),
        f"{module}.weight_scale_2": torch.tensor(0.037, dtype=torch.float32),  # 0-d
        f"{module}.input_scale": torch.tensor(1.0, dtype=torch.float32),  # 0-d
    }


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
    pytest.importorskip("transformers")
    from transformers import MixtralConfig, MixtralForCausalLM

    try:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    except ImportError:
        pytest.skip("transformers build has no conversion_mapping API")
    if not get_checkpoint_conversion_mapping("mixtral"):
        pytest.skip("transformers build has no mixtral conversion_mapping")

    from modelopt.torch.export.quant_aware_conversion import revert_weight_conversion_quant_aware

    cfg = MixtralConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=2,
        vocab_size=64,
        max_position_embeddings=64,
    )
    model = MixtralForCausalLM(cfg)

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
