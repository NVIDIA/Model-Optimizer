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

"""The LiLiCorr export must name the reranker the way the serving loaders do.

``LiLiCorrHead`` builds its MLPs as ``nn.Sequential``, so the raw parameter names are
positional. Both serving stacks build the same arithmetic as named submodules and lift
the feature LayerNorm out to a sibling:

    Sequential(LayerNorm, Linear, SiLU, Linear)  ==  feature_norm + up_proj + silu + down_proj

Exporting the positional names yields a checkpoint neither loader can resolve:

    vLLM    ValueError: There is no module or parameter named 'feature_mlp.0' in
            LiLiCorrHead. The available parameters belonging to feature_mlp
            (NemotronHMLP) are: {'feature_mlp.down_proj...', ...}
    SGLang  ValueError: LiLiCorr checkpoint is missing 14 head parameters
            (e.g. lilicorr.feature_mlp.down_proj.bias, ...)
"""

import torch

from modelopt.torch.export.plugins.hf_spec_export import LiLiCorrExporter


def _raw_state_dict(num_layers: int = 2):
    """A draft state dict with the positional names ``nn.Sequential`` produces."""
    sd = {
        "dflash_module.lilicorr.feature_mlp.0.weight": torch.zeros(5),
        "dflash_module.lilicorr.feature_mlp.0.bias": torch.zeros(5),
        "dflash_module.lilicorr.feature_mlp.1.weight": torch.zeros(8, 5),
        "dflash_module.lilicorr.feature_mlp.1.bias": torch.zeros(8),
        "dflash_module.lilicorr.feature_mlp.3.weight": torch.zeros(8, 8),
        "dflash_module.lilicorr.feature_mlp.3.bias": torch.zeros(8),
        # Names that must pass through untouched.
        "dflash_module.lilicorr.out_head.weight": torch.zeros(4, 8),
        "dflash_module.lilicorr.slot_embedding": torch.zeros(1, 1, 3, 1, 8),
        "dflash_module.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8),
    }
    for i in range(num_layers):
        sd[f"dflash_module.lilicorr.layers.{i}.mlp.0.weight"] = torch.zeros(16, 8)
        sd[f"dflash_module.lilicorr.layers.{i}.mlp.0.bias"] = torch.zeros(16)
        sd[f"dflash_module.lilicorr.layers.{i}.mlp.2.weight"] = torch.zeros(8, 16)
        sd[f"dflash_module.lilicorr.layers.{i}.mlp.2.bias"] = torch.zeros(8)
        # Attention already matches both loaders; it must not be renamed.
        sd[f"dflash_module.lilicorr.layers.{i}.attn.in_proj_weight"] = torch.zeros(24, 8)
    return sd


def _export(sd):
    """Run the rename without constructing a real exporter.

    ``_extract_state_dict`` only reads its argument and the class-level rename table, so
    an uninitialised instance is enough -- and avoids needing a converted model here.
    """
    exporter = object.__new__(LiLiCorrExporter)
    return exporter._extract_state_dict(sd)


def test_reranker_mlps_are_renamed_for_the_serving_loaders():
    out = _export(_raw_state_dict())

    # The names both stacks look for.
    for name in (
        "lilicorr.feature_norm.weight",
        "lilicorr.feature_norm.bias",
        "lilicorr.feature_mlp.up_proj.weight",
        "lilicorr.feature_mlp.up_proj.bias",
        "lilicorr.feature_mlp.down_proj.weight",
        "lilicorr.feature_mlp.down_proj.bias",
        "lilicorr.layers.0.mlp.up_proj.weight",
        "lilicorr.layers.1.mlp.down_proj.bias",
    ):
        assert name in out, f"{name} missing from the export"

    # No positional MLP child may survive.
    assert not [k for k in out if ".mlp.0." in k or ".mlp.1." in k or ".mlp.3." in k]


def test_rename_preserves_every_tensor_and_its_shape():
    raw = _raw_state_dict()
    out = _export(raw)
    assert len(out) == len(raw)
    # Shapes are carried over, so a rename can never be mistaken for a reshape.
    assert out["lilicorr.feature_mlp.up_proj.weight"].shape == torch.Size([8, 5])
    assert out["lilicorr.feature_mlp.down_proj.weight"].shape == torch.Size([8, 8])
    assert out["lilicorr.feature_norm.weight"].shape == torch.Size([5])


def test_unrelated_names_pass_through_untouched():
    out = _export(_raw_state_dict())
    # Attention uses the same names on both sides already; renaming it would break loading.
    assert "lilicorr.layers.0.attn.in_proj_weight" in out
    assert "lilicorr.out_head.weight" in out
    assert "lilicorr.slot_embedding" in out
    assert "layers.0.self_attn.q_proj.weight" in out
