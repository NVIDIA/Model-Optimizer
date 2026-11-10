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
    """A draft state dict with the positional names ``nn.Sequential`` produces.

    Every tensor gets a DISTINCT fill value. Several entries share a shape --
    ``feature_mlp.0.weight``/``.bias`` are both (5,), and ``feature_mlp.1.bias``,
    ``feature_mlp.3.bias`` and ``layers.N.mlp.2.bias`` are all (8,) -- so a rename that
    swapped two of them would be invisible to a shape-only check. The values make a swap
    detectable, which is the whole point of a test guarding a name mapping.
    """
    v = iter(range(1, 1000))

    def t(*shape):
        return torch.full(shape, float(next(v)))

    sd = {
        "dflash_module.lilicorr.feature_mlp.0.weight": t(5),
        "dflash_module.lilicorr.feature_mlp.0.bias": t(5),
        "dflash_module.lilicorr.feature_mlp.1.weight": t(8, 5),
        "dflash_module.lilicorr.feature_mlp.1.bias": t(8),
        "dflash_module.lilicorr.feature_mlp.3.weight": t(8, 8),
        "dflash_module.lilicorr.feature_mlp.3.bias": t(8),
        # Names that must pass through untouched.
        "dflash_module.lilicorr.out_head.weight": t(4, 8),
        "dflash_module.lilicorr.slot_embedding": t(1, 1, 3, 1, 8),
        "dflash_module.layers.0.self_attn.q_proj.weight": t(8, 8),
    }
    for i in range(num_layers):
        sd[f"dflash_module.lilicorr.layers.{i}.mlp.0.weight"] = t(16, 8)
        sd[f"dflash_module.lilicorr.layers.{i}.mlp.0.bias"] = t(16)
        sd[f"dflash_module.lilicorr.layers.{i}.mlp.2.weight"] = t(8, 16)
        sd[f"dflash_module.lilicorr.layers.{i}.mlp.2.bias"] = t(8)
        # Attention already matches both loaders; it must not be renamed.
        sd[f"dflash_module.lilicorr.layers.{i}.attn.in_proj_weight"] = t(24, 8)
    return sd


# Exported name -> the raw name its tensor must have come from. Anything not listed here
# is expected to keep its name (minus the dflash_module. prefix).
_EXPECTED_SOURCE = {
    "lilicorr.feature_norm.weight": "lilicorr.feature_mlp.0.weight",
    "lilicorr.feature_norm.bias": "lilicorr.feature_mlp.0.bias",
    "lilicorr.feature_mlp.up_proj.weight": "lilicorr.feature_mlp.1.weight",
    "lilicorr.feature_mlp.up_proj.bias": "lilicorr.feature_mlp.1.bias",
    "lilicorr.feature_mlp.down_proj.weight": "lilicorr.feature_mlp.3.weight",
    "lilicorr.feature_mlp.down_proj.bias": "lilicorr.feature_mlp.3.bias",
    **{
        f"lilicorr.layers.{i}.mlp.{dst}.{suffix}": f"lilicorr.layers.{i}.mlp.{src}.{suffix}"
        for i in (0, 1)
        for src, dst in (("0", "up_proj"), ("2", "down_proj"))
        for suffix in ("weight", "bias")
    },
}


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
    """Every exported tensor is the SAME tensor its name says it came from.

    Shape equality alone cannot catch a swap between same-shaped entries, so compare
    values: each exported tensor must equal the raw tensor it was mapped from.
    """
    raw = _raw_state_dict()
    out = _export(raw)
    assert len(out) == len(raw)

    stripped = {k.split("dflash_module.", 1)[1]: v for k, v in raw.items()}
    for exported_name, tensor in out.items():
        source_name = _EXPECTED_SOURCE.get(exported_name, exported_name)
        assert source_name in stripped, f"{exported_name} maps to unknown source {source_name}"
        expected = stripped[source_name]
        assert tensor.shape == expected.shape, exported_name
        assert torch.equal(tensor, expected), (
            f"{exported_name} holds the wrong tensor: got fill {tensor.flatten()[0].item()}, "
            f"expected {expected.flatten()[0].item()} from {source_name}"
        )

    # And every renamed source is consumed exactly once -- no raw name left behind.
    for source_name in _EXPECTED_SOURCE.values():
        assert source_name not in out, f"{source_name} survived the rename"


def test_unrelated_names_pass_through_untouched():
    out = _export(_raw_state_dict())
    # Attention uses the same names on both sides already; renaming it would break loading.
    assert "lilicorr.layers.0.attn.in_proj_weight" in out
    assert "lilicorr.out_head.weight" in out
    assert "lilicorr.slot_embedding" in out
    assert "layers.0.self_attn.q_proj.weight" in out


def test_quantized_scale_leaves_are_renamed_with_their_linear():
    """A quantized linear's scale tensors must follow it through the rename.

    ``_export_transformers_checkpoint`` registers ``weight_scale`` (and, depending on the
    format, ``input_scale`` / ``weight_scale_2``) next to each packed ``weight``. These
    hang off the same positional module path as the weight, so a rename anchored on
    ``(weight|bias)$`` would leave them behind -- and the positional-name guard would then
    reject the export of a model that used to export fine, only with the wrong names.
    """
    quant_leaves = ("weight_scale", "weight_scale_2", "input_scale")
    raw = _raw_state_dict()
    for leaf in quant_leaves:
        raw[f"dflash_module.lilicorr.feature_mlp.1.{leaf}"] = torch.ones(1)
        raw[f"dflash_module.lilicorr.feature_mlp.3.{leaf}"] = torch.ones(1)
        raw[f"dflash_module.lilicorr.layers.0.mlp.0.{leaf}"] = torch.ones(1)
        raw[f"dflash_module.lilicorr.layers.1.mlp.2.{leaf}"] = torch.ones(1)

    out = _export(raw)  # must not raise

    for leaf in quant_leaves:
        assert f"lilicorr.feature_mlp.up_proj.{leaf}" in out
        assert f"lilicorr.feature_mlp.down_proj.{leaf}" in out
        assert f"lilicorr.layers.0.mlp.up_proj.{leaf}" in out
        assert f"lilicorr.layers.1.mlp.down_proj.{leaf}" in out
    assert len(out) == len(raw)
    assert not [k for k in out if ".feature_mlp.1." in k or ".mlp.0." in k]


def test_quant_config_module_references_use_the_export_names():
    """The quantization config must name the tensors the checkpoint actually contains.

    ``_map_export_name`` is handed to ``revert_quant_config_names`` on the export path, so
    config references (exclusions, mixed-precision overrides) have to come back renamed the
    same way the tensors did -- otherwise they point at modules no longer in the file.
    """
    exporter = object.__new__(LiLiCorrExporter)
    cases = {
        "dflash_module.lilicorr.feature_mlp.0": "lilicorr.feature_norm",
        "dflash_module.lilicorr.feature_mlp.1": "lilicorr.feature_mlp.up_proj",
        "dflash_module.lilicorr.feature_mlp.3": "lilicorr.feature_mlp.down_proj",
        "dflash_module.lilicorr.layers.3.mlp.0": "lilicorr.layers.3.mlp.up_proj",
        "dflash_module.lilicorr.layers.3.mlp.2": "lilicorr.layers.3.mlp.down_proj",
        # A bare module path, no leaf -- what the config carries.
        "dflash_module.lilicorr.layers.0.attn": "lilicorr.layers.0.attn",
    }
    for raw_name, expected in cases.items():
        assert exporter._map_export_name(raw_name) == expected, raw_name

    # Not a draft module: left entirely alone.
    assert exporter._map_export_name("model.layers.0.mlp.0") == "model.layers.0.mlp.0"


def test_similar_names_are_not_renamed_by_prefix_accident():
    """The rename pins to a whole path component, so lookalikes are untouched."""
    exporter = object.__new__(LiLiCorrExporter)
    for name in (
        "dflash_module.lilicorr.feature_mlp.10.weight",  # 10, not 1
        "dflash_module.lilicorr.feature_mlp.2.weight",  # SiLU index, no params
        "dflash_module.lilicorr_extra.feature_mlp.1.weight",  # different module
    ):
        mapped = exporter._map_export_name(name)
        assert "up_proj" not in mapped and "down_proj" not in mapped, name
