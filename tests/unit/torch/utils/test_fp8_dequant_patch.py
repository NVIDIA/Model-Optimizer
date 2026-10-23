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

"""``_patch_finegrained_fp8_dequant``: transformers' block-FP8 dequant, fixed for DeepSeek-V3.

transformers 5.9's ``Fp8Dequantize._dequantize_one`` finds the block size by dividing the weight
by the scale grid, which breaks on DeepSeek-V3's partial last blocks (a 576-row weight under a
5-row grid of 128-row blocks). The patch installs a tile-and-crop dequant instead, and only below
transformers 5.10.1, so the tests present an older version to it. The dequant is checked against a
plain blockwise reference: ``weight[..., i, j] * scale_inv[..., i // block_n, j // block_k]``.
"""

from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version

transformers = pytest.importorskip("transformers")
finegrained_fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
quantizer_fp8 = pytest.importorskip("transformers.quantizers.quantizer_finegrained_fp8")

from transformers import FineGrainedFP8Config
from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter

import modelopt.torch.utils.plugins.model_load_utils as model_load_utils

if not hasattr(finegrained_fp8, "Fp8Dequantize"):
    pytest.skip("transformers has no Fp8Dequantize", allow_module_level=True)
if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("torch has no float8_e4m3fn", allow_module_level=True)

# The installed version: the fixture below makes the patch see an older one.
_INSTALLED_TRANSFORMERS = Version(transformers.__version__)
Fp8Dequantize = finegrained_fp8.Fp8Dequantize
FineGrainedFP8HfQuantizer = quantizer_fp8.FineGrainedFP8HfQuantizer

_MISSING = object()
# What the patch may rebind: the dequant method, and the FP8 quantizer's converter hook plus the
# flag guarding it.
_PATCH_TARGETS = (
    (Fp8Dequantize, "_dequantize_one"),
    (FineGrainedFP8HfQuantizer, "update_weight_conversions"),
    (FineGrainedFP8HfQuantizer, "_modelopt_uwc_patched"),
)


@pytest.fixture
def pre_5_10_transformers(monkeypatch):
    """Unpatched transformers, reporting a version the patch targets; restored afterwards."""
    saved = [(cls, name, vars(cls).get(name, _MISSING)) for cls, name in _PATCH_TARGETS]
    monkeypatch.setattr(transformers, "__version__", "5.9.0")
    monkeypatch.setattr(model_load_utils, "_FINEGRAINED_FP8_PATCHED", False)
    monkeypatch.setattr(model_load_utils, "_FP8_DEQUANT_DTYPE", None)
    yield
    for cls, name, value in saved:
        if value is not _MISSING:
            setattr(cls, name, value)
        elif name in vars(cls):
            delattr(cls, name)


def _patched_state():
    return [vars(cls).get(name, _MISSING) for cls, name in _PATCH_TARGETS]


def _dequant_op(block_size):
    """The dequant op of an FP8 quantizer configured with ``weight_block_size=block_size``."""
    config = SimpleNamespace(weight_block_size=block_size)
    return Fp8Dequantize(SimpleNamespace(quantization_config=config))


def _block_fp8(shape, grid, seed=0):
    """A random fp8 weight and its fp32 ``scale_inv`` grid."""
    gen = torch.Generator().manual_seed(seed)
    weight = (4 * torch.randn(shape, generator=gen)).to(torch.float8_e4m3fn)
    return weight, torch.rand(grid, generator=gen) + 0.5


def _ceil_grid(shape, block):
    *lead, rows, cols = shape
    return (*lead, -(-rows // block[0]), -(-cols // block[1]))


def _reference(weight, scale_inv, block):
    """``weight[..., i, j] * scale_inv[..., i // block_n, j // block_k]``, in fp32."""
    rows, cols = weight.shape[-2:]
    scale = scale_inv[..., torch.arange(rows) // block[0], :][..., torch.arange(cols) // block[1]]
    return weight.float() * scale


@pytest.mark.parametrize(
    ("shape", "block", "configured", "grid"),
    [
        ((8, 12), (4, 4), (4, 4), None),
        ((10, 13), (4, 4), (4, 4), None),
        ((9, 8), (4, 4), (4, 4), None),
        ((6, 10), (2, 4), (2, 4), None),
        ((3, 10, 13), (4, 4), (4, 4), None),
        ((576, 7168), (128, 128), (128, 128), None),
        ((8, 8), (4, 2), (128, 128), (2, 4)),
        ((130, 260), (128, 128), None, (2, 3)),
        ((0, 13), (4, 4), (4, 4), None),
    ],
    ids=[
        "whole-blocks",
        "partial-last-blocks",
        "grid-divides-the-rows",
        "non-square-blocks",
        "per-expert",
        "deepseek-kv-a-proj",
        "grid-finer-than-configured",
        "nothing-configured",
        "empty-weight",
    ],
)
def test_dequant_matches_the_blockwise_reference(
    pre_5_10_transformers, shape, block, configured, grid
):
    """``configured`` is the quantization config's block size, ``grid`` the scale grid when it is not
    that block's ceil grid. ``grid-divides-the-rows``: 9 rows of 4-row blocks make a 3-row grid,
    which also divides 9, yet the blocks are 4 rows, not 3. A grid that is not the configured
    block's ceil grid sets the block by division; with nothing configured, DeepSeek's 128 x 128
    applies."""
    weight, scale_inv = _block_fp8(shape, grid or _ceil_grid(shape, block))
    model_load_utils._patch_finegrained_fp8_dequant(torch.float32)

    out = _dequant_op(configured)._dequantize_one(weight, scale_inv)

    assert torch.equal(out, _reference(weight, scale_inv, block))


@pytest.mark.parametrize(
    ("target_dtype", "expected"),
    [(None, torch.bfloat16), (torch.float16, torch.float16)],
    ids=["unrequested", "requested"],
)
def test_dequant_emits_the_requested_model_dtype(pre_5_10_transformers, target_dtype, expected):
    """Not the scales' dtype: DeepSeek's fp32 ``weight_scale_inv`` would give fp32 weights that
    then meet bf16 activations in the calibration matmuls. Unrequested, it is bf16."""
    weight, scale_inv = _block_fp8((10, 13), (3, 4))
    model_load_utils._patch_finegrained_fp8_dequant(target_dtype)

    out = _dequant_op((4, 4))._dequantize_one(weight, scale_inv)

    assert out.dtype == expected
    assert torch.equal(out, _reference(weight, scale_inv, (4, 4)).to(expected))


@pytest.mark.parametrize(
    ("grid", "match"),
    [((0, 4), "empty scale grid"), ((3, 0), "empty scale grid"), ((4, 4), "fits neither")],
    ids=["no-row-scales", "no-col-scales", "grid-no-block-places"],
)
def test_dequant_refuses_a_scale_grid_it_cannot_place(pre_5_10_transformers, grid, match):
    """Rather than load wrong weights: an empty grid for a non-empty weight (the weight and its
    scales were sliced apart, as 5.9's expert ``tensor_idx`` bug did), or a 4-row grid over 10 rows,
    which no 4-row block (the configured one) or even split gives."""
    weight, scale_inv = _block_fp8((10, 13), grid)
    model_load_utils._patch_finegrained_fp8_dequant(torch.float32)

    with pytest.raises(ValueError, match=match):
        _dequant_op((4, 4))._dequantize_one(weight, scale_inv)


def test_the_expert_converter_leads_with_a_merge_that_dequantizes_each_expert(
    pre_5_10_transformers,
):
    """transformers prepends ``Fp8Dequantize`` to the routed-expert converter, so its first op is
    no longer the ``MergeModulelist`` the loader keys each expert's ``tensor_idx`` on, and the EP
    load then slices every expert by rows. The patch fuses the two into one ``MergeModulelist``
    that dequantizes each expert with its own scales and drops them before ``Concatenate``."""
    if not hasattr(FineGrainedFP8HfQuantizer, "update_weight_conversions"):
        pytest.skip("this transformers' FP8 quantizer prepends no dequant to the expert converter")
    stock_dequant = Fp8Dequantize._dequantize_one
    model_load_utils._patch_finegrained_fp8_dequant(torch.float32)
    # This transformers' chain passes output_dtype, which the patched dequant (written for the
    # versions it installs on) does not take; its arithmetic is tested above.
    Fp8Dequantize._dequantize_one = stock_dequant
    experts = WeightConverter(
        source_patterns=["mlp.experts.*.gate_proj.weight", "mlp.experts.*.up_proj.weight"],
        target_patterns="mlp.experts.gate_up_proj",
        operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
    )
    quantizer = FineGrainedFP8HfQuantizer(
        FineGrainedFP8Config(weight_block_size=(4, 4), dequantize=True), pre_quantized=True
    )

    [converter] = [
        conv
        for conv in quantizer.update_weight_conversions([experts])
        if conv.target_patterns == ["mlp.experts.gate_up_proj"]
    ]

    assert isinstance(converter.operations[0], MergeModulelist)
    blocks = {  # every expert's weight and scales differ, so a mixed-up pair cannot pass
        name: [_block_fp8((8, 12), (2, 3), seed=3 * p + e) for e in range(3)]
        for p, name in enumerate(("gate_proj", "up_proj"))
    }
    tensors = {}  # what the loader collects for each of the converter's source patterns
    for name, pairs in blocks.items():
        tensors[f"mlp.experts.*.{name}.weight$"] = [weight for weight, _ in pairs]
        tensors[f"mlp.experts.*.{name}.weight_scale_inv$"] = [scale_inv for _, scale_inv in pairs]
    for op in converter.operations:
        tensors = op.convert(
            tensors,
            source_patterns=converter.source_patterns,
            target_patterns=converter.target_patterns,
        )
    [fused] = tensors.values()
    expected = torch.cat(
        [
            torch.stack([_reference(weight, scale_inv, (4, 4)) for weight, scale_inv in pairs])
            for pairs in blocks.values()
        ],
        dim=1,
    )
    assert torch.equal(fused, expected.to(fused.dtype))


def test_patching_again_keeps_the_install_and_takes_the_new_dtype(pre_5_10_transformers):
    """Every block-fp8 load patches again, with its own model dtype."""
    model_load_utils._patch_finegrained_fp8_dequant(torch.bfloat16)
    installed = _patched_state()

    model_load_utils._patch_finegrained_fp8_dequant(torch.float16)

    assert _patched_state() == installed
    weight, scale_inv = _block_fp8((10, 13), (3, 4))
    assert _dequant_op((4, 4))._dequantize_one(weight, scale_inv).dtype == torch.float16


def test_a_reimported_module_does_not_wrap_the_quantizer_hook_twice(
    pre_5_10_transformers, monkeypatch
):
    """Re-importing ``model_load_utils`` loses its module flag; the quantizer class carries its own."""
    if not hasattr(FineGrainedFP8HfQuantizer, "update_weight_conversions"):
        pytest.skip("this transformers' FP8 quantizer has no update_weight_conversions")
    model_load_utils._patch_finegrained_fp8_dequant()
    hook = vars(FineGrainedFP8HfQuantizer)["update_weight_conversions"]
    monkeypatch.setattr(model_load_utils, "_FINEGRAINED_FP8_PATCHED", False)

    model_load_utils._patch_finegrained_fp8_dequant()

    assert vars(FineGrainedFP8HfQuantizer)["update_weight_conversions"] is hook


def test_undoing_the_patch_restores_stock_transformers(pre_5_10_transformers, monkeypatch):
    """The patch rebinds only the dequant method and, where the quantizer has one, its converter
    hook and guard flag. Putting those back restores the stock classes exactly, after which the
    patch installs again once the module flag is cleared."""
    has_hook = hasattr(FineGrainedFP8HfQuantizer, "update_weight_conversions")
    classes = (Fp8Dequantize, FineGrainedFP8HfQuantizer)
    stock = {cls: dict(vars(cls)) for cls in classes}

    model_load_utils._patch_finegrained_fp8_dequant()

    changed = {
        (cls, name)
        for cls in classes
        for name in stock[cls].keys() | vars(cls).keys()
        if vars(cls).get(name, _MISSING) is not stock[cls].get(name, _MISSING)
    }
    assert changed == set(_PATCH_TARGETS if has_hook else _PATCH_TARGETS[:1])
    for cls, name in changed:
        if name in stock[cls]:
            setattr(cls, name, stock[cls][name])
        else:
            delattr(cls, name)
    assert all(dict(vars(cls)) == stock[cls] for cls in classes)

    monkeypatch.setattr(model_load_utils, "_FINEGRAINED_FP8_PATCHED", False)
    model_load_utils._patch_finegrained_fp8_dequant()
    assert vars(Fp8Dequantize).get("_dequantize_one") is not stock[Fp8Dequantize].get(
        "_dequantize_one"
    )


@pytest.mark.parametrize(
    ("shape", "grid"), [((576, 7168), (5, 56)), ((512, 256), (4, 2))], ids=["partial", "whole"]
)
def test_from_5_10_1_its_own_dequant_also_takes_partial_last_blocks(
    pre_5_10_transformers, monkeypatch, shape, grid
):
    """Upstream fixed the expert converter there and decodes ue8m0 and FP4 scales, which the override
    does not, so its dequant is kept and only wrapped. A partial last block (DeepSeek-V3's 576-row
    kv_a_proj_with_mqa under a 5-row grid of 128-row blocks) is padded to whole blocks for it and
    cropped back; a weight of whole blocks goes to it as is."""
    if Version("5.10.1") > _INSTALLED_TRANSFORMERS:
        pytest.skip("needs transformers' own dequant from 5.10.1")
    monkeypatch.setattr(transformers, "__version__", "5.10.1")
    hook = vars(FineGrainedFP8HfQuantizer).get("update_weight_conversions", _MISSING)
    weight, scale_inv = _block_fp8(shape, grid)

    model_load_utils._patch_finegrained_fp8_dequant(torch.bfloat16)
    out = _dequant_op((128, 128))._dequantize_one(weight, scale_inv, torch.float32)

    assert torch.equal(out, _reference(weight, scale_inv, (128, 128)))
    assert vars(FineGrainedFP8HfQuantizer).get("update_weight_conversions", _MISSING) is hook
