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

transformers = pytest.importorskip("transformers")
finegrained_fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
quantizer_fp8 = pytest.importorskip("transformers.quantizers.quantizer_finegrained_fp8")

import modelopt.torch.utils.moe as moe

if not hasattr(finegrained_fp8, "Fp8Dequantize"):
    pytest.skip("transformers has no Fp8Dequantize", allow_module_level=True)
if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("torch has no float8_e4m3fn", allow_module_level=True)

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
    monkeypatch.setattr(moe, "_FINEGRAINED_FP8_PATCHED", False)
    monkeypatch.setattr(moe, "_FP8_DEQUANT_DTYPE", None)
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


def _block_fp8(shape, grid):
    """A random fp8 weight and its fp32 ``scale_inv`` grid."""
    gen = torch.Generator().manual_seed(0)
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
    ("shape", "block"),
    [
        ((8, 12), (4, 4)),
        ((10, 13), (4, 4)),
        ((9, 8), (4, 4)),
        ((6, 10), (2, 4)),
        ((3, 10, 13), (4, 4)),
        ((576, 7168), (128, 128)),
    ],
    ids=[
        "whole-blocks",
        "partial-last-blocks",
        "grid-divides-the-rows",
        "non-square-blocks",
        "per-expert",
        "deepseek-kv-a-proj",
    ],
)
def test_dequant_matches_the_blockwise_reference(pre_5_10_transformers, shape, block):
    """``grid-divides-the-rows``: 9 rows of 4-row blocks make a 3-row grid, which also divides 9,
    yet the blocks are 4 rows, not 3."""
    weight, scale_inv = _block_fp8(shape, _ceil_grid(shape, block))
    moe._patch_finegrained_fp8_dequant(torch.float32)

    out = _dequant_op(block)._dequantize_one(weight, scale_inv)

    assert torch.equal(out, _reference(weight, scale_inv, block))


@pytest.mark.parametrize(
    ("configured", "shape", "grid", "block"),
    [((128, 128), (8, 8), (2, 4), (4, 2)), (None, (130, 260), (2, 3), (128, 128))],
    ids=["grid-finer-than-configured", "nothing-configured"],
)
def test_the_block_size_comes_from_the_grid_or_defaults_to_deepseeks(
    pre_5_10_transformers, configured, shape, grid, block
):
    """A grid that is not the configured block's ceil grid sets the block by division; with no
    block size configured, DeepSeek's 128 x 128 applies."""
    weight, scale_inv = _block_fp8(shape, grid)
    moe._patch_finegrained_fp8_dequant(torch.float32)

    out = _dequant_op(configured)._dequantize_one(weight, scale_inv)

    assert torch.equal(out, _reference(weight, scale_inv, block))


@pytest.mark.parametrize(
    ("target_dtype", "expected"),
    [
        (None, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.float32, torch.float32),
    ],
)
def test_dequant_emits_the_requested_model_dtype(pre_5_10_transformers, target_dtype, expected):
    """Not the scales' dtype: DeepSeek's fp32 ``weight_scale_inv`` would give fp32 weights that
    then meet bf16 activations in the calibration matmuls. Unrequested, it is bf16."""
    weight, scale_inv = _block_fp8((10, 13), (3, 4))
    moe._patch_finegrained_fp8_dequant(target_dtype)

    out = _dequant_op((4, 4))._dequantize_one(weight, scale_inv)

    assert out.dtype == expected
    assert torch.equal(out, _reference(weight, scale_inv, (4, 4)).to(expected))


@pytest.mark.parametrize("grid", [(0, 4), (3, 0)])
def test_an_empty_scale_grid_passes_the_weight_through_unscaled(pre_5_10_transformers, grid):
    """Sharded or merged expert loads can hand over an empty grid; it must not crash the load."""
    weight, scale_inv = _block_fp8((10, 13), grid)
    moe._patch_finegrained_fp8_dequant(torch.float32)

    out = _dequant_op((4, 4))._dequantize_one(weight, scale_inv)

    assert torch.equal(out, weight.float())


def test_patching_again_keeps_the_install_and_takes_the_new_dtype(pre_5_10_transformers):
    """Every block-fp8 load patches again, with its own model dtype."""
    moe._patch_finegrained_fp8_dequant(torch.bfloat16)
    installed = _patched_state()

    moe._patch_finegrained_fp8_dequant(torch.float16)

    assert _patched_state() == installed
    weight, scale_inv = _block_fp8((10, 13), (3, 4))
    assert _dequant_op((4, 4))._dequantize_one(weight, scale_inv).dtype == torch.float16


def test_a_reimported_module_does_not_wrap_the_quantizer_hook_twice(
    pre_5_10_transformers, monkeypatch
):
    """Re-importing ``moe`` loses its module flag; the quantizer class carries its own."""
    if not hasattr(FineGrainedFP8HfQuantizer, "update_weight_conversions"):
        pytest.skip("this transformers' FP8 quantizer has no update_weight_conversions")
    moe._patch_finegrained_fp8_dequant()
    hook = vars(FineGrainedFP8HfQuantizer)["update_weight_conversions"]
    monkeypatch.setattr(moe, "_FINEGRAINED_FP8_PATCHED", False)

    moe._patch_finegrained_fp8_dequant()

    assert vars(FineGrainedFP8HfQuantizer)["update_weight_conversions"] is hook


def test_undoing_the_patch_restores_stock_transformers(pre_5_10_transformers, monkeypatch):
    """The patch rebinds only the dequant method and, where the quantizer has one, its converter
    hook and guard flag. Putting those back restores the stock classes exactly, after which the
    patch installs again once the module flag is cleared."""
    has_hook = hasattr(FineGrainedFP8HfQuantizer, "update_weight_conversions")
    classes = (Fp8Dequantize, FineGrainedFP8HfQuantizer)
    stock = {cls: dict(vars(cls)) for cls in classes}

    moe._patch_finegrained_fp8_dequant()

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

    monkeypatch.setattr(moe, "_FINEGRAINED_FP8_PATCHED", False)
    moe._patch_finegrained_fp8_dequant()
    assert vars(Fp8Dequantize).get("_dequantize_one") is not stock[Fp8Dequantize].get(
        "_dequantize_one"
    )


@pytest.mark.parametrize("version", ["5.10.1", "5.17.0"])
def test_transformers_from_5_10_1_keeps_its_own_dequant(
    pre_5_10_transformers, monkeypatch, version
):
    """Upstream fixed the dequant there and decodes ue8m0 scales, which the patch does not."""
    monkeypatch.setattr(transformers, "__version__", version)
    stock = _patched_state()

    moe._patch_finegrained_fp8_dequant(torch.bfloat16)

    assert _patched_state() == stock
