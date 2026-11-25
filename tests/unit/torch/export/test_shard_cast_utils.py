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

"""Tests for shared streaming checkpoint-cast utilities."""

import os
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

import modelopt.torch.export as modelopt_export
import modelopt.torch.export.shard_cast_utils as shard_cast_utils
from modelopt.torch.export.shard_cast_utils import (
    build_w13_kmax_overrides,
    dequantize_fp8_ue8m0_to_bf16,
    dequantize_mxfp4_to_bf16,
    link_aux_files,
    mxfp4_kmax,
    quantize_bf16_to_fp8_ue8m0,
    quantize_mxfp4_to_nvfp4_lossless,
    resolve_checkpoint_file,
    validate_aux_files,
)
from modelopt.torch.quantization.qtensor import MXFP4QTensor, NVFP4QTensor


@pytest.mark.parametrize("block", [32, 128])
def test_native_fp8_roundtrip_partial_zero_and_scale_boundary_blocks(block, tmp_path):
    """Native FP8-representable BF16 values survive encode/decode, including edge blocks."""
    exponents = torch.tensor([-127, -8, -1, 0, 8, 119])
    rows, cols = (len(exponents) + 1) * block - 3, block + 5
    codes = torch.tensor([448.0, -224.0, 1.125, -0.5, 0.0]).repeat(rows, (cols + 4) // 5)
    codes = codes[:, :cols]
    row_exponents = torch.cat((exponents, torch.tensor([0]))).repeat_interleave(block)[:rows]
    weight = torch.ldexp(codes, row_exponents[:, None]).to(torch.bfloat16)
    weight[-block + 3 :] = 0
    codes[-block + 3 :] = 0

    encoded, scales = quantize_bf16_to_fp8_ue8m0(weight, block)
    checkpoint = tmp_path / "native.safetensors"
    save_file({"weight": encoded, "scale": scales}, checkpoint)
    restored = load_file(checkpoint)
    encoded, scales = restored["weight"], restored["scale"]
    expected_scales = torch.cat((exponents + 127, torch.tensor([0]))).to(torch.uint8)
    torch.testing.assert_close(scales, expected_scales[:, None].expand(-1, 2))
    torch.testing.assert_close(encoded.float(), codes, rtol=0, atol=0)
    torch.testing.assert_close(
        dequantize_fp8_ue8m0_to_bf16(encoded, scales, block), weight, rtol=0, atol=0
    )
    torch.testing.assert_close(
        dequantize_fp8_ue8m0_to_bf16(encoded, scales.view(torch.float8_e8m0fnu), block),
        weight,
        rtol=0,
        atol=0,
    )
    above = torch.nextafter(weight, torch.full_like(weight, float("inf")))
    _, above_scales = quantize_bf16_to_fp8_ue8m0(above, block)
    torch.testing.assert_close(above_scales[:-1], (expected_scales[:-1] + 1)[:, None].expand(-1, 2))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_native_fp8_export_rejects_nonfinite_weights(value):
    weight = torch.zeros(2, 3, dtype=torch.bfloat16)
    weight[0, 0] = value
    with pytest.raises(ValueError):
        quantize_bf16_to_fp8_ue8m0(weight)


@pytest.mark.parametrize(
    ("shape", "dtype", "block"),
    [
        ((1, 32, 32), torch.bfloat16, 32),
        ((32, 32), torch.float32, 32),
        ((32, 32), torch.bfloat16, 16),
        ((0, 32), torch.bfloat16, 32),
    ],
)
def test_native_fp8_export_rejects_unsupported_matrices(shape, dtype, block):
    with pytest.raises(ValueError):
        quantize_bf16_to_fp8_ue8m0(torch.zeros(shape, dtype=dtype), block)


@pytest.mark.parametrize(
    "scale",
    [
        torch.ones(2, 1, dtype=torch.uint8),
        torch.ones(1, 1),
        torch.full((1, 1), 255, dtype=torch.uint8),
    ],
)
def test_native_fp8_load_rejects_invalid_scales(scale):
    weight = torch.zeros(32, 32).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        dequantize_fp8_ue8m0_to_bf16(weight, scale, block=32)


def test_mxfp4_to_nvfp4_lossless_cast_preserves_dequantized_values():
    torch.manual_seed(7)
    weight = torch.randn(4, 64, dtype=torch.bfloat16)
    qtensor, scale = MXFP4QTensor.quantize(weight, block_size=32)
    packed = qtensor._quantized_data
    scale = scale.reshape(4, 2)

    nvfp4_packed, nvfp4_scale, scale_2, n_blocks, n_lossless = quantize_mxfp4_to_nvfp4_lossless(
        packed,
        scale,
        mxfp4_kmax(scale),
        "cpu",
    )
    source_dequant = dequantize_mxfp4_to_bf16(packed, scale, "cpu")
    nvfp4_dequant = NVFP4QTensor(weight.shape, weight.dtype, nvfp4_packed).dequantize(
        scale=nvfp4_scale,
        double_scale=scale_2,
        block_sizes={-1: 16},
        dtype=weight.dtype,
    )

    assert torch.equal(nvfp4_dequant, source_dequant)
    assert scale_2.shape == torch.Size([])
    assert n_blocks == n_lossless == scale.numel()


def test_w13_pair_requires_both_projections():
    base = "layers.1.ffn.experts.0.w1"

    with pytest.raises(RuntimeError, match="split across shards"):
        build_w13_kmax_overrides(
            [base],
            lambda _: torch.tensor([127], dtype=torch.uint8),
            "cpu",
        )


def test_public_api_contains_only_cast_helpers():
    assert set(shard_cast_utils.__all__) == {
        "build_w13_amax_overrides",
        "build_w13_kmax_overrides",
        "dequantize_fp8_ue8m0_to_bf16",
        "dequantize_mxfp4_to_bf16",
        "mxfp4_kmax",
        "quantize_bf16_to_fp8_ue8m0",
        "quantize_mxfp4_to_nvfp4",
        "quantize_mxfp4_to_nvfp4_lossless",
    }
    assert not hasattr(modelopt_export, "log")


def test_link_aux_files_preserves_sidecars_and_applies_skips(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "assets").mkdir(parents=True)
    (source / ".cache").mkdir()
    (source / "tokenizer_config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"shard")
    (source / "assets" / "config.txt").write_text("keep")
    (source / ".cache" / "stale.json").write_text("{}")

    link_aux_files(
        source,
        output,
        skip_top_level={".cache"},
        skip_dir_names={".cache"},
        skip_file=lambda path: path.suffix == ".safetensors",
    )

    assert (output / "tokenizer_config.json").read_text() == "{}"
    assert (output / "assets" / "config.txt").read_text() == "keep"
    assert not (output / "model.safetensors").exists()
    assert not (output / ".cache").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX symlink support")
def test_link_aux_files_accepts_huggingface_snapshot_blobs(tmp_path):
    repository = tmp_path / "models--moonshotai--Kimi-K3"
    snapshot = repository / "snapshots" / "revision"
    blob = repository / "blobs" / "tokenizer-blob"
    snapshot.mkdir(parents=True)
    blob.parent.mkdir()
    blob.write_text("tokenizer")
    (snapshot / "tokenizer.json").symlink_to(os.path.relpath(blob, snapshot))

    assert resolve_checkpoint_file(snapshot, "tokenizer.json") == blob.resolve()
    validate_aux_files(snapshot)
    output = tmp_path / "output"
    link_aux_files(snapshot, output)

    assert (output / "tokenizer.json").read_text() == "tokenizer"
    assert not (output / "tokenizer.json").is_symlink()


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX symlink and FIFO support")
@pytest.mark.parametrize(
    ("source_kind", "message"),
    [("symlink", "outside the checkpoint directory"), ("fifo", "regular file")],
)
def test_link_aux_files_rejects_unsafe_sources(tmp_path, source_kind, message):
    source = tmp_path / "source"
    source.mkdir()
    unsafe = source / "unsafe"
    if source_kind == "symlink":
        outside = tmp_path / "outside"
        outside.write_text("secret")
        unsafe.symlink_to(outside)
    else:
        os.mkfifo(unsafe)

    with pytest.raises(ValueError, match=message):
        resolve_checkpoint_file(source, unsafe.name)
    with pytest.raises(ValueError, match=message):
        link_aux_files(source, tmp_path / "output")

    assert not (tmp_path / "output").exists()


def test_resolve_checkpoint_file_rejects_oversized_metadata(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("12345")

    with pytest.raises(ValueError, match="4-byte size limit"):
        resolve_checkpoint_file(source, "config.json", max_bytes=4)
