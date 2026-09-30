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

import modelopt.torch.export as modelopt_export
import modelopt.torch.export.shard_cast_utils as shard_cast_utils
from modelopt.torch.export.shard_cast_utils import (
    build_w13_kmax_overrides,
    dequantize_mxfp4_to_bf16,
    link_aux_files,
    mxfp4_kmax,
    quantize_mxfp4_to_nvfp4_lossless,
    resolve_checkpoint_file,
    validate_aux_files,
)
from modelopt.torch.quantization.qtensor import MXFP4QTensor, NVFP4QTensor


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
        "dequantize_mxfp4_to_bf16",
        "mxfp4_kmax",
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
