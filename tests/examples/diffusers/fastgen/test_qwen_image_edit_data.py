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

"""Focused CPU tests for the Qwen-Image-Edit preprocessing/data contracts."""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))

import preprocess_qwen_image_edit as edit_preprocess


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def test_spatialedit_tar_parser_maps_source_target_and_instruction(tmp_path):
    shard = tmp_path / "object_rotation" / "worker0-000000.tar"
    shard.parent.mkdir()
    metadata = {
        "conversations": [
            {"from": "human", "value": "<image>\nRotate the object to the right."},
            {"from": "gpt", "value": "<image>\n"},
        ],
        "meta": {"sample_id": "sample-7"},
    }
    with tarfile.open(shard, "w") as archive:
        _add_tar_bytes(archive, "key.0.jpg", _jpeg_bytes((255, 0, 0)))
        _add_tar_bytes(archive, "key.1.jpg", _jpeg_bytes((0, 0, 255)))
        _add_tar_bytes(archive, "key.json", json.dumps(metadata).encode())

    sample = next(edit_preprocess.iter_spatialedit_samples(tmp_path))

    # The WebDataset key is pair-unique; legacy ``meta.sample_id`` is only asset-unique.
    assert sample.sample_id == "key"
    assert sample.prompt == "Rotate the object to the right."
    assert len(sample.conditioning_images) == 1
    assert sample.conditioning_paths[0].endswith("::key.0.jpg")
    assert sample.target_path.endswith("::key.1.jpg")
    # JPEG is lossy, so compare dominant channels rather than exact values.
    assert sample.conditioning_images[0].getpixel((0, 0))[0] > 200
    assert sample.target_image.getpixel((0, 0))[2] > 200


def test_jsonl_parser_supports_archive_descriptors_and_normalized_field_names(tmp_path):
    archive_path = tmp_path / "images.tar"
    with tarfile.open(archive_path, "w") as archive:
        _add_tar_bytes(archive, "ref1.jpg", _jpeg_bytes((255, 0, 0)))
        _add_tar_bytes(archive, "ref2.jpg", _jpeg_bytes((0, 255, 0)))
        _add_tar_bytes(archive, "target.jpg", _jpeg_bytes((0, 0, 255)))
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "multi-ref",
                "conditioning_images": [
                    {"archive": "images.tar", "member": "ref1.jpg"},
                    {"archive": "images.tar", "member": "ref2.jpg"},
                ],
                "reference_image": {"archive": "images.tar", "member": "target.jpg"},
                "prompt": "<image>\nCombine both references.",
            }
        )
        + "\n"
    )

    sample = next(edit_preprocess.iter_jsonl_samples(manifest))

    assert sample.sample_id == "multi-ref"
    assert sample.prompt == "Combine both references."
    assert sample.negative_prompt == " "
    assert len(sample.conditioning_images) == 2
    assert sample.target_path.endswith("images.tar::target.jpg")


def test_launcher_cli_aliases_map_to_canonical_arguments():
    args = edit_preprocess.build_parser().parse_args(
        [
            "--input-dir",
            "raw",
            "--output-dir",
            "cache",
            "--gpu-id",
            "3",
            "--shard-idx",
            "1",
            "--shard-count",
            "4",
            "--max-samples",
            "25",
        ]
    )

    assert args.webdataset_root == Path("raw")
    assert args.gpu_id == 3
    assert (args.shard_rank, args.shard_world, args.limit) == (1, 4, 25)


def test_edit_collate_pads_multimodal_positive_and_negative_sequences():
    torch = pytest.importorskip("torch")
    pytest.importorskip("nemo_automodel")
    from fastgen_data import collate_fn_image_to_image

    def sample(pos_length: int, neg_length: int, sample_id: str):
        return {
            "latent": torch.randn(16, 8, 8),
            "conditioning_latents": [torch.randn(16, 8, 8)],
            "prompt_embeds": torch.randn(pos_length, 32),
            "prompt_embeds_mask": torch.ones(pos_length, dtype=torch.long),
            "negative_prompt_embeds": torch.randn(neg_length, 32),
            "negative_prompt_embeds_mask": torch.ones(neg_length, dtype=torch.long),
            "crop_resolution": torch.tensor([64, 64]),
            "original_resolution": torch.tensor([64, 64]),
            "crop_offset": torch.tensor([0, 0]),
            "prompt": "edit",
            "negative_prompt": " ",
            "image_path": f"{sample_id}.jpg",
            "conditioning_image_paths": [f"{sample_id}-ref.jpg"],
            "conditioning_resolutions": [(64, 64)],
            "target_latent_shape": (16, 8, 8),
            "conditioning_latent_shapes": [(16, 8, 8)],
            "sample_id": sample_id,
            "bucket_id": 0,
            "aspect_ratio": 1.0,
        }

    output = collate_fn_image_to_image([sample(3, 2, "a"), sample(5, 4, "b")])

    assert output["image_latents"].shape == (2, 16, 8, 8)
    assert len(output["conditioning_latents"]) == 1
    assert output["conditioning_latents"][0].shape == (2, 16, 8, 8)
    assert output["text_embeddings"].shape == (2, 5, 32)
    assert output["text_embeddings_mask"].tolist() == [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ]
    assert output["negative_text_embeddings"].shape == (2, 4, 32)
    assert output["negative_text_embeddings_mask"].tolist() == [
        [1, 1, 0, 0],
        [1, 1, 1, 1],
    ]


def test_qwen_image_edit_processor_is_registered():
    pytest.importorskip("torch")
    from preprocess.processors import ProcessorRegistry

    assert ProcessorRegistry.is_registered("qwen_image_edit")
