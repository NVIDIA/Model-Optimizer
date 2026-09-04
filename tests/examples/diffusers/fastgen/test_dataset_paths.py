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

"""Portable and contained path contract for the shared FastGen cache."""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nemo_automodel")
pytest.importorskip("torchdata")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))

from fastgen_data import build_text_to_image_multiresolution_dataloader
from fastgen_data.paths import resolve_cache_root, resolve_under_root
from fastgen_data.text_to_image_dataset import TextToImageDataset


def test_cache_root_uses_unset_or_empty_fallback(make_fastgen_cache, monkeypatch, tmp_path):
    cache = make_fastgen_cache(tmp_path / "cache")

    monkeypatch.delenv("MODELOPT_FASTGEN_DATASET_CACHE_DIR", raising=False)
    assert resolve_cache_root(cache) == cache.resolve()

    monkeypatch.setenv("MODELOPT_FASTGEN_DATASET_CACHE_DIR", "")
    assert resolve_cache_root(cache) == cache.resolve()


@pytest.mark.parametrize("override", ["relative/cache", "~/cache", " "])
def test_cache_root_rejects_nonempty_relative_override(monkeypatch, override, tmp_path):
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    monkeypatch.setenv("MODELOPT_FASTGEN_DATASET_CACHE_DIR", override)

    with pytest.raises(ValueError, match="absolute"):
        resolve_cache_root(fallback)


def test_cache_root_rejects_missing_or_non_directory_override(monkeypatch, tmp_path):
    fallback = tmp_path / "fallback"
    fallback.mkdir()

    monkeypatch.setenv("MODELOPT_FASTGEN_DATASET_CACHE_DIR", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        resolve_cache_root(fallback)

    regular_file = tmp_path / "file"
    regular_file.write_text("not a directory")
    monkeypatch.setenv("MODELOPT_FASTGEN_DATASET_CACHE_DIR", str(regular_file))
    with pytest.raises(NotADirectoryError):
        resolve_cache_root(fallback)


def test_resolve_under_root_rejects_traversal_absolute_and_symlink_escape(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    inside = root / "inside.pt"
    inside.write_bytes(b"inside")
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    (root / "escape.pt").symlink_to(outside)

    assert resolve_under_root(root, "inside.pt", "sample") == inside.resolve()
    for candidate in ("../outside.pt", outside, "escape.pt"):
        with pytest.raises(ValueError, match="sample"):
            resolve_under_root(root, candidate, "sample")


def test_dataset_rejects_metadata_shard_escape(make_fastgen_cache, tmp_path):
    cache = make_fastgen_cache(tmp_path / "cache")
    safe_item = json.loads((cache / "metadata_shard_0.json").read_text())[0]
    outside_shard = tmp_path / "outside.json"
    outside_shard.write_text(json.dumps([safe_item]))
    (cache / "metadata.json").write_text(json.dumps({"shards": [str(outside_shard)]}))

    with pytest.raises(ValueError, match="metadata shard"):
        TextToImageDataset(cache)


def test_dataset_rejects_payload_symlink_escape(make_fastgen_cache, tmp_path):
    cache = make_fastgen_cache(tmp_path / "cache")
    outside = tmp_path / "outside.pt"
    torch.save({"latent": torch.zeros(1)}, outside)
    (cache / "escape.pt").symlink_to(outside)
    shard = json.loads((cache / "metadata_shard_0.json").read_text())
    shard[0]["cache_file"] = "escape.pt"
    (cache / "metadata_shard_0.json").write_text(json.dumps(shard))

    dataset = TextToImageDataset(cache)
    with pytest.raises(ValueError, match="sample cache file"):
        dataset[0]


def test_dataset_accepts_absolute_payload_beneath_root(make_fastgen_cache, tmp_path):
    cache = make_fastgen_cache(tmp_path / "cache", absolute_payloads=True)
    dataset = TextToImageDataset(cache)

    assert "latent" in dataset[0]


def test_prompt_only_dataset_and_loader_do_not_emit_image_latents(make_fastgen_cache, tmp_path):
    cache = make_fastgen_cache(tmp_path / "cache")
    dataset = TextToImageDataset(cache, prompt_only=True)
    assert "latent" not in dataset[0]

    loader, _ = build_text_to_image_multiresolution_dataloader(
        cache_dir=str(cache),
        prompt_only=True,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        negative_prompt_embedding_path="negative_prompt_embedding.pt",
    )
    batch = next(iter(loader))

    assert "image_latents" not in batch
    assert {"text_embeddings", "text_embeddings_mask"}.issubset(batch)
    assert "negative_text_embeddings" in batch


def test_environment_redirects_samples_and_relative_negative_embedding(
    make_fastgen_cache, monkeypatch, tmp_path
):
    fallback = make_fastgen_cache(tmp_path / "fallback", marker=1.0)
    override = make_fastgen_cache(tmp_path / "override", marker=9.0)
    monkeypatch.setenv("MODELOPT_FASTGEN_DATASET_CACHE_DIR", str(override.resolve()))

    loader, _ = build_text_to_image_multiresolution_dataloader(
        cache_dir=str(fallback),
        batch_size=1,
        num_workers=0,
        shuffle=False,
        negative_prompt_embedding_path="negative_prompt_embedding.pt",
    )
    batch = next(iter(loader))

    assert loader.dataset.cache_root == override.resolve()
    assert batch["metadata"]["prompts"][0].startswith("prompt-9.0-")
    assert torch.equal(batch["negative_text_embeddings"], torch.full((1, 2, 3), 9.0))


def test_builder_rejects_negative_embedding_escape(make_fastgen_cache, tmp_path):
    cache = make_fastgen_cache(tmp_path / "cache")
    outside = tmp_path / "negative.pt"
    torch.save(torch.zeros(2, 3), outside)

    with pytest.raises(ValueError, match="negative prompt embedding"):
        build_text_to_image_multiresolution_dataloader(
            cache_dir=str(cache),
            num_workers=0,
            negative_prompt_embedding_path=str(outside),
        )


def test_builder_logs_effective_root_once_on_rank_zero(make_fastgen_cache, caplog, tmp_path):
    cache = make_fastgen_cache(tmp_path / "cache")
    caplog.set_level(logging.INFO, logger="fastgen_data.collate_fns")

    build_text_to_image_multiresolution_dataloader(
        cache_dir=str(cache), dp_rank=1, dp_world_size=2, num_workers=0
    )
    assert not [record for record in caplog.records if "effective_cache_root=" in record.message]

    caplog.clear()
    build_text_to_image_multiresolution_dataloader(
        cache_dir=str(cache), dp_rank=0, dp_world_size=2, num_workers=0
    )
    messages = [
        record.message for record in caplog.records if "effective_cache_root=" in record.message
    ]
    assert len(messages) == 1
    assert str(cache.resolve()) in messages[0]
    assert "selected=6/6" in messages[0]


def test_preprocessing_publishes_absolute_paths_for_relative_output(monkeypatch, tmp_path):
    # The metadata publisher does not use OpenCV. Stub that optional video dependency so this
    # CPU-only test executes the real publisher in the lean AutoModel test environment.
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    from preprocess.preprocessing_multiprocess import _save_metadata_shards

    monkeypatch.chdir(tmp_path)
    output = pathlib.Path("relative-cache")
    output.mkdir()
    payload = output / "sample.pt"
    torch.save({"latent": torch.zeros(1)}, payload)

    _save_metadata_shards(
        [{"cache_file": str(payload)}],
        output,
        "qwen_image",
        "Qwen/Qwen-Image",
        "qwen_image",
        10,
        {},
    )

    shard = json.loads((output / "metadata_shard_s0000.json").read_text())
    published = pathlib.Path(shard[0]["cache_file"])
    assert published.is_absolute()
    assert published == payload.resolve()
    published.relative_to(output.resolve())
