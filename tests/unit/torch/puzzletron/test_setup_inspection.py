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

"""Focused source-normalization tests for the Puzzletron setup wizard."""

from types import SimpleNamespace

import pytest

from puzzletron_setup import SetupError, inspection
from puzzletron_setup.inspection import normalize_dataset_source, normalize_model_source


def test_normalizes_hugging_face_web_urls():
    assert normalize_model_source("https://huggingface.co/Qwen/Qwen3.5-0.8B") == (
        "Qwen/Qwen3.5-0.8B"
    )
    assert normalize_model_source("huggingface.com/Qwen/Qwen3.5-0.8B/") == ("Qwen/Qwen3.5-0.8B")
    assert (
        normalize_dataset_source("https://huggingface.com/datasets/nvidia/Some-Dataset")
        == "nvidia/Some-Dataset"
    )


def test_normalizes_existing_local_paths_and_rejects_other_uris(tmp_path):
    model = tmp_path / "Model"
    dataset = tmp_path / "Dataset"
    model.mkdir()
    dataset.mkdir()

    assert normalize_model_source(str(model)) == str(model.resolve())
    assert normalize_dataset_source(str(dataset)) == str(dataset.resolve())
    with pytest.raises(SetupError, match="Unsupported model source"):
        normalize_model_source("s3://bucket/model")
    with pytest.raises(SetupError, match="does not exist"):
        normalize_dataset_source("../missing-dataset")


def test_inspect_model_uses_cached_config_when_hub_resolution_is_unavailable(tmp_path, monkeypatch):
    revision = "a" * 40
    config_path = tmp_path / "models--Qwen--cached" / "snapshots" / revision / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text('{"model_type": "cached"}\n')

    class OfflineApi:
        def model_info(self, source, revision=None):
            del source, revision
            raise RuntimeError("offline")

    monkeypatch.setattr(inspection, "HfApi", OfflineApi)
    monkeypatch.setattr(
        inspection,
        "try_to_load_from_cache",
        lambda source, filename, revision: None,
    )
    monkeypatch.setattr(
        inspection,
        "scan_cache_dir",
        lambda: SimpleNamespace(
            repos=[
                SimpleNamespace(
                    repo_type="model",
                    repo_id="Qwen/cached",
                    revisions=[
                        SimpleNamespace(
                            snapshot_path=config_path.parent,
                            commit_hash=revision,
                        )
                    ],
                )
            ]
        ),
    )
    monkeypatch.setattr(
        inspection,
        "resolve_profile",
        lambda config: SimpleNamespace(inventory=lambda value: "cached-inventory"),
    )

    model = inspection.inspect_model("Qwen/cached")

    assert model.source == "Qwen/cached"
    assert model.resolved_revision == revision
    assert model.config == {"model_type": "cached"}
    assert model.inventory == "cached-inventory"


def test_cached_model_ref_preserves_snapshot_commit_across_blob_symlink(tmp_path, monkeypatch):
    revision = "b" * 40
    blob = tmp_path / "blobs" / "config"
    blob.parent.mkdir()
    blob.write_text("{}\n")
    config_path = tmp_path / "snapshots" / revision / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.symlink_to(blob)
    monkeypatch.setattr(
        inspection,
        "try_to_load_from_cache",
        lambda source, filename, revision: str(config_path),
    )

    cached = inspection._cached_remote_config("Qwen/cached", None)

    assert cached == (config_path.absolute(), revision)
