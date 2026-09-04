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

"""Tests for the resumable dataset-preparation worker stage."""

import hashlib
import json

import pytest

from examples.puzzletron import prepare_dataset as module
from examples.puzzletron.evaluation.vlm.preparation import benchmark_data
from puzzletron_orchestrator.adapters.stage_compat import stage_is_complete


def test_prepare_dataset_stage_materializes_and_seals_completion(tmp_path, monkeypatch):
    output = tmp_path / "datasets" / "vlm"
    hf_home = tmp_path / "hf-home"
    catalog = {
        "realworldqa": {
            "repository": "lmms-lab/RealWorldQA",
            "revision": "907c4e5228fd1703c710ed937601cb5f89ab8d5c",
            "requires_media": False,
            "preparation_dir": None,
        }
    }
    config = {
        "puzzle_dir": str(tmp_path),
        "dataset_path": str(output),
        "data": {"revision": "pinned-sha"},
        "prepare_dataset": {
            "enabled": True,
            "adapter": "nemotron_vlm_v2",
            "output": str(output),
            "subsets": ["sparsetables", "plotqa_cot", "wiki_en"],
            "num_samples": 8,
            "seed": 17,
            "max_shards_per_subset": 1,
            "evaluation_tasks": ["realworldqa"],
            "evaluation_hf_home": str(hf_home),
            "evaluation_datasets": catalog,
        },
    }

    def materialize(spec):
        output.mkdir(parents=True)
        (output / "images").mkdir()
        samples = []
        images = []
        for index in range(8):
            relative = f"images/{index:04d}_00.png"
            image_payload = f"image-{index}".encode()
            (output / relative).write_bytes(image_payload)
            samples.append({"conversation": [{"content": [{"type": "image", "image": relative}]}]})
            images.append({"path": relative, "sha256": hashlib.sha256(image_payload).hexdigest()})
        samples_payload = json.dumps(samples).encode()
        payload = {
            "sample_count": 8,
            "samples_sha256": hashlib.sha256(samples_payload).hexdigest(),
            "image_count": 8,
            "images": images,
            "acquisition": spec.identity(revision="pinned-sha"),
        }
        (output / "samples.json").write_bytes(samples_payload)
        (output / "manifest.json").write_text(json.dumps(payload))
        (output / "puzzletron_acquisition.json").write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(module, "materialize_nemotron_vlm_dataset", materialize)

    def prepare_benchmarks(root, tasks, **_kwargs):
        assert tuple(tasks) == ("realworldqa",)
        task = "realworldqa"
        repository = catalog[task]["repository"]
        revision = catalog[task]["revision"]
        snapshot = root / "hub" / "datasets--lmms-lab--RealWorldQA" / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "dataset-info.json").write_text("{}")
        return [
            {
                "task": task,
                "repository": repository,
                "revision": revision,
                "snapshot": str(snapshot),
                "requires_media": False,
                "preparation_dir": None,
                "snapshot_inventory": benchmark_data._snapshot_inventory_report(
                    root, task, snapshot
                ),
                "status": "downloaded",
            }
        ]

    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setattr(module, "prepare_benchmark_datasets", prepare_benchmarks)

    result = module.prepare_dataset_stage(config)

    assert result.status == "success"
    assert stage_is_complete(config, "prepare_dataset")
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["outputs"]["sample_count"] == 8
    assert manifest["outputs"]["acquisition"]["revision"] == "pinned-sha"
    assert manifest["outputs"]["evaluation_hf_home"] == str(hf_home)
    assert [row["task"] for row in manifest["outputs"]["evaluation_data"]] == ["realworldqa"]


def test_prepare_dataset_stage_requires_runner_hf_home_for_evaluations(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    config = {
        "puzzle_dir": str(tmp_path),
        "dataset_path": str(tmp_path / "dataset"),
        "prepare_dataset": {
            "enabled": True,
            "output": str(tmp_path / "dataset"),
            "evaluation_tasks": ["realworldqa"],
        },
    }
    monkeypatch.setattr(
        module,
        "materialize_nemotron_vlm_dataset",
        lambda _spec: {"sample_count": 1, "acquisition": {}},
    )

    with pytest.raises(ValueError, match="runner HF_HOME"):
        module.prepare_dataset_stage(config)
