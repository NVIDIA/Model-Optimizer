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

"""Opt-in real-checkpoint smoke for the Qwen 3.5 0.8B VLM MIP route."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

from modelopt.torch.puzzletron.dataset.multimodal import materialize_normalized_conversation_samples

RUN_PATH = "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/mip_vlm_smoke.yaml"
EXECUTION_PATH = "examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.vlm_smoke.yaml"


def _materialize_image_conversations(path: Path) -> None:
    images = []
    samples = []
    answer = (
        "The image contains a colored square used to validate multimodal model pruning. " * 16
    ).strip()
    for index in range(8):
        image = Image.new("RGB", (32, 32), color=(index * 8, 32, 255 - index * 8))
        images.append(image)
        samples.append(
            {
                "source": {
                    "dataset": "qwen3p5-vlm-smoke-fixture",
                    "revision": "1",
                    "row_id": f"row-{index}",
                },
                "image_count": 1,
                "conversation": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Describe the colored square."},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}],
                    },
                ],
            }
        )
    try:
        manifest = materialize_normalized_conversation_samples(
            samples,
            path,
            expected_count=len(samples),
        )
    finally:
        for image in images:
            image.close()
    assert manifest["sample_count"] == manifest["image_count"] == len(samples)


def _write_local_runner(path: Path, project_root: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "runner": {
                    "kind": "slurm",
                    "slurm": {"account": "local-smoke", "max_nodes": 1},
                    "execution_contract": {
                        "repository": str(project_root),
                        "venv": sys.prefix,
                        "container": None,
                        "container_mounts": None,
                        "prerun_commands": [],
                        "postrun_commands": [],
                    },
                }
            },
            sort_keys=False,
        )
    )


@pytest.mark.integration
@pytest.mark.manual(reason="downloads and prunes the real Qwen 3.5 0.8B VLM checkpoint")
@pytest.mark.timeout(2400)
def test_qwen3p5_0p8b_orchestrated_vlm_mip_smoke_completes(
    project_root_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Run through MIP and prove that image inputs reach the vision tower."""

    dataset = tmp_path / "dataset"
    results = tmp_path / "results"
    cache = tmp_path / "cache"
    runner = tmp_path / "runner.yaml"
    _materialize_image_conversations(dataset)
    _write_local_runner(runner, project_root_path)
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(results))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(dataset))
    monkeypatch.setenv("PUZZLETRON_DATASET_REVISION", "fixture-revision")

    environment = os.environ.copy()
    environment.update(
        {
            "HF_HOME": str(cache / "huggingface"),
            "HF_DATASETS_CACHE": str(cache / "datasets"),
            "TORCH_HOME": str(cache / "torch"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root_path / "examples/puzzletron/orchestrate.py"),
            "--experiment",
            str(project_root_path / RUN_PATH),
            "--runner",
            str(runner),
            "--execution",
            str(project_root_path / EXECUTION_PATH),
            "--stage",
            "full",
            "--local",
            "--color",
            "never",
        ],
        cwd=project_root_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=2300,
        check=False,
    )
    if completed.returncode:
        pytest.fail(
            "Qwen 3.5 0.8B VLM MIP smoke failed.\n"
            f"stdout tail:\n{completed.stdout[-12000:]}\n"
            f"stderr tail:\n{completed.stderr[-12000:]}"
        )

    width_manifest = json.loads(
        (results / "manifests/width_importance.json").read_text(encoding="utf-8")
    )
    activation_root = Path(width_manifest["outputs"]["activations_log_dir"])
    activation_markers = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in activation_root.glob("**/args.json")
    ]
    assert activation_markers
    vision_markers = [
        marker
        for marker in activation_markers
        if marker.get("observability", {}).get("vision_forward_count", 0) > 0
    ]
    assert vision_markers
    for marker in vision_markers:
        observability = marker["observability"]
        assert observability["vision_forward_count"] > 0
        assert observability["vision_output_checksums"]
        assert observability["batch_fingerprints"]

    replacement_results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in results.glob(
            "scenarios/width-*/depth-*/"
            "single_subblock_replacement_solutions--validation/solution_*.json"
        )
    ]
    assert len(replacement_results) == 48
    for result in replacement_results:
        observability = result["observability"]
        assert observability["vision_forward_count"] > 0
        assert observability["vision_output_checksums"]

    mip_manifest = json.loads((results / "manifests/mip.json").read_text(encoding="utf-8"))
    assert mip_manifest["status"] == "success"
    active_profiles = json.loads((results / "mip/active_profiles.json").read_text())
    assert active_profiles["status"] == "success"
    assert active_profiles["profile_ids"] == ["params-90"]
