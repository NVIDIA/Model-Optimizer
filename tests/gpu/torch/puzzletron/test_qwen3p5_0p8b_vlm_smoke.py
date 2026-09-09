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

from examples.puzzletron.evaluation.vlm import contracts as evaluation_contracts
from examples.puzzletron.evaluation.vlm import profile as evaluation_profile
from examples.puzzletron.evaluation.vlm import suites as evaluation_suites
from modelopt.torch.puzzletron.dataset.multimodal import materialize_normalized_conversation_samples
from puzzletron_orchestrator.recipe_config import (
    materialize_resolved_bundle,
    resolve_recipe_run,
    site_template,
)
from tests._test_utils.torch.puzzletron.checkpoint_evaluation import (
    assert_pruned_checkpoints_completed_benchmark,
)


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


def _write_local_site(path: Path, project_root: Path, hf_home: Path) -> None:
    payload = site_template()
    payload["site"]["environment"].update({"repository": str(project_root), "venv": sys.prefix})
    payload["site"]["paths"]["hf_home"] = str(hf_home)
    payload["site"]["slurm"].update({"account": "local-smoke", "partition": "local"})
    payload["resources"]["selected"] = {
        "mode": "per_attempt",
        "gpus_per_node": 1,
        "max_nodes": 1,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


@pytest.mark.integration
@pytest.mark.manual(
    reason=(
        "downloads and prunes the real checkpoint, evaluates the cached pinned RealWorldQA "
        "snapshot on saved pre-KD and post-KD checkpoints through lmms-eval/vLLM, and "
        "benchmarks it with AIPerf"
    )
)
@pytest.mark.timeout(8700)
def test_qwen3p5_0p8b_orchestrated_vlm_full_smoke_completes(
    project_root_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Run a VLM bundle generated through the public configuration contract locally."""

    dataset = tmp_path / "dataset"
    results = tmp_path / "results"
    benchmark_hf_home = os.environ.get("PUZZLETRON_VLM_BENCHMARK_HF_HOME")
    if not benchmark_hf_home:
        pytest.fail("requires PUZZLETRON_VLM_BENCHMARK_HF_HOME with the pinned RealWorldQA cache")
    benchmark_hf_home_path = Path(benchmark_hf_home).expanduser().absolute()
    if not benchmark_hf_home_path.is_dir():
        pytest.fail(f"benchmark cache is not a directory: {benchmark_hf_home_path}")
    benchmark_hub_cache = benchmark_hf_home_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(benchmark_hub_cache))
    try:
        evaluation_suites.offline_dataset_snapshot(
            benchmark_hf_home_path,
            "realworldqa",
            evaluation_profile.VLM_BENCHMARK_DATASETS["realworldqa"].revision,
        )
    except ValueError as error:
        pytest.fail(str(error))
    recipe = tmp_path / "recipe.yaml"
    site = tmp_path / "site.yaml"
    _materialize_image_conversations(dataset)
    recipe_payload = yaml.safe_load(
        (
            project_root_path / "examples/puzzletron/configs/recipes/qwen3p5_0p8b_vlm_smoke.yaml"
        ).read_text()
    )
    recipe_payload.update(
        {
            "run_root": str(results),
            "resource_profile": "selected",
            "data": {"path": str(dataset), "revision": "fixture-revision"},
            "advanced": {"experiment": {"prepare_dataset.enabled": False}},
        }
    )
    recipe.write_text(yaml.safe_dump(recipe_payload, sort_keys=False))
    _write_local_site(site, project_root_path, benchmark_hf_home_path)
    bundle = materialize_resolved_bundle(resolve_recipe_run(recipe, site), activate=True)
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HOME": str(benchmark_hf_home_path),
            "HF_HUB_CACHE": str(benchmark_hub_cache),
            "HF_DATASETS_CACHE": str(benchmark_hf_home_path / "datasets"),
            "TORCH_HOME": str(tmp_path / "cache/torch"),
            "XDG_CACHE_HOME": str(tmp_path / "cache/xdg"),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root_path / "examples/puzzletron/orchestrate.py"),
            "--experiment",
            str(bundle / "experiment.runtime.yaml"),
            "--runner",
            str(bundle / "runner.yaml"),
            "--execution",
            str(bundle / "execution.yaml"),
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
        timeout=8600,
        check=False,
    )
    if completed.returncode:
        pytest.fail(
            "Qwen 3.5 0.8B full VLM smoke failed.\n"
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
    vision_markers = [
        marker
        for marker in activation_markers
        if marker.get("observability", {}).get("vision_forward_count", 0) > 0
    ]
    assert vision_markers
    for marker in vision_markers:
        observability = marker["observability"]
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
    smoke_rows = evaluation_contracts.load_profile("core-3_24-examples_r1-vllm").exact_rows
    assert smoke_rows is not None
    expected_realworldqa_samples = evaluation_suites.manifest_task_denominators(smoke_rows)[
        "realworldqa"
    ]["selected_rows"]
    assert expected_realworldqa_samples is not None
    for checkpoint_node, evaluation_node in (
        ("materialized", "checkpoint_eval"),
        ("short_vlm_kd", "post_kd_checkpoint_eval"),
    ):
        assert_pruned_checkpoints_completed_benchmark(
            results,
            checkpoint_node=checkpoint_node,
            evaluation_node=evaluation_node,
            task=evaluation_suites.task_name("realworldqa"),
            limit=expected_realworldqa_samples,
        )
