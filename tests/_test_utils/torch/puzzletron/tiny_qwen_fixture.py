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

"""Materialize and inspect the tiny-Qwen Puzzletron integration fixture."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from _test_utils.torch.transformers_models import create_tiny_qwen3_5_dir
from datasets import Dataset, DatasetDict

import puzzletron_orchestrator.recipe_config as recipe_config
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)
from puzzletron_orchestrator.config import _compose, _config_root, _merge

__all__ = ["TinyQwenCampaign", "build_tiny_qwen_campaign"]

if TYPE_CHECKING:
    from puzzletron_orchestrator.schema import CampaignPlan

_CONFIG_DIR = Path(__file__).with_name("configs")
_RECIPE = _CONFIG_DIR / "tiny_qwen.recipe.yaml"
_SITE = _CONFIG_DIR / "tiny_qwen.site.yaml"
_EXPERIMENT_OVERLAY = _CONFIG_DIR / "tiny_qwen_lifecycle.overlay.yaml"


@dataclass(frozen=True)
class TinyQwenCampaign:
    """Generated campaign bundle plus its exact execution contract."""

    project_root: Path
    smoke_bundle: Path
    smoke_root: Path
    flow_id: str
    overrides: tuple[str, ...]
    environment: dict[str, str]
    config: dict[str, Any]
    compiled_plan: CampaignPlan

    def run(self, *, timeout: int = 720) -> subprocess.CompletedProcess[str]:
        """Run or resume the full campaign through the public local orchestrator."""

        command = [
            sys.executable,
            str(self.project_root / "examples/puzzletron/orchestrate.py"),
            "--experiment",
            str(self.smoke_bundle / "experiment.runtime.yaml"),
            "--runner",
            str(self.smoke_bundle / "runner.yaml"),
            "--execution",
            str(self.smoke_bundle / "execution.yaml"),
            "--stage",
            "full",
            "--local",
            "--poll-interval",
            "0.05",
            "--color",
            "never",
        ]
        for override in self.overrides:
            command.extend(("--override", override))
        try:
            return subprocess.run(
                command,
                cwd=self.project_root,
                env=self.environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            stderr = error.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            logs = sorted(
                self.smoke_root.glob("logs/**/*.log"),
                key=lambda path: path.stat().st_mtime_ns,
            )
            log_tail = (
                logs[-1].read_text(errors="replace")[-12000:] if logs else "no task log found"
            )
            raise AssertionError(
                f"Tiny Qwen Puzzletron campaign timed out after {timeout}s.\n"
                f"stderr tail:\n{stderr[-12000:]}\n"
                f"latest task-log tail:\n{log_tail}"
            ) from error

    def require_success(self, completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
        """Return the controller result or raise with the most useful task log."""

        if completed.returncode != 0:
            logs = sorted(
                self.smoke_root.glob("logs/**/*.log"),
                key=lambda path: path.stat().st_mtime_ns,
            )
            log_tail = (
                logs[-1].read_text(errors="replace")[-12000:] if logs else "no task log found"
            )
            raise AssertionError(
                "Tiny Qwen Puzzletron campaign failed.\n"
                f"stdout tail:\n{completed.stdout[-12000:]}\n"
                f"stderr tail:\n{completed.stderr[-12000:]}\n"
                f"latest task-log tail:\n{log_tail}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise AssertionError(
                "Puzzletron orchestrator did not emit its JSON result.\n"
                f"stdout tail:\n{completed.stdout[-12000:]}\n"
                f"stderr tail:\n{completed.stderr[-12000:]}"
            ) from error
        if not isinstance(payload, dict):
            raise AssertionError(f"unexpected Puzzletron result payload: {payload!r}")
        return payload


def _save_messages_dataset(path: Path) -> None:
    response = (
        "Compression removes redundant parameters while preserving useful model behavior. " * 16
    ).strip()
    messages = [
        {"role": "user", "content": "What is model compression?"},
        {"role": "assistant", "content": response},
    ]
    rows = [{"messages": messages}] * 2
    DatasetDict(
        {
            "train": Dataset.from_list(rows),
            "validation": Dataset.from_list(rows),
        }
    ).save_to_disk(str(path))


def _write_tiny_route_template(project_root: Path, tmp_path: Path, model_dir: Path) -> Path:
    """Compose the checked-in tiny overlay onto the maintained lifecycle."""

    source = (
        project_root
        / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/full_smoke.yaml"
    )
    experiment = _compose(source, root=_config_root(source), stack=())
    overlay = yaml.safe_load(_EXPERIMENT_OVERLAY.read_text())
    experiment = _merge(experiment, overlay)
    # The hermetic test replaces external downstream benchmarks with the
    # checked-in local-only flow rather than merging the two node mappings.
    experiment["post_mip"]["flows"]["params-90"]["nodes"] = overlay["post_mip"]["flows"][
        "params-90"
    ]["nodes"]
    experiment["input_hf_model_path"] = str(model_dir)
    experiment["model_info"]["hf_repo"] = str(model_dir)
    catalog = tmp_path / "test-route-catalog"
    catalog.mkdir()
    (catalog / "tiny_qwen.yaml").write_text(yaml.safe_dump(experiment, sort_keys=False))
    return catalog


def build_tiny_qwen_campaign(project_root: Path, tmp_path: Path) -> TinyQwenCampaign:
    """Generate the sole tiny-Qwen public-route E2E fixture."""

    model_dir = create_tiny_qwen3_5_dir(
        tmp_path / "model",
        with_tokenizer=True,
        hidden_size=512,
        intermediate_size=768,
        head_dim=16,
        max_position_embeddings=128,
        num_hidden_layers=2,
        layer_types=["full_attention"] * 2,
    )
    dataset_dir = tmp_path / "dataset"
    result_root = tmp_path / "fast-e2e"
    cache_dir = tmp_path / "cache"
    recipe_path = tmp_path / "puzzletron.recipe.yaml"
    site_path = tmp_path / "puzzletron.site.yaml"
    _save_messages_dataset(dataset_dir)

    recipe = yaml.safe_load(_RECIPE.read_text())
    recipe["run_root"] = str(result_root)
    recipe["data"]["path"] = str(dataset_dir)
    site = yaml.safe_load(_SITE.read_text())
    site["site"]["environment"].update({"repository": str(project_root), "venv": sys.prefix})
    site["site"]["paths"]["hf_home"] = str(cache_dir / "huggingface")
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    site_path.write_text(yaml.safe_dump(site, sort_keys=False))

    catalog_root = _write_tiny_route_template(project_root, tmp_path, model_dir)
    route_key = (recipe["model"], recipe["workflow"], recipe["mode"])
    original_root = recipe_config._CONFIG_ROOT
    original_routes = recipe_config.ROUTES_BY_KEY
    test_routes = dict(original_routes)
    test_routes[route_key] = replace(
        original_routes[route_key],
        experiment_template="tiny_qwen.yaml",
    )
    try:
        recipe_config._CONFIG_ROOT = catalog_root
        recipe_config.ROUTES_BY_KEY = test_routes
        resolved = recipe_config.resolve_recipe_run(recipe_path, site_path)
    finally:
        recipe_config._CONFIG_ROOT = original_root
        recipe_config.ROUTES_BY_KEY = original_routes

    smoke_bundle = recipe_config.materialize_resolved_bundle(resolved, activate=True)
    experiment_path = smoke_bundle / "experiment.runtime.yaml"
    overrides: tuple[str, ...] = ()
    config = pipeline_config_from_path(experiment_path, overrides=overrides)
    compiled_plan = compile_campaign_plan(
        experiment_config_path=experiment_path,
        runner=load_runner_config(smoke_bundle / "runner.yaml"),
        execution=load_execution_config(smoke_bundle / "execution.yaml"),
        overrides=overrides,
        stage_filter="full",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(cache_dir / "huggingface"),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_CACHE": str(cache_dir / "datasets"),
            "AIPERF_TOKENIZER_ALIAS_DIR": str(cache_dir / "aiperf-tokenizers"),
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_HOME": str(cache_dir / "torch"),
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_CACHE_ROOT": str(cache_dir / "vllm"),
            "WANDB_DISABLED": "true",
            "XDG_CACHE_HOME": str(cache_dir / "xdg"),
        }
    )
    return TinyQwenCampaign(
        project_root=project_root,
        smoke_bundle=smoke_bundle,
        smoke_root=result_root,
        flow_id="params-90",
        overrides=overrides,
        environment=environment,
        config=config,
        compiled_plan=compiled_plan,
    )
