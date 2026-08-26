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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from _test_utils.torch.transformers_models import create_tiny_qwen3_5_dir
from datasets import Dataset, DatasetDict

from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)
from puzzletron_setup.v2.wizard import _DEFAULT_DATA_SOURCE, _DEFAULT_MODEL_SOURCE, run_wizard_v2

__all__ = ["TinyQwenCampaign", "build_tiny_qwen_campaign"]

if TYPE_CHECKING:
    from collections.abc import Sequence

    from puzzletron_orchestrator.schema import CampaignPlan
    from puzzletron_setup.v2.prompts import PromptChoice


_SETUP_DEFAULTS = Path(__file__).with_name("tiny_qwen_setup_defaults.yaml")
_EXPERIMENT_OVERLAY = Path(__file__).with_name("tiny_qwen_experiment_overlay.yaml")


class _DefaultsBackend:
    """Select resolved guided defaults while supplying the campaign directory."""

    def __init__(self, campaign_dir: Path) -> None:
        self.campaign_dir = campaign_dir

    def text(self, message: str, default: str) -> Any:
        if message == "Campaign directory:":
            return str(self.campaign_dir)
        return default

    def select(
        self,
        message: str,
        choices: Sequence[PromptChoice],
        default: Any,
    ) -> Any:
        if message == "Model:":
            return _DEFAULT_MODEL_SOURCE
        if message == "Dataset:":
            return _DEFAULT_DATA_SOURCE
        if default is not None:
            return default
        return next(choice.value for choice in choices if choice.disabled is None)

    def checkbox(
        self,
        message: str,
        choices: Sequence[PromptChoice],
        defaults: Sequence[Any],
    ) -> Any:
        del message, choices
        return list(defaults)


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

    def run(self, *, timeout: int = 2100) -> subprocess.CompletedProcess[str]:
        """Run or resume the full campaign through the public local orchestrator."""

        command = [
            sys.executable,
            str(self.project_root / "examples/puzzletron/orchestrate.py"),
            "--experiment",
            str(self.smoke_bundle / "experiment.yaml"),
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
        return subprocess.run(
            command,
            cwd=self.project_root,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

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
    rows = [{"messages": messages}] * 8
    DatasetDict(
        {
            "train": Dataset.from_list(rows),
            "validation": Dataset.from_list(rows),
        }
    ).save_to_disk(str(path))


def _merge_config(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively apply a declarative test overlay to a generated config."""

    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_config(base[key], value)
        else:
            base[key] = value
    return base


def _write_setup_defaults(
    path: Path,
    *,
    model_dir: Path,
    dataset_dir: Path,
    result_root: Path,
    project_root: Path,
) -> None:
    """Materialize the checked-in setup config with per-test runtime paths."""

    defaults = yaml.safe_load(_SETUP_DEFAULTS.read_text())
    defaults["model"]["source"] = str(model_dir)
    defaults["data"]["source"] = str(dataset_dir)
    defaults["output"]["result_root"] = str(result_root)
    contract = defaults["infrastructure"]["execution_contract"]
    contract["repository"] = str(project_root)
    contract["venv"] = sys.prefix
    path.write_text(yaml.safe_dump(defaults, sort_keys=False))


def build_tiny_qwen_campaign(
    project_root: Path,
    tmp_path: Path,
) -> TinyQwenCampaign:
    """Generate the sole tiny-Qwen setup-to-resume Puzzletron E2E fixture."""

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
    campaign_dir = tmp_path / "campaign"
    result_root = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    defaults_path = tmp_path / "defaults.yaml"
    _save_messages_dataset(dataset_dir)
    _write_setup_defaults(
        defaults_path,
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        result_root=result_root,
        project_root=project_root,
    )

    generated = run_wizard_v2(
        resume=None,
        defaults_path=defaults_path,
        backend=_DefaultsBackend(campaign_dir),
    )
    smoke_bundle = generated / "smoke"
    experiment_path = smoke_bundle / "experiment.yaml"
    experiment = yaml.safe_load(experiment_path.read_text())
    flows = dict((experiment.get("post_mip") or {}).get("flows") or {})
    if len(flows) != 1:
        raise AssertionError(f"expected one recommended post-MIP flow, found {sorted(flows)}")
    flow_id = next(iter(flows))
    overlay = yaml.safe_load(_EXPERIMENT_OVERLAY.read_text())
    overlay_flows = dict(overlay["post_mip"]["flows"])
    if tuple(overlay_flows) != (flow_id,):
        raise AssertionError(
            f"tiny-Qwen overlay targets {sorted(overlay_flows)}, generated flow is {flow_id!r}"
        )
    _merge_config(experiment, overlay)
    experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False))
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
        smoke_root=result_root / "smoke",
        flow_id=flow_id,
        overrides=overrides,
        environment=environment,
        config=config,
        compiled_plan=compiled_plan,
    )
