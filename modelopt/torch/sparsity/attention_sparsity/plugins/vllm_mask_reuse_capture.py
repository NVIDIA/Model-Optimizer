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

"""vLLM worker bootstrap and RPCs for policy-free mask-reuse calibration."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu_worker import Worker as BaseWorker

if TYPE_CHECKING:
    from types import ModuleType

CAPTURE_ENV = "MASK_REUSE_FA4_CALIBRATION_CAPTURE"
PLAN_ENV = "MASK_REUSE_FA4_PLAN"
_REQUIRED_API = (
    "configure_capture_runtime",
    "capture_status",
    "begin_capture",
    "drain_capture",
)

__all__ = ["MaskReuseCaptureWorker"]


def _capture_api() -> ModuleType:
    if os.environ.get(CAPTURE_ENV) != "1":
        raise RuntimeError(f"{CAPTURE_ENV}=1 is required for mask-reuse calibration capture")
    try:
        api = importlib.import_module("mask_reuse_vllm.capture")
    except ImportError as error:
        raise RuntimeError(
            "the custom mask-reuse backend does not expose mask_reuse_vllm.capture"
        ) from error
    missing = [name for name in _REQUIRED_API if not callable(getattr(api, name, None))]
    if missing:
        raise RuntimeError(f"mask-reuse capture API is incomplete; missing {missing}")
    return api


def _configure_capture_before_model_load() -> ModuleType:
    """Install a planner and capture provider without loading a serving policy."""
    plan_name = os.environ.get(PLAN_ENV)
    if not plan_name:
        raise RuntimeError(f"{PLAN_ENV} must name the explicit calibration topology preset")
    api = _capture_api()
    api.configure_capture_runtime(plan_name)
    return api


class MaskReuseCaptureWorker(BaseWorker):
    """Run the custom backend in env-gated, policy-free capture mode."""

    def load_model(self, *args, **kwargs) -> None:
        """Install capture runtime before vLLM constructs attention modules."""
        # The attention implementation resolves process-local runtime state
        # while the model is loading.  Install the policy-free capture provider
        # first; a promoted v3 policy must not be required to collect its own
        # calibration evidence.
        api = _configure_capture_before_model_load()
        super().load_model(*args, **kwargs)
        status = api.capture_status()
        if not isinstance(status, Mapping) or status.get("available") is not True:
            reason = status.get("reason") if isinstance(status, Mapping) else None
            raise RuntimeError(f"mask-reuse capture backend is unavailable: {reason or status!r}")

    def mask_reuse_capture_status(self) -> dict[str, object]:
        """Return this worker rank's fail-closed capture status."""
        return dict(_capture_api().capture_status())

    def mask_reuse_capture_begin(self, invocation: dict[str, object]) -> dict[str, object]:
        """Arm exactly one prompt/target invocation on this worker rank."""
        return dict(_capture_api().begin_capture(invocation))

    def mask_reuse_capture_drain(self) -> dict[str, object]:
        """Drain one completed rank-local sufficient-stat payload."""
        return dict(_capture_api().drain_capture())
