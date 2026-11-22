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
from pathlib import Path
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu_worker import Worker as BaseWorker

from .mask_reuse_stream import validate_capture_stream

if TYPE_CHECKING:
    from types import ModuleType

CAPTURE_ENV = "MASK_REUSE_FA4_CALIBRATION_CAPTURE"
PLAN_ENV = "MASK_REUSE_FA4_PLAN"
PROVIDER_ENV = "MASK_REUSE_CALIBRATION_PROVIDER"
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
    provider = os.environ.get(PROVIDER_ENV, "fa4")
    modules = {
        "fa4": "mask_reuse_vllm.capture",
        "reference": "mask_reuse_vllm.reference.capture",
    }
    if provider not in modules:
        raise RuntimeError(f"{PROVIDER_ENV} must be fa4 or reference")
    module = modules[provider]
    try:
        api = importlib.import_module(module)
    except ImportError as error:
        raise RuntimeError(f"the {provider} mask-reuse backend does not expose {module}") from error
    if provider == "reference":
        root = os.environ.get("MASK_REUSE_CALIBRATION_SOURCE_ROOT")
        expected = (
            Path(root) / "src/attention_backend/src/mask_reuse_vllm/reference/capture.py"
            if root
            else None
        )
        actual = getattr(api, "__file__", None)
        if expected is None or not actual or Path(actual).resolve() != expected.resolve():
            raise RuntimeError(
                "reference capture module is not loaded from the witnessed source root"
            )
    missing = [name for name in _REQUIRED_API if not callable(getattr(api, name, None))]
    if provider == "reference" and not callable(getattr(api, "install_capture", None)):
        missing.append("install_capture")
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

    def _validate_capture_cache_dtype(self) -> None:
        cache_dtype = self.vllm_config.cache_config.cache_dtype
        if cache_dtype != "bfloat16":
            raise RuntimeError(
                "mask-reuse capture requires explicit kv_cache_dtype='bfloat16'; "
                f"vLLM resolved {cache_dtype!r}. Checkpoint weight quantization is unchanged."
            )

    def load_model(self, *args, **kwargs) -> None:
        """Install capture runtime before vLLM constructs attention modules."""
        self._validate_capture_cache_dtype()
        # The attention implementation resolves process-local runtime state
        # while the model is loading.  Install the policy-free capture provider
        # first; a promoted v3 policy must not be required to collect its own
        # calibration evidence.
        api = _configure_capture_before_model_load()
        super().load_model(*args, **kwargs)
        self._validate_capture_cache_dtype()
        if os.environ.get(PROVIDER_ENV, "fa4") == "reference":
            api.install_capture(self.model_runner)
        status = api.capture_status()
        if not isinstance(status, Mapping) or status.get("available") is not True:
            reason = status.get("reason") if isinstance(status, Mapping) else None
            raise RuntimeError(f"mask-reuse capture backend is unavailable: {reason or status!r}")

    def mask_reuse_capture_status(self) -> dict[str, object]:
        """Return this worker rank's fail-closed capture status."""
        return dict(_capture_api().capture_status())

    def mask_reuse_capture_runtime_geometry(self) -> dict[str, int]:
        """Report actual cache pages after vLLM finishes hybrid-cache allocation."""
        page = self.vllm_config.cache_config.block_size
        reference = os.environ.get(PROVIDER_ENV, "fa4") == "reference"
        if type(page) is not int or page not in ((16, 4160) if reference else (16,)):
            raise RuntimeError("capture backend does not support the resolved KV page size")
        geometry = getattr(_capture_api(), "runtime_geometry", None)
        if callable(geometry):
            if geometry() != {"kv_page_tokens": page}:
                raise RuntimeError("reference capture and vLLM disagree on actual KV pages")
        elif page != 16:
            raise RuntimeError("nondefault KV pages require an explicit provider geometry API")
        return {"kv_page_tokens": page}

    def mask_reuse_capture_begin(self, invocation: dict[str, object]) -> dict[str, object]:
        """Arm exactly one prompt/target invocation on this worker rank."""
        return dict(_capture_api().begin_capture(invocation))

    def mask_reuse_capture_drain(self) -> dict[str, object]:
        """Drain one completed rank-local sufficient-stat payload."""
        return dict(_capture_api().drain_capture())

    def mask_reuse_capture_stream_begin(
        self, request: dict[str, object], spool_directory: str, completed_chunk_count: int = 0
    ) -> dict[str, object]:
        """Arm reference-only streaming capture without changing final-chunk behavior."""
        if os.environ.get(PROVIDER_ENV) != "reference":
            raise RuntimeError("whole-prefill streaming capture requires the reference provider")
        validate_capture_stream(request)
        begin = getattr(_capture_api(), "begin_capture_stream", None)
        if not callable(begin):
            raise RuntimeError(
                "reference provider does not support whole-prefill streaming capture"
            )
        return dict(begin(request, spool_directory, completed_chunk_count=completed_chunk_count))

    def mask_reuse_capture_stream_drain(self) -> dict[str, object]:
        """Return stream-completion evidence; statistics are read from rank-local spools."""
        if os.environ.get(PROVIDER_ENV) != "reference":
            raise RuntimeError("whole-prefill streaming capture requires the reference provider")
        drain = getattr(_capture_api(), "drain_capture_stream", None)
        if not callable(drain):
            raise RuntimeError(
                "reference provider does not support whole-prefill streaming capture"
            )
        return dict(drain())
