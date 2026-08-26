# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Kernel integrations for sparse attention: Triton FA and diffusers/LTX backends."""

import contextlib
import threading

from modelopt.torch.kernels.common.attention import (
    IS_AVAILABLE,
    attention,
    register_triton_attention,
)

# ---------------------------------------------------------------------------
# Optional backend registrations (depend on diffusers / ltx_core)
# ---------------------------------------------------------------------------
register_diffusers_triton_attention = None
register_ltx_triton_attention = None

# Suppress ImportError (missing package) and RuntimeError (triton without GPU driver)
with contextlib.suppress(ImportError, RuntimeError):
    from .diffusers_triton_attention import register_diffusers_triton_attention

with contextlib.suppress(ImportError, RuntimeError):
    from .ltx_triton_attention import register_ltx_triton_attention

# ---------------------------------------------------------------------------
# Thread-local flag for flash_skip_softmax's eager-attention context
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def set_skip_softmax_context(active: bool) -> None:
    """Set whether skip-softmax softmax patching is active (thread-local)."""
    _thread_local.skip_softmax_active = active


def get_skip_softmax_context() -> bool:
    """Return whether skip-softmax softmax patching is active."""
    return getattr(_thread_local, "skip_softmax_active", False)


__all__ = [
    "IS_AVAILABLE",
    "attention",
    "get_skip_softmax_context",
    "register_diffusers_triton_attention",
    "register_ltx_triton_attention",
    "register_triton_attention",
    "set_skip_softmax_context",
]
