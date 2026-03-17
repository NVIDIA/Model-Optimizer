# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Eager attention backends for skip-softmax sparse attention."""

import contextlib
import threading

# ---------------------------------------------------------------------------
# Thread-local context: shared by diffusers and LTX backends
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def set_skip_softmax_context(active: bool) -> None:
    """Set thread-local flag indicating skip-softmax eager attention is active."""
    _thread_local.skip_softmax_active = active


def get_skip_softmax_context() -> bool:
    """Return True if skip-softmax eager attention is active in this thread."""
    return getattr(_thread_local, "skip_softmax_active", False)


# ---------------------------------------------------------------------------
# Optional backend registrations (depend on diffusers / ltx_core)
# ---------------------------------------------------------------------------
register_diffusers_eager_attention = None
register_ltx_eager_attention = None

with contextlib.suppress(ImportError):
    from .diffusers_eager_attention import register_diffusers_eager_attention

with contextlib.suppress(ImportError):
    from .ltx_eager_attention import register_ltx_eager_attention

__all__ = [
    "get_skip_softmax_context",
    "register_diffusers_eager_attention",
    "register_ltx_eager_attention",
    "set_skip_softmax_context",
]
