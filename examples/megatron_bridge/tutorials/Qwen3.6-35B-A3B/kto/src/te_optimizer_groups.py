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

"""Avoid FusedAdam tensor-handle exhaustion during optimizer updates.

The tutorial needs this workaround on NeMo 26.08.00. In NeMo 26.10 RC8,
FusedAdam already limits tensor-handle use by splitting large updates.
"""

from functools import wraps

import transformer_engine

__all__ = ["bound_fused_adam_groups"]


def bound_fused_adam_groups() -> None:
    """Keep FusedAdam from exhausting tensor handles during an optimizer update.

    On the pinned NeMo 26.08 build, split large parameter groups into chunks
    of at most 512 tensors to limit how many handles FusedAdam uses at once.
    Each chunk keeps the original group's optimizer settings.
    """
    if transformer_engine.__version__ != "2.17.1+4329ff84":
        return

    from transformer_engine.pytorch.optimizers import FusedAdam

    if getattr(FusedAdam, "_qakto_bounded_groups", False):
        return
    original_init = FusedAdam.__init__

    @wraps(original_init)
    def init(self, params, *args, **kwargs):
        params = list(params)
        groups = params if not params or isinstance(params[0], dict) else [{"params": params}]
        bounded = []
        for group in groups:
            parameters = list(group["params"])
            # Keep each group's Adam settings and preserve its parameter order.
            bounded.extend(
                dict(group, params=parameters[start : start + 512])
                for start in range(0, max(1, len(parameters)), 512)
            )
        original_init(self, bounded, *args, **kwargs)

    FusedAdam.__init__ = init
    FusedAdam._qakto_bounded_groups = True
