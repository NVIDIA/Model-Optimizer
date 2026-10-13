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

"""The ``MODELOPT_*`` environment-variable knobs of distributed HuggingFace model loading.

Each knob is a small accessor that reads ``os.environ`` *at call time*, not at import: a launcher sets
them before it forks the distributed workers, and tests may override them, so the live value is what
matters. Callers do ``from modelopt.torch.utils import env`` then ``env.ep_degree()``.

Distribution plan (see ``resolve_distribution_plan`` and ``decide_distribution_plan``):
  MODELOPT_EP_DEGREE        expert-parallel degree; ``""``/unset -> ``"auto"``.
  MODELOPT_FSDP_SIZE        dense-FSDP degree; ``""``/unset -> ``"auto"`` (``"world"``/int also accepted).
"""

import os

__all__ = [
    "ep_degree",
    "fsdp_size",
]


def ep_degree() -> str:
    """Expert-parallel degree knob for ``decide_distribution_plan``; ``""``/unset -> ``"auto"``."""
    return os.environ.get("MODELOPT_EP_DEGREE") or "auto"


def fsdp_size() -> str:
    """Dense-FSDP degree knob for ``decide_distribution_plan``; ``""``/unset -> ``"auto"``."""
    return os.environ.get("MODELOPT_FSDP_SIZE") or "auto"
