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

"""NeMo Gym adapter of the verbosity gate: its MLflow metric names.

Gym metrics in MLflow: ``<agent>/mean/output_tokens`` with sample count ``coverage/scored``.
Retiring a tool means deleting its adapter and its references.
"""

import re

from _metric_util import positive

_GYM_TOKENS = re.compile(r"^(?P<agent>[^/]+)/mean/output_tokens$")


def tokens_from_run(metrics, tags, load_artifact=None):
    """Gym's ``((avg_tokens, count), None)`` or ``(None, reason)``; ``None`` if the run is not Gym's."""
    gym = {m.group("agent"): k for k in metrics if (m := _GYM_TOKENS.match(k))}
    if len(gym) > 1:
        return None, f"several agents report output tokens: {sorted(gym)}"
    if not gym:
        return None
    key = next(iter(gym.values()))
    count = metrics.get("coverage/scored")
    if not positive(metrics[key]) or not positive(count):
        return None, f"{key}={metrics[key]!r} coverage/scored={count!r}"
    return (metrics[key], count), None
