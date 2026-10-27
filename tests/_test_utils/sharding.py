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

"""Duration-balanced sharding of test files, to run one suite on several CI runners.

``pytest --shard-count N --shard-index I`` runs the files of shard I out of N. Whole files go to a shard, so
module-scoped fixtures (worker pools, model builds) are not repeated across shards. The assignment is
deterministic, so every shard computes the same partition from ``tests/shard_durations.json`` (seconds per
file, see ``shard_durations.py`` to refresh it); a file without an entry counts as the median file.
"""

import json
from pathlib import Path

import pytest

DURATIONS_FILE = Path(__file__).resolve().parents[1] / "shard_durations.json"


def assign_files(durations: dict[str, float], files: list[str], count: int) -> list[list[str]]:
    """Longest file first onto the least loaded shard. Returns ``count`` lists that partition ``files``."""
    known = sorted(durations[f] for f in files if f in durations)
    default = known[len(known) // 2] if known else 1.0
    seconds = {f: durations.get(f, default) for f in files}
    loads = [0.0] * count
    shards: list[list[str]] = [[] for _ in range(count)]
    for f in sorted(files, key=lambda f: (-seconds[f], f)):
        i = loads.index(min(loads))
        loads[i] += seconds[f]
        shards[i].append(f)
    return shards


def select_shard(config, items, index: int, count: int) -> None:
    """Keep only the items of shard ``index`` (1-based) of ``count``, deselecting the rest in place."""
    if not 1 <= index <= count:
        raise pytest.UsageError(
            f"--shard-index must be between 1 and --shard-count ({count}), got {index}"
        )

    def file_of(item):
        return item.path.relative_to(config.rootpath).as_posix()

    durations = json.loads(DURATIONS_FILE.read_text()) if DURATIONS_FILE.exists() else {}
    mine = set(assign_files(durations, sorted({file_of(item) for item in items}), count)[index - 1])
    kept = [item for item in items if file_of(item) in mine]
    deselected = [item for item in items if file_of(item) not in mine]
    config.hook.pytest_deselected(items=deselected)
    items[:] = kept
