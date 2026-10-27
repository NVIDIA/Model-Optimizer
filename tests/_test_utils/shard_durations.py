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

"""Write ``tests/shard_durations.json`` from the logs of CI lane jobs that ran ``pytest -v``.

Usage: ``python tests/_test_utils/shard_durations.py LOG [LOG ...] > tests/shard_durations.json``, with the log of
each shard of each sharded lane (``gh run view --job <id> --log``, or the log file downloaded from the job page).
Actions timestamps every line, so the time of a file is the time between its result lines; the median over the
logs is kept. A file that ran in several logs is counted per log.
"""

import collections
import datetime
import json
import re
import statistics
import sys

RESULT = re.compile(r"^(\S+Z)\s+(tests/\S+?)::\S+ .*?(?:PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)")


def parse(log: str) -> dict[str, float]:
    seconds: dict[str, float] = collections.defaultdict(float)
    previous = None
    for line in log.splitlines():
        if not (m := RESULT.match(line)):
            continue
        # Actions timestamps carry 7 fractional digits, which fromisoformat does not accept
        stamp = datetime.datetime.fromisoformat(re.sub(r"(\.\d{6})\d*Z$", r"\1+00:00", m[1]))
        seconds[m[2]] += (stamp - previous).total_seconds() if previous else 0.0
        previous = stamp
    return seconds


def main(paths: list[str]) -> dict[str, float]:
    per_file: dict[str, list[float]] = collections.defaultdict(list)
    for path in paths:
        with open(path, errors="replace") as f:
            for name, s in parse(f.read()).items():
                per_file[name].append(s)
    return {name: round(statistics.median(v), 1) for name, v in sorted(per_file.items())}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1:]), indent=1))
