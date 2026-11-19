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

"""Synthetic CPU workflow fixture; the values are not model benchmark results."""

import json
import os
import sys
import time
from pathlib import Path

__all__ = ["main"]


def main():
    """Produce a toy checkpoint, exercise a parallel branch and join its outputs."""
    stage = sys.argv[1]
    if stage == "produce":
        result = {"synthetic": True, "value": int(os.environ["SYNTHETIC_VALUE"])}
    elif stage in ("accuracy", "speed"):
        model_path = Path(os.environ["MODEL_INPUT"])
        model = json.loads(
            (model_path / "checkpoint.json" if model_path.is_dir() else model_path).read_text()
        )
        barrier = Path(os.environ["MODELOPT_WORKSPACE"]) / "barrier"
        barrier.mkdir(exist_ok=True)
        (barrier / stage).touch()
        other = "speed" if stage == "accuracy" else "accuracy"
        timeout = float(os.environ.get("WORKFLOW_BARRIER_TIMEOUT_SECONDS", "15"))
        deadline = time.monotonic() + timeout
        while timeout > 0 and not (barrier / other).exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("Independent tasks did not execute concurrently")
            time.sleep(0.05)
        result = {"synthetic": True, stage: model["value"]}
    elif stage == "report":
        result = {"synthetic": True}
        for key in ("ACCURACY_INPUT", "SPEED_INPUT"):
            result.update(json.loads(Path(os.environ[key]).read_text()))
    else:
        raise ValueError("Unknown smoke stage")
    Path(os.environ["RESULT"]).write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
