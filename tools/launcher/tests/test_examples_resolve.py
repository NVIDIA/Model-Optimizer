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

"""Structural validation of every launcher example YAML.

CPU-only (no GPU, no cluster): parses each `examples/**/*.yaml` and checks the
launcher-relevant shape — well-formed YAML, a `script:` per task, `common/*`
scripts that actually exist on disk (catches renamed/typo'd wrapper paths), a
`_factory_`, and list-shaped `args`. The actual quantize/train execution is
covered by cluster integration runs, not here; this guards against config
regressions (e.g. a new example pointing at a missing wrapper).
"""

import glob
import os

import pytest
import yaml

_LAUNCHER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLES = sorted(
    glob.glob(os.path.join(_LAUNCHER_DIR, "examples", "**", "*.yaml"), recursive=True)
)

# Pre-existing examples that reference a common/ script which no longer exists on
# disk (renamed/removed upstream, example not updated). Excluded from the
# script-exists check so this test guards NEW examples without failing on
# unrelated pre-existing drift. These are worth fixing separately:
#   common/smoke/hostname.sh          — examples/smoke/hostname.yaml
#   common/eagle3/offline_training.sh — examples/Qwen/Qwen3-8B/hf_offline_eagle3_ptq.yaml
#                                       (common/eagle3/ has train_eagle.sh, not offline_training.sh)
_KNOWN_MISSING_SCRIPTS = {
    "common/smoke/hostname.sh",
    "common/eagle3/offline_training.sh",
}


def _tasks(cfg):
    """Yield (name, task_dict) for a single-task or job_name+pipeline example."""
    if not isinstance(cfg, dict):
        return
    pipeline = cfg.get("pipeline")
    if isinstance(pipeline, dict):
        for key, val in pipeline.items():
            if key.startswith("task_") and isinstance(val, dict):
                yield key, val
    elif "script" in cfg:
        yield "task", cfg


def test_examples_present():
    """At least one launcher example YAML is discovered (guards the glob itself)."""
    assert _EXAMPLES, "no launcher example YAMLs found"


@pytest.mark.parametrize(
    "path", _EXAMPLES, ids=[os.path.relpath(p, _LAUNCHER_DIR) for p in _EXAMPLES]
)
def test_example_yaml_valid(path):
    """Each example parses and every task has a valid script/factory/args shape."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, (dict, list)), f"{path}: top-level YAML is not a mapping/list"

    for name, task in _tasks(cfg):
        script = task.get("script")
        assert isinstance(script, str) and script.strip(), f"{path}:{name}: missing `script`"

        # Scripts shipped with the launcher live under common/; verify the path is
        # real so a renamed/typo'd wrapper is caught at unit-test time.
        first_token = script.split()[0]
        if first_token.startswith("common/") and first_token not in _KNOWN_MISSING_SCRIPTS:
            assert os.path.exists(os.path.join(_LAUNCHER_DIR, first_token)), (
                f"{path}:{name}: script not found: {first_token}"
            )

        slurm_config = task.get("slurm_config")
        if slurm_config is not None:
            assert slurm_config.get("_factory_"), (
                f"{path}:{name}: slurm_config is missing `_factory_`"
            )

        args = task.get("args")
        assert args is None or isinstance(args, list), f"{path}:{name}: `args` must be a list"

        env = task.get("environment")
        assert env is None or isinstance(env, (list, dict)), (
            f"{path}:{name}: `environment` must be a list or dict"
        )
