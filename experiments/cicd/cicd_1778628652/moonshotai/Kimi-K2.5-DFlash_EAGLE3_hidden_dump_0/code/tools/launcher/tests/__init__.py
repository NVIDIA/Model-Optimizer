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

"""Unit tests for the ModelOpt Launcher.

Coverage:
    - test_core.py: Shared dataclasses, factory registry, global_vars interpolation,
      version reporting, default env generation, and the run_jobs loop (mocked).
    - test_slurm_config.py: SlurmConfig dataclass defaults and slurm_factory behavior
      with environment variable overrides.
    - test_yaml_formats.py: YAML parsing for --yaml format, pipeline=@ format, and
      task_configs resolution via registered factories.

Not covered (requires live infrastructure):
    - Actual Slurm job submission (SSH tunnel, sbatch)
    - Docker container launch
    - nemo experiment status/logs polling
    - PatternPackager tar.gz creation and rsync
"""
