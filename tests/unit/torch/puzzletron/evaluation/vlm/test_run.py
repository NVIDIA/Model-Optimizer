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

"""Tests for the VLM evaluation command-line entry point."""

import subprocess
import sys
from pathlib import Path

from examples.puzzletron.evaluation.vlm import run as evaluation


def test_direct_launcher_does_not_shadow_standard_library_profile():
    script = Path(evaluation.__file__).absolute()
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy, sys; "
                "sys.path.insert(0, sys.argv[1]); "
                "runpy.run_path(sys.argv[2], run_name='modelopt_vlm_launcher'); "
                "import cProfile; "
                "assert callable(cProfile.run)"
            ),
            str(script.parent),
            str(script),
        ],
        check=True,
    )


def test_vlm_parser_exposes_only_suite_owned_sample_limits():
    parser = evaluation._build_parser()
    help_text = parser.format_help()
    assert "--limit" not in help_text
    assert "--full" not in help_text
    assert "--tasks" not in help_text
    assert "--evaluation-profile" not in help_text
    assert parser.get_default("suite") == "short"
