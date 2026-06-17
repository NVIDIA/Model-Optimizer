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
"""Re-export ``examples/hf_ptq/example_utils`` so tests can import it via
``from _test_utils.examples.llm_ptq_example_utils import example_utils``
without per-file ``sys.path`` shims.
"""

import sys

from _test_utils.examples.run_command import MODELOPT_ROOT

_LLM_PTQ_DIR = MODELOPT_ROOT / "examples" / "hf_ptq"
if str(_LLM_PTQ_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_PTQ_DIR))

import example_utils

__all__ = ["example_utils"]
