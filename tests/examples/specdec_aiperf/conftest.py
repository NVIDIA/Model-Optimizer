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

# bench.py, plus specdec_bench for the format-parity test.
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
for _path in (_EXAMPLES / "specdec_aiperf", _EXAMPLES / "specdec_bench"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
