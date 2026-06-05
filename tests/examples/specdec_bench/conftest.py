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

# Make examples/specdec_bench/specdec_bench/ + upload_to_s3.py importable from
# these tests. Anchored on the repo root via parents[3]:
#   __file__ = <repo>/tests/examples/specdec_bench/conftest.py
#   parents[3] = <repo>
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[3] / "examples" / "specdec_bench"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
