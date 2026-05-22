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

"""Set ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` before vLLM is imported so
``LLM.collective_rpc(callable)`` can pickle worker callables. pytest loads
conftests before sibling test modules, so this beats the top-level
``from vllm import LLM`` in ``test_*.py``.
"""

import os

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
