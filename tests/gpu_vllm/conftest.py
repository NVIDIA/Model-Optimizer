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

"""Shared setup for vLLM tests.

vLLM handles its own distributed init, current-vllm-config context, and
parallel-state setup when ``LLM(...)`` is constructed, so this conftest only
opts into ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` *before* importing vLLM so
``LLM.collective_rpc(callable)`` can ship our worker callables over the engine
IPC channel via pickle. Without this, the default msgpack encoder rejects raw
functions and the call raises ``TypeError``. Only safe in a controlled test
environment.
"""

import os

# Must precede any ``import vllm``: the env is read at module-import time.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
