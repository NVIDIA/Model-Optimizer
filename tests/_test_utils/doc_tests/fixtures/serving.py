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

"""Real, opt-in vLLM adapter serving verification."""

import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path


def prepare_adapter_server(ctx):
    """Link an existing QLoRA export into an isolated example workspace."""
    checkpoint = Path(os.environ["MODELOPT_DOC_QLORA_CHECKPOINT"]).resolve()
    assert (checkpoint / "adapter_config.json").is_file()
    assert (checkpoint / "base_model/config.json").is_file()
    ctx.cwd = ctx.tmp
    (ctx.cwd / "qwen3-8b-fp4-qlora-hf").symlink_to(checkpoint, target_is_directory=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        ctx.env["PORT"] = str(sock.getsockname()[1])


def verify_adapter_server(ctx):
    """Wait for readiness and require an actual completion from the served adapter."""
    url = f"http://127.0.0.1:{ctx.env['PORT']}"
    deadline = time.monotonic() + 600
    while True:
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                assert response.status == 200
            break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise TimeoutError("vLLM did not become ready") from None
            time.sleep(1)
    request = urllib.request.Request(
        url + "/v1/completions",
        data=json.dumps(
            {"model": "adapter", "prompt": "The capital of France is", "max_tokens": 4}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    assert result["choices"] and result["usage"]["completion_tokens"] > 0
