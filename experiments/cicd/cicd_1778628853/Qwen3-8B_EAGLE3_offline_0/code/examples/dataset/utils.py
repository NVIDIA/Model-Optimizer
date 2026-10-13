# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Utility functions."""

import hashlib
import json
from pathlib import Path

import aiohttp


async def download_file(url: str, destination: Path) -> None:
    """Download a file from a URL to a specified destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession() as session, session.get(url) as response:
        if response.status != 200:
            msg = f"Failed to download {url}: {response.status}"
            raise RuntimeError(msg)
        content = await response.read()
        destination.write_bytes(content)
        print(f"Downloaded {url} to {destination}")


def id_for_conversation(conversation: list) -> str:
    """Generate a unique ID for a conversation based on its content."""
    json_str = json.dumps(conversation, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    json_bytes = json_str.encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()
