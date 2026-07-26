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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace

from modelopt.torch.utils.nemotron_vlm_dataset_utils import list_repo_files_cached


def test_repo_listing_forwards_immutable_revision(monkeypatch):
    calls = []
    fake_hub = SimpleNamespace(
        list_repo_files=lambda **kwargs: calls.append(kwargs) or ["subset/media/shard.tar"]
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    list_repo_files_cached.cache_clear()

    files = list_repo_files_cached("nvidia/repo", revision="abc123")

    assert files == ["subset/media/shard.tar"]
    assert calls == [
        {
            "repo_id": "nvidia/repo",
            "repo_type": "dataset",
            "revision": "abc123",
        }
    ]
