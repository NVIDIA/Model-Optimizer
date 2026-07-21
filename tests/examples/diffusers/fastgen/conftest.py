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

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def make_fastgen_cache():
    """Create a tiny, fully local FastGen latent cache."""
    torch = pytest.importorskip("torch")

    def _make(
        root: Path,
        *,
        count: int = 6,
        marker: float = 0.0,
        absolute_payloads: bool = False,
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        payload_dir = root / "payloads"
        payload_dir.mkdir()

        metadata = []
        for sample_id in range(count):
            payload_path = payload_dir / f"sample_{sample_id}.pt"
            torch.save(
                {
                    "latent": torch.full((4, 2, 2), marker + sample_id),
                    "crop_offset": (0, 0),
                    "prompt": f"prompt-{marker}-{sample_id}",
                    "image_path": f"/source/image_{sample_id}.png",
                    "prompt_embeds": torch.full((1, 2, 3), marker + sample_id),
                    "prompt_embeds_mask": torch.ones((1, 2), dtype=torch.long),
                },
                payload_path,
            )
            cache_file = payload_path if absolute_payloads else payload_path.relative_to(root)
            resolution = [64, 64] if sample_id % 2 == 0 else [64, 128]
            metadata.append(
                {
                    "cache_file": str(cache_file),
                    "bucket_resolution": resolution,
                    "original_resolution": resolution,
                    "bucket_id": sample_id % 2,
                    "aspect_ratio": resolution[0] / resolution[1],
                }
            )

        midpoint = max(1, count // 2)
        shard_names = ["metadata_shard_0.json", "metadata_shard_1.json"]
        for shard_name, items in zip(
            shard_names, (metadata[:midpoint], metadata[midpoint:]), strict=True
        ):
            (root / shard_name).write_text(json.dumps(items))
        (root / "metadata.json").write_text(json.dumps({"shards": shard_names}))
        torch.save(
            {
                "embed": torch.full((2, 3), marker),
                "mask": torch.ones(2, dtype=torch.long),
            },
            root / "negative_prompt_embedding.pt",
        )
        return root

    return _make
