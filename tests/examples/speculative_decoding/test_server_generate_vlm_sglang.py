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

"""Tests for the multimodal SGLang generation client."""

import importlib.util
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).parents[3]
    / "examples/speculative_decoding/distributed_generate/server_generate_vlm_sglang.py"
)
_spec = importlib.util.spec_from_file_location("server_generate_vlm_sglang", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
server_generate_vlm_sglang = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server_generate_vlm_sglang)


def test_resolve_media_path_supports_local_and_remote_media(tmp_path):
    """Existing local media and supported remote media remain usable."""

    local_media = tmp_path / "image.jpg"
    local_media.touch()
    remote_media = "https://example.com/image.jpg"

    assert server_generate_vlm_sglang._resolve_media_path("image.jpg", str(tmp_path), None) == str(
        local_media
    )
    assert (
        server_generate_vlm_sglang._resolve_media_path(remote_media, str(tmp_path), None)
        == remote_media
    )


def test_resolve_media_path_returns_none_and_warns_once_for_missing_media(monkeypatch, capsys):
    """Missing local media can fall back to video or be skipped by the caller."""

    monkeypatch.setattr(server_generate_vlm_sglang, "_UNRESOLVED_MEDIA_PATHS", set())

    assert server_generate_vlm_sglang._resolve_media_path("missing.mp4", None, None) is None
    assert server_generate_vlm_sglang._resolve_media_path("missing.mp4", None, None) is None

    assert capsys.readouterr().out == "WARNING: could not resolve media path: missing.mp4\n"


def test_openai_media_value_is_relative_to_the_media_root(tmp_path):
    """The local HTTP server exposes media files, not the container root."""

    media_root = tmp_path / "media"
    media_path = media_root / "videos" / "clip.mp4"
    media_path.parent.mkdir(parents=True)
    media_path.touch()
    input_path = tmp_path / "input" / "private.json"
    input_path.parent.mkdir()
    input_path.touch()

    assert (
        server_generate_vlm_sglang._as_openai_media_value(
            str(media_path), "http://127.0.0.1:18080", str(media_root), None
        )
        == "http://127.0.0.1:18080/videos/clip.mp4"
    )
    assert server_generate_vlm_sglang._as_openai_media_value(
        str(input_path), "http://127.0.0.1:18080", str(media_root), None
    ) == str(input_path)
