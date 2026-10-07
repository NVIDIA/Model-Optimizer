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

import base64
import importlib.util
from pathlib import Path

import pytest

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


def test_openai_media_value_inlines_local_media_as_a_data_uri(tmp_path):
    """--media_inline makes a request self-contained, with no media server."""

    media = tmp_path / "frame.png"
    media.write_bytes(b"\x89PNG\r\n\x1a\n")

    value = server_generate_vlm_sglang._as_openai_media_value(
        str(media), None, None, None, media_inline=True
    )

    assert value.startswith("data:image/png;base64,")
    assert base64.b64decode(value.split(",", 1)[1]) == b"\x89PNG\r\n\x1a\n"


def test_openai_media_value_inline_rejects_missing_media(tmp_path):
    """Inlining a file that is not there must fail loudly, not send an empty payload."""

    with pytest.raises(FileNotFoundError):
        server_generate_vlm_sglang._as_openai_media_value(
            str(tmp_path / "gone.png"), None, None, None, media_inline=True
        )


def test_openai_media_value_warns_once_when_passing_a_bare_local_path(monkeypatch, capsys):
    """Without a delivery option the path only resolves on servers that read local files.

    SGLang native does; vLLM does not unless started with --allowed-local-media-path. The
    behavior is kept for backward compatibility, so the warning is what makes an
    unfetchable request diagnosable instead of a confusing server-side error.
    """

    monkeypatch.setattr(server_generate_vlm_sglang, "_WARNED_LOCAL_MEDIA", False)

    assert server_generate_vlm_sglang._as_openai_media_value("/data/a.mp4", None, None, None) == (
        "/data/a.mp4"
    )
    assert server_generate_vlm_sglang._as_openai_media_value("/data/b.mp4", None, None, None) == (
        "/data/b.mp4"
    )

    assert capsys.readouterr().out.count("WARNING:") == 1


def test_openai_media_value_prefers_remote_urls_over_inlining(tmp_path):
    """An http(s)/data value is already fetchable and must pass through untouched."""

    remote = "https://example.com/clip.mp4"
    assert (
        server_generate_vlm_sglang._as_openai_media_value(remote, None, None, None, True) == remote
    )
