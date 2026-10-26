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

import hashlib
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("cv2")
_SCRIPT = (
    Path(__file__).parents[3] / "examples/windows/accuracy_benchmark/fvd_metrics/compute_fvd.py"
)
_SPEC = importlib.util.spec_from_file_location("sonar_compute_fvd", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
compute_fvd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compute_fvd)


def test_cached_weights_are_verified_on_every_load(tmp_path, monkeypatch):
    cache = tmp_path / "cached.pt"
    cache.write_bytes(b"trusted checkpoint")
    monkeypatch.setattr(compute_fvd, "DEFAULT_CACHE", cache)
    monkeypatch.setattr(
        compute_fvd, "WEIGHTS_SHA256", hashlib.sha256(cache.read_bytes()).hexdigest()
    )
    assert compute_fvd.resolve_weights(None) == cache
    cache.write_bytes(b"modified checkpoint")
    with pytest.raises(RuntimeError, match="Cached I3D weights failed"):
        compute_fvd.resolve_weights(None)


def test_custom_weights_checksum_is_enforced(tmp_path):
    path = tmp_path / "custom.pt"
    path.write_bytes(b"custom checkpoint")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert compute_fvd.resolve_weights(str(path), digest.upper()) == path
    with pytest.raises(RuntimeError, match="Supplied I3D weights failed"):
        compute_fvd.resolve_weights(str(path), "0" * 64)


def test_checksum_option_requires_custom_weights(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["compute_fvd.py", "--ref-dir", "ref", "--gen-dir", "gen", "--weights-sha256", "0" * 64],
    )
    with pytest.raises(SystemExit) as exc:
        compute_fvd.parse_args()
    assert exc.value.code == 2
