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

"""Tests for the Puzzletron worker-image build command."""

import pytest

from examples.puzzletron.build_worker_image import artifact_names


def test_exported_artifacts_share_a_git_revision_identity():
    revision = "ba737f1f2301d0526c7d4674e1d21bf3d8c1ff14"

    assert artifact_names(revision) == {
        "archive": "modelopt-puzzletron-linux-amd64-git-ba737f1f2301.tar.zst",
        "sqsh": "modelopt-puzzletron-linux-amd64-git-ba737f1f2301.sqsh",
    }


def test_artifact_names_require_a_full_git_revision():
    with pytest.raises(ValueError, match="full lowercase Git commit"):
        artifact_names("ba737f1f2301")
