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

"""Tests for modelopt/torch/export/plugins/hf_checkpoint_utils.py"""

from unittest.mock import patch

import pytest

pytest.importorskip("huggingface_hub")

from modelopt.torch.export import copy_hf_ckpt_remote_code


def test_copy_hf_ckpt_remote_code_local_dir(tmp_path):
    """copy_hf_ckpt_remote_code copies top-level .py files from a local directory."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "modeling_custom.py").write_text("# custom model")
    (src_dir / "configuration_custom.py").write_text("# custom config")
    (src_dir / "not_python.txt").write_text("not python")
    (src_dir / "subdir").mkdir()
    (src_dir / "subdir" / "nested.py").write_text("# nested — should not be copied")

    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()

    copy_hf_ckpt_remote_code(src_dir, dst_dir)

    assert (dst_dir / "modeling_custom.py").read_text() == "# custom model"
    assert (dst_dir / "configuration_custom.py").read_text() == "# custom config"
    assert not (dst_dir / "not_python.txt").exists(), "non-.py files should not be copied"
    assert not (dst_dir / "nested.py").exists(), "nested .py files should not be copied"


def test_copy_hf_ckpt_remote_code_local_dir_no_py_files(tmp_path):
    """copy_hf_ckpt_remote_code is a no-op when the local directory has no .py files."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "config.json").write_text("{}")

    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()

    copy_hf_ckpt_remote_code(src_dir, dst_dir)  # should not raise

    assert list(dst_dir.iterdir()) == [], "no files should be copied"


def test_copy_hf_ckpt_remote_code_hub_id(tmp_path):
    """copy_hf_ckpt_remote_code delegates to snapshot_download for a Hub model ID."""
    dst_dir = tmp_path / "dst"

    with patch("modelopt.torch.export.plugins.hf_checkpoint_utils.snapshot_download") as mock_sd:
        copy_hf_ckpt_remote_code("nvidia/NVIDIA-Nemotron-Nano-12B-v2", dst_dir)

    mock_sd.assert_called_once_with(
        repo_id="nvidia/NVIDIA-Nemotron-Nano-12B-v2",
        local_dir=str(dst_dir),
        allow_patterns=["*.py"],
    )
