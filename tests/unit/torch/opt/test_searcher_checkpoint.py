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

"""Search checkpoints must support fresh directories as well as single files."""

import pytest

from modelopt.torch.opt import searcher as searcher_module
from modelopt.torch.quantization.algorithms import AutoQuantizeGradientSearcher


@pytest.mark.parametrize("layout", ["existing_directory", "new_directory", "file", "legacy_file"])
def test_search_checkpoint_fresh_and_resume(tmp_path, monkeypatch, layout):
    checkpoint = tmp_path / ("state.pth" if layout.endswith("file") else "state")
    if layout == "existing_directory":
        checkpoint.mkdir()
    searcher = AutoQuantizeGradientSearcher()
    searcher.config = {"checkpoint": str(checkpoint)}
    searcher.reset_search()
    assert not searcher.load_search_checkpoint()
    searcher.best["score"] = 3.0
    searcher.save_search_checkpoint()
    searcher.reset_search()
    if layout == "legacy_file":
        monkeypatch.setattr(searcher_module.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(searcher_module.dist, "rank", lambda group=None: 0)
        assert not checkpoint.with_name("state0.pth").exists()
        with pytest.warns(UserWarning, match="falling back to"):
            assert searcher.load_search_checkpoint()
    else:
        assert searcher.load_search_checkpoint()
    assert searcher.best["score"] == 3.0
