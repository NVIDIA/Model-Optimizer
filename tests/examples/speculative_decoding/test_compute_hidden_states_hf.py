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

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_EXAMPLE = Path(__file__).parents[3] / "examples/speculative_decoding/collect_hidden_states"
sys.path.insert(0, str(_EXAMPLE))
_SPEC = importlib.util.spec_from_file_location(
    "sonar_hidden_states", _EXAMPLE / "compute_hidden_states_hf.py"
)
assert _SPEC is not None and _SPEC.loader is not None
hidden_states = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hidden_states)


@pytest.mark.parametrize("write_fails", [False, True])
def test_threaded_hidden_state_save_is_atomic(tmp_path, monkeypatch, write_fails):
    output = tmp_path / "conversation.pt"
    torch.save({"previous": True}, output)
    payload = {"hidden_states": torch.ones(2, 3)}
    if write_fails:

        def fail_after_partial_write(state, path):
            path.write_bytes(b"partial")
            raise OSError("disk write failed")

        monkeypatch.setattr(hidden_states.torch, "save", fail_after_partial_write)
        with pytest.raises(OSError, match="disk write failed"):
            asyncio.run(asyncio.to_thread(hidden_states._save_hidden_states, output, payload))
        assert torch.load(output, weights_only=True) == {"previous": True}
    else:
        asyncio.run(asyncio.to_thread(hidden_states._save_hidden_states, output, payload))
        torch.testing.assert_close(
            torch.load(output, weights_only=True)["hidden_states"], payload["hidden_states"]
        )
    assert not output.with_suffix(".pt.tmp").exists()
