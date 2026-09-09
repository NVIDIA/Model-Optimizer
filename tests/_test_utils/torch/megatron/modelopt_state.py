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
"""ModelOpt-state checks for Megatron distributed checkpoints."""

from pathlib import Path

from megatron.bridge.training.post_training.checkpointing import has_modelopt_state
from torch.distributed.checkpoint import FileSystemReader

__all__ = ["assert_has_modelopt_state", "assert_no_quantizers_matching"]


def assert_has_modelopt_state(megatron_path: Path | str) -> None:
    """Assert a Megatron checkpoint carries restorable ModelOpt state.

    ``rglob("modelopt_state")`` passes on an empty state, which exports unquantized.
    """

    state_dirs = list(Path(megatron_path).rglob("modelopt_state"))
    assert state_dirs, f"No modelopt_state directory under {megatron_path}"
    assert has_modelopt_state(str(megatron_path)), (
        f"modelopt_state under {megatron_path} holds no restorable mode (only 'kd_loss' or "
        "empty), so the quantizers would not survive a reload"
    )


def assert_no_quantizers_matching(megatron_path: Path | str, *substrings: str) -> None:
    """Assert no calibrated quantizer under ``megatron_path`` matches ``substrings``.

    Disabled-quantizer patterns are written against HuggingFace names, so they silently
    no-op wherever Megatron names the module differently.
    """
    iter_dirs = sorted(Path(megatron_path).glob("iter_*"))
    assert iter_dirs, f"No iter_* checkpoint under {megatron_path}"
    keys = FileSystemReader(str(iter_dirs[-1])).read_metadata().state_dict_metadata
    quantizers = [k for k in keys if "_quantizer." in k]
    assert quantizers, f"No quantizers at all under {megatron_path}; was the model quantized?"
    for substring in substrings:
        hits = sorted(k for k in quantizers if substring in k)
        assert not hits, f"Expected no quantizer matching {substring!r}, found: {hits[:4]}"
