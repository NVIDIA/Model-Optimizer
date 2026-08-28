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

__all__ = ["assert_has_modelopt_state"]


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
