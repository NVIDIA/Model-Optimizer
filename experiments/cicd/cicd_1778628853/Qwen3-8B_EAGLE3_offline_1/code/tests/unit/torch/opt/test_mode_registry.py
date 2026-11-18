# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from modelopt.torch.opt.mode import ModeDescriptor, _ModeRegistryCls


def test_mode_registry():
    """Test basic functionality of _ModeRegistryCls."""
    # Create a new registry
    registry = _ModeRegistryCls("test_registry")

    @registry.register_mode
    class TestMode(ModeDescriptor):
        @property
        def name(self) -> str:
            """Returns the value (str representation) of the mode."""
            return "a_test_mode"

        @property
        def config_class(self):
            """Returns the config class for the mode."""
            return None

        @property
        def convert(self):
            """Returns the convert function for the mode."""
            return None

        @property
        def restore(self):
            """Returns the restore function for the mode."""
            return None

    assert isinstance(_ModeRegistryCls.get_from_any("a_test_mode"), TestMode)

    registry.remove_mode("a_test_mode")

    assert not _ModeRegistryCls.contained_in_any("a_test_mode")

    del registry

    assert "test_registry" not in _ModeRegistryCls._all_registries
