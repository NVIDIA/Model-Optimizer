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

from collections import defaultdict

import pytest

from modelopt.torch.opt.mode import _ModeRegistryCls


@pytest.fixture
def use_channel_div_4():
    """To use channel_divisor=4 in any test, just add this fixture as an argument to the test.

    If using this fixture, make sure to not overwrite the config further in the test otherwise
    it would globally change the config for other tests.
    """
    # modes to patch
    modes_to_patch = ["fastnas", "autonas", "gradnas"]

    # lookup to store original divisor values
    lookups_divisor = defaultdict(lambda: defaultdict(dict))

    # patch with channel_divisor=4
    for mode in modes_to_patch:
        config_class = _ModeRegistryCls.get_from_any(mode).config_class
        for name, field_info in config_class.model_fields.items():
            for regex, val in field_info.default.items():
                for k in val:
                    if "divisor" in k:
                        lookups_divisor[mode][name][regex] = (k, val[k])
                        val[k] = 4

    # yield, i.e., run test
    yield

    # restore original divisor values
    for mode, lookup in lookups_divisor.items():
        config_class = _ModeRegistryCls.get_from_any(mode).config_class
        for name, kv_for_all in lookup.items():
            for regex, (k, v) in kv_for_all.items():
                config_class.model_fields[name].default[regex][k] = v
