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

from contextlib import nullcontext

import pytest
import torch.nn as nn

from modelopt.torch.sparsity.weight_sparsity.config import SparseMagnitudeConfig
from modelopt.torch.sparsity.weight_sparsity.module import SparseModule, SpDMRegistry


def test_rule_based_config():
    """Try testing the rule-based config."""
    # some reused attributes
    rule_cls = SparseMagnitudeConfig
    original_len = len(rule_cls())

    lin_alias = "nn.Linear"
    lin_name = "nn_linear"
    lin_value = {}
    lin_test_config = {lin_alias: lin_value}
    lin_expected_value = {"*": lin_value}

    new_name = "nn.Conv1d"
    new_cls = nn.Conv1d
    new_value = {}
    new_test_config = {new_name: new_value}
    new_expected_value = {"*": new_value}
    new_default_value = {"*": {}, "*lm_head*": None}

    def _run_test(is_new_registered):
        # just initialize the config and check if it works
        config = rule_cls()
        config.model_dump()

        # check default value for the newly registered field
        if is_new_registered:
            assert config[new_name] == new_default_value

        # try customizing a field
        config = SparseMagnitudeConfig(**lin_test_config)
        assert config.model_dump()[lin_alias] == lin_expected_value

        # try adding an extra field that is not in the registry
        with nullcontext() if is_new_registered else pytest.raises(KeyError):
            config = SparseMagnitudeConfig(**new_test_config, **lin_test_config)
        if is_new_registered:
            assert config[new_name] == new_expected_value
            assert new_name in config.model_dump()

        # try various operators

        # __contains__
        assert lin_alias in config
        assert lin_name in config
        assert (new_name in config) == is_new_registered

        # __getitem__, __getattr__
        assert config[lin_name] == lin_expected_value
        assert config[lin_alias] == lin_expected_value
        assert getattr(config, lin_name) == lin_expected_value
        with nullcontext() if is_new_registered else pytest.raises(AttributeError):
            config[new_name]

        # get
        assert config.get(lin_alias) == lin_expected_value
        assert config.get(new_name) == (new_expected_value if is_new_registered else None)

        # len
        assert len(config) == original_len + (1 if is_new_registered else 0)

        # __iter__
        keys = set(config)
        assert lin_alias in keys
        assert lin_name not in keys
        assert (new_name in keys) == is_new_registered

        # keys
        assert lin_alias in config.keys()  # noqa: SIM118
        assert lin_name not in config.keys()  # noqa: SIM118
        assert (new_name in config.keys()) == is_new_registered  # noqa: SIM118

        # setitem (re-assign and re-validate)
        config[lin_alias] = lin_value
        assert config[lin_alias] == lin_expected_value

    # run the test with the new class not registered
    _run_test(False)

    # now try adding a new DM in the registry and see if it's dynamically detected
    SpDMRegistry.register({new_cls: new_name})(SparseModule)
    rule_cls.register_default({new_name: new_default_value})
    _run_test(True)

    # check if we can update the default value again for plugins
    rule_cls.register_default({new_name: None})

    # check that we cannot update the default value for a regular DM
    with pytest.raises(AssertionError):
        rule_cls.register_default({lin_name: None})

    # now unregister it again and run one more time
    SpDMRegistry.unregister(new_cls)
    rule_cls.unregister_default(new_name)
    _run_test(False)
