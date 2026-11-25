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

import pytest
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from modelopt.torch.nas.registry import DMRegistry


def _get_nn_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding="same"),
        nn.Sequential(nn.BatchNorm2d(16)),
        nn.Conv2d(16, 3, 3, padding="same"),
        nn.InstanceNorm2d(3),
    )


def _get_dnn_model():
    model = _get_nn_model()
    for mod in model.modules():
        if mod in DMRegistry:
            DMRegistry.convert(mod)
    return model


@pytest.mark.parametrize("get_model", [_get_nn_model, _get_dnn_model])
def test_syncbn_conversion(get_model):
    model = get_model()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # check that all BNs are SyncBNs as expected
    assert all(
        isinstance(m, nn.SyncBatchNorm) or not isinstance(m, _BatchNorm) for m in model.modules()
    )


def test_dnn_syncbn_conversion():
    # get model
    model = _get_dnn_model()
    fake_attr = "_fake_attr"
    active = 8

    # modify hparams in the bn of the model to see if everything carries over...
    dyn_bn = model[1][0]
    hp = dyn_bn.get_hparam("num_features")
    setattr(hp, fake_attr, None)
    hp.active = active

    # check if hp works as expected
    assert dyn_bn.weight.shape == (active,)

    # convert
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # check that now sync bn is correctly set up...
    dyn_syncbn = model[1][0]
    assert isinstance(dyn_syncbn, DMRegistry[nn.SyncBatchNorm]), "Expected dynamic SyncBN!"
    assert dyn_syncbn.get_hparam("num_features") is hp, "Expected same hparam object!"
    assert dyn_syncbn.weight.shape == (active,), "Expected same weight shape!"

    # check if we can still set sync_bn to max value and it works
    hp.active = hp.max
    assert dyn_syncbn.weight.shape == (hp.max,)
