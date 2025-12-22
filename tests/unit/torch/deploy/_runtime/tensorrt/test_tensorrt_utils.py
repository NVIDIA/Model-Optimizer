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

import sys
from types import ModuleType
from unittest import mock

import numpy as np
import pytest
from _test_utils.torch.deploy.lib_test_models import LeNet5TwoInputs

tensorrt = ModuleType("tensorrt")
tensorrt.Logger = mock.Mock(name="tensorrt_logger")  # type: ignore[attr-defined]
tensorrt.tensorrt = mock.Mock(name="tensorrt_tensorrt")  # type: ignore[attr-defined]
tensorrt.__version__ = "8.2"  # type: ignore[attr-defined]
sys.modules[tensorrt.__name__] = tensorrt

from modelopt.torch._deploy._runtime.tensorrt.tensorrt_utils import calib_data_generator


@pytest.mark.parametrize(
    ("batch_size", "num_batches", "total_batch"),
    [
        (1, 1, 1),
        (1, 2, 2),
        (2, 1, 2),
        (2, 2, 4),
        (2, 2, 5),
    ],
)
def test_calib_data_generator(batch_size, num_batches, total_batch):
    model = LeNet5TwoInputs()
    onnx_bytes = model.to_onnx_bytes(batch_size)

    inputs_1 = LeNet5TwoInputs.gen_input(0, total_batch)
    inputs_2 = LeNet5TwoInputs.gen_input(1, total_batch)

    generator = calib_data_generator(onnx_bytes, [inputs_1, inputs_2])
    count = 0
    for feed_dict in generator:
        assert len(feed_dict) == 2

        assert np.array_equal(
            feed_dict["x"].detach().numpy(),
            inputs_1[count * batch_size : (count + 1) * batch_size].detach().numpy(),
        )
        assert np.array_equal(
            feed_dict["x1"].detach().numpy(),
            inputs_2[count * batch_size : (count + 1) * batch_size].detach().numpy(),
        )

        count += 1

    assert count == num_batches
