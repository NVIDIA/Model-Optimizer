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
from _test_utils.torch.deploy.device_model import device_model_tester
from _test_utils.torch.deploy.lib_test_models import BaseDeployModel, get_deploy_models

deploy_benchmark = get_deploy_models()


@pytest.mark.parametrize("model", deploy_benchmark.values(), ids=deploy_benchmark.keys())
def test_ort_device_model(model: BaseDeployModel):
    return device_model_tester(model, {"runtime": "ORT"})
