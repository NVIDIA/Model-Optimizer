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


import pytest
import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir


@pytest.fixture(scope="session")
def num_gpus():
    return torch.cuda.device_count()


@pytest.fixture(scope="session")
def cuda_capability():
    return torch.cuda.get_device_capability()


@pytest.fixture(scope="session")
def tiny_llama_path(tmp_path_factory):
    return str(
        create_tiny_llama_dir(
            tmp_path_factory.mktemp("tiny_llama"),
            with_tokenizer=True,
            hidden_size=512,
            intermediate_size=512,
        )
    )
