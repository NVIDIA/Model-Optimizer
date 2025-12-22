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

pytest.importorskip("transformers")
from functools import partial

from _test_utils.torch.distributed.utils import get_device_counts, spawn_multiprocess_job
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from transformers import AutoModelForCausalLM

from modelopt.torch.utils.distributed import is_dtensor_sharded


def _test_transformers_multi_process(rank, size, tmp_path):
    tiny_llama_dir = create_tiny_llama_dir(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(tiny_llama_dir, device_map="auto")
    assert not is_dtensor_sharded(model)


# TODO: remove this test and add test for TP sharded transformers models after we support TP sharded models
@pytest.mark.parametrize("device_count", get_device_counts())
def test_transformers_multi_process(tmp_path, device_count):
    spawn_multiprocess_job(
        size=device_count,
        job=partial(_test_transformers_multi_process, tmp_path=tmp_path),
        backend="nccl",
    )
