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

from functools import partial

from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.megatron.models import MegatronModel
from _test_utils.torch.megatron.utils import initialize_for_megatron, sharded_state_dict_test_helper
from _test_utils.torch.sparsity.utils import sample_subnet_with_sparsity

from modelopt.torch.opt.conversion import apply_mode


def _test_sharded_state_dict(tmp_path, rank, size):
    initialize_for_megatron()

    model_ref = MegatronModel(size).cuda()
    input = model_ref.get_dummy_input().cuda()

    model_ref = apply_mode(model_ref, "sparse_magnitude")
    sample_subnet_with_sparsity(model_ref)

    model_test = MegatronModel(size).cuda()

    sharded_state_dict_test_helper(tmp_path, model_ref, model_test, lambda model: model(input))


def test_sharded_state_dict(tmp_path):
    spawn_multiprocess_job(size=1, job=partial(_test_sharded_state_dict, tmp_path), backend="nccl")
