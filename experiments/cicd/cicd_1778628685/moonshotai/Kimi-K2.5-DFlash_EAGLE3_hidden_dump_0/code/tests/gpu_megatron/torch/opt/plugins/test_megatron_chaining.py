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

from _test_utils.torch.megatron.models import MegatronModel
from _test_utils.torch.megatron.utils import initialize_for_megatron, sharded_state_dict_test_helper
from _test_utils.torch.sparsity.utils import sample_subnet_with_sparsity
from megatron.core.parallel_state import destroy_model_parallel

import modelopt.torch.quantization as mtq
import modelopt.torch.sparsity as mts


def test_sharded_state_dict(tmp_path, distributed_setup_size_1):
    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    model_ref = MegatronModel().cuda()
    input = model_ref.get_dummy_input().cuda()

    model_ref = mts.sparsify(model_ref, mode="sparse_magnitude")
    sample_subnet_with_sparsity(model_ref)

    def forward_fn(model):
        return model(input)

    model_ref = mtq.quantize(model_ref, mtq.INT8_DEFAULT_CFG, forward_fn)

    model_test = MegatronModel().cuda()

    sharded_state_dict_test_helper(tmp_path, model_ref, model_test, forward_fn)

    # Clean up since this is not a spawned process
    destroy_model_parallel()
