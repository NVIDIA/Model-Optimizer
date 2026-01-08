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

import modelopt.torch.prune as mtp


def prune_minitron(model, constraints, config, channel_divisor=64):
    return mtp.prune(
        model,
        mode=[
            (
                "mcore_minitron",
                mtp.mcore_minitron.get_mcore_minitron_config(
                    hidden_size_divisor=channel_divisor,
                    ffn_hidden_size_divisor=channel_divisor,
                    mamba_head_dim_divisor=4,
                    num_moe_experts_divisor=1,
                    num_layers_divisor=1,
                ),
            )
        ],
        constraints=constraints,
        dummy_input=None,  # Not used
        config=config,
    )
