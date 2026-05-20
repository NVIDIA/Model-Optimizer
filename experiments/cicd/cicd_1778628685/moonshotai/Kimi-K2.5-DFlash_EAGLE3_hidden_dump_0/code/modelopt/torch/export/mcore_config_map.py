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

"""Define the config mapping between Mcore and modelopt."""

# the map is a list of tuples, and each tuple has two elements
# first element: a list of possible fields on Mcore
# second element: a name of the layer config field inside modelopt
MCORE_CONFIG_MAP = [
    (["moe_router_topk"], "moe_top_k"),
    (["num_moe_experts"], "moe_num_experts"),
    (["position_embedding_type"], "position_embedding_type"),
]
