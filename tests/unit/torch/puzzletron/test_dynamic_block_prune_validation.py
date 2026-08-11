# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Validation tests for dynamic single-block pruning metadata."""

import pytest

from modelopt.torch.puzzletron.pruning.dynamic_block_prune import build_block_prune_specs


def test_build_specs_rejects_missing_ffn_module_name():
    with pytest.raises(ValueError, match="down_proj_name"):
        build_block_prune_specs(
            down_proj_name=None,
            o_proj_name=None,
            orig_intermediate=12,
            target_intermediate=8,
            orig_num_q=None,
            orig_num_kv=None,
            target_num_q=None,
            target_num_kv=None,
            head_dim=None,
        )


@pytest.mark.parametrize(
    ("orig_num_q", "o_proj_name", "head_dim", "missing_metadata"),
    [
        (None, "o", 4, "orig_num_q"),
        (24, None, 4, "o_proj_name"),
        (24, "o", None, "head_dim"),
    ],
)
def test_build_specs_rejects_missing_attention_metadata(
    orig_num_q, o_proj_name, head_dim, missing_metadata
):
    with pytest.raises(ValueError, match=missing_metadata):
        build_block_prune_specs(
            down_proj_name=None,
            o_proj_name=o_proj_name,
            orig_intermediate=None,
            target_intermediate=None,
            orig_num_q=orig_num_q,
            orig_num_kv=4,
            target_num_q=12,
            target_num_kv=2,
            head_dim=head_dim,
        )
