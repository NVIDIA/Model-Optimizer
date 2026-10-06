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

from modelopt.torch.trace import Symbol
from modelopt.torch.trace.analyzer import NodeProcessor, is_in_slice_range


@pytest.mark.parametrize(
    ("slice_obj", "num", "expected"),
    [
        (slice(0, 5), 0, True),
        (slice(0, 5), 5, False),
        (slice(0, 5), 1, True),
        (slice(5), 0, True),
        (slice(5), 5, False),
        (slice(2, None), 2, True),
        (slice(2, None), 0, False),
    ],
)
def test_is_in_slice_range(slice_obj, num, expected):
    assert is_in_slice_range(slice_obj, num) == expected


@pytest.mark.parametrize(
    ("edims", "edims2", "expected_result"),
    [
        ([{0, 1}], [{0}], {0: 0}),
        ([{0, 1}], [{2}], {}),
        ([{0, 1}, {2, 3}], [{2, 3}], {}),
        ([{0, 1}, {2, 3}], [{2, 3}, {1, 3}], {0: 1, 1: 0}),
        ([{0, 1}, {2, 3}], [{1, 2, 3}, {1, 3}], {0: 0, 1: 1}),
        ([{0, 1}, {2, 3}], [{0, 2}, {1, 3}], {0: 0, 1: 1}),  # all combos allowed but picking one!
        ([{0, 1}, {2, 3}], [{4, 5}, {1, 3}], {}),
        ([{0, 10}, {2, 3}], [{4, 5}, {1, 3}], {}),
        ([{0, 10}, {2, 3}, {5, 6}], [{4, 5}, {1, 3}, {6}], {}),  # 2x2++ works if there is no match!
    ],
)
def test_sym_matching(edims, edims2, expected_result):
    # process hps
    syms = [Symbol(elastic_dims=edim, cl_type=Symbol.CLType.INCOMING) for edim in edims]
    syms2 = [Symbol(elastic_dims=edim, cl_type=Symbol.CLType.INCOMING) for edim in edims2]

    # process expected result
    expected_result = {syms[i]: syms2[j] for i, j in expected_result.items()}

    assert NodeProcessor.build_sym_matching(syms, syms2) == expected_result
