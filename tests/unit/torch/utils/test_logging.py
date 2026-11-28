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

from io import StringIO

import pytest
import tqdm

from modelopt.torch.utils.logging import no_stdout


@pytest.mark.parametrize("disable_arg", ["omitted", "positional", "keyword"])
def test_no_stdout_disables_tqdm_with_positional_arguments(disable_arg):
    output = StringIO()
    # Supply every positional argument immediately preceding disable.
    args = (range(3), "progress", 3, True, output, 80, 0.1, 10.0, 2, True)
    kwargs = {}
    if disable_arg == "positional":
        args += (False, "items")
    elif disable_arg == "keyword":
        kwargs["disable"] = False

    with no_stdout(), tqdm.tqdm(*args, **kwargs) as bar:
        assert bar.disable
        assert list(bar) == [0, 1, 2]
    assert output.getvalue() == ""


def test_no_stdout_nested_contexts():
    output = StringIO()
    with no_stdout(), no_stdout(), tqdm.tqdm(range(3), file=output) as bar:
        assert list(bar) == [0, 1, 2]
    assert output.getvalue() == ""
