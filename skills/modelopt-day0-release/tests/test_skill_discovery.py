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

"""Check canonical evaluation datasets."""

from pathlib import Path

import pytest

_SKILLS = Path(__file__).resolve().parents[3] / "skills"


@pytest.mark.parametrize(
    "dataset", sorted(_SKILLS.glob("modelopt-*/evals/evals.json")), ids=lambda p: p.parents[1].name
)
def test_canonical_dataset_is_regular_file(dataset):
    assert dataset.is_file() and not dataset.is_symlink()
    assert not dataset.parent.is_symlink()
