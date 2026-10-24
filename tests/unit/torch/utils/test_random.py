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

import random as pyrandom

import pytest

from modelopt.torch.utils import random as mrandom


@pytest.fixture(autouse=True)
def _reset_generator():
    for attr in ("generator", "is_manual", "is_synced"):
        if hasattr(mrandom._get_generator, attr):
            delattr(mrandom._get_generator, attr)
    yield
    for attr in ("generator", "is_manual", "is_synced"):
        if hasattr(mrandom._get_generator, attr):
            delattr(mrandom._get_generator, attr)


def test_unseeded_generator_is_not_manual():
    mrandom._get_generator()
    assert mrandom._get_generator.is_manual is False


def test_seeded_generator_is_manual():
    mrandom._get_generator(seed=5)
    assert mrandom._get_generator.is_manual is True


def test_zero_seed_is_respected():
    mrandom._get_generator(seed=0)
    assert mrandom._get_generator.is_manual is True
    assert mrandom.random() == pyrandom.Random(0).random()


def test_unseeded_generator_resyncs_when_distributed(monkeypatch):
    mrandom._get_generator()
    first = mrandom._get_generator.generator

    monkeypatch.setattr(mrandom.dist, "size", lambda: 2)
    monkeypatch.setattr(mrandom.dist, "broadcast", lambda obj, *a, **k: 1234)
    assert mrandom._get_generator() is not first
    assert mrandom._get_generator.is_synced is True
    assert mrandom.random() == pyrandom.Random(1234).random()
