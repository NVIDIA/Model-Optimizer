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

"""Regression tests for seed handling in ``modelopt.torch.utils.random``."""

import random as stdlib_random

import pytest

from modelopt.torch.utils import random as mtu_random
from modelopt.torch.utils.random import _get_generator


def _clear_generator_cache():
    for attr in ("generator", "is_manual", "is_synced"):
        if hasattr(_get_generator, attr):
            delattr(_get_generator, attr)


@pytest.fixture(autouse=True)
def _fresh_generator_cache():
    """Clear the cached generator state on ``_get_generator`` around each test."""
    _clear_generator_cache()
    yield
    _clear_generator_cache()


def test_explicit_seed_zero_is_honored():
    """An explicit seed of 0 must not be silently replaced by random bits."""
    _get_generator(seed=0)
    stream_0a = [mtu_random.random() for _ in range(5)]

    _get_generator(seed=0)
    stream_0b = [mtu_random.random() for _ in range(5)]

    _get_generator(seed=1)
    stream_1 = [mtu_random.random() for _ in range(5)]

    reference = stdlib_random.Random(0)
    assert stream_0a == stream_0b
    assert stream_0a == [reference.random() for _ in range(5)]
    assert stream_0a != stream_1


def test_is_manual_flag_reflects_original_argument():
    """``is_manual`` must be False for auto-seeding and True for a manual seed."""
    _get_generator()
    assert _get_generator.is_manual is False

    _get_generator(seed=42)
    assert _get_generator.is_manual is True


def test_deterministic_seed_restores_generator_on_exception():
    """``_deterministic_seed`` must restore the outer generator even if the body raises."""
    _get_generator(seed=7)
    reference = stdlib_random.Random(7)
    assert [mtu_random.random() for _ in range(3)] == [reference.random() for _ in range(3)]

    with pytest.raises(RuntimeError, match="boom"), mtu_random._deterministic_seed():
        mtu_random.random()
        raise RuntimeError("boom")

    # The outer stream must continue exactly where it left off.
    assert [mtu_random.random() for _ in range(3)] == [reference.random() for _ in range(3)]
