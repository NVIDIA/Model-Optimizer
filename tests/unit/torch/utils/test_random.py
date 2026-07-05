# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import random as _stdlib_random

import pytest

from modelopt.torch.utils import random as mo_random
from modelopt.torch.utils.random import _deterministic_seed, _get_generator, _set_deterministic_seed


def _clear_generator_state():
    for attr in ("generator", "is_manual", "is_synced"):
        if hasattr(_get_generator, attr):
            delattr(_get_generator, attr)


@pytest.fixture(autouse=True)
def _fresh_generator_state():
    """Ensure each test starts and ends without a cached module-level generator."""
    _clear_generator_state()
    yield
    _clear_generator_state()


def test_manual_seed_matches_stdlib_random():
    """A manual seed must produce exactly the stdlib ``random.Random(seed)`` stream."""
    _set_deterministic_seed(1234)
    expected = _stdlib_random.Random(1234)
    assert [mo_random.random() for _ in range(5)] == [expected.random() for _ in range(5)]


def test_default_seed_is_one():
    _set_deterministic_seed()
    assert mo_random.random() == _stdlib_random.Random(1).random()


def test_reseeding_reproduces_sequence():
    _set_deterministic_seed(77)
    first_run = [mo_random.random() for _ in range(4)]
    _set_deterministic_seed(77)
    assert [mo_random.random() for _ in range(4)] == first_run


def test_different_seeds_produce_different_sequences():
    _set_deterministic_seed(1)
    seq_a = [mo_random.random() for _ in range(4)]
    _set_deterministic_seed(2)
    assert [mo_random.random() for _ in range(4)] != seq_a


def test_independent_of_global_random_state():
    """Reseeding the global ``random`` module must not affect the modelopt generator."""
    _set_deterministic_seed(3)
    expected = _stdlib_random.Random(3)
    assert mo_random.random() == expected.random()
    _stdlib_random.seed(999)  # must have no effect on the modelopt stream
    assert mo_random.random() == expected.random()


def test_random_in_unit_interval_with_auto_seed():
    # no manual seed -> generator is auto-seeded lazily
    for _ in range(100):
        assert 0.0 <= mo_random.random() < 1.0


def test_auto_seeded_generator_is_cached():
    assert _get_generator() is _get_generator()


def test_manual_seed_replaces_existing_generator():
    generator_before = _get_generator()  # auto-seeded
    _set_deterministic_seed(5)
    assert _get_generator() is not generator_before
    assert mo_random.random() == _stdlib_random.Random(5).random()


def test_choice_deterministic():
    seq = list(range(100))
    _set_deterministic_seed(7)
    assert mo_random.choice(seq) == _stdlib_random.Random(7).choice(seq)


def test_sample_deterministic():
    _set_deterministic_seed(11)
    assert mo_random.sample(range(20), 5) == _stdlib_random.Random(11).sample(range(20), 5)


def test_shuffle_deterministic_and_in_place():
    _set_deterministic_seed(42)
    data = list(range(10))
    assert mo_random.shuffle(data) is None  # in-place, like random.shuffle
    expected = list(range(10))
    _stdlib_random.Random(42).shuffle(expected)
    assert data == expected


def test_centroid_exact_values():
    # scalars: median of [1, 2, 3, 4, 5] is 3
    assert mo_random.centroid([1, 2, 3, 4, 5]) == 3
    # tuples are reduced via torch.prod: [6, 1, 16] -> median 6 -> (2, 3)
    assert mo_random.centroid([(2, 3), (1, 1), (4, 4)]) == (2, 3)


def test_centroid_even_length_returns_first_closest():
    # median of [1, 2, 3, 4] is 2.5; 2 and 3 are equally close, first match wins
    assert mo_random.centroid([1, 2, 3, 4]) == 2


def test_original_returns_none():
    assert mo_random.original([1, 2, 3]) is None
    assert mo_random.original([]) is None


def test_deterministic_seed_context_uses_seed_1024_and_restores_state():
    # NOTE: `_deterministic_seed` does not wrap its body in try/finally, so the previous
    # generator is NOT restored if the body raises. Only the happy path is tested here.
    _set_deterministic_seed(99)
    outer = _stdlib_random.Random(99)
    inner = _stdlib_random.Random(1024)

    assert mo_random.random() == outer.random()
    with _deterministic_seed():
        assert mo_random.random() == inner.random()
        assert mo_random.random() == inner.random()
    # the outer stream continues exactly where it left off
    assert mo_random.random() == outer.random()


def test_is_manual_flag_after_seeding():
    # NOTE: `_get_generator` computes `is_manual = seed is not None` *after* reassigning
    # `seed = dist.broadcast(seed or _random.getrandbits(64))`, so `is_manual` is True even
    # for auto-generated seeds. In a distributed run, an auto-seeded generator created before
    # process-group initialization would therefore never be re-synced. This test documents
    # the (correct) manual-seed half of that contract only.
    _set_deterministic_seed(13)
    assert _get_generator.is_manual is True
