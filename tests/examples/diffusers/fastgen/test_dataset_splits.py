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

"""Deterministic stable-ID split contract for the shared FastGen cache."""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nemo_automodel")
pytest.importorskip("torchdata")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))

from fastgen_data import (
    build_text_to_image_multiresolution_dataloader,
    make_train_validation_indices,
)


def _snapshot(root: pathlib.Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_split_has_frozen_membership_and_does_not_change_global_rng():
    torch.manual_seed(1234)
    rng_before = torch.random.get_rng_state().clone()

    train, validation = make_train_validation_indices(10, validation_count=3, seed=17)

    assert validation == [0, 7, 9]
    assert train == [1, 2, 3, 4, 5, 6, 8]
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(10))
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert make_train_validation_indices(10, 3, 17) == (train, validation)


@pytest.mark.parametrize(
    ("num_samples", "validation_count", "seed", "error"),
    [
        (True, 1, 0, TypeError),
        (1, 1, 0, ValueError),
        (4, False, 0, TypeError),
        (4, 0, 0, ValueError),
        (4, 4, 0, ValueError),
        (4, 1, True, TypeError),
        (4, 1, -1, ValueError),
    ],
)
def test_split_rejects_invalid_inputs(num_samples, validation_count, seed, error):
    with pytest.raises(error):
        make_train_validation_indices(num_samples, validation_count, seed)


def test_train_and_validation_loaders_are_disjoint_stable_and_read_only(
    make_fastgen_cache, tmp_path
):
    cache = make_fastgen_cache(tmp_path / "cache")
    before = _snapshot(cache)
    train_ids, validation_ids = make_train_validation_indices(6, validation_count=2, seed=17)

    train_loader, train_sampler = build_text_to_image_multiresolution_dataloader(
        cache_dir=str(cache),
        split="train",
        validation_count=2,
        split_seed=17,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )
    validation_loader, validation_sampler = build_text_to_image_multiresolution_dataloader(
        cache_dir=str(cache),
        split="validation",
        validation_count=2,
        split_seed=17,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    assert train_loader.dataset.sample_ids == train_ids
    assert validation_loader.dataset.sample_ids == validation_ids
    assert (
        train_loader.dataset.cache_root == validation_loader.dataset.cache_root == cache.resolve()
    )
    assert train_sampler.shuffle_buckets and train_sampler.shuffle_within_bucket
    assert not validation_sampler.shuffle_buckets
    assert not validation_sampler.shuffle_within_bucket
    assert validation_sampler.drop_last is False
    assert _snapshot(cache) == before
