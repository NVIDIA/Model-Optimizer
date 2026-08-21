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

"""Random number generator with a deterministic, synchronized seed for sampling."""

import random as _random
from collections.abc import Callable, MutableSequence, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

import torch

from . import distributed as dist
from .list import list_closest_to_median

T = TypeVar("T")  # pylint: disable=invalid-name
FSample = Callable[[Sequence[T]], T]


def _get_generator(seed: int | None = None) -> _random.Random:
    # delete existing generator if manual seed is provided OR if we are now in distributed setting
    # but weren't previously and so generator is not yet synced across GPUs (manual seed still takes
    # precedence though).
    if hasattr(_get_generator, "generator") and (
        seed is not None
        or (
            dist.size() > 1
            and not getattr(_get_generator, "is_synced", False)
            and not getattr(_get_generator, "is_manual", False)
        )
    ):
        delattr(_get_generator, "generator")
    if not hasattr(_get_generator, "generator"):
        # synchronizing random seed and initialize generator
        seed = dist.broadcast(seed or _random.getrandbits(64))
        _get_generator.generator = _random.Random(seed)  # type: ignore[attr-defined]
        _get_generator.is_manual = seed is not None  # type: ignore[attr-defined]
        _get_generator.is_synced = dist.size() > 1  # type: ignore[attr-defined]
    return _get_generator.generator  # type: ignore[attr-defined]


def _set_deterministic_seed(seed: int = 1):
    """Set the default seed for the random number generator  and synchronize it across GPUs."""
    _get_generator(seed=seed)


def random() -> float:
    """Generate a random number from [0, 1) with a deterministic seed."""
    return _get_generator().random()


def choice(seq: Sequence[T]) -> T:
    """Return a random element from the sequence using a synchronized seed.

    Args:
        seq: Sequence to sample from.

    Returns:
        Random element from the sequence.

    This function is synchronized across all GPUs and can be used to sample a random subnet from a
    search space via :meth:`mtn.sample()<modelopt.torch.nas.utils.sample>` such that the resulting
    subnet/configuration is the same across all GPUs.

    Example:
    .. code-block:: python

        from modelopt.torch.nas import random
        import modelopt.torch.nas as mtn

        # Sample a random subnet of a converted model
        config = mtn.sample(model, random.choice)

        # random.choice is also the default option for sample
        config = mtn.sample(model)
    """
    return _get_generator().choice(seq)


def sample(*args, **kwargs):
    """Sample elements from a given population with a deterministic seed."""
    return _get_generator().sample(*args, **kwargs)


def centroid(seq: Sequence[T]) -> T:
    """Reduce each element of the seq via ``torch.prod()`` and then return seq element closest.

    Args:
        seq: Sequence to determine centroid.

    Returns:
        Centroid of the sequence.

    This function can be used to sample the centroid subnet of an search space via
    :meth:`mtn.sample()<modelopt.torch.nas.utils.sample>`. The centroid subnet aims to cheaply
    approximate the median of the search space defined by the ``model``.

    Example:
    .. code-block:: python

        from modelopt.torch.nas import random
        import modelopt.torch.nas as mtn

        # Sample the centroid subnet of a converted model
        config = mtn.sample(model, random.centroid)
    """
    seq_reduced = [torch.prod(torch.tensor(x)).item() for x in seq]
    return seq[seq_reduced.index(list_closest_to_median(seq_reduced))]


def original(seq: Sequence[T]) -> None:
    """Return an indicator (None) that can be recognized internally to sample the original choice.

    Args:
        seq: Sequence of choices from which we want to "choose" original choice.

    Returns:
        None indicating to internally select the original choice from the sequence.

    This function can be used to sample the original subnet of a search space via
    :meth:`mtn.sample()<modelopt.torch.nas.utils.sample>`. The original subnet corresponds to the
    model architecture before the conversion process.

    Example:
    .. code-block:: python

        from modelopt.torch.nas import random
        import modelopt.torch.nas as mtn

        # Sample the original subnet of a converted model
        config = mtn.sample(model, random.original)
    """
    return None


def shuffle(seq: MutableSequence[Any]):
    """Shuffle the sequence in-place with a deterministic seed."""
    return _get_generator().shuffle(seq)


@contextmanager
def _deterministic_seed():
    """Sets a deterministic seed within the context.

    Resets the random state to prior upon exit.
    """
    old_random_generator = _get_generator()
    _set_deterministic_seed(1024)
    yield
    delattr(_get_generator, "generator")
    setattr(_get_generator, "generator", old_random_generator)
