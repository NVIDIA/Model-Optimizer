# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref

import torch
from torch import nn

from modelopt.torch.puzzletron.plugins.automodel.hooks.moe import _grouped_cache_matches


def test_grouped_route_cache_requires_the_live_source_tensor() -> None:
    module = nn.Identity()
    source = torch.ones(2, 3)
    cache = {"module": module, "input_ref": weakref.ref(source)}

    assert _grouped_cache_matches(cache, module, source)
    assert not _grouped_cache_matches(cache, module, source.clone())

    source_ref = weakref.ref(source)
    del source
    gc.collect()
    assert source_ref() is None
    assert not _grouped_cache_matches(cache, module, torch.ones(2, 3))
