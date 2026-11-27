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

"""Skip an unselected task head while preserving its module and outer hooks."""

from contextlib import contextmanager

import torch


@contextmanager
def bypass_head_context(model, units, head, *, head_paths=None):
    """Avoid allocating unused vocabulary logits; restore any existing forward override."""
    task = getattr(model, "thinker", model)
    getter = getattr(task, "get_input_embeddings", None)
    try:
        input_embeddings = getter() if callable(getter) else None
    except NotImplementedError:
        input_embeddings = None
    # Some tasks expose the input embedding module itself as their output head.
    # Replacing its forward would corrupt tokens before the selected boundaries.
    if head is None or head is input_embeddings:
        yield
        return
    paths = (
        [path for path, child in model.named_modules() if child is head]
        if head_paths is None
        else head_paths
    )
    selected = any(
        not unit.teacher_path
        or path == unit.teacher_path
        or path.startswith(unit.teacher_path + ".")
        for unit in units
        for path in paths
    )
    if selected or not paths:
        yield
        return
    had_override, previous = "forward" in head.__dict__, head.__dict__.get("forward")

    def unused_logits(*args, **kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states", kwargs.get("input"))
        if not isinstance(hidden, torch.Tensor):
            raise ValueError("A Bypass output head requires tensor hidden states")
        return hidden.new_zeros((*hidden.shape[:-1], 1))

    head.forward = unused_logits
    try:
        yield
    finally:
        if had_override:
            head.forward = previous
        else:
            delattr(head, "forward")
