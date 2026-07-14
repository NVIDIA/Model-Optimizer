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

"""Bounded teacher-target cache for AutoModel replace-1-block scoring.

The replace-1-block scorer runs the teacher once and reuses its outputs as the
comparison target for every candidate solution (the user-selected "cache teacher
targets" strategy). To make this feasible at large vocab, the cache deliberately
stores only what the metric needs and never the full ``[tokens, vocab]`` teacher
logits (the known OOM source):

* per batch: the teacher's final hidden state ``[b, t, d]`` — both the cosine/MSE
  target *and* the input the ``flash_kd`` path streams through the teacher LM head
  to reconstruct teacher logits chunk-by-chunk for KL/CE; and
* once: the teacher LM-head weight ``[vocab, d]``.

Tensors are held on CPU (``d``-sized, not ``vocab``-sized) so they survive the
teacher's GPU teardown before the candidate loop. Sized to the (small)
solution-validation set, not the activation pass.
"""

import torch

__all__ = ["TeacherTargetCache"]


class TeacherTargetCache:
    """Holds per-batch teacher hidden states + the teacher LM-head weight on CPU."""

    def __init__(self):
        self._hidden_per_batch: list[torch.Tensor] = []
        self.lm_head_weight: torch.Tensor | None = None
        self._sealed = False

    def set_lm_head_weight(self, weight: torch.Tensor) -> None:
        """Store the teacher LM-head weight ``[vocab, d]`` once (moved to CPU)."""
        self.lm_head_weight = weight.detach().to("cpu", dtype=torch.float32)

    def append_hidden(self, hidden_states: torch.Tensor) -> None:
        """Append one batch of final hidden states ``[b, t, d]`` (moved to CPU)."""
        assert not self._sealed, "cache already sealed; cannot append after extraction"
        self._hidden_per_batch.append(hidden_states.detach().to("cpu", dtype=torch.float32))

    def seal(self) -> None:
        """Finalize after teacher extraction; from here the cache is read-only."""
        assert self.lm_head_weight is not None, "teacher LM-head weight was never captured"
        self._sealed = True

    def __len__(self) -> int:
        return len(self._hidden_per_batch)

    def hidden(self, batch_idx: int, device=None, dtype=None) -> torch.Tensor:
        """Return the cached teacher hidden state for ``batch_idx`` (optionally moved)."""
        h = self._hidden_per_batch[batch_idx]
        if device is not None or dtype is not None:
            h = h.to(device=device, dtype=dtype)
        return h

    def lm_head(self, device=None, dtype=None) -> torch.Tensor:
        """Return the teacher LM-head weight (optionally moved)."""
        w = self.lm_head_weight
        if device is not None or dtype is not None:
            w = w.to(device=device, dtype=dtype)
        return w
