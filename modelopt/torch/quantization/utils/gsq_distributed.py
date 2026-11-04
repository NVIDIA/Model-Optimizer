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

"""WORLD coordination for replicated staged GSQ."""

from contextlib import contextmanager

import torch
import torch.distributed as dist

__all__ = []


class DistributedGSQ:
    def __init__(self):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.active = self.world_size > 1
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if self.active and "nccl" in dist.get_backend()
            else torch.device("cpu")
        )

    @contextmanager
    def check(self):
        # Local-only phases must finish or refuse together before the next collective.
        error = None
        try:
            yield
        except Exception as exc:
            error = exc
        failed = error is not None
        if self.active:
            status = torch.tensor(int(failed), device=self.device)
            dist.all_reduce(status, op=dist.ReduceOp.MAX)
            failed = bool(status)
        if error is not None:
            raise error
        if failed:
            raise RuntimeError("GSQ failed on another rank; see that rank's exception")

    def consistent(self, value):
        if self.active:
            values = [None] * self.world_size
            dist.all_gather_object(values, value)
            if any(other != value for other in values):
                raise ValueError("GSQ configuration or model layout differs across ranks")

    def sum(self, value):
        if self.active:
            dist.all_reduce(value)
        return value

    def broadcast(self, value):
        if self.active:
            dist.broadcast(value, 0)

    def local_count(self, global_count):
        return len(range(self.rank, global_count, self.world_size))
