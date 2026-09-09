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

"""Public-API dataloader reconstruction for deterministic mid-epoch resume."""

from __future__ import annotations

from typing import Any

from torchdata.stateful_dataloader import StatefulDataLoader

__all__ = ["rebuild_stateful_dataloader"]


def rebuild_stateful_dataloader(
    dataloader: StatefulDataLoader,
    sampler: Any,
    step_scheduler: Any,
    global_step: int,
) -> StatefulDataLoader:
    """Return a fresh loader positioned at the deterministic cursor for ``global_step``.

    Fresh starts and schedulers without a positive epoch length return the original loader.
    Otherwise the sampler is positioned through its public ``load_state_dict`` method, and every
    public ``StatefulDataLoader`` construction option is preserved.
    """
    epoch_len = int(getattr(step_scheduler, "epoch_len", 0) or 0)
    grad_acc_steps = int(getattr(step_scheduler, "grad_acc_steps", 1) or 1)
    if global_step <= 0 or epoch_len <= 0 or sampler is None:
        return dataloader

    epoch = global_step // epoch_len
    batches_yielded = (global_step % epoch_len) * grad_acc_steps
    sampler.load_state_dict({"epoch": epoch, "batches_yielded": batches_yielded})

    new_loader = StatefulDataLoader(
        dataloader.dataset,
        batch_sampler=sampler,
        collate_fn=dataloader.collate_fn,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers,
        pin_memory_device=dataloader.pin_memory_device,
        in_order=dataloader.in_order,
        snapshot_every_n_steps=dataloader.snapshot_every_n_steps,
    )
    step_scheduler.epoch = epoch
    step_scheduler.dataloader = new_loader
    return new_loader
