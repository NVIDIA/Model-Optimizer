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

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    return port


def init_process(rank, size, job=None, backend="gloo", port=None):
    """Initialize the distributed environment."""

    os.environ["MASTER_ADDR"] = "localhost"

    port = str(get_free_port()) if port is None else str(port)

    # We need to use a different port for each tests to avoid conflicts
    os.environ["MASTER_PORT"] = port

    dist.init_process_group(backend, rank=rank, world_size=size)
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(rank)
    torch.manual_seed(1234)
    if job is not None:
        job(rank, size)
    # Explicitly destroy the process group to avoid leaving the process alive
    if dist.is_initialized():
        dist.destroy_process_group()


def spawn_multiprocess_job(size, job, backend="gloo"):
    port = get_free_port()

    processes = []
    mp.set_start_method("spawn", force=True)
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, job, backend, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

        # Ensure that all processes have exited successfully
        assert not p.exitcode


def get_device_counts():
    num_gpus = torch.cuda.device_count()
    return [
        1,
        pytest.param(2, marks=pytest.mark.skipif(num_gpus < 2, reason="need 2 GPUs!")),
    ]


def synchronize_state_dict(model: nn.Module):
    state_dict = model.state_dict()
    for v in state_dict.values():
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
    model.load_state_dict(state_dict)
