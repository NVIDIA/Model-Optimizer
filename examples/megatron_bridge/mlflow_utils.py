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

"""Shared MLflow wiring for the Megatron-Bridge examples, mirroring ``examples/hf_ptq``.

Each script declares what it records as a :class:`~modelopt.torch.utils.mlflow.Tool` beside
its own flags; this module knows none of them, only how a run is opened and closed here.

Every rank parses the same flags, so a typo in the URI fails identically everywhere rather
than on one rank while the others wait in a collective.
"""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager

import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils.mlflow import Tool, tracked_run

# These scripts' own bookkeeping, on top of the tracking settings the library already keeps
# out of the params. Every Tool here passes it as ``non_params``.
NON_PARAMS = frozenset({"checkpoint_exported"})


@contextmanager
def mlflow_run(args: argparse.Namespace, tool: Tool) -> Iterator[None]:
    """Track this invocation for the duration of the block."""
    with tracked_run(
        args,
        tool,
        is_main=dist.is_master(),
        exported=lambda: args.checkpoint_exported,
        world_size=dist.size(),
    ):
        yield
