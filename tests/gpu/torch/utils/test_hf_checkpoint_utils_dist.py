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

"""Multi-rank tests for ``ensure_local_checkpoint`` (each case spawns a 2-process gloo job)."""

import os
from functools import partial

import pytest
import torch.distributed as dist
from _test_utils.torch.distributed.utils import spawn_multiprocess_job

from modelopt.torch.utils.plugins import hf_checkpoint_utils

pytest.importorskip("huggingface_hub")


class _FakeHfApi:
    """``HfApi`` whose ``repo_info`` succeeds, so the Hub counts as reachable."""

    def repo_info(self, repo_id):
        pass


def _ensure_local_checkpoint_job(tmp_path, fail, group_ranks, rank, size):
    utils = hf_checkpoint_utils
    snapshot = tmp_path / "snapshot"

    def fake_snapshot_download(repo_id, **kwargs):
        with (tmp_path / f"downloaded-by-rank{rank}").open("a") as marker:
            marker.write(f"{repo_id}\n")  # one line per call, so a repeat download shows
        if fail:
            raise OSError("hub unreachable")
        snapshot.mkdir(exist_ok=True)
        return str(snapshot)

    utils.snapshot_download = fake_snapshot_download
    utils.HfApi = _FakeHfApi
    os.environ.pop("HF_HUB_OFFLINE", None)  # inherited from tests/unit/conftest.py
    # new_group is collective over every rank, members or not.
    group = None if group_ranks is None else dist.new_group(group_ranks, backend="gloo")
    if group_ranks is not None and rank not in group_ranks:
        return
    if fail:
        with pytest.raises(RuntimeError, match="hub unreachable"):
            utils.ensure_local_checkpoint("org/model", group=group)
    else:
        result = utils.ensure_local_checkpoint("org/model", group=group)
        assert result == ("org/model", str(snapshot))


def _ensure_local_checkpoint_local_path_job(tmp_path, hidden_from_rank1, rank, size):
    utils = hf_checkpoint_utils

    def fail(*args, **kwargs):
        raise AssertionError("a local path must not be downloaded")

    utils.snapshot_download = fail
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    if hidden_from_rank1 and rank == 1:
        # As if the directory sat on rank 0's node-local disk only.
        isdir = os.path.isdir
        utils.os.path.isdir = lambda path: path != str(checkpoint) and isdir(path)
        with pytest.raises(RuntimeError, match="this rank cannot see"):
            utils.ensure_local_checkpoint(str(checkpoint))
    else:
        assert utils.ensure_local_checkpoint(str(checkpoint)) == (None, str(checkpoint))


@pytest.mark.parametrize("hidden_from_rank1", [False, True])
def test_ensure_local_checkpoint_lets_rank_0_resolve_a_local_path(tmp_path, hidden_from_rank1):
    """Ranks that each checked a local path could disagree about it and strand one another in the
    collective; rank 0 decides, and a rank that cannot see its answer raises instead."""
    job = partial(_ensure_local_checkpoint_local_path_job, tmp_path, hidden_from_rank1)
    spawn_multiprocess_job(2, job, backend="gloo")


@pytest.mark.parametrize(
    ("fail", "group_ranks", "downloader"),
    [
        (False, None, "rank0"),
        (True, None, "rank0"),
        (False, [0, 1], "rank0"),
        # Rank 0 of the group, not of the world: global rank 1.
        (False, [1], "rank1"),
    ],
)
def test_ensure_local_checkpoint_downloads_once_per_group(tmp_path, fail, group_ranks, downloader):
    """Rank 0 of the group downloads once and every member gets its result -- including its
    failure, so the waiting ranks raise instead of hanging."""
    job = partial(_ensure_local_checkpoint_job, tmp_path, fail, group_ranks)
    spawn_multiprocess_job(2, job, backend="gloo")
    assert [p.name for p in tmp_path.glob("downloaded-by-*")] == [f"downloaded-by-{downloader}"]
    assert (tmp_path / f"downloaded-by-{downloader}").read_text().splitlines() == ["org/model"]
