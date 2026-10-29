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

"""Publication retains one complete generation while failures leave its pointer intact."""

import pytest
import torch

from modelopt.torch.quantization import _calibration_store
from modelopt.torch.quantization._calibration_store import _CalibrationStore
from modelopt.torch.utils import safe_load


@pytest.mark.usefixtures("skip_on_windows")
def test_checkpoint_generation_retention_and_failure(tmp_path, monkeypatch):
    store = _CalibrationStore(tmp_path)
    checkpoint = tmp_path / "search"
    with store.exclusive(checkpoint):
        with pytest.raises(ValueError), _CalibrationStore(tmp_path).exclusive(checkpoint):
            pytest.fail("A second coordinator acquired the same checkpoint")
        for step in range(3):
            payload = {"step": step, "weight": torch.full((4,), step)}
            store._publish(checkpoint, payload)
            assert list(tmp_path.glob(".search-*")) == [checkpoint.resolve()]
            loaded = safe_load(checkpoint / "rank0.pth")
            assert loaded["step"] == step
            torch.testing.assert_close(loaded["weight"], payload["weight"])
        previous = checkpoint.resolve()

        def fail_publish(*args):
            raise OSError("Interrupted publication")

        with monkeypatch.context() as failure:
            failure.setattr(_calibration_store.os, "replace", fail_publish)
            with pytest.raises(ValueError):
                store._publish(checkpoint, {"step": 4})
        assert checkpoint.resolve() == previous
        assert list(tmp_path.glob(".search-*")) == [previous]
        assert safe_load(checkpoint / "rank0.pth")["step"] == 2

    # A symlink outside the managed generation namespace must not be replaced or reclaimed.
    external = tmp_path / "external"
    external.mkdir()
    (external / "rank0.pth").write_bytes(b"retain")
    checkpoint.unlink()
    checkpoint.symlink_to(external, target_is_directory=True)
    with store.exclusive(checkpoint), pytest.raises(ValueError):
        store._publish(checkpoint, {"step": 5})
    assert checkpoint.resolve() == external
    assert (external / "rank0.pth").read_bytes() == b"retain"


@pytest.mark.timeout(45)
def test_distributed_checkpoint_retains_previous_generation(cpu_dist_workers, tmp_path):
    cpu_dist_workers.run(_distributed_publication, tmp_path)


def _distributed_publication(rank, size, directory):
    store = _CalibrationStore(directory)
    checkpoint = directory / "distributed"
    with store.exclusive(checkpoint):
        store._publish(checkpoint, {"rank": rank, "weight": torch.ones(4)})
        previous = checkpoint.resolve()
        save = _calibration_store.safe_save

        def fail_one_writer(payload, path):
            if rank == size - 1:
                raise OSError("One rank failed during serialization")
            save(payload, path)

        with pytest.MonkeyPatch.context() as failure:
            failure.setattr(_calibration_store, "safe_save", fail_one_writer)
            with pytest.raises(ValueError):
                store._publish(checkpoint, {"rank": rank, "weight": torch.zeros(4)})
        torch.distributed.barrier()
        assert checkpoint.resolve() == previous
        assert list(directory.glob(".distributed-*")) == [previous]
        state = safe_load(checkpoint / f"rank{rank}.pth")
        assert state["rank"] == rank
        torch.testing.assert_close(state["weight"], torch.ones(4))
