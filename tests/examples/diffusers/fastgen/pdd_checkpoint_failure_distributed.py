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

"""Two-rank proof that rank-0 checkpoint failures propagate instead of deadlocking."""

from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))

import pdd.checkpoint as pdd_checkpoint_module
from pdd.checkpoint import PDDCheckpointManager


class _State:
    def state_dict(self):
        return {"value": 1}


class _Sampler(_State):
    def state_dict(self):
        return {
            "epoch": 0,
            "committed_batches": 1,
            "sample_slots_consumed": 1,
            "plan_sha256": "0" * 64,
            "next_sample_ids": ["next"],
        }


class _Trainer(_State):
    def __init__(self, completed_steps: int = 1) -> None:
        self.completed_steps = completed_steps


class _StepScheduler:
    def __init__(self, trainer: _Trainer) -> None:
        self.trainer = trainer

    def state_dict(self):
        return {"step": self.trainer.completed_steps, "epoch": 0}


class _Checkpointer:
    def __init__(self, rank: int, *, fail_sidecar: bool = False) -> None:
        self.config = SimpleNamespace(is_async=False)
        self.rank = rank
        self.fail_sidecar = fail_sidecar

    def save_model(self, model, path: str) -> None:
        del model
        if self.rank == 0:
            root = pathlib.Path(path) / "model"
            root.mkdir(parents=True)
            (root / ".metadata").write_bytes(b"metadata")
            (root / "__0_0.distcp").write_bytes(b"model")

    def save_optimizer(self, optimizer, model, path: str, scheduler) -> None:
        del optimizer, model, scheduler
        if self.rank == 0:
            root = pathlib.Path(path) / "optim"
            root.mkdir(parents=True)
            (root / ".metadata").write_bytes(b"metadata")
            (root / "__0_0.distcp").write_bytes(b"optim")

    def save_on_dp_ranks(self, state, state_name: str, path: str) -> None:
        if self.fail_sidecar and self.rank == 1 and state_name == "sampler":
            raise OSError("injected rank-1 sidecar failure")
        root = pathlib.Path(path) / state_name
        root.mkdir(parents=True, exist_ok=True)
        torch.save(state.state_dict(), root / f"{state_name}_dp_rank_{self.rank}.pt")


class _FailingManager(PDDCheckpointManager):
    def __init__(self, *, failure_stage: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.failure_stage = failure_stage

    def _prepare_staging(self, final: pathlib.Path) -> str:
        if self.failure_stage == "prepare":
            raise OSError("injected preparation failure")
        return super()._prepare_staging(final)

    def _publish_staging(self, **kwargs) -> None:
        if self.failure_stage == "publish":
            raise OSError("injected publication failure")
        if self.failure_stage != "latest":
            super()._publish_staging(**kwargs)
            return
        original = pdd_checkpoint_module._atomic_text

        def fail_latest(path: pathlib.Path, text: str) -> None:
            if path.name == "LATEST":
                raise OSError("injected LATEST update failure")
            original(path, text)

        pdd_checkpoint_module._atomic_text = fail_latest
        try:
            super()._publish_staging(**kwargs)
        finally:
            pdd_checkpoint_module._atomic_text = original


def _run_failure(root: pathlib.Path, stage: str) -> None:
    rank = dist.get_rank()
    trainer = _Trainer()
    if stage == "latest":
        initial = _FailingManager(
            failure_stage="none",
            root=root / stage,
            checkpointer=_Checkpointer(rank),
            model=object(),
            optimizer=SimpleNamespace(param_groups=[{"lr": 2.0e-5}]),
            scheduler=object(),
            step_scheduler=_StepScheduler(trainer),
            trainer=trainer,
            sampler=_Sampler(),
            rng=_State(),
            identity={"schema_version": 4, "topology": {"world_size": 2}},
        )
        initial.save()
        trainer.completed_steps = 2
    manager = _FailingManager(
        failure_stage=stage,
        root=root / stage,
        checkpointer=_Checkpointer(rank, fail_sidecar=stage == "sidecar"),
        model=object(),
        optimizer=SimpleNamespace(param_groups=[{"lr": 2.0e-5}]),
        scheduler=object(),
        step_scheduler=_StepScheduler(trainer),
        trainer=trainer,
        sampler=_Sampler(),
        rng=_State(),
        identity={"schema_version": 4, "topology": {"world_size": 2}},
    )
    message = None
    try:
        manager.save()
    except RuntimeError as error:
        message = str(error)
    messages: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(messages, message)
    if stage == "sidecar":
        assert all(
            item is not None and "checkpoint sidecar save failed" in item for item in messages
        )
    else:
        expected = "preparation" if stage == "prepare" else "publication"
        assert all(
            item is not None and f"rank-0 checkpoint {expected}" in item for item in messages
        )
    if stage == "latest":
        assert (root / stage / "LATEST").read_text().strip() == "step_00000001"
        assert manager.resolve("LATEST").name == "step_00000002"


def main() -> None:
    dist.init_process_group("gloo")
    root_payload = [
        tempfile.mkdtemp(prefix="modelopt-pdd-rank0-failure-") if dist.get_rank() == 0 else None
    ]
    dist.broadcast_object_list(root_payload, src=0)
    root = pathlib.Path(root_payload[0])
    try:
        _run_failure(root, "prepare")
        dist.barrier()
        _run_failure(root, "publish")
        dist.barrier()
        _run_failure(root, "sidecar")
        dist.barrier()
        _run_failure(root, "latest")
    finally:
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(root)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
