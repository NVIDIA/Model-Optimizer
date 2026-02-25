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

"""torch.distribute utils."""

import json
from contextlib import contextmanager
from io import BytesIO
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import torch

from modelopt.torch.utils import distributed as dist

from .model_config_utils import (
    model_config_from_dict,
    model_config_to_dict,
    restore_model_config,
    split_config_and_weights,
)


class NFSWorkspace:
    """A shared workspace implementation using Network File Storage (NFS).

    NOTE: all read/write/modifition to the NFS dir do not involve any collective
          communication nor barrier. It is users' responsibility to synchronize
          all ranks (local and remove processes).

    This implementation uses `torch.save` and `torch.load` for serialization.

    Args:
        workspace_path: the path to the NFS directory for postprocess cross rank communication.
            If not provided, SharedMemory will be used instead.
    """

    def __init__(self, workspace_path: Path | str | None = None):
        """Create the NFS work dir and clean up existing existing state files."""
        self.path = Path("") if workspace_path is None else Path(workspace_path)
        self._is_initialized = workspace_path is not None
        self.rank = dist.rank()
        if self.is_initialized:
            if self.rank == 0:
                self.path.mkdir(parents=True, exist_ok=True)
            self.state_path = self._get_state_path(self.rank)
            self._clean_up()

    @property
    def is_initialized(self):
        """Whether the workspace is initialized."""
        return self._is_initialized

    def write_configs_and_weights(self, config_json: dict[str, Any], weights: dict[str, Any]):
        """All ranks write the state file to the shared NFS dir.

        Args:
            config_json: model or module config in json
            weights: module weights in torch's state_dict format
        """
        if not self.is_initialized:
            raise ValueError("NFSWorkspace is not initialized!")
        self._clean_up()
        torch.save({"config": config_json, "weight": weights}, self.state_path)

    def read_configs_and_weights_from_rank(
        self, target_rank: int
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """All ranks read the target_rank state file.

        Args:
            target_rank: the target rank

        Returns:
            the model/module config and the weights
        """
        if not self.is_initialized:
            raise ValueError("NFSWorkspace is not initialized!")
        state_path = self._get_state_path(target_rank)
        if state_path.exists():
            # Security NOTE: weights_only=False is used here on ModelOpt-generated ckpt, not on untrusted user input
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            return state["config"], state["weight"]
        else:
            return None, None

    def _get_state_path(self, target_rank: int) -> Path:
        """Return the state file name of a particular rank.

        Args:
            target_rank: the target rank

        Returns:
            the state file path of the target rank
        """
        if not self.is_initialized:
            raise ValueError("NFSWorkspace is not initialized!")
        return self.path.joinpath(f"rank_{target_rank}_state.pth")

    def _clean_up(self):
        """Remove existing state files."""
        if not self.is_initialized:
            raise ValueError("NFSWorkspace is not initialized!")
        self.state_path.unlink(missing_ok=True)


@contextmanager
def get_tensors_parallel(tensor: torch.Tensor, ranks: list[int], group=None):
    """Gathers the tensors across distributed processes using shm.

    Args:
        tensor: the tensor that each rank want to pass to the first rank.
            The tensors across the ranks need to have the same size.
        ranks: the list of the ranks
        group: the barrier sync group.

    Yields:
        the first rank in the ranks has the full access of the tensors across all the ranks.
        the other ranks returns an empty list

    The shm will be destroyed after consumption.
    """
    assert tensor is not None
    assert len(ranks) > 1
    local_rank = dist.rank()
    shm_writer = None
    shm_readers = []
    tensor = tensor.cpu()

    is_merged_rank = local_rank == ranks[0]
    # Create shm and copy the tensor to the shm if not the merged rank.
    # Assume each tensor need up to 2KB additional space for metadata.
    if not is_merged_rank:
        shm_writer = SharedMemory(name=f"rank_{local_rank}", create=True, size=tensor.nbytes + 2048)
        torch.save(tensor, shm_writer._mmap)  # type: ignore[attr-defined]
    # All ranks wait for this to complete.
    dist.barrier(group)

    tensors = []
    # The merged rank gather the tensor from the other ranks (including itself).
    if is_merged_rank:
        for rank in ranks:
            if rank == ranks[0]:
                tensors.append(tensor)
            else:
                shm = SharedMemory(name=f"rank_{rank}", create=False)
                shared_tensor = torch.load(BytesIO(shm.buf))
                tensors.append(shared_tensor)
                shm_readers.append(shm)
    try:
        # Send the tensor list to the consumer.
        # The merged rank will get a valid tensor while the other ranks an empty tensor.
        yield tensors
    finally:
        # Reader closes the shms.
        if shm_readers:
            for shm in shm_readers:
                shm.close()

        # All ranks wait for the reader to close the shms.
        dist.barrier(group)

        # Writer frees the shm resource.
        if shm_writer is not None:
            shm_writer.close()
            shm_writer.unlink()


@contextmanager
def get_configs_parallel(config, ranks: list[int], group, workspace_path: Path | str | None = None):
    """Gathers the layer config across distributed processes using shm or NFS.

    Args:
        config: the config (nullable) that each rank want to pass to the first rank.
        ranks: the list of the ranks
        group: the barrier sync group.
        workspace_path: the path to the NFS directory for postprocess cross rank communication.

    Yields:
        the first rank in the ranks has the full access of the configs across all the ranks.
        the other ranks returns an empty list

    When workspace_path is provided, an NFSWorkspace object is created to perform communication
    across ranks. Otherwise, `SharedMemory` is used for local multi-process communication.
    The shm will be destroyed after consumption.
    """
    assert len(ranks) > 1
    local_rank = dist.rank()
    shm_writer = None
    shm_readers = []
    nfs_workspace = NFSWorkspace(workspace_path)

    is_merged_rank = local_rank == ranks[0]

    def _get_weights_nbytes(weights_dict: dict[str, torch.Tensor]):
        total_nbytes = 0
        for k, v in weights_dict.items():
            # Assume each tensor need up to 2KB additional space for metadata.
            # In reality this should be much smaller.
            total_nbytes = total_nbytes + len(k) + v.nbytes + 2048

        return total_nbytes

    # Create shm and copy the serialized config to the shm if not the merged rank.
    if not is_merged_rank:
        if config is not None:
            config_dict = model_config_to_dict(config)
            # Add additional config type name to the dict so we can later pick the right config type.
            config_dict["__name__"] = str(type(config).__name__)
            weights = {}
            split_config_and_weights(config_dict, weights)

            config_json = json.dumps(config_dict)

            if nfs_workspace.is_initialized:
                # All ranks except for the master merge rank write to the NFS dir.
                nfs_workspace.write_configs_and_weights(config_dict, weights)
            else:
                # SHM data structure: 8B json size, serialized json bytes and the weights dict.
                shm_writer = SharedMemory(
                    name=f"rank_{local_rank}_config",
                    create=True,
                    size=(8 + len(config_json) + _get_weights_nbytes(weights)),
                )

                # Write json length to the shm
                shm_writer.buf[:8] = len(config_json).to_bytes(8, "little")

                # Write json to the shm
                shm_writer.buf[8 : len(config_json) + 8] = config_json.encode()

                # Write np tensors to the shm.
                shm_writer._mmap.seek(len(config_json) + 8)  # type: ignore[attr-defined]
                torch.save(weights, shm_writer._mmap)  # type: ignore[attr-defined]
        else:
            # If the config is None, we just store the empty 0.
            shm_writer = SharedMemory(
                name=f"rank_{local_rank}_config",
                create=True,
                size=8,
            )

            shm_writer.buf[:8] = (0).to_bytes(8, "little")

    # All ranks wait for this to complete.
    dist.barrier(group)

    configs = []
    if is_merged_rank:
        for rank in ranks:
            if rank == ranks[0]:
                configs.append(config)
            elif nfs_workspace.is_initialized:
                # The master merge rank read other configs from the NFS dir.
                config_dict, weights = nfs_workspace.read_configs_and_weights_from_rank(rank)
                if config_dict is not None:
                    restore_model_config(config_dict, weights)
                    config = model_config_from_dict(config_dict)
                    configs.append(config)
            else:
                shm = SharedMemory(name=f"rank_{rank}_config", create=False)
                len_json = int.from_bytes(shm.buf[:8], "little")

                if len_json != 0:
                    config_dict = json.loads(shm.buf[8 : 8 + len_json].tobytes().decode())
                    weights = torch.load(BytesIO(shm.buf[8 + len_json :]))
                    restore_model_config(config_dict, weights)
                    config = model_config_from_dict(config_dict)

                    configs.append(config)
                    shm_readers.append(shm)
    try:
        # Send the config list to the consumer.
        # The merged rank will get a valid config list while the other ranks an empty list.
        yield configs
    finally:
        # Reader closes the shms.
        if shm_readers:
            for shm in shm_readers:
                shm.close()

        # All ranks wait for the reader to close the shms.
        dist.barrier(group)

        # Writer frees the shm resource.
        if shm_writer is not None:
            shm_writer.close()
            shm_writer.unlink()
