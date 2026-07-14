"""Slurm allocation policy for the two-node cross-model campaign."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import ParallelTopology

__all__ = ["CampaignAllocation", "allocation_for_topology"]


@dataclass(frozen=True)
class CampaignAllocation:
    nodes: int
    gpus_per_node: int
    nproc_per_node: int
    exclusive: bool


def allocation_for_topology(topology: ParallelTopology) -> CampaignAllocation:
    topology.validate()
    world_size = topology.world_size
    if world_size > 16:
        raise ValueError(f"campaign topology needs {world_size} ranks but only 16 GPUs are allowed")
    nodes = 1 if world_size <= 8 else 2
    if world_size % nodes:
        raise ValueError(
            f"world size {world_size} cannot be spread evenly over {nodes} campaign nodes"
        )
    gpus_per_node = world_size // nodes
    return CampaignAllocation(
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        nproc_per_node=gpus_per_node,
        exclusive=gpus_per_node == 8,
    )
