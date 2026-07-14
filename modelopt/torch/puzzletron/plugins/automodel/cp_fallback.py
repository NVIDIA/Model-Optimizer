"""Exact context-parallel fallback for recurrent token mixers.

FLA's native CP GatedDeltaNet path can fail to return from its first forward on
some supported GPU/software combinations.  Replace-block scoring still needs
the requested CP input/output layout, so this fallback gathers the dense
sequence inside each GDN layer, executes the ordinary exact GDN kernel, and
selects this CP rank's original load-balanced tokens from the result.

The fallback deliberately changes only the implementation of the recurrent
token mixer.  TP, PP, FSDP, sequence parallelism, the dynamic block slicing,
and the LM-head metrics remain active.
"""

from __future__ import annotations

from types import MethodType

import torch
import torch.distributed as torch_dist

__all__ = ["install_exact_replicated_gdn_cp"]


def _dtensor_type():
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()
    return DTensor


def _rewrap_like(local: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    DTensor = _dtensor_type()
    if not isinstance(reference, DTensor):
        return local
    from torch.distributed.tensor import distribute_tensor

    return distribute_tensor(
        local,
        device_mesh=reference.device_mesh,
        placements=reference.placements,
    )


def _cp_all_gather_concat(tensor: torch.Tensor, group, *, dim: int) -> torch.Tensor:
    """All-gather local payloads without dispatching c10d on a DTensor."""
    DTensor = _dtensor_type()

    is_dtensor = isinstance(tensor, DTensor)
    local = tensor.to_local() if is_dtensor else tensor
    gathered = [torch.empty_like(local) for _ in range(torch_dist.get_world_size(group))]
    torch_dist.all_gather(gathered, local.contiguous(), group=group)
    return torch.cat(gathered, dim=dim)


def _exact_replicated_forward_with_cp(
    self,
    hidden_states: torch.Tensor,
    *,
    position_ids: torch.Tensor | None,
    seq_index: torch.Tensor | None,
) -> torch.Tensor:
    """Run an exact full-sequence GDN and restore this rank's CP token layout."""
    cp_group = self._cp_mesh.get_group()
    DTensor = _dtensor_type()
    hidden_local = hidden_states.to_local() if isinstance(hidden_states, DTensor) else hidden_states
    local_positions = self._extract_local_positions(
        position_ids,
        seq_index,
        hidden_local.shape[1],
    )
    if local_positions is None:
        local_positions = self._extract_local_positions(
            getattr(self, "_cached_position_ids", None),
            None,
            hidden_local.shape[1],
        )
    self._cached_position_ids = None
    if local_positions is None:
        raise RuntimeError(
            f"CP GDN layer {self.layer_idx} requires local position metadata"
        )

    if isinstance(hidden_states, DTensor):
        # Reconstruct this CP rank's complete sequence from the TP/SP shards.
        hidden_cp_local = hidden_states.full_tensor()
        tp_group = hidden_states.device_mesh.get_group()
        positions_cp_local = _cp_all_gather_concat(
            local_positions,
            tp_group,
            dim=0,
        )
    else:
        hidden_cp_local = hidden_local
        positions_cp_local = local_positions

    cp_order_hidden = _cp_all_gather_concat(
        hidden_cp_local,
        cp_group,
        dim=1,
    )
    cp_order_positions = _cp_all_gather_concat(
        positions_cp_local,
        cp_group,
        dim=0,
    )
    sort_order = torch.argsort(cp_order_positions)
    sorted_positions = cp_order_positions.index_select(0, sort_order)
    expected_positions = torch.arange(
        sorted_positions.numel(),
        device=sorted_positions.device,
        dtype=sorted_positions.dtype,
    )
    if not torch.equal(sorted_positions, expected_positions):
        raise RuntimeError(
            f"CP GDN layer {self.layer_idx} expected dense positions 0..S-1"
        )

    full_hidden_local = cp_order_hidden.index_select(1, sort_order)
    # GDN is not part of NeMo's base TP plan: its FSDP-managed weights are
    # ordinary local tensors.  Keep computation local and restore the decoder's
    # DTensor layout only after the token mixer returns.
    full_output = self._forward_no_cp(full_hidden_local)

    restore_indices = torch.searchsorted(sorted_positions, positions_cp_local)
    restored_positions = sorted_positions.index_select(0, restore_indices)
    if not torch.equal(restored_positions, positions_cp_local):
        raise RuntimeError(
            f"CP GDN layer {self.layer_idx} could not restore the local token layout"
        )
    local_output = full_output.index_select(1, restore_indices)
    return _rewrap_like(local_output, hidden_states)


def install_exact_replicated_gdn_cp(model_parts) -> int:
    """Install the exact fallback on CP-aware recurrent mixers in ``model_parts``."""
    count = 0
    for part in model_parts or []:
        for module in part.modules():
            cp_mesh = getattr(module, "_cp_mesh", None)
            if cp_mesh is None or cp_mesh.size() <= 1:
                continue
            required = (
                "_forward_with_cp",
                "_forward_no_cp",
                "_extract_local_positions",
                "_all_gather_concat",
            )
            if not all(hasattr(module, name) for name in required):
                continue
            module._forward_with_cp = MethodType(
                _exact_replicated_forward_with_cp,
                module,
            )
            count += 1
    return count
