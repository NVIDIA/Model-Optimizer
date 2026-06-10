# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Monkey-patch for accelerate's fsdp2_load_full_state_dict buffer handling.

Problem
-------
accelerate's ``fsdp2_load_full_state_dict`` (called during model preparation
when ``cpu_ram_efficient_loading=True``) iterates ``model.state_dict()`` and
unconditionally accesses ``.device_mesh`` on every entry, assuming they are all
DTensors.  After ``fully_shard()``, **parameters** become DTensors but
**persistent buffers** (e.g., rotary-embedding ``inv_freq``) remain plain
``torch.Tensor``.  This crashes with::

    AttributeError: 'Tensor' object has no attribute 'device_mesh'

Additionally, ``cpu_ram_efficient_loading`` causes a dtype divergence: rank 0
loads the model on CPU (inheriting the checkpoint's dtype, e.g. bfloat16) while
other ranks use ``meta`` device (defaulting to float32 for newly-added modules
like the DFlash head).  After ``fully_shard()``, the DTensor dtypes differ
across ranks for these modules.  Since ``dist.broadcast()`` requires matching
dtypes and element sizes on all ranks, broadcasting a bfloat16 tensor (2
bytes/elem) to a float32 receive buffer (4 bytes/elem) causes an NCCL deadlock.

Why we need FSDP2 via accelerate config (not ParallelismConfig)
---------------------------------------------------------------
MiniMax-M2.7's ``trust_remote_code`` model code requires **transformers 4.57.x**.
Transformers' native FSDP2 support via ``ParallelismConfig`` requires
**transformers 5.x**.  So we fall back to configuring FSDP2 through an
``accelerate`` YAML config file (``accelerate_fsdp2.yaml``), which works with
transformers 4.57.x.  We set ``dp_shard_size=1`` to prevent ``main.py`` from
creating a ``ParallelismConfig``, letting the accelerate config handle sharding.

Why we need cpu_ram_efficient_loading
-------------------------------------
MiniMax-M2.7 is a 229B MoE model (~230 GB in FP8).  Each GB200 node has 4 GPUs
and ~800 GB system RAM.  Without ``cpu_ram_efficient_loading``, all 4 ranks per
node load the model to CPU simultaneously (4 × 230 GB ≈ 920 GB), exceeding
system RAM and triggering OOM kills.  With ``cpu_ram_efficient_loading``, only
rank 0 loads the model; other ranks initialize on ``meta`` device.  The weights
are then broadcast via ``fsdp2_load_full_state_dict`` — which is where the bug
hits.

What this patch does
--------------------
1. Before accessing ``.device_mesh``, checks ``isinstance(entry, DTensor)``.
   For non-DTensor entries (persistent buffers), broadcasts the raw tensor from
   rank 0 without calling ``distribute_tensor()``.

2. All ranks iterate ``model.state_dict()`` (post-shard) in the same order so
   broadcast calls match 1-to-1.  Rank 0 looks up the full parameter **by key
   name** from the pre-shard state dict — never by positional ``zip``, because
   ``model.to("meta")`` + ``fully_shard()`` can reorder keys.

3. **Dtype synchronization**: rank 0 broadcasts a dtype code for each entry
   before the main loop.  All ranks then use the same dtype for their broadcast
   tensors.  This fixes the dtype divergence caused by rank 0 loading in
   bfloat16 while other ranks default to float32 for newly-added modules.

Accelerate config constraints (for reference)
----------------------------------------------
``accelerate_fsdp2.yaml`` also requires:

- ``fsdp_use_orig_params: true`` — without this, FSDP flattens all params into
  FlatParameter, losing per-parameter ``requires_grad`` flags.  The DFlash head
  can't train because its gradients mix with frozen base model zeros.
- ``fsdp_transformer_layer_cls_to_wrap: MiniMaxM2DecoderLayer,DFlashModule`` —
  DFlash head params at the model root must be in the wrap policy so they become
  DTensors.  Without this, ``fsdp2_load_full_state_dict`` also crashes.
- ``fsdp_sync_module_states: true`` — accelerate's launch validator requires it
  when ``cpu_ram_efficient_loading`` is enabled, even though FSDP2 ignores it at
  runtime (sets it to None with a warning).

Does NOT affect models on transformers 5.x
-------------------------------------------
This entire workaround exists ONLY because MiniMax-M2.7 requires
transformers 4.57.x.  Models that support transformers 5.x (Qwen, Llama,
Nemotron, etc.) use ``ParallelismConfig`` natively by setting
``dp_shard_size > 1`` in the training args.  That code path handles buffers
correctly and does not go through ``fsdp2_load_full_state_dict`` at all.
No accelerate config file, no ``PATCH_FSDP2_BUFFERS``, no
``OVERRIDE_TRANSFORMERS`` needed.

When to remove
--------------
This patch can be removed when EITHER of these happens:

1. MiniMax updates ``trust_remote_code`` for transformers 5.x, allowing native
   ``ParallelismConfig`` (which handles this correctly).
2. Upstream accelerate fixes ``fsdp2_load_full_state_dict`` to skip non-DTensor
   entries.  Track: https://github.com/huggingface/accelerate

Activation
----------
Set ``PATCH_FSDP2_BUFFERS=1`` in the environment to activate.  Off by default.
Only needed in MiniMax-M2.7 pipeline YAMLs.
"""

import torch


# Dtype encoding for the broadcast dtype-sync step.
_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.bfloat16: 1,
    torch.float16: 2,
    torch.float8_e4m3fn: 3 if hasattr(torch, "float8_e4m3fn") else -1,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items() if v >= 0}


def apply():
    """Patch fsdp2_load_full_state_dict if the buffer bug is present."""
    try:
        import accelerate.utils.fsdp_utils as fsdp_utils
        from torch.distributed.tensor import DTensor

        _orig = fsdp_utils.fsdp2_load_full_state_dict  # noqa: F841

        def _patched(accelerator, model, full_sd, cpu_offload=False):
            import time
            import torch.distributed as dist
            from torch.distributed.tensor import distribute_tensor

            meta_sharded_sd = model.state_dict()
            sharded_sd = {}
            n_total = len(meta_sharded_sd)
            n_dtensor = sum(1 for v in meta_sharded_sd.values() if isinstance(v, DTensor))
            n_buffer = n_total - n_dtensor

            if accelerator.is_main_process:
                print(f"[fsdp2_buffer_patch] State dict: {n_total} entries "
                      f"({n_dtensor} DTensor, {n_buffer} buffer), full_sd: {len(full_sd)}")
            else:
                print(f"[fsdp2_buffer_patch] State dict: {n_total} entries "
                      f"({n_dtensor} DTensor, {n_buffer} buffer)")
            t0 = time.time()

            # --- Step 0: broadcast dtype codes from rank 0 ---
            # cpu_ram_efficient_loading causes rank 0 to load in bfloat16 while
            # other ranks default to float32 for newly-added modules (DFlash).
            # After fully_shard(), DTensor dtypes diverge across ranks.
            # Broadcast rank 0's dtypes so all ranks use the same dtype for
            # each broadcast tensor.
            if accelerator.is_main_process:
                dtype_codes = torch.tensor(
                    [_DTYPE_TO_CODE.get(full_sd[name].dtype, 0)
                     for name in meta_sharded_sd.keys()],
                    dtype=torch.int32, device=accelerator.device,
                )
            else:
                dtype_codes = torch.empty(
                    n_total, dtype=torch.int32, device=accelerator.device,
                )
            dist.broadcast(dtype_codes, src=0, group=dist.group.WORLD)
            broadcast_dtypes = [_CODE_TO_DTYPE[c.item()] for c in dtype_codes]
            del dtype_codes

            # Infer dtype/contiguity for cast — copied from upstream
            def _infer_parameter_dtype(mdl, param_name, empty_param):
                try:
                    old_param = mdl.get_parameter_or_buffer(param_name)
                except AttributeError:
                    base, local = param_name.rsplit(".", 1)
                    old_param = getattr(mdl.get_submodule(base), local)
                is_f8 = hasattr(torch, "float8_e4m3fn") and empty_param.dtype == torch.float8_e4m3fn
                casting_dtype = (
                    old_param.dtype if (empty_param.dtype.is_floating_point and not is_f8) else None
                )
                return old_param is not None and old_param.is_contiguous(), casting_dtype

            def _finish(st, contig, dtype, offload):
                if dtype is not None:
                    st = st.to(dtype=dtype)
                if contig:
                    st = st.contiguous()
                if offload:
                    st = st.to("cpu")
                return st

            # --- Step 1: broadcast all entries ---
            # All ranks iterate meta_sharded_sd in the same order to ensure
            # identical broadcast sequences.  Rank 0 looks up the full parameter
            # by name — never positional zip (model.to("meta") + fully_shard()
            # can reorder keys).  broadcast_dtypes[idx] is used for the tensor
            # dtype on ALL ranks to prevent NCCL deadlocks from dtype divergence.
            for idx, (param_name, sharded_param) in enumerate(meta_sharded_sd.items()):
                is_dtensor = isinstance(sharded_param, DTensor)
                bcast_dtype = broadcast_dtypes[idx]

                if not is_dtensor:
                    # Persistent buffer — broadcast raw, no distribute_tensor
                    if accelerator.is_main_process:
                        t = full_sd[param_name].detach().to(
                            device=accelerator.device, dtype=bcast_dtype)
                    else:
                        t = torch.empty(
                            sharded_param.size(),
                            device=accelerator.device,
                            dtype=bcast_dtype,
                        )
                    dist.broadcast(t, src=0, group=dist.group.WORLD)
                    sharded_sd[param_name] = t
                    continue

                device_mesh = sharded_param.device_mesh
                if accelerator.is_main_process:
                    ft = full_sd[param_name].detach().to(
                        device=device_mesh.device_type, dtype=bcast_dtype)
                    if isinstance(ft, DTensor):
                        ft = ft.to_local()
                else:
                    ft = torch.empty(
                        sharded_param.size(),
                        device=device_mesh.device_type,
                        dtype=bcast_dtype,
                    )
                dist.broadcast(ft, src=0, group=dist.group.WORLD)
                st = distribute_tensor(ft, device_mesh, sharded_param.placements)
                contig, _ = _infer_parameter_dtype(model, param_name, ft)
                # Use bcast_dtype (from rank 0) instead of the model's local
                # param dtype — with cpu_ram_efficient_loading, non-rank-0
                # processes have fp32 meta-device params for DFlash, and
                # _infer_parameter_dtype would incorrectly cast bf16 back to fp32.
                is_f8 = hasattr(torch, "float8_e4m3fn") and bcast_dtype == torch.float8_e4m3fn
                final_dtype = None if is_f8 else bcast_dtype
                sharded_sd[param_name] = _finish(st, contig, final_dtype, cpu_offload)

            elapsed = time.time() - t0
            print(f"[fsdp2_buffer_patch] Broadcast done in {elapsed:.1f}s, "
                  f"loading {len(sharded_sd)} entries into model...")
            model.load_state_dict(sharded_sd, assign=True)
            print(f"[fsdp2_buffer_patch] State dict loaded successfully "
                  f"({time.time() - t0:.1f}s total)")
            return model

        fsdp_utils.fsdp2_load_full_state_dict = _patched
        print("[fsdp2_buffer_patch] Patched fsdp2_load_full_state_dict for buffer compatibility")
    except Exception as e:
        print(f"[fsdp2_buffer_patch] Patch skipped: {e}")


_clip_grad_norm_call_count = 0


def _clip_grad_norm(parameters, max_norm, norm_type=2):
    """Clip gradient norms for FSDP2 DTensor parameters.

    Bypasses DTensor dispatch (which deadlocks with partially-frozen models
    on the accelerate FSDP2 path) by extracting local tensor shards and
    doing an explicit all_reduce for the global norm.

    Handles Shard (need all_reduce) and Replicate/regular (already global)
    placements.  Safe for DFlash-only and LoRA co-training.
    """
    global _clip_grad_norm_call_count
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Shard

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters]  # materialize generator
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        # Match the device of the normal return path (GPU when training on CUDA) so
        # callers don't hit a device mismatch on the empty-grad case.
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.tensor(0.0, device=dev)

    # Shard DTensors hold partial data — need all_reduce for global norm.
    # Replicate DTensors and regular tensors already hold full data.
    device = grads[0]._local_tensor.device if isinstance(grads[0], DTensor) else grads[0].device
    sharded_norm_p = torch.tensor(0.0, device=device)
    local_norm_p = torch.tensor(0.0, device=device)

    n_sharded = 0
    n_replicate = 0
    n_regular = 0

    for g in grads:
        if isinstance(g, DTensor):
            is_sharded = any(isinstance(p, Shard) for p in g.placements)
            t = g._local_tensor.detach().to(torch.float32)
            n = torch.linalg.vector_norm(t, norm_type)
            if is_sharded:
                sharded_norm_p += n.pow(norm_type)
                n_sharded += 1
            else:
                local_norm_p += n.pow(norm_type)
                n_replicate += 1
        else:
            n = torch.linalg.vector_norm(g.detach().to(torch.float32), norm_type)
            local_norm_p += n.pow(norm_type)
            n_regular += 1

    dist.all_reduce(sharded_norm_p, op=dist.ReduceOp.SUM)
    total_norm = (sharded_norm_p + local_norm_p).pow(1.0 / norm_type)

    clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)

    # Debug: log computation breakdown on first 5 calls (no collectives — safe).
    _clip_grad_norm_call_count += 1
    if _clip_grad_norm_call_count <= 5 and dist.get_rank() == 0:
        print(
            f"[clip_grad_norm debug] call={_clip_grad_norm_call_count} "
            f"total_norm={total_norm.item():.6f} "
            f"sharded_norm_p={sharded_norm_p.item():.6f} local_norm_p={local_norm_p.item():.6f} "
            f"grads={len(grads)} (sharded={n_sharded} replicate={n_replicate} regular={n_regular}) "
            f"max_norm={max_norm} clip_coef={clip_coef.item():.6f}"
        )
    for g in grads:
        if isinstance(g, DTensor):
            g._local_tensor.mul_(clip_coef)
        else:
            g.mul_(clip_coef)

    return total_norm


def patch_accelerator(accelerator):
    """Replace accelerator's clip_grad_norm_ with FSDP2-safe version."""
    accelerator.clip_grad_norm_ = _clip_grad_norm
    print("[fsdp2_buffer_patch] Patched accelerator.clip_grad_norm_ "
          "for FSDP2 DTensor compatibility")
