# Adapted from https://github.com/IST-DASLab/FP-Quant/blob/d2e3092/src/quantization/gptq.py
# with minor modifications to the original forms to accommodate minor architectural differences
# to be reused in the Model-Optimizer pipeline.
# Copyright (c) Andrei Panferov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

"""GPTQ helper and Hessian utilities for calibration."""

import functools
import queue
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch
import torch.distributed as dist

import modelopt.torch.kernels.quantization.gemm as triton_kernel
from modelopt.torch.utils import (
    get_available_gpu_memory,
    get_process_gpus,
    print_rank_0,
    warn_rank_0,
)
from modelopt.torch.utils.network import bind_forward_method, unpatch_forward_method

from .numeric_utils import E4M3_MAX, fp8_max_for_normalization

# A GPU takes a new Hessian only if enough memory stays available after it for the calibration
# forward that follows. Until GPTQ has measured that forward on the GPU, keep this share free.
_HESSIAN_UNMEASURED_RESERVE = 0.35
# Once measured, keep this multiple of the most a calibration forward added on the GPU, and at
# least this share of the GPU.
_HESSIAN_RESERVE_MARGIN = 1.5
_HESSIAN_MIN_RESERVE = 0.1
# The most memory a GPTQ calibration forward has added on each GPU, by device index.
_calib_forward_peak: dict[int, int] = {}


def _gpu_has_room(gpu: int, nbytes: int) -> bool:
    """Whether ``nbytes`` more on ``gpu`` leaves the calibration forward the memory it needs."""
    total = torch.cuda.get_device_properties(gpu).total_memory
    peak = _calib_forward_peak.get(gpu)
    if peak is None:
        reserve = _HESSIAN_UNMEASURED_RESERVE * total
    else:
        reserve = max(_HESSIAN_MIN_RESERVE * total, _HESSIAN_RESERVE_MARGIN * peak)
    return get_available_gpu_memory(gpu) - nbytes >= reserve


def reset_calib_forward_memory():
    """Forget measured calibration forwards; a new run, maybe of another model, measures afresh."""
    _calib_forward_peak.clear()


@contextmanager
def track_calib_forward_memory():
    """Record the memory the enclosed calibration forward adds on each GPU, for Hessian placement.

    PyTorch's peak counters are reset first: otherwise an earlier peak, such as the previous
    layer's Hessians and GPTQ updates, would count as forward memory and send Hessians to the CPU
    for nothing.
    """
    gpus = get_process_gpus() if torch.cuda.is_available() else []
    allocated = {}
    for gpu in gpus:
        torch.cuda.reset_peak_memory_stats(gpu)
        allocated[gpu] = torch.cuda.memory_allocated(gpu)
    yield
    for gpu in gpus:
        added = torch.cuda.max_memory_allocated(gpu) - allocated[gpu]
        _calib_forward_peak[gpu] = max(_calib_forward_peak.get(gpu, 0), added)


def _hessian_offload_device(nbytes: int) -> torch.device:
    """Where a Hessian goes when its weight's GPU lacks room: the emptiest GPU with room, else CPU.

    A model sharded over several GPUs leaves the ones not holding the current layer idle, and a
    Hessian there costs a device-to-device copy per update instead of a host copy and a CPU
    rescale of the whole matrix. Only GPUs this process already holds memory on are candidates,
    and only outside distributed runs, so no other job's or rank's GPU is touched.
    """
    for gpu in sorted(_own_gpus(), key=get_available_gpu_memory, reverse=True):
        if _gpu_has_room(gpu, nbytes):
            return torch.device("cuda", gpu)
    return torch.device("cpu")


def _own_gpus() -> list[int]:
    """GPUs this process already holds memory on; none in a multi-rank run, where they may be another rank's."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return []
    return get_process_gpus()


def _update_room(helper, weight, block_size) -> int:
    """Bytes a GPU needs to run ``helper``'s update of ``weight``, as measured with the CUDA codecs.

    The Hessian, the inverse and their temporaries; the float32 copy of the weight with a
    dtype-conversion temporary (a projection much wider than its input makes the weight the larger
    part); the block's errors and up to four temporaries of a group no wider than the block, each
    as large as the weight when the input is no wider than a block; and the codec's tables and
    scratch, about 35 MiB on a GPU's first update.
    """
    columns = weight.shape[-1]
    rows = weight.numel() // columns
    block = min(block_size, columns)
    return 2**26 + 4 * (8 * helper.hessian.numel() + rows * (2 * columns + 5 * block))


def update_expert_weights(work, block_size, perc_damp):
    """Run the GPTQ updates of ``(helper, weight)`` pairs, spread over this process's GPUs.

    Expert updates are independent, and in a layerwise run of a model sharded over several GPUs
    only the current layer's GPU works. Helpers whose update can run on any device run in one
    worker thread per GPU this process uses that has room for an update, each writing its result
    back to its weight; the others, or all of them when fewer than two GPUs qualify, run in turn on
    their weight's device.
    """
    movable = [(h, w) for h, w in work if w.is_cuda and h.updates_on_any_device()]
    # A worker holds one expert at a time.
    room = max((_update_room(h, w, block_size) for h, w in movable), default=0)
    gpus = [torch.device("cuda", g) for g in _own_gpus() if get_available_gpu_memory(g) > room]
    if len(gpus) < 2 or len(movable) < 2:
        movable = []

    def update(item, device):
        helper, weight = item
        helper.update_weights(block_size, perc_damp, weight, device=device)
        helper.free()

    moved = {id(helper) for helper, _ in movable}
    for item in work:
        if id(item[0]) not in moved:
            update(item, None)
    if not movable:
        return

    pending: queue.SimpleQueue = queue.SimpleQueue()
    for item in movable:
        pending.put(item)

    def next_item(device):
        try:
            return pending.get_nowait()
        except queue.Empty:
            return None

    run_on_each_gpu(gpus, next_item, update)
    for device in {weight.device for _, weight in movable}:
        torch.cuda.synchronize(device)


def run_on_each_gpu(gpus, next_item, run):
    """Call ``run(item, device)`` on the items ``next_item(device)`` hands out, one thread per GPU.

    ``next_item`` returns the next item for that GPU, or ``None`` when there is none left; each GPU's
    items can be its own or come from a queue all of them share. The first item runs in the caller's
    thread with ``device=None``, on the device its data is on, so lazy setup that is not thread-safe,
    such as loading a format's CUDA extension, happens once before the workers start.
    """
    first = next_item(gpus[0])
    if first is not None:
        run(first, None)
    inference = torch.is_inference_mode_enabled()

    def worker(device):
        # The device, grad mode and inference mode are per thread: take the caller's.
        torch.cuda.set_device(device)
        with torch.inference_mode() if inference else torch.no_grad():
            while (item := next_item(device)) is not None:
                run(item, device)
        torch.cuda.synchronize(device)

    with ThreadPoolExecutor(len(gpus)) as pool:
        for future in [pool.submit(worker, device) for device in gpus]:
            future.result()


def update_hessian(input, hessian, n_samples, input_dtype=torch.float32):
    """Update hessian matrix with new input samples using incremental formula.

    Args:
        input: Input tensor (batch_size, ..., features)
        hessian: Current Hessian matrix to update in-place
        n_samples: Number of samples already processed
        input_dtype: The dtype the GEMM reads ``input`` in: ``torch.float32``, or
            ``torch.bfloat16`` for a tensor-core GEMM that accumulates in float32.

    Returns:
        Tuple of (updated_hessian, new_sample_count)

    Note: input must be non-empty (batch_size > 0); a zero-sized input causes division by zero.
    """
    # Flatten to 2D (total_tokens, features) first, so batch_size counts tokens
    input_flat = input.reshape(-1, input.shape[-1])
    batch_size = input_flat.shape[0]

    if batch_size == 0:  # in MOEs some experts receive no tokens
        return hessian, n_samples

    # Incremental averaging, H = H * n / (n + b) + (2 / (n + b)) * X^T X, in one GEMM: this runs
    # once per expert per forward, and separate scale and add passes each rewrite all of H.
    scale = n_samples / (n_samples + batch_size)
    n_samples += batch_size
    alpha = 2 / n_samples
    if bfloat16_hessian_gemm(input_dtype, input_flat):
        # BF16 products are exact in float32, so a tensor-core GEMM writing float32 differs from a
        # float32 GEMM only in how it rounds the sums: up to ~3e-5 of the Hessian's diagonal at 32K
        # tokens per batch. It is 9-16x faster on GB300 and needs no float32 copy of the inputs.
        input_flat = input_flat.to(torch.bfloat16)
        if hessian.device == input_flat.device:
            torch.addmm(
                hessian,
                input_flat.t(),
                input_flat,
                beta=scale,
                alpha=alpha,
                out_dtype=torch.float32,
                out=hessian,
            )
        else:
            gram = torch.mm(input_flat.t(), input_flat, out_dtype=torch.float32)
            hessian.mul_(scale).add_(gram.to(hessian.device), alpha=alpha)
        return hessian, n_samples

    input_flat = input_flat.float()
    if hessian.device == input_flat.device:
        hessian.addmm_(input_flat.t(), input_flat, beta=scale, alpha=alpha)
    else:
        hessian.mul_(scale).add_((input_flat.t() @ input_flat).to(hessian.device), alpha=alpha)

    return hessian, n_samples


def bfloat16_hessian_gemm(input_dtype: torch.dtype, inputs: torch.Tensor) -> bool:
    """Whether a Hessian GEMM over ``inputs`` runs on bfloat16 with float32 output.

    True when ``input_dtype`` asks for bfloat16 and the inputs' GPU has such GEMMs; when they are
    missing, warns once per device and returns False, so the caller computes in float32.
    """
    if input_dtype != torch.bfloat16:
        return False
    if inputs.is_cuda and _has_float32_out_gemm(inputs.device):
        return True
    _warn_bfloat16_hessian_unavailable(inputs.device)
    return False


@functools.cache
def _has_float32_out_gemm(device: torch.device) -> bool:
    """Whether GEMMs on CUDA ``device`` take bfloat16 inputs and write float32 (``out_dtype``)."""
    try:
        x = torch.zeros(1, 1, device=device, dtype=torch.bfloat16)
        h = torch.mm(x, x, out_dtype=torch.float32)
        torch.addmm(h, x, x, out_dtype=torch.float32, out=h)
        torch.bmm(x[None], x[None], out_dtype=torch.float32)
    except (TypeError, RuntimeError):
        return False
    return True


@functools.cache
def _warn_bfloat16_hessian_unavailable(device: torch.device) -> None:
    warn_rank_0(
        f"hessian_input_dtype='bfloat16' needs CUDA GEMMs that take out_dtype, which {device} lacks "
        "with this torch; computing its Hessians in float32."
    )


def compute_hessian_inverse(hessian, weight, perc_damp):
    """Compute damped upper-Cholesky inverse Hessian.

    Dead-neuron columns (all-zero in ``weight``) are zeroed in the
    Hessian before inversion, matching the FP-Quant reference:
    https://github.com/IST-DASLab/FP-Quant/blob/d2e3092f968262c4de5fb050e1aef568a280dadd/src/quantization/gptq.py#L200

    Args:
        hessian: Hessian matrix ``[in_features, in_features]``.
        weight: Weight matrix ``[out_features, in_features]`` for dead-neuron detection.
        perc_damp: Percentage of average Hessian diagonal for damping.

    Returns:
        Upper-triangular Cholesky factor of the damped inverse Hessian
        ``[in_features, in_features]``.  Falls back to the identity matrix
        when the Hessian is not positive definite.
    """
    h = hessian.clone()
    zero_cols = torch.nonzero(weight.eq(0).all(dim=0)).unsqueeze(-1)

    h[zero_cols, :] = 0
    h[:, zero_cols] = 0
    h[zero_cols, zero_cols] = 1

    damp = perc_damp * torch.mean(torch.diag(h))
    diag_indices = torch.arange(h.shape[0], device=h.device)
    h[diag_indices, diag_indices] += damp

    try:
        h = torch.cholesky_inverse(torch.linalg.cholesky(h))
        return torch.linalg.cholesky(h, upper=True)
    except (RuntimeError, torch.linalg.LinAlgError):
        print_rank_0("Warning: Hessian is not positive definite, using identity matrix")
        return torch.eye(h.shape[0], device=h.device, dtype=h.dtype)


class GPTQHelper:
    """Encapsulates per-module GPTQ state and operations.

    Owns the Hessian, patches the forward during collection, and contains
    the blockwise weight-update logic.

    Instance attributes set during ``__init__``:
        module, name, hessian, n_samples

    Instance attributes set during ``update_weights``:
        weight: float working copy of module weights (mutated in-place by update methods)
        h_inv: upper-triangular Cholesky factor of the damped inverse Hessian
    """

    CACHE_NAME = "_forward_no_gptq_hessian"

    def __init__(
        self,
        module,
        name,
        offload_to_cpu=False,
        fused=False,
        importance_weighted=False,
        weight_quantizer=None,
        weight=None,
        log_mse=True,
        hessian_input_dtype="float32",
    ):
        """Initialize GPTQHelper with module state and Hessian storage.

        ``weight_quantizer`` and ``weight`` select one of several weights a module holds, such as
        one expert's slice of an HF fused-MoE parameter; by default the helper owns
        ``module.weight`` and ``module.weight_quantizer``. ``log_mse=False`` skips the logged
        Hessian-weighted error, which costs two weight-by-Hessian matmuls.
        ``hessian_input_dtype`` is the dtype the Hessian GEMM reads the inputs in, by name.
        """
        self.module = module
        self.name = name
        self.fused = fused
        self.importance_weighted = importance_weighted
        self.hessian_input_dtype = getattr(torch, hessian_input_dtype)
        self._weight_quantizer = weight_quantizer
        self.log_mse = log_mse
        weight = module.weight if weight is None else weight
        in_features = weight.shape[-1]
        device = weight.device
        if device.type == "meta":
            device = torch.device("cpu")
        elif offload_to_cpu and device.type == "cuda":
            index = torch.cuda.current_device() if device.index is None else device.index
            if not _gpu_has_room(index, 4 * in_features**2):
                device = _hessian_offload_device(4 * in_features**2)
        self.hessian = torch.zeros(in_features, in_features, dtype=torch.float32, device=device)
        self.n_samples = 0
        # Set by update_weights(); listed here for documentation.
        self.weight: torch.Tensor | None = None
        self.h_inv: torch.Tensor | None = None

    @property
    def weight_quantizer(self):
        """The quantizer GPTQ quantizes this helper's weight with."""
        if self._weight_quantizer is not None:
            return self._weight_quantizer
        return self.module.weight_quantizer

    def setup(self):
        """Patch the module's forward to accumulate Hessian during the collection pass."""
        gptq_helper = self

        def hessian_forward(self, input, *args, **kwargs):
            inp = input.to_local() if hasattr(input, "to_local") else input
            if self.input_quantizer is not None and self.input_quantizer.is_enabled:
                hessian_input = self.input_quantizer(inp)
            else:
                hessian_input = inp
            gptq_helper.accumulate(hessian_input)

            out = self._forward_no_gptq_hessian(input, *args, **kwargs)

            return out

        bind_forward_method(self.module, hessian_forward, self.CACHE_NAME)

    def accumulate(self, inputs):
        """Add a batch of this weight's inputs, after input quantization, to the Hessian."""
        inputs = inputs.to_local() if hasattr(inputs, "to_local") else inputs
        self.hessian, self.n_samples = update_hessian(
            inputs, self.hessian, self.n_samples, self.hessian_input_dtype
        )

    def cleanup(self):
        """Unpatch the module's forward method."""
        unpatch_forward_method(self.module, self.CACHE_NAME)

    def sync_hessian_across_dp(self):
        """Merge the Hessians of this module's data-parallel replicas, which saw different data."""
        parallel_state = getattr(self.module, "parallel_state", None)
        group = parallel_state.data_parallel_group if parallel_state is not None else None
        if group is None or not group.is_initialized() or group.world_size() == 1:
            return
        on_cuda = "nccl" in dist.get_backend(group.group)
        device = (
            torch.device("cuda", torch.cuda.current_device()) if on_cuda else torch.device("cpu")
        )
        n_samples = torch.tensor(self.n_samples, dtype=torch.float64, device=device)
        hessian = self.hessian.to(device).mul_(self.n_samples)
        dist.all_reduce(n_samples, group=group.group)
        dist.all_reduce(hessian, group=group.group)
        self.n_samples = int(n_samples.item())
        if self.n_samples:
            hessian /= self.n_samples
        self.hessian = hessian.to(self.hessian.device)

    def free(self):
        """Release Hessian and working tensors to reclaim memory."""
        self.hessian = None
        self.weight = None
        self.h_inv = None

    def updates_on_any_device(self) -> bool:
        """Whether :meth:`update_weights` may run on a device other than its weight's.

        True only for helpers whose update reads no quantizer state, which lives on the weight's
        device, and fits the room :func:`update_expert_weights` checks for.
        """
        return False

    def update_weights(self, block_size, perc_damp, weight=None, device=None):
        """Run GPTQ blockwise weight update on this helper's weight.

        Populates ``self.weight`` and ``self.h_inv``, runs the blockwise update, logs MSE, and
        overwrites ``weight`` (default ``module.weight``) in place, so an expert's slice of a
        fused 3-D parameter is written back into that parameter. The update runs on ``device``,
        by default the weight's; only a helper whose :meth:`updates_on_any_device` is true may
        pass another.
        """
        weight = self.module.weight if weight is None else weight
        device = weight.device if device is None else device
        hessian = self.hessian.to(device)
        self.weight = weight.data.to(device, torch.float32, copy=True)
        self._prepare_hessian_inverse(hessian, perc_damp)
        self._blockwise_update(block_size)
        if self.log_mse:
            self._print_mse_error(hessian, weight.data.to(device))
        weight.data.copy_(self.weight.reshape(weight.shape))

    # ------------------------------------------------------------------
    # Quantize helpers — all read from self.weight_quantizer, self.weight, self.h_inv
    # ------------------------------------------------------------------

    def _prepare_hessian_inverse(self, hessian, perc_damp):
        """Compute damped inverse Hessian and store as ``self.h_inv``."""
        assert self.weight is not None, "_prepare_hessian_inverse called before update_weights()"
        self.h_inv = compute_hessian_inverse(hessian, self.weight, perc_damp)

    def _blockwise_update(self, block_size):
        """Column-wise GPTQ update.

        When ``self.fused`` is True and :meth:`_fused_supported` holds, uses
        :func:`gptq_blockwise_update_fused_scalar` (a fused Triton kernel).  Otherwise falls back to
        :func:`gptq_blockwise_update` (unfused column-by-column loop).
        """
        assert self.weight is not None and self.h_inv is not None, (
            "_blockwise_update called before _prepare_hessian_inverse()"
        )
        quantizer = self.weight_quantizer

        if self.fused and self._fused_supported(quantizer, self.weight):
            block_sizes = quantizer.block_sizes
            quant_block_size = block_sizes.get(-1) or block_sizes.get(1)
            if quant_block_size is not None and block_size % quant_block_size != 0:
                raise ValueError(
                    f"GPTQ block_size ({block_size}) must be divisible by the quantizer"
                    f" group_size ({quant_block_size})"
                )
            out_features, num_cols = self.weight.shape
            n_blocks = num_cols // quant_block_size
            block_amax = quantizer.amax.reshape(out_features, n_blocks).float()
            global_scale = quantizer.global_amax.float().item() / (6.0 * 448.0)
            gptq_blockwise_update_fused_scalar(
                self.weight, block_amax, global_scale, self.h_inv, block_size, quant_block_size
            )
        else:
            gptq_blockwise_update(self.weight, self.h_inv, block_size, quantizer)

    def _fused_supported(self, quantizer, weight):
        # The fused kernel needs FP8 Triton casts (SM89+) and the default FP8 scale range (no 4/6).
        return (
            getattr(quantizer, "_is_nvfp4_static_quantizer", False)
            and fp8_max_for_normalization(quantizer) == E4M3_MAX
            and triton_kernel.IS_AVAILABLE
            and weight.is_cuda
            and torch.cuda.get_device_capability(weight.device) >= (8, 9)
        )

    def _print_mse_error(self, hessian, weight):
        """Log Hessian-weighted relative MSE between ``self.weight`` and original ``weight``."""
        w_orig = weight.float()
        delta = self.weight - w_orig
        mse = (delta).mm(hessian).mul(delta).mean() / (w_orig.mm(hessian).mul(w_orig).mean() + 1e-6)
        suffix = f", n_hessian_samples: {self.n_samples}" if self.n_samples else ""
        print_rank_0(f"[{self.name}] Relative MSE error: {mse.item():.2e}{suffix}")


def gptq_blockwise_update(weight, h_inv, block_size, quantize_fn):
    """Column-wise GPTQ update using full-matrix fake quantization.

    For each column, quantizes the full weight matrix via ``quantize_fn`` and
    extracts the quantized column.  Error is propagated to remaining columns
    within the block and then to all subsequent columns via the inverse Hessian.

    Args:
        weight: Weight tensor ``[out_features, in_features]``, modified **in-place**
            with fake-quantized values.
        h_inv: Upper-triangular Cholesky factor of the damped inverse Hessian
            ``[in_features, in_features]``.
        block_size: Number of columns to process per GPTQ block.
        quantize_fn: Callable ``(weight) -> qdq_weight`` that fake-quantizes
            the full weight matrix.
    """
    num_cols = weight.shape[1]

    for block_start in range(0, num_cols, block_size):
        block_end = min(block_start + block_size, num_cols)
        n_cols_blk = block_end - block_start
        h_inv_cho_blk = h_inv[block_start:block_end, block_start:block_end]

        wblk = weight.clone()
        errs = torch.zeros_like(weight[:, block_start:block_end])

        for i in range(n_cols_blk):
            w_ci = wblk[:, block_start + i]
            d = h_inv_cho_blk[i, i]
            qdq = quantize_fn(wblk)
            weight[:, block_start + i] = qdq[:, block_start + i]
            err = (w_ci - qdq[:, block_start + i]) / d
            wblk[:, block_start + i : block_end].addr_(err, h_inv_cho_blk[i, i:], alpha=-1)
            errs[:, i] = err

        weight[:, block_end:].addmm_(errs, h_inv[block_start:block_end, block_end:], alpha=-1)


def gptq_blockwise_update_fused_scalar(
    weight, block_amax, global_scale, h_inv, block_size, quant_block_size
):
    """Fused GPTQ blockwise update for NVFP4 scalar quantization.

    Uses a fused Triton kernel that combines scale computation, quantization,
    and per-column error propagation into one launch per GPTQ block, avoiding
    the Python-level per-column loop in :func:`gptq_blockwise_update`.

    Args:
        weight: Weight tensor ``[out_features, in_features]``, modified **in-place**
            with fake-quantized values.
        block_amax: Per-block amax values ``[out_features, n_amax_blocks]``.
        global_scale: Pre-computed ``global_amax / (6.0 * 448.0)`` (scalar).
        h_inv: Upper-triangular Cholesky factor of the damped inverse Hessian
            ``[in_features, in_features]``.
        block_size: Number of columns to process per GPTQ block.
        quant_block_size: Number of elements sharing one quantization scale factor.
    """
    from modelopt.torch.kernels.quantization.gemm.gptq_fused_kernel import gptq_fused_block_scalar

    num_cols = weight.shape[1]
    for bs in range(0, num_cols, block_size):
        be = min(bs + block_size, num_cols)
        qw, err = gptq_fused_block_scalar(
            weight[:, bs:be].clone().contiguous(),
            block_amax,
            global_scale,
            h_inv[bs:be, bs:be].contiguous(),
            quant_block_size,
            bs,
        )
        weight[:, bs:be] = qw
        if be < num_cols:
            weight[:, be:].addmm_(err, h_inv[bs:be, be:], alpha=-1)


_GPTQ_HELPER_REGISTRY: dict[str, type[GPTQHelper]] = {}


def register_gptq_helper(backend: str, factory: type[GPTQHelper]) -> None:
    """Register a :class:`GPTQHelper` subclass for a quantizer backend.

    When :func:`modelopt.torch.quantization.model_calib.gptq` encounters a
    module whose ``weight_quantizer.backend`` matches ``backend``, it will
    construct ``factory`` instead of the default ``GPTQHelper``.
    """
    _GPTQ_HELPER_REGISTRY[backend] = factory


class ImatrixEncoder:
    """Encodes a weight once, its search weighted by its inputs' per-column second moments.

    :func:`modelopt.torch.quantization.model_calib.imatrix_calibrate` collects ``E[x_j^2]`` for
    every weight whose quantizer an encoder accepts, then calls :meth:`encode` with it.
    """

    def accepts(self, quantizer) -> bool:
        """Whether this encoder handles ``quantizer``; decided before calibration runs."""
        raise NotImplementedError

    def validate(self, weight: torch.Tensor, quantizer) -> None:
        """Raise if :meth:`encode` would refuse ``weight``; checked before calibration runs."""

    def encode(
        self, weight: torch.Tensor, quantizer, second_moments: torch.Tensor | None
    ) -> torch.Tensor:
        """Return ``weight`` fake-quantized by ``quantizer``, weighted by ``second_moments``.

        ``second_moments`` is ``None`` when no calibration input reached the weight.
        """
        raise NotImplementedError


_IMATRIX_ENCODER_REGISTRY: dict[str, ImatrixEncoder] = {}


def register_imatrix_encoder(backend: str, encoder: ImatrixEncoder) -> None:
    """Register the :class:`ImatrixEncoder` for a quantizer backend.

    :func:`modelopt.torch.quantization.model_calib.imatrix_calibrate` only encodes weights whose
    ``weight_quantizer.backend`` has a registered encoder.
    """
    _IMATRIX_ENCODER_REGISTRY[backend] = encoder
