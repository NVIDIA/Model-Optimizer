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

"""Calibration utilities."""

import fnmatch
import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import TypeAlias

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from modelopt.torch.opt.searcher import ForwardLoop
from modelopt.torch.quantization.utils.layerwise_calib import (
    LayerActivationCollector,
    _CheckpointState,
)
from modelopt.torch.utils import print_rank_0, warn_rank_0
from modelopt.torch.utils.distributed import DistributedProcessGroup, ParallelState
from modelopt.torch.utils.distributed import is_initialized as dist_is_initialized
from modelopt.torch.utils.distributed import size as dist_size
from modelopt.torch.utils.network import bind_forward_method, unpatch_forward_method

from .calib import MseCalibrator, NVFP4MSECalibrator, _Calibrator
from .conversion import create_and_replace_svdquant_linear_on_the_fly, set_quantizer_by_cfg_context
from .nn import NVFP4StaticQuantizer, QuantModule, SequentialQuantizer, TensorQuantizer
from .utils import (
    SHARED_PATTERNS,
    SharedWeightGlobalAmaxState,
    disable_calib,
    enable_fake_quant,
    enable_quant,
    enable_weight_access_and_writeback,
    is_quantized_column_parallel_linear,
    is_quantized_linear,
    is_quantized_row_parallel_linear,
    persistent_materialization,
    promote_nvfp4_static_quantizers,
)
from .utils.calib_utils import _GPTQ_HELPER_REGISTRY, GPTQHelper

__all__ = [
    "CalibratorFactory",
    "awq",
    "layerwise_calibrate",
    "local_hessian_calibrate",
    "max_calibrate",
    "smoothquant",
    "svdquant",
]


def _collect_weight_stats(quantizer: nn.Module, weight: torch.Tensor) -> None:
    quantizer(weight)


def _is_calibrated_nvfp4_static(q) -> bool:
    """True iff ``q`` is an enabled NVFP4-static weight quantizer with ``_amax`` set."""
    return (
        isinstance(q, NVFP4StaticQuantizer)
        and not q._disabled
        and q.is_nvfp4_static
        and getattr(q, "_amax", None) is not None
    )


def _collect_grouped_linears(model: nn.Module) -> list[list[nn.Module]]:
    """Collect name-based sibling groups (Q/K/V, gate/up, w1/w3) of calibrated NVFP4-static linears."""
    # Inline import: layer_utils -> quant_utils -> model_calib cycle.
    from modelopt.torch.export.layer_utils import _GATE_UP_PAIRS

    patterns: tuple[tuple[str, ...], ...] = (("q_proj", "k_proj", "v_proj"), *_GATE_UP_PAIRS)
    groups: list[list[nn.Module]] = []
    for parent in model.modules():
        for sibling_names in patterns:
            members = [
                child
                for child in (getattr(parent, n, None) for n in sibling_names)
                if child is not None
                and _is_calibrated_nvfp4_static(getattr(child, "weight_quantizer", None))
            ]
            if len(members) >= 2:
                groups.append(members)
    return groups


@torch.no_grad()
def _check_grouped_weight_global_amax_synced(model: nn.Module) -> None:
    """Verify shared NVFP4 state unified each name-based fusible group's weight global_amax.

    The default name-based grouping (Q/K/V, gate/up, w1/w3) is kept here as a *check*
    rather than performed: after attach/populate/promote, the promoted static-NVFP4 weight
    quantizers in each name group must already share one ``global_amax``. This catches the
    SharedWeightGlobalAmaxState path failing to form or sync a group it should have (e.g. a
    default-pattern regression, or an architecture the regexes miss)
    before the MSE per-block search — computed against ``global_amax`` — bakes in the
    inconsistency. Run only when the default patterns are in effect (custom
    ``shared_states`` may intentionally group differently). Members whose ``global_amax``
    is not materialized (``None``/meta, e.g. an ``init_empty_weights`` model) are skipped.
    """
    for group in _collect_grouped_linears(model):
        amaxes = [m.weight_quantizer.global_amax for m in group]
        amaxes = [a for a in amaxes if a is not None and not a.is_meta]
        if len(amaxes) < 2:
            continue
        ref = amaxes[0]
        assert all(torch.equal(a, ref) for a in amaxes), (
            "A fusible sibling group (q/k/v or gate/up) was not unified to a shared weight "
            "global_amax; SharedWeightGlobalAmaxState failed to sync it, so the per-block "
            "MSE scales would be inconsistent across the group."
        )


def _finalize_with_shared_state(model: nn.Module, weight_patterns: list[str]) -> None:
    """Finalize quantization from the attached shared state: aggregate, promote, verify.

    Aggregates each fusible group's shared weight ``global_amax`` and promotes it onto the
    member NVFP4-static quantizers, so siblings read the unified value instead of their own
    ``_amax``; under the default patterns, verifies the name groups were actually synced.
    Call once ``_amax`` is final: single-process, or after the distributed amax sync.
    """
    SharedWeightGlobalAmaxState.populate(model)
    promote_nvfp4_static_quantizers(model)
    # Under the default patterns, verify the fusible name groups were actually synced.
    if weight_patterns == list(SHARED_PATTERNS):
        _check_grouped_weight_global_amax_synced(model)


CalibratorFactory: TypeAlias = Callable[
    [torch.Tensor, int | tuple | list | None, Callable[..., torch.Tensor]], _Calibrator
]

_FP8_SWEEP_CALIBRATOR_REGISTRY: dict[str, CalibratorFactory] = {}


def _register_fp8_sweep_calibrator(backend: str, calibrator_factory: CalibratorFactory) -> None:
    """Register a custom calibrator factory for a quantization backend.

    When ``fp8_scale_sweep=True`` is passed to :func:`mse_calibrate`, any weight
    quantizer whose ``backend`` attribute matches a registered key will use the
    corresponding factory instead of the default :class:`MseCalibrator`.

    Args:
        backend: Backend name string (must match ``TensorQuantizer.backend``).
        calibrator_factory: Callable with signature
            ``(amax: Tensor, axis: int | tuple | list | None, quant_func: Callable)``
            that returns a :class:`_Calibrator` instance.
    """
    _FP8_SWEEP_CALIBRATOR_REGISTRY[backend] = calibrator_factory


def _uses_modelopt_fp8_weight_scales(weight_quantizer: TensorQuantizer) -> bool:
    """Whether the internal ModelOpt FP8-scale MSE sweep applies to this quantizer."""
    return weight_quantizer.backend is None and weight_quantizer.is_nvfp4_static


def weight_only_quantize(model: nn.Module):
    """Just quantize the weights of the model."""
    name_to_module = dict(model.named_modules())
    seen_modules = set()
    for module in name_to_module.values():
        if module in seen_modules:
            continue

        if isinstance(module, QuantModule):
            with enable_weight_access_and_writeback(module, model, name_to_module):
                for weight, weight_quantizer in module.iter_weights_for_calibration():
                    weight_quantizer(weight)
        seen_modules.add(module)


def _run_and_load_max_stats(model: nn.Module, forward_loop: ForwardLoop | None = None):
    """Run max-stat collection and load collected stats without post-processing."""
    enable_stats_collection(model)
    if forward_loop is None:
        weight_only_quantize(model)
    else:
        forward_loop(model)
    finish_stats_collection(model)


def _has_expert_parallelism(module: nn.Module) -> bool:
    """Check if module has expert parallelism enabled."""
    ps = getattr(module, "parallel_state", None)
    return ps is not None and ps.expert_model_parallel_group.is_initialized()


def _iter_leaf_quantizers(quantizer):
    if isinstance(quantizer, SequentialQuantizer):
        for _q in quantizer:
            yield from _iter_leaf_quantizers(_q)
        return
    yield quantizer


def _check_moe_calibration_complete(quantizer, parallel_state):
    """Raise error if MoE calibration is incomplete across distributed MoE ranks."""
    for leaf_quantizer in _iter_leaf_quantizers(quantizer):
        has_amax = getattr(leaf_quantizer, "_amax", None) is not None
        for group in [
            parallel_state.data_parallel_group,
            parallel_state.expert_model_parallel_group,
            parallel_state.tensor_parallel_group,
        ]:
            if not group.is_initialized():
                continue
            amax_states = DistributedProcessGroup.get_dist_syncd_obj(
                has_amax, group, lambda objs: objs
            )
            if any(amax_states) and not all(amax_states):
                raise RuntimeError(
                    "MoE calibration incomplete: some experts received no tokens during "
                    "calibration. Increase --calib-size to ensure all experts see calibration "
                    "data."
                )


def _is_routed_expert(parent_name: str) -> bool:
    """Routed-expert FQN contains ``experts`` but not ``shared_experts`` (covers SequentialMLP and TEGroupedMLP)."""
    return "experts" in parent_name and "shared_experts" not in parent_name


def _should_sync_amax_across_ep(
    parent_name: str, child_name: str, sync_expert_weight_amax: bool
) -> bool:
    """Skip EP sync for routed-expert weights (per-rank shards differ).

    SequentialMLP opts in via sync_expert_weight_amax.
    """
    if "weight_quantizer" in child_name and _is_routed_expert(parent_name):
        return sync_expert_weight_amax
    return True


@torch.no_grad()
def max_calibrate(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    distributed_sync=True,
    sync_expert_weight_amax=False,
    shared_states: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
):
    """Calibrate the model using max.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.
        distributed_sync: Whether to sync input_quantizer amax across distributed processes.
        sync_expert_weight_amax: SequentialMLP only — share one weight amax across all experts
            in a MoE layer (within-rank sync + EP all-reduce when EP>1).
        shared_states: Optional dict keyed by shared-state name. ``"weight_global_amax"`` is
            implemented today and accepts ``{"patterns": [...]}``; omitted patterns use
            ``SHARED_PATTERNS``, while an empty list disables the state.

    See :class:`MaxCalibConfig <modelopt.torch.quantization.config.MaxCalibConfig>` for
    details on the remaining arguments.
    """
    # Discover fusible sibling groups by name regex and attach the (initially empty) shared
    # state up front, so parent-level runtime hooks can be installed by future concrete
    # states. Discovery is structural (a pattern over the module tree), so it needs no
    # ``_amax``; per-member values are aggregated later by
    # SharedWeightGlobalAmaxState.populate, after the forward and any cross-rank ``_amax`` sync.
    weight_patterns = SharedWeightGlobalAmaxState.resolve_patterns(shared_states=shared_states)
    SharedWeightGlobalAmaxState.attach(model, patterns=weight_patterns)

    # Always run weight calibration on the weight tensor directly so every weight
    # quantizer gets ``_amax``, regardless of MoE routing. Downstream algorithms
    # (MSE, AWQ, export) then no longer need to patch in a missing ``_amax``.
    enable_stats_collection(model)
    weight_only_quantize(model)
    if forward_loop is not None:
        forward_loop(model)
    finish_stats_collection(model)

    # Sync quantizer amax across local experts within each rank (for SequentialMLP)
    for name, module in model.named_modules():
        if hasattr(module, "layer_sync_moe_local_experts_amax"):
            module.layer_sync_moe_local_experts_amax(sync_weight_amax=sync_expert_weight_amax)

    # Fail fast on NVFP4 static-block with TP>1 (sharded_state_dict treats _amax as replicated).
    try:
        from .plugins.megatron import _check_nvfp4_static_tp_supported
    except ImportError:
        pass
    else:
        _check_nvfp4_static_tp_supported(model)

    if not distributed_sync:
        # Single-process: _amax is final.
        _finalize_with_shared_state(model, weight_patterns)
        return

    # Check MoE calibration completeness before sync
    for name, module in model.named_modules():
        if isinstance(module, QuantModule) and _has_expert_parallelism(module):
            for child in module.children():
                if isinstance(child, (TensorQuantizer, SequentialQuantizer)):
                    _check_moe_calibration_complete(child, module.parallel_state)

    def sync_quantizer_amax_across_dp_ep(quantizer, parallel_state, parent_name, child_name):
        """Sync amax across DP (always) and EP (filtered — see _should_sync_amax_across_ep)."""
        if isinstance(quantizer, SequentialQuantizer):
            for _q in quantizer:
                sync_quantizer_amax_across_dp_ep(_q, parallel_state, parent_name, child_name)
            return
        if getattr(quantizer, "_amax", None) is None:
            return
        quantizer.sync_amax_across_distributed_group(parallel_state.data_parallel_group)
        if _should_sync_amax_across_ep(parent_name, child_name, sync_expert_weight_amax):
            quantizer.sync_amax_across_distributed_group(parallel_state.expert_model_parallel_group)

    # Step 2:Sync amax across data parallelism
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            for child_name, child in module.named_children():
                if isinstance(child, (TensorQuantizer, SequentialQuantizer)):
                    sync_quantizer_amax_across_dp_ep(child, module.parallel_state, name, child_name)
    # Step 3: TP sync
    # Objective: the quantization parameters when TP = 8 then changed to TP=4 then back to TP=8 should be the same

    # ColumnParallel: X @ [A_1, A_2] (weights split along Cout)
    #   activations:  TPG should have the same amax if axis in [None, -1]
    #   weights:      TPG should have the same amax if axis in [None, -1] (note: we dont use -1 axis for weights)

    # RowParallel:    [X_1, X_2] @  [A_1
    #                                A_2] (weights split along Cin)
    #   activations:  TPG should have the same amax if axis in [None]
    #   weights:      TPG should have the same amax if axis in [None, 0]

    def sync_quantizer_amax_across_tp(
        quantizer: TensorQuantizer | SequentialQuantizer,
        linear_name: str,
        quantizer_type: str,
        axes_for_sync: list,
        parallel_state: ParallelState,
    ):
        # Syncing amax across TP for sequential quantizer
        if isinstance(quantizer, SequentialQuantizer):
            for _q in quantizer:
                sync_quantizer_amax_across_tp(
                    _q, linear_name, quantizer_type, axes_for_sync, parallel_state
                )
            return
        # sync is not needed for block quantization
        if quantizer.block_sizes is not None:
            if hasattr(quantizer, "_padding"):
                warnings.warn(
                    f"Found block-quantized padded {quantizer_type} for {linear_name}, amax will"
                    " not be synced correctly."
                )
            # Skip amax sync for INT4 / W4A8 block quantization
            # Sync amax for NVFP4 (dynamic per-block, static per-tensor quantized scale)
            if getattr(quantizer.block_sizes, "type", None) == "dynamic":
                return

        if quantizer.axis in axes_for_sync and quantizer.amax is not None:
            quantizer.sync_amax_across_distributed_group(parallel_state.tensor_parallel_group)

    # Step 2: Sync amax across relevant parallelism (such as TP / EP)
    for name, module in model.named_modules():
        if getattr(module, "_parallel_state", None) is None:
            continue

        if is_quantized_column_parallel_linear(module):
            sync_quantizer_amax_across_tp(
                module.input_quantizer,
                name,
                "input_quantizer",
                axes_for_sync=[None, -1],
                parallel_state=module.parallel_state,
            )
            sync_quantizer_amax_across_tp(
                module.weight_quantizer,
                name,
                "weight_quantizer",
                axes_for_sync=[None, -1],
                parallel_state=module.parallel_state,
            )

        if is_quantized_row_parallel_linear(module):
            sync_quantizer_amax_across_tp(
                module.input_quantizer,
                name,
                "input_quantizer",
                axes_for_sync=[None],
                parallel_state=module.parallel_state,
            )

            sync_quantizer_amax_across_tp(
                module.weight_quantizer,
                name,
                "weight_quantizer",
                axes_for_sync=[None, 0],
                parallel_state=module.parallel_state,
            )

        # KV Cache Quantization
        if hasattr(module, "k_bmm_quantizer") and hasattr(module, "v_bmm_quantizer"):
            # We only support KVCache quantization with scalar per-tensor states for now (NVFP4 & FP8 KV cache)
            # So we should sync amax across DP and TP for these quantizers (DP is already synced from above)
            for quantizer in [module.k_bmm_quantizer, module.v_bmm_quantizer]:
                if isinstance(quantizer, TensorQuantizer) and quantizer.amax is not None:
                    quantizer.sync_amax_across_distributed_group(
                        module.parallel_state.tensor_parallel_group
                    )

    # _amax is now cross-rank consistent across ranks.
    _finalize_with_shared_state(model, weight_patterns)


def _mse_quant_func(x, amax, quantizer):
    """Quantization function for MSE calibration."""
    original_amax = quantizer._amax.clone() if hasattr(quantizer, "_amax") else None
    quantizer._amax = amax

    try:
        with (
            enable_quant(quantizer),
            disable_calib(quantizer),
            enable_fake_quant(quantizer),
        ):
            if hasattr(quantizer, "_original_shape"):
                x = quantizer._reset_to_original_shape(x)
            xq = quantizer(x)
            if hasattr(quantizer, "_block_reshape_size"):
                # Reapply static block padding before returning to the calibration block layout.
                xq = quantizer._process_for_blockquant(xq)
    finally:
        if original_amax is not None:
            quantizer._amax = original_amax
        else:
            delattr(quantizer, "_amax")

    return xq


def _make_weight_mse_calibrator(
    weight_quantizer: TensorQuantizer,
    step_size: float,
    start_multiplier: float,
    stop_multiplier: float,
    fp8_scale_sweep: bool,
    error_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    hessian: torch.Tensor | None = None,
) -> _Calibrator | None:
    """Create the MSE calibrator for one eligible weight quantizer (``None`` if ineligible).

    ``error_func`` overrides the squared-error metric (local-Hessian's per-block weighting).
    ``hessian`` (the same per-cin-block metric as a raw tensor) enables NVFP4's Hessian-weighted
    Triton fast path; ``error_func`` then serves only as the reference fallback.
    """
    if (
        not isinstance(weight_quantizer, TensorQuantizer)
        or not weight_quantizer.is_enabled
        or weight_quantizer._dynamic
        or weight_quantizer._calibrator is None
        or getattr(weight_quantizer, "_amax", None) is None
    ):
        return None

    initial_amax = weight_quantizer._amax.clone().detach()
    axis = weight_quantizer._calibrator._axis
    quant_func = partial(_mse_quant_func, quantizer=weight_quantizer)

    if fp8_scale_sweep:
        backend: str | None = getattr(weight_quantizer, "backend", None)
        backend_factory = (
            _FP8_SWEEP_CALIBRATOR_REGISTRY.get(backend) if backend is not None else None
        )
        if backend is not None and backend_factory is not None:
            if error_func is not None:
                # Registered backend factories don't accept a custom error_func.
                warnings.warn(
                    f"backend '{backend}' does not support a custom error function; skipping "
                    "error-function-weighted MSE calibration for this quantizer."
                )
                return None
            return backend_factory(initial_amax, axis, quant_func)
        if _uses_modelopt_fp8_weight_scales(weight_quantizer):
            return NVFP4MSECalibrator(
                amax=initial_amax,
                axis=axis,
                global_amax=weight_quantizer.global_amax,
                quant_func=quant_func,
                error_func=error_func,
                hessian=hessian,
            )
        # fp8_scale_sweep applies only to registered backends and static NVFP4; skip others.
        return None

    # No fp8_scale_sweep: multiplier-search MSE for all quantizers.
    return MseCalibrator(
        amax=initial_amax,
        axis=axis,
        step_size=step_size,
        start_multiplier=start_multiplier,
        stop_multiplier=stop_multiplier,
        quant_func=quant_func,
        error_func=error_func,
    )


@torch.no_grad()
def mse_calibrate(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    distributed_sync=True,
    step_size: float = 0.1,
    start_multiplier: float = 0.25,
    stop_multiplier: float = 4.0,
    fp8_scale_sweep: bool = False,
    shared_states: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
):
    """Calibrate weight quantizers using MSE-based amax search.

    This calibration method first uses max calibration to initialize amax values for
    all quantizers, then searches for better weight amax values by minimizing the MSE
    between original and quantized weights.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.
        distributed_sync: Whether to sync amax across distributed processes.
        step_size: Step size for amax search (default: 0.1).
        start_multiplier: Starting multiplier for amax search (default: 0.25).
        stop_multiplier: Ending multiplier for amax search (default: 4.0).
        fp8_scale_sweep: If True, only ModelOpt static NVFP4 weights and registered
            custom backends are MSE-calibrated (via FP8 E4M3 scale-value sweep); all
            other weight quantizers (INT8, plain FP8, unregistered backends, etc.) are
            skipped and left at their max-calibrated amax. If False, all weight
            quantizers use the multiplier search.

    See :class:`MseCalibConfig <modelopt.torch.quantization.config.MseCalibConfig>` for
    details on the remaining arguments.
    """
    # max_calibrate initializes activations and weights; MSE only refines weights below.
    max_calibrate(model, forward_loop, distributed_sync, shared_states=shared_states)
    name_to_module = dict(model.named_modules())
    _mse_calibrate_weights(
        model,
        name_to_module,
        step_size=step_size,
        start_multiplier=start_multiplier,
        stop_multiplier=stop_multiplier,
        fp8_scale_sweep=fp8_scale_sweep,
    )


@torch.no_grad()
def _mse_calibrate_weights(
    model: nn.Module,
    name_to_module: dict[str, nn.Module],
    step_size: float,
    start_multiplier: float,
    stop_multiplier: float,
    fp8_scale_sweep: bool,
    error_func_for: Callable[[TensorQuantizer], Callable | None] | None = None,
    hessian_for: Callable[[TensorQuantizer], torch.Tensor | None] | None = None,
):
    """Run MSE weight calibration over all eligible quantizers (shared by mse / local-Hessian).

    ``error_func_for`` maps a weight quantizer to an optional per-weight error function
    (local-Hessian's Hessian metric); ``None`` means plain squared error. ``hessian_for``
    maps a weight quantizer to the same metric as a raw per-cin-block Hessian tensor,
    enabling the Hessian-weighted Triton fast path.
    """
    seen_modules: set[int] = set()
    pbar = tqdm(desc="MSE weight calibration")
    for parent_module in name_to_module.values():
        if id(parent_module) in seen_modules or not isinstance(parent_module, QuantModule):
            continue
        seen_modules.add(id(parent_module))
        with enable_weight_access_and_writeback(parent_module, model, name_to_module):
            for weight, weight_quantizer in parent_module.iter_weights_for_calibration():
                error_func = error_func_for(weight_quantizer) if error_func_for else None
                hessian = hessian_for(weight_quantizer) if hessian_for else None
                cal = _make_weight_mse_calibrator(
                    weight_quantizer,
                    step_size,
                    start_multiplier,
                    stop_multiplier,
                    fp8_scale_sweep,
                    error_func=error_func,
                    hessian=hessian,
                )
                if cal is None:
                    continue
                weight_quantizer._calibrator = cal
                _run_and_load_max_stats(
                    weight_quantizer, partial(_collect_weight_stats, weight=weight)
                )
                if hasattr(cal, "reset"):
                    cal.reset()

                pbar.update(1)
    pbar.close()


class _LocalHessianAccumulator:
    """Per-block local Hessian ``H = ΣXᵀX`` for one weight quantizer.

    Partitioned over ``cin`` into ``cin // block_size`` blocks to match the NVFP4 per-block
    scale; the buffer is allocated lazily so never-routed experts cost nothing.
    """

    def __init__(self, cout: int, cin: int, block_size: int):
        self.cout = cout
        self.cin = cin
        self.block_size = block_size
        self.num_blocks_per_cin = cin // block_size
        # Not block-divisible -> no Hessian (falls back to plain MSE).
        self.is_enabled = cin % block_size == 0
        self.hessian_per_block: torch.Tensor | None = None
        self._normalized_hessian: torch.Tensor | None = None
        self.num_samples = 0

    @torch.no_grad()
    def accumulate(self, input_tensor: torch.Tensor) -> None:
        """Accumulate ``XᵀX`` per block from an activation of shape ``(..., cin)``."""
        if not self.is_enabled:
            return
        # fp32 GEMM avoids bf16/fp16 precision loss; (cin, tokens) -> (n_blocks, bs, tokens).
        x = input_tensor.reshape(-1, self.cin).to(torch.float32).T
        x = x.reshape(self.num_blocks_per_cin, self.block_size, -1)
        hessian_batch = x @ x.transpose(-1, -2)
        if self.hessian_per_block is None:
            self.hessian_per_block = hessian_batch
        else:
            self.hessian_per_block += hessian_batch
        self.num_samples += input_tensor.numel() // self.cin

    def normalized_hessian(self) -> torch.Tensor | None:
        """Per-cin-block Hessian ``H / num_samples`` (``None`` if no samples).

        Shared by both the Triton fast path and the reference ``error_func`` so the two
        consume one tensor; cached because the accumulated buffer may be freed afterwards.
        """
        if (
            self._normalized_hessian is None
            and self.hessian_per_block is not None
            and self.num_samples
        ):
            self._normalized_hessian = self.hessian_per_block / self.num_samples
        return self._normalized_hessian

    def build_error_func(
        self, keep_buffer: bool = False
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None:
        """Hessian-weighted error function (``None`` if no samples).

        Frees the raw Hessian buffer unless ``keep_buffer`` (kept for debug inspection).
        """
        hessian = self.normalized_hessian()
        if hessian is None:
            return None
        cout = self.cout
        bs = self.block_size
        if not keep_buffer:
            self.hessian_per_block = None

        def local_hessian_error(x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
            original_shape = x.shape
            # Per-block weighted error: dw (cout,n,bs) · H (n,bs,bs) -> (cout,n).
            dw = (x - xq).view(cout, -1, bs)
            block_loss = torch.einsum("cnb,nbd,cnd->cn", dw, hessian, dw).reshape(-1)
            return block_loss.unsqueeze(-1).expand(-1, bs).reshape(original_shape)

        return local_hessian_error


def _warn_if_block_size_mismatch(weight_quantizer: TensorQuantizer, block_size: int, name: str):
    """Warn if the Hessian block_size differs from the quantizer's scale block (misaligns)."""
    block_sizes = getattr(weight_quantizer, "block_sizes", None)
    quant_block = block_sizes.get(-1) if block_sizes else None
    if quant_block is not None and quant_block != block_size:
        warn_rank_0(
            f"local_hessian: block_size ({block_size}) != quantizer scale block "
            f"({quant_block}) for {name}; Hessian weighting will not align with the scale blocks."
        )


def _warn_local_hessian_fallback(name, weight, weight_quantizer, block_size, warned: set):
    """Warn once per ``(name, cin)`` when a captured layer falls back to plain MSE."""
    if weight.dim() < 2:
        return
    cin = weight.shape[1]
    if (name, cin) in warned:
        return
    warned.add((name, cin))
    if cin % block_size != 0:
        warn_rank_0(
            f"local_hessian: {name} input features ({cin}) not divisible by block_size "
            f"({block_size}); falling back to plain MSE for these weights."
        )
    _warn_if_block_size_mismatch(weight_quantizer, block_size, name)


def _is_quant_fused_experts(module: nn.Module) -> bool:
    """Whether ``module`` is a converted HF fused-MoE-experts wrapper with per-expert quantizers."""
    first_proj_attr = getattr(module, "_first_proj_attr", "gate_up_proj")
    return hasattr(module, "_current_expert_idx") and hasattr(
        module, f"{first_proj_attr}_weight_quantizers"
    )


def _register_local_hessian_input_hooks(model, name_to_module, capture, block_size, warned):
    """Register forward hooks feeding each weight's input activations to ``capture``.

    Local-Hessian-specific (kept here rather than as a general ``QuantModule`` API): dense
    quantized linears hook the layer input; HF fused-MoE experts hook the shared input quantizers,
    keyed by the active expert (``_current_expert_idx``). Weights without a hook (conv,
    SequentialQuantizer, non-eager experts) fall back to plain MSE. Returns removable handles.
    """
    handles: list = []

    def _make_expert_hook(expert_module, weight_name, quantizers, enabled):
        def _expert_hook(_input_quantizer, args):
            if not args:
                return
            idx = expert_module._current_expert_idx
            if idx in enabled:
                # Read the weight fresh (valid under accelerate/FSDP re-materialization).
                capture(quantizers[idx], getattr(expert_module, weight_name)[idx], args[0])

        return _expert_hook

    for name, module in name_to_module.items():
        if is_quantized_linear(module) and isinstance(module.weight_quantizer, TensorQuantizer):
            with enable_weight_access_and_writeback(module, model, name_to_module):
                # ``weight`` may be absent (e.g. TE GroupedLinear exposes weight0..N, not weight);
                # such modules have no single 2-D weight to pair and fall back to plain MSE.
                weight = getattr(module, "weight", None)
                if weight is None or weight.dim() != 2 or not module.weight_quantizer.is_enabled:
                    continue
                _warn_local_hessian_fallback(
                    name, weight, module.weight_quantizer, block_size, warned
                )

            def _dense_hook(linear, args):
                if args:
                    capture(linear.weight_quantizer, linear.weight, args[0])

            handles.append(module.register_forward_pre_hook(_dense_hook))
        elif _is_quant_fused_experts(module):
            with enable_weight_access_and_writeback(module, model, name_to_module):
                first_proj_attr = getattr(module, "_first_proj_attr", "gate_up_proj")
                for weight_name, quantizers_name, input_q_name in (
                    (
                        first_proj_attr,
                        f"{first_proj_attr}_weight_quantizers",
                        f"{first_proj_attr}_input_quantizer",
                    ),
                    ("down_proj", "down_proj_weight_quantizers", "down_proj_input_quantizer"),
                ):
                    weight = getattr(module, weight_name, None)
                    quantizers = getattr(module, quantizers_name, None)
                    input_quantizer = getattr(module, input_q_name, None)
                    if weight is None or quantizers is None or input_quantizer is None:
                        continue
                    _warn_local_hessian_fallback(
                        f"{name}.{weight_name}", weight[0], quantizers[0], block_size, warned
                    )
                    # Snapshot which experts are enabled now, before the caching forward silences
                    # all weight quantizers — so we don't capture (and discard) disabled experts.
                    enabled = {i for i, q in enumerate(quantizers) if q.is_enabled}
                    handles.append(
                        input_quantizer.register_forward_pre_hook(
                            _make_expert_hook(module, weight_name, quantizers, enabled)
                        )
                    )
    return handles


@torch.no_grad()
def local_hessian_calibrate(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    distributed_sync: bool = True,
    step_size: float = 0.1,
    start_multiplier: float = 0.25,
    stop_multiplier: float = 4.0,
    fp8_scale_sweep: bool = True,
    block_size: int = 16,
    debug: bool = False,
    shared_states: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
):
    """Calibrate weight quantizers by minimizing the Hessian-weighted error.

    Minimizes ``(W - Wq)ᵀ H (W - Wq)`` with per-block Hessian ``H = ΣXᵀX`` (approximating the
    output error ``||WX - WqX||²``), built from a forward with weight fake-quant disabled
    (input quantizers untouched) and fed to :func:`mse_calibrate`'s weight search via ``error_func``.

    Like :func:`mse_calibrate`, TensorQuantizer weights are calibrated — with the Hessian
    metric where a weight pairs with its input activations (dense linears and HF fused-MoE
    experts), plain MSE otherwise. Other quantizer types (e.g. SequentialQuantizer) are
    unsupported and left at their max-calibrated scale.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model. Required for this algorithm.
        distributed_sync: Whether to sync amax across distributed processes.
        step_size: Step size for amax search (default: 0.1).
        start_multiplier: Starting multiplier for amax search (default: 0.25).
        stop_multiplier: Ending multiplier for amax search (default: 4.0).
        fp8_scale_sweep: If True, sweep over all 128 possible FP8 E4M3 scale values
            for NVFP4 per-block quantization (default: True).
        block_size: Block size for local Hessian computation (default: 16).
        debug: If True, retain the per-quantizer Hessian accumulators on the model
            (``model._local_hessian_accumulators``) for inspection.

    See :class:`LocalHessianCalibConfig <modelopt.torch.quantization.config.LocalHessianCalibConfig>`
    for details on the configuration options.
    """
    if forward_loop is None:
        warnings.warn("forward_loop must be provided for local_hessian; skipping local_hessian")
        return

    # Phase 1: max-calibrate (also bootstraps dead experts + promotes/syncs NVFP4 static).
    print_rank_0("local_hessian: Running max calibration for all quantizers...")
    max_calibrate(model, forward_loop, distributed_sync, shared_states=shared_states)

    name_to_module = dict(model.named_modules())

    # Hessians keyed by id(weight_quantizer); modules pair weights<->activations via the hook.
    accumulators: dict[int, _LocalHessianAccumulator] = {}

    def capture(weight_quantizer, weight, input_tensor):
        input_local = input_tensor.to_local() if hasattr(input_tensor, "to_local") else input_tensor
        acc = accumulators.get(id(weight_quantizer))
        if acc is None:
            acc = _LocalHessianAccumulator(weight.shape[0], weight.shape[1], block_size)
            accumulators[id(weight_quantizer)] = acc
        acc.accumulate(input_local)

    # Phase 2: capture each weight's input activations during a forward with weight fake-quant
    # disabled (so H = ΣXᵀX reflects full-precision weights); input quantizers are left as-is.
    warned: set = set()
    handles = _register_local_hessian_input_hooks(
        model, name_to_module, capture, block_size, warned
    )
    print_rank_0("local_hessian: Caching activations and computing local Hessian...")
    try:
        with set_quantizer_by_cfg_context(
            model, [{"quantizer_name": "*weight_quantizer", "enable": False}]
        ):
            forward_loop(model)
    finally:
        for handle in handles:
            handle.remove()

    # TODO(fridah-nv): the per-block Hessian is not synced across TP/DP ranks (max_calibrate's
    # amax sync runs before this), so refined amaxes can diverge. All-reduce Hessian / re-sync.
    if dist_is_initialized() and dist_size() > 1:
        warn_rank_0(
            "local_hessian: Hessian is not synced across ranks; refined weight amaxes may "
            "diverge under tensor/data parallelism. Treat local_hessian as single-rank for now."
        )

    # Phase 3: weight search. Build error_funcs first so build_error_func caches the normalized
    # Hessian (freeing the raw buffer) before normalized_hessian() reuses it; the fast path
    # (tensor) and reference fallback (error_func) then share that one tensor.
    error_funcs = {
        qid: acc.build_error_func(keep_buffer=debug) for qid, acc in accumulators.items()
    }
    hessians = {qid: acc.normalized_hessian() for qid, acc in accumulators.items()}
    print_rank_0("local_hessian: Running MSE calibration with local Hessian loss...")
    _mse_calibrate_weights(
        model,
        name_to_module,
        step_size=step_size,
        start_multiplier=start_multiplier,
        stop_multiplier=stop_multiplier,
        fp8_scale_sweep=fp8_scale_sweep,
        error_func_for=lambda q: error_funcs.get(id(q)),
        hessian_for=lambda q: hessians.get(id(q)),
    )

    # Release the per-block Hessians (held by the error_func closures, calibrators, and the
    # accumulators' cache) before empty_cache so export starts defragmented; keep only for debug.
    error_funcs.clear()
    hessians.clear()
    for module in name_to_module.values():
        if isinstance(module, TensorQuantizer) and isinstance(module._calibrator, MseCalibrator):
            module._calibrator._error_func = None
            if isinstance(module._calibrator, NVFP4MSECalibrator):
                module._calibrator._hessian = None
    if debug:
        model._local_hessian_accumulators = accumulators
    else:
        accumulators.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print_rank_0("local_hessian: Calibration complete.")


def enable_stats_collection(model: nn.Module):
    """Enable stats collection for all quantizers in the model."""
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and not module._disabled:
            if module._use_constant_amax:
                # use_constant_amax quantizers use a fixed amax and don't need calibration.
                # Disable quantization during calibration so it doesn't affect other quantizers.
                module.disable_quant()
                continue
            elif module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()


def finish_stats_collection(model: nn.Module, method: str | None = None, **kwargs):
    """Finish stats collection for all quantizers in the model."""
    for _, module in model.named_modules():
        if not isinstance(module, TensorQuantizer) or module._disabled:
            continue

        if module._use_constant_amax:
            # Re-enable quantization for use_constant_amax quantizers disabled in enable_stats_collection.
            module.enable_quant()
            continue

        cal = getattr(module, "_calibrator", None)
        if cal and not getattr(module, "_dynamic", False):
            if method == "entropy":
                if cal.compute_amax(method) is not None:
                    module.load_calib_amax("entropy", **kwargs)
            elif cal.compute_amax(**kwargs) is not None:
                module.load_calib_amax(**kwargs)

        if module.bias_calibrator is not None and module.bias_type == "static":
            module.load_calib_bias()

        module.enable_quant()
        module.disable_calib()


@torch.no_grad()
def disable_pre_quant_scale_and_resmooth(linear: nn.Module, delete_pre_quant_scale: bool = False):
    """Disable pre_quant_scale and resmooth the quantized linear weights."""
    assert is_quantized_linear(linear), "Only quantized linear modules are supported"
    assert linear.input_quantizer._enable_pre_quant_scale, (
        "pre_quant_scale should be enabled first!"
    )
    assert hasattr(linear.input_quantizer, "_pre_quant_scale"), (
        "pre_quant_scale should be available"
    )

    pre_quant_scale = linear.input_quantizer._pre_quant_scale.to(torch.float32)

    linear.weight.copy_(
        (linear.weight * pre_quant_scale.squeeze()[None, :]).to(linear.weight.dtype)
    )
    linear.weight_quantizer.reset_amax()
    max_calibrate(linear, lambda linear: linear.weight_quantizer(linear.weight))

    # Lets not delete the _pre_quant_scale, it might useful later; Instead we will disable it
    linear.input_quantizer._enable_pre_quant_scale = False

    if linear.input_quantizer.amax is not None:
        assert hasattr(linear.input_quantizer, "_amax_for_smoothing")
        device, dtype = linear.weight.device, linear.weight.dtype
        linear.input_quantizer.amax = linear.input_quantizer._amax_for_smoothing.amax().to(
            device=device, dtype=dtype
        )

    if delete_pre_quant_scale:
        delattr(linear.input_quantizer, "_pre_quant_scale")
        linear.input_quantizer._enable_pre_quant_scale = False


# A global variable used during auto_quantize to avoid folding pre_quant_scale to weights
_ENABLE_FOLDING_PQS_TO_WEIGHTS = True


@torch.no_grad()
def _apply_weight_pre_quant_scale(linear, pre_quant_scale):
    if _ENABLE_FOLDING_PQS_TO_WEIGHTS:
        linear.weight.data.copy_(
            (linear.weight * pre_quant_scale.to(linear.weight.device).squeeze()[None, :]).to(
                linear.weight.dtype
            )
        )
    else:
        linear.weight_quantizer._enable_pre_quant_scale = True
        linear.weight_quantizer.pre_quant_scale = pre_quant_scale.squeeze()[None, :].to(
            linear.weight.dtype
        )

    linear.weight_quantizer.reset_amax()
    max_calibrate(linear, lambda linear: linear.weight_quantizer(linear.weight))


@torch.no_grad()
def apply_pre_quant_scale_and_smooth(
    linear: nn.Module, pre_quant_scale: torch.Tensor | None = None
):
    """Apply pre_quant_scale and smooth the quantized linear weights.

    If pre_quant_scale is not provided, the existing pre_quant_scale of input_quantizer will be used.
    """
    assert is_quantized_linear(linear), "Only quantized linear modules are supported"
    assert linear.input_quantizer.pre_quant_scale is None, "pre_quant_scale should be None first!"

    if pre_quant_scale is None:
        pre_quant_scale = (
            linear.input_quantizer._pre_quant_scale
            if hasattr(linear.input_quantizer, "_pre_quant_scale")
            else None
        )

    assert pre_quant_scale is not None, "pre_quant_scale should be provided or already set"

    assert torch.all(pre_quant_scale > 0), "pre_quant_scale should be positive"

    # pre_quant_scale should be in fp32 for the scaling math to be numerically safe
    pre_quant_scale = pre_quant_scale.to(torch.float32)

    linear.input_quantizer._enable_pre_quant_scale = True
    linear.input_quantizer.pre_quant_scale = pre_quant_scale.to(linear.weight.dtype)

    inv_scale = 1.0 / pre_quant_scale
    _apply_weight_pre_quant_scale(linear, inv_scale)

    if linear.input_quantizer.amax is not None:
        assert hasattr(linear.input_quantizer, "_amax_for_smoothing")
        device, dtype = linear.weight.device, linear.weight.dtype
        _amax_for_smoothing = linear.input_quantizer._amax_for_smoothing.to(
            device=device, dtype=dtype
        )
        linear.input_quantizer.amax = (
            (_amax_for_smoothing * pre_quant_scale.to(device)).amax().to(dtype)
        )

        if is_quantized_column_parallel_linear(linear) or is_quantized_row_parallel_linear(linear):
            linear.input_quantizer.sync_amax_across_distributed_group(
                linear.parallel_state.tensor_parallel_group
            )


@torch.no_grad()
def smoothquant(model: nn.Module, forward_loop: ForwardLoop | None = None, alpha=1.0):
    """Smooth-Quant variant with per-channel weight scaling.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.

    See :class:`SmoothQuantCalibConfig <modelopt.torch.quantization.config.SmoothQuantCalibConfig>` for
    details on the remaining arguments.
    """
    # distributed synchronization
    # max_calibrate performs amax sync for data parallel

    # Column parallel:
    # activations:  TPG should have the same pre_quant_scale
    #               This is achieved by syncing act_amax and weight_scale across TPG which is used to
    #               compute pre_quant_scale
    # weights:      no-op

    # Row parallel:
    # activations:  TPG should have same activation amax
    # weights:      TPG should have the same weight amax

    assert forward_loop is not None, "forward_loop must be provided for smoothquant"
    for name, module in model.named_modules():
        if (
            is_quantized_linear(module)
            and module.input_quantizer.is_enabled
            and module.input_quantizer.axis is None
        ):
            module.input_quantizer.axis = -1

    max_calibrate(model, forward_loop)

    def postprocess(module):
        # It is important to keep scaling math in fp32 to be numerically safe
        act_amax = module.input_quantizer.amax.float()
        weight_scale = module.weight.abs().amax(dim=0, keepdim=True)
        device, dtype = module.weight.device, module.weight.dtype

        parallel_group = module.parallel_state.tensor_parallel_group
        if is_quantized_column_parallel_linear(module) and parallel_group.is_initialized():
            dist.all_reduce(act_amax, op=dist.ReduceOp.MAX, group=parallel_group.group)
            dist.all_reduce(weight_scale, op=dist.ReduceOp.MAX, group=parallel_group.group)

        scale_a = (weight_scale.pow(1 - alpha) / act_amax.pow(alpha)).squeeze()

        # Now that activation per-channel amax have been collected, use per-tensor quantization for activation
        # TODO: make this a buffer after we support only heterogeneous checkpointing for MCore
        module.input_quantizer._amax_for_smoothing = act_amax.cpu()
        module.input_quantizer.reset_amax()
        module.input_quantizer.axis = None
        module.input_quantizer.amax = act_amax.amax().to(dtype=dtype, device=device)

        # Some channel could have 0 amax which causes scale_a to overflow. Explicitly mask them out here
        epsilon = 1.0 / (1 << 31)
        if scale_a.min() <= epsilon:
            zero_mask = act_amax <= epsilon
            scale_a[zero_mask] = 1
        scale_a = scale_a.clamp(min=1e-4, max=1e4)
        apply_pre_quant_scale_and_smooth(module, scale_a)

    name_to_module = dict(model.named_modules())
    smoothed_modules = 0
    for name, module in name_to_module.items():
        if is_quantized_linear(module):
            if not hasattr(module.input_quantizer, "_amax"):
                warnings.warn(f"{name} is not calibrated, skip smoothing")
                continue
            if module.input_quantizer.num_bits != 8 or module.weight_quantizer.num_bits != 8:
                warnings.warn(f"Only int8 smoothing is supported, skip {name}")
                continue
            if module.input_quantizer.axis != -1:
                warnings.warn(f"Only per-channel smoothing is supported, skip {name}")
                continue

            assert module.input_quantizer._amax.numel() > 1, (
                f"Error: {name} has only one channel to smooth"
            )

            with enable_weight_access_and_writeback(module, model, name_to_module):
                postprocess(module)

            smoothed_modules += 1
    print_rank_0(f"Smoothed {smoothed_modules} modules")


def awq(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    algorithm: str = "awq_lite",
    **kwargs,
):
    """Apply AWQ to the model.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.

    See :class:`AWQFullCalibConfig <modelopt.torch.quantization.config.AWQFullCalibConfig>` for
    details on the remaining arguments.
    """
    with SequentialQuantizer.convert_to_single_quantizer(model):
        if algorithm in ["awq_full", "awq_lite"]:
            awq_lite(model, forward_loop, **kwargs)

        if algorithm in ["awq_full", "awq_clip"]:
            awq_clip(model, forward_loop, **kwargs)

    # Special handling for SequentialQuantizer
    # Pre-compute name_to_module dict to avoid O(n^2) complexity in enable_weight_access_and_writeback
    name_to_module = dict(model.named_modules())
    for name, module in model.named_modules():
        if is_quantized_linear(module) and isinstance(module.weight_quantizer, SequentialQuantizer):
            with enable_weight_access_and_writeback(module, model, name_to_module):
                max_calibrate(module, lambda linear: linear.weight_quantizer(module.weight))


@torch.no_grad()
def awq_lite(
    model: nn.Module,
    forward_loop: ForwardLoop,
    alpha_step: float = 0.1,
    debug: bool = False,
    **kwargs,
):
    """Lite version of AWQ.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.

    See :class:`AWQLiteCalibConfig <modelopt.torch.quantization.config.AWQLiteCalibConfig>` for
    details on the remaining arguments.
    """
    if forward_loop is None:
        warnings.warn("forward_loop must be provided for awq_lite; skipping awq_lite")
        return

    class AWQLiteHelper:
        cache_mode: bool = False

        def __init__(self, module, name):
            self.name = name
            self.act_scale = 0.0
            self.num_cache_steps = 0
            self.num_search_steps = 0
            self.block_size = _get_awq_quantizer_block_size(module.weight, module.weight_quantizer)
            self.weight_scale = get_weight_scale(module.weight, self.block_size)
            self.loss = {
                k.item(): torch.zeros((), device=module.weight.device, dtype=torch.float32)
                for k in torch.arange(0, 1.0 + alpha_step, alpha_step)
            }
            self.best_scale = None
            self.best_alpha = None
            self.is_input_quantized = module.input_quantizer.is_enabled
            self.num_tokens = 0
            self.module = module
            self.is_enabled = True

        def setup(self):
            module = self.module
            bind_forward_method(module, forward, "_forward_no_awq")
            if module.input_quantizer.is_enabled:
                module.input_quantizer.disable()
                if module.input_quantizer.axis not in [None, -1]:
                    self.is_enabled = False
                    return
                module.input_quantizer.axis = -1

        def cleanup(self):
            module = self.module
            if hasattr(module, "_if_calib"):
                delattr(module, "_if_calib")
            unpatch_forward_method(module, "_forward_no_awq")

    def get_weight_scale(weight, block_size=None):
        org_shape = weight.shape
        slice_after_padding = None
        if block_size:
            if org_shape[-1] % block_size != 0:
                slice_after_padding = slice(org_shape[-1])
                weight = F.pad(weight, (0, block_size - org_shape[-1] % block_size), "constant", 0)
                org_shape = weight.shape
            weight = weight.contiguous().view(-1, block_size)
        weight_abs = weight.abs()  # Cache to avoid redundant computation
        weight_abs_amax = weight_abs.amax(dim=1, keepdim=True)
        scale = weight_abs / (weight_abs_amax + torch.finfo(weight.dtype).tiny)
        scale = scale.view(org_shape)
        if slice_after_padding is not None:
            scale = scale[..., slice_after_padding]
        scale = scale.mean(0).to(torch.float32)
        return scale

    def get_act_scale(x):
        return x.abs().contiguous().view(-1, x.shape[-1]).mean(0).to(torch.float32)

    def get_scale(x_max, w_max, alpha, tensor_parallel_group=None):
        scales = (
            (
                x_max.pow(alpha)
                / (w_max.to(x_max.device).pow(1 - alpha) + torch.finfo(torch.float32).tiny)
            )
            .clamp(min=1e-4, max=1e4)
            .view(-1)
        )
        scales = (scales / (scales.max() * scales.min()).sqrt()).view(-1)
        if tensor_parallel_group and tensor_parallel_group.is_initialized():
            dist.all_reduce(scales, op=dist.ReduceOp.SUM, group=tensor_parallel_group.group)
            scales /= tensor_parallel_group.world_size()
        return scales

    def update_loss(self, out, out_actual, alpha):
        out_actual = out_actual[0] if isinstance(out_actual, tuple) else out_actual
        out = out[0] if isinstance(out, tuple) else out
        out = out.to_local() if hasattr(out, "to_local") else out
        out_actual = out_actual.to_local() if hasattr(out_actual, "to_local") else out_actual
        loss = (out - out_actual).float().pow(2).mean()
        self.awq_lite.loss[alpha] += loss.to(self.awq_lite.loss[alpha].device)

    def update_best_params(self):
        if not self.awq_lite.is_enabled:
            return
        self.awq_lite.loss.update({k: float(v) for k, v in self.awq_lite.loss.items()})
        self.awq_lite.best_alpha = min(self.awq_lite.loss, key=self.awq_lite.loss.get)
        self.awq_lite.best_scale = get_scale(
            self.awq_lite.act_scale,
            self.awq_lite.weight_scale,
            self.awq_lite.best_alpha,
            (
                self.parallel_state.tensor_parallel_group
                if is_quantized_column_parallel_linear(self)
                else None
            ),
        )

    def forward(self, input, *args, **kwargs):
        # Collect actual output without quantization
        self.weight_quantizer.disable()
        if hasattr(self.input_quantizer, "_pre_quant_scale"):
            delattr(self.input_quantizer, "_pre_quant_scale")
        if hasattr(self.weight_quantizer, "_pre_quant_scale"):
            delattr(self.weight_quantizer, "_pre_quant_scale")
        out_actual = self._forward_no_awq(input, *args, **kwargs)
        self.weight_quantizer.enable()

        if input.numel() == 0 or not self.awq_lite.is_enabled:
            # For MoEs, some experts might see 0 tokens
            return out_actual

        if AWQLiteHelper.cache_mode:
            # Get local tensor from Dtensor
            input = input.to_local() if hasattr(input, "to_local") else input

            self.awq_lite.act_scale += get_act_scale(self.input_quantizer(input))
            self.awq_lite.num_cache_steps += 1
            self.awq_lite.num_tokens += input.numel() / input.shape[-1]
            if self.awq_lite.is_input_quantized:
                with set_quantizer_by_cfg_context(
                    self.input_quantizer, [{"quantizer_name": "*", "enable": True}]
                ):
                    max_calibrate(self.input_quantizer, lambda quantizer: quantizer(input), False)
            return out_actual

        for alpha in self.awq_lite.loss:
            awq_scale = get_scale(
                self.awq_lite.act_scale,
                self.awq_lite.weight_scale,
                alpha,
                (
                    self.parallel_state.tensor_parallel_group
                    if is_quantized_column_parallel_linear(self)
                    else None
                ),
            )
            self.input_quantizer.pre_quant_scale = (1 / awq_scale).to(self.weight.dtype)
            self.weight_quantizer.pre_quant_scale = awq_scale.to(self.weight.dtype)
            out = self._forward_no_awq(input, *args, **kwargs)
            update_loss(self, out, out_actual, alpha)

        self.awq_lite.num_search_steps += 1

        # Now forward the actual output without any quantization
        return out_actual

    # Pre-compute name_to_module dict ONCE to avoid O(n^2) complexity in enable_weight_access_and_writeback
    name_to_module = dict(model.named_modules())
    for name, module in name_to_module.items():
        if is_quantized_linear(module) and module.weight_quantizer.is_enabled:
            with enable_weight_access_and_writeback(module, model, name_to_module):
                module.awq_lite = AWQLiteHelper(module, name)
            module.awq_lite.setup()

    # Collect activation scale values
    AWQLiteHelper.cache_mode = True
    print_rank_0("awq_lite: Caching activation statistics...")

    # Lets enable stats collection
    # This will collect amax for input_quantizers and KV quantizers during the caching mode forward pass
    enable_stats_collection(model)
    forward_loop(model)

    # Load the amax values collected during the caching mode forward pass
    # This will also perform distributed amax sync for input_quantizers
    with set_quantizer_by_cfg_context(
        model,
        [{"quantizer_name": "*weight_quantizer", "enable": False}],
    ):
        max_calibrate(model, lambda model: None, distributed_sync=True)
    finish_stats_collection(model)

    def sync_act_scale_across_dp(module, data_parallel_group):
        """Sync activation scale across Data Parallel (DP)."""
        if data_parallel_group.is_initialized():
            dist.all_reduce(
                module.awq_lite.act_scale, op=dist.ReduceOp.AVG, group=data_parallel_group.group
            )

    for name, module in model.named_modules():
        if (
            is_quantized_linear(module)
            and hasattr(module, "awq_lite")
            and module.awq_lite.num_cache_steps > 0
        ):
            # Hack: MoEs forward all tokens through all experts if _if_calib is True
            module._if_calib = True
            module.awq_lite.act_scale = module.awq_lite.act_scale / module.awq_lite.num_cache_steps

            has_nan_local = torch.any(torch.isnan(module.awq_lite.act_scale)) or torch.any(
                torch.isnan(module.awq_lite.weight_scale)
            )
            has_nan = DistributedProcessGroup.get_dist_syncd_obj(
                has_nan_local, module.parallel_state.data_parallel_group, lambda objs: any(objs)
            )

            if has_nan:
                module.awq_lite.is_enabled = False
            else:
                sync_act_scale_across_dp(
                    module,
                    module.parallel_state.data_parallel_group,
                )

    # Disable AWQ search for uncalibrated experts (num_cache_steps == 0) to
    # prevent get_scale() crash on float act_scale. Max calibration and neutral
    # pre_quant_scale are applied in the postprocessing loop below.
    for name, module in model.named_modules():
        if (
            is_quantized_linear(module)
            and hasattr(module, "awq_lite")
            and module.awq_lite.num_cache_steps == 0
        ):
            module.awq_lite.is_enabled = False

    AWQLiteHelper.cache_mode = False
    print_rank_0("awq_lite: Searching parameters...")
    with torch.no_grad():
        forward_loop(model)

    def postprocess(module, name):
        update_best_params(module)
        if hasattr(module.weight_quantizer, "_pre_quant_scale"):
            delattr(module.weight_quantizer, "_pre_quant_scale")
        if hasattr(module.input_quantizer, "_pre_quant_scale"):
            delattr(module.input_quantizer, "_pre_quant_scale")
        if module.awq_lite.is_input_quantized:
            if module.input_quantizer.amax is not None:
                act_amax = module.input_quantizer.amax
                # TODO: make this a buffer after we support only heterogeneous checkpointing for MCore
                module.input_quantizer._amax_for_smoothing = act_amax.cpu()
                module.input_quantizer.reset_amax()
                module.input_quantizer.axis = None
                module.input_quantizer.amax = act_amax.amax()
                module.input_quantizer.enable()
            # for dynamic quantization, there is no amax, so we just enable the quantizer
            else:
                module.input_quantizer.enable()

        if module.awq_lite.is_enabled:
            apply_pre_quant_scale_and_smooth(module, 1.0 / module.awq_lite.best_scale)
        else:
            warnings.warn(f"awq_lite: Disabling for {name}, quantizing with max calibration.")
            max_calibrate(module, lambda module: module.weight_quantizer(module.weight))

    for name, module in model.named_modules():
        if hasattr(module, "awq_lite"):
            # Flag modules whose search pass missed them despite cache hits, so
            # they fall through to the neutral-scale path below.
            if module.awq_lite.num_cache_steps > 0 and module.awq_lite.num_search_steps == 0:
                module.awq_lite.is_enabled = False
                warnings.warn(
                    "awq_lite: Calling `forward_loop(model)` the second time did not forward"
                    f" data through the {name}. Please provide a valid `forward_loop` function"
                    " that can be used to forward data through the model many times."
                )

            if not module.awq_lite.is_enabled:
                # Expert is disabled — uncalibrated (no cache-pass tokens, set
                # at the pre-search pass above), had NaN in act/weight scales,
                # or saw no search-pass tokens. Max-calibrate weights and apply
                # a neutral (all-ones) pre_quant_scale so the exporter sees a
                # consistent nvfp4_awq format across all expert linears in an
                # MoE group.
                # NOTE: ones-scale must be registered OUTSIDE enable_weight_access_and_writeback
                # because HF accelerate post_forward drops newly-registered submodule buffers.
                warnings.warn(
                    f"awq_lite: Forcing pre_quant_scale=1 for {name} because the expert "
                    "was not properly exercised during calibration. This may degrade accuracy; "
                    "consider increasing calibration size or using a more diverse dataset."
                )
                with enable_weight_access_and_writeback(module, model, name_to_module):
                    max_calibrate(module, lambda module: module.weight_quantizer(module.weight))
                    w_shape, w_dtype, w_device = (
                        module.weight.shape[1],
                        module.weight.dtype,
                        module.weight.device,
                    )
                module.input_quantizer._enable_pre_quant_scale = True
                module.input_quantizer.pre_quant_scale = torch.ones(
                    w_shape,
                    dtype=w_dtype,
                    device=w_device,
                )
                # Mirror the calibrated postprocess path, gated on
                # is_input_quantized so weight-only AWQ configs (where
                # setup() never disabled input_quantizer) stay untouched.
                # Collapse any per-channel _amax left over from cache_mode
                # max_calibrate into a per-tensor scalar so
                # preprocess_linear_fusion's numel==1 assertion passes, and
                # re-enable the quantizer (awq_lite.setup disabled it).
                if module.awq_lite.is_input_quantized:
                    if module.input_quantizer.amax is not None:
                        act_amax = module.input_quantizer.amax
                        module.input_quantizer._amax_for_smoothing = act_amax.cpu()
                        module.input_quantizer.reset_amax()
                        module.input_quantizer.axis = None
                        module.input_quantizer.amax = act_amax.amax()
                    module.input_quantizer.enable()
            else:
                with enable_weight_access_and_writeback(module, model, name_to_module):
                    postprocess(module, name)

            module.awq_lite.cleanup()
            if not debug:
                delattr(module, "awq_lite")


@torch.no_grad()
def awq_clip(
    model: nn.Module,
    forward_loop: ForwardLoop,
    max_co_batch_size: int = 1024,
    max_tokens_per_batch: int = 64,
    min_clip_ratio: float = 0.5,
    shrink_step: float = 0.05,
    debug: bool = False,
    **kwargs,
):
    """AWQ-Clip variant.

    Args:
        model: Model to calibrate.
        forward_loop: A callable that runs the forward pass of the model.

    See :class:`AWQClipCalibConfig <modelopt.torch.quantization.config.AWQClipCalibConfig>` for
    details on the remaining arguments.
    """
    assert forward_loop is not None, "forward_loop must be provided for awq_clip"

    class AWQClipHelper:
        def __init__(self, module):
            self.num_tokens = 0
            self.block_size = _get_awq_quantizer_block_size(module.weight, module.weight_quantizer)

            # Cache the original amax
            module.weight_quantizer.reset_amax()
            enable_stats_collection(module.weight_quantizer)
            module.weight_quantizer(module.weight)
            finish_stats_collection(module.weight_quantizer)
            self.w_amax = module.weight_quantizer.amax.clone()

            co, ci = module.weight.shape
            clip_ratios = [
                round(float(k), 2) for k in torch.arange(min_clip_ratio, 1.0, shrink_step)
            ] + [1.0]
            if self.is_per_tensor_clip(module):
                self.loss = {k: torch.tensor(0.0, device=module.weight.device) for k in clip_ratios}
            else:
                self.loss = {
                    k: torch.zeros(
                        (co, math.ceil(ci / self.block_size)),
                        device=module.weight.device,
                    )
                    for k in clip_ratios
                }
            self.best_clip_val = None
            self.best_loss = None

            self.is_input_quantized = module.input_quantizer.is_enabled
            module.weight_quantizer.disable()

        def is_per_tensor_clip(self, module):
            quantizer = module.weight_quantizer
            is_dynamic_w_per_tensor = (
                hasattr(quantizer, "block_sizes")
                and quantizer.block_sizes.get("type", None) == "dynamic"
                and quantizer.axis is None
            )
            is_per_tensor = quantizer.axis is None and quantizer.block_sizes is None
            return is_dynamic_w_per_tensor or is_per_tensor

    def update_best_params(self):
        self.awq_clip.best_loss = torch.ones_like(self.awq_clip.w_amax) * float("inf")
        self.awq_clip.best_clip_val = torch.zeros_like(self.awq_clip.w_amax)

        for shrink, loss in self.awq_clip.loss.items():
            loss = loss.view_as(self.awq_clip.w_amax)
            indices = loss < self.awq_clip.best_loss
            self.awq_clip.best_loss = torch.where(indices, loss, self.awq_clip.best_loss)
            self.awq_clip.best_clip_val = torch.where(
                indices, self.awq_clip.w_amax * shrink, self.awq_clip.best_clip_val
            )

    def _clip_search(self, inputs, co_bsz=256, max_tokens=16):
        weight = self.weight
        self.weight_quantizer.enable()

        if self.awq_clip.is_per_tensor_clip(self):
            # In NVFP4, only the per-tensor amax is clipped
            out_actual = inputs @ self.weight.T
            original_amax = self.weight_quantizer.amax.clone()
            self.awq_clip.num_tokens += inputs.shape[0]
            for shrink in self.awq_clip.loss:
                self.weight_quantizer.amax = original_amax * shrink
                out = inputs @ self.weight_quantizer(self.weight).T
                loss = (out - out_actual).float().pow(2).mean()
                self.awq_clip.loss[shrink] += loss
        else:
            # weight  [co, ci] -> [co, 1, n_block, block_size]
            # inputs  [..., ci] -> [1, max_tokens, n_block, block_size]

            inputs = inputs.view(-1, inputs.shape[-1])  # _, ci
            # Select max_tokens from the total input tokens of count batch * n_token
            inputs = inputs[0 :: max(1, inputs.shape[0] // max_tokens)]  # max_tokens, ci
            self.awq_clip.num_tokens += inputs.shape[0]

            block_size = self.awq_clip.block_size
            co, ci = weight.shape
            if ci % block_size != 0:
                weight = F.pad(weight, (0, block_size - ci % block_size), "constant", 0)
                inputs = F.pad(inputs, (0, block_size - ci % block_size), "constant", 0)
                ci = weight.shape[-1]

            weight = weight.reshape(co, 1, -1, block_size)  # co, 1, n_block, block_size

            # 1, max_tokens, n_block, block_size
            inputs = inputs.reshape(1, inputs.shape[0], -1, block_size)

            for co_batch in range(math.ceil(co / co_bsz)):
                w = weight[co_batch * co_bsz : min((co_batch + 1) * co_bsz, co)]

                org_out = (inputs * w).sum(dim=-1)  # co_bsz, max_tokens, n_block

                for shrink in self.awq_clip.loss:
                    self.weight_quantizer.amax = self.awq_clip.w_amax * shrink
                    quantized_clipped_weight = self.weight_quantizer(self.weight)
                    cur_w = quantized_clipped_weight[
                        co_batch * co_bsz : min((co_batch + 1) * co_bsz, co)
                    ]
                    if cur_w.shape[-1] % block_size != 0:
                        cur_w = F.pad(
                            cur_w,
                            (0, block_size - cur_w.shape[-1] % block_size),
                            "constant",
                            0,
                        )
                    cur_w = cur_w.reshape(w.shape)
                    cur_out = (inputs * cur_w).sum(dim=-1)  # co_bsz, max_tokens, n_block

                    # co_bsz, n_block
                    loss = (cur_out - org_out).float().pow(2).mean(dim=1)

                    parallel_group = self.parallel_state.data_parallel_group
                    if parallel_group.is_initialized():
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=parallel_group.group)
                        loss /= parallel_group.world_size()

                    del cur_out, cur_w
                    self.awq_clip.loss[shrink][
                        co_batch * co_bsz : min((co_batch + 1) * co_bsz, co)
                    ] += loss
                del org_out

    def forward(name, self, input, *args, **kwargs):
        # input shape : (..., cin)
        # weight shape : (cout, cin)
        if self.awq_clip.is_input_quantized:
            self.input_quantizer.enable()
            max_calibrate(self.input_quantizer, lambda input_quantizer: input_quantizer(input))
            self.input_quantizer.disable()
        try:
            _clip_search(
                self,
                self.input_quantizer(input),
                max_co_batch_size,
                max_tokens_per_batch,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                raise RuntimeError(
                    f"Clip search on {name} failed due to CUDA out of memory, try reducing"
                    " max_co_batch_size"
                ) from e
            raise RuntimeError(e)

        # Disable quantization
        self.weight_quantizer.disable()
        return self._forward_no_awq(input, *args, **kwargs)

    # Pre-compute name_to_module dict to avoid O(n^2) complexity in enable_weight_access_and_writeback
    name_to_module = dict(model.named_modules())
    for name, module in model.named_modules():
        if (
            is_quantized_linear(module)
            and module.weight_quantizer.is_enabled
            and module.weight_quantizer.block_sizes is not None
        ):
            bind_forward_method(module, partial(forward, name), "_forward_no_awq")
            with enable_weight_access_and_writeback(module, model, name_to_module):
                module.awq_clip = AWQClipHelper(module)

    print_rank_0("awq_clip: Estimating parameters...")
    # Lets enable stats collection
    # This will collect amax for input_quantizers and KV quantizers during the caching mode forward pass
    enable_stats_collection(model)
    forward_loop(model)
    # Load the amax values collected during the caching mode forward pass
    # This will also perform distributed amax sync for input_quantizers
    with set_quantizer_by_cfg_context(
        model,
        [{"quantizer_name": "*weight_quantizer", "enable": False}],
    ):
        max_calibrate(model, lambda model: None, distributed_sync=True)
    finish_stats_collection(model)

    def postprocess(module):
        update_best_params(module)

        # Load the best clip value (amax)
        module.weight_quantizer.amax = module.awq_clip.best_clip_val
        module.weight_quantizer.enable()
        if module.awq_clip.is_input_quantized:
            module.input_quantizer.enable()

    for name, module in model.named_modules():
        if is_quantized_linear(module) and hasattr(module, "awq_clip"):
            if module.awq_clip.num_tokens > 0:
                with enable_weight_access_and_writeback(module, model, name_to_module):
                    postprocess(module)

            if not debug:
                delattr(module, "awq_clip")

            unpatch_forward_method(module, "_forward_no_awq")


def _get_awq_quantizer_block_size(tensor: torch.Tensor, quantizer: TensorQuantizer):
    if quantizer.block_sizes is None:
        return None
    if -1 in quantizer.block_sizes:
        blocksize = quantizer.block_sizes[-1]
    elif 1 in quantizer.block_sizes:
        blocksize = quantizer.block_sizes[1]
    else:
        raise ValueError("AWQ requires block quantization along -1 axis")
    return blocksize


def svd(weight, rank):
    original_device = weight.device
    original_dtype = weight.dtype
    weight_f64 = weight.to(dtype=torch.float64, device=original_device)
    u, s, vt = torch.linalg.svd(weight_f64, full_matrices=False)
    us = u[:, :rank] * s[:rank]
    vt = vt[:rank]
    us = us.to(device=original_device, dtype=original_dtype)
    vt = vt.to(device=original_device, dtype=original_dtype)
    if us.shape[1] < rank or vt.shape[0] < rank:
        warnings.warn(
            "The low-rank dimensions do not match the layer dimensions. "
            "Please verify your configuration and model settings. "
            f"Rank is {us.shape[1]} and {vt.shape[0]}"
        )
        us_temp = torch.zeros((us.shape[0], rank), dtype=us.dtype, device=us.device)
        vt_temp = torch.zeros((rank, vt.shape[1]), dtype=vt.dtype, device=vt.device)
        us_temp[:, : us.shape[1]] = us
        vt_temp[: vt.shape[0], :] = vt
        us = us_temp
        vt = vt_temp
    return us, vt


@torch.no_grad()
def svdquant(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    lowrank: int = 32,
    skip_layers: list[str] | None = None,
    **kwargs,
):
    """Lite version of SVDQuant.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.

    See :class:`SVDQuantConfig <modelopt.torch.quantization.config.SVDQuantConfig>` for
    details on the remaining arguments.
    """

    def is_skipped(name):
        return any(fnmatch.fnmatch(name, pattern) for pattern in skip_layers or [])

    def postprocess(module, name):
        print_rank_0(f"SVD {name}")
        weight = module.weight.data
        us, vt = svd(weight, lowrank)
        module.weight_quantizer.svdquant_lora_a = vt
        module.weight_quantizer.svdquant_lora_b = us
        module.weight.data.sub_(
            module.weight_quantizer.svdquant_lora_b @ module.weight_quantizer.svdquant_lora_a
        )
        module.weight_quantizer.reset_amax()
        module.input_quantizer.reset_amax()

    create_and_replace_svdquant_linear_on_the_fly(model=model)

    # Modules matching `skip_layers` opt out of the SVDQuant algorithm but stay
    # quantized: temporarily disable their quantizers so awq_lite neither smooths
    # their weights nor attaches a pre_quant_scale, then re-enable them so the
    # final max calibration collects their amax like a plain max recipe.
    skipped_quantizers = []
    if skip_layers:
        for name, module in model.named_modules():
            if (
                is_quantized_linear(module)
                and module.weight_quantizer.is_enabled
                and is_skipped(name)
            ):
                print_rank_0(f"SVDQuant skips {name}; quantizing with max calibration.")
                for quantizer in (module.weight_quantizer, module.input_quantizer):
                    if quantizer.is_enabled:
                        quantizer.disable()
                        skipped_quantizers.append(quantizer)

    awq(model, forward_loop, "awq_lite", **kwargs)

    for quantizer in skipped_quantizers:
        quantizer.enable()

    name_to_module = dict(model.named_modules())
    for name, module in name_to_module.items():
        if (
            is_quantized_linear(module)
            and module.weight_quantizer.is_enabled
            and not is_skipped(name)
        ):
            with enable_weight_access_and_writeback(module, model, name_to_module):
                postprocess(module, name)
    max_calibrate(model, forward_loop)


@torch.no_grad()
def layerwise_calibrate(
    model: nn.Module,
    forward_loop: ForwardLoop,
    calib_func: Callable,
    **calib_kwargs,
):
    """Layerwise calibration - a layer-by-layer calibration algorithm.

    Runs the full model forward per layer but patches decoder layers with a
    skip / run / capture strategy so that inter-layer logic in parent modules
    (e.g. mask construction) executes naturally without model-specific hooks.

    If ``checkpoint_dir`` is passed (via ``calib_kwargs``), per-layer checkpoints
    are saved after each layer completes. On restart, calibration resumes from
    the last completed layer.

    ``get_qdq_activations_from_prev_layer`` (via ``calib_kwargs``) controls
    whether the cached inputs handed to layer N+1 come from a forward through
    the just-calibrated layer with quantizers active (True; e.g. GPTQ) or
    temporarily disabled (False; matches non-layerwise max-calib semantics).
    """
    checkpoint_dir = calib_kwargs.pop("checkpoint_dir", None)
    qdq_from_prev = calib_kwargs.pop("get_qdq_activations_from_prev_layer", False)
    save_every = calib_kwargs.pop("save_every", 1)

    if forward_loop is None:
        raise ValueError(
            "forward_loop must not be None for layerwise calibration. "
            "Please provide a valid forward_loop callable."
        )

    transformer_layers = LayerActivationCollector.get_decoder_layers(model)
    if transformer_layers is None or len(transformer_layers) == 0:
        raise ValueError(
            "Could not find transformer layers in model. "
            "Layerwise calibration requires a model with identifiable transformer layers."
        )

    num_layers = len(transformer_layers)
    print_rank_0(f"Layerwise calibration: Found {num_layers} transformer layers")

    ckpt = _CheckpointState.from_folder(
        checkpoint_dir,
        num_layers,
        save_every=save_every,
    )
    start_layer = ckpt.start_layer if ckpt else 0

    input_getter = LayerActivationCollector(model)
    input_getter._patch_all_layers(decoder_layers=transformer_layers)

    resumed_inputs = ckpt.setup_resume(transformer_layers) if ckpt and start_layer > 0 else None

    try:
        # Bootstrap: get first layer's inputs (or use resumed inputs).
        layer_inputs = input_getter.get_first_layer_inputs(
            start_layer, resumed_inputs, forward_loop
        )

        for layer_idx in range(start_layer, num_layers):
            layer = transformer_layers[layer_idx]

            def _layer_forward_loop(m, _inputs=layer_inputs):
                for args, kwargs_input in _inputs:
                    # Reset past_key_values to prevent the KV cache from
                    # accumulating across multiple forward replays (e.g.
                    # max_calibrate then Hessian collection in GPTQ).
                    # The layer doesn't need stale KV data — each replay
                    # should start with a fresh cache.
                    if (
                        "past_key_values" in kwargs_input
                        and kwargs_input["past_key_values"] is not None
                    ):
                        kwargs_input = dict(kwargs_input)
                        cache = kwargs_input["past_key_values"]
                        if hasattr(cache, "reset"):
                            cache.reset()
                        else:
                            kwargs_input["past_key_values"] = None
                    m(*args, **kwargs_input)

            is_last = layer_idx + 1 >= num_layers

            with persistent_materialization(layer):
                # qdq_from_prev=False: capture before calib_func so the forward
                # replay uses the original FP weights. Disable quantizers too in
                # case any pre-calibration observer behavior would perturb the
                # captured activations.
                if not is_last and not qdq_from_prev:
                    with set_quantizer_by_cfg_context(
                        layer, [{"quantizer_name": "*", "enable": False}]
                    ):
                        next_inputs = input_getter.cache_outputs_for_next_layer_calib(
                            layer, forward_loop
                        )
                    # cache_outputs left this layer in "run" mode with an empty
                    # deque; reset so calib_func's replay hits the real forward.
                    layer._layerwise_calib.mode = "original"

                calib_func(layer, _layer_forward_loop, **calib_kwargs)

                # qdq_from_prev=True: capture after calib_func so the next layer
                # sees QDQ error and any in-place weight updates from this layer.
                if not is_last and qdq_from_prev:
                    next_inputs = input_getter.cache_outputs_for_next_layer_calib(
                        layer, forward_loop
                    )
                elif is_last:
                    next_inputs = None

                if ckpt:
                    ckpt.save(layer_idx, model, transformer_layers, next_inputs)

            del layer_inputs
            torch.cuda.empty_cache()
            layer_inputs = next_inputs  # noqa: F841 (used in next iteration's closure)
    finally:
        input_getter._unpatch_all_layers()

    if ckpt:
        ckpt.full_restore(transformer_layers, model)

    print_rank_0("Layerwise calibration completed")


@torch.no_grad()
def gptq(
    model: nn.Module,
    forward_loop: ForwardLoop,
    perc_damp: float = 0.01,
    block_size: int = 128,
    fused: bool = False,
):
    """GPTQ quantization.

    Works in two modes depending on ``layerwise.enable`` in the config:

    * **Layerwise** (``layerwise.enable=True``): ``layerwise_calibrate`` calls
      this function once per decoder layer with updated activations, producing
      more accurate Hessian estimates.
    * **Non-layerwise** (``layerwise.enable=False``): called once on the full
      model. All layers are quantized in parallel from the original activations.

    Per-module steps:

    1. ``max_calibrate`` to set amax values from the current activations.
    2. Promote eligible quantizers to ``NVFP4StaticQuantizer`` (two-level scaling).
    3. Collect per-linear-layer Hessian matrices via forward hooks.
    4. Blockwise weight updates using the inverse Hessian to compensate for
       rounding error (the core GPTQ column-wise update).

    Args:
        model: The module to quantize — either the full model or a single decoder
            layer when invoked by ``layerwise_calibrate``.
        forward_loop: Callable that replays calibration inputs through *model*.
        perc_damp: Percentage of avg Hessian diagonal for damping (default: 0.01).
        block_size: Block size for GPTQ weight update.
        fused: If True, use fused Triton kernel for NVFP4 static quantization.
    """
    total_start = time.time()

    # TODO: Add support for other scale setting strateiges like weight-mse or local-hessian
    max_calibrate(model, forward_loop=forward_loop)

    quantized_layers = [
        (n, m)
        for n, m in model.named_modules()
        if is_quantized_linear(m) and m.weight_quantizer.is_enabled
    ]
    if not quantized_layers:
        print_rank_0("No quantized linear layers found, skipping GPTQ")
        return

    def _make_gptq_handle(name, m):
        backend = getattr(m.weight_quantizer, "backend", None)
        if backend is None:
            cls = GPTQHelper
        else:
            cls = _GPTQ_HELPER_REGISTRY.get(backend, GPTQHelper)
        return cls(m, name, offload_to_cpu=True, fused=fused)

    gptq_handles = {name: _make_gptq_handle(name, m) for name, m in quantized_layers}
    for handle in gptq_handles.values():
        handle.setup()

    print_rank_0(f"Computing Hessians for {len(gptq_handles)} linear layers...")

    with set_quantizer_by_cfg_context(
        model, [{"quantizer_name": "*weight_quantizer", "enable": False}]
    ):
        forward_loop(model)

    for handle in gptq_handles.values():
        handle.cleanup()

    print_rank_0("Updating weights using GPTQ algorithm...")
    name_to_module = dict(model.named_modules())
    for handle in gptq_handles.values():
        with enable_weight_access_and_writeback(handle.module, model, name_to_module):
            handle.update_weights(block_size, perc_damp)
        handle.free()
    del gptq_handles

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_rank_0(f"GPTQ time: {time.time() - total_start:.2f}s")
