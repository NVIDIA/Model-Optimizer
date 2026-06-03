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

"""Custom vLLM worker that calibrates skip-softmax thresholds in-engine.

Unlike ``SparseAttnWorker`` (which reads an already-calibrated
``sparse_attention_config`` from the checkpoint and serves), this worker
*produces* that config. It force-swaps ``ModelOptSparseAttentionImpl`` onto
every attention layer and exposes RPC methods the driver
(``calibrate_sparse_attn.py``) calls via ``LLM.collective_rpc``:

- ``sparse_calib_enable``: put every layer's impl in calibration mode.
- ``sparse_calib_fit``: stop measuring, fit the exponential ``(a, b)`` model
  from the tile-skip stats collected during ``llm.generate``, and return an
  export-format ``sparse_attention_config`` dict.

Calibration uses the ModelOpt Triton calibration kernel through the paged KV
cache (see ``modelopt.torch.sparsity.attention_sparsity.plugins.vllm``), so the
numbers match the HF calibration path and drop straight into the existing
serving path.
"""

import importlib
from typing import Any

try:
    _has_legacy_attention_layer = importlib.util.find_spec("vllm.attention.layer") is not None
except (ModuleNotFoundError, ValueError):
    _has_legacy_attention_layer = False

if _has_legacy_attention_layer:
    from vllm.attention.layer import Attention as VLLMAttention
else:
    from vllm.model_executor.layers.attention import Attention as VLLMAttention

from vllm.v1.worker.gpu_worker import Worker as BaseWorker

import modelopt
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import (
    ModelOptSparseAttentionImpl,
    _clone_sparse_impl,
    disable_calibration,
    enable_calibration,
    fit_calibration,
    get_flashinfer_sparse_impl_cls,
    iter_sparse_impls,
    patch_flashinfer_metadata_builder,
)

# Default threshold sweep — should span sparsities from ~10% to ~95%.
DEFAULT_THRESHOLD_TRIALS = [
    1e-4,
    1e-3,
    5e-3,
    1e-2,
    3e-2,
    5e-2,
    1e-1,
    2e-1,
    3e-1,
    5e-1,
    7e-1,
    9e-1,
]


def _sparse_impl_cls_for(impl):
    """Pick the ModelOpt sparse impl class matching the layer's vLLM backend.

    Returns ``None`` for already-swapped or unsupported backends. The FlashInfer
    path also installs the metadata-builder patch that exposes the dense paged
    metadata the calibration kernel needs.
    """
    name = type(impl).__name__
    if name.startswith("ModelOptSparse"):
        return None  # already swapped (idempotent across reloads)
    if name == "FlashAttentionImpl":
        return ModelOptSparseAttentionImpl
    if name == "FlashInferImpl":
        if not patch_flashinfer_metadata_builder():
            return None
        return get_flashinfer_sparse_impl_cls()
    return None


def _force_replace_attention_impls(worker) -> int:
    """Swap the ModelOpt sparse impl onto every supported attention layer.

    Calibration has no checkpoint metadata to match against, so every attention
    layer is converted unconditionally (with empty ``sparse_kw``; calibration
    mode is toggled separately by ``sparse_calib_enable``). Supports the
    FlashAttention and FlashInfer backends; other backends are left untouched.
    """
    model = worker.model_runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()

    patched = 0
    skipped_backends: set[str] = set()
    for _, module in model.named_modules():
        if not isinstance(module, VLLMAttention):
            continue
        impl = module.impl
        new_cls = _sparse_impl_cls_for(impl)
        if new_cls is None:
            if not type(impl).__name__.startswith("ModelOptSparse"):
                skipped_backends.add(type(impl).__name__)
            continue
        try:
            new_impl = _clone_sparse_impl(impl, new_cls)
        except NotImplementedError:
            # e.g. FlashAttention sinks — leave those layers on vLLM's impl.
            skipped_backends.add(type(impl).__name__)
            continue
        new_impl.sparse_kw = {}
        module.impl = new_impl
        patched += 1
    print(f"[ModelOpt] Calibration: swapped impl on {patched} attention layers")
    if skipped_backends:
        print(
            f"[ModelOpt] Calibration: left {sorted(skipped_backends)} layers unchanged "
            "(unsupported backend — calibrate under FLASH_ATTN or FLASHINFER)."
        )
    return patched


class SparseAttnCalibWorker(BaseWorker):
    """vLLM worker that calibrates skip-softmax thresholds through the engine."""

    def load_model(self, *args, **kwargs) -> None:
        """Load the model, then force the sparse impl onto every attention layer."""
        super().load_model(*args, **kwargs)
        _force_replace_attention_impls(self)

    # -- RPC methods (invoked via LLM.collective_rpc) ----------------------

    def sparse_calib_status(self) -> dict[str, Any]:
        """Report which sparse impls are active and how many records each holds.

        Lets the driver confirm calibration actually routes through the expected
        backend (e.g. ``ModelOptSparseFlashInferImpl``) rather than a fallback.
        """
        impls = list(iter_sparse_impls(self.model_runner.model))
        impl_types: dict[str, int] = {}
        total_records = 0
        for impl in impls:
            impl_types[type(impl).__name__] = impl_types.get(type(impl).__name__, 0) + 1
            total_records += len(getattr(impl, "_calib_records", []))
        return {
            "num_sparse_layers": len(impls),
            "impl_types": impl_types,
            "calibrating": any(getattr(i, "_calibrate", False) for i in impls),
            "total_records": total_records,
        }

    def sparse_calib_enable(self, threshold_trials: list[float] | None = None) -> int:
        """Enter calibration mode on all sparse impls; returns layer count."""
        trials = threshold_trials or DEFAULT_THRESHOLD_TRIALS
        impls = list(iter_sparse_impls(self.model_runner.model))
        enable_calibration(impls, trials)
        return len(impls)

    def sparse_calib_fit(
        self,
        target_sparse_ratio: dict[str, float] | float = 0.5,
        threshold_trials: list[float] | None = None,
        fit_logspace: bool = False,
    ) -> dict[str, Any] | None:
        """Stop measuring, fit ``(a, b)``, and return an export-format config.

        Returns ``None`` if no phase produced a valid fit (e.g. too little data).
        """
        trials = threshold_trials or DEFAULT_THRESHOLD_TRIALS
        impls = list(iter_sparse_impls(self.model_runner.model))
        disable_calibration(impls)
        calibration_params = fit_calibration(impls, trials, fit_logspace=fit_logspace)
        if not calibration_params:
            return None
        return _build_sparse_attention_config(calibration_params, target_sparse_ratio)


def _normalize_target(target_sparse_ratio: dict[str, float] | float) -> dict[str, float]:
    if isinstance(target_sparse_ratio, (int, float)):
        return {"prefill": float(target_sparse_ratio), "decode": float(target_sparse_ratio)}
    return {
        "prefill": float(target_sparse_ratio.get("prefill", 0.5)),
        "decode": float(target_sparse_ratio.get("decode", 0.5)),
    }


def _build_sparse_attention_config(
    calibration_params: dict[str, dict[str, float]],
    target_sparse_ratio: dict[str, float] | float,
) -> dict[str, Any]:
    """Build the ``sparse_attention_config`` block consumed at serving time.

    Matches ``modelopt.torch.sparsity.attention_sparsity.conversion.
    export_sparse_attention_config`` so ``load_from_checkpoint_metadata`` (the
    serving path) recognizes it without changes.
    """
    threshold_scale_factor: dict[str, Any] = {"formula": "a * exp(b * target_sparsity)"}
    for phase in ("prefill", "decode"):
        if phase in calibration_params:
            threshold_scale_factor[phase] = {
                "a": calibration_params[phase]["a"],
                "b": calibration_params[phase]["b"],
            }
    return {
        "config_groups": {
            "group_0": {"sparse_algo": "softmax_skip", "targets": ["Attention"]},
        },
        "threshold_scale_factor": threshold_scale_factor,
        "target_sparse_ratio": _normalize_target(target_sparse_ratio),
        "producer": {"name": "modelopt", "version": modelopt.__version__},
    }
