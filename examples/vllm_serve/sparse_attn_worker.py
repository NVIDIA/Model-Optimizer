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

"""vLLM worker lifecycle wiring for ModelOpt attention transforms."""

from types import SimpleNamespace

from vllm.v1.worker.gpu_worker import Worker as BaseWorker

from modelopt.torch.sparsity.attention_sparsity.plugins.sparse_attn_calibration import (
    DEFAULT_THRESHOLD_TRIALS,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.sparse_attn_config import (
    load_from_checkpoint_metadata,
    match_sparse_config,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import (
    _build_sparse_kw,
    collect_calibration_counts,
    disable_calibration,
    enable_calibration,
    iter_sparse_impls,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm_runtime import (
    install_vllm_nvfp4_attention,
    install_vllm_skip_softmax_calibration,
    install_vllm_sparse_attention_from_checkpoint,
)

__all__ = ["SparseAttnWorker", "QuantSparseAttnWorker", "SkipSoftmaxCalibWorker"]  # noqa: RUF022

_QUANT_FORMAT_KEYS = ("q_format", "k_format", "p_format", "v_format")


def _unwrapped_model(worker):
    model = worker.model_runner.model
    return model.unwrap() if hasattr(model, "unwrap") else model


def _print_install_report(policy, report) -> None:
    if report.installed_count:
        if policy != "Sparse attention":
            print(
                f"[ModelOpt] Installed {policy} (quant+sparse) on "
                f"{report.installed_count} layers: {dict(report.backend_counts)}"
            )
        else:
            if report.sparse_algorithm:
                print(f"[ModelOpt] Sparse attention config: algo -> {report.sparse_algorithm}")
            print(
                f"[ModelOpt] Sparse attention: replaced impl on {report.installed_count} "
                f"attention layers: {dict(report.backend_counts)}"
            )
    elif report.sparse_algorithm:
        print(
            f"[ModelOpt] Sparse attention config {report.sparse_algorithm} matched no active "
            "attention layers; vLLM remains unchanged"
        )
    else:
        print(
            "[ModelOpt] No sparse_attention_config found in the checkpoint; "
            "skipping sparse attention. Run examples/llm_sparsity/attention_sparsity/"
            "hf_sa.py to calibrate and export a checkpoint with the config embedded."
        )


class SparseAttnWorker(BaseWorker):
    """Install checkpoint-driven sparse attention after model loading."""

    def load_model(self, *args, **kwargs) -> None:
        """Load the model, then install checkpoint-configured attention."""
        super().load_model(*args, **kwargs)
        report = install_vllm_sparse_attention_from_checkpoint(self.model_runner)
        _print_install_report("Sparse attention", report)


class SkipSoftmaxCalibWorker(BaseWorker):
    """Calibrate skip-softmax thresholds through the engine.

    Unlike :class:`SparseAttnWorker` (which serves an already-calibrated
    ``sparse_attention_config``), this worker *produces* that config. The
    library installer swaps calibration-capable adapters onto every attention
    layer at load; measurement starts only when the driver calls
    ``sparse_calib_enable`` (so warmup launches are never recorded) and raw
    per-threshold tile counts are harvested with ``sparse_calib_counts`` for
    the driver to aggregate across TP ranks and fit.
    """

    def load_model(self, *args, **kwargs) -> None:
        """Load the model, then install calibration adapters on every layer."""
        super().load_model(*args, **kwargs)
        report = install_vllm_skip_softmax_calibration(self.model_runner)
        print(
            f"[ModelOpt] Skip-softmax calibration installed on {report.installed_count} "
            f"attention layers: {dict(report.backend_counts)}"
        )

    # -- RPC methods (invoked via LLM.collective_rpc) ----------------------

    def sparse_calib_enable(self, threshold_trials: list[float] | None = None) -> int:
        """Enter calibration mode on all installed impls; returns layer count."""
        impls = list(iter_sparse_impls(_unwrapped_model(self)))
        enable_calibration(impls, list(threshold_trials or DEFAULT_THRESHOLD_TRIALS))
        return len(impls)

    def sparse_calib_status(self) -> dict:
        """Report active impls and record counts, so the backend is verifiable."""
        impls = list(iter_sparse_impls(_unwrapped_model(self)))
        impl_types: dict[str, int] = {}
        total_records = 0
        for impl in impls:
            impl_types[type(impl).__name__] = impl_types.get(type(impl).__name__, 0) + 1
            total_records += len(getattr(impl, "_calib_records", []))
        return {
            "num_sparse_layers": len(impls),
            "impl_types": impl_types,
            "calibrating": any(getattr(impl, "_calibrate", False) for impl in impls),
            "total_records": total_records,
        }

    def sparse_calib_counts(self) -> dict[str, list[dict]]:
        """Stop measuring and return this rank's layer-merged raw tile counts."""
        model = _unwrapped_model(self)
        disable_calibration(list(iter_sparse_impls(model)))
        return collect_calibration_counts(model)

    def sparse_validation_enable(self, sparse_config: dict) -> int:
        """Apply the candidate through the serving loader and reset tile counters."""
        model = _unwrapped_model(self)
        impls = list(iter_sparse_impls(model))
        detected = load_from_checkpoint_metadata(
            SimpleNamespace(sparse_attention_config=sparse_config)
        )
        if detected is None:
            raise ValueError("Candidate has no serving sparse attention configuration")
        resolved, _ = detected
        disable_calibration(impls)
        impl_ids = {id(impl) for impl in impls}
        selected = 0
        self._sparse_validation_phases = tuple(
            sparse_config["config_groups"]["group_0"]["target_sparsity"]
        )
        for name, module in model.named_modules():
            impl = getattr(module, "impl", None)
            if impl is None or id(impl) not in impl_ids:
                continue
            cfg = match_sparse_config(name, resolved)
            impl.sparse_kw = _build_sparse_kw(cfg) if cfg and cfg.get("enable", True) else {}
            active = "threshold_scale_factor" in impl.sparse_kw
            impl._sparse_validation_stats = {} if active else None
            selected += int(active)
        return selected

    def sparse_validation_collect(self) -> dict[str, dict[str, int]]:
        """Sum counters across selected layers, rejecting any missing layer/phase."""
        result: dict[str, dict[str, int]] = {}
        for impl in iter_sparse_impls(_unwrapped_model(self)):
            stats_by_phase = getattr(impl, "_sparse_validation_stats", None)
            if stats_by_phase is None:
                continue
            for phase in self._sparse_validation_phases:
                stats = stats_by_phase.get(phase, {})
                record = result.setdefault(
                    phase,
                    {
                        "total": 0,
                        "skipped": 0,
                        "launches": 0,
                        "unmeasured_launches": 0,
                        "min_seq_len": stats.get("min_seq_len", 0),
                    },
                )
                for key in ("total", "skipped", "launches", "unmeasured_launches"):
                    record[key] += stats.get(key, 0)
                if stats.get("launches", 0) == 0:
                    record["unmeasured_launches"] += 1
                record["min_seq_len"] = min(record["min_seq_len"], stats.get("min_seq_len", 0))
            impl._sparse_validation_stats = None
        return result


class QuantSparseAttnWorker(BaseWorker):
    """Install quantized attention plus optional checkpoint sparsity.

    Per-operand formats come from vLLM's ``--additional-config``; absent keys
    default to NVFP4 on all four operands (Q/K/P/V)::

        --additional-config '{"modelopt_attn_quant": {"p_format": "fp8", "v_format": "fp8"}}'
    """

    def _quant_formats(self) -> dict[str, str]:
        additional = getattr(self.vllm_config, "additional_config", None) or {}
        formats = additional.get("modelopt_attn_quant", {})
        unknown = set(formats) - set(_QUANT_FORMAT_KEYS)
        if unknown:
            raise ValueError(
                f"unknown modelopt_attn_quant keys {sorted(unknown)}; "
                f"allowed: {list(_QUANT_FORMAT_KEYS)}"
            )
        return dict(formats)

    def load_model(self, *args, **kwargs) -> None:
        """Load the model, then install the configured attention quant recipe."""
        super().load_model(*args, **kwargs)
        formats = self._quant_formats()
        report = install_vllm_nvfp4_attention(self.model_runner, sparse_cfg="checkpoint", **formats)
        policy = "NVFP4 attention" if not formats else f"Quant attention ({formats})"
        _print_install_report(policy, report)

    def determine_available_memory(self) -> int:
        """Profile memory without compiling the dynamically converted modules."""
        # Sparse-only imports must remain independent of quantization-specific APIs.
        import torch

        from modelopt.torch.quantization.plugins.vllm import disable_compilation

        with torch.inference_mode(), disable_compilation(_unwrapped_model(self)):
            return BaseWorker.determine_available_memory(self)
