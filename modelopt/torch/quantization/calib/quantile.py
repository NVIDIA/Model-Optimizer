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

"""Streaming P^2 quantile calibrator.

Estimates one or more quantiles of ``|activation|`` with O(1) memory per
quantile using the Jain & Chlamtac (1985) P^2 algorithm with the corrections
documented at https://aakinshin.net/vignettes/p2-quantile-estimator/.

The per-tensor amax produced by :class:`QuantileCalibrator` is taken as the
``max`` of the tracked quantile estimates. For NVFP4, this amax drives both
the FP32 per-tensor global scale and the FP8 representable range of the
per-block scales (see ``modelopt/torch/kernels/quantization/gemm/fp4_kernel_hopper.py``).
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

import numpy as np
import torch

from .calibrator import _Calibrator

__all__ = ["P2QuantileEstimator", "QuantileCalibrator", "save_quantile_data"]


DEFAULT_QUANTILES: tuple[float, ...] = (0.99, 0.999, 0.9999, 0.99999)

# Per-call subsample cap for ``QuantileCalibrator.collect``. Activation tensors
# in real models routinely contain millions of values per batch; the P^2
# marker walk is Python-level and scales with chunk size, so processing every
# scalar is infeasible. For the quantiles of interest (>= 0.99), uniform
# subsampling of O(10^4) values per call still leaves enough samples in the
# tail across a normal calibration set (256 batches x 16k = 4M draws) to
# estimate the quantile to within a few percent. Set to None to disable.
_DEFAULT_SUBSAMPLE_CAP: int = 16384


class P2QuantileEstimator:
    """Streaming P^2 quantile estimator for a single probability.

    Holds five markers ``q[0..4]`` and their integer ranks ``n[0..4]``.
    ``estimate()`` returns the current estimate of the ``p``-quantile,
    available after at least one value has been observed.
    """

    def __init__(self, p: float) -> None:
        """Initialise the estimator for probability ``p`` in (0, 1)."""
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be in (0, 1), got {p}")
        self.p: float = float(p)
        self._init_buffer: list[float] = []
        self._initialized: bool = False
        self.q: list[float] = [0.0] * 5
        self.n: list[float] = [0.0] * 5  # floats so n[i] += dInt stays float-safe
        self.count: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _initialize_from_buffer(self) -> None:
        """Initialise markers from the 5-value buffer.

        Uses the adaptive snap recommended by the vignette when the rounded
        ideal ranks ``[0, round(2p), round(4p), round(2+2p), 4]`` are strictly
        increasing; otherwise falls back to the canonical non-adaptive
        ``n = [0, 1, 2, 3, 4]`` initialisation. Adaptive snapping collapses
        interior markers to the same rank for extreme quantiles (e.g. p=0.999
        yields n[2]=n[3]=4), which would create zero denominators in the
        parabolic / linear update formulas.
        """
        sorted_buf = sorted(self._init_buffer)
        p = self.p
        n1 = max(0, min(4, round(2.0 * p)))
        n2 = max(0, min(4, round(4.0 * p)))
        n3 = max(0, min(4, round(2.0 + 2.0 * p)))
        if 0 < n1 < n2 < n3 < 4:
            self.q = list(sorted_buf)
            self.n = [0.0, float(n1), float(n2), float(n3), 4.0]
            self.q[1] = sorted_buf[n1]
            self.q[2] = sorted_buf[n2]
            self.q[3] = sorted_buf[n3]
        else:
            self.q = list(sorted_buf)
            self.n = [0.0, 1.0, 2.0, 3.0, 4.0]
        self._initialized = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _desired_ranks(self) -> tuple[float, float, float, float, float]:
        """Recompute desired positions from ``count`` (avoids float drift)."""
        c_minus_one = float(self.count - 1)
        p = self.p
        return (
            0.0,
            c_minus_one * p / 2.0,
            c_minus_one * p,
            c_minus_one * (1.0 + p) / 2.0,
            c_minus_one,
        )

    def _parabolic(self, i: int, d: int) -> float:
        q = self.q
        n = self.n
        return q[i] + d / (n[i + 1] - n[i - 1]) * (
            (n[i] - n[i - 1] + d) * (q[i + 1] - q[i]) / (n[i + 1] - n[i])
            + (n[i + 1] - n[i] - d) * (q[i] - q[i - 1]) / (n[i] - n[i - 1])
        )

    def _linear(self, i: int, d: int) -> float:
        q = self.q
        n = self.n
        j = i + d
        return q[i] + d * (q[j] - q[i]) / (n[j] - n[i])

    def _adjust_one_step(self, i: int, d: int) -> None:
        """Move marker i by one position (d == ±1) using parabolic + linear fallback."""
        new_q = self._parabolic(i, d)
        if not (self.q[i - 1] < new_q < self.q[i + 1]):
            new_q = self._linear(i, d)
        self.q[i] = new_q
        self.n[i] += d

    def _adjust_markers(self) -> None:
        """Cascade interior markers to their desired positions (bulk-update).

        Each interior marker can advance until it bumps up against a neighbour
        that has not yet been advanced. We therefore run round-robin passes
        across the markers until no further movement is possible — this lets
        a marker fully reach its target after its upper neighbour has caught
        up. For per-value updates this loop terminates after at most a single
        step; for bulk updates the outer loop runs ``O(max d_i)`` times.
        """
        ns = self._desired_ranks()
        order = (1, 2, 3) if self.p >= 0.5 else (3, 2, 1)
        moved = True
        while moved:
            moved = False
            for i in order:
                d = ns[i] - self.n[i]
                if d >= 1.0 and (self.n[i + 1] - self.n[i]) > 1.0:
                    self._adjust_one_step(i, 1)
                    moved = True
                elif d <= -1.0 and (self.n[i - 1] - self.n[i]) < -1.0:
                    self._adjust_one_step(i, -1)
                    moved = True

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------
    def update(self, x: float) -> None:
        """Feed a single scalar value into the estimator."""
        self._update_bulk_numpy(np.asarray([float(x)], dtype=np.float64))

    def update_bulk(self, values: torch.Tensor | np.ndarray | list[float]) -> None:
        """Feed a 1-D array of values into the estimator.

        Vectorised path used by :class:`QuantileCalibrator` for activation
        tensors. The per-value P^2 update is fundamentally sequential, but the
        cell-count increments and the extremes can be computed in one pass on
        the chunk; marker adjustment then runs in a loop bounded by the
        per-marker displacement (small after the first batch).
        """
        if isinstance(values, torch.Tensor):
            arr = values.detach().to("cpu", dtype=torch.float64).numpy()
        else:
            arr = np.asarray(values, dtype=np.float64)
        arr = arr.reshape(-1)
        if arr.size == 0:
            return
        self._update_bulk_numpy(arr)

    # Mini-chunk size for bulk processing. Smaller = more accurate (markers
    # stay closer to current during cell-counting), larger = less Python
    # overhead. 1024 is a reasonable compromise that keeps tail accuracy good
    # while still doing ~4000 numpy reductions per million values rather than
    # millions of Python iterations.
    _BULK_SUBCHUNK: int = 1024

    def _update_bulk_numpy(self, arr: np.ndarray) -> None:
        # Phase 1: fill the 5-value init buffer.
        if not self._initialized:
            need = 5 - len(self._init_buffer)
            if need > 0:
                head = arr[:need]
                self._init_buffer.extend(float(v) for v in head)
                arr = arr[need:]
                if len(self._init_buffer) < 5:
                    self.count += int(head.size)
                    return
                self.count += int(head.size)
                self._initialize_from_buffer()
            if arr.size == 0:
                return

        # Phase 2: process in mini-chunks so markers stay current during
        # cell-counting. A single huge bulk update biases extreme quantiles
        # high because *all* chunk values get counted against the initial
        # markers and then the cascade can overshoot.
        step = self._BULK_SUBCHUNK
        for start in range(0, arr.size, step):
            sub = arr[start : start + step]
            chunk_min = float(sub.min())
            chunk_max = float(sub.max())
            self.q[0] = min(self.q[0], chunk_min)
            self.q[4] = max(self.q[4], chunk_max)
            for i in (1, 2, 3):
                self.n[i] += float(np.count_nonzero(sub < self.q[i]))
            self.n[4] += float(sub.size)
            self.count += int(sub.size)
            self._adjust_markers()

    # ------------------------------------------------------------------
    # Read-out
    # ------------------------------------------------------------------
    def estimate(self) -> float:
        """Return the current estimate of the p-quantile.

        If fewer than 5 values have been observed, returns the max of the
        buffered values — degrades gracefully to "max" semantics for tiny
        calibration sets.
        """
        if not self._initialized:
            if not self._init_buffer:
                return 0.0
            return max(self._init_buffer)
        return float(self.q[2])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the estimator state."""
        return {
            "p": self.p,
            "q": list(self.q),
            "n": list(self.n),
            "count": int(self.count),
            "initialized": bool(self._initialized),
            "init_buffer": list(self._init_buffer),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore estimator state from a dict produced by :meth:`state_dict`."""
        self.p = float(state["p"])
        self.q = [float(v) for v in state["q"]]
        self.n = [float(v) for v in state["n"]]
        self.count = int(state["count"])
        self._initialized = bool(state["initialized"])
        self._init_buffer = [float(v) for v in state["init_buffer"]]

    def __repr__(self) -> str:
        return (
            f"P2QuantileEstimator(p={self.p}, count={self.count}, "
            f"initialized={self._initialized}, estimate={self.estimate():.6g})"
        )


class QuantileCalibrator(_Calibrator):
    """Calibrator that estimates per-tensor amax via streaming quantiles.

    Tracks one P^2 estimator per probability in ``quantiles``. ``compute_amax``
    returns ``max`` over the tracked estimates as a 0-D tensor. The toolkit's
    ``_override_quantile_levels`` is supported by reassigning
    ``_quantile_probs`` and ``_estimators`` after construction.

    **Per-tensor only.** ``axis is not None`` is not supported; the framework
    cannot ask the calibrator for a per-block ``_amax`` here. NVFP4 weight
    quantizers (which need per-block calibrated amax) should use the MSE
    calibrator instead.
    """

    def __init__(
        self,
        num_bits: int = 8,
        axis: Any = None,
        unsigned: bool = False,
        quantiles: list[float] | tuple[float, ...] | None = None,
        subsample_cap: int | None = _DEFAULT_SUBSAMPLE_CAP,
    ) -> None:
        """Initialise the quantile calibrator (per-tensor only; ``axis`` must be ``None``)."""
        super().__init__(num_bits, axis, unsigned)
        if axis is not None:
            raise NotImplementedError(
                "QuantileCalibrator only supports per-tensor calibration (axis=None). "
                f"Got axis={axis}. For per-channel/per-block calibration use MSE or histogram."
            )
        probs = list(DEFAULT_QUANTILES if quantiles is None else quantiles)
        if not probs:
            raise ValueError("`quantiles` must be a non-empty list of probabilities.")
        for p in probs:
            if not (0.0 < p < 1.0):
                raise ValueError(f"Each quantile must be in (0, 1); got {p}.")
        # Keep ascending for readability and so estimate-ordering is intuitive.
        probs = sorted(set(probs))
        self._quantile_probs: list[float] = probs
        self._estimators: dict[float, P2QuantileEstimator] = {
            p: P2QuantileEstimator(p) for p in probs
        }
        self._calib_amax: torch.Tensor | None = None
        # Per-call subsample cap. ``None`` keeps every value (good for tiny
        # synthetic tensors / unit tests); a positive int keeps a uniform
        # random subsample of that many values per ``collect`` (required for
        # real LLM activation tensors, which can be 10^7+ values per batch).
        self._subsample_cap: int | None = subsample_cap

    @torch.no_grad()
    def collect(self, x: torch.Tensor) -> None:
        """Feed an activation tensor into the per-quantile estimators."""
        # Meta-device path mirrors MaxCalibrator: defer real work until on a real device.
        if x.device.type == "meta":
            return
        flat = x.detach().abs().reshape(-1)
        if flat.numel() == 0:
            return
        if torch.isnan(flat).any():
            raise AssertionError("detected nan values during quantile calibration")
        if torch.isinf(flat).any():
            raise AssertionError("detected inf values during quantile calibration")

        # Subsample on-device before the CPU transfer when the tensor is large.
        # Uniform subsampling preserves the empirical distribution for the
        # quantile estimate while making the Python-side P^2 walk tractable.
        cap = self._subsample_cap
        if cap is not None and flat.numel() > cap:
            idx = torch.randperm(flat.numel(), device=flat.device)[:cap]
            flat = flat[idx]

        # Single CPU/float64 transfer per collect; estimators share the array.
        arr = flat.to("cpu", dtype=torch.float64).numpy()
        for est in self._estimators.values():
            est.update_bulk(arr)

    def reset(self) -> None:
        """Discard collected state and reinitialise all estimators."""
        self._calib_amax = None
        self._estimators = {p: P2QuantileEstimator(p) for p in self._quantile_probs}

    def compute_amax(self) -> torch.Tensor | None:
        """Return the per-tensor amax as the max over tracked quantile estimates."""
        if not self._estimators:
            return None
        # Pick the max across tracked quantile estimates. For monotonically
        # converged estimators this equals the highest tracked level; using
        # max is robust to small-N noise where higher-p estimates can momentarily
        # land below lower-p estimates.
        amax_val = max(est.estimate() for est in self._estimators.values())
        if amax_val == 0.0 and self._estimators:
            # No data seen yet; mirror MaxCalibrator which returns None pre-collect.
            return None
        self._calib_amax = torch.tensor(amax_val, dtype=torch.float32)
        return self._calib_amax

    def __str__(self) -> str:
        return f"QuantileCalibrator(quantiles={self._quantile_probs})"

    def __repr__(self) -> str:
        return (
            "QuantileCalibrator("
            + super().__repr__()
            + f" quantiles={self._quantile_probs}"
            + f" calib_amax={self._calib_amax}"
            + ")"
        )


def save_quantile_data(model: torch.nn.Module, path: str) -> int:
    """Dump every :class:`QuantileCalibrator`'s estimator state to ``path``.

    Returns the number of quantizers saved. The output is a JSON file with the
    shape ``{module_name: {str(p): estimator_state_dict, ...}}``.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for name, module in model.named_modules():
        cal = getattr(module, "_calibrator", None)
        if not isinstance(cal, QuantileCalibrator):
            continue
        out[name] = {
            f"{p:.10g}": copy.deepcopy(est.state_dict()) for p, est in cal._estimators.items()
        }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return len(out)
