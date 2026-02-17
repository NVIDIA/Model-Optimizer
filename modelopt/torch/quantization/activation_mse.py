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

"""Per-layer activation MSE measurement for quantization analysis.

This module provides utilities to measure per-linear-layer MSE between a model's
activations before and after quantization.  Inspired by FP-Quant's two-phase approach:

- **Phase 1** (before quantization): ``collect_activations()`` runs the model on
  calibration data and stores per-layer outputs in CPU RAM.
- **Phase 2** (after quantization): ``measure_activation_mse()`` runs the quantized
  model on the same data and computes MSE on-the-fly against the stored Phase 1
  outputs.  Only running scalar accumulators are kept -- no second set of tensors
  is stored.

Typical usage in hf_ptq.py::

    # Phase 1: before quantization
    orig_acts = mtq.collect_activations(model, mse_dataloader, max_samples=16)

    # Quantize
    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

    # Phase 2: after quantization -- computes MSE incrementally
    mse = mtq.measure_activation_mse(model, mse_dataloader, orig_acts, max_samples=16)
"""

import contextlib
import fnmatch
import hashlib
import os
from collections.abc import Iterable
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from modelopt.torch.utils.network import get_decoder_layers

__all__ = ["ActivationMSELogger", "collect_activations", "measure_activation_mse"]


def _tensor_from_output(out) -> torch.Tensor:
    """Extract a single tensor from a layer's output (handles tuple returns)."""
    if isinstance(out, torch.Tensor):
        return out.detach()
    return out[0].detach()


def _is_linear(module: nn.Module) -> bool:
    """Check if a module is a linear layer (covers both nn.Linear and quantized linear)."""
    return isinstance(module, nn.Linear)


def _matches_filter(name: str, layer_filter: str | None) -> bool:
    """Check if a layer name matches the optional filter pattern (fnmatch-style)."""
    if layer_filter is None:
        return True
    return fnmatch.fnmatch(name, layer_filter)


def _discover_target_layers(
    model: nn.Module,
    layer_filter: str | None = None,
) -> dict[str, nn.Module]:
    """Discover linear layers within decoder blocks of the model.

    Uses get_decoder_layers() to find transformer blocks, then finds all linear
    submodules within those blocks.  Falls back to all linear layers in the model
    if decoder blocks cannot be identified.

    Args:
        model: The model to inspect.
        layer_filter: Optional fnmatch pattern to select specific layers
            (e.g., ``"*self_attn*"``).

    Returns:
        Dict mapping full module path -> module reference.
    """
    decoder_layers = get_decoder_layers(model)

    targets: dict[str, nn.Module] = {}

    if decoder_layers is not None:
        # Build a reverse lookup: module id -> full name in model
        module_to_name: dict[int, str] = {id(m): n for n, m in model.named_modules()}

        for block in decoder_layers:
            block_name = module_to_name.get(id(block), "")
            for sub_name, sub_mod in block.named_modules():
                if _is_linear(sub_mod):
                    full_name = f"{block_name}.{sub_name}" if block_name else sub_name
                    if _matches_filter(full_name, layer_filter):
                        targets[full_name] = sub_mod
    else:
        # Fallback: scan all modules
        for name, module in model.named_modules():
            if _is_linear(module):
                if _matches_filter(name, layer_filter):
                    targets[name] = module

    return targets


def _run_batch(model: nn.Module, batch) -> None:
    """Run a single batch through the model."""
    if isinstance(batch, dict):
        model(**batch)
    elif isinstance(batch, (list, tuple)):
        model(*batch)
    else:
        model(batch)


@torch.no_grad()
def collect_activations(
    model: nn.Module,
    dataloader: Iterable,
    max_samples: int | None = None,
    layer_filter: str | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Collect per-linear-layer output activations into CPU memory (Phase 1).

    Registers forward hooks on linear layers within the model's decoder blocks,
    runs calibration data through the model, and returns captured per-layer outputs.

    Args:
        model: The model to collect activations from (typically pre-quantization).
        dataloader: An iterable yielding batches (dicts with ``input_ids``, etc.).
            Use batch_size=1 to minimize memory.
        max_samples: Maximum number of batches to process.  ``None`` means all.
        layer_filter: Optional fnmatch pattern to restrict which layers are
            collected (e.g., ``"*self_attn*"``).  ``None`` means all linear layers
            inside decoder blocks.

    Returns:
        Dict mapping layer name to a list of output tensors (one per batch, on CPU).
    """
    was_training = model.training
    model.eval()

    # Discover target linear layers
    targets = _discover_target_layers(model, layer_filter)
    if not targets:
        raise ValueError(
            f"No linear layers found matching the given filter. layer_filter={layer_filter!r}"
        )

    print(f"Collecting activations for {len(targets)} layers...")

    # Storage: {layer_name: [tensor_per_batch, ...]}
    saved: dict[str, list[torch.Tensor]] = {name: [] for name in targets}
    captured: dict[str, torch.Tensor] = {}

    def _make_hook(key: str):
        def hook(_module: nn.Module, _input, output) -> None:
            captured[key] = _tensor_from_output(output).cpu()

        return hook

    # Register hooks
    hooks = []
    for name, module in targets.items():
        hooks.append(module.register_forward_hook(_make_hook(name)))

    try:
        n_batches = 0
        for batch in tqdm(dataloader, desc="Collecting activations", leave=False):
            if max_samples is not None and n_batches >= max_samples:
                break

            captured.clear()
            _run_batch(model, batch)

            for name in targets:
                if name in captured:
                    saved[name].append(captured[name])

            n_batches += 1
    finally:
        for h in hooks:
            h.remove()

    model.train(was_training)

    print(f"Collected {n_batches} samples across {len(targets)} layers")
    return saved


@torch.no_grad()
def measure_activation_mse(
    model: nn.Module,
    dataloader: Iterable,
    orig_activations: dict[str, list[torch.Tensor]],
    max_samples: int | None = None,
    layer_filter: str | None = None,
) -> dict[str, float]:
    """Compute per-layer MSE between stored and live activations (Phase 2).

    Runs the (quantized) model on calibration data and computes MSE on-the-fly
    against the pre-quantization activations stored by :func:`collect_activations`.

    Only scalar accumulators (sum of squared errors and element count) are kept
    per layer -- no second set of activation tensors is stored.

    The MSE for each layer is computed as::

        MSE = sum_over_all_elements((orig - quant) ^ 2) / total_elements

    Args:
        model: The quantized model to measure.
        dataloader: Same dataloader used for :func:`collect_activations`
            (must yield batches in the same order).
        orig_activations: Output of :func:`collect_activations` -- dict mapping
            layer name to a list of pre-quantization output tensors.
        max_samples: Maximum number of batches to process (should match Phase 1).
        layer_filter: Optional fnmatch pattern (should match Phase 1).

    Returns:
        Dict mapping layer name to its MSE value.
    """
    was_training = model.training
    model.eval()

    # Discover target layers on the (now-quantized) model
    targets = _discover_target_layers(model, layer_filter)

    # Only measure layers that exist in both the model and orig_activations
    common_keys = sorted(set(targets.keys()) & set(orig_activations.keys()))
    if not common_keys:
        raise ValueError(
            "No matching layers between the quantized model and stored activations. "
            "Ensure the same layer_filter is used for both phases."
        )

    skipped = set(orig_activations.keys()) - set(targets.keys())
    if skipped:
        print(f"Warning: {len(skipped)} layers in orig_activations not found in model (skipped)")

    print(f"Computing activation MSE for {len(common_keys)} layers...")

    # Scalar accumulators
    sum_sq: dict[str, float] = dict.fromkeys(common_keys, 0.0)
    count: dict[str, int] = dict.fromkeys(common_keys, 0)

    captured: dict[str, torch.Tensor] = {}

    def _make_hook(key: str):
        def hook(_module: nn.Module, _input, output) -> None:
            captured[key] = _tensor_from_output(output).cpu()

        return hook

    # Register hooks only on common layers
    hooks = [targets[name].register_forward_hook(_make_hook(name)) for name in common_keys]

    try:
        batch_idx = 0
        for batch in tqdm(dataloader, desc="Computing activation MSE", leave=False):
            if max_samples is not None and batch_idx >= max_samples:
                break

            captured.clear()
            _run_batch(model, batch)

            for name in common_keys:
                if name not in captured:
                    continue
                if batch_idx >= len(orig_activations.get(name, [])):
                    continue

                o = orig_activations[name][batch_idx].float()
                q = captured[name].float()

                if o.shape != q.shape:
                    print(
                        f"Warning: shape mismatch for {name} batch {batch_idx}: "
                        f"{o.shape} vs {q.shape}, skipping"
                    )
                    continue

                sum_sq[name] += F.mse_loss(o, q, reduction="sum").item()
                count[name] += o.numel()

            batch_idx += 1
    finally:
        for h in hooks:
            h.remove()

    model.train(was_training)

    mse = {
        key: (sum_sq[key] / count[key]) if count[key] > 0 else float("nan") for key in common_keys
    }

    return mse


# ---------------------------------------------------------------------------
# Portable ActivationMSELogger class
# ---------------------------------------------------------------------------


def _portable_discover_target_layers(
    model: nn.Module,
    layer_filter: str | None = None,
) -> dict[str, nn.Module]:
    """Discover linear layers in decoder blocks with a portable fallback chain.

    Strategy:
      1. Try modelopt's ``get_decoder_layers`` (available inside ModelOpt).
      2. Try common HuggingFace attribute paths (``model.model.layers``, etc.).
      3. Fall back to scanning **all** ``nn.Linear`` in ``model.named_modules()``.

    Within each set of decoder blocks the function collects every ``nn.Linear``
    sub-module and optionally filters by *layer_filter* (fnmatch pattern).
    """
    decoder_layers = None

    # 1. Try modelopt helper (may not exist when file is copied elsewhere)
    with contextlib.suppress(Exception):
        decoder_layers = get_decoder_layers(model)

    # 2. Try common HF / other patterns
    if decoder_layers is None:
        for attr_chain in (
            ("model", "layers"),
            ("decoder", "layers"),
            ("transformer", "h"),
            ("backbone", "layers"),
        ):
            obj = model
            try:
                for attr in attr_chain:
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.ModuleList):
                    decoder_layers = obj
                    break
            except AttributeError:
                continue

    targets: dict[str, nn.Module] = {}

    if decoder_layers is not None:
        module_to_name: dict[int, str] = {id(m): n for n, m in model.named_modules()}
        for block in decoder_layers:
            block_name = module_to_name.get(id(block), "")
            for sub_name, sub_mod in block.named_modules():
                if isinstance(sub_mod, nn.Linear):
                    full_name = f"{block_name}.{sub_name}" if block_name else sub_name
                    if _matches_filter(full_name, layer_filter):
                        targets[full_name] = sub_mod
    else:
        # 3. Fallback: all linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if _matches_filter(name, layer_filter):
                    targets[name] = module

    return targets


class ActivationMSELogger:
    """Portable activation MSE logger for comparing original vs quantized models.

    Works with both:

    - ``List[Tensor]`` data (**FP-Quant** style): each element is ``[1, seq_len]``
      or ``[B, seq_len]``, consumed via ``model(tensor)``.
    - ``DataLoader`` / ``Iterable`` yielding dicts (**ModelOpt** style):
      ``{"input_ids": tensor, ...}``, consumed via ``model(**batch)``.

    Guarantees same samples are used for both phases via SHA-256 hashing of
    input tensors.  Supports saving / loading all activations to disk for
    later cross-codebase comparison.

    Example (ModelOpt -- DataLoader with dict batches)::

        mse_logger = ActivationMSELogger(max_samples=16, save_dir="./mse_logs")
        mse_logger.collect(model, dataloader, phase="original")
        model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
        mse_logger.collect(model, dataloader, phase="quantized")
        results = mse_logger.compute_mse()
        print(mse_logger.summary())
        mse_logger.save()

    Example (FP-Quant -- List[Tensor])::

        mse_logger = ActivationMSELogger(max_samples=16, save_dir="./mse_logs")
        mse_logger.collect(model_orig, calibration_data, phase="original")
        mse_logger.collect(model_quant, calibration_data, phase="quantized")
        results = mse_logger.compute_mse()
        print(mse_logger.summary())
        mse_logger.save()
    """

    def __init__(
        self,
        max_samples: int = 16,
        layer_filter: str | None = None,
        save_dir: str | None = None,
    ):
        """Initialize the ActivationMSELogger.

        Args:
            max_samples: Maximum number of calibration batches to process per phase.
            layer_filter: Optional glob pattern to restrict which layers are tracked.
            save_dir: Optional directory path for persisting activation data to disk.
        """
        self.max_samples = max_samples
        self.layer_filter = layer_filter
        self.save_dir = save_dir

        # Per-phase state
        self.original_activations: dict[str, list[torch.Tensor]] = {}
        self.quantized_activations: dict[str, list[torch.Tensor]] = {}
        self.input_hashes: list[str] = []  # hashes for "original" phase
        self.quant_input_hashes: list[str] = []  # hashes for "quantized" phase

        # Computed after both phases
        self.mse_results: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect(
        self,
        model: nn.Module,
        data: Iterable,
        phase: str,
        target_modules: dict[str, nn.Module] | None = None,
    ) -> None:
        """Collect per-linear-layer output activations for a given phase.

        Args:
            model: The model to run (original or quantized).
            data: An iterable of batches.  Each batch can be:

                - ``torch.Tensor`` with shape ``[B, seq_len]`` (FP-Quant style).
                - ``dict`` with at least an ``"input_ids"`` key (ModelOpt style).
                - ``list`` / ``tuple`` of tensors.
            phase: ``"original"`` or ``"quantized"``.
            target_modules: Optional explicit mapping of ``{name: nn.Module}``
                to attach hooks to.  If *None*, layers are auto-discovered
                via decoder-block scanning.
        """
        if phase not in ("original", "quantized"):
            raise ValueError(f"phase must be 'original' or 'quantized', got {phase!r}")

        was_training = model.training
        model.eval()

        # ----- layer discovery -----
        targets = (
            target_modules
            if target_modules is not None
            else (_portable_discover_target_layers(model, self.layer_filter))
        )
        if not targets:
            raise ValueError(
                "No linear layers found. Provide target_modules explicitly or "
                f"check layer_filter={self.layer_filter!r}."
            )

        print(
            f"[ActivationMSELogger] Phase '{phase}': hooking {len(targets)} layers, "
            f"max_samples={self.max_samples}"
        )

        # ----- storage -----
        saved: dict[str, list[torch.Tensor]] = {name: [] for name in targets}
        captured: dict[str, torch.Tensor] = {}
        hashes: list[str] = []

        def _make_hook(key: str):
            def hook(_module: nn.Module, _input, output) -> None:
                captured[key] = _tensor_from_output(output).cpu()

            return hook

        hooks = []
        for name, module in targets.items():
            hooks.append(module.register_forward_hook(_make_hook(name)))

        try:
            n_batches = 0
            for batch in tqdm(data, desc=f"Collecting ({phase})", leave=False):
                if self.max_samples is not None and n_batches >= self.max_samples:
                    break

                captured.clear()
                self._run_batch(model, batch)

                for name in targets:
                    if name in captured:
                        saved[name].append(captured[name])

                hashes.append(self._hash_batch(batch))
                n_batches += 1
        finally:
            for h in hooks:
                h.remove()

        model.train(was_training)

        # ----- store results on self -----
        if phase == "original":
            self.original_activations = saved
            self.input_hashes = hashes
        else:
            self.quantized_activations = saved
            self.quant_input_hashes = hashes
            # Verify sample consistency
            if self.input_hashes:
                self._verify_hashes()

        # Invalidate any previous MSE since we have new activations
        self.mse_results = None

        print(f"[ActivationMSELogger] Collected {n_batches} batches for phase '{phase}'")

    def compute_mse(self) -> dict[str, float]:
        """Compute per-layer MSE between original and quantized activations.

        Returns:
            Dict mapping layer name to its MSE value.

        Raises:
            ValueError: If either phase has not been collected yet.
        """
        if not self.original_activations:
            raise ValueError(
                "No original activations collected. Call collect(..., phase='original') first."
            )
        if not self.quantized_activations:
            raise ValueError(
                "No quantized activations collected. Call collect(..., phase='quantized') first."
            )

        common_keys = sorted(
            set(self.original_activations.keys()) & set(self.quantized_activations.keys())
        )
        if not common_keys:
            raise ValueError(
                "No matching layer names between original and quantized activations. "
                "Ensure the same model architecture / layer_filter is used for both phases."
            )

        orig_only = set(self.original_activations.keys()) - set(self.quantized_activations.keys())
        quant_only = set(self.quantized_activations.keys()) - set(self.original_activations.keys())
        if orig_only:
            print(
                f"[ActivationMSELogger] Warning: {len(orig_only)} layers only in original (skipped)"
            )
        if quant_only:
            print(
                f"[ActivationMSELogger] Warning: {len(quant_only)} layers only in quantized (skipped)"
            )

        sum_sq: dict[str, float] = dict.fromkeys(common_keys, 0.0)
        count: dict[str, int] = dict.fromkeys(common_keys, 0)

        for name in common_keys:
            orig_list = self.original_activations[name]
            quant_list = self.quantized_activations[name]
            n = min(len(orig_list), len(quant_list))
            for i in range(n):
                o = orig_list[i].float()
                q = quant_list[i].float()
                if o.shape != q.shape:
                    print(
                        f"[ActivationMSELogger] Warning: shape mismatch for {name} "
                        f"batch {i}: {o.shape} vs {q.shape}, skipping"
                    )
                    continue
                sum_sq[name] += F.mse_loss(o, q, reduction="sum").item()
                count[name] += o.numel()

        self.mse_results = {
            key: (sum_sq[key] / count[key]) if count[key] > 0 else float("nan")
            for key in common_keys
        }
        return self.mse_results

    def save(self, path: str | None = None) -> str:
        """Save all state (activations, hashes, MSE) to disk via ``torch.save``.

        Args:
            path: Explicit file path.  If *None*, a timestamped file is created
                inside ``self.save_dir`` (which must be set).

        Returns:
            The path where the file was saved.
        """
        if path is None:
            if self.save_dir is None:
                raise ValueError("Provide a path or set save_dir in the constructor.")
            os.makedirs(self.save_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.save_dir, f"activation_mse_{ts}.pt")

        payload = {
            "max_samples": self.max_samples,
            "layer_filter": self.layer_filter,
            "input_hashes": self.input_hashes,
            "quant_input_hashes": self.quant_input_hashes,
            "original_activations": self.original_activations,
            "quantized_activations": self.quantized_activations,
            "mse": self.mse_results,
        }
        torch.save(payload, path)
        print(f"[ActivationMSELogger] Saved to {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "ActivationMSELogger":
        """Load a previously saved ``ActivationMSELogger`` from disk.

        Args:
            path: Path to the ``.pt`` file created by :meth:`save`.

        Returns:
            A new ``ActivationMSELogger`` instance with restored state.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        logger = cls(
            max_samples=payload.get("max_samples", 16),
            layer_filter=payload.get("layer_filter"),
        )
        logger.original_activations = payload.get("original_activations", {})
        logger.quantized_activations = payload.get("quantized_activations", {})
        logger.input_hashes = payload.get("input_hashes", [])
        logger.quant_input_hashes = payload.get("quant_input_hashes", [])
        logger.mse_results = payload.get("mse")
        print(f"[ActivationMSELogger] Loaded from {path}")
        return logger

    def summary(self) -> str:
        """Return a formatted string summarising per-layer MSE results.

        Computes MSE first if not already done.
        """
        if self.mse_results is None:
            self.compute_mse()
        assert self.mse_results is not None

        lines = ["Per-layer activation MSE (original vs quantized):"]
        lines.extend(
            f"  {key}: {self.mse_results[key]:.6e}" for key in sorted(self.mse_results.keys())
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Pre-materialized MSE data (cross-run / cross-codebase safety)
    # ------------------------------------------------------------------

    @staticmethod
    def materialize_data(
        data: Iterable,
        path: str,
        max_samples: int | None = None,
    ) -> list[torch.Tensor]:
        """Freeze the first *max_samples* batches from *data* into a ``.pt`` file.

        Each batch (``dict``, ``Tensor``, or ``list/tuple``) is normalised to a
        single ``input_ids`` CPU tensor before saving.  The resulting file is a
        plain ``List[Tensor]`` that can be loaded in **any** codebase and passed
        straight to :meth:`collect`.

        If *path* already exists it is **not** overwritten -- call
        :meth:`load_data` instead.

        Args:
            data: Iterable of batches (DataLoader, List[Tensor], etc.).
            path: Destination ``.pt`` file path.
            max_samples: How many batches to keep. ``None`` means all.

        Returns:
            The materialised list of CPU tensors (same object that was saved).
        """
        samples: list[torch.Tensor] = []
        for batch in data:
            if max_samples is not None and len(samples) >= max_samples:
                break
            if isinstance(batch, dict):
                t = batch.get("input_ids", next(iter(batch.values())))
            elif isinstance(batch, torch.Tensor):
                t = batch
            elif isinstance(batch, (list, tuple)):
                t = batch[0]
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")
            samples.append(t.cpu())

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(samples, path)
        print(f"[ActivationMSELogger] Materialised {len(samples)} MSE input samples -> {path}")
        return samples

    @staticmethod
    def load_data(path: str) -> list[torch.Tensor]:
        """Load a previously materialised MSE input set.

        Args:
            path: Path to the ``.pt`` file created by :meth:`materialize_data`.

        Returns:
            ``List[Tensor]`` of input batches (on CPU).
        """
        samples = torch.load(path, map_location="cpu", weights_only=True)
        print(f"[ActivationMSELogger] Loaded {len(samples)} MSE input samples from {path}")
        return samples

    # ------------------------------------------------------------------
    # Static / private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_batch(model: nn.Module, batch) -> None:
        """Run a single batch through the model (handles Tensor, dict, list/tuple).

        Automatically moves inputs to the model's device so that CPU-stored
        materialized data works transparently with a CUDA model.
        """
        device = next(model.parameters()).device
        if isinstance(batch, dict):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            model(**batch)
        elif isinstance(batch, torch.Tensor):
            model(batch.to(device))
        elif isinstance(batch, (list, tuple)):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            model(*batch)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

    @staticmethod
    def _hash_batch(batch) -> str:
        """Compute SHA-256 hash of the primary input tensor in *batch*.

        - ``dict``  -> hashes ``batch["input_ids"]`` (falls back to first value).
        - ``Tensor`` -> hashes the tensor directly.
        - ``list/tuple`` -> hashes the first element.
        """
        if isinstance(batch, dict):
            t = batch.get("input_ids", next(iter(batch.values())))
        elif isinstance(batch, torch.Tensor):
            t = batch
        elif isinstance(batch, (list, tuple)):
            t = batch[0] if batch else None
        else:
            return ""

        if t is None or not isinstance(t, torch.Tensor):
            return ""
        return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()

    def _verify_hashes(self) -> None:
        """Compare input hashes between original and quantized phases."""
        n = min(len(self.input_hashes), len(self.quant_input_hashes))
        mismatches = sum(1 for i in range(n) if self.input_hashes[i] != self.quant_input_hashes[i])
        if mismatches:
            print(
                f"[ActivationMSELogger] WARNING: {mismatches}/{n} batches have "
                f"different input hashes between original and quantized phases. "
                f"The same data may not have been used for both phases!"
            )
        else:
            print(f"[ActivationMSELogger] Input hash verification passed ({n}/{n} match)")
