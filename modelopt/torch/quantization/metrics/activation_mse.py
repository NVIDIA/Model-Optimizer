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

# mypy: ignore-errors
# ruff: noqa: D107, D205, PERF401, PLR0124

"""Per-layer activation MSE between original (unquantized) and quantized model.

Includes the portable ``ActivationMSELogger`` class that works across codebases
(FP-Quant List[Tensor] style *and* ModelOpt DataLoader-of-dicts style).

Ported from FP-Quant: https://github.com/IST-DASLab/FP-Quant
"""

import fnmatch
import gc
import hashlib
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def _get_module(block: nn.Module, name: str) -> nn.Module:
    """Get submodule from block by dotted name, e.g. 'self_attn.q_proj'."""
    obj = block
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def _get_linear_layer_names(block: nn.Module) -> list[str]:
    """Collect relative names of linear layers in a transformer block (same as GPTQ)."""
    names = []
    for name, layer in block.named_modules():
        if isinstance(layer, nn.Linear):
            names.append(name)
    return names


def _tensor_from_output(out) -> torch.Tensor:
    """Extract a single tensor from layer output (handle tuple return)."""
    if isinstance(out, torch.Tensor):
        return out.detach()
    return out[0].detach()


def _discover_layer_keys(blocks, layer_names, num_blocks):
    """Build list of valid layer keys."""
    keys = []
    for i in range(num_blocks):
        for name in layer_names:
            try:
                _get_module(blocks[i], name)
            except AttributeError:
                continue
            keys.append(f"model.layers.{i}.{name}")
    return keys


def _collect_outputs(
    model: nn.Module,
    blocks: nn.ModuleList,
    layer_names: list[str],
    layer_keys: list[str],
    calibration_data: list[torch.Tensor],
    device: torch.device | str,
    num_blocks: int,
    desc: str,
) -> dict[str, list[torch.Tensor]]:
    """Run model on calibration data, capture per-layer outputs (moved to CPU)."""
    captured: dict[str, torch.Tensor] = {}
    saved: dict[str, list[torch.Tensor]] = {k: [] for k in layer_keys}

    def make_hook(key: str):
        def hook(_module: nn.Module, _input: tuple, output) -> None:
            captured[key] = _tensor_from_output(output).cpu()

        return hook

    hooks = []
    for i in range(num_blocks):
        for name in layer_names:
            key = f"model.layers.{i}.{name}"
            if key not in saved:
                continue
            try:
                mod = _get_module(blocks[i], name)
            except AttributeError:
                continue
            hooks.append(mod.register_forward_hook(make_hook(key)))

    try:
        for sample in tqdm(calibration_data, desc=desc, leave=False):
            inp = sample.unsqueeze(0) if sample.dim() == 1 else sample
            inp = inp.to(device)
            captured.clear()
            with torch.no_grad():
                _ = model(inp)
            for key in layer_keys:
                if key in captured:
                    saved[key].append(captured[key])
    finally:
        for h in hooks:
            h.remove()
    return saved


@torch.no_grad()
def measure_per_layer_activation_mse(
    model_orig: nn.Module,
    model_quant: nn.Module,
    calibration_data: list[torch.Tensor],
    device: torch.device | str,
    log_wandb: bool = False,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Measure per-linear-layer MSE between activations of the original (unquantized)
    model and the quantized model on the same calibration data.

    Runs each model on GPU one at a time to avoid OOM.
    Returns a dict mapping layer key (e.g. "model.layers.0.self_attn.q_proj") to MSE.
    """
    if max_samples is not None and max_samples > 0:
        calibration_data = calibration_data[:max_samples]

    blocks_quant = model_quant.model.layers
    blocks_orig = model_orig.model.layers
    num_blocks = len(blocks_quant)
    assert len(blocks_orig) == num_blocks

    layer_names = _get_linear_layer_names(blocks_quant[0])
    layer_keys = _discover_layer_keys(blocks_quant, layer_names, num_blocks)

    # --- Phase 1: run quantized model on GPU, save outputs to CPU ---
    print("  Phase 1/2: collecting quantized model outputs...")
    model_quant.to(device)
    quant_outputs = _collect_outputs(
        model_quant,
        blocks_quant,
        layer_names,
        layer_keys,
        calibration_data,
        device,
        num_blocks,
        desc="Activation MSE (quant)",
    )
    # Free GPU for original model
    model_quant.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    # --- Phase 2: run original model on GPU, compute MSE vs stored quant ---
    print("  Phase 2/2: collecting original model outputs and computing MSE...")
    model_orig.to(device)

    # Instead of storing orig outputs, compute MSE on the fly per sample
    sum_sq: dict[str, float] = dict.fromkeys(layer_keys, 0.0)
    count: dict[str, int] = dict.fromkeys(layer_keys, 0)

    captured: dict[str, torch.Tensor] = {}

    def make_hook(key: str):
        def hook(_module: nn.Module, _input: tuple, output) -> None:
            captured[key] = _tensor_from_output(output).cpu()

        return hook

    hooks = []
    for i in range(num_blocks):
        for name in layer_names:
            key = f"model.layers.{i}.{name}"
            if key not in sum_sq:
                continue
            try:
                mod = _get_module(blocks_orig[i], name)
            except AttributeError:
                continue
            hooks.append(mod.register_forward_hook(make_hook(key)))

    try:
        for sample_idx, sample in enumerate(
            tqdm(calibration_data, desc="Activation MSE (orig)", leave=False)
        ):
            inp = sample.unsqueeze(0) if sample.dim() == 1 else sample
            inp = inp.to(device)
            captured.clear()
            _ = model_orig(inp)
            for key in layer_keys:
                if key not in captured:
                    continue
                if sample_idx >= len(quant_outputs.get(key, [])):
                    continue
                o = captured[key].float()
                q = quant_outputs[key][sample_idx].float()
                if o.shape != q.shape:
                    continue
                sum_sq[key] += F.mse_loss(o, q, reduction="sum").item()
                count[key] += o.numel()
    finally:
        for h in hooks:
            h.remove()

    # Free original model from GPU
    model_orig.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    # Move quantized model back to GPU for downstream usage
    model_quant.to(device)

    mse = {
        key: (sum_sq[key] / count[key]) if count[key] > 0 else float("nan") for key in layer_keys
    }

    if log_wandb:
        try:
            import wandb

            for key, val in mse.items():
                if val == val:  # skip nan
                    wandb.log({f"activation_mse/{key}": val})
        except ImportError:
            pass

    return mse


# ---------------------------------------------------------------------------
# Portable ActivationMSELogger class
# ---------------------------------------------------------------------------


def _matches_filter(name: str, layer_filter: str | None) -> bool:
    """Check if a layer name matches the optional filter pattern (fnmatch-style)."""
    if layer_filter is None:
        return True
    return fnmatch.fnmatch(name, layer_filter)


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

    # 1. Try modelopt helper
    try:
        from modelopt.torch.quantization.utils.activation_collector import LayerActivationCollector

        decoder_layers = LayerActivationCollector.get_decoder_layers(model)
    except Exception:
        pass

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

    Example (FP-Quant -- List[Tensor])::

        mse_logger = ActivationMSELogger(max_samples=16, save_dir="./mse_logs")
        mse_logger.collect(model_orig, calibration_data, phase="original")
        mse_logger.collect(model_quant, calibration_data, phase="quantized")
        results = mse_logger.compute_mse()
        print(mse_logger.summary())
        mse_logger.save()

    Example (ModelOpt -- DataLoader with dict batches)::

        mse_logger = ActivationMSELogger(max_samples=16, save_dir="./mse_logs")
        mse_logger.collect(model, dataloader, phase="original")
        model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
        mse_logger.collect(model, dataloader, phase="quantized")
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
        data,
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
        for key in sorted(self.mse_results.keys()):
            lines.append(f"  {key}: {self.mse_results[key]:.6e}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Pre-materialized MSE data (cross-run / cross-codebase safety)
    # ------------------------------------------------------------------

    @staticmethod
    def materialize_data(
        data,
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
    # Raw-text materialization (cross-model / cross-tokenizer reuse)
    # ------------------------------------------------------------------

    @staticmethod
    def materialize_raw_text(
        data,
        path: str,
        tokenizer=None,
        max_samples: int | None = None,
    ) -> list[str]:
        """Save raw text strings to a JSON file for cross-model reuse.

        Extracts text from batches by decoding ``input_ids`` with the provided
        *tokenizer*.  The saved JSON file can be loaded by any model regardless
        of its vocabulary and re-tokenized via :meth:`tokenize_raw_text`.

        Args:
            data: Iterable of batches (DataLoader, ``List[Tensor]``, etc.).
            path: Destination ``.json`` file path.
            tokenizer: A HuggingFace tokenizer with a ``decode`` method.
                Required to convert token IDs back to text.
            max_samples: How many batches to keep.  ``None`` means all.

        Returns:
            The list of decoded text strings (same content that was saved).
        """
        if tokenizer is None:
            raise ValueError(
                "tokenizer is required for materialize_raw_text to decode input_ids back to text."
            )

        texts: list[str] = []
        for batch in data:
            if max_samples is not None and len(texts) >= max_samples:
                break
            if isinstance(batch, dict):
                t = batch.get("input_ids", next(iter(batch.values())))
            elif isinstance(batch, torch.Tensor):
                t = batch
            elif isinstance(batch, (list, tuple)):
                t = batch[0]
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            if t.dim() == 1:
                t = t.unsqueeze(0)
            for row in t:
                if max_samples is not None and len(texts) >= max_samples:
                    break
                texts.append(tokenizer.decode(row, skip_special_tokens=True))

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {"texts": texts, "max_samples": len(texts)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[ActivationMSELogger] Saved {len(texts)} raw text samples -> {path}")
        return texts

    @staticmethod
    def load_raw_text(path: str) -> list[str]:
        """Load raw text strings from a JSON file created by :meth:`materialize_raw_text`.

        Args:
            path: Path to the ``.json`` file.

        Returns:
            List of raw text strings.
        """
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        texts = payload["texts"]
        print(f"[ActivationMSELogger] Loaded {len(texts)} raw text samples from {path}")
        return texts

    @staticmethod
    def tokenize_raw_text(
        texts: list[str],
        tokenizer,
        max_length: int = 2048,
    ) -> list[torch.Tensor]:
        """Tokenize raw text strings into a ``List[Tensor]`` for :meth:`collect`.

        Each string is independently tokenized and truncated to *max_length*.
        Returns one ``[1, seq_len]`` tensor per string — the same format
        expected by :meth:`collect` and :func:`compute_perplexity`.

        Args:
            texts: List of raw text strings (from :meth:`load_raw_text`).
            tokenizer: A HuggingFace tokenizer.
            max_length: Maximum token length per sample (default: 2048).

        Returns:
            ``List[Tensor]`` of tokenized inputs on CPU.
        """
        samples: list[torch.Tensor] = []
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                add_special_tokens=False,
            )
            samples.append(encoded.input_ids.cpu())
        print(f"[ActivationMSELogger] Tokenized {len(samples)} samples (max_length={max_length})")
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
