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

"""Layerwise calibration layer patching, activation capture, and checkpoint save/resume.

This module provides :class:`LayerActivationCollector`, a stateful helper that
patches decoder layers with a skip / run / capture strategy for efficient
layer-by-layer calibration, and :class:`_CheckpointState` for persisting
per-layer calibration progress to disk.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils import print_rank_0
from modelopt.torch.utils.network import (
    bind_forward_method,
    get_module_device,
    unpatch_forward_method,
)

if TYPE_CHECKING:
    from modelopt.torch.opt.searcher import ForwardLoop


class _EarlyStopForwardError(Exception):
    """Raised to halt the forward pass after capturing layer inputs."""


@dataclass
class _LayerCalibState:
    """Mutable per-layer state used during layerwise calibration.

    Attached to each decoder layer as ``_layerwise_calib`` and accessed by the
    patched forward to decide skip / run / capture / original behaviour.
    """

    mode: str = "original"
    name: str = ""
    cached_inputs: deque = field(default_factory=deque)
    collected_inputs: list = field(default_factory=list)
    output_meta: tuple | None = None


class _SkipLayer(nn.Module):
    """Parameter-free stand-in for a fully calibrated decoder layer.

    Replaces the real layer in the ModuleList so that framework hooks
    (accelerate, FSDP2, etc.) have no parameters to transfer. Holds a
    reference to the original layer for restoration during cleanup.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        # Bypass nn.Module.__setattr__ to avoid registering original as a submodule.
        object.__setattr__(self, "_original", original)
        self._layerwise_calib = _LayerCalibState(mode="skip")

    _PROXY_BLOCKLIST = frozenset({"_hf_hook", "_old_forward"})

    def __getattr__(self, name: str):
        # Proxy non-special attribute lookups to the original layer so that
        # parent-model code that accesses layer-level attributes (e.g.,
        # NemotronH's ``block_type``) still works when the layer is replaced
        # with a _SkipLayer.  Accelerate hook attrs are blocked so the
        # framework does not attempt to manage this parameter-free stand-in.
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in self._PROXY_BLOCKLIST:
                raise
            return getattr(object.__getattribute__(self, "_original"), name)

    def forward(self, *args, **kwargs):
        return LayerActivationCollector._zeros_from_meta(
            self._original._layerwise_calib.output_meta
        )


class LayerActivationCollector:
    """Collects layer activations for layerwise (layer-by-layer) calibration.

    Each decoder layer is patched with a unified forward whose behaviour is
    governed by a per-layer :class:`_LayerCalibState`:

    * **skip** — return a zero-filled dummy whose shape and type match the
      layer's real output (reconstructed from lightweight metadata).  No
      computation is performed.  The correctly shaped dummy ensures un-patched
      inter-layer operations in the parent forward (e.g. LayerNorm, tuple
      unpacking) do not raise shape or type errors.
    * **run** — replay previously captured inputs through the original forward,
      ignoring whatever the parent passes in.  Only the just-calibrated layer
      uses this mode, so its output reflects updated weights.
    * **capture** — record ``(args, kwargs)`` and raise
      ``_EarlyStopForwardError`` to halt the forward pass early.
    * **original** — call the original forward unchanged.

    Because the *run* layer discards upstream values, skip-layer outputs are
    never consumed for real computation.
    """

    _decoder_layer_support: list[tuple[Any, Any]] = []
    _LAYER_ATTR = "_layerwise_calib"

    def __init__(self, model: nn.Module):
        """Initialize the collector for the given model."""
        self.model = model
        self._decoder_layers: nn.ModuleList | None = None
        self._layer_to_idx: dict[nn.Module, int] = {}
        self._patched = False

    def _swap_to_dummy(self, idx: int):
        """Replace decoder layer *idx* with a parameter-free dummy.

        ``output_meta`` is intentionally preserved on the original layer: the
        ``_SkipLayer`` reads it to produce correctly shaped zero-filled outputs
        for the parent forward pass.
        """
        assert self._decoder_layers is not None
        layer = self._decoder_layers[idx]
        layer._layerwise_calib.mode = "skip"
        layer._layerwise_calib.cached_inputs.clear()
        self._decoder_layers[idx] = _SkipLayer(layer)

    @staticmethod
    def get_decoder_layers(model: nn.Module) -> nn.ModuleList | None:
        """Return decoder layers supported by layerwise calibration."""
        for is_supported, discoverer in LayerActivationCollector._decoder_layer_support:
            if not is_supported(model):
                continue
            decoder_layers = discoverer(model)
            if decoder_layers is not None:
                return decoder_layers
        return None

    @staticmethod
    def is_supported(model: nn.Module) -> bool:
        """Whether the model supports decoder-layer layerwise calibration."""
        return LayerActivationCollector.get_decoder_layers(model) is not None

    @classmethod
    def register_decoder_layer_support(cls, is_supported: Any, discoverer: Any):
        """Register a (predicate, discoverer) pair for decoder-layer detection."""
        entry = (is_supported, discoverer)
        if entry not in cls._decoder_layer_support:
            cls._decoder_layer_support.append(entry)

    @staticmethod
    def _extract_output_meta(output):
        """Extract lightweight (shape, dtype, device) metadata from a layer output.

        Recursively handles tensors, tuples, lists, and non-tensor values (e.g. None).
        The returned structure can be passed to ``_zeros_from_meta`` to reconstruct a
        zero-filled output with identical shape and type.
        """
        if isinstance(output, torch.Tensor):
            return ("tensor", output.shape, output.dtype, output.device)
        if isinstance(output, tuple):
            return (
                "tuple",
                tuple(LayerActivationCollector._extract_output_meta(o) for o in output),
            )
        if isinstance(output, list):
            return ("list", [LayerActivationCollector._extract_output_meta(o) for o in output])
        return ("other", output)

    @staticmethod
    def _zeros_from_meta(meta):
        """Reconstruct a zero-filled output from metadata produced by ``_extract_output_meta``."""
        tag = meta[0]
        if tag == "tensor":
            _, shape, dtype, device = meta
            return torch.zeros(shape, dtype=dtype, device=device)
        if tag == "tuple":
            return tuple(LayerActivationCollector._zeros_from_meta(m) for m in meta[1])
        if tag == "list":
            return [LayerActivationCollector._zeros_from_meta(m) for m in meta[1]]
        # "other" values are lightweight non-tensors (e.g. None, small scalars).
        # Returned directly (not copied); safe because skip-mode outputs are
        # immediately discarded by the downstream run-mode layer.
        return meta[1]

    def _patch_all_layers(self, decoder_layers: nn.ModuleList | None = None):
        """Bind the unified forward to every decoder layer and the model. Called once.

        Args:
            decoder_layers: Pre-resolved decoder layers. If *None*, layers are
                discovered via :meth:`get_decoder_layers`.
        """

        def _patched_forward(self, *args, **kwargs):
            info: _LayerCalibState = self._layerwise_calib

            if info.mode == "skip":
                if info.output_meta is None:
                    raise RuntimeError(
                        f"Layer {info.name} is in 'skip' mode but has no output_meta. "
                        "This indicates a state-machine bug: the layer should have run "
                        "in 'run' mode (which sets output_meta) before transitioning to 'skip'."
                    )
                return LayerActivationCollector._zeros_from_meta(info.output_meta)

            if info.mode == "run":
                assert info.cached_inputs, (
                    f"Layer {info.name} is in 'run' mode but has no cached inputs to replay."
                )
                real_args, real_kwargs = info.cached_inputs.popleft()
                output = self._original_forward(*real_args, **real_kwargs)
                info.output_meta = LayerActivationCollector._extract_output_meta(output)
                return output

            if info.mode == "capture":
                info.collected_inputs.append((args, kwargs))
                raise _EarlyStopForwardError()

            return self._original_forward(*args, **kwargs)

        if decoder_layers is not None:
            self._decoder_layers = decoder_layers
        else:
            self._decoder_layers = self.get_decoder_layers(self.model)
        assert self._decoder_layers is not None

        self._layer_to_idx = {layer: i for i, layer in enumerate(self._decoder_layers)}
        module_to_name = {m: name for name, m in self.model.named_modules()}

        try:
            for layer in self._decoder_layers:
                layer._layerwise_calib = _LayerCalibState(
                    name=module_to_name.get(layer, type(layer).__name__),
                )
                bind_forward_method(layer, _patched_forward, "_original_forward")

            def _early_stop_forward(module_self, *args, **kwargs):
                try:
                    return module_self._original_forward(*args, **kwargs)
                except _EarlyStopForwardError:
                    return None

            bind_forward_method(self.model, _early_stop_forward, "_original_forward")
        except Exception:
            self._cleanup_layers()
            raise

        self._patched = True

    def _cleanup_layers(self):
        """Best-effort cleanup of any patched layers and model forward."""
        if self._decoder_layers is not None:
            for idx, layer in enumerate(self._decoder_layers):
                if isinstance(layer, _SkipLayer):
                    self._decoder_layers[idx] = layer._original

        if hasattr(self.model, "_original_forward"):
            unpatch_forward_method(self.model, "_original_forward")

        if self._decoder_layers is not None:
            for layer in self._decoder_layers:
                if hasattr(layer, "_original_forward"):
                    unpatch_forward_method(layer, "_original_forward")
                if hasattr(layer, self._LAYER_ATTR):
                    delattr(layer, self._LAYER_ATTR)

    def _unpatch_all_layers(self):
        """Restore original forwards and clean up state attributes. Called once."""
        if not self._patched:
            return
        self._cleanup_layers()
        self._patched = False

    def _set_layer_states(self, layer_idx: int):
        """Transition layer modes for the next calibration step.

        When calibrating layer *i*, three transitions happen:

        * Layer ``i - 2`` → **skip** (fully done, free its cached inputs).
        * Layer ``i - 1`` → **run** (replay captured inputs with calibrated weights).
        * Layer ``i``     → **capture** (record inputs, then early-stop).
        """
        assert self._decoder_layers is not None

        if layer_idx > 1:
            idx = layer_idx - 2
            if not isinstance(self._decoder_layers[idx], _SkipLayer):
                self._swap_to_dummy(idx)

        if layer_idx > 0:
            prev = self._decoder_layers[layer_idx - 1]._layerwise_calib
            if not prev.collected_inputs:
                raise RuntimeError(
                    f"Layer {layer_idx - 1} ({prev.name!r}) has no collected inputs to replay. "
                    "Layers must be calibrated sequentially — ensure get_input_activations() "
                    "was called for every preceding layer in order."
                )
            prev.mode = "run"
            prev.cached_inputs = deque(prev.collected_inputs)
            prev.collected_inputs = []

        cur = self._decoder_layers[layer_idx]._layerwise_calib
        cur.mode = "capture"
        cur.collected_inputs = []

    def _log_layer_summary(self, layer_idx: int):
        """Log a one-line summary of layer modes for the current calibration step."""
        assert self._decoder_layers is not None
        n = len(self._decoder_layers)
        groups: dict[str, list[int]] = {}
        for i, layer in enumerate(self._decoder_layers):
            mode = layer._layerwise_calib.mode
            if mode in ("skip", "run", "capture"):
                groups.setdefault(mode, []).append(i + 1)

        parts = []
        for mode in ("skip", "run", "capture"):
            if mode not in groups:
                continue
            ids = groups[mode]
            parts.append(f"{mode}: {len(ids)}" if mode == "skip" else f"{mode}: {ids}")
        print_rank_0(f"Calibrating layer {layer_idx + 1}/{n} | {' | '.join(parts)}")

    @torch.no_grad()
    def get_input_activations(self, layer: torch.nn.Module, forward_loop: ForwardLoop) -> list:
        """Collect input activations for *layer* by running a full model forward.

        Layers before the target are skipped or re-run (if just calibrated), the
        target layer captures its inputs, and an early-stop prevents unnecessary
        computation beyond the target.

        :meth:`_patch_all_layers` must be called before this method.

        Note: the model forward returns ``None`` for every batch during capture
        (because ``_EarlyStopForwardError`` short-circuits the forward pass).
        Callers should not rely on the model's return value within *forward_loop*.
        """
        if not self._patched:
            raise RuntimeError(
                "get_input_activations() requires _patch_all_layers() to be called first."
            )
        layer_idx = self._layer_to_idx[layer]
        self._set_layer_states(layer_idx)
        self._log_layer_summary(layer_idx)

        info = layer._layerwise_calib
        try:
            forward_loop(self.model)
        except Exception:
            # Reset the current layer so subsequent calls don't see stale state.
            info.mode = "original"
            info.collected_inputs = []
            raise

        if not info.collected_inputs:
            info.mode = "original"
            raise RuntimeError(
                f"Layer {info.name!r} collected no inputs during forward_loop. "
                "The forward loop did not reach this layer — check that forward_loop() "
                "actually calls the model and that the layer is in the forward path."
            )

        inputs = list(info.collected_inputs)
        # Reset to original so calib_func can call the layer's real forward
        # directly.  The layer will transition to run → skip in subsequent
        # iterations via _set_layer_states.
        info.mode = "original"
        return inputs

    def get_first_layer_inputs(
        self,
        start_layer: int,
        resumed_inputs: list | None,
        forward_loop: ForwardLoop,
    ) -> list:
        """Get inputs for the first layer to calibrate, handling resume.

        If *resumed_inputs* is provided, sets skip mode on layers ``0..start_layer-1``
        and seeds the start layer's ``collected_inputs`` for subsequent
        ``cache_outputs_for_next_layer_calib`` calls.  Otherwise, captures inputs
        via a normal forward pass.
        """
        assert self._decoder_layers is not None

        if resumed_inputs is not None:
            print_rank_0(f"Calibrating layer {start_layer + 1} (resumed)")
            for i in range(start_layer):
                self._swap_to_dummy(i)
            layer = self._decoder_layers[start_layer]
            layer._layerwise_calib.collected_inputs = resumed_inputs
            layer._layerwise_calib.mode = "original"
            return resumed_inputs

        return self.get_input_activations(self._decoder_layers[start_layer], forward_loop)

    @torch.no_grad()
    def cache_outputs_for_next_layer_calib(
        self, layer: torch.nn.Module, forward_loop: ForwardLoop
    ) -> list:
        """Run a forward pass after calibrating *layer* to capture the next layer's inputs.

        This puts *layer* into "run" mode (setting its ``output_meta``) and the
        next layer into "capture" mode, then runs *forward_loop*.  Returns the
        captured inputs for the next layer.

        Must be called only when a next layer exists (i.e. *layer* is not the
        last decoder layer).
        """
        assert self._decoder_layers is not None
        layer_idx = self._layer_to_idx[layer]
        next_idx = layer_idx + 1
        assert next_idx < len(self._decoder_layers), "No next layer to capture inputs for."
        from .core_utils import persistent_materialization

        next_layer = self._decoder_layers[next_idx]
        with persistent_materialization(layer):
            return self.get_input_activations(next_layer, forward_loop)


def _move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors to *device*. Non-tensors are returned as-is."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [_move_to_device(v, device) for v in obj]
        return type(obj)(moved)
    return obj


def _remap_output_metadata_device(meta: tuple, device: torch.device) -> tuple:
    """Patch the device field inside output_meta tuples so _zeros_from_meta uses *device*."""
    tag = meta[0]
    if tag == "tensor":
        _, shape, dtype, _old_device = meta
        return ("tensor", shape, dtype, device)
    if tag == "tuple":
        return ("tuple", tuple(_remap_output_metadata_device(m, device) for m in meta[1]))
    if tag == "list":
        return ("list", [_remap_output_metadata_device(m, device) for m in meta[1]])
    return meta


def _read_manifest(checkpoint_dir: str) -> dict | None:
    """Read manifest.json from *checkpoint_dir*. Returns None if missing or corrupt."""
    path = os.path.join(checkpoint_dir, "manifest.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_manifest(checkpoint_dir: str, last_completed_layer: int, num_layers: int) -> None:
    """Atomically write manifest.json."""
    path = os.path.join(checkpoint_dir, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(
            {"last_completed_layer": last_completed_layer, "num_layers": num_layers},
            f,
        )
    os.replace(tmp, path)


def _layer_dir(checkpoint_dir: str, idx: int) -> str:
    return os.path.join(checkpoint_dir, f"layer_{idx:04d}")


def _save_layer(
    checkpoint_dir: str,
    idx: int,
    weights: dict,
    qstate: dict,
    output_meta: tuple,
    next_inputs: list | None,
    num_layers: int,
) -> None:
    """Save a single layer checkpoint and update the manifest atomically."""
    d = _layer_dir(checkpoint_dir, idx)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    torch.save(weights, os.path.join(d, "weights.pt"))
    torch.save(qstate, os.path.join(d, "quantizer_state.pt"))
    torch.save(output_meta, os.path.join(d, "output_meta.pt"))
    if next_inputs is not None:
        torch.save(next_inputs, os.path.join(d, "next_inputs.pt"))
    _write_manifest(checkpoint_dir, idx, num_layers)


def detect_resume_point(checkpoint_dir: str) -> tuple[int, dict] | None:
    """Detect where to resume from an existing checkpoint directory.

    Returns ``(start_layer, manifest)`` if there is work to resume,
    or ``None`` if the directory is empty, corrupt, or calibration was already complete.
    """
    manifest = _read_manifest(checkpoint_dir)
    if manifest is None:
        return None
    last = manifest.get("last_completed_layer")
    total = manifest.get("num_layers")
    if last is None or total is None:
        return None
    if last + 1 >= total:
        return None
    return (last + 1, manifest)


class _CheckpointState:
    """Manages checkpoint save and restore for layerwise calibration.

    Handles both saving per-layer checkpoints during calibration and
    restoring from a previous partial run.

    .. todo::
        Support distributed checkpoint save/restore for FSDP2:
        use ``torch.distributed.checkpoint`` (or save only from rank 0 + barrier)
        and broadcast restored state to all ranks during resume.
    """

    def __init__(self, checkpoint_dir: str, num_layers: int, start_layer: int = 0):
        if dist.is_initialized() and dist.size() > 1:
            raise RuntimeError(
                "Layerwise calibration checkpointing is not supported in "
                "multi-process distributed jobs (e.g. FSDP2). "
                "Use single-process calibration or disable checkpointing."
            )

        self.checkpoint_dir = checkpoint_dir
        self.num_layers = num_layers
        self.start_layer = start_layer

    @classmethod
    def from_folder(cls, checkpoint_dir: str | None, num_layers: int) -> _CheckpointState | None:
        """Create from folder. Detects resume point. Returns None if no checkpoint_dir."""
        if not checkpoint_dir:
            return None
        os.makedirs(checkpoint_dir, exist_ok=True)
        info = detect_resume_point(checkpoint_dir)
        if info is not None:
            manifest_num_layers = info[1].get("num_layers")
            if manifest_num_layers is not None and manifest_num_layers != num_layers:
                raise ValueError(
                    f"Checkpoint num_layers mismatch: manifest has {manifest_num_layers} "
                    f"but model has {num_layers}. Use a fresh checkpoint directory."
                )
        start = info[0] if info else 0
        if start > 0:
            print_rank_0(
                f"Checkpoint: resuming layerwise calibration from layer {start}/{num_layers}"
            )
        return cls(checkpoint_dir, num_layers, start_layer=start)

    def setup_resume(self, layers: nn.ModuleList) -> list | None:
        """Load output_meta for skip layers 0..K-1, return next_inputs for layer K.

        Sets ``output_meta`` on each already-calibrated layer so that
        skip mode can produce correctly shaped dummy outputs.
        """
        if self.start_layer == 0:
            return None

        last_ckpt = self.start_layer - 1

        for i in range(self.start_layer):
            d = _layer_dir(self.checkpoint_dir, i)
            # weights_only=False is safe: file is internally generated by _save_layer, not user-supplied
            meta = torch.load(
                os.path.join(d, "output_meta.pt"), map_location="cpu", weights_only=False
            )
            layer_device = get_module_device(layers[i])
            meta = _remap_output_metadata_device(meta, layer_device)
            layers[i]._layerwise_calib.output_meta = meta

        d = _layer_dir(self.checkpoint_dir, last_ckpt)
        next_inputs_path = os.path.join(d, "next_inputs.pt")
        if not os.path.isfile(next_inputs_path):
            raise FileNotFoundError(f"Cannot resume: next_inputs.pt missing for layer {last_ckpt}")
        # weights_only=False is safe: file is internally generated by _save_layer, not user-supplied
        next_inputs = torch.load(next_inputs_path, map_location="cpu", weights_only=False)
        resume_device = get_module_device(layers[self.start_layer])
        next_inputs = _move_to_device(next_inputs, resume_device)
        return next_inputs

    def full_restore(self, layers: nn.ModuleList, model: nn.Module) -> None:
        """Restore weights and quantizer state for layers 0..K-1 after the calibration loop."""
        from modelopt.torch.quantization.config import QuantizeConfig
        from modelopt.torch.quantization.conversion import restore_quantizer_state
        from modelopt.torch.quantization.utils.core_utils import enable_weight_access_and_writeback

        if self.start_layer == 0:
            return

        dummy_config = QuantizeConfig()
        name_to_module = dict(model.named_modules())
        for i in range(self.start_layer):
            layer = layers[i]
            d = _layer_dir(self.checkpoint_dir, i)

            # Resolve layer_device and load inside the context so params are
            # materialized — otherwise get_module_device can return meta.
            with enable_weight_access_and_writeback(layer, model, name_to_module):
                layer_device = get_module_device(layer)
                # weights_only=False is safe: files are internally generated by _save_layer
                qstate = torch.load(
                    os.path.join(d, "quantizer_state.pt"),
                    map_location=layer_device,
                    weights_only=False,
                )
                weights = torch.load(
                    os.path.join(d, "weights.pt"),
                    map_location=layer_device,
                    weights_only=False,
                )
                restore_quantizer_state(layer, dummy_config, {"quantizer_state": qstate})
                layer.load_state_dict(weights, strict=False, assign=True)

        print_rank_0(f"Checkpoint: restored {self.start_layer} previously calibrated layers")

    def save(
        self,
        layer_idx: int,
        layer: nn.Module,
        model: nn.Module,
        layers: nn.ModuleList,
        next_layer_inputs: list | None = None,
    ) -> None:
        """Snapshot layer state and write checkpoint to disk in one step.

        Args:
            layer_idx: Index of the layer just calibrated.
            layer: The layer module (weights may be on GPU or managed by accelerate/FSDP2).
            model: The full model (needed for ``enable_weight_access_and_writeback``).
            layers: The decoder layer list (to read ``output_meta``).
            next_layer_inputs: Inputs for the next layer (``None`` for the final layer).
        """
        from modelopt.torch.quantization.conversion import quantizer_state
        from modelopt.torch.quantization.utils.core_utils import enable_weight_access_and_writeback

        _cpu = torch.device("cpu")
        with enable_weight_access_and_writeback(layer, model):
            weights = _move_to_device(layer.state_dict(), _cpu)
            qstate = _move_to_device(quantizer_state(layer), _cpu)

        output_meta = getattr(layer._layerwise_calib, "output_meta", None)
        if output_meta is None:
            # Placeholder for the last layer: output_meta is never used for skip mode
            # since there is no subsequent layer that needs a correctly shaped dummy output.
            output_meta = LayerActivationCollector._extract_output_meta(torch.zeros(1))

        _save_layer(
            self.checkpoint_dir,
            layer_idx,
            weights,
            qstate,
            _move_to_device(output_meta, _cpu),
            _move_to_device(next_layer_inputs, _cpu) if next_layer_inputs is not None else None,
            self.num_layers,
        )
        suffix = " (final)" if next_layer_inputs is None else ""
        print_rank_0(f"Checkpoint: saved layer {layer_idx}{suffix}")
