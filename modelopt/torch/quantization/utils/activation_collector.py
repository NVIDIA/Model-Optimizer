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

"""Sequential calibration layer patching and activation capture.

This module provides :class:`LayerActivationCollector`, a stateful helper that
patches decoder layers with a skip / run / capture strategy for efficient
layer-by-layer calibration.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from modelopt.torch.opt.searcher import ForwardLoop
from modelopt.torch.utils import print_rank_0
from modelopt.torch.utils.network import bind_forward_method, unpatch_forward_method


class _EarlyStopForwardError(Exception):
    """Raised to halt the forward pass after capturing layer inputs."""


@dataclass
class _LayerCalibState:
    """Mutable per-layer state used during sequential calibration.

    Attached to each decoder layer as ``_seq_calib`` and accessed by the
    patched forward to decide skip / run / capture / original behaviour.
    """

    mode: str = "original"
    name: str = ""
    cached_inputs: deque = field(default_factory=deque)
    collected_inputs: list = field(default_factory=list)
    output_meta: tuple | None = None


class LayerActivationCollector:
    """Collects layer activations for sequential (layer-by-layer) calibration.

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

    # Global registry of (predicate, discoverer) pairs. Populated at import time
    # by plugins (e.g. huggingface.py, megatron.py). Order matters: the first
    # matching entry wins, so more specific predicates (e.g. Nemotron-H) must be
    # registered before generic ones (e.g. homogeneous HF models).
    #
    # This is intentionally a mutable class variable shared across all instances:
    # plugins register once at import time, and the registry is read-only after
    # that.  register_decoder_layer_support() guards against duplicate entries.
    _decoder_layer_support: list[tuple[Any, Any]] = []
    _LAYER_ATTR = "_seq_calib"

    def __init__(self, model: nn.Module):
        """Initialize the collector for the given model."""
        self.model = model
        self._decoder_layers: nn.ModuleList | None = None
        self._layer_to_idx: dict[nn.Module, int] = {}
        self._patched = False

    @staticmethod
    def get_decoder_layers(model: nn.Module) -> nn.ModuleList | None:
        """Return decoder layers supported by sequential calibration."""
        for is_supported, discoverer in LayerActivationCollector._decoder_layer_support:
            if not is_supported(model):
                continue
            decoder_layers = discoverer(model)
            if decoder_layers is not None:
                return decoder_layers
        return None

    @staticmethod
    def is_supported(model: nn.Module) -> bool:
        """Whether the model supports decoder-layer sequential calibration."""
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
        # "other" values are expected to be lightweight non-tensors (e.g. None, small scalars).
        # The value is returned directly (not copied); callers must not mutate it.
        # In practice this is safe because skip-mode outputs are immediately discarded by the
        # downstream run-mode layer, which replays from its own cached inputs instead.
        return meta[1]

    def _patch_all_layers(self, decoder_layers: nn.ModuleList | None = None):
        """Bind the unified forward to every decoder layer and the model. Called once.

        Args:
            decoder_layers: Pre-resolved decoder layers. If *None*, layers are
                discovered via :meth:`get_decoder_layers`.
        """

        def _patched_forward(self, *args, **kwargs):
            """Unified forward bound to every decoder layer during sequential calibration.

            ``self`` here is the decoder layer module (bound via ``bind_forward_method``).
            All per-layer state is accessed through ``self._seq_calib``.
            """
            info: _LayerCalibState = self._seq_calib

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
                layer._seq_calib = _LayerCalibState(
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
            done = self._decoder_layers[layer_idx - 2]._seq_calib
            # output_meta is intentionally kept: skip mode needs it to produce
            # correctly shaped zero-filled outputs for the parent forward.
            done.mode = "skip"
            done.cached_inputs.clear()

        if layer_idx > 0:
            prev = self._decoder_layers[layer_idx - 1]._seq_calib
            if not prev.collected_inputs:
                raise RuntimeError(
                    f"Layer {layer_idx - 1} ({prev.name!r}) has no collected inputs to replay. "
                    "Layers must be calibrated sequentially — ensure get_input_activations() "
                    "was called for every preceding layer in order."
                )
            prev.mode = "run"
            prev.cached_inputs = deque(prev.collected_inputs)
            prev.collected_inputs = []

        cur = self._decoder_layers[layer_idx]._seq_calib
        cur.mode = "capture"
        cur.collected_inputs = []

    def _log_layer_summary(self, layer_idx: int):
        """Log a one-line summary of layer modes for the current calibration step."""
        assert self._decoder_layers is not None
        n = len(self._decoder_layers)
        groups: dict[str, list[int]] = {}
        for i, layer in enumerate(self._decoder_layers):
            mode = layer._seq_calib.mode
            if mode in ("skip", "run", "capture"):
                groups.setdefault(mode, []).append(i + 1)
        parts = [f"{mode}: {groups[mode]}" for mode in ("skip", "run", "capture") if mode in groups]
        print_rank_0(f"Calibrating layer {layer_idx + 1}/{n} | {' | '.join(parts)}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        info = layer._seq_calib
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
        # After capture, set to original so calib_func can call the layer's
        # real forward directly.  The layer will transition to run → skip
        # in subsequent iterations via _set_layer_states.
        info.mode = "original"
        return inputs
