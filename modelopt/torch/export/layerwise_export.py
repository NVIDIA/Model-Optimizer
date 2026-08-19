# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Write each decoder layer's quantized checkpoint shard as soon as it is calibrated."""

import contextlib
import json
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .model_config import FUSION_FREE_FORMATS, QUANTIZATION_NVFP4
from .quant_utils import get_quant_config, get_quantization_format

__all__ = [
    "LayerwiseExporter",
    "assert_layerwise_export_supported",
    "layer_shard_name",
    "transient_module_state",
]

# Fusing formats this path can handle itself. The groups _fuse_shared_input_modules works
# on -- q/k/v behind input_layernorm, gate/up behind post_attention_layernorm -- live
# inside one decoder layer, so export_layer rediscovers them per layer instead of needing
# the whole-model forward. AWQ and SVDQuant are excluded: they additionally need
# requantize_resmooth_fused_llm_layers' pre-quant-scale steps, which are still model-wide.
_PER_LAYER_FUSABLE_FORMATS = frozenset({QUANTIZATION_NVFP4})

SUPPORTED_FORMATS = FUSION_FREE_FORMATS | _PER_LAYER_FUSABLE_FORMATS

_TAIL_SHARD = "model-tail.safetensors"
_INDEX_FILE = "model.safetensors.index.json"


def layer_shard_name(layer_idx: int) -> str:
    """Shard filename for one decoder layer.

    Derived from the index rather than a running counter so that re-exporting a layer
    overwrites its shard instead of leaving a stale copy behind for the index to pick up.
    """
    return f"model-layer-{layer_idx:05d}.safetensors"


def _is_quantized_module(module: nn.Module) -> bool:
    """Whether this module carries quantizers of its own.

    Detected by type, not attribute name: fused-expert modules name theirs after the weight
    (``gate_up_proj_weight_quantizer``), so a name-based test misses whole MoE blocks.
    """
    from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer

    return any(
        isinstance(child, (TensorQuantizer, SequentialQuantizer)) for child in module.children()
    )


def _module_formats(model: nn.Module) -> set:
    """Every distinct quantization format present, not just the first one found.

    ``get_quantization_format(model)`` stops at the first quantized child, so gating on it
    would let an NVFP4 layer slip through a check meant to exclude it.
    """
    return {
        get_quantization_format(module)
        for _, module in model.named_modules()
        if _is_quantized_module(module)
    }


def _tied_quantized_modules(model: nn.Module) -> list[str]:
    """Names of quantized modules that share a weight tensor with another quantized module.

    The whole-model export merges their amaxes via ``sync_tied_input_amax``; a per-layer
    pass cannot, since a tie partner may be uncalibrated or already written.
    """
    by_ptr: dict[int, list[str]] = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or not _is_quantized_module(module) or weight.is_meta:
            continue
        ptr = weight.data_ptr()
        # data_ptr() is 0 for meta tensors and DTensors, grouping unrelated modules.
        if ptr:
            by_ptr.setdefault(ptr, []).append(name)
    return sorted(n for names in by_ptr.values() if len(names) > 1 for n in names)


def assert_layerwise_export_supported(model: nn.Module) -> None:
    """Raise ``NotImplementedError`` unless per-layer export is valid for this model.

    Each case would otherwise produce a checkpoint differing from a whole-model export
    without failing, so all are rejected before the first shard is written.

    The central one is the format gate. Both other export paths begin with
    ``requantize_resmooth_fused_llm_layers``, which this one never calls: its core step
    discovers modules sharing an input via a dummy forward over the *whole* model, and
    there is no such forward here. Restricting to ``FUSION_FREE_FORMATS`` is what makes
    that omission invisible -- for those formats all three of its steps are no-ops, since
    pre-quant-scale fusion requires ``nvfp4_awq`` and MoE expert resmoothing requires AWQ
    or SVDQuant. Any other format would silently lose them.

    .. todo::
        Support the fusing formats (NVFP4 above all) by making
        ``collect_shared_input_modules`` operate on a single decoder layer rather than the
        whole model. The groups it finds -- q/k/v, gate/up -- are intra-layer, so a dummy
        forward over one layer can discover them; the whole-model scope is an artifact of
        how the batch exporter happens to call it, not a requirement. AWQ and SVDQuant
        additionally need the pre-quant-scale steps made per-layer.
    """
    from modelopt.torch.quantization.utils.core_utils import has_accelerate_offload
    from modelopt.torch.utils import distributed as dist

    unsupported = sorted(str(f) for f in _module_formats(model) - SUPPORTED_FORMATS)
    if unsupported:
        raise NotImplementedError(
            f"layerwise export does not support quantization format(s) {unsupported}: they "
            "need requantize_resmooth_fused_llm_layers' pre-quant-scale steps, which are "
            "still whole-model. Supported today: "
            f"{sorted(str(f) for f in SUPPORTED_FORMATS if f)}."
        )

    tied = _tied_quantized_modules(model)
    if has_accelerate_offload(model) and not tied:
        # Offloaded weights sit on meta between windows, so the grouping above inspected
        # nothing; an empty result is not a clean bill of health. Resolving ties by name
        # instead would survive weight moves -- the same fix ExportContext.__post_init__
        # already carries a TODO for.
        warnings.warn(
            "Weight-tie detection is skipped for offloaded models: data_ptr() cannot group "
            "weights that are not resident. A model with tied quantized modules would "
            "export per-module input_scale instead of the merged value."
        )

    if tied:
        raise NotImplementedError(
            f"layerwise export does not support weight-tied quantized modules {tied[:6]}: "
            "the whole-model path merges their input_quantizer amaxes via "
            "sync_tied_input_amax so both sides share one input_scale, which a per-layer "
            "pass cannot do because a tie partner may be uncalibrated or already written."
        )

    if dist.is_initialized() and dist.size() > 1:
        raise NotImplementedError(
            "layerwise export does not support multi-process jobs (e.g. FSDP2): every rank "
            "would write the same shard files. Use single-process calibration."
        )


@contextlib.contextmanager
def transient_module_state(module: nn.Module):
    """Undo everything export does to ``module``, so calibration can continue through it.

    Export is destructive -- packed weights, new scale buffers, grafted per-expert
    submodules. An offloaded model discards that when its materialization window closes; a
    resident one has no window, and calibration still has every later layer to run.

    Restoring the dicts suffices, and costs references rather than a deep copy, because
    export rebinds them instead of mutating tensors in place.
    """
    snapshot = [
        (m, dict(m._parameters), dict(m._buffers), dict(m._modules)) for m in module.modules()
    ]
    try:
        yield
    finally:
        for m, params, buffers, children in snapshot:
            m._parameters.clear()
            m._parameters.update(params)
            m._buffers.clear()
            m._buffers.update(buffers)
            m._modules.clear()
            m._modules.update(children)


class LayerwiseExporter:
    """Writes one decoder layer's quantized shard per call, then the tail and index.

    Constructed before calibration begins, driven once per layer from inside the window
    calibration already opens, and finalized after the last one::

        exporter = LayerwiseExporter(model, export_dir)
        ...
        with persistent_materialization(layer, writeback=False):
            calib_func(layer, ...)
            exporter.export_layer(layer_idx, layer)
        ...
        quant_config = exporter.finalize(extra_state_dict=mtp_state_dict)

    ``finalize()`` rebuilds the index from the shards present on disk, so layers exported
    by an earlier run that this one skipped are picked up without being re-exported.
    """

    def __init__(
        self,
        model: nn.Module,
        export_dir: Path | str,
        dtype: torch.dtype | None = None,
        is_modelopt_qlora: bool = False,
    ) -> None:
        """Validate support and capture model-level state, before calibration runs.

        Only quantizer *configuration* is read here, which ``mtq.quantize`` fixes when it
        swaps modules; anything amax-dependent belongs in :meth:`finalize`.
        """
        from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

        from .layer_utils import is_moe
        from .quant_aware_conversion import build_reverse_name_mapper
        from .registry import ExportContext, PrepareMoEInputsRegistry
        from .unified_export_hf import _resolve_export_dtype
        from .unified_export_hf_streaming import _assert_no_split_rules

        assert_layerwise_export_supported(model)
        # Splits regroup tensors across the whole state dict; no per-layer pass can reverse it.
        _assert_no_split_rules(model)

        for _, sub_module in model.named_modules():
            if (
                is_moe(sub_module)
                and hasattr(sub_module, "experts")
                and PrepareMoEInputsRegistry.match(sub_module.experts) is None
            ):
                raise NotImplementedError(
                    f"MoE model with experts type '{type(sub_module.experts).__name__}' is "
                    "not supported in export."
                )

        layers = LayerActivationCollector.get_decoder_layers(model)
        if layers is None:
            raise RuntimeError(
                "Layerwise export requires discoverable decoder layers. The model "
                "architecture is not supported by LayerActivationCollector."
            )
        # The same call calibration uses, so layer_idx means the same thing on both sides.
        self._layers = layers
        layer_ids = {id(m): i for i, m in enumerate(layers)}
        self._layer_names: dict[int, str] = {}
        for name, module in model.named_modules():
            idx = layer_ids.get(id(module))
            if idx is not None:
                self._layer_names[idx] = name
        # Descendants too: the tail pass must skip anything a layer shard already covered.
        self._decoder_owned_ids = {id(m) for layer in layers for m in layer.modules()}
        # Materialization dispatch rebuilds this map per call when not supplied; only the
        # handful of tail modules reach it, but the map is model-sized either way.
        self._name_to_module = dict(model.named_modules())

        # The context is the single owner of the model, dtype and qlora flag; keeping
        # parallel copies on self would leave two sources of truth for the same facts.
        # Dedup is off for the reason the offload path turns it off (registry.py
        # __post_init__): data_ptr() cannot identify a tensor across an export that keeps
        # rolling packed weights back. Ties are refused anyway, and with both caches None
        # the context is immutable, so one instance serves every pass.
        self._ctx = ExportContext(
            model=model,
            dtype=_resolve_export_dtype(model, dtype),
            is_modelopt_qlora=is_modelopt_qlora,
            tied_cache=None,
            moe_tied_cache=None,
        )

        self._export_dir = Path(export_dir)
        self._export_dir.mkdir(parents=True, exist_ok=True)
        # Not get_kv_cache_dtype(model): it does not recurse, so given the root it always
        # answers None, which then trips the KV assert in the per-tensor pass.
        self._kv_cache_format = get_quant_config(model, is_modelopt_qlora=is_modelopt_qlora)[
            "quantization"
        ]["kv_cache_quant_algo"]
        self._finalized = False

        self._name_mapper = None
        try:
            self._name_mapper = build_reverse_name_mapper(model)
        except Exception as exc:
            warnings.warn(
                f"Reverse name mapper unavailable ({exc}); exported tensor names may not "
                "match the original HF hub checkpoint."
            )
        # By name, not data_ptr: layers and tail are separate passes, so there is never a
        # whole-dict view to compare pointers across.
        raw_tied_keys: set[str] = (
            set(getattr(model, "_tied_weights_keys", None) or [])
            if getattr(model.config, "tie_word_embeddings", False)
            else set()
        )
        self._tied_alias_keys: set[str] = (
            {self._name_mapper(k) for k in raw_tied_keys}
            if self._name_mapper is not None
            else raw_tied_keys
        )

    def export_layer(
        self,
        layer_idx: int,
        layer_module: nn.Module,
        probe_forward: Callable[[nn.Module], None] | None = None,
    ) -> None:
        """Pack one calibrated layer into its shard, leaving the layer itself untouched.

        ``probe_forward`` runs the layer once on real activations; a fusing format needs it
        to rediscover which modules share an input. Omitting it is only valid when no
        format present fuses.
        """
        from modelopt.torch.quantization.plugins.huggingface import _reconstruct_fused_moe_linear

        from .layer_utils import sync_moe_gate_up_amax
        from .unified_export_hf import _dispatch_export_handler, _prepare_moe_inputs

        assert not self._finalized, "export_layer() called after finalize()"
        assert layer_module is self._layers[layer_idx], (
            f"layer_idx {layer_idx} does not match the module passed; calibration and export "
            "disagree on decoder layer order."
        )

        layer_name = self._layer_names[layer_idx]
        tensors: dict[str, torch.Tensor] = {}
        with transient_module_state(layer_module):
            # Per-block, so neither fits a whole-model prep pass: earlier there is no amax
            # yet, later the layer is already written.
            _prepare_moe_inputs(layer_module, self._ctx.dtype, self._ctx.is_modelopt_qlora)
            self._fuse_shared_inputs(layer_module, probe_forward)
            sync_moe_gate_up_amax(layer_module)

            for sub_name, sub_mod in layer_module.named_modules():
                full_name = f"{layer_name}.{sub_name}" if sub_name else layer_name
                _dispatch_export_handler(full_name, sub_mod, self._ctx)
            _reconstruct_fused_moe_linear(layer_module)

            prefix = f"{layer_name}." if layer_name else ""
            for key, tensor in layer_module.state_dict().items():
                self._collect(tensors, prefix + key, tensor)

        save_file(_copy_storage_aliases(tensors), str(self._export_dir / layer_shard_name(layer_idx)))

    def _fuse_shared_inputs(
        self, layer_module: nn.Module, probe_forward: Callable[[nn.Module], None] | None
    ) -> None:
        """Unify scales across the modules of this layer that share an input.

        The whole-model exporters get these groups from one forward over the entire model.
        Rediscovering them per layer is equivalent because the groups never cross a layer
        boundary, and it uses the layer's real activations rather than a synthetic probe.
        """
        from .quant_utils import get_quantization_format
        from .unified_export_hf import _fuse_shared_input_modules, collect_shared_input_modules

        # Per-module scan, not get_quantization_format(layer_module): that returns the first
        # format found, so a layer with FP8 attention and NVFP4 experts reports fp8 and
        # would skip fusing its NVFP4 groups. _fuse_shared_input_modules re-evaluates the
        # format per group, so the value passed below is only a fallback.
        if not (_module_formats(layer_module) - FUSION_FREE_FORMATS):
            return
        layer_format = get_quantization_format(layer_module)
        if probe_forward is None:
            raise RuntimeError(
                f"layer format {layer_format!r} needs input-sharing groups to fuse its "
                "scales, but no probe_forward was supplied to rediscover them."
            )

        input_to_linear, _ = collect_shared_input_modules(
            layer_module, lambda: probe_forward(layer_module)
        )
        _fuse_shared_input_modules(self._ctx.model, input_to_linear, quantization_format=layer_format)

    def finalize(self, extra_state_dict: dict[str, torch.Tensor] | None = None) -> dict:
        """Export the tail, write every config artifact, and index all shards.

        Leaves ``export_dir`` a complete, loadable checkpoint, so no separate
        ``export_hf_checkpoint()`` call is needed. Returns the quant config.
        """
        from modelopt.torch.quantization.utils.core_utils import (
            enable_weight_access_and_writeback,
            requires_weight_materialization,
        )

        from .quant_aware_conversion import revert_quant_config_names
        from .unified_export_hf import (
            _add_mtp_exclusions,
            _dispatch_export_handler,
            _warn_on_unsynced_moe_gate_up,
            _write_hf_export_config,
            save_non_weight_artifacts,
        )

        assert not self._finalized, "finalize() called twice"
        self._finalized = True

        model = self._ctx.model
        quant_config = get_quant_config(model, is_modelopt_qlora=self._ctx.is_modelopt_qlora)
        _add_mtp_exclusions(model, quant_config)
        _warn_on_unsynced_moe_gate_up(model)
        if getattr(model, "hf_quantizer", None) is not None:
            model.hf_quantizer = None
        # Module references in the config must use the same hub names the tensors were
        # written under, or a loader will treat an excluded BF16 layer as quantized.
        if self._name_mapper is not None and quant_config:
            with contextlib.suppress(Exception):
                revert_quant_config_names(quant_config.get("quantization", {}), self._name_mapper)

        tail: dict[str, torch.Tensor] = {}
        seen_keys: set[str] = set()
        handled_ids: set[int] = set()
        # Decoder tensors are already in their own shards.
        skip_prefixes = tuple(f"{n}." for n in self._layer_names.values() if n)

        # Non-decoder modules whose weights are not directly readable -- embeddings, norms,
        # lm_head on an offloaded model. Each needs its own materialization window, or its
        # tensors are still on meta here and _collect drops them silently. Containers are
        # skipped: their children get their own window.
        for name, module in model.named_modules():
            if id(module) in self._decoder_owned_ids:
                continue
            if not requires_weight_materialization(module, model, self._name_to_module):
                continue
            with enable_weight_access_and_writeback(
                module, model, self._name_to_module, writeback=False
            ):
                for sub_name, sub_mod in module.named_modules():
                    full_name = f"{name}.{sub_name}" if sub_name else name
                    _dispatch_export_handler(full_name, sub_mod, self._ctx)
                    handled_ids.add(id(sub_mod))
                prefix = f"{name}." if name else ""
                for key, tensor in module.state_dict().items():
                    seen_keys.add(prefix + key)
                    self._collect(tail, prefix + key, tensor)

        # Everything already resident. On a model with no offload this is the whole tail.
        for name, module in model.named_modules():
            if id(module) in self._decoder_owned_ids or id(module) in handled_ids:
                continue
            if _holds_meta_tensor(module):
                # requires_weight_materialization said no window was needed, yet the weights
                # are not here. Packing would raise deep inside the export handler; skipping
                # would drop the tensor silently. Neither is acceptable.
                raise RuntimeError(
                    f"{name!r} holds meta tensors but was not offered a materialization "
                    "window, so its weights cannot be exported. Export without export_dir "
                    "and use export_hf_checkpoint() for this model."
                )
            _dispatch_export_handler(name, module, self._ctx)
        for name, tensor in model.state_dict().items():
            if name.startswith(skip_prefixes) or name in seen_keys:
                continue
            self._collect(tail, name, tensor)

        # Tensors the model never held -- e.g. MTP weights, which HF leaves orphaned because
        # it only builds num_hidden_layers decoders. Already materialized and already in
        # export form, so only the hub-name reversal applies.
        for name, tensor in (extra_state_dict or {}).items():
            mapped = self._name_mapper(name) if self._name_mapper is not None else name
            tail.setdefault(mapped, tensor.detach().contiguous().cpu())

        save_file(_copy_storage_aliases(tail), str(self._export_dir / _TAIL_SHARD))
        self._write_index()
        save_non_weight_artifacts(model, self._export_dir)
        _write_hf_export_config(model, quant_config, self._export_dir)
        return quant_config

    def assert_shards_present(self, upto: int) -> None:
        """Require shards for layers ``[0, upto)``, which a resume intends to skip.

        Calibration resumes from its own checkpoint directory, which knows nothing about
        what was exported. If the two were produced by different runs, the skipped layers
        have no shards and the gap would only surface at :meth:`finalize`, after the whole
        calibration had run. Fail before any of that work instead.
        """
        missing = [i for i in range(upto) if not (self._export_dir / layer_shard_name(i)).exists()]
        if missing:
            raise RuntimeError(
                f"Resuming calibration at layer {upto} would skip layers {missing}, but "
                f"their shards are missing from {self._export_dir}. The checkpoint and "
                "export directories are from different runs; delete one and restart."
            )

    def _collect(self, out: dict[str, torch.Tensor], full_key: str, tensor: torch.Tensor) -> None:
        """Apply per-tensor export postprocessing and hub-name reversal, or drop the tensor."""
        from .quant_utils import _postprocess_single_tensor

        if tensor is None or tensor.is_meta:
            return
        new_key, new_value = _postprocess_single_tensor(
            full_key, tensor, 448, self._kv_cache_format, self._ctx.is_modelopt_qlora
        )
        if new_key is None or new_value is None:
            return
        if self._name_mapper is not None:
            new_key = self._name_mapper(new_key)
        if new_key in self._tied_alias_keys:
            return
        out[new_key] = new_value.detach().contiguous().cpu()

    def _write_index(self) -> None:
        """Build ``model.safetensors.index.json`` by reading back the shards on disk.

        Read from disk rather than accumulated in memory, because shards this run resumed
        past were never seen by this process. Enumerated from the layer count rather than
        globbed, so leftovers from a longer previous run cannot leak into the index.
        """
        from safetensors import safe_open

        shards = [self._export_dir / layer_shard_name(i) for i in range(len(self._layers))]
        shards.append(self._export_dir / _TAIL_SHARD)

        weight_map: dict[str, str] = {}
        total_size = 0
        for shard in shards:
            with safe_open(str(shard), framework="pt") as f:
                for key in f.keys():  # noqa: SIM118 -- safe_open has no __iter__
                    weight_map[key] = shard.name
            total_size += _shard_data_bytes(shard)
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        (self._export_dir / _INDEX_FILE).write_text(json.dumps(index, indent=2))


def _holds_meta_tensor(module: nn.Module) -> bool:
    """Whether this module's own parameters or buffers are still on meta."""
    return any(
        t is not None and t.is_meta
        for t in (*module._parameters.values(), *module._buffers.values())
    )


def _copy_storage_aliases(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy tensors that share storage with an earlier key.

    ``save_file`` rejects two keys backed by the same storage, and ``_collect``'s ``.cpu()``
    is a no-op rather than a copy when the tensor is already there. Copy rather than drop:
    every key has to survive. Mirrors ``_StreamingShardWriter.add``.
    """
    seen: set[int] = set()
    for key, tensor in tensors.items():
        if tensor.data_ptr() in seen:
            tensors[key] = tensor.clone()
        else:
            seen.add(tensor.data_ptr())
    return tensors


def _shard_data_bytes(path: Path) -> int:
    """Payload size of a safetensors file, excluding its header.

    Layout is an 8-byte little-endian header length, that much JSON, then tensor data.
    Subtracting is exact and avoids a dtype-size table that would have to track every
    safetensors dtype name.
    """
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
    return path.stat().st_size - 8 - header_len
