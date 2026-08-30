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
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.utils.core_utils import (
    enable_weight_access_and_writeback,
    requires_weight_materialization,
)
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
from modelopt.torch.utils import distributed as dist

from .layer_utils import is_moe, sync_moe_gate_up_amax
from .model_config import FUSION_FREE_FORMATS, QUANTIZATION_NVFP4
from .model_utils import TiedWeightMap
from .quant_aware_conversion import build_reverse_name_mapper, revert_quant_config_names
from .quant_utils import _postprocess_single_tensor, get_quant_config, get_quantization_format
from .registry import ExportContext, PrepareMoEInputsRegistry
from .unified_export_hf import (
    _add_mtp_exclusions,
    _dispatch_export_handler,
    _fuse_shared_input_modules,
    _prepare_moe_inputs,
    _resolve_export_dtype,
    _write_hf_export_config,
    collect_shared_input_modules,
    save_non_weight_artifacts,
)
from .unified_export_hf_streaming import _assert_no_split_rules

# Fusable per layer because the groups (q/k/v, gate/up) never cross a decoder boundary.
# AWQ and SVDQuant also need pre-quant-scale steps, which are still whole-model.
_PER_LAYER_FUSABLE_FORMATS = frozenset({QUANTIZATION_NVFP4})

SUPPORTED_FORMATS = FUSION_FREE_FORMATS | _PER_LAYER_FUSABLE_FORMATS

_TAIL_SHARD = "model-tail.safetensors"
_INDEX_FILE = "model.safetensors.index.json"


def layer_shard_name(layer_idx: int) -> str:
    """Shard filename for one decoder layer, keyed by index so a re-export overwrites."""
    return f"model-layer-{layer_idx:05d}.safetensors"


def _is_quantized_module(module: nn.Module) -> bool:
    """By type, not name: fused experts name theirs ``gate_up_proj_weight_quantizer``."""
    return any(
        isinstance(child, (TensorQuantizer, SequentialQuantizer)) for child in module.children()
    )


def _module_formats(model: nn.Module) -> set:
    """Every distinct format present. ``get_quantization_format`` stops at the first."""
    return {
        get_quantization_format(module)
        for _, module in model.named_modules()
        if _is_quantized_module(module)
    }


def _tied_quantized_modules(model: nn.Module) -> list[str]:
    """Quantized modules sharing a weight with another.

    Grouped by name, which survives offload: a ``data_ptr`` grouping sees nothing when the
    weights are on meta and would pass vacuously. Falls back to ``data_ptr`` when the model
    publishes no map (transformers < 5).
    """
    tied_map = TiedWeightMap(model)
    groups: dict[str, list[str]] = {}
    by_ptr: dict[int, list[str]] = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or not _is_quantized_module(module):
            continue
        key = tied_map.group_key(f"{name}.weight")
        if key is not None:
            groups.setdefault(key, []).append(name)
        elif not weight.is_meta and weight.data_ptr():
            # data_ptr() is 0 for meta tensors and DTensors, grouping unrelated modules.
            by_ptr.setdefault(weight.data_ptr(), []).append(name)
    tied = {n for names in groups.values() if len(names) > 1 for n in names}
    tied |= {n for names in by_ptr.values() if len(names) > 1 for n in names}
    return sorted(tied)


def assert_formats_supported(module: nn.Module, scope: str) -> None:
    """Raise unless every format in ``module`` can be reproduced per layer.

    Called before calibration to fail early, and again per exported layer -- AWQ and
    SVDQuant only become visible once the calibrator registers their discriminators.
    """
    unsupported = sorted(str(f) for f in _module_formats(module) - SUPPORTED_FORMATS)
    if unsupported:
        raise NotImplementedError(
            f"layerwise export does not support quantization format(s) {unsupported} "
            f"({scope}): they need requantize_resmooth_fused_llm_layers' pre-quant-scale "
            "steps, which are still whole-model. Supported today: "
            f"{sorted(str(f) for f in SUPPORTED_FORMATS if f)}."
        )


def assert_layerwise_export_supported(model: nn.Module) -> None:
    """Raise unless per-layer export is valid for this model."""
    assert_formats_supported(model, "before calibration")

    tied = _tied_quantized_modules(model)
    if tied:
        raise NotImplementedError(
            f"layerwise export does not support weight-tied quantized modules {tied[:6]}: "
            "the whole-model path merges their input_quantizer amaxes via "
            "sync_tied_input_amax so both sides share one input_scale, which a per-layer "
            "pass cannot do because a tie partner may be uncalibrated or already written. "
            "Conversion quantizes every nn.Linear and nn.Embedding, so disabling their "
            "quantizers does not lift this -- tie_word_embeddings models need "
            "export_hf_checkpoint()."
        )

    if dist.is_initialized() and dist.size() > 1:
        raise NotImplementedError(
            "layerwise export does not support multi-process jobs (e.g. FSDP2): every rank "
            "would write the same shard files. Use single-process calibration."
        )


class LayerwiseExporter:
    """Writes one decoder layer's quantized shard per call, then the tail and index.

    Built before calibration, driven per layer, finalized after the last. ``finalize``
    indexes the shards on disk, so an earlier run's layers are picked up as they are.
    """

    def __init__(
        self,
        model: nn.Module,
        export_dir: Path | str,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Validate support and capture model-level state.

        Runs before calibration, so nothing amax-dependent exists yet.
        """
        assert_layerwise_export_supported(model)
        # Splits regroup tensors across the whole state dict; no per-layer pass reverses that.
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

        self._ctx = ExportContext(model=model, dtype=_resolve_export_dtype(model, dtype))

        self._export_dir = Path(export_dir)
        self._export_dir.mkdir(parents=True, exist_ok=True)
        # Read here, not in finalize(): it reports on the quantizer modules, which
        # export_layer replaces as it goes, so by finalize() the model looks unquantized.
        self._quant_config = get_quant_config(model, is_modelopt_qlora=self._ctx.is_modelopt_qlora)
        # Not get_kv_cache_dtype: it does not recurse, so on the root it answers None.
        self._kv_cache_format = self._quant_config["quantization"]["kv_cache_quant_algo"]
        self._finalized = False

        self._name_mapper = None
        try:
            self._name_mapper = build_reverse_name_mapper(model)
        except Exception as exc:
            warnings.warn(
                f"Reverse name mapper unavailable ({exc}); exported tensor names may not "
                "match the original HF hub checkpoint."
            )

    def export_layer(
        self,
        layer_idx: int,
        layer_module: nn.Module,
        layer_inputs: list | None = None,
    ) -> None:
        """Pack one calibrated layer into its shard, converting it in place.

        ``layer_inputs`` are the layer's cached calibration activations, replayed once so
        a fusing format can rediscover which modules share an input; omit them only when
        nothing fuses.
        """
        # Local, as in every other export module: the plugin imports transformers.
        from modelopt.torch.quantization.plugins.huggingface import _reconstruct_fused_moe_linear

        assert not self._finalized, "export_layer() called after finalize()"
        if layer_module is not self._layers[layer_idx]:
            # Not an assert: -O would strip it, and the failure is silent -- layer N's
            # tensors land in layer M's shard and the index looks perfectly well formed.
            raise RuntimeError(
                f"layer_idx {layer_idx} does not match the module passed; calibration and "
                "export disagree on decoder layer order."
            )

        assert_formats_supported(layer_module, "once calibrated")

        layer_name = self._layer_names[layer_idx]
        tensors: dict[str, torch.Tensor] = {}

        # Order matters at both seams: scales derive from amax, so they must be final
        # before packing, and the restack consumes packed per-expert tensors.
        _prepare_moe_inputs(layer_module, self._ctx.dtype, self._ctx.is_modelopt_qlora)
        self._unify_shared_quantization_params(layer_module, layer_inputs)

        for sub_name, sub_mod in layer_module.named_modules():
            full_name = f"{layer_name}.{sub_name}" if sub_name else layer_name
            _dispatch_export_handler(full_name, sub_mod, self._ctx)
        _reconstruct_fused_moe_linear(layer_module)

        prefix = f"{layer_name}." if layer_name else ""
        for key, tensor in layer_module.state_dict().items():
            self._collect(tensors, prefix + key, tensor)

        save_file(tensors, str(self._export_dir / layer_shard_name(layer_idx)))

    def _unify_shared_quantization_params(
        self, layer_module: nn.Module, layer_inputs: list | None
    ) -> None:
        """Unify the quantization parameters across the modules a fused kernel merges.

        The per-layer half of ``requantize_resmooth_fused_llm_layers``: one input scale per
        shared-input group, one weight_scale_2 per expert gate/up pair. Its pre-quant-scale
        steps are AWQ/SVDQuant-only and refused.
        """
        # A set, not get_quantization_format: that stops at the first hit, so a mixed
        # FP8-attention/NVFP4-expert layer would report fp8 and skip fusing entirely.
        if _module_formats(layer_module) - FUSION_FREE_FORMATS:
            self._fuse_shared_input_scales(layer_module, layer_inputs)
        sync_moe_gate_up_amax(layer_module)

    def _fuse_shared_input_scales(self, layer_module: nn.Module, layer_inputs: list | None) -> None:
        """Rediscover the groups that share an input, on real activations, and fuse them."""
        layer_format = get_quantization_format(layer_module)
        if not layer_inputs:
            raise RuntimeError(
                f"layer format {layer_format!r} needs input-sharing groups to fuse its "
                "scales, but no layer_inputs were supplied to rediscover them."
            )

        args, kwargs = layer_inputs[0]
        input_to_linear, _ = collect_shared_input_modules(
            layer_module, lambda: layer_module(*args, **kwargs)
        )
        _fuse_shared_input_modules(
            self._ctx.model, input_to_linear, quantization_format=layer_format
        )

    def finalize(self) -> dict:
        """Export the tail, write the config artifacts, and index all shards.

        Leaves ``export_dir`` a complete checkpoint; no ``export_hf_checkpoint()`` needed.
        """
        assert not self._finalized, "finalize() called twice"
        self._finalized = True

        model = self._ctx.model
        quant_config = self._quant_config
        _add_mtp_exclusions(model, quant_config)
        # No gate/up sync here: export_layer did every layer, and the tail has no experts.
        if getattr(model, "hf_quantizer", None) is not None:
            model.hf_quantizer = None
        # Names must match the tensors', or a loader reads an excluded BF16 layer as quantized.
        if self._name_mapper is not None and quant_config:
            with contextlib.suppress(Exception):
                revert_quant_config_names(quant_config.get("quantization", {}), self._name_mapper)

        name_to_module = dict(model.named_modules())
        # Recomputed, not snapshotted in __init__: calibration adds modules inside the
        # layers (SharedQuantState), and a stale set would leave them to the tail pass.
        decoder_owned_ids = {id(m) for layer in self._layers for m in layer.modules()}

        tail: dict[str, torch.Tensor] = {}
        seen_keys: set[str] = set()
        handled_ids: set[int] = set()
        # Decoder tensors are already in their own shards.
        skip_prefixes = tuple(f"{n}." for n in self._layer_names.values() if n)

        # Offloaded tail modules are on meta and _collect drops meta silently, so each
        # needs its own materialization window.
        for name, module in model.named_modules():
            if id(module) in decoder_owned_ids:
                continue
            if not requires_weight_materialization(module, model, name_to_module):
                continue
            with enable_weight_access_and_writeback(module, model, name_to_module, writeback=False):
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
            if id(module) in decoder_owned_ids or id(module) in handled_ids:
                continue
            if _holds_meta_tensor(module):
                # Packing would raise deep in the handler; skipping would drop it silently.
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

        save_file(tail, str(self._export_dir / _TAIL_SHARD))
        self._write_index()
        save_non_weight_artifacts(model, self._export_dir)
        _write_hf_export_config(model, quant_config, self._export_dir)
        warnings.warn(
            "The exported checkpoint is complete, but per-layer export leaves the model in "
            "export form: it must not be used for inference."
        )
        return quant_config

    def completed_layers(self) -> int:
        """How many leading layers have a shard. Contiguous: a gap means the rest never ran."""
        n = 0
        while (self._export_dir / layer_shard_name(n)).exists():
            n += 1
        return n

    def assert_no_orphan_shards(self) -> None:
        """Refuse to redo work when shards exist but no usable resume record does."""
        done = self.completed_layers()
        if not done:
            return
        raise RuntimeError(
            f"{self._export_dir} already holds shards for layers 0..{done - 1}, but the "
            "layerwise checkpoint directory has no usable resume record, so calibration "
            "would restart at layer 0 and overwrite them. Either restore the checkpoint "
            f"directory that produced these shards, or delete {self._export_dir} to "
            "re-export."
        )

    def assert_shards_present(self, upto: int) -> None:
        """Require shards for layers ``[0, upto)``, which a resume intends to skip.

        Otherwise a mismatched checkpoint/export pair only surfaces after a full run.
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
        if tensor is None or tensor.is_meta:
            return
        new_key, new_value = _postprocess_single_tensor(
            full_key, tensor, 448, self._kv_cache_format, self._ctx.is_modelopt_qlora
        )
        if new_key is None or new_value is None:
            return
        if self._name_mapper is not None:
            new_key = self._name_mapper(new_key)
        out[new_key] = new_value.detach().contiguous().cpu()

    def _write_index(self) -> None:
        """Build ``model.safetensors.index.json`` from the shards on disk.

        From disk because a resumed run never saw the earlier shards in memory; by layer
        count rather than a glob, so a longer previous run's leftovers cannot leak in.
        """
        # Out of the index already; delete them so the directory *is* the checkpoint.
        for stale in self._export_dir.glob("model-layer-*.safetensors"):
            if int(stale.stem.rsplit("-", 1)[1]) >= len(self._layers):
                stale.unlink()

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


def _shard_data_bytes(path: Path) -> int:
    """Payload size of a safetensors file: total minus the 8-byte prefix and header."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
    return path.stat().st_size - 8 - header_len
