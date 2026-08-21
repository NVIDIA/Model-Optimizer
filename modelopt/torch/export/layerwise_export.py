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
import hashlib
import json
import re
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .model_config import FUSION_FREE_FORMATS, QUANTIZATION_NVFP4
from .quant_utils import get_quant_config, get_quantization_format

# Fusing formats this path can handle itself. The groups _fuse_shared_input_modules works
# on -- q/k/v behind input_layernorm, gate/up behind post_attention_layernorm -- live
# inside one decoder layer, so export_layer rediscovers them per layer instead of needing
# the whole-model forward. AWQ and SVDQuant are excluded: they additionally need
# requantize_resmooth_fused_llm_layers' pre-quant-scale steps, which are still model-wide.
_PER_LAYER_FUSABLE_FORMATS = frozenset({QUANTIZATION_NVFP4})

SUPPORTED_FORMATS = FUSION_FREE_FORMATS | _PER_LAYER_FUSABLE_FORMATS

_TAIL_SHARD = "model-tail.safetensors"
_INDEX_FILE = "model.safetensors.index.json"
_IDENTITY_FILE = ".layerwise_export.json"


def layer_shard_name(layer_idx: int) -> str:
    """Shard filename for one decoder layer.

    Keyed by index, not a counter, so re-exporting overwrites rather than leaving a stale
    shard for the index to find.
    """
    return f"model-layer-{layer_idx:05d}.safetensors"


def _is_quantized_module(module: nn.Module) -> bool:
    """Whether this module carries quantizers of its own.

    By type, not attribute name: fused experts name theirs after the weight
    (``gate_up_proj_weight_quantizer``), which a name test misses.
    """
    from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer

    return any(
        isinstance(child, (TensorQuantizer, SequentialQuantizer)) for child in module.children()
    )


def _fuse_unrouted_experts(layer_module: nn.Module, fused_linears: dict[str, list[str]]) -> None:
    """Fuse the sibling experts the probe never routed a token to.

    ``sync_moe_gate_up_amax`` covers ``weight_quantizer.amax`` but not a static
    quantizer's ``global_amax``, so unrouted pairs would keep unmerged scales.
    """
    from .quant_utils import preprocess_linear_fusion

    names = {name for name, _ in layer_module.named_modules()}
    for group, members in fused_linears.items():
        if not re.search(r"experts?\.\d+", group):
            continue
        expert_id = 0
        while True:
            sibling = re.sub(r"(experts?\.)\d+", rf"\g<1>{expert_id}", group, count=1)
            if sibling in fused_linears:  # the probe already fused this one
                expert_id += 1
                continue
            if sibling not in names:
                break
            preprocess_linear_fusion(
                [
                    layer_module.get_submodule(
                        re.sub(r"(experts?\.)\d+", rf"\g<1>{expert_id}", member)
                    )
                    for member in members
                ]
            )
            expert_id += 1


def _module_formats(model: nn.Module) -> set:
    """Every distinct format present. ``get_quantization_format`` stops at the first."""
    return {
        get_quantization_format(module)
        for _, module in model.named_modules()
        if _is_quantized_module(module)
    }


def _tied_quantized_modules(model: nn.Module) -> list[str]:
    """Quantized modules sharing a weight with another.

    The whole-model export merges their amaxes via ``sync_tied_input_amax``; a per-layer
    pass cannot, since a partner may be uncalibrated or already written.

    Grouped by name via :class:`TiedWeightMap`, which survives offload -- a ``data_ptr``
    grouping sees nothing when the weights are on meta and would pass vacuously. Falls back
    to ``data_ptr`` when the map is empty (transformers <5.0 does not publish one).
    """
    from .model_utils import TiedWeightMap

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

    Called twice, because AWQ and SVDQuant only become visible once the calibrator has
    registered ``_pre_quant_scale`` / ``svdquant_lora_a``: before calibration to fail
    early, and on each exported layer, which is the authority.
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
    """Raise unless per-layer export is valid for this model.

    Structural cases only; each would otherwise differ from a whole-model export without
    failing. Formats are settled by :func:`assert_formats_supported`.
    """
    from modelopt.torch.utils import distributed as dist

    assert_formats_supported(model, "before calibration")

    tied = _tied_quantized_modules(model)
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
    """Undo the parameter, buffer and child-module changes export makes to ``module``.

    A resident model has no materialization window to discard what export leaves behind.
    Restoring the dicts covers rebinding; buffers are additionally clone-restored because
    scale fusion writes amax in place. Parameters are not -- that would copy the layer.
    Attributes outside those three dicts are not restored (e.g. the layer-relative ``name``
    that ``collect_shared_input_modules`` sets), so handlers must keep their state there.
    """
    snapshot = [
        (
            m,
            dict(m._parameters),
            {k: (v.clone() if v is not None else None) for k, v in m._buffers.items()},
            dict(m._modules),
        )
        for m in module.modules()
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

    Built before calibration, driven per layer from inside the window calibration opens,
    finalized after the last. ``finalize`` indexes the shards on disk, so layers an earlier
    run exported are picked up without being re-exported.
    """

    def __init__(
        self,
        model: nn.Module,
        export_dir: Path | str,
        dtype: torch.dtype | None = None,
        is_modelopt_qlora: bool = False,
    ) -> None:
        """Validate support and capture model-level state, before calibration runs.

        Only quantizer *configuration* is read here; anything amax-dependent belongs in
        :meth:`finalize`.
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
        # Materialization dispatch rebuilds this map per call when not supplied.
        self._name_to_module = dict(model.named_modules())

        # Sole owner of model/dtype/qlora: parallel copies on self would be a second
        # source of truth. Tie state is no longer carried here -- the driver owns a
        # name-based TiedWeightMap instead.
        self._ctx = ExportContext(
            model=model,
            dtype=_resolve_export_dtype(model, dtype),
            is_modelopt_qlora=is_modelopt_qlora,
        )

        self._export_dir = Path(export_dir)
        self._export_dir.mkdir(parents=True, exist_ok=True)
        # Not get_kv_cache_dtype(model): it does not recurse, so given the root it answers
        # None, which trips the KV assert in the per-tensor pass.
        # One walk, shared with _bind_identity: get_quant_config re-derives every module's
        # format, which is not free on a large MoE.
        quant_config = get_quant_config(model, is_modelopt_qlora=is_modelopt_qlora)
        self._kv_cache_format = quant_config["quantization"]["kv_cache_quant_algo"]
        self._finalized = False

        self._name_mapper = None
        try:
            self._name_mapper = build_reverse_name_mapper(model)
        except Exception as exc:
            warnings.warn(
                f"Reverse name mapper unavailable ({exc}); exported tensor names may not "
                "match the original HF hub checkpoint."
            )
        # By name, not data_ptr: layers and tail are separate passes.
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

        self._bind_identity(quant_config)

    def export_layer(
        self,
        layer_idx: int,
        layer_module: nn.Module,
        probe_forward: Callable[[nn.Module], None] | None = None,
    ) -> None:
        """Pack one calibrated layer into its shard, leaving that layer untouched.

        ``probe_forward`` runs the layer on real activations so a fusing format can
        rediscover which modules share an input; only omit it when nothing fuses.
        """
        from modelopt.torch.quantization.plugins.huggingface import _reconstruct_fused_moe_linear

        from .layer_utils import sync_moe_gate_up_amax
        from .unified_export_hf import _dispatch_export_handler, _prepare_moe_inputs

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
        with transient_module_state(layer_module):
            # Per-block: earlier there is no amax yet, later the layer is written.
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

        save_file(
            _copy_storage_aliases(tensors), str(self._export_dir / layer_shard_name(layer_idx))
        )

    def _fuse_shared_inputs(
        self, layer_module: nn.Module, probe_forward: Callable[[nn.Module], None] | None
    ) -> None:
        """Unify scales across the modules of this layer that share an input.

        Parity with ``requantize_resmooth_fused_llm_layers``, whose steps are all
        intra-layer, so a per-layer pass can reproduce them:

        ==============================  ==========================================
        Whole-model step                Here
        ==============================  ==========================================
        ``fuse_prequant_to_linear``     refused (AWQ/SVDQuant)
        layernorm pre_quant_scale fold  refused (AWQ/SVDQuant)
        MoE expert resmooth             refused (AWQ/SVDQuant)
        shared-input group fusion       this method, on real activations
        sibling-expert replay           :func:`_fuse_unrouted_experts`
        gate/up amax sync               ``sync_moe_gate_up_amax`` in export_layer
        ==============================  ==========================================
        """
        from .quant_utils import get_quantization_format
        from .unified_export_hf import _fuse_shared_input_modules, collect_shared_input_modules

        # Per-module scan: get_quantization_format returns the first format found, so an
        # FP8-attention/NVFP4-expert layer would report fp8 and skip fusing. The value
        # passed below is only a fallback -- _fuse_shared_input_modules re-checks per group.
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
        fused = _fuse_shared_input_modules(
            self._ctx.model, input_to_linear, quantization_format=layer_format
        )
        _fuse_unrouted_experts(layer_module, fused)

    def finalize(self, extra_state_dict: dict[str, torch.Tensor] | None = None) -> dict:
        """Export the tail, write the config artifacts, and index all shards.

        Leaves ``export_dir`` a complete checkpoint; no ``export_hf_checkpoint()`` needed.
        """
        from modelopt.torch.quantization.utils.core_utils import (
            enable_weight_access_and_writeback,
            requires_weight_materialization,
        )

        from .quant_aware_conversion import revert_quant_config_names
        from .unified_export_hf import (
            _add_mtp_exclusions,
            _dispatch_export_handler,
            _write_hf_export_config,
            save_non_weight_artifacts,
        )

        assert not self._finalized, "finalize() called twice"
        # Unlike export_layer, the tail is converted in place: its tensors are read from
        # model.state_dict() after dispatch, so restoring first would collect unpacked
        # weights. The model is left in export form and must not be used for inference.
        self._finalized = True

        model = self._ctx.model
        quant_config = get_quant_config(model, is_modelopt_qlora=self._ctx.is_modelopt_qlora)
        _add_mtp_exclusions(model, quant_config)
        # No _warn_on_unsynced_moe_gate_up: export_layer syncs inside
        # transient_module_state, so the shards are synced and the live model is not.
        if getattr(model, "hf_quantizer", None) is not None:
            model.hf_quantizer = None
        # Config module names must match the tensors', or a loader treats an excluded
        # BF16 layer as quantized.
        if self._name_mapper is not None and quant_config:
            with contextlib.suppress(Exception):
                revert_quant_config_names(quant_config.get("quantization", {}), self._name_mapper)

        tail: dict[str, torch.Tensor] = {}
        seen_keys: set[str] = set()
        handled_ids: set[int] = set()
        # Decoder tensors are already in their own shards.
        skip_prefixes = tuple(f"{n}." for n in self._layer_names.values() if n)

        # Offloaded embeddings/norms/lm_head are on meta here, and _collect would drop them
        # silently; each needs its own window. Containers are skipped -- children get one.
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
                # No window was offered yet the weights are absent. Packing would raise
                # deep in the handler, skipping would drop it silently.
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

        # Tensors the model never held (e.g. orphaned MTP weights), already in export
        # form, so only the hub-name reversal applies.
        for name, tensor in (extra_state_dict or {}).items():
            mapped = self._name_mapper(name) if self._name_mapper is not None else name
            tail.setdefault(mapped, tensor.detach().contiguous().cpu())

        save_file(_copy_storage_aliases(tail), str(self._export_dir / _TAIL_SHARD))
        self._write_index()
        save_non_weight_artifacts(model, self._export_dir)
        _write_hf_export_config(model, quant_config, self._export_dir)
        return quant_config

    def _bind_identity(self, quant_config: dict) -> None:
        """Tie the shards to the run that produced them.

        The manifest records no model or quantization identity and
        :meth:`assert_shards_present` only checks existence, so one run's manifest could
        otherwise finalize another's shards into a valid-looking, wrong checkpoint.
        """
        model = self._ctx.model
        # The quant config carries the per-module contract, so two runs differing in which
        # modules are quantized hash differently even when the format names match.
        quant_contract = hashlib.sha256(
            json.dumps(
                quant_config,
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()[:16]
        identity = {
            "model_class": type(model).__name__,
            # Digesting the weights would mean reading the whole model, so retrained
            # weights at the same path are not distinguished.
            "source": str(getattr(model.config, "_name_or_path", "") or ""),
            "num_layers": len(self._layers),
            "formats": sorted(str(f) for f in _module_formats(model) if f),
            "kv_cache": str(self._kv_cache_format),
            "quant_contract": quant_contract,
        }
        path = self._export_dir / _IDENTITY_FILE
        previous = json.loads(path.read_text()) if path.exists() else None
        # Only committed shards make it binding: this is written before layer 0, so a run
        # that died early would otherwise poison the directory with nothing to protect.
        if previous is not None and previous != identity and self.completed_layers():
            differing = sorted(
                k for k in set(previous) | set(identity) if previous.get(k) != identity.get(k)
            )
            raise RuntimeError(
                f"{self._export_dir} holds shards from a different run (differing: "
                f"{differing}); resuming would finalize them against this run's "
                "manifest. Use a fresh export directory."
            )
        if previous != identity:
            path.write_text(json.dumps(identity, indent=2))

    def completed_layers(self) -> int:
        """How many leading layers already have a shard on disk.

        A *contiguous* run from layer 0: a gap means the layers after it were never
        finished, and resuming past one would leave the checkpoint missing them.
        """
        n = 0
        while (self._export_dir / layer_shard_name(n)).exists():
            n += 1
        return n

    def assert_no_orphan_shards(self, manifest_present: bool) -> None:
        """Refuse to silently redo work when shards exist but the resume record does not.

        The resume point comes from the manifest -- the shards cannot supply it. Without
        one ``start_layer`` is 0, :meth:`assert_shards_present` checks an empty range, and
        calibration overwrites every finished layer without a word.
        """
        if manifest_present:
            return
        done = self.completed_layers()
        if not done:
            return
        raise RuntimeError(
            f"{self._export_dir} already holds shards for layers 0..{done - 1}, but the "
            "layerwise checkpoint directory has no manifest, so calibration would restart "
            "at layer 0 and overwrite them. Either restore the checkpoint directory that "
            f"produced these shards, or delete {self._export_dir} to re-export."
        )

    def assert_shards_present(self, upto: int) -> None:
        """Require shards for layers ``[0, upto)``, which a resume intends to skip.

        The checkpoint directory knows nothing about what was exported, so a mismatched
        pair would only surface at :meth:`finalize` -- after a full calibration.
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
        """Build ``model.safetensors.index.json`` from the shards on disk.

        From disk because shards this run resumed past were never seen in memory; from the
        layer count rather than a glob, so a longer previous run's leftovers cannot leak in.
        """
        from safetensors import safe_open

        # A longer previous run's shards are already out of the index; delete them too,
        # so the directory *is* the checkpoint rather than the checkpoint plus leftovers.
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


def _copy_storage_aliases(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy tensors sharing storage with an earlier key.

    ``save_file`` rejects those, and ``_collect``'s ``.cpu()`` is a no-op when the tensor is
    already there. Copy rather than drop: every key must survive.
    """
    seen: set[int] = set()
    for key, tensor in tensors.items():
        if tensor.data_ptr() in seen:
            tensors[key] = tensor.clone()
        else:
            seen.add(tensor.data_ptr())
    return tensors


def _shard_data_bytes(path: Path) -> int:
    """Payload size of a safetensors file: total minus its 8-byte length prefix and header.

    Exact, and avoids a dtype-size table that would have to track every safetensors name.
    """
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
    return path.stat().st_size - 8 - header_len
