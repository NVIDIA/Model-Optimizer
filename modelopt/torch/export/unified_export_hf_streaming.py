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

"""Streaming HF checkpoint export for disk/CPU-offloaded models.

Kept apart from :mod:`unified_export_hf` so the resident exporter cannot drift back into
being offload-aware: the only edge between them is the dispatch in
``export_hf_checkpoint``, which imports :func:`_export_transformers_checkpoint_streaming`
lazily to keep the dependency acyclic.
"""

import itertools
import json
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .quant_aware_conversion import build_reverse_name_mapper
from .quant_utils import _postprocess_single_tensor, get_quant_config
from .registry import ExportContext
from .unified_export_hf import (
    _add_mtp_exclusions,
    _dispatch_export_handler,
    _prepare_moe_inputs,
    _resolve_export_dtype,
    _warn_on_unsynced_moe_gate_up,
    requantize_resmooth_fused_llm_layers,
    save_non_weight_artifacts,
)

__all__ = ["_export_transformers_checkpoint_streaming"]


class _StreamingShardWriter:
    """Write tensors to safetensors shard files without accumulating the full state dict.

    Buffers tensors up to ``max_shard_size`` bytes, flushes to a numbered temp file, then
    at :meth:`finalize` renames temp files to canonical shard names once the total shard
    count is known.

    Peak memory = 1 layer (being materialized) + 1 shard buffer, not the full checkpoint.
    """

    def __init__(self, export_dir: Path | str, max_shard_size: int) -> None:
        self._export_dir = Path(export_dir)
        self._max_shard_size = max_shard_size
        self._buffer: dict[str, torch.Tensor] = {}
        self._buffer_bytes: int = 0
        self._part_files: list[Path] = []
        self._total_bytes: int = 0
        # Maps tensor key → part-file index (recorded at flush time)
        self._key_to_part: dict[str, int] = {}
        # data_ptr of every buffered tensor, so aliases never reach save_file.
        self._buffer_storage: set[int] = set()

    def _flush(self) -> None:
        if not self._buffer:
            return
        part_idx = len(self._part_files)
        part_path = self._export_dir / f"__shard_part_{part_idx:05d}.safetensors"
        save_file(self._buffer, str(part_path))
        for key in self._buffer:
            self._key_to_part[key] = part_idx
        self._part_files.append(part_path)
        self._total_bytes += self._buffer_bytes
        self._buffer = {}
        self._buffer_storage = set()
        self._buffer_bytes = 0

    def add(self, key: str, tensor: torch.Tensor) -> None:
        """Buffer a tensor, flushing the current shard to disk when it is full.

        ``save_file`` rejects tensors sharing storage, which two keys can still do here
        when the tensor reaches us already on CPU (so ``_stream_tensor``'s ``.cpu()`` was
        a no-op rather than a copy). Copy on collision rather than dropping one of them:
        offloaded export writes tied weights as separate entries, so every key must
        survive. ``data_ptr()`` only has to hold within one buffer, whose entries stay
        alive until :meth:`_flush`.
        """
        if tensor.data_ptr() in self._buffer_storage:
            tensor = tensor.clone()

        self._buffer_storage.add(tensor.data_ptr())
        self._buffer[key] = tensor
        self._buffer_bytes += tensor.nbytes
        if self._buffer_bytes >= self._max_shard_size:
            self._flush()

    def finalize(self) -> dict[str, str]:
        """Flush remaining buffer, rename part files, write model.safetensors.index.json.

        Returns the weight_map ``{key: shard_filename}`` written to the index.
        Single-shard exports use ``model.safetensors`` without an index file.
        """
        self._flush()
        n_shards = len(self._part_files)
        if n_shards == 0:
            return {}

        if n_shards == 1:
            final_name = "model.safetensors"
            self._part_files[0].rename(self._export_dir / final_name)
            return dict.fromkeys(self._key_to_part, final_name)

        for i, part_path in enumerate(self._part_files):
            part_path.rename(self._export_dir / f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors")

        weight_map = {
            key: f"model-{part_idx + 1:05d}-of-{n_shards:05d}.safetensors"
            for key, part_idx in self._key_to_part.items()
        }
        total_size = self._total_bytes
        index_path = self._export_dir / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f)
        return weight_map


def _parse_shard_size(size: int | str) -> int:
    """Convert a shard-size string (e.g. ``"10GB"``, ``"500MB"``) to bytes.

    Mirrors transformers' ``convert_file_size_to_int``, which reads ``GB``/``MB``/``KB``
    as decimal and only ``GiB``/``MiB``/``KiB`` as binary. That helper was removed from
    ``transformers.utils`` in transformers 5.x, so the fallback below is the live path
    there, not a rarely-taken branch.
    """
    try:
        from transformers.utils import convert_file_size_to_int

        return convert_file_size_to_int(size)
    except ImportError:
        pass
    if isinstance(size, int):
        return size
    s = size.strip().upper()
    for suffix, multiplier in (
        ("GIB", 1024**3),
        ("MIB", 1024**2),
        ("KIB", 1024),
        ("GB", 1000**3),
        ("MB", 1000**2),
        ("KB", 1000),
    ):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * multiplier)
    return int(s)


def _assert_no_split_rules(model: nn.Module) -> None:
    """Refuse to stream a model whose conversion mapping needs tensor-level splits."""
    from .quant_aware_conversion import _build_reverse_rules

    try:
        split_rules, _, _ = _build_reverse_rules(model)
    except Exception:
        return  # build_reverse_name_mapper reports the failure with a warning
    if split_rules:
        raise NotImplementedError(
            "Disk/CPU-offloaded export cannot reverse tensor-level split rules in this "
            "model's transformers conversion mapping: the streaming path reverses names "
            "per tensor, while splits need the full state dict. Export without offloading."
        )


def _export_transformers_checkpoint_streaming(
    model: nn.Module,
    dtype: torch.dtype | None = None,
    is_modelopt_qlora: bool = False,
    export_dir: Path | str = ".",
    max_shard_size: int | str = "10GB",
    extra_state_dict: dict[str, torch.Tensor] | None = None,
    **kwargs,
) -> tuple[None, dict[str, Any]]:
    """Export a disk/CPU-offloaded model by streaming tensors layer-by-layer to shard files.

    The offloaded counterpart of :func:`_export_transformers_checkpoint`, which builds the
    whole quantized state dict at once and so needs every weight resident. Here each
    decoder layer is materialized, exported, and written to a shard file before the next
    one is touched, bounding peak memory at one layer plus one shard buffer.

    Model-level preparation (MoE input handling, resmooth/requantize, quant config) matches
    the resident path. The per-tensor work does not: instead of ``postprocess_state_dict``
    over a finished dict, each tensor goes through :func:`_postprocess_single_tensor` as it
    is produced. Two consequences follow from having no whole-dict view:

    - Tied weights are dropped by *name* from ``_tied_weights_keys`` (data_ptr is meaningless
      once weights move host<->device); see the TODO below on adopting ``all_tied_weights_keys``.
    - Conversion mappings that need tensor-level splits cannot be reversed one tensor at a
      time, so they are rejected up front rather than exported incorrectly.

    Args:
        model: the full torch model to export, carrying accelerate offload hooks.
        dtype: weight dtype for unquantized layers, or the model's dtype if None.
        is_modelopt_qlora: whether the model is a ModelOpt QLoRA model.
        export_dir: directory to write shards and config artifacts into.
        max_shard_size: shard size limit, as bytes or a string such as ``"10GB"``.
        extra_state_dict: tensors the model itself never holds (e.g. MTP weights, which HF
            leaves orphaned) and which would otherwise be missing from the export.

    Returns:
        ``(None, quant_config)``. No state dict is returned because none is ever
        assembled; shards, ``config.json``, and ``generation_config.json`` are written to
        ``export_dir`` directly. The caller writes ``hf_quant_config.json`` and merges
        ``quantization_config`` into ``config.json``.

    Raises:
        NotImplementedError: if the model's conversion mapping contains split rules.
        RuntimeError: if decoder layers cannot be discovered for layer-wise materialization.
    """
    from modelopt.torch.quantization.plugins.huggingface import _reconstruct_fused_moe_linear
    from modelopt.torch.quantization.utils.core_utils import (
        enable_weight_access_and_writeback,
        requires_weight_materialization,
    )
    from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

    export_dir = Path(export_dir)
    # Materialization dispatch walks the module tree from the root; without this cache each
    # call re-derives it, which is O(N^2) over a MoE model's expert modules.
    name_to_module = dict(model.named_modules())

    # --- Same model-level setup as _export_transformers_checkpoint ---
    dtype = _resolve_export_dtype(model, dtype)
    _prepare_moe_inputs(model, dtype, is_modelopt_qlora)

    requantize_resmooth_fused_llm_layers(model)

    quant_config = get_quant_config(model, is_modelopt_qlora=is_modelopt_qlora)

    _add_mtp_exclusions(model, quant_config)

    _warn_on_unsynced_moe_gate_up(model)

    # --- Per-tensor constants ---
    kv_cache_max_bound = 448
    kv_cache_format = quant_config["quantization"]["kv_cache_quant_algo"]

    # --- Tied alias keys to skip ---
    # data_ptr() is unreliable for disk-offloaded weights, so we use _tied_weights_keys.
    # Only apply when tie_word_embeddings=True: _tied_weights_keys can list keys whose
    # weights are not actually shared (e.g. if the model was saved with tie_word_embeddings=False
    # but the attribute was never cleared), which would incorrectly drop lm_head.weight.
    #
    # TODO(tied-map): the resident path reads HF's ``all_tied_weights_keys`` (covers dict-style/MoE
    # ties); this path could too, to close the streaming gap for offloaded 5.x models -- but that
    # swap needs offload-specific validation (meta tensors, per-tensor order, disk round-trip) first.
    raw_tied_keys: set[str] = (
        set(getattr(model, "_tied_weights_keys", None) or [])
        if getattr(model.config, "tie_word_embeddings", False)
        else set()
    )

    # --- Name mapper for per-tensor key reversal ---
    # Tensor names are applied inline; quant config names are handled by the caller.
    # Renames are all a per-tensor pass can reverse. The batch path additionally runs
    # revert_weight_conversion_quant_aware() for split rules, which need the whole state
    # dict to regroup tensors, so refuse rather than emit fused tensors under unfused
    # hub keys.
    _assert_no_split_rules(model)
    name_mapper = None
    try:
        name_mapper = build_reverse_name_mapper(model)
    except Exception as exc:
        warnings.warn(
            f"Reverse name mapper unavailable ({exc}); exported tensor names may not match "
            "the original HF hub checkpoint."
        )

    tied_alias_keys: set[str] = (
        {name_mapper(k) for k in raw_tied_keys} if name_mapper is not None else raw_tied_keys
    )

    # --- Decoder layers ---
    decoder_layers = LayerActivationCollector.get_decoder_layers(model)
    if decoder_layers is None:
        raise RuntimeError(
            "Streaming export requires discoverable decoder layers. "
            "The model architecture is not supported by LayerActivationCollector."
        )
    decoder_layer_ids = {id(m) for m in decoder_layers}
    # Descendants too, not just the layers: an offloaded layer's children return to meta
    # when its window closes, so a child-level check would re-enter and re-export weights
    # this pass already packed.
    decoder_owned_ids = {id(m) for layer in decoder_layers for m in layer.modules()}

    # --- Persistent-buffer predicate (mirrors state_dict() which excludes non-persistent) ---
    def _is_persistent_buffer(name: str) -> bool:
        parts = name.split(".")
        mod: nn.Module = model
        for part in parts[:-1]:
            mod = getattr(mod, part, mod)
        return parts[-1] not in getattr(mod, "_non_persistent_buffers_set", frozenset())

    # --- Stream tensors to shard files ---
    shard_size_bytes = _parse_shard_size(max_shard_size)
    writer = _StreamingShardWriter(export_dir, shard_size_bytes)
    ctx = ExportContext(model=model, dtype=dtype, is_modelopt_qlora=is_modelopt_qlora)
    seen_keys: set[str] = set()

    def _stream_tensor(full_key: str, tensor: torch.Tensor) -> None:
        new_key, new_value = _postprocess_single_tensor(
            full_key, tensor, kv_cache_max_bound, kv_cache_format, is_modelopt_qlora
        )
        if new_key is None or new_value is None:
            return
        if name_mapper is not None:
            new_key = name_mapper(new_key)
        if new_key in tied_alias_keys:
            return
        writer.add(new_key, new_value.detach().contiguous().cpu())

    # Decoder layers: materialize one at a time
    for layer_name, layer_module in model.named_modules():
        if id(layer_module) not in decoder_layer_ids:
            continue
        with enable_weight_access_and_writeback(
            layer_module, model, name_to_module, writeback=False
        ):
            for sub_name, sub_mod in layer_module.named_modules():
                full_name = f"{layer_name}.{sub_name}" if sub_name else layer_name
                _dispatch_export_handler(full_name, sub_mod, ctx)
            _reconstruct_fused_moe_linear(layer_module)
            prefix = f"{layer_name}." if layer_name else ""
            for key, tensor in layer_module.state_dict().items():
                full_key = prefix + key
                if full_key in seen_keys:
                    continue
                seen_keys.add(full_key)
                _stream_tensor(full_key, tensor)
            # Release GPU tensors added by export handlers before hook.post_forward
            # runs, to prevent cross-layer accumulation on disk-offloaded models.
            #
            # Two categories accumulate without explicit cleanup:
            #
            # 1. CUDA *buffers* on any sub-module (weight_scale, weight_scale_2,
            #    input_scale): AlignDevicesHook.post_forward uses offload_buffers=False
            #    by default, so it never offloads buffers.  Pre-existing buffers in
            #    disk-offloaded layers live on CPU, so any CUDA buffer encountered here
            #    was registered by the export handlers and is safe to drop.
            #
            # 2. CUDA *parameters* on modules WITHOUT _hf_hook: _export_fused_experts
            #    creates fresh nn.Module objects (one per expert x projection) and adds
            #    them to the layer via add_module() *after* weight_access_and_writeback
            #    captured its materialized list.  hook.post_forward never visits these
            #    new modules, so their packed NVFP4 weight parameters (~5 GB per MoE
            #    layer) stay live on GPU.  Modules WITH _hf_hook are original model
            #    modules whose parameters hook.post_forward will meta-ify; leave those
            #    alone.
            for sub_mod in layer_module.modules():
                for buf_name in list(sub_mod._buffers):
                    buf = sub_mod._buffers[buf_name]
                    if buf is not None and buf.device.type == "cuda":
                        sub_mod._buffers[buf_name] = None
                if not hasattr(sub_mod, "_hf_hook"):
                    for param_name, param in list(sub_mod._parameters.items()):
                        if param is not None and param.device.type == "cuda":
                            sub_mod._parameters[param_name] = None
        torch.cuda.empty_cache()

    # Non-decoder modules whose weights are not directly readable (embed_tokens, norm,
    # lm_head, ...). Containers are skipped: their children get their own window.
    for name, module in model.named_modules():
        if id(module) in decoder_owned_ids:
            continue
        if not requires_weight_materialization(module, model, name_to_module):
            continue
        with enable_weight_access_and_writeback(module, model, name_to_module, writeback=False):
            for sub_name, sub_mod in module.named_modules():
                full_name = f"{name}.{sub_name}" if sub_name else name
                _dispatch_export_handler(full_name, sub_mod, ctx)
            prefix = f"{name}." if name else ""
            for key, tensor in module.state_dict().items():
                full_key = prefix + key
                if full_key in seen_keys or tensor.is_meta:
                    continue
                seen_keys.add(full_key)
                _stream_tensor(full_key, tensor)

    # GPU-resident parameters and persistent buffers (not covered by the above loops).
    # named_buffers() includes non-persistent buffers that state_dict() excludes; filter them.
    for name, tensor in itertools.chain(
        model.named_parameters(),
        ((n, b) for n, b in model.named_buffers() if _is_persistent_buffer(n)),
    ):
        if name in seen_keys or tensor is None or tensor.is_meta:
            continue
        seen_keys.add(name)
        _stream_tensor(name, tensor)

    # Tensors the model never held — e.g. MTP weights, which HF leaves orphaned because it
    # builds only num_hidden_layers decoders. They are already materialized and skip the
    # per-tensor postprocessing, matching how the batch path merges them after
    # postprocess_state_dict; only the hub-name reversal applies.
    for name, tensor in (extra_state_dict or {}).items():
        if name in seen_keys:
            continue
        seen_keys.add(name)
        writer.add(
            name_mapper(name) if name_mapper is not None else name,
            tensor.detach().contiguous().cpu(),
        )

    writer.finalize()

    save_non_weight_artifacts(model, export_dir)

    return None, quant_config
