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

"""On-the-fly decompression of compressed-tensors quantized MoE experts into the transformers-5.x
fused-experts layout, DURING from_pretrained — so a2a PTQ can run on the ORIGINAL compressed
checkpoint with no decompressed intermediate on disk, EP-sharded (each rank decompresses only its
own experts).

FORMAT-AGNOSTIC. The dequant math is delegated to the ``compressed_tensors`` library and driven
entirely by the checkpoint's own ``quantization_config`` (the ``format`` string + the per-group
``QuantizationArgs``). Any format the library registers works with ZERO per-format code here:
``pack-quantized`` (INT, symmetric or asymmetric, any bit-width / group_size / strategy),
``nvfp4-pack-quantized`` (FP4 + FP8 block scale + FP32 global), ``naive-quantized`` /
``float-quantized`` / ``int-quantized`` (unpacked FP8/INT), etc. The two universal hooks the library
exposes per format are all we need:
  - ``compressor.compression_param_names(scheme)`` -> the exact set of stored keys for one weight
    (e.g. ``("weight_packed","weight_scale","weight_shape")`` for pack, ``("weight","weight_scale")``
    for float), i.e. which per-expert sibling tensors to pull.
  - ``compressor.decompress({<those keys>: tensor}, scheme)["weight"]`` -> the dequantized weight.
``compressed_tensors`` must be importable at load time, and transformers must be 5.8 or newer: earlier
releases' ``Concatenate`` looks up every source pattern, including the sibling keys the decompression
already consumed.

The model-specific knobs remain parameters; defaults target moonshotai/Kimi-K2.7-Code (DeepSeek-V3
backbone):
  - install_onfly_decompress(hf_config)         — auto-detects format + quant args from the config
  - to_native_backbone_config(vlm_config, ...)  — build the native backbone config

Mechanism (mirrors transformers' FP8 quantizer ``update_weight_conversions``): the fused-MoE model
already ships a ``WeightConverter`` that fuses per-expert ``gate/up/down_proj.weight`` -> 3-D
``gate_up_proj``/``down_proj`` via ``MergeModulelist``+``Concatenate``. We rewrite those converters so
each ``.weight`` source becomes the checkpoint's actual stored keys for that weight
(``compression_param_names``), and PREPEND a ``DecompressExperts`` op that pairs those per expert and
dequantizes -> bf16 BEFORE the merge/concat collapse the per-expert structure.
``install_onfly_decompress()`` monkeypatches ``get_model_conversion_mapping``.
"""

import torch


def to_native_backbone_config(
    vlm_config,
    model_type: str = "deepseek_v3",
    architectures: "tuple[str, ...]" = ("DeepseekV3ForCausalLM",),
):
    """Derive the transformers-native fused backbone config from a VLM config's ``text_config``, so
    ``from_pretrained`` builds the native model directly from the ORIGINAL checkpoint's compressed
    shards — NO preprocessed native-config "view" dir.

    A VLM's ``text_config`` is typically the bundled remote-code config (its ``auto_map`` points at a
    ModuleList modeling file, and it carries the stale compressed-tensors ``quantization_config``).
    Dropping ``auto_map`` + ``quantization_config`` and setting ``model_type`` to the native type makes
    it resolve to the stock transformers native config (which, for the a2a path, supplies
    ``base_model_ep_plan``). ``architectures`` must be set to the native class (NOT left None — hf_ptq's
    VL-detection iterates ``config.architectures``).

    Defaults target the Kimi DeepSeek-V3 backbone; pass ``model_type`` / ``architectures`` for another.
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    text_config = getattr(vlm_config, "text_config", None)
    if text_config is None:
        raise ValueError("to_native_backbone_config: config has no .text_config (not a VLM?)")
    if model_type not in CONFIG_MAPPING:
        raise ValueError(f"to_native_backbone_config: unknown native model_type {model_type!r}")
    d = text_config.to_dict() if hasattr(text_config, "to_dict") else dict(text_config)
    for key in ("auto_map", "quantization_config", "_name_or_path"):
        d.pop(key, None)
    d["model_type"] = model_type
    d["architectures"] = list(architectures)
    return CONFIG_MAPPING[model_type].from_dict(d)


def _find_quantization_config(hf_config):
    """Locate the compressed-tensors ``quantization_config`` (top-level or under ``text_config``) and
    return it as a plain dict."""
    qc = getattr(hf_config, "quantization_config", None)
    if qc is None:
        tc = getattr(hf_config, "text_config", None)
        qc = getattr(tc, "quantization_config", None) if tc is not None else None
    if qc is None and isinstance(hf_config, dict):
        qc = hf_config.get("quantization_config") or (hf_config.get("text_config") or {}).get(
            "quantization_config"
        )
    if qc is None:
        raise ValueError(
            "install_onfly_decompress: config has no compressed-tensors quantization_config"
        )
    if hasattr(qc, "to_dict"):
        qc = qc.to_dict()
    elif not isinstance(qc, dict):
        qc = dict(qc)
    return qc


def _read_compression(hf_config):
    """Parse the checkpoint's compressed-tensors ``quantization_config`` and return
    ``(compressor, scheme, param_names, format_str)`` via the ``compressed_tensors`` registry.

    ``compressor`` is the registered compressor for ``format`` (its ``decompress`` classmethod is the
    dequant); ``scheme`` is the ``QuantizationScheme`` (supplies ``scheme.weights`` to ``decompress``);
    ``param_names`` is the exact set of stored sibling keys per weight. The experts' group is chosen as
    the sole config group, else the one whose ``targets`` reference "expert", else the first."""
    # Look the config up first: with none, there is nothing to decompress (ValueError), and
    # compressed_tensors need not even be installed.
    quantization_config = _find_quantization_config(hf_config)
    from compressed_tensors.compressors.base import BaseCompressor
    from compressed_tensors.quantization import QuantizationConfig

    qconfig = QuantizationConfig.model_validate(quantization_config)
    groups = qconfig.config_groups
    if len(groups) == 1:
        scheme = next(iter(groups.values()))
    else:
        scheme = next(
            (
                g
                for g in groups.values()
                if any("expert" in str(t).lower() for t in (getattr(g, "targets", None) or []))
            ),
            next(iter(groups.values())),
        )
    fmt = qconfig.format
    fmt = fmt.value if hasattr(fmt, "value") else str(fmt)
    compressor = BaseCompressor.get_value_from_registry(fmt)
    param_names = tuple(compressor.compression_param_names(scheme))
    return compressor, scheme, param_names, fmt


def _make_op_class(compressor, scheme, param_names):
    """Build the per-expert decompress op lazily (so this module imports without transformers present),
    bound to a compressed_tensors compressor + scheme + the checkpoint's stored param names. The FIRST
    param name is the "anchor" (the source key the decompressed list is emitted under, replacing the
    original ``.weight`` so the downstream MergeModulelist/Concatenate fuse exactly as uncompressed).

    NOTE: ``DecompressExperts`` subclasses ``MergeModulelist`` so that it shards experts AT LOAD. The
    transformers loader (``convert_and_load_state_dict_in_model``) only computes the per-expert
    ``shard_index`` (``tensor_idx``) when ``isinstance(converter.operations[0], MergeModulelist)``; that
    ``tensor_idx`` is what makes ``GroupedGemmParallel.shard_tensor`` keep ONLY this rank's local experts
    (returning ``None`` for the rest). If our op were a plain ``ConversionOps`` prepended before the
    original ``MergeModulelist``, ``operations[0]`` would not match, ``tensor_idx`` would be ``None``, and
    every rank would materialize + decompress ALL experts (garbled, and a ~device-max transient -> OOM at
    scale, e.g. Kimi-K2.7-Code EP32). Subclassing keeps ``operations[0]`` a ``MergeModulelist`` instance so
    per-expert sharding fires; we still override ``convert`` to decompress (the ``dim``/merge behavior of
    the base class is unused -- the ORIGINAL ``MergeModulelist`` later in the chain does the stacking)."""
    from transformers.core_model_loading import MergeModulelist

    anchor = "." + param_names[0]
    others = [(pn, "." + pn) for pn in param_names[1:]]

    class DecompressExperts(MergeModulelist):
        """First op in a fused-experts WeightConverter chain.

        For each anchor source (e.g. ``*.weight_packed``), pair it with its sibling stored params (``*.weight_scale``,
        ``*.weight_shape``, ``*.weight_zero_point``, ``*.weight_global_scale``, ... — whatever the format stores),
        decompress each expert via ``compressed_tensors`` -> bf16, and emit the decompressed list UNDER the anchor key
        (dropping the siblings). The downstream MergeModulelist/Concatenate then fuse as for an uncompressed checkpoint.

        Subclasses ``MergeModulelist`` purely so the loader recognizes ``operations[0]`` and shards experts
        at load (see ``_make_op_class`` docstring); non-local experts arrive as ``None`` (from
        ``GroupedGemmParallel.shard_tensor``) and are skipped, so only this rank's local experts are
        decompressed.
        """

        def __init__(self):
            super().__init__(
                dim=0
            )  # dim unused (convert overridden); needed so isinstance -> shard_index

        @torch.no_grad()
        def convert(self, input_dict, *args, **kwargs):
            out = {}
            for pat in list(input_dict.keys()):
                marker = "$" if pat.endswith("$") else ""
                base = pat[:-1] if marker else pat
                if not base.endswith(anchor):
                    continue
                stem = base[: -len(anchor)]
                lists = {param_names[0]: input_dict.pop(pat)}
                # Pop each sibling if PRESENT; a checkpoint may store only a SUBSET of the scheme's
                # possible params (e.g. mxfp4-pack-quantized declares weight_global_scale via
                # compression_param_names, but the public moonshotai/Kimi-K3 checkpoint stores only
                # weight_packed + weight_scale). Missing optional siblings simply aren't passed to the
                # compressor (whose decompress uses whatever the scheme requires).
                for pn, suf in others:
                    sib = input_dict.pop(stem + suf + marker, None)
                    if sib is not None:
                        lists[pn] = sib
                anchor_list = lists[param_names[0]]
                present = [pn for pn in param_names if pn in lists]
                # Non-local experts were sharded away at load (GroupedGemmParallel -> None); decompress
                # only the local experts this rank owns. Keeps peak memory at the per-rank shard.
                out[pat] = [
                    compressor.decompress({pn: lists[pn][i] for pn in present}, scheme)[
                        "weight"
                    ].to(torch.bfloat16)
                    for i in range(len(anchor_list))
                    if anchor_list[i] is not None
                ]
            out.update(input_dict)  # nothing else expected, but pass through defensively
            return out

    return DecompressExperts


def _compressedify(conv, compressor, scheme, param_names):
    """Rewrite a per-expert WeightConverter (.weight sources) into a compressed-aware one.

    each ``.weight`` -> the checkpoint's stored keys ``[.<pn>$ for pn in param_names]`` (anchor first, preserving
    gate/up order so Concatenate fuses gate-then-up), and prepend DecompressExperts.
    """
    from transformers.core_model_loading import WeightConverter

    DecompressExperts = _make_op_class(compressor, scheme, param_names)  # noqa: N806 (a class, keeping its own name)
    by_pname = {pn: [] for pn in param_names}
    for p in conv.source_patterns:
        base = p.removesuffix("$")
        if base.endswith(".weight"):
            stem = base[: -len(".weight")]
            for pn in param_names:
                by_pname[pn].append(stem + "." + pn + "$")
        else:
            by_pname[param_names[0]].append(p)  # leave non-weight sources as-is (none expected)
    new_sources = [s for pn in param_names for s in by_pname[pn]]
    tgt = getattr(conv, "_original_target_patterns", conv.target_patterns)
    return WeightConverter(
        source_patterns=new_sources,
        target_patterns=tgt,
        operations=[DecompressExperts(), *conv.operations],
    )


# id(converter) -> {source_pattern: [expert indices in append order]}; used to re-sort experts numerically.
_EXPERT_ORDER: dict = {}


def _install_numeric_expert_ordering():
    """Patch WeightConverter.add_tensor / materialize_tensors so per-expert tensors are merged in NUMERIC index order.

    The loader appends in checkpoint-iteration order, which is LEXICOGRAPHIC for many indices (0,1,10,100,...,2,...) —
    that scrambles fused-expert slots. A no-op for numerically-ordered checkpoints.
    """
    import re as _re

    from transformers.core_model_loading import WeightConverter

    if getattr(WeightConverter.add_tensor, "_onfly_patched", False):
        return
    orig_add = WeightConverter.add_tensor
    orig_mat = WeightConverter.materialize_tensors

    def add_tensor(self, target_key, source_key, source_pattern, future):
        orig_add(self, target_key, source_key, source_pattern, future)
        m = _re.search(r"experts\.(\d+)\.", source_key)
        if m is not None:
            _EXPERT_ORDER.setdefault(id(self), {}).setdefault(source_pattern, []).append(
                int(m.group(1))
            )

    def materialize_tensors(self):
        order = _EXPERT_ORDER.pop(id(self), None)
        if order:
            for pat, idxs in order.items():
                lst = self.collected_tensors.get(pat)
                if lst is not None and len(lst) == len(idxs):
                    self.collected_tensors[pat] = [
                        f for _, f in sorted(zip(idxs, lst), key=lambda x: x[0])
                    ]
        return orig_mat(self)

    setattr(add_tensor, "_onfly_patched", True)
    WeightConverter.add_tensor = add_tensor
    WeightConverter.materialize_tensors = materialize_tensors


def _install_keep_int_native():
    """Patch _materialize_copy so INTEGER source tensors are never cast to a FLOATING dtype during load.

    Without an hf_quantizer the loader casts every source to the target param dtype (bf16); that destroys the integer
    bit pattern of packed weights (`weight_packed`), shapes (`weight_shape`), zero-points and g_idx. Floats (the scale,
    global scale, dense weights) still cast normally.
    """
    import torch
    import transformers.core_model_loading as cml

    if getattr(cml._materialize_copy, "_onfly_patched", False):
        return

    def _materialize_copy(tensor, device=None, dtype=None):
        t = tensor[...]
        if (
            dtype is not None
            and not t.is_floating_point()
            and torch.empty((), dtype=dtype).is_floating_point()
        ):
            dtype = None  # keep packed/shape/zero-point/g_idx integers native
        if dtype is not None or device is not None:
            t = t.to(device=device, dtype=dtype)
        return t

    setattr(_materialize_copy, "_onfly_patched", True)
    cml._materialize_copy = _materialize_copy


def install_onfly_decompress(hf_config):
    """Monkeypatch get_model_conversion_mapping so the fused-MoE per-expert converters decompress the
    checkpoint's compressed-tensors weights on the fly, in whatever ``format`` the checkpoint declares.
    Idempotent. Call once before from_pretrained of a compressed checkpoint. ``hf_config`` is the model
    config carrying the compressed-tensors ``quantization_config`` (top-level or under ``text_config``);
    the format, bit-width, group size, strategy and stored param names are all read from it."""
    import transformers.modeling_utils as mu
    from transformers.core_model_loading import WeightConverter

    from modelopt.torch.utils.moe import require_transformers

    if getattr(mu.get_model_conversion_mapping, "_onfly_patched", False):
        return
    # Raises ValueError for a config with nothing to decompress, before anything is patched.
    compressor, scheme, param_names, fmt = _read_compression(hf_config)
    require_transformers("5.8", "on-the-fly compressed-tensors decompression")
    _install_numeric_expert_ordering()
    _install_keep_int_native()
    orig = mu.get_model_conversion_mapping

    def patched(model, key_mapping=None, hf_quantizer=None, *a, **k):
        convs = orig(model, key_mapping=key_mapping, hf_quantizer=hf_quantizer, *a, **k)
        out = []
        for c in convs:
            # only the per-expert MoE converters (have `.weight` sources under `experts`)
            if isinstance(c, WeightConverter) and any(
                "experts" in p and p.endswith((".weight", ".weight$")) for p in c.source_patterns
            ):
                out.append(_compressedify(c, compressor, scheme, param_names))
            else:
                out.append(c)
        return out

    setattr(patched, "_onfly_patched", True)
    mu.get_model_conversion_mapping = patched
    print(
        f"[onfly_decompress] installed: compressed_tensors '{fmt}' fused-MoE experts decompress "
        f"(stored params={param_names}) -> bf16 during from_pretrained",
        flush=True,
    )
