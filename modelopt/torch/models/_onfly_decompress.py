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

"""Decompress a compressed-tensors MoE checkpoint's experts as ``from_pretrained`` loads them.

Each rank decompresses only its own experts into the transformers 5.x fused-experts layout, so
expert-parallel PTQ runs on the original compressed checkpoint, with no decompressed copy on disk.
The math is the ``compressed_tensors`` library's, driven by the checkpoint's ``quantization_config``:
``compressor.compression_param_names(scheme)`` names the stored keys of one weight, and
``compressor.decompress(...)["weight"]`` dequantizes them, so every format the library registers
works whose stored floating-point tensors bf16 holds exactly (:func:`check_compressed_keys`). It
needs ``compressed_tensors`` importable and transformers 5.8 or newer (earlier releases'
``Concatenate`` looks up the sibling keys the decompression consumed).

:func:`install_onfly_decompress` rewrites the per-expert converters ``get_model_conversion_mapping``
returns, for the model built from the config it marks: each ``.weight`` source becomes the stored
keys, and a ``DecompressExperts`` op ahead of the merge dequantizes each expert.
:func:`to_native_backbone_config` builds a VLM's transformers-native backbone config.
"""

import os

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
    native = CONFIG_MAPPING[model_type].from_dict(d)
    stored_as = getattr(vlm_config, _MARK, None) or getattr(text_config, _MARK, None)
    if stored_as is not None:
        # A backbone loaded from this config on its own decompresses as the VLM would.
        _mark(native, stored_as)
    return native


def _config_levels(hf_config) -> list:
    """``hf_config`` and the nested language-model configs the expert-parallel load reads."""
    from modelopt.torch.utils.moe import _SUB_CONFIG_ATTRS

    nested = (getattr(hf_config, attr, None) for attr in _SUB_CONFIG_ATTRS)
    return [c for c in (hf_config, *nested) if c is not None]


def _find_quantization_config(hf_config):
    """Locate the compressed-tensors ``quantization_config`` (top-level, else on the nested
    language-model config) and return it as a plain dict, or ``None`` for a config without one."""
    qc = next(
        (
            found
            for c in _config_levels(hf_config)
            if (found := getattr(c, "quantization_config", None)) is not None
        ),
        None,
    )
    if qc is None:
        return None
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
    ``param_names`` is the exact set of stored sibling keys per weight. The checkpoint's one config
    group is the experts' scheme; a checkpoint with several, or whose one group does not target the
    experts, is refused. ``None`` for a config with no ``quantization_config``: nothing to decompress."""
    # Look the config up first: with none, compressed_tensors need not even be installed. A config
    # that fails to parse raises pydantic's ValidationError, a ValueError of its own.
    quantization_config = _find_quantization_config(hf_config)
    if quantization_config is None:
        return None
    try:
        from compressed_tensors.compressors.base import BaseCompressor
        from compressed_tensors.quantization import QuantizationConfig
    except ImportError as e:
        # No pip extra of ModelOpt installs it: only a compressed checkpoint needs it.
        raise ImportError(
            "On-the-fly decompression of a compressed-tensors checkpoint needs the "
            "compressed-tensors package. Install it with: pip install compressed-tensors"
        ) from e

    qconfig = QuantizationConfig.model_validate(quantization_config)
    groups = qconfig.config_groups
    if not groups:
        raise ValueError(
            "install_onfly_decompress: the checkpoint's compressed-tensors quantization_config has "
            "no config groups: nothing to decompress."
        )
    if len(groups) > 1:
        # Only the expert converters are rewritten, so another group's weights would load newly
        # initialized, and which group holds the experts would be a guess.
        raise NotImplementedError(
            f"install_onfly_decompress: the checkpoint has {len(groups)} compressed-tensors config "
            f"groups ({', '.join(groups)}); on-the-fly decompression handles a single group, "
            "compressing only the experts. Decompress the checkpoint first."
        )
    [scheme] = groups.values()
    # A group of regex targets alone must name the experts, or it leaves them stored as plain weights
    # the rewritten converters would not find. A class target such as Linear does not say which
    # modules it compressed (the experts, and any Linear the ignore list leaves): the expert-parallel
    # load checks the names the checkpoint stores instead (check_compressed_keys).
    if not any(not t.startswith("re:") or "experts" in t for t in scheme.targets):
        raise NotImplementedError(
            f"install_onfly_decompress: the checkpoint's compressed-tensors config group targets "
            f"{scheme.targets}, not the experts; on-the-fly decompression handles a checkpoint "
            "whose experts are what it compresses. Decompress the checkpoint first."
        )
    fmt = qconfig.format
    fmt = fmt.value if hasattr(fmt, "value") else str(fmt)
    compressor = BaseCompressor.get_value_from_registry(fmt)
    param_names = tuple(compressor.compression_param_names(scheme))
    return compressor, scheme, param_names, fmt


def _make_op_class(compressor, scheme, param_names):
    """Build the per-expert decompress op, bound to a compressor, its scheme and the stored keys.

    The first stored key is the anchor the decompressed experts are emitted under, in place of the
    ``.weight`` the downstream merge expects. The op subclasses ``MergeModulelist`` only because the
    loader shards experts at load (``tensor_idx``, so non-local experts arrive as ``None``) only when
    a converter's first op is one; the original merge later in the chain still does the stacking.
    """
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
                if any(len(lists[pn]) != len(anchor_list) for pn in present):
                    # Paired by position: an expert short of one would pair every later expert's
                    # weight with the next one's scale.
                    counts = ", ".join(f"{len(lists[pn])} {pn}" for pn in present)
                    raise ValueError(
                        f"DecompressExperts: {stem} stores {counts}, not one per expert"
                    )
                # Non-local experts were sharded away at load (GroupedGemmParallel -> None); decompress
                # only the local experts this rank owns. Keeps peak memory at the per-rank shard.
                out[pat] = [
                    compressor.decompress({pn: lists[pn][i] for pn in present}, scheme)[
                        "weight"
                    ].to(torch.bfloat16)
                    for i in range(len(anchor_list))
                    if anchor_list[i] is not None
                ]
            if input_dict:
                # A stored param without its weight, or a source that is no weight: passed on, the
                # merge would fuse it into the wrong slot or fail far from here.
                raise ValueError(
                    f"DecompressExperts: unconsumed checkpoint keys {sorted(input_dict)}"
                )
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
            by_pname[param_names[0]].append(p)  # DecompressExperts refuses it (none expected)
    new_sources = [s for pn in param_names for s in by_pname[pn]]
    tgt = getattr(conv, "_original_target_patterns", conv.target_patterns)
    new = WeightConverter(
        source_patterns=new_sources,
        target_patterns=tgt,
        operations=[DecompressExperts(), *conv.operations],
    )
    # What else transformers sets on a converter when it maps a model -- the sub-model scope it
    # matches keys under, whether it loads on the CPU -- whichever of them this release has.
    for attr in ("scope_prefix", "base_model_prefix", "force_cpu"):
        if hasattr(conv, attr):
            setattr(new, attr, getattr(conv, attr))
    return new


def _install_keep_int_native():
    """Patch _materialize_copy so INTEGER source tensors are never cast to a FLOATING dtype during load.

    Without an hf_quantizer the loader casts every source to the target param dtype (bf16); that destroys the integer
    bit pattern of packed weights (`weight_packed`), shapes (`weight_shape`), zero-points and g_idx. Floats (the scale,
    global scale, dense weights) still cast normally, so the expert-parallel load refuses a checkpoint whose compressed
    experts store a float bf16 cannot hold exactly (:func:`check_compressed_keys`).

    Process-wide, unlike the converter rewrite: any later load keeps an integer tensor native where it would have cast
    it to a floating-point dtype. Only a compressed checkpoint loaded without its quantizer asks for that cast: with a
    quantizer the loader keeps integer tensors as stored, and an unquantized checkpoint stores none for a float param.
    """
    import transformers.core_model_loading as cml

    if getattr(cml._materialize_copy, "_onfly_patched", False):
        return
    orig = cml._materialize_copy

    # Any other argument passes through, positionally as the loader passes the ones it has, or by
    # keyword: the patch sits on every later load's copy path.
    def _materialize_copy(tensor, device=None, dtype=None, *args, **kwargs):
        t = tensor[...]
        if dtype is not None and dtype.is_floating_point and not t.is_floating_point():
            dtype = None  # keep packed/shape/zero-point/g_idx integers native
        return orig(t, device, dtype, *args, **kwargs)

    setattr(_materialize_copy, "_onfly_patched", True)
    cml._materialize_copy = _materialize_copy


# Marks a config whose model's converters the installed hooks rewrite (holds the names each
# compressed weight is stored under).
_MARK = "_modelopt_onfly_decompress"


def _mark(hf_config, stored_as):
    """Mark ``hf_config``'s load for the installed hooks; ``from_pretrained``'s copy of it keeps the mark."""
    setattr(hf_config, _MARK, stored_as)


def is_marked(hf_config) -> bool:
    """Whether :func:`install_onfly_decompress` marked ``hf_config``, or its nested language-model
    config, for its hooks."""
    return any(getattr(c, _MARK, None) is not None for c in _config_levels(hf_config))


def install_onfly_decompress(hf_config):
    """Monkeypatch get_model_conversion_mapping so the fused-MoE per-expert converters decompress the
    checkpoint's compressed-tensors weights on the fly, in whatever ``format`` the checkpoint declares.
    Call before from_pretrained of a compressed checkpoint. ``hf_config`` is the model config carrying
    the compressed-tensors ``quantization_config`` (top-level or under ``text_config``); the format,
    bit-width, group size, strategy and stored param names are all read from it.

    The hooks are process-wide, so a process decompresses one compressed checkpoint: installing again
    for it marks that config too, and installing for a checkpoint compressed differently raises. Only a
    model built from a marked config, or from one whose ``text_config`` is marked, has its converters
    rewritten; any other model ``from_pretrained`` loads in the process, an uncompressed MoE say,
    keeps its stock converters. :func:`to_native_backbone_config` carries the mark to the backbone
    config it derives, and the expert-parallel load checks the checkpoint against it
    (:func:`check_compressed_keys`)."""
    from modelopt.torch.utils import print_rank_0
    from modelopt.torch.utils.moe import require_transformers

    if isinstance(hf_config, dict):
        # The load finds the hooks by a mark on the config object, which a dict cannot carry.
        raise TypeError("install_onfly_decompress: pass the model's config object, not a dict")
    compression = _read_compression(hf_config)
    require_transformers("5.8", "on-the-fly compressed-tensors decompression")
    # Imported once the release is known to have them: transformers 4 has no core_model_loading.
    import transformers.modeling_utils as mu
    from transformers.core_model_loading import WeightConverter

    installed = getattr(mu.get_model_conversion_mapping, "_onfly_compression", None)
    if installed is not None:
        if compression is None:
            # A config its prepare dropped quantization_config from after installing (Kimi-K3's),
            # already marked, may be prepared again: nothing new to decompress. An unmarked one would
            # load its experts through the stock converters, newly initialized.
            if not is_marked(hf_config):
                raise ValueError(
                    "install_onfly_decompress: config has no compressed-tensors "
                    "quantization_config and is not marked: nothing would decompress it"
                )
            return
        if compression != installed:
            raise RuntimeError(
                f"install_onfly_decompress: this process already decompresses a '{installed[3]}' "
                f"checkpoint (stored params {installed[2]}); one compressed differently ('{compression[3]}', "
                f"stored params {compression[2]}) needs a process of its own"
            )
        # The same checkpoint through a config of its own, which its load must find marked too.
        _mark(hf_config, installed[2])
        return
    if compression is None:
        # Nothing to decompress: refused before anything is patched.
        raise ValueError(
            "install_onfly_decompress: config has no compressed-tensors quantization_config"
        )
    compressor, scheme, param_names, fmt = compression
    _install_keep_int_native()
    orig = mu.get_model_conversion_mapping

    def patched(*args, **kwargs):
        convs = orig(*args, **kwargs)
        config = getattr(args[0] if args else kwargs["model"], "config", None)
        if not is_marked(config):
            # Not the compressed checkpoint: rewritten, an uncompressed MoE's experts would match no
            # stored key and load newly initialized without an error.
            return convs
        out, rewritten = [], 0
        for c in convs:
            # only the per-expert MoE converters (have `.weight` sources under `.experts.`, which
            # leaves a shared expert's `shared_experts.` out, as check_compressed_keys does)
            if isinstance(c, WeightConverter) and any(
                ".experts." in p and p.endswith((".weight", ".weight$")) for p in c.source_patterns
            ):
                out.append(_compressedify(c, compressor, scheme, param_names))
                rewritten += 1
            else:
                out.append(c)
        if not rewritten:
            # The stock converters would read the experts' plain .weight keys, which the checkpoint
            # does not store.
            raise RuntimeError(
                "install_onfly_decompress: the model's conversion mapping has no per-expert "
                "converter to rewrite, so its compressed experts would load newly initialized."
            )
        # from_pretrained loads from a copy of the marked config; the mark must not reach the config
        # a later save writes, at any level. Cleared once the experts' converters are rewritten.
        for c in _config_levels(config):
            if hasattr(c, _MARK):
                delattr(c, _MARK)
        return out

    setattr(patched, "_onfly_compression", compression)
    mu.get_model_conversion_mapping = patched
    _mark(hf_config, param_names)
    print_rank_0(
        f"[onfly_decompress] installed: compressed_tensors '{fmt}' fused-MoE experts decompress "
        f"(stored params={param_names}) -> bf16 during from_pretrained"
    )


def check_compressed_keys(hf_config, ckpt_dir) -> None:
    """Refuse a checkpoint the decompression would load wrong, by the tensors ``ckpt_dir`` stores.

    The hooks rewrite only the experts' converters, and the expert-parallel load drops the
    quantization_config that would load any other compressed weight: a weight compressed outside the
    experts would load newly initialized, and so would an expert stored uncompressed, which the
    rewritten converters do not read. A weight counts as compressed when it stores a name
    ``hf_config``'s mark holds other than ``weight``, which an uncompressed weight is stored under
    too.

    The load may also cast a compressed weight's floating-point tensors to bf16, the dtype its
    experts decompress to, before the decompression reads them: one that bf16 cannot hold exactly,
    such as an fp32 global scale or an fp16 scale, would shift every weight it scales. Only the
    tensors' names and dtypes are read.
    """
    from safetensors import safe_open

    from modelopt.torch.utils.plugins.hf_checkpoint_utils import checkpoint_tensor_files

    stored_as = next(
        getattr(c, _MARK) for c in _config_levels(hf_config) if getattr(c, _MARK, None) is not None
    )
    tensor_files = checkpoint_tensor_files(ckpt_dir)
    compressed: set[str] = set()
    expert_weights: set[str] = set()
    for name in tensor_files:
        stem, _, stored = name.rpartition(".")
        if stored in stored_as and stored != "weight":
            compressed.add(stem)
        elif stored == "weight" and ".experts." in name:
            expert_weights.add(stem)
    outside = sorted(stem for stem in compressed if ".experts." not in stem)
    plain = sorted(expert_weights - compressed)
    if outside or plain or not compressed:
        raise NotImplementedError(
            f"The checkpoint compresses {len(compressed)} weights, {len(outside)} of them outside the "
            f"experts {outside[:1]}, and stores {len(plain)} expert weights uncompressed {plain[:1]}: "
            "on-the-fly decompression loads the experts alone, every one compressed, so the rest "
            "would load newly initialized. Decompress the checkpoint first."
        )
    by_file: dict[str, list[str]] = {}
    for name, file in tensor_files.items():
        stem, _, stored = name.rpartition(".")
        if stem in compressed and stored in stored_as:
            by_file.setdefault(file, []).append(name)
    inexact: set[str] = set()
    for file, names in by_file.items():
        with safe_open(os.path.join(ckpt_dir, file), framework="pt") as f:
            for name in names:
                dtype = f.get_slice(name).get_dtype()
                if dtype in ("F16", "F32", "F64"):
                    inexact.add(f"{name.rpartition('.')[2]} ({dtype})")
    if inexact:
        raise NotImplementedError(
            f"The checkpoint's compressed experts store {', '.join(sorted(inexact))}, which bf16 "
            "cannot hold exactly: the load may cast them to bf16, the dtype the experts decompress "
            "to, before decompressing, shifting every weight they scale. Decompress the checkpoint "
            "first."
        )
