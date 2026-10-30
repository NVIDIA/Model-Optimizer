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

Each rank decompresses only its own experts, with the ``compressed_tensors`` library, so expert-parallel
PTQ runs on the original checkpoint. Needs transformers 5.8, whose ``Concatenate`` reads only its keys.
"""

import functools
import os

import torch
from safetensors import safe_open

from modelopt.torch.utils import print_rank_0
from modelopt.torch.utils.moe import _SUB_CONFIG_ATTRS, require_transformers


def to_native_backbone_config(
    vlm_config,
    model_type: str = "deepseek_v3",
    architectures: "tuple[str, ...]" = ("DeepseekV3ForCausalLM",),
):
    """The transformers-native backbone config for a VLM config's bundled ``text_config``.

    Drops the remote ``auto_map`` and the stale ``quantization_config``, and names the native model
    type and class (hf_ptq's VLM detection reads ``architectures``). The decompression mark carries
    over. Defaults target the Kimi DeepSeek-V3 backbone.
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
    """``(compressor, scheme, param_names, format)`` for the checkpoint's compressed-tensors config.

    ``param_names`` are the keys one weight is stored under. The one config group must target the
    experts; several groups are refused. ``None`` for a config with no ``quantization_config``."""
    # With no config, compressed_tensors need not even be installed.
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
        # Only the expert converters are rewritten: which group holds the experts would be a guess.
        raise NotImplementedError(
            f"install_onfly_decompress: the checkpoint has {len(groups)} compressed-tensors config "
            f"groups ({', '.join(groups)}); on-the-fly decompression handles a single group, "
            "compressing only the experts. Decompress the checkpoint first."
        )
    [scheme] = groups.values()
    # Regex targets alone must name the experts. A class target such as Linear does not say which
    # modules it compressed: check_compressed_keys checks the names the checkpoint stores instead.
    if not any(not t.startswith("re:") or "experts" in t for t in scheme.targets):
        raise NotImplementedError(
            f"install_onfly_decompress: the checkpoint's compressed-tensors config group targets "
            f"{scheme.targets}, not the experts; on-the-fly decompression handles a checkpoint "
            "whose experts are what it compresses. Decompress the checkpoint first."
        )
    # A group's own format wins: the config's top level then says "mixed-precision".
    fmt = getattr(scheme, "format", None) or qconfig.format
    fmt = fmt.value if hasattr(fmt, "value") else str(fmt)
    if fmt not in BaseCompressor.registered_names():
        raise NotImplementedError(
            f"install_onfly_decompress: compressed-tensors registers no compressor for the format "
            f"{fmt!r}, so on-the-fly decompression cannot read it. Decompress the checkpoint first."
        )
    compressor = BaseCompressor.get_value_from_registry(fmt)
    param_names = tuple(compressor.compression_param_names(scheme))
    return compressor, scheme, param_names, fmt


def _make_op_class(compressor, scheme, param_names):
    """Build the per-expert decompress op, bound to a compressor, its scheme and the stored keys.

    It subclasses ``MergeModulelist`` only so the loader shards experts at load, handing it ``None``
    for each expert another rank owns; the merge later in the chain still does the stacking.
    """
    from transformers.core_model_loading import MergeModulelist

    anchor = "." + param_names[0]
    others = [(pn, "." + pn) for pn in param_names[1:]]

    class DecompressExperts(MergeModulelist):
        """Decompress each owned expert's stored params to bf16, under the first stored key."""

        def __init__(self):
            super().__init__(dim=0)  # dim unused: convert is overridden

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
                # A checkpoint may store a subset of the scheme's params (Kimi-K3's mxfp4 stores no
                # weight_global_scale): only those present go to the compressor.
                for pn, suf in others:
                    sib = input_dict.pop(stem + suf + marker, None)
                    if sib is not None:
                        lists[pn] = sib
                anchor_list = lists[param_names[0]]
                present = [pn for pn in param_names if pn in lists]
                if any(len(lists[pn]) != len(anchor_list) for pn in present):
                    counts = ", ".join(f"{len(lists[pn])} {pn}" for pn in present)
                    raise ValueError(
                        f"DecompressExperts: {stem} stores {counts}, not one per expert"
                    )
                # Experts other ranks own arrive as None: only this rank's are decompressed.
                out[pat] = [
                    compressor.decompress({pn: lists[pn][i] for pn in present}, scheme)[
                        "weight"
                    ].to(torch.bfloat16)
                    for i in range(len(anchor_list))
                    if anchor_list[i] is not None
                ]
            if input_dict:
                # A stored param without its weight: the merge would fuse it into the wrong slot.
                raise ValueError(
                    f"DecompressExperts: unconsumed checkpoint keys {sorted(input_dict)}"
                )
            return out

    return DecompressExperts


@functools.cache
def _decompressing_converter_class():
    """The ``WeightConverter`` subclass a rewritten converter is, reversing as the one it rewrote."""
    from transformers.core_model_loading import WeightConverter

    class DecompressingConverter(WeightConverter):
        """Reverses as the converter it rewrote: a save names the experts back as ``.weight`` keys."""

        __slots__ = ("_reversible",)

        def reverse_transform(self):
            return self._reversible.reverse_transform()

    return DecompressingConverter


def _compressedify(conv, compressor, scheme, param_names):
    """Rewrite a per-expert converter to read the stored keys, with ``DecompressExperts`` first.

    Each ``.weight`` source becomes its stored keys, grouped by key with the anchor first, which
    keeps gate before up for the merge. The rewritten converter reverses as ``conv`` does.
    """
    DecompressExperts = _make_op_class(compressor, scheme, param_names)  # noqa: N806 (a class, keeping its own name)
    by_pname = {pn: [] for pn in param_names}
    for p in conv.source_patterns:
        base = p.removesuffix("$")
        if not base.endswith(".weight") or ".experts." not in base:
            # Only the experts' weights are stored compressed: rewritten, a dense weight would match
            # no stored key and load newly initialized. A bias is expected in a converter of its own
            # (check_compressed_keys lets one through); grouped with the weight, the merge takes it.
            raise NotImplementedError(
                f"install_onfly_decompress: the per-expert converter for {conv.target_patterns} "
                f"reads {p!r} alongside the experts' weights, which is no compressed tensor; "
                "on-the-fly decompression rewrites a converter that reads the experts' weights only. "
                "Decompress the checkpoint first."
            )
        stem = base[: -len(".weight")]
        for pn in param_names:
            by_pname[pn].append(stem + "." + pn + "$")
    new_sources = [s for pn in param_names for s in by_pname[pn]]
    tgt = getattr(conv, "_original_target_patterns", conv.target_patterns)
    new = _decompressing_converter_class()(
        source_patterns=new_sources,
        target_patterns=tgt,
        operations=[DecompressExperts(), *conv.operations],
    )
    new._reversible = conv
    # What else transformers sets on a converter it maps, whichever of them this release has.
    for attr in ("scope_prefix", "base_model_prefix", "force_cpu"):
        if hasattr(conv, attr):
            setattr(new, attr, getattr(conv, attr))
    return new


# Whether the load under way is one whose converters the hooks rewrote (from_pretrained asks for the
# mapping before it reads a tensor). Module state: the loader materializes in worker threads.
_LOAD = {"decompressing": False}


def _install_keep_int_native():
    """Patch ``spawn_materialize`` so a decompressing load never casts an integer source to a float.

    Without a quantizer the loader casts each source to its parameter's dtype in the copy, or from
    transformers 5.14 in ``DtensorShardOperation.shard_tensor``: both start here. Other loads cast.
    """
    import transformers.core_model_loading as cml

    if getattr(cml.spawn_materialize, "_onfly_patched", False):
        return
    orig = cml.spawn_materialize

    # Any other argument passes through, positionally or by keyword: a release may add some.
    def spawn_materialize(thread_pool, tensor, device=None, dtype=None, *args, **kwargs):
        # A safetensors slice is judged by its stored dtype, without reading it.
        stored = tensor.get_dtype() if hasattr(tensor, "get_dtype") else None
        floats = stored.startswith(("F", "BF")) if stored else tensor.is_floating_point()
        if _LOAD["decompressing"] and dtype is not None and dtype.is_floating_point and not floats:
            dtype = None  # keep packed/shape/zero-point/g_idx integers native
        return orig(thread_pool, tensor, device, dtype, *args, **kwargs)

    setattr(spawn_materialize, "_onfly_patched", True)
    cml.spawn_materialize = spawn_materialize


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


def stored_names(hf_config) -> tuple[str, ...]:
    """The names the installed compressed-tensors stores each weight under for ``hf_config``'s scheme."""
    for c in _config_levels(hf_config):
        if (stored_as := getattr(c, _MARK, None)) is not None:
            return stored_as
    raise ValueError("The config is not marked for on-the-fly decompression: install it first.")


def install_onfly_decompress(hf_config):
    """Make ``from_pretrained`` decompress the experts of the checkpoint ``hf_config`` describes.

    Call before ``from_pretrained``. The hooks are process-wide but rewrite only the converters of a
    model built from a config they marked: one compressed checkpoint per process, installing again
    for it marks that config too, and one compressed differently is refused.
    """
    if isinstance(hf_config, dict):
        # The load finds the hooks by a mark on the config object, which a dict cannot carry.
        raise TypeError("install_onfly_decompress: pass the model's config object, not a dict")
    # The loader sets each parameter from what the conversion hands it, uncast. Every level counts:
    # a backbone config derived from the text config (to_native_backbone_config) keeps its dtype.
    for level in _config_levels(hf_config):
        dtype = getattr(level, "dtype", None)
        if dtype is not None and str(dtype).removeprefix("torch.") != "bfloat16":
            raise NotImplementedError(
                f"install_onfly_decompress: the experts decompress to bfloat16, but this config "
                f"loads in {dtype}. Load the checkpoint in bfloat16."
            )
    # Before compressed-tensors is needed: a release below the floor is what to fix first.
    require_transformers("5.8", "on-the-fly compressed-tensors decompression")
    compression = _read_compression(hf_config)
    # Imported once the release is known to have them: transformers 4 has no core_model_loading.
    import transformers.modeling_utils as mu
    from transformers.core_model_loading import WeightConverter

    installed = getattr(mu.get_model_conversion_mapping, "_onfly_compression", None)
    if installed is not None:
        if compression is None:
            # A marked config its prepare dropped quantization_config from (Kimi-K3's) may be prepared
            # again; an unmarked one would load its experts newly initialized.
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
        _LOAD["decompressing"] = is_marked(config)
        if not _LOAD["decompressing"]:
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
            # The stock converters would read .weight keys the checkpoint does not store.
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


def check_compressed_keys(hf_config, ckpt_dir) -> dict[str, str]:
    """Refuse a checkpoint the decompression would load wrong, by the tensors ``ckpt_dir`` stores.

    Only the experts decompress, and the load drops the config any other compressed weight would need:
    the experts alone must be compressed, every one of them. The load may cast their float tensors to
    bf16 before decompressing, so one bf16 cannot hold exactly (an fp32 scale) is refused too. Reads
    only the tensors' names and dtypes, and returns the ``tensor name -> file`` map it read.
    """
    # A plugin: it imports huggingface_hub, which this module needs nowhere else.
    from modelopt.torch.utils.plugins.hf_checkpoint_utils import checkpoint_tensor_files

    stored_as = stored_names(hf_config)
    tensor_files = checkpoint_tensor_files(ckpt_dir)
    compressed: set[str] = set()
    expert_weights: set[str] = set()
    stems_storing: dict[str, set[str]] = {}
    for name in tensor_files:
        stem, _, stored = name.rpartition(".")
        if stored in stored_as:
            stems_storing.setdefault(stored, set()).add(stem)
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
    # The names are what the installed compressed-tensors gives the scheme: a weight's name a newer
    # release added would match no rewritten source, and the experts would decompress without it.
    # Only weight* names are checked (every compressor's today), so an expert's bias, which loads
    # through its own converter, is not refused.
    unknown = [n for n in sorted(tensor_files) if n.rpartition(".")[0] in compressed]
    unknown = [n for n in unknown if n.rpartition(".")[2] not in stored_as]
    unknown = [n for n in unknown if n.rpartition(".")[2].startswith("weight")]
    if unknown:
        raise NotImplementedError(
            f"The checkpoint stores {len(unknown)} tensors under its compressed weights that this "
            f"compressed-tensors release does not name for the scheme {unknown[:1]}: the decompression "
            f"reads only {list(stored_as)}. Upgrade compressed-tensors, or decompress it first."
        )
    # The load pairs each expert's stored tensors by the order it collects them, not by expert id: one
    # missing a scale would pair a later expert's weight with an earlier one's scale. A name no weight
    # stores (Kimi-K3's weight_global_scale) is simply not read.
    unpaired = sorted(
        (stored, sorted(compressed - stems))
        for stored, stems in stems_storing.items()
        if stems & compressed and compressed - stems
    )
    if unpaired:
        stored, missing = unpaired[0]
        raise NotImplementedError(
            f"The checkpoint stores {stored} for some compressed weights but not for {len(missing)} "
            f"others {missing[:1]}: the decompression pairs each expert's stored tensors by position, "
            "so every compressed weight must store the same names. Decompress the checkpoint first."
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
    return tensor_files
