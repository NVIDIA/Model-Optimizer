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

"""On-the-fly decompression of compressed-tensors MoE experts while ``from_pretrained`` loads.

The tests drive transformers' own weight-loading objects the way its loader does --
``rename_source_key`` routes a checkpoint key, ``add_tensor`` collects it, ``convert`` fuses it --
with a stand-in compressor, so they need neither a checkpoint nor ``compressed_tensors``. The two
tests that use the real ``compressed_tensors`` registry skip when it cannot be imported.
"""

import copy
import sys
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from packaging.version import Version

if Version(transformers.__version__) < Version("5.8"):
    pytest.skip("the on-the-fly decompression needs transformers >= 5.8", allow_module_level=True)

import transformers.core_model_loading as cml
import transformers.modeling_utils as mu
from transformers import DeepseekV3Config, PretrainedConfig
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import Chunk, MergeModulelist, WeightConverter, WeightRenaming

from modelopt.torch.models import _onfly_decompress as onfly

PACKED = ("weight_packed", "weight_scale", "weight_shape")  # what pack-quantized stores per weight


@pytest.fixture
def restore_loader_hooks(monkeypatch):
    """Undo the process-wide transformers hooks the decompressor installs."""
    for owner, name in (
        (WeightConverter, "add_tensor"),
        (WeightConverter, "materialize_tensors"),
        (cml, "_materialize_copy"),
        (mu, "get_model_conversion_mapping"),
    ):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    monkeypatch.setattr(onfly, "_EXPERT_ORDER", {})


class _ScaledCompressor:
    """Stands in for a compressed_tensors compressor: ``weight = weight_packed * weight_scale``."""

    @classmethod
    def decompress(cls, state_dict, scheme):
        return {"weight": state_dict["weight_packed"].float() * state_dict["weight_scale"]}


def _stock_expert_converters():
    """transformers' own per-expert -> fused converters for DeepSeek-V3: ``(gate_up, down)``."""
    by_target = {
        conv.target_patterns[0]: conv
        for conv in get_checkpoint_conversion_mapping("deepseek_v3")
        if isinstance(conv, WeightConverter)
    }
    return by_target["mlp.experts.gate_up_proj"], by_target["mlp.experts.down_proj"]


def _load(converters, checkpoint):
    """Route, collect and fuse ``checkpoint`` the way transformers' loader does.

    The first converter whose sources match a key claims it; every fused target gets its own deep
    copy of that converter, fed in checkpoint order with completed futures (async loading).
    """
    per_target = {}
    for key, tensor in checkpoint.items():
        for conv in converters:
            target, pattern = conv.rename_source_key(key)
            if pattern is not None:
                break
        else:
            raise AssertionError(f"no converter claims {key}")
        future = Future()
        future.set_result(tensor)
        per_target.setdefault(target, copy.deepcopy(conv)).add_tensor(target, key, pattern, future)
    fused = {}
    for target, conv in per_target.items():
        fused.update(conv.convert(target))
    return fused


# ------------------------------------------------------------- rewriting the per-expert converters


def test_weight_sources_become_the_stored_keys_anchor_first_and_gate_before_up():
    """Each ``.weight`` source becomes the checkpoint's stored keys, grouped anchor first with gate
    ahead of up in every group (``Concatenate`` stacks in source order, so rows stay [gate; up]).
    Targets are kept, and the decompress op runs first, as a ``MergeModulelist`` so that loaders
    which only shard experts when the first op is one still do."""
    gate_up, _ = _stock_expert_converters()
    rewritten = onfly._compressedify(gate_up, _ScaledCompressor, None, PACKED)

    assert rewritten.source_patterns == [
        "mlp.experts.*.gate_proj.weight_packed$",
        "mlp.experts.*.up_proj.weight_packed$",
        "mlp.experts.*.gate_proj.weight_scale$",
        "mlp.experts.*.up_proj.weight_scale$",
        "mlp.experts.*.gate_proj.weight_shape$",
        "mlp.experts.*.up_proj.weight_shape$",
    ]
    assert rewritten.target_patterns == ["mlp.experts.gate_up_proj"]
    decompress, *stock_ops = rewritten.operations
    assert isinstance(decompress, MergeModulelist)
    assert [repr(op) for op in stock_ops] == [repr(op) for op in gate_up.operations]


def test_stored_keys_are_anchored_so_a_weight_never_claims_its_scale():
    """Float formats store ``weight`` next to ``weight_scale``, and transformers regex-*searches*
    source patterns: an unanchored ``gate_proj.weight`` would also claim ``gate_proj.weight_scale``."""
    gate_up, _ = _stock_expert_converters()
    rewritten = onfly._compressedify(gate_up, _ScaledCompressor, None, ("weight", "weight_scale"))
    key = "model.layers.0.mlp.experts.7."
    fused = "model.layers.0.mlp.experts.gate_up_proj"

    assert rewritten.rename_source_key(key + "gate_proj.weight_scale") == (
        fused,
        "mlp.experts.*.gate_proj.weight_scale$",
    )
    assert rewritten.rename_source_key(key + "gate_proj.weight") == (
        fused,
        "mlp.experts.*.gate_proj.weight$",
    )
    assert rewritten.rename_source_key(key + "up_proj.weight_scale")[1] == (
        "mlp.experts.*.up_proj.weight_scale$"
    )
    assert rewritten.rename_source_key(key + "down_proj.weight")[1] is None


def test_non_weight_sources_are_kept_verbatim():
    """Only ``.weight`` sources name a compressed tensor; any other source passes through as is."""
    conv = WeightConverter(
        source_patterns=["mlp.experts.*.gate_proj.weight", "mlp.experts.*.gate_proj.bias"],
        target_patterns="mlp.experts.gate_proj",
        operations=[MergeModulelist(dim=0)],
    )
    rewritten = onfly._compressedify(conv, _ScaledCompressor, None, PACKED)

    assert "mlp.experts.*.gate_proj.bias" in rewritten.source_patterns
    assert sorted(p for p in rewritten.source_patterns if "weight" in p) == sorted(
        f"mlp.experts.*.gate_proj.{name}$" for name in PACKED
    )


# ----------------------------------------------------------------- decompressing during the load


def test_packed_experts_decompress_then_fuse_like_an_uncompressed_checkpoint():
    """Through transformers' loader objects, per-expert packed weights and scales come out as the
    fused bf16 ``gate_up_proj`` ([gate; up] per expert) and the stacked ``down_proj``."""
    n_experts, inter, hidden = 3, 2, 4
    shapes = {"gate": (inter, hidden), "up": (inter, hidden), "down": (hidden, inter)}
    gen = torch.Generator().manual_seed(0)
    dense, checkpoint = {}, {}
    for e in range(n_experts):
        for proj, shape in shapes.items():
            packed = torch.randint(-8, 8, shape, generator=gen, dtype=torch.int32)
            scale = torch.rand(shape[0], 1, generator=gen) + 0.5
            dense[proj, e] = (packed.float() * scale).bfloat16()
            key = f"model.layers.0.mlp.experts.{e}.{proj}_proj."
            checkpoint |= {
                key + "weight_packed": packed,
                key + "weight_scale": scale,
                key + "weight_shape": torch.tensor(shape),
            }
    converters = [
        onfly._compressedify(conv, _ScaledCompressor, None, PACKED)
        for conv in _stock_expert_converters()
    ]

    fused = _load(converters, checkpoint)
    gate_up = fused.pop("model.layers.0.mlp.experts.gate_up_proj")
    down = fused.pop("model.layers.0.mlp.experts.down_proj")

    assert not fused
    assert gate_up.dtype == down.dtype == torch.bfloat16
    expected = [torch.cat([dense["gate", e], dense["up", e]]) for e in range(n_experts)]
    assert torch.equal(gate_up, torch.stack(expected))
    assert torch.equal(down, torch.stack([dense["down", e] for e in range(n_experts)]))


def test_only_owned_experts_and_stored_siblings_reach_the_compressor():
    """Under EP the loader hands experts this rank does not own over as ``None``; they are skipped,
    so a rank decompresses only its own. A checkpoint may store fewer params than the scheme
    declares (no ``weight_global_scale`` in Kimi-K3's mxfp4); missing ones are not passed."""
    seen = []

    class RecordingCompressor(_ScaledCompressor):
        @classmethod
        def decompress(cls, state_dict, scheme):
            seen.append((sorted(state_dict), scheme))
            return super().decompress(state_dict, scheme)

    _, down = _stock_expert_converters()
    names = ("weight_packed", "weight_scale", "weight_global_scale")
    decompress = onfly._compressedify(down, RecordingCompressor, "scheme", names).operations[0]
    packed = [torch.full((2, 2), e + 1, dtype=torch.int32) for e in range(3)]
    scale = torch.full((2, 1), 0.5)

    out = decompress.convert(
        {
            "mlp.experts.*.down_proj.weight_packed$": [packed[0], None, packed[2]],
            "mlp.experts.*.down_proj.weight_scale$": [scale, None, scale],
        }
    )

    [(pattern, experts)] = out.items()
    assert pattern == "mlp.experts.*.down_proj.weight_packed$"  # the anchor, for the stock merge
    assert [t.dtype for t in experts] == [torch.bfloat16] * 2
    assert [t[0, 0].item() for t in experts] == [0.5, 1.5]
    assert seen == [(["weight_packed", "weight_scale"], "scheme")] * 2


@pytest.mark.usefixtures("restore_loader_hooks")
def test_experts_merge_in_numeric_not_lexicographic_order():
    """Checkpoint keys iterated as strings yield experts 0, 1, 10, 11, 2, ...; the fused tensor must
    still hold expert i in slot i (unfixed, expert 10 lands in slot 2)."""
    onfly._install_numeric_expert_ordering()
    _, down = _stock_expert_converters()
    keys = {e: f"model.layers.0.mlp.experts.{e}.down_proj.weight" for e in range(12)}
    checkpoint = {keys[e]: torch.full((2, 3), float(e)) for e in sorted(keys, key=str)}
    assert list(checkpoint)[2] == keys[10]  # the order the fix exists for

    fused = _load([down], checkpoint)["model.layers.0.mlp.experts.down_proj"]

    assert fused[:, 0, 0].tolist() == [float(e) for e in range(12)]


@pytest.mark.usefixtures("restore_loader_hooks")
def test_integer_checkpoint_tensors_keep_their_dtype_on_load():
    """Without a quantizer the loader casts every tensor to the model's float dtype, which destroys
    packed weights, shapes and zero-points. Integers stay integers; floats are still cast, and
    integer-to-integer casts still happen."""
    onfly._install_keep_int_native()
    packed = torch.tensor([[2**31 - 15, -(2**31) + 3]], dtype=torch.int32)  # not exact in bf16

    kept = cml._materialize_copy(packed, dtype=torch.bfloat16)

    assert kept.dtype == torch.int32
    assert torch.equal(kept, packed)
    assert cml._materialize_copy(torch.ones(2), dtype=torch.bfloat16).dtype == torch.bfloat16
    assert cml._materialize_copy(packed, dtype=torch.int64).dtype == torch.int64
    # the loader reaches the hook through spawn_materialize, which looks it up at call time
    assert cml.spawn_materialize(None, packed, "cpu", torch.bfloat16)().dtype == torch.int32


# ------------------------------------------------------------------------ installing the rewrite


@pytest.mark.usefixtures("restore_loader_hooks")
def test_install_rewrites_only_the_per_expert_weight_converters(monkeypatch):
    """The loader's conversion list also holds renamings and other converters; only converters that
    fuse per-expert ``.weight`` tensors are rewritten, and from_pretrained's arguments reach the
    stock mapping unchanged."""
    renaming = WeightRenaming(r"experts\.(\d+)\.w1\.weight", r"experts.\1.gate_proj.weight")
    qkv = WeightConverter(
        "self_attn.qkv_proj.weight",
        ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
        [Chunk(dim=0)],
    )
    expert_bias = WeightConverter(
        "mlp.experts.*.down_proj.bias", "mlp.experts.down_proj_bias", [MergeModulelist(dim=0)]
    )
    gate_up, down = _stock_expert_converters()
    stock = [renaming, qkv, expert_bias, gate_up, down]
    calls = []

    def stock_mapping(model, key_mapping=None, hf_quantizer=None, add_legacy=True):
        calls.append((model, key_mapping, hf_quantizer))
        return list(stock)

    compression = (_ScaledCompressor, None, PACKED, "stub-format")
    monkeypatch.setattr(mu, "get_model_conversion_mapping", stock_mapping)
    monkeypatch.setattr(onfly, "_read_compression", lambda cfg: compression)
    config = SimpleNamespace()
    onfly.install_onfly_decompress(config)
    model = SimpleNamespace(config=config)

    # positionally, the way from_pretrained calls it
    out = mu.get_model_conversion_mapping(model, {"old": "new"}, "quantizer")

    assert calls == [(model, {"old": "new"}, "quantizer")]
    assert len(out) == len(stock)
    assert all(kept is original for kept, original in zip(out[:3], stock[:3]))
    assert [conv.source_patterns[0] for conv in out[3:]] == [
        "mlp.experts.*.gate_proj.weight_packed$",
        "mlp.experts.*.down_proj.weight_packed$",
    ]


@pytest.mark.usefixtures("restore_loader_hooks")
def test_installing_twice_wraps_the_loader_once(monkeypatch):
    """``prepare`` may run more than once per process. Installing again for the same checkpoint must
    not wrap the hooks again: with a ``weight`` anchor the outer wrap would rewrite the rewritten
    converter and decompress twice."""
    _, down = _stock_expert_converters()
    reads = []

    def read(cfg):
        reads.append(cfg)
        return _ScaledCompressor, None, ("weight", "weight_scale"), "stub-format"

    def stock_mapping(model, **kwargs):
        return [copy.deepcopy(down)]

    monkeypatch.setattr(mu, "get_model_conversion_mapping", stock_mapping)
    monkeypatch.setattr(onfly, "_read_compression", read)

    def hooks():
        return (
            mu.get_model_conversion_mapping,
            WeightConverter.add_tensor,
            WeightConverter.materialize_tensors,
            cml._materialize_copy,
        )

    onfly.install_onfly_decompress(SimpleNamespace())
    installed = hooks()
    again = SimpleNamespace()
    onfly.install_onfly_decompress(again)

    assert hooks() == installed
    assert len(reads) == 2
    [conv] = mu.get_model_conversion_mapping(SimpleNamespace(config=again))
    assert len(conv.operations) == len(down.operations) + 1


@pytest.mark.usefixtures("restore_loader_hooks")
def test_a_checkpoint_compressed_differently_needs_a_process_of_its_own(monkeypatch):
    """The hooks are process-wide: decompressing a second checkpoint with the first one's scheme and
    stored params would load it wrong without a word, so its install is refused and the hooks stay as
    they are. A config the installing prepare already stripped of its quantization_config is the
    first checkpoint again."""
    _, down = _stock_expert_converters()
    compressions = {
        "a": (_ScaledCompressor, None, PACKED, "stub-format"),
        "b": (_ScaledCompressor, None, (*PACKED, "weight_zero_point"), "stub-format"),
    }

    def read(cfg):
        if cfg.checkpoint is None:
            raise ValueError("install_onfly_decompress: config has no quantization_config")
        return compressions[cfg.checkpoint]

    monkeypatch.setattr(
        mu, "get_model_conversion_mapping", lambda model, **kw: [copy.deepcopy(down)]
    )
    monkeypatch.setattr(onfly, "_read_compression", read)

    onfly.install_onfly_decompress(SimpleNamespace(checkpoint="a"))
    installed = mu.get_model_conversion_mapping
    onfly.install_onfly_decompress(SimpleNamespace(checkpoint=None))
    with pytest.raises(RuntimeError, match="needs a process of its own"):
        onfly.install_onfly_decompress(SimpleNamespace(checkpoint="b"))

    assert mu.get_model_conversion_mapping is installed


def _install_for_deepseek_v3(monkeypatch):
    """Install for a compressed DeepSeek-V3 config; a config without quantization_config reads as none."""
    compression = (_ScaledCompressor, None, PACKED, "stub-format")

    def read(cfg):
        if getattr(cfg, "quantization_config", None) is None:
            raise ValueError("install_onfly_decompress: config has no quantization_config")
        return compression

    monkeypatch.setattr(
        mu, "get_model_conversion_mapping", lambda model, **kw: list(_stock_expert_converters())
    )
    monkeypatch.setattr(onfly, "_read_compression", read)
    config = DeepseekV3Config(num_hidden_layers=1)
    config.quantization_config = {"format": "stub-format"}
    onfly.install_onfly_decompress(config)
    return config


@pytest.mark.usefixtures("restore_loader_hooks")
def test_another_model_loaded_in_the_process_keeps_its_stock_expert_converters(monkeypatch):
    """The hooks outlive the compressed checkpoint's load. Rewritten, an uncompressed MoE's experts
    would match no stored key, load newly initialized, and from_pretrained would not complain."""
    _install_for_deepseek_v3(monkeypatch)
    other = SimpleNamespace(config=DeepseekV3Config(num_hidden_layers=1))

    out = mu.get_model_conversion_mapping(other)

    stock = [conv.source_patterns for conv in _stock_expert_converters()]
    assert [conv.source_patterns for conv in out] == stock
    assert all(p.endswith(".weight") for patterns in stock for p in patterns)


@pytest.mark.usefixtures("restore_loader_hooks")
@pytest.mark.parametrize("repeat", ["stripped", "fresh"])
def test_the_compressed_checkpoint_is_rewritten_through_the_copy_from_pretrained_loads(
    monkeypatch, repeat
):
    """from_pretrained builds the model from a deep copy of the config ``prepare`` handed over. That
    copy must still select the rewrite -- whether ``prepare`` ran again on the config it stripped of
    its quantization_config (Kimi-K3) or on a fresh config of the same checkpoint -- and the mark
    must not reach the config a later save writes."""
    config = _install_for_deepseek_v3(monkeypatch)
    if repeat == "stripped":
        config.quantization_config = None
    else:
        config = DeepseekV3Config(num_hidden_layers=1)
        config.quantization_config = {"format": "stub-format"}
    onfly.install_onfly_decompress(config)
    model = SimpleNamespace(config=copy.deepcopy(config))

    out = mu.get_model_conversion_mapping(model)

    assert [conv.source_patterns[0] for conv in out] == [
        "mlp.experts.*.gate_proj.weight_packed$",
        "mlp.experts.*.down_proj.weight_packed$",
    ]
    assert onfly._MARK not in model.config.to_dict()


# ------------------------------------------------------------------------ reading the checkpoint


@pytest.mark.parametrize(
    ("config", "found"),
    [
        (
            SimpleNamespace(
                quantization_config={"at": "top"},
                text_config=SimpleNamespace(quantization_config={"at": "text"}),
            ),
            {"at": "top"},
        ),
        (
            SimpleNamespace(text_config=SimpleNamespace(quantization_config={"at": "text"})),
            {"at": "text"},
        ),
        ({"text_config": {"quantization_config": {"at": "dict"}}}, {"at": "dict"}),
        (
            SimpleNamespace(quantization_config=SimpleNamespace(to_dict=lambda: {"at": "obj"})),
            {"at": "obj"},
        ),
    ],
    ids=["top-level-first", "text-config", "raw-dict", "config-object"],
)
def test_the_quantization_config_is_found_at_the_top_level_then_on_the_text_config(config, found):
    """A VLM may carry its compressed-tensors config at the top level or on ``text_config``, as a
    config object or a raw dict; it is always returned as a plain dict."""
    assert onfly._find_quantization_config(config) == found


def test_a_config_without_a_quantization_config_is_a_value_error():
    """Callers rely on exactly this error to mean "plain experts, nothing to decompress"."""
    with pytest.raises(ValueError, match="quantization_config"):
        onfly._find_quantization_config(SimpleNamespace(text_config=SimpleNamespace()))


@pytest.mark.usefixtures("restore_loader_hooks")
def test_a_config_with_nothing_to_decompress_leaves_the_loader_alone(monkeypatch):
    """By the time a bf16 checkpoint's caller sees that error, nothing may be patched yet -- and
    compressed_tensors, which such a checkpoint never needs, need not be installed."""
    monkeypatch.setitem(sys.modules, "compressed_tensors", None)
    hooks = (
        mu.get_model_conversion_mapping,
        WeightConverter.add_tensor,
        WeightConverter.materialize_tensors,
        cml._materialize_copy,
    )

    with pytest.raises(ValueError, match="quantization_config"):
        onfly.install_onfly_decompress(SimpleNamespace(text_config=SimpleNamespace()))

    assert (
        mu.get_model_conversion_mapping,
        WeightConverter.add_tensor,
        WeightConverter.materialize_tensors,
        cml._materialize_copy,
    ) == hooks


class _RemoteTextConfig(PretrainedConfig):
    """Stands in for the remote-code DeepSeek-V3 config a VLM bundles as its ``text_config``."""

    model_type = "remote_deepseek_v3"


@pytest.mark.parametrize("as_dict", [False, True], ids=["config-object", "raw-dict"])
def test_the_native_backbone_config_is_a_clean_stock_deepseek_v3(as_dict):
    """The bundled text config becomes transformers' own ``DeepseekV3Config`` with the same
    architecture, minus ``auto_map`` (which routes back to the remote code) and the stale
    ``quantization_config`` (which would build a competing quantizer). ``architectures`` names the
    native class: hf_ptq's VLM detection iterates it."""
    fields = {
        "hidden_size": 64,
        "num_hidden_layers": 3,
        "n_routed_experts": 8,
        "moe_intermediate_size": 16,
        "auto_map": {"AutoConfig": "configuration_deepseek.DeepseekV3Config"},
        "quantization_config": {"quant_method": "compressed-tensors"},
    }
    if as_dict:
        text = {**fields, "model_type": "remote_deepseek_v3"}
    else:
        text = _RemoteTextConfig(**fields)

    native = onfly.to_native_backbone_config(SimpleNamespace(text_config=text))

    assert type(native) is DeepseekV3Config
    assert native.model_type == "deepseek_v3"
    assert native.architectures == ["DeepseekV3ForCausalLM"]
    assert (native.hidden_size, native.num_hidden_layers) == (64, 3)
    assert (native.n_routed_experts, native.moe_intermediate_size) == (8, 16)
    assert getattr(native, "auto_map", None) is None
    assert getattr(native, "quantization_config", None) is None


def test_the_native_backbone_config_needs_a_text_config_and_a_known_model_type():
    with pytest.raises(ValueError, match="text_config"):
        onfly.to_native_backbone_config(SimpleNamespace())
    with pytest.raises(ValueError, match="unknown native model_type"):
        onfly.to_native_backbone_config(SimpleNamespace(text_config={}), model_type="not_a_model")


# -------------------------------------------------------- with the real compressed_tensors library


def _int4(symmetric, targets):
    """A compressed-tensors INT4, group-32 config group."""
    weights = {"num_bits": 4, "type": "int", "symmetric": symmetric, "strategy": "group"}
    return {"targets": targets, "weights": {**weights, "group_size": 32}}


def _pack_quantized(**groups):
    """A compressed-tensors ``quantization_config`` in the ``pack-quantized`` format."""
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": groups,
    }


ATTN = ["re:.*self_attn.*"]
EXPERTS = ["re:.*experts.*"]


@pytest.mark.parametrize(
    ("groups", "zero_point"),
    [
        ({"attn": _int4(True, ATTN), "moe": _int4(False, EXPERTS)}, True),
        ({"moe": _int4(True, EXPERTS), "attn": _int4(False, ATTN)}, False),
        ({"attn": _int4(False, ATTN), "dense": _int4(True, ["Linear"])}, True),
    ],
    ids=["expert-group-second", "expert-group-first", "no-expert-group-takes-the-first"],
)
def test_the_expert_config_group_decides_the_stored_params(groups, zero_point):
    """With several config groups the one targeting experts wins, else the first; an asymmetric
    scheme stores a zero-point next to the packed weight, a symmetric one does not."""
    pytest.importorskip(
        "compressed_tensors", reason="needs compressed-tensors (CI's unit session installs it)"
    )
    config = SimpleNamespace(quantization_config=_pack_quantized(**groups))

    _, scheme, names, fmt = onfly._read_compression(config)

    assert fmt == "pack-quantized"
    assert names[: len(PACKED)] == PACKED
    assert ("weight_zero_point" in names) is zero_point
    assert scheme.weights.symmetric is not zero_point


def test_real_pack_quantized_experts_round_trip_exactly():
    """With the real compressed_tensors INT4 compressor, packed experts come back out of the fused
    converter as the original weights (chosen to be exact in INT4 x 0.5 and in bf16)."""
    pytest.importorskip(
        "compressed_tensors", reason="needs compressed-tensors (CI's unit session installs it)"
    )
    config = SimpleNamespace(quantization_config=_pack_quantized(experts=_int4(True, ["Linear"])))
    compressor, scheme, names, _ = onfly._read_compression(config)
    n_experts, inter, hidden = 3, 4, 64
    gen = torch.Generator().manual_seed(0)
    dense, checkpoint = {}, {}
    for e in range(n_experts):
        for proj in ("gate", "up"):
            dense[proj, e] = torch.randint(-8, 8, (inter, hidden), generator=gen).float() * 0.5
            scale = torch.full((inter, hidden // 32), 0.5)
            stored = compressor.compress({"weight": dense[proj, e], "weight_scale": scale}, scheme)
            key = f"model.layers.0.mlp.experts.{e}.{proj}_proj."
            checkpoint |= {key + name: stored[name] for name in names}
    gate_up, _ = _stock_expert_converters()

    fused = _load([onfly._compressedify(gate_up, compressor, scheme, names)], checkpoint)

    expected = [torch.cat([dense["gate", e], dense["up", e]]) for e in range(n_experts)]
    assert torch.equal(
        fused["model.layers.0.mlp.experts.gate_up_proj"], torch.stack(expected).bfloat16()
    )
