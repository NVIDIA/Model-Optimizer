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
with a stand-in compressor, so they need neither a checkpoint nor ``compressed_tensors``. The tests
that use the real ``compressed_tensors`` registry, among them real ``from_pretrained`` loads of tiny
compressed checkpoints, skip when it cannot be imported.
"""

import copy
import json
import sys
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

transformers = pytest.importorskip("transformers")

from packaging.version import Version

if Version(transformers.__version__) < Version("5.8"):
    pytest.skip("the on-the-fly decompression needs transformers >= 5.8", allow_module_level=True)

import transformers.core_model_loading as cml
import transformers.modeling_utils as mu
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    DeepseekV3Config,
    LlamaConfig,
    MixtralConfig,
    PretrainedConfig,
)
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import (
    Chunk,
    Concatenate,
    MergeModulelist,
    WeightConverter,
    WeightRenaming,
)

from modelopt.torch.models import _onfly_decompress as onfly

PACKED = ("weight_packed", "weight_scale", "weight_shape")  # what pack-quantized stores per weight


@pytest.fixture
def restore_loader_hooks(monkeypatch):
    """Undo the process-wide transformers hooks the decompressor installs, and start the test with
    no decompressing load under way, whatever an earlier test's load left."""
    for owner, name in ((cml, "spawn_materialize"), (mu, "get_model_conversion_mapping")):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    monkeypatch.setattr(onfly, "_LOAD", {"decompressing": False})


class _ScaledCompressor:
    """Stands in for a compressed_tensors compressor: ``weight = stored weight * weight_scale``,
    the stored weight being ``weight_packed`` (pack-quantized) or ``weight`` (float formats)."""

    @classmethod
    def decompress(cls, state_dict, scheme):
        stored = state_dict.get("weight_packed", state_dict.get("weight"))
        return {"weight": stored.float() * state_dict["weight_scale"]}


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


def test_only_expert_weight_sources_are_rewritten_and_stored_params_need_their_weight():
    """Only the experts' ``.weight`` sources name a compressed tensor: a converter reading another --
    an expert's bias, or a dense weight fused with the experts' -- is refused where it is rewritten,
    before any tensor is read. A stored param without its weight still reaches the decompress op,
    which refuses it rather than hand the merge a tensor it would fuse into the wrong slot."""
    for other in ("mlp.experts.*.gate_proj.bias", "mlp.gate_proj.weight"):
        conv = WeightConverter(
            source_patterns=["mlp.experts.*.gate_proj.weight", other],
            target_patterns="mlp.experts.gate_proj",
            operations=[MergeModulelist(dim=0)],
        )
        with pytest.raises(NotImplementedError, match=r"alongside the experts. weights"):
            onfly._compressedify(conv, _ScaledCompressor, None, PACKED)

    gate_up, _ = _stock_expert_converters()
    scales_alone = {"model.layers.0.mlp.experts.0.gate_proj.weight_scale": torch.ones(2, 2)}
    with pytest.raises(ValueError, match=r"unconsumed checkpoint keys .*gate_proj\.weight_scale"):
        _load([onfly._compressedify(gate_up, _ScaledCompressor, None, PACKED)], scales_alone)


def test_a_rewritten_converter_keeps_the_scope_transformers_set_on_it():
    """transformers scopes a sub-model's converters (a VLM's language model) once it has built them;
    the rewrite keeps whichever of those fields this release has."""
    _, down = _stock_expert_converters()
    down = copy.deepcopy(down)
    fields = {"scope_prefix": "language_model", "base_model_prefix": "model", "force_cpu": True}
    fields = {name: value for name, value in fields.items() if hasattr(type(down), name)}
    for name, value in fields.items():
        setattr(down, name, value)

    rewritten = onfly._compressedify(down, _ScaledCompressor, None, PACKED)

    assert {name: getattr(rewritten, name) for name in fields} == fields


# ----------------------------------------------------------------- decompressing during the load


@pytest.mark.parametrize(
    "stored", [PACKED, ("weight", "weight_scale")], ids=["pack-quantized", "weight-and-scale"]
)
def test_packed_experts_decompress_then_fuse_like_an_uncompressed_checkpoint(stored):
    """Through transformers' loader objects, per-expert compressed weights and scales come out as
    the fused bf16 ``gate_up_proj`` ([gate; up] per expert) and the stacked ``down_proj``.

    The decompress op runs first, as a ``MergeModulelist``: loaders shard experts at load only when
    the first op is one. A float format stores ``weight`` next to ``weight_scale``, and transformers
    regex-*searches* source patterns, so only anchored patterns keep a weight from claiming its
    scale.
    """
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
            values = {"weight_packed": packed, "weight": packed, "weight_scale": scale}
            values["weight_shape"] = torch.tensor(shape)
            checkpoint |= {key + name: values[name] for name in stored}
    converters = [
        onfly._compressedify(conv, _ScaledCompressor, None, stored)
        for conv in _stock_expert_converters()
    ]
    assert all(isinstance(conv.operations[0], MergeModulelist) for conv in converters)

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
    declares (no ``weight_global_scale`` in Kimi-K3's mxfp4); missing ones are not passed, but
    one stored for some experts and not others is refused."""
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
    # Paired by position, an expert missing one of them would shift every later expert's pairing.
    short = {
        "mlp.experts.*.down_proj.weight_packed$": packed,
        "mlp.experts.*.down_proj.weight_scale$": [scale, scale],
    }
    with pytest.raises(ValueError, match="3 weight_packed, 2 weight_scale, not one per expert"):
        decompress.convert(short)


PACKED_INT32 = torch.tensor([[2**31 - 15, -(2**31) + 3]], dtype=torch.int32)  # not exact in bf16


def _read(path, sharding_op=None, dtype=torch.bfloat16):
    """Read ``path``'s one tensor the way the loader does: a safetensors slice, through spawn_materialize."""
    with safe_open(str(path), framework="pt") as stored:
        [key] = stored.keys()
        kwargs = {"sharding_op": sharding_op, "tensor_idx": 0} if sharding_op else {}
        return cml.spawn_materialize(None, stored.get_slice(key), "cpu", dtype, **kwargs)()


@pytest.mark.usefixtures("restore_loader_hooks")
def test_integer_checkpoint_tensors_keep_their_dtype_on_load(tmp_path):
    """Without a quantizer the loader casts every tensor to the model's float dtype as it reads it,
    which destroys packed weights, shapes and zero-points. In a decompressing load integers stay
    integers; floats are still cast, and integer-to-integer casts still happen. Any other load casts
    as before."""
    save_file({"packed": PACKED_INT32}, str(tmp_path / "packed.safetensors"))
    onfly._install_keep_int_native()
    assert _read(tmp_path / "packed.safetensors").dtype == torch.bfloat16
    onfly._LOAD["decompressing"] = True

    kept = _read(tmp_path / "packed.safetensors")

    assert kept.dtype == torch.int32
    assert torch.equal(kept, PACKED_INT32)
    assert _read(tmp_path / "packed.safetensors", dtype=torch.int64).dtype == torch.int64
    assert (
        cml.spawn_materialize(None, torch.ones(2), "cpu", torch.bfloat16)().dtype == torch.bfloat16
    )
    assert cml.spawn_materialize(None, PACKED_INT32, "cpu", torch.bfloat16)().dtype == torch.int32


@pytest.mark.usefixtures("restore_loader_hooks")
def test_an_integer_sharded_onto_the_expert_mesh_keeps_its_dtype(tmp_path, gloo_world_of_one):
    """From transformers 5.14 an expert-parallel load shards each expert's tensor onto the mesh as it
    reads it, and the shard op casts it to the parameter's dtype, not the copy: a decompressing load
    keeps the packed integers there too."""
    sharding_utils = pytest.importorskip("transformers.distributed.sharding_utils")
    from torch.distributed.tensor import Shard, distribute_tensor

    experts = torch.zeros(2, 4, 8, dtype=torch.bfloat16)  # a fused expert param, sharded by expert
    shard = sharding_utils.DtensorShardOperation(
        distribute_tensor(experts, gloo_world_of_one, [Shard(0)])
    )
    save_file({"packed": PACKED_INT32}, str(tmp_path / "packed.safetensors"))
    onfly._install_keep_int_native()
    onfly._LOAD["decompressing"] = True

    kept = _read(tmp_path / "packed.safetensors", sharding_op=shard)

    assert kept.dtype == torch.int32
    assert torch.equal(kept, PACKED_INT32)


@pytest.mark.usefixtures("restore_loader_hooks")
def test_a_read_parameter_a_release_adds_passes_through(monkeypatch):
    """The patch sits on every later load's reads, so a parameter past the dtype (5.14's sharding_op
    and tensor_idx, or one a release adds) reaches the loader's own, positionally or by keyword."""
    seen = []

    def release_spawn(
        thread_pool, tensor, device=None, dtype=None, sharding_op=None, tensor_idx=None
    ):
        seen.append((sharding_op, tensor_idx))
        return tensor

    monkeypatch.setattr(cml, "spawn_materialize", release_spawn)
    onfly._install_keep_int_native()

    cml.spawn_materialize(None, torch.ones(2), "cpu", torch.bfloat16, "op", 3)
    cml.spawn_materialize(
        None, torch.ones(2), "cpu", torch.bfloat16, sharding_op="op", tensor_idx=3
    )

    assert seen == [("op", 3), ("op", 3)]


# ------------------------------------------------------------------------ installing the rewrite


@pytest.mark.usefixtures("restore_loader_hooks")
def test_install_rewrites_only_the_per_expert_weight_converters(monkeypatch):
    """The loader's conversion list also holds renamings and other converters; only converters that
    fuse per-expert ``.weight`` tensors are rewritten, not a shared expert's, and from_pretrained's
    arguments reach the stock mapping unchanged."""
    renaming = WeightRenaming(r"experts\.(\d+)\.w1\.weight", r"experts.\1.gate_proj.weight")
    qkv = WeightConverter(
        "self_attn.qkv_proj.weight",
        ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
        [Chunk(dim=0)],
    )
    expert_bias = WeightConverter(
        "mlp.experts.*.down_proj.bias", "mlp.experts.down_proj_bias", [MergeModulelist(dim=0)]
    )
    shared_expert = WeightConverter(
        ["mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight"],
        "mlp.shared_experts.gate_up_proj",
        [Concatenate(dim=0)],
    )
    gate_up, down = _stock_expert_converters()
    stock = [renaming, qkv, expert_bias, shared_expert, gate_up, down]
    calls = []

    def stock_mapping(model, key_mapping=None, hf_quantizer=None, add_legacy=True):
        calls.append((model, key_mapping, hf_quantizer, add_legacy))
        return list(stock)

    compression = (_ScaledCompressor, None, PACKED, "stub-format")
    monkeypatch.setattr(mu, "get_model_conversion_mapping", stock_mapping)
    monkeypatch.setattr(onfly, "_read_compression", lambda cfg: compression)
    config = SimpleNamespace()
    onfly.install_onfly_decompress(config)
    model = SimpleNamespace(config=config)

    # positionally, the way from_pretrained calls it, and a later parameter too
    out = mu.get_model_conversion_mapping(model, {"old": "new"}, "quantizer", False)

    assert calls == [(model, {"old": "new"}, "quantizer", False)]
    assert len(out) == len(stock)
    assert all(kept is original for kept, original in zip(out[:4], stock[:4]))
    assert [conv.source_patterns[0] for conv in out[4:]] == [
        "mlp.experts.*.gate_proj.weight_packed$",
        "mlp.experts.*.down_proj.weight_packed$",
    ]


@pytest.mark.usefixtures("restore_loader_hooks")
def test_a_marked_model_with_no_expert_converter_to_rewrite_is_refused(monkeypatch):
    """With nothing rewritten, the stock converters would look for plain ``.weight`` keys the
    checkpoint does not store, and the experts would load newly initialized. The load is refused,
    and the config, whose converters were not rewritten, keeps its mark."""
    renaming = WeightRenaming(r"experts\.(\d+)\.w1\.weight", r"experts.\1.gate_proj.weight")
    monkeypatch.setattr(mu, "get_model_conversion_mapping", lambda model, **kw: [renaming])
    compression = (_ScaledCompressor, None, PACKED, "stub-format")
    monkeypatch.setattr(onfly, "_read_compression", lambda cfg: compression)
    config = SimpleNamespace()
    onfly.install_onfly_decompress(config)
    model = SimpleNamespace(config=copy.deepcopy(config))

    with pytest.raises(RuntimeError, match="no per-expert converter to rewrite"):
        mu.get_model_conversion_mapping(model)
    assert onfly.is_marked(model.config)


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
        return mu.get_model_conversion_mapping, cml.spawn_materialize

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
    first checkpoint again; one with none that was never marked is refused, since nothing would
    decompress its experts; one whose quantization_config fails to parse is not, and its error
    goes through."""
    _, down = _stock_expert_converters()
    compressions = {
        "a": (_ScaledCompressor, None, PACKED, "stub-format"),
        "b": (_ScaledCompressor, None, (*PACKED, "weight_zero_point"), "stub-format"),
        None: None,
    }

    def read(cfg):
        if cfg.checkpoint == "malformed":
            # compressed_tensors' parse raises pydantic's ValidationError, a ValueError.
            raise ValueError("1 validation error for QuantizationConfig")
        return compressions[cfg.checkpoint]

    monkeypatch.setattr(
        mu, "get_model_conversion_mapping", lambda model, **kw: [copy.deepcopy(down)]
    )
    monkeypatch.setattr(onfly, "_read_compression", read)

    first = SimpleNamespace(checkpoint="a")
    onfly.install_onfly_decompress(first)
    installed = mu.get_model_conversion_mapping
    first.checkpoint = None  # its prepare dropped the quantization_config after installing
    onfly.install_onfly_decompress(first)
    with pytest.raises(ValueError, match="is not marked"):
        onfly.install_onfly_decompress(SimpleNamespace(checkpoint=None))
    with pytest.raises(RuntimeError, match="needs a process of its own"):
        onfly.install_onfly_decompress(SimpleNamespace(checkpoint="b"))
    with pytest.raises(ValueError, match="validation error"):
        onfly.install_onfly_decompress(SimpleNamespace(checkpoint="malformed"))

    assert mu.get_model_conversion_mapping is installed


def _install_for_deepseek_v3(monkeypatch):
    """Install for a compressed DeepSeek-V3 config; a config without quantization_config reads as none."""
    compression = (_ScaledCompressor, None, PACKED, "stub-format")

    def read(cfg):
        return None if getattr(cfg, "quantization_config", None) is None else compression

    monkeypatch.setattr(
        mu, "get_model_conversion_mapping", lambda model, **kw: list(_stock_expert_converters())
    )
    monkeypatch.setattr(onfly, "_read_compression", read)
    config = DeepseekV3Config(num_hidden_layers=1)
    config.quantization_config = {"format": "stub-format"}
    onfly.install_onfly_decompress(config)
    return config


@pytest.mark.usefixtures("restore_loader_hooks")
@pytest.mark.parametrize("loaded", ["another-model", "stripped-copy", "fresh-config"])
def test_only_the_compressed_checkpoint_loads_through_the_rewritten_converters(monkeypatch, loaded):
    """The hooks outlive the compressed checkpoint's load. Rewritten, another (uncompressed) MoE's
    experts would match no stored key, load newly initialized, and from_pretrained would not
    complain. from_pretrained builds the model from a deep copy of the config ``prepare`` handed
    over; that copy must still select the rewrite, whether ``prepare`` ran again on the config it
    stripped of its quantization_config (Kimi-K3) or on a fresh config of the same checkpoint, and
    the mark must not reach the config a later save writes."""
    config = _install_for_deepseek_v3(monkeypatch)
    if loaded == "another-model":
        config = DeepseekV3Config(num_hidden_layers=1)
    else:
        if loaded == "stripped-copy":
            config.quantization_config = None
        else:
            config = DeepseekV3Config(num_hidden_layers=1)
            config.quantization_config = {"format": "stub-format"}
        onfly.install_onfly_decompress(config)
    model = SimpleNamespace(config=copy.deepcopy(config))

    out = mu.get_model_conversion_mapping(model)

    stored = "weight" if loaded == "another-model" else "weight_packed$"
    assert [conv.source_patterns[0] for conv in out] == [
        f"mlp.experts.*.gate_proj.{stored}",
        f"mlp.experts.*.down_proj.{stored}",
    ]
    assert onfly._MARK not in model.config.to_dict()


@pytest.mark.usefixtures("restore_loader_hooks")
@pytest.mark.parametrize("marked", ["vlm", "text-config"])
@pytest.mark.parametrize("loaded", ["vlm", "backbone"])
def test_a_vlm_marked_at_either_level_loads_through_the_rewritten_converters(
    monkeypatch, marked, loaded
):
    """A VLM carries its compressed-tensors config at the top level or on its text config, and the
    install marks the config it is handed. Whether the load builds the VLM, its text config replaced
    by the native backbone's, or that backbone on its own, its expert converters are rewritten, and
    neither level keeps the mark for a later save."""
    monkeypatch.setattr(
        mu, "get_model_conversion_mapping", lambda model, **kw: list(_stock_expert_converters())
    )
    compression = (_ScaledCompressor, None, PACKED, "stub-format")
    monkeypatch.setattr(onfly, "_read_compression", lambda cfg: compression)
    text = _RemoteTextConfig(hidden_size=64, quantization_config={"format": "stub-format"})
    vlm = SimpleNamespace(text_config=text)
    onfly.install_onfly_decompress(vlm if marked == "vlm" else text)
    vlm.text_config = onfly.to_native_backbone_config(vlm)
    model = SimpleNamespace(config=copy.deepcopy(vlm if loaded == "vlm" else vlm.text_config))

    out = mu.get_model_conversion_mapping(model)

    assert [conv.source_patterns[0] for conv in out] == [
        "mlp.experts.*.gate_proj.weight_packed$",
        "mlp.experts.*.down_proj.weight_packed$",
    ]
    assert not onfly.is_marked(model.config)


@pytest.mark.usefixtures("restore_loader_hooks")
def test_a_language_model_config_named_llm_config_is_read_as_text_config_is(monkeypatch):
    """Some VLMs name their language-model config llm_config, which the expert-parallel load reads
    as it reads text_config: marked there, the VLM loads through the rewritten converters, and the
    mark is cleared."""
    monkeypatch.setattr(
        mu, "get_model_conversion_mapping", lambda model, **kw: list(_stock_expert_converters())
    )
    compression = (_ScaledCompressor, None, PACKED, "stub-format")
    monkeypatch.setattr(onfly, "_read_compression", lambda cfg: compression)
    llm = SimpleNamespace()
    onfly.install_onfly_decompress(llm)
    model = SimpleNamespace(config=SimpleNamespace(llm_config=copy.deepcopy(llm)))

    out = mu.get_model_conversion_mapping(model)

    assert [conv.source_patterns[0] for conv in out] == [
        "mlp.experts.*.gate_proj.weight_packed$",
        "mlp.experts.*.down_proj.weight_packed$",
    ]
    assert not onfly.is_marked(model.config)


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
        (
            SimpleNamespace(llm_config=SimpleNamespace(quantization_config={"at": "llm"})),
            {"at": "llm"},
        ),
        (
            SimpleNamespace(quantization_config=SimpleNamespace(to_dict=lambda: {"at": "obj"})),
            {"at": "obj"},
        ),
    ],
    ids=["top-level-first", "text-config", "llm-config", "config-object"],
)
def test_the_quantization_config_is_found_at_the_top_level_then_on_the_text_config(config, found):
    """A VLM may carry its compressed-tensors config at the top level or on its language-model
    config (``text_config`` or ``llm_config``), as a config object or a dict; it is always returned
    as a plain dict."""
    assert onfly._find_quantization_config(config) == found


@pytest.mark.usefixtures("restore_loader_hooks")
@pytest.mark.parametrize(
    ("config", "error", "match"),
    [
        (SimpleNamespace(text_config=SimpleNamespace()), ValueError, "quantization_config"),
        ({"quantization_config": {"format": "pack-quantized"}}, TypeError, "not a dict"),
        (
            SimpleNamespace(dtype=torch.float16, quantization_config={"format": "pack-quantized"}),
            NotImplementedError,
            "loads in torch.float16",
        ),
        (
            SimpleNamespace(
                text_config=SimpleNamespace(dtype=torch.float16),
                quantization_config={"format": "pack-quantized"},
            ),
            NotImplementedError,
            "loads in torch.float16",
        ),
    ],
    ids=["nothing-to-decompress", "raw-dict", "not-bf16", "not-bf16-text-config"],
)
def test_a_config_with_nothing_to_decompress_leaves_the_loader_alone(
    monkeypatch, config, error, match
):
    """By the time a bf16 checkpoint's caller sees that error, nothing may be patched yet -- and
    compressed_tensors, which such a checkpoint never needs, need not be installed. Nor for a raw
    dict, which cannot carry the mark its load would be found by, or for a load in a dtype other
    than the bf16 the experts decompress to, at any config level (a backbone config derived from the
    text config keeps its dtype): the loader would set them beside the rest uncast."""
    monkeypatch.setitem(sys.modules, "compressed_tensors", None)
    hooks = (
        mu.get_model_conversion_mapping,
        WeightConverter.add_tensor,
        WeightConverter.materialize_tensors,
        cml.spawn_materialize,
    )

    with pytest.raises(error, match=match):
        onfly.install_onfly_decompress(config)

    assert (
        mu.get_model_conversion_mapping,
        WeightConverter.add_tensor,
        WeightConverter.materialize_tensors,
        cml.spawn_materialize,
    ) == hooks


@pytest.mark.usefixtures("restore_loader_hooks")
def test_a_transformers_below_the_floor_is_named_before_compressed_tensors(monkeypatch):
    """On a transformers the decompression cannot run on, the first error names the release, not
    the compressed-tensors package a user would install only to hit the release next."""
    monkeypatch.setitem(sys.modules, "compressed_tensors", None)
    monkeypatch.setattr(transformers, "__version__", "5.5.4")

    with pytest.raises(RuntimeError, match=r"transformers >= 5\.8"):
        onfly.install_onfly_decompress(
            SimpleNamespace(quantization_config={"format": "pack-quantized"})
        )


EXPERTS = [f"model.layers.1.mlp.experts.{e}.gate_proj" for e in range(3)]
EXPERT = EXPERTS[0]
ATTENTION = "model.layers.1.self_attn.q_proj"
SHARED_EXPERT = "model.layers.1.mlp.shared_experts.up_proj"  # a dense MLP, not the routed experts
STORED_DTYPES = {  # as compressed-tensors stores them; every other tensor is bf16 here
    "weight_packed": torch.int32,
    "weight_shape": torch.int32,
    "weight_global_scale": torch.float32,
}


@pytest.mark.parametrize(
    ("stored_as", "names", "match"),
    [
        (
            PACKED,
            [f"{EXPERT}.{n}" for n in PACKED] + [f"{ATTENTION}.weight", f"{SHARED_EXPERT}.weight"],
            None,
        ),
        (
            ("weight", "weight_scale"),
            [f"{EXPERT}.weight", f"{EXPERT}.weight_scale", f"{ATTENTION}.weight"],
            None,
        ),
        (PACKED, [f"{p}.{n}" for p in (EXPERT, ATTENTION) for n in PACKED], "1 of them outside"),
        (PACKED, [f"{EXPERT}.weight"] + [f"{ATTENTION}.{n}" for n in PACKED], "1 of them outside"),
        (
            PACKED,
            [f"{EXPERT}.{n}" for n in PACKED] + ["model.layers.1.mlp.experts.1.gate_proj.weight"],
            "1 expert weights uncompressed",
        ),
        (PACKED, [f"{ATTENTION}.weight", f"{SHARED_EXPERT}.weight"], "compresses 0 weights"),
        (
            (*PACKED, "weight_global_scale"),
            [f"{EXPERT}.{n}" for n in (*PACKED, "weight_global_scale")],
            r"store weight_global_scale \(F32\), which bf16 cannot hold exactly",
        ),
        (
            PACKED,
            [f"{EXPERT}.{n}" for n in PACKED] + [f"{EXPERTS[1]}.weight_scale"],
            "stores weight_packed for some compressed weights but not for 1 others",
        ),
        (
            PACKED,
            [f"{EXPERT}.{n}" for n in (*PACKED, "weight_zero_point")],
            "stores 1 tensors under its compressed weights that this compressed-tensors release",
        ),
        (PACKED, [f"{EXPERT}.{n}" for n in (*PACKED, "bias")], None),
        (
            PACKED,
            [f"{EXPERTS[e]}.{n}" for e in (0, 1) for n in ("weight_packed", "weight_shape")]
            + [f"{EXPERTS[e]}.weight_scale" for e in (1, 2)],
            "stores weight_packed for some compressed weights but not for 1 others",
        ),
    ],
    ids=[
        "experts-alone",
        "experts-alone-float-format",
        "every-linear",
        "all-but-the-experts",
        "an-expert-left-uncompressed",
        "nothing-compressed",
        "an-fp32-scale",
        "a-scale-alone",
        "a-name-this-release-does-not-know",
        "an-expert-bias-loads-apart",
        "equal-counts-other-experts",
    ],
)
def test_a_checkpoint_must_compress_its_experts_and_nothing_else(tmp_path, stored_as, names, match):
    """The config cannot say which modules a class target such as Linear compressed, so the names
    the checkpoint stores decide: anything else compressed, or an expert stored uncompressed, would
    load newly initialized, without a quantizer to load it or a converter that reads it. A float
    stored for the experts must be one bf16 holds exactly: the load may cast it to bf16 first. And
    every compressed weight stores the same names, all of them names the installed compressed-tensors
    gives the scheme: the load pairs them by position, not by expert, so a scale stored for other
    experts in equal number would pair with the wrong weights, and a name it does not know would be
    left out of the decompression."""
    tensors = {
        name: torch.zeros(1, dtype=STORED_DTYPES.get(name.rpartition(".")[2], torch.bfloat16))
        for name in names
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    config = SimpleNamespace(text_config=SimpleNamespace())
    onfly._mark(config.text_config, stored_as)

    if match is None:
        onfly.check_compressed_keys(config, tmp_path)
    else:
        with pytest.raises(NotImplementedError, match=match):
            onfly.check_compressed_keys(config, tmp_path)


def test_an_unmarked_config_has_no_stored_names_to_check(tmp_path):
    """The check reads what the checkpoint stores by the names the mark holds, so a config the
    decompressor never marked is refused, by name, rather than with a bare StopIteration."""
    save_file({f"{EXPERT}.weight_packed": torch.zeros(1)}, str(tmp_path / "model.safetensors"))

    with pytest.raises(ValueError, match="not marked for on-the-fly decompression"):
        onfly.check_compressed_keys(SimpleNamespace(text_config=SimpleNamespace()), tmp_path)


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
    ("groups", "error", "match"),
    [
        (
            {"attn": _int4(True, ATTN), "moe": _int4(False, EXPERTS)},
            NotImplementedError,
            "2 compressed-tensors config groups",
        ),
        (
            {"attn": _int4(False, ATTN), "dense": _int4(True, ["Linear"])},
            NotImplementedError,
            "2 compressed-tensors config groups",
        ),
        ({"attn": _int4(True, ATTN)}, NotImplementedError, "not the experts"),
        ({}, ValueError, "nothing to decompress"),
    ],
    ids=["an-expert-group-among-others", "no-expert-group", "one-group-not-the-experts", "none"],
)
def test_a_checkpoint_whose_experts_are_not_its_one_compressed_group_is_refused(
    groups, error, match
):
    """Only the expert converters are rewritten: another group's weights would load newly
    initialized, a group guessed for the experts would decompress them with the wrong scheme, and
    experts no group targets are stored as plain weights the rewritten converters would not find."""
    pytest.importorskip(
        "compressed_tensors", reason="needs compressed-tensors (CI's unit session installs it)"
    )
    config = SimpleNamespace(quantization_config=_pack_quantized(**groups))

    with pytest.raises(error, match=match):
        onfly._read_compression(config)


@pytest.mark.parametrize(
    ("top", "group", "match"),
    [("mixed-precision", "pack-quantized", None), ("fakequant", None, "no compressor")],
    ids=["the-group-names-it", "none-registered"],
)
def test_the_format_is_the_one_group_s_own_and_must_have_a_compressor(top, group, match):
    """compressed-tensors writes "mixed-precision" at the top level once a group names its own
    format; the one group's format is the one to decompress with. A format nothing registers is
    refused by this module, rather than failing inside the library's registry."""
    pytest.importorskip(
        "compressed_tensors", reason="needs compressed-tensors (CI's unit session installs it)"
    )
    experts = _int4(True, EXPERTS)
    if group is not None:
        experts["format"] = group
    config = SimpleNamespace(quantization_config={**_pack_quantized(moe=experts), "format": top})

    if match is None:
        assert onfly._read_compression(config)[3] == group
    else:
        with pytest.raises(NotImplementedError, match=match):
            onfly._read_compression(config)


# The compressions the published Kimi checkpoints use: INT4 (Kimi-K2.5) and MXFP4 (Kimi-K3), each
# with the values its format holds exactly, and a scale it holds exactly.
_INT4 = {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 32}
_FP4_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
_FORMATS = {
    "pack-quantized": (_INT4, torch.arange(-8, 8) * 0.5, 0.5),
    "mxfp4-pack-quantized": (
        {**_INT4, "type": "float", "scale_dtype": "torch.uint8"},
        torch.cat([_FP4_E2M1, -_FP4_E2M1[1:]]),
        1.0,
    ),
}


def _compressed_mixtral(path, fmt):
    """Save a tiny Mixtral whose experts ``fmt`` compresses; return its config and dense experts.

    Mixtral's checkpoint names its experts ``w1``/``w3``/``w2``, which the model fuses and the export
    names back: names a save gets wrong would show.
    """
    weights, values, scale = _FORMATS[fmt]
    config = MixtralConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=1,
        tie_word_embeddings=False,
    )
    AutoModelForCausalLM.from_config(config).to(torch.bfloat16).save_pretrained(path)
    quantization_config = {
        "quant_method": "compressed-tensors",
        "format": fmt,
        "config_groups": {"experts": {"targets": ["Linear"], "weights": weights, "format": fmt}},
    }
    compressor, scheme, _, _ = onfly._read_compression(
        SimpleNamespace(quantization_config=quantization_config)
    )
    state = load_file(str(path / "model.safetensors"))
    gen = torch.Generator().manual_seed(0)
    dense = {}
    for key in sorted(k for k in list(state) if ".experts." in k):  # ...experts.<e>.w1.weight
        rows, cols = state.pop(key).shape
        dense[key] = values[torch.randint(0, len(values), (rows, cols), generator=gen)]
        scales = torch.full((rows, cols // 32), scale)
        stored = compressor.compress({"weight": dense[key], "weight_scale": scales}, scheme)
        for name, tensor in stored.items():  # floats as bf16, as the Kimi checkpoints store them
            state[key.removesuffix("weight") + name] = (
                tensor.bfloat16() if tensor.is_floating_point() else tensor
            )
    save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})
    config_json = json.loads((path / "config.json").read_text())
    config_json["quantization_config"] = quantization_config
    (path / "config.json").write_text(json.dumps(config_json))
    return AutoConfig.from_pretrained(path), dense


@pytest.mark.usefixtures("restore_loader_hooks")
@pytest.mark.parametrize("fmt", sorted(_FORMATS))
def test_a_compressed_checkpoint_loads_exactly_and_saves_under_its_own_names(tmp_path, fmt):
    """A real ``from_pretrained`` through the installed decompressor: the fused experts hold the
    checkpoint's weights exactly. The converters the load kept reverse as the original fusion, so a
    save writes the experts back as the checkpoint's own ``w1``/``w3``/``w2`` weights, decompressed,
    and the export names its expanded experts after them."""
    pytest.importorskip(
        "compressed_tensors", reason="needs compressed-tensors (CI's unit session installs it)"
    )
    from modelopt.torch.export.quant_aware_conversion import build_reverse_name_mapper

    config, dense = _compressed_mixtral(tmp_path, fmt)
    onfly.install_onfly_decompress(config)
    del config.quantization_config  # as the expert-parallel load does: no quantizer competes

    model = AutoModelForCausalLM.from_pretrained(tmp_path, config=config, dtype=torch.bfloat16)

    experts, stored = model.model.layers[0].mlp.experts, "model.layers.0.block_sparse_moe.experts"
    for e in range(2):
        gate_up = torch.cat([dense[f"{stored}.{e}.w1.weight"], dense[f"{stored}.{e}.w3.weight"]])
        assert torch.equal(experts.gate_up_proj[e], gate_up.bfloat16())
        assert torch.equal(experts.down_proj[e], dense[f"{stored}.{e}.w2.weight"].bfloat16())
    saved = cml.revert_weight_conversion(model, model.state_dict())
    for key, weight in dense.items():
        assert torch.equal(saved[key], weight.bfloat16())
    export_name = build_reverse_name_mapper(model, tensor_keys=True)
    assert [
        export_name(f"model.layers.0.mlp.experts.1.{part}.weight")
        for part in ("gate_proj", "up_proj", "down_proj")
    ] == [f"{stored}.1.{hub}.weight" for hub in ("w1", "w3", "w2")]


@pytest.mark.usefixtures("restore_loader_hooks")
def test_a_load_the_decompressor_did_not_mark_casts_as_before(tmp_path):
    """The hooks are process-wide, but only a decompressing load keeps an integer source integer:
    a later load of a model no config of which was marked casts it to its float parameter."""
    pytest.importorskip(
        "compressed_tensors", reason="needs compressed-tensors (CI's unit session installs it)"
    )
    onfly.install_onfly_decompress(
        SimpleNamespace(quantization_config=_pack_quantized(experts=_int4(True, ["Linear"])))
    )
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(tmp_path)
    state = load_file(str(tmp_path / "model.safetensors"))
    state["model.norm.weight"] = torch.full_like(state["model.norm.weight"], 3, dtype=torch.int8)
    save_file(state, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})

    model = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)

    assert model.model.norm.weight.dtype == torch.bfloat16
    assert torch.equal(model.model.norm.weight, torch.full_like(model.model.norm.weight, 3))
