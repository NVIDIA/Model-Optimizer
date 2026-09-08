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

"""Per-layer export must match whole-model export, and refuse what it cannot match."""

import contextlib
import copy
import json
from unittest.mock import patch

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama, get_tiny_qwen3_moe
from safetensors.torch import load_file

import modelopt.torch.quantization as mtq
from modelopt.torch.export.layerwise_export import LayerwiseExporter, layer_shard_name
from modelopt.torch.export.unified_export_hf import export_hf_checkpoint

NUM_LAYERS = 4
CALIB_BATCHES = [torch.randint(0, 32, (1, 16)) for _ in range(2)]


def _calib(model):
    for batch in CALIB_BATCHES:
        model(batch.cuda())


def _build_model():
    torch.manual_seed(0)
    model = get_tiny_llama(num_hidden_layers=NUM_LAYERS).cuda().eval()
    # get_tiny_llama leaves this unset, but export reads it to detect multimodal models.
    model.config.architectures = ["LlamaForCausalLM"]
    return model


def _layerwise_cfg(export_dir, checkpoint_dir, base=None):
    cfg = copy.deepcopy(base or mtq.FP8_DEFAULT_CFG)
    cfg["algorithm"] = {
        "method": "max",
        "layerwise": {
            "enable": True,
            "export_dir": str(export_dir),
            "checkpoint_dir": str(checkpoint_dir),
            # max is amax-only, so the layer weights the shard captured stay valid.
            "calib_mutates_weights": False,
        },
    }
    return cfg


def _load_checkpoint(export_dir):
    index = export_dir / "model.safetensors.index.json"
    shards = (
        set(json.loads(index.read_text())["weight_map"].values())
        if index.exists()
        else ["model.safetensors"]
    )
    tensors = {}
    for shard in shards:
        tensors.update(load_file(str(export_dir / shard)))
    return tensors


def _assert_same_checkpoint(expected, actual):
    assert set(expected) == set(actual), (
        f"key mismatch: missing={sorted(set(expected) - set(actual))}, "
        f"extra={sorted(set(actual) - set(expected))}"
    )
    for key, want in expected.items():
        got = actual[key]
        assert got.dtype == want.dtype and got.shape == want.shape, f"{key}: dtype/shape differs"
        assert torch.equal(got.float(), want.float()), f"{key}: values differ"


def _assert_same_quant_config(baseline_dir, export_dir):
    """The metadata a loader reads, which tensor equality never covers.

    ``get_quant_config`` reports on the quantizer modules, and per-layer export replaces
    them as it goes, so a config read too late describes an unquantized model while the
    weights are packed.
    """
    for name, key in (
        ("hf_quant_config.json", "quantization"),
        ("config.json", "quantization_config"),
    ):
        want, got = baseline_dir / name, export_dir / name
        assert want.is_file() == got.is_file(), (
            f"{name}: present in baseline={want.is_file()} but exported={got.is_file()}"
        )
        if not want.is_file():
            continue
        expected = json.loads(want.read_text()).get(key)
        actual = json.loads(got.read_text()).get(key)
        assert actual == expected, (
            f"{name}[{key}] differs:\n  baseline={expected}\n  fused={actual}"
        )


def _fp8_cfg():
    return copy.deepcopy(mtq.FP8_DEFAULT_CFG)


def _kv_cache_cfg():
    return mtq.update_quant_cfg_with_kv_cache_quant(
        copy.deepcopy(mtq.FP8_DEFAULT_CFG), copy.deepcopy(mtq.FP8_KV_CFG["quant_cfg"])
    )


def _nvfp4_cfg():
    """NVFP4 with o_proj left unquantized.

    Layerwise calibration leaves ``self_attn.o_proj``'s input amax at 0 on every layer but
    the last, so a full-NVFP4 model cannot be exported by *any* path -- a pre-existing bug
    unrelated to per-layer export. The shipped NVFP4 layerwise recipes are experts-only and
    never quantize o_proj, which is why it has gone unnoticed. Excluding it here keeps this
    test on the behaviour it is meant to cover: q/k/v and gate/up scale fusion.
    """
    cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    cfg["quant_cfg"].append({"quantizer_name": "*o_proj*", "enable": False})
    return cfg


def _mixed_fp8_nvfp4_cfg():
    """FP8 attention, NVFP4 MLP -- a layer whose format depends on where you look.

    ``get_quantization_format`` returns the first format found, so gating fusion on it
    reports fp8 here and silently skips fusing the NVFP4 groups. o_proj stays unquantized
    for the reason in :func:`_nvfp4_cfg`.
    """
    nvfp4 = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    numerics = next(
        e["cfg"] for e in nvfp4["quant_cfg"] if e.get("quantizer_name") == "*weight_quantizer"
    )
    fp8 = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    fp8_numerics = next(
        e["cfg"] for e in fp8["quant_cfg"] if e.get("quantizer_name") == "*weight_quantizer"
    )
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*self_attn*weight_quantizer", "cfg": copy.deepcopy(fp8_numerics)},
            {"quantizer_name": "*self_attn*input_quantizer", "cfg": copy.deepcopy(fp8_numerics)},
            {"quantizer_name": "*mlp*weight_quantizer", "cfg": copy.deepcopy(numerics)},
            {"quantizer_name": "*mlp*input_quantizer", "cfg": copy.deepcopy(numerics)},
            {"quantizer_name": "*o_proj*", "enable": False},
        ]
    }


def _int4_awq_cfg():
    return copy.deepcopy(mtq.INT4_AWQ_CFG)


def _nvfp4_awq_cfg():
    cfg = copy.deepcopy(mtq.NVFP4_AWQ_LITE_CFG)
    cfg["quant_cfg"].append({"quantizer_name": "*o_proj*", "enable": False})
    return cfg


@pytest.fixture(scope="module")
def baseline_checkpoint(tmp_path_factory):
    """A normal layerwise calibration followed by a separate whole-model export."""
    export_dir = tmp_path_factory.mktemp("baseline")
    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    cfg["algorithm"] = {"method": "max", "layerwise": {"enable": True}}
    model = mtq.quantize(_build_model(), cfg, _calib)
    export_hf_checkpoint(model, export_dir=export_dir)
    return _load_checkpoint(export_dir)


@contextlib.contextmanager
def _dies_at_layer(layer_idx: int):
    """Lose the session mid-model, the way a GPU timeout would."""
    real = LayerwiseExporter.export_layer

    def die(self, idx, *args, **kwargs):
        if idx == layer_idx:
            raise RuntimeError("interrupted")
        return real(self, idx, *args, **kwargs)

    with patch.object(LayerwiseExporter, "export_layer", die):
        yield


@pytest.mark.parametrize("interrupt_at", [None, 2], ids=["fresh", "resumed"])
@pytest.mark.parametrize(
    ("make_cfg", "layerwise_extra", "expected_key_suffix"),
    [
        # NVFP4 fuses q/k/v and gate/up scales, so per-layer rediscovery has to match.
        pytest.param(_nvfp4_cfg, {}, ("weight_scale_2",), id="nvfp4"),
        # The probe runs the layer directly, so the capture must leave it in "original".
        pytest.param(
            _nvfp4_cfg,
            {"get_qdq_activations_from_prev_layer": True},
            ("weight_scale_2",),
            id="nvfp4_qdq_from_prev_layer",
        ),
        # A layer holding two formats must still fuse the one that needs it.
        pytest.param(_mixed_fp8_nvfp4_cfg, {}, None, id="mixed_fp8_nvfp4"),
        # KV scales only survive if the format is read off the whole quant config: from
        # the root module alone it is None, and the per-tensor pass then asserts on the
        # first *_bmm_quantizer._amax it sees.
        pytest.param(_kv_cache_cfg, {}, ("k_scale", "v_scale"), id="kv_cache"),
        pytest.param(_fp8_cfg, {}, None, id="fp8"),
    ],
)
def test_export_matches_whole_model_export(
    tmp_path, make_cfg, layerwise_extra, expected_key_suffix, interrupt_at
):
    """Exporting per layer during calibration must yield the same checkpoint.

    ``interrupt_at`` crosses every config with a lost-session resume, since the two
    interact: a resumed run re-enters the export path part-way through the model.
    """
    baseline_dir = tmp_path / "baseline"
    base = make_cfg()
    base["algorithm"] = {"method": "max", "layerwise": {"enable": True, **layerwise_extra}}
    export_hf_checkpoint(mtq.quantize(_build_model(), base, _calib), export_dir=baseline_dir)

    export_dir = tmp_path / "fused"
    cfg = _layerwise_cfg(export_dir, tmp_path / "ckpt", base=make_cfg())
    cfg["algorithm"]["layerwise"].update(layerwise_extra)
    if interrupt_at is not None:
        with _dies_at_layer(interrupt_at), pytest.raises(RuntimeError, match="interrupted"):
            mtq.quantize(_build_model(), cfg, _calib)
    mtq.quantize(_build_model(), cfg, _calib)

    exported = _load_checkpoint(export_dir)
    if expected_key_suffix:
        assert any(k.endswith(expected_key_suffix) for k in exported), (
            f"no {expected_key_suffix} keys in the exported checkpoint"
        )
    _assert_same_checkpoint(_load_checkpoint(baseline_dir), exported)
    _assert_same_quant_config(baseline_dir, export_dir)
    # The directory must be loadable on its own, with no follow-up export call.
    for artifact in ("config.json", "hf_quant_config.json", "model.safetensors.index.json"):
        assert (export_dir / artifact).is_file(), f"{artifact} missing"


def test_index_resolves_every_key_to_the_shard_holding_it(tmp_path):
    """A loader resolves keys through the index; tensor equality never exercises that.

    A weight_map entry naming the wrong shard compares equal to a whole-model export and
    still fails in vLLM or transformers.
    """
    export_dir = tmp_path / "fused"
    mtq.quantize(_build_model(), _layerwise_cfg(export_dir, tmp_path / "ckpt"), _calib)

    weight_map = json.loads((export_dir / "model.safetensors.index.json").read_text())["weight_map"]
    on_disk = {}
    for shard in sorted(set(weight_map.values())):
        assert (export_dir / shard).is_file(), f"index names a missing shard {shard}"
        on_disk.update(dict.fromkeys(load_file(export_dir / shard), shard))

    assert set(weight_map) == set(on_disk), "index and shards disagree on which keys exist"
    assert all(on_disk[k] == v for k, v in weight_map.items()), "key routed to the wrong shard"


def test_layerwise_export_replaces_resume_artifacts(tmp_path):
    """The shards are the resume artifact, so per-layer weight copies are not written."""
    checkpoint_dir = tmp_path / "ckpt"
    mtq.quantize(_build_model(), _layerwise_cfg(tmp_path / "fused", checkpoint_dir), _calib)

    assert not list(checkpoint_dir.rglob("weights.pt"))
    assert not list(checkpoint_dir.rglob("quantizer_buffers.pt"))
    # output_meta is not reconstructible from exported weights, so it stays.
    assert list(checkpoint_dir.rglob("output_meta.pt"))
    # The cached activations are the bulk of the resume dir, and only the committed
    # boundary's are resumable -- keeping one per layer would dwarf the checkpoint.
    assert not list(checkpoint_dir.rglob("next_inputs.pt")), (
        "a completed run has nothing to resume from, so no activation cache should remain"
    )


def test_resume_skips_exported_layers(tmp_path, baseline_checkpoint):
    """A run resuming mid-model must still produce the full, correct checkpoint."""
    export_dir = tmp_path / "fused"
    checkpoint_dir = tmp_path / "ckpt"
    # Die partway, the way a lost GPU session would: only the committed boundary is
    # resumable, so rewinding a *finished* run's manifest would not reproduce this state.
    with _dies_at_layer(2), pytest.raises(RuntimeError, match="interrupted"):
        mtq.quantize(_build_model(), _layerwise_cfg(export_dir, checkpoint_dir), _calib)

    assert (export_dir / layer_shard_name(1)).is_file(), "layer 1 was never committed"
    assert not (export_dir / layer_shard_name(2)).exists(), "layer 2 should not have landed"

    # Shards 0..1 are on disk and must be reused rather than recalculated.
    mtq.quantize(_build_model(), _layerwise_cfg(export_dir, checkpoint_dir), _calib)

    _assert_same_checkpoint(baseline_checkpoint, _load_checkpoint(export_dir))


def test_resume_without_matching_shards_fails_fast(tmp_path):
    """Mismatched checkpoint/export dirs must fail before recalibrating, not at the end."""
    checkpoint_dir = tmp_path / "ckpt"
    mtq.quantize(_build_model(), _layerwise_cfg(tmp_path / "fused", checkpoint_dir), _calib)

    manifest_path = checkpoint_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["last_completed_layer"] = 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="shards are missing"):
        mtq.quantize(
            _build_model(), _layerwise_cfg(tmp_path / "empty_export", checkpoint_dir), _calib
        )


def test_complete_manifest_finalizes_without_recalibrating(tmp_path, baseline_checkpoint):
    """A crash between the last shard and finalize must cost only the finalize.

    detect_resume_point returns None once the manifest is complete, so start_layer falls
    back to 0 and every layer would be recalculated and overwritten.
    """
    export_dir = tmp_path / "fused"
    checkpoint_dir = tmp_path / "ckpt"
    mtq.quantize(_build_model(), _layerwise_cfg(export_dir, checkpoint_dir), _calib)

    # What a crash after the final ckpt.save looks like: every shard and a complete
    # manifest on disk, but no tail, index or config yet.
    (export_dir / "model-tail.safetensors").unlink()
    (export_dir / "model.safetensors.index.json").unlink()
    layer_mtimes = {p.name: p.stat().st_mtime for p in export_dir.glob("model-layer-*.safetensors")}
    assert layer_mtimes

    mtq.quantize(_build_model(), _layerwise_cfg(export_dir, checkpoint_dir), _calib)

    # The layer shards must be reused verbatim, not rewritten.
    for name, mtime in layer_mtimes.items():
        assert (export_dir / name).stat().st_mtime == mtime, f"{name} was rewritten"
    _assert_same_checkpoint(baseline_checkpoint, _load_checkpoint(export_dir))


@pytest.mark.parametrize("damage", ["deleted", "no_resume_point"])
def test_shards_without_resume_record_refuse(tmp_path, damage):
    """A lost resume record must not silently overwrite finished shards.

    Either way start_layer falls back to 0, so assert_shards_present checks nothing.
    """
    export_dir = tmp_path / "fused"
    checkpoint_dir = tmp_path / "ckpt"
    mtq.quantize(_build_model(), _layerwise_cfg(export_dir, checkpoint_dir), _calib)

    manifest = checkpoint_dir / "manifest.json"
    if damage == "deleted":
        manifest.unlink()
    else:
        record = json.loads(manifest.read_text())
        record.pop("last_completed_layer")
        manifest.write_text(json.dumps(record))

    with pytest.raises(RuntimeError, match="no usable resume record"):
        mtq.quantize(_build_model(), _layerwise_cfg(export_dir, checkpoint_dir), _calib)


def test_export_without_checkpoint_dir_may_overwrite(tmp_path):
    """Used directly, without checkpoint_dir, there is no resume to lose.

    ``hf_ptq`` derives one from ``--export_path`` so its users get resume by default; a
    library caller that omits it is opting out, and re-exporting from scratch is then the
    documented behaviour rather than an error.
    """
    export_dir = tmp_path / "fused"
    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    cfg["algorithm"] = {
        "method": "max",
        "layerwise": {"enable": True, "export_dir": str(export_dir)},
    }
    mtq.quantize(_build_model(), cfg, _calib)
    mtq.quantize(_build_model(), copy.deepcopy(cfg), _calib)  # must not raise


def _build_moe_model():
    torch.manual_seed(0)
    model = get_tiny_qwen3_moe(num_experts=16, num_experts_per_tok=1).cuda().eval()
    model.config.architectures = ["Qwen3MoeForCausalLM"]
    return model


@pytest.mark.parametrize(
    "make_cfg",
    [
        pytest.param(_fp8_cfg, id="quantized"),
        # `enable: False` leaves the quantizer module in place, so this refuses too --
        # excluding lm_head from quantization is not a way past the gate.
        pytest.param(
            lambda: {**_fp8_cfg(), "quant_cfg": [{"quantizer_name": "*", "enable": False}]},
            id="quantizers_disabled",
        ),
    ],
)
def test_tied_embeddings_are_refused(tmp_path, make_cfg):
    """Every tie_word_embeddings model is refused, which is why export needs no tie handling."""
    torch.manual_seed(0)
    model = get_tiny_llama(num_hidden_layers=NUM_LAYERS, tie_word_embeddings=True).cuda().eval()
    model.config.architectures = ["LlamaForCausalLM"]

    cfg = _layerwise_cfg(tmp_path / "fused", tmp_path / "ckpt", base=make_cfg())
    with pytest.raises(NotImplementedError, match="weight-tied quantized modules"):
        mtq.quantize(model, cfg, _calib)


def test_moe_export_matches(tmp_path):
    """MoE layers take a different path: fused expert inputs and gate/up amax sync."""
    baseline_dir = tmp_path / "baseline"
    base = _nvfp4_cfg()
    base["algorithm"] = {"method": "max", "layerwise": {"enable": True}}
    export_hf_checkpoint(mtq.quantize(_build_moe_model(), base, _calib), export_dir=baseline_dir)

    export_dir = tmp_path / "fused"
    mtq.quantize(
        _build_moe_model(), _layerwise_cfg(export_dir, tmp_path / "ckpt", base=_nvfp4_cfg()), _calib
    )

    _assert_same_checkpoint(_load_checkpoint(baseline_dir), _load_checkpoint(export_dir))
    _assert_same_quant_config(baseline_dir, export_dir)


def test_export_consumes_the_model_without_affecting_the_checkpoint(tmp_path):
    """Per-layer export converts each layer in place; the shard is written first.

    The layer is dead state by then -- the next layer's inputs were captured before the
    call, and resume skips finished layers in favour of their shards. What must not drift
    is the checkpoint, so this pins that the in-place conversion is invisible to it.
    """
    baseline_dir = tmp_path / "baseline"
    base = _nvfp4_cfg()
    base["algorithm"] = {"method": "max", "layerwise": {"enable": True}}
    export_hf_checkpoint(mtq.quantize(_build_model(), base, _calib), export_dir=baseline_dir)

    export_dir = tmp_path / "fused"
    model = mtq.quantize(
        _build_model(),
        _layerwise_cfg(export_dir, tmp_path / "ckpt", base=_nvfp4_cfg()),
        _calib,
    )

    _assert_same_checkpoint(_load_checkpoint(baseline_dir), _load_checkpoint(export_dir))
    # The returned model is in export form, which is why hf_ptq forces --skip_generate.
    packed = [
        n
        for n, m in model.named_modules()
        if hasattr(m, "weight_scale") or hasattr(m, "input_scale")
    ]
    assert packed, "expected the exported model to be left in export form"


@pytest.mark.parametrize(
    ("make_cfg", "method", "match"),
    [
        # Visible from the config: int4_awq is keyed on num_bits/SequentialQuantizer.
        pytest.param(_int4_awq_cfg, "max", "awq", id="int4_awq_from_config"),
        # Only visible afterwards: the NVFP4 discriminators (_pre_quant_scale,
        # svdquant_lora_a) are registered by the calibrator, so the constructor's gate
        # sees plain nvfp4 and the check has to run again on the first exported layer.
        pytest.param(_nvfp4_awq_cfg, "awq_lite", "nvfp4_awq", id="nvfp4_awq_after_calibration"),
    ],
)
def test_awq_is_refused(tmp_path, make_cfg, method, match):
    """AWQ needs the pre-quant-scale steps, which are still whole-model."""
    cfg = _layerwise_cfg(tmp_path / "fused", tmp_path / "ckpt", base=make_cfg())
    cfg["algorithm"]["method"] = method
    cfg["algorithm"]["layerwise"]["calib_mutates_weights"] = method != "max"

    with pytest.raises(NotImplementedError, match=match):
        mtq.quantize(_build_model(), cfg, _calib)
