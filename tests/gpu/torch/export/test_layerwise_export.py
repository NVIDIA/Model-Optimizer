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

import copy
import json
import shutil

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama
from safetensors.torch import load_file

import modelopt.torch.quantization as mtq
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


def _fp8_cfg():
    return copy.deepcopy(mtq.FP8_DEFAULT_CFG)


def _narrowed_fp8_cfg():
    """FP8 with the MLP left unquantized: same format, different module selection."""
    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    cfg["quant_cfg"].append({"quantizer_name": "*mlp*", "enable": False})
    return cfg


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
    tmp_path, make_cfg, layerwise_extra, expected_key_suffix
):
    """Exporting per layer during calibration must yield the same checkpoint."""
    baseline_dir = tmp_path / "baseline"
    base = make_cfg()
    base["algorithm"] = {"method": "max", "layerwise": {"enable": True, **layerwise_extra}}
    export_hf_checkpoint(mtq.quantize(_build_model(), base, _calib), export_dir=baseline_dir)

    export_dir = tmp_path / "fused"
    cfg = _layerwise_cfg(export_dir, tmp_path / "ckpt", base=make_cfg())
    cfg["algorithm"]["layerwise"].update(layerwise_extra)
    mtq.quantize(_build_model(), cfg, _calib)

    exported = _load_checkpoint(export_dir)
    if expected_key_suffix:
        assert any(k.endswith(expected_key_suffix) for k in exported), (
            f"no {expected_key_suffix} keys in the exported checkpoint"
        )
    _assert_same_checkpoint(_load_checkpoint(baseline_dir), exported)
    # The directory must be loadable on its own, with no follow-up export call.
    for artifact in ("config.json", "hf_quant_config.json", "model.safetensors.index.json"):
        assert (export_dir / artifact).is_file(), f"{artifact} missing"


def test_layerwise_export_replaces_resume_artifacts(tmp_path):
    """The shards are the resume artifact, so per-layer weight copies are not written."""
    checkpoint_dir = tmp_path / "ckpt"
    mtq.quantize(_build_model(), _layerwise_cfg(tmp_path / "fused", checkpoint_dir), _calib)

    assert not list(checkpoint_dir.rglob("weights.pt"))
    assert not list(checkpoint_dir.rglob("quantizer_buffers.pt"))
    # next_inputs and output_meta are not reconstructible from exported weights, so they stay.
    assert list(checkpoint_dir.rglob("output_meta.pt"))


def test_resume_skips_exported_layers(tmp_path, baseline_checkpoint):
    """A run resuming mid-model must still produce the full, correct checkpoint."""
    export_dir = tmp_path / "fused"
    checkpoint_dir = tmp_path / "ckpt"
    mtq.quantize(_build_model(), _layerwise_cfg(export_dir, checkpoint_dir), _calib)

    # Rewind the manifest so the next run believes only layers 0..1 finished; their shards
    # are on disk and must be reused rather than recalculated.
    manifest_path = checkpoint_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["last_completed_layer"] = 1
    manifest_path.write_text(json.dumps(manifest))

    resumed_dir = tmp_path / "resumed"
    shutil.copytree(export_dir, resumed_dir)
    mtq.quantize(_build_model(), _layerwise_cfg(resumed_dir, checkpoint_dir), _calib)

    _assert_same_checkpoint(baseline_checkpoint, _load_checkpoint(resumed_dir))


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


def test_shards_without_manifest_refuse(tmp_path):
    """A lost resume record must not silently overwrite finished shards."""
    export_dir = tmp_path / "fused"
    checkpoint_dir = tmp_path / "ckpt"
    mtq.quantize(_build_model(), _layerwise_cfg(export_dir, checkpoint_dir), _calib)

    # What an ephemeral checkpoint_dir looks like on the next run: shards survive, the
    # manifest does not. start_layer is then 0, so assert_shards_present checks nothing.
    (checkpoint_dir / "manifest.json").unlink()

    with pytest.raises(RuntimeError, match="no manifest"):
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


@pytest.mark.parametrize(
    "make_second_cfg",
    [
        # Same layer count, NVFP4 instead of FP8.
        pytest.param(_nvfp4_cfg, id="different_format"),
        # Same format and layer count: only the per-module contract in the quant config
        # tells these apart, so format names alone are not enough to bind a run.
        pytest.param(_narrowed_fp8_cfg, id="different_module_selection"),
    ],
)
def test_shards_from_a_different_run_refuse(tmp_path, make_second_cfg):
    """One run's manifest must not finalize another run's shards.

    The resume manifest records no identity of its own and assert_shards_present only
    checks that files exist, so without a binding this would produce a valid-looking,
    wrong checkpoint.
    """
    export_dir = tmp_path / "fused"
    mtq.quantize(_build_model(), _layerwise_cfg(export_dir, tmp_path / "ckpt_a"), _calib)

    second = _layerwise_cfg(export_dir, tmp_path / "ckpt_b", base=make_second_cfg())
    with pytest.raises(RuntimeError, match="different run"):
        mtq.quantize(_build_model(), second, _calib)


def test_identity_without_shards_does_not_block_a_rerun(tmp_path):
    """A run that dies before layer 0 leaves an identity file guarding nothing."""
    export_dir = tmp_path / "fused"
    export_dir.mkdir()
    (export_dir / ".layerwise_export.json").write_text('{"model_class": "Other"}')

    mtq.quantize(_build_model(), _layerwise_cfg(export_dir, tmp_path / "ckpt"), _calib)

    assert _load_checkpoint(export_dir), "rerun produced no checkpoint"


def test_export_does_not_mutate_the_model(tmp_path):
    """Exporting a layer must leave the model exactly as calibration left it.

    Scale fusion unifies a group's amax with an in-place ``_amax.data.copy_()``, which
    restoring the buffer *dict* does not undo -- only restoring buffer contents does. FP8
    cannot catch this: it never fuses.
    """
    plain = _nvfp4_cfg()
    plain["algorithm"] = {"method": "max", "layerwise": {"enable": True}}
    expected = mtq.quantize(_build_model(), plain, _calib).state_dict()
    expected = {k: v.clone() for k, v in expected.items()}

    exported = mtq.quantize(
        _build_model(),
        _layerwise_cfg(tmp_path / "fused", tmp_path / "ckpt", base=_nvfp4_cfg()),
        _calib,
    ).state_dict()

    drifted = [k for k in expected if k in exported and not torch.equal(expected[k], exported[k])]
    assert not drifted, f"per-layer export mutated the model: {drifted[:5]}"


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
