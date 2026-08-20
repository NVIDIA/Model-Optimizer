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


@pytest.fixture(scope="module")
def baseline_checkpoint(tmp_path_factory):
    """A normal layerwise calibration followed by a separate whole-model export."""
    export_dir = tmp_path_factory.mktemp("baseline")
    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    cfg["algorithm"] = {"method": "max", "layerwise": {"enable": True}}
    model = mtq.quantize(_build_model(), cfg, _calib)
    export_hf_checkpoint(model, export_dir=export_dir)
    return _load_checkpoint(export_dir)


def test_layerwise_export_matches_whole_model_export(tmp_path, baseline_checkpoint):
    """Exporting per layer during calibration must yield the same checkpoint."""
    export_dir = tmp_path / "fused"
    mtq.quantize(_build_model(), _layerwise_cfg(export_dir, tmp_path / "ckpt"), _calib)

    _assert_same_checkpoint(baseline_checkpoint, _load_checkpoint(export_dir))
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


def test_shards_from_a_different_config_refuse(tmp_path):
    """Same layer count, different quantization: the shards must not be reused.

    The resume manifest records no model or quantization identity and
    assert_shards_present only checks that files exist, so without a binding one run's
    manifest could finalize another run's shards into a valid-looking, wrong checkpoint.
    """
    export_dir = tmp_path / "fused"
    mtq.quantize(_build_model(), _layerwise_cfg(export_dir, tmp_path / "ckpt_fp8"), _calib)

    # Same model and layer count, NVFP4 instead of FP8, pointed at the same shards.
    nvfp4 = _layerwise_cfg(export_dir, tmp_path / "ckpt_nvfp4", base=_nvfp4_cfg())
    with pytest.raises(RuntimeError, match="different run"):
        mtq.quantize(_build_model(), nvfp4, _calib)


def test_shards_from_a_different_module_selection_refuse(tmp_path):
    """Same format and layer count, different modules quantized: still a different run.

    Format names alone cannot tell these apart -- both are FP8 -- so the identity has to
    carry the per-module contract from the quant config.
    """
    export_dir = tmp_path / "fused"
    mtq.quantize(_build_model(), _layerwise_cfg(export_dir, tmp_path / "ckpt_a"), _calib)

    narrowed = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    narrowed["quant_cfg"].append({"quantizer_name": "*mlp*", "enable": False})
    with pytest.raises(RuntimeError, match="different run"):
        mtq.quantize(
            _build_model(), _layerwise_cfg(export_dir, tmp_path / "ckpt_b", base=narrowed), _calib
        )


def test_kv_cache_quantized_export_matches(tmp_path):
    """KV-cache scales must survive: the format has to be read off the whole quant config.

    Deriving it from the root module alone yields None, which makes the per-tensor pass
    assert on the first ``*_bmm_quantizer._amax`` it sees.
    """
    kv_cfg = mtq.update_quant_cfg_with_kv_cache_quant(
        copy.deepcopy(mtq.FP8_DEFAULT_CFG), copy.deepcopy(mtq.FP8_KV_CFG["quant_cfg"])
    )

    baseline_dir = tmp_path / "baseline"
    base = copy.deepcopy(kv_cfg)
    base["algorithm"] = {"method": "max", "layerwise": {"enable": True}}
    export_hf_checkpoint(mtq.quantize(_build_model(), base, _calib), export_dir=baseline_dir)

    export_dir = tmp_path / "fused"
    mtq.quantize(_build_model(), _layerwise_cfg(export_dir, tmp_path / "ckpt", base=kv_cfg), _calib)

    exported = _load_checkpoint(export_dir)
    assert any(k.endswith(("k_scale", "v_scale")) for k in exported), (
        "no KV cache scales were exported"
    )
    _assert_same_checkpoint(_load_checkpoint(baseline_dir), exported)


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


def test_nvfp4_export_matches(tmp_path):
    """NVFP4 fuses q/k/v and gate/up scales; per-layer rediscovery must match."""
    baseline_dir = tmp_path / "baseline"
    base = _nvfp4_cfg()
    base["algorithm"] = {"method": "max", "layerwise": {"enable": True}}
    export_hf_checkpoint(mtq.quantize(_build_model(), base, _calib), export_dir=baseline_dir)

    export_dir = tmp_path / "fused"
    mtq.quantize(
        _build_model(), _layerwise_cfg(export_dir, tmp_path / "ckpt", base=_nvfp4_cfg()), _calib
    )

    exported = _load_checkpoint(export_dir)
    assert any(k.endswith("weight_scale_2") for k in exported), "no NVFP4 global scales exported"
    _assert_same_checkpoint(_load_checkpoint(baseline_dir), exported)


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


def test_mixed_format_export_matches(tmp_path):
    """A layer holding two formats must still fuse the one that needs it."""
    baseline_dir = tmp_path / "baseline"
    base = _mixed_fp8_nvfp4_cfg()
    base["algorithm"] = {"method": "max", "layerwise": {"enable": True}}
    export_hf_checkpoint(mtq.quantize(_build_model(), base, _calib), export_dir=baseline_dir)

    export_dir = tmp_path / "fused"
    mtq.quantize(
        _build_model(),
        _layerwise_cfg(export_dir, tmp_path / "ckpt", base=_mixed_fp8_nvfp4_cfg()),
        _calib,
    )
    _assert_same_checkpoint(_load_checkpoint(baseline_dir), _load_checkpoint(export_dir))


def test_awq_is_refused(tmp_path):
    """AWQ needs the pre-quant-scale steps, which are still whole-model."""
    cfg = _layerwise_cfg(tmp_path / "fused", tmp_path / "ckpt", base=mtq.INT4_AWQ_CFG)
    with pytest.raises(NotImplementedError, match="awq"):
        mtq.quantize(_build_model(), cfg, _calib)
