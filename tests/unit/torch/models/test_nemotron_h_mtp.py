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

import copy
import fnmatch
import json
import shutil
import threading
import warnings
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("transformers")

from huggingface_hub import constants as hub_constants
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel, conversion_mapping

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_utils import get_activation_scaling_factor, postprocess_state_dict
from modelopt.torch.export.unified_export_hf import _process_quantized_modules
from modelopt.torch.models import hf
from modelopt.torch.models.nemotron_h import mtp as adapter
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.conversion import set_quantizer_by_cfg_context
from modelopt.torch.quantization.model_calib import (
    _needs_activation_forward_for_max_calib,
    awq,
    local_hessian_calibrate,
    max_calibrate,
)
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils.calib_utils import GPTQHelper
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector


@pytest.mark.parametrize(
    ("config_key", "declared", "invalid"),
    [
        (None, False, False),
        (None, True, False),
        ("llm_config", True, False),
        ("text_config", True, True),
    ],
)
def test_missing_mtp_weights_warn_without_constructing(
    tmp_path, monkeypatch, config_key, declared, invalid
):
    config = {"num_nextn_predict_layers": int(declared)}
    if config_key:
        config = {config_key: config}
    (tmp_path / "config.json").write_text(json.dumps(config))
    if invalid:
        (tmp_path / "model.safetensors").write_bytes(b"not safetensors")
    loader = Mock(side_effect=AssertionError("No remote code is needed without MTP weights"))
    monkeypatch.setattr(adapter, "get_class_from_dynamic_module", loader)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with hf.prepare_model_for_loading("nemotron_h", tmp_path, False):
            pass
    assert len(caught) == int(declared)
    if declared:
        assert "config declares MTP but no MTP tensors were found" in str(caught[0].message)
    loader.assert_not_called()


@pytest.mark.parametrize(
    "keys",
    [
        ["mtp.layers.0.eh_proj.weight"],
        ["mtp.layers.0.eh_proj.weight", "mtp.layers.3.final_layernorm.weight"],
        ["mtp.layers.0.eh_proj.weight", "language_model.mtp.layers.1.final_layernorm.weight"],
        ["decoder.mtp.layers.0.eh_proj.weight", "decoder.mtp.layers.1.final_layernorm.weight"],
    ],
)
def test_unsupported_mtp_layout_rejects_silent_passthrough(tmp_path, keys):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(keys, "model-00001.safetensors")})
    )
    with (
        pytest.raises(ValueError, match="Unsupported Nemotron-H MTP tensor layout"),
        adapter.prepare_for_loading(tmp_path, False),
    ):
        pytest.fail("Unsupported MTP tensors would be silently passed through")


@pytest.fixture(params=[("plain", 2), ("wrapped", 4)])
def tiny_checkpoint(tmp_path, monkeypatch, request):
    native = pytest.importorskip("transformers.models.nemotron_h.modeling_nemotron_h")
    layout, num_blocks = request.param
    config = native.NemotronHConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=32,
        layers_block_type=["attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        n_routed_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32,
        use_mamba_kernels=False,
        attn_implementation="eager",
        num_nextn_predict_layers=num_blocks // 2,
        mtp_layers_block_type=["attention", "moe"] * (num_blocks // 2),
    )

    class TinyOmni(PreTrainedModel):
        config_class = native.NemotronHConfig
        base_model_prefix = "language_model"

        def __init__(self, config):
            super().__init__(config)
            self.language_model = native.NemotronHForCausalLM(config)
            self.post_init()

        def forward(self, *args, **kwargs):
            return self.language_model(*args, **kwargs)

    cls = TinyOmni if layout == "wrapped" else native.NemotronHForCausalLM
    model = cls(config).eval()
    language_model = getattr(model, "language_model", model)
    # Exercise checkpoint aliases without asking the native config parser to accept them.
    mtp_config = copy.deepcopy(config)
    attention_alias = "full_attention" if "attention" in native.MIXER_TYPES else "attention"
    mtp_config.mtp_layers_block_type = [attention_alias, "moe"] * (num_blocks // 2)
    language_model.mtp = adapter._NemotronHMTP(mtp_config)
    # Fused expert parameters are allocated with empty(); a real checkpoint supplies their values.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    # Write the source layout directly; older HF save_pretrained rewrites the language prefix.
    config.save_pretrained(tmp_path)
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    monkeypatch.setattr(adapter, "get_class_from_dynamic_module", Mock(return_value=cls))
    return cls, model, tmp_path


def test_remote_consent_and_constructor_cleanup(tiny_checkpoint):
    """Consent, nested construction, and failures preserve the original model class."""
    cls, source, checkpoint = tiny_checkpoint
    config = json.loads((checkpoint / "config.json").read_text())
    config["auto_map"] = {"AutoModelForCausalLM": "modeling_local.NemotronHWithMTP"}
    (checkpoint / "config.json").write_text(json.dumps(config))
    loader = adapter.get_class_from_dynamic_module
    original_init = cls.__init__
    if hasattr(source, "language_model"):
        with (
            pytest.raises(ValueError, match="trust_remote_code=True"),
            adapter.prepare_for_loading(checkpoint, False),
        ):
            pytest.fail("Remote MTP class was loaded without consent")
    else:
        with adapter.prepare_for_loading(checkpoint, False):
            assert hasattr(cls(source.config), "mtp")
    loader.assert_not_called()
    with adapter.prepare_for_loading(checkpoint, True):
        with adapter.prepare_for_loading(checkpoint, True):
            loaded = cls.from_pretrained(checkpoint)
        assert hasattr(getattr(loaded, "language_model", loaded), "mtp")
        if hasattr(source, "language_model"):
            assert not hasattr(type(source.language_model)(source.config), "mtp")
    loader.assert_called_with("modeling_local.NemotronHWithMTP", checkpoint)
    with (
        pytest.raises(RuntimeError, match="load failure"),
        adapter.prepare_for_loading(checkpoint, True),
    ):
        raise RuntimeError("load failure")
    assert cls.__init__ is original_init
    fresh = cls(source.config)
    assert not hasattr(getattr(fresh, "language_model", fresh), "mtp")


def test_overlapping_loads_take_turns(tiny_checkpoint, monkeypatch):
    """A load in another thread waits, so it neither reads nor restores this load's patches."""
    cls, source, checkpoint = tiny_checkpoint
    original_init = cls.__init__
    lock = adapter._LOAD_LOCK
    at_lock, entered = threading.Event(), threading.Event()
    errors = []

    class SignalingLock:
        # Marks the other load reaching the lock, so its wait is contention and not slow setup.
        def __enter__(self):
            if threading.current_thread() is other:
                at_lock.set()
            return lock.__enter__()

        def __exit__(self, *exc_info):
            return lock.__exit__(*exc_info)

    monkeypatch.setattr(adapter, "_LOAD_LOCK", SignalingLock())

    def other_load():
        try:
            with adapter.prepare_for_loading(checkpoint, True):
                entered.set()
                cls(source.config)
        except Exception as error:  # surfaced in the main thread
            errors.append(error)
        finally:
            at_lock.set()  # a load that skips the lock fails the check below, not a timeout

    other = threading.Thread(target=other_load, daemon=True)
    with adapter.prepare_for_loading(checkpoint, True):
        other.start()
        assert at_lock.wait(timeout=30)
        other.join(timeout=0.5)
        assert other.is_alive(), "a load in another thread ran inside this one"
        assert not entered.is_set()
        model = cls(source.config)
    other.join(timeout=30)
    assert not other.is_alive()
    assert not errors
    assert hasattr(getattr(model, "language_model", model), "mtp")
    assert cls.__init__ is original_init


def _hub_checkpoint(checkpoint, source, monkeypatch, sharded):
    """Populate a hermetic Hub cache so the lifecycle tests can use real Hub-ID resolution."""
    repo_id = "test/nemotron-h-mtp"
    cache = checkpoint / "hub"
    repo_cache = cache / "models--test--nemotron-h-mtp"
    revision = "a" * 40
    snapshot = repo_cache / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (repo_cache / "refs").mkdir()
    (repo_cache / "refs" / "main").write_text(revision)
    shutil.copyfile(checkpoint / "config.json", snapshot / "config.json")
    weight_file = "model-00001-of-00001.safetensors" if sharded else "model.safetensors"
    shutil.copyfile(checkpoint / "model.safetensors", snapshot / weight_file)
    if sharded:
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {}, "weight_map": dict.fromkeys(source.state_dict(), weight_file)}
            )
        )
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    return repo_id


@pytest.mark.parametrize(
    ("blocks", "error"), [(["attention"], "MTP block count"), (["unknown"], "MTP block type")]
)
def test_invalid_mtp_config_restores_constructor(tiny_checkpoint, blocks, error):
    cls, source, checkpoint = tiny_checkpoint
    config = copy.deepcopy(source.config)
    config.mtp_layers_block_type = blocks
    original_init = cls.__init__
    with (
        pytest.raises(ValueError, match=error),
        adapter.prepare_for_loading(checkpoint, True),
    ):
        cls(config)
    assert cls.__init__ is original_init


def test_mtp_loading_context_warns_if_constructor_is_bypassed(tiny_checkpoint):
    cls, _, checkpoint = tiny_checkpoint
    original_init = cls.__init__
    with (
        pytest.warns(UserWarning, match="MTP weights may remain unquantized"),
        adapter.prepare_for_loading(checkpoint, True),
    ):
        pass
    assert cls.__init__ is original_init


def test_mtp_later_attention_blocks_remain_causal(tiny_checkpoint):
    _, model, _ = tiny_checkpoint
    mtp = getattr(model, "language_model", model).mtp.eval()
    hidden = torch.randn(1, 4, model.config.hidden_size)
    embeddings = torch.randn_like(hidden)
    with torch.no_grad():
        expected = mtp(hidden, embeddings)
        hidden[:, -1] += 10
        embeddings[:, -1] += 10
        actual = mtp(hidden, embeddings)
    # MTP uses next-token embeddings, so only the last two positions may change. The tolerance
    # covers float noise (below 1e-7 over 400 RNG states); a real leak changes them by ~0.1.
    torch.testing.assert_close(actual[:, :-2], expected[:, :-2], rtol=1e-5, atol=1e-6)


def test_fused_weight_quantizers_need_no_activation_forward():
    mtp = torch.nn.Module()
    mtp.up_proj_weight_quantizers = torch.nn.ModuleList(
        [TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3)))]
    )
    assert not _needs_activation_forward_for_max_calib(mtp)


_PROJECTION = "layers.0.eh_proj"
_SHARED = "layers.1.mixer.shared_experts.up_proj"
_KV = "layers.0.mixer.[kv]_bmm_quantizer"


@pytest.mark.parametrize(
    ("location", "activation", "attributes", "needs_forward"),
    [
        ("local", f"{_PROJECTION}.input_quantizer", {}, True),
        ("local", f"{_SHARED}.input_quantizer", {}, True),
        ("local", _KV, {"num_bits": (4, 3)}, True),
        ("local", f"{_PROJECTION}.output_quantizer", {"num_bits": (4, 3)}, True),
        ("local", None, {}, False),
        ("local", _KV, {"num_bits": (4, 3), "constant_amax": 448.0}, False),
        ("local", f"{_PROJECTION}.input_quantizer", {"type": "dynamic"}, False),
        ("hub-single", f"{_PROJECTION}.input_quantizer", {}, True),
        ("hub-sharded", f"{_PROJECTION}.input_quantizer", {}, True),
    ],
)
def test_loading_calibration_and_export(
    tiny_checkpoint, monkeypatch, location, activation, attributes, needs_forward
):
    """Local and Hub checkpoints place exact weights, calibrate MTP, and export matching scales."""
    cls, source, checkpoint = tiny_checkpoint
    if location.startswith("hub"):
        checkpoint = _hub_checkpoint(checkpoint, source, monkeypatch, location == "hub-sharded")
    wrapped = hasattr(source, "language_model")
    model_type = "nemotron_h_omni" if wrapped else "nemotron_h"
    loader = cls if wrapped else AutoModelForCausalLM
    original_init = cls.__init__
    with hf.prepare_model_for_loading(model_type, checkpoint, trust_remote_code=wrapped):
        model, info = loader.from_pretrained(
            checkpoint, local_files_only=True, output_loading_info=True
        )
    assert cls.__init__ is original_init
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert hf.checkpoint_has_mtp(model_type, checkpoint)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    if wrapped:
        adapter.get_class_from_dynamic_module.assert_called_once_with(
            "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3", checkpoint
        )
    language_model = getattr(model, "language_model", model)
    prefix = "language_model.mtp." if wrapped else "mtp."
    projection_name = prefix + (
        _SHARED if activation == f"{_SHARED}.input_quantizer" else _PROJECTION
    )
    cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    cfg["quant_cfg"].append({"quantizer_name": "*", "enable": False})
    cfg["quant_cfg"].append(
        {"quantizer_name": f"{projection_name}.weight_quantizer", "enable": True}
    )
    if activation:
        cfg["quant_cfg"].append(
            {
                "quantizer_name": prefix + activation,
                "enable": True,
                **({"cfg": attributes} if attributes else {}),
            }
        )
    mtq.quantize(model, {**cfg, "algorithm": None})
    mtp = language_model.mtp
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(inputs, use_cache=False).logits
    original_forward = language_model.forward
    hf.prepare_model_for_calibration(model)
    wrapped_forward = language_model.forward
    assert wrapped_forward is not original_forward
    hf.prepare_model_for_calibration(model)
    assert language_model.forward is wrapped_forward
    calls = []
    mtp.register_forward_hook(lambda *_: calls.append(True))

    outputs = []
    max_calibrate(
        model, lambda calibrated: outputs.append(calibrated(inputs, use_cache=False).logits)
    )
    torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)
    assert model.get_submodule(projection_name).weight_quantizer.amax is not None
    assert calls == ([True] if needs_forward else [])
    assert not language_model.model.norm_f._forward_pre_hooks
    if needs_forward or "constant_amax" in attributes:
        names = fnmatch.filter(dict(model.named_modules()), prefix + activation)
        assert names
        for name in names:
            amax = model.get_submodule(name).amax
            assert amax is not None and torch.isfinite(amax).all() and amax.max() > 0
    if needs_forward and activation.endswith(".input_quantizer"):
        projection = model.get_submodule(projection_name)
        assert projection.input_quantizer.amax.max() > 0
        expected_scale = get_activation_scaling_factor(projection).squeeze().clone()
        _process_quantized_modules(model, torch.bfloat16)
        exported = postprocess_state_dict(model.state_dict(), maxbound=448, quantization=None)
        torch.testing.assert_close(exported[f"{projection_name}.input_scale"], expected_scale)


@pytest.mark.parametrize("algorithm", ["awq_lite", "awq_clip", "awq_full"])
@pytest.mark.parametrize("activation", [False, True])
def test_awq_runs_mtp_with_disabled_activation_quantizers(tiny_checkpoint, algorithm, activation):
    """AWQ must exercise MTP even while it disables the input quantizers for its search."""
    _, model, _ = tiny_checkpoint
    language_model = getattr(model, "language_model", model)
    mtp = language_model.mtp
    cfg = copy.deepcopy(mtq.INT4_AWQ_CFG)
    cfg["quant_cfg"].extend(
        [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": f"{_PROJECTION}.weight_quantizer",
                "enable": True,
                "cfg": {"num_bits": 4, "block_sizes": {-1: 16}},
            },
            {"quantizer_name": f"{_PROJECTION}.input_quantizer", "enable": activation},
        ]
    )
    mtq.quantize(mtp, {**cfg, "algorithm": None})
    hf.prepare_model_for_calibration(model)
    inputs = torch.tensor([[1, 2, 3, 4]])
    calls = []
    handle = mtp.register_forward_hook(lambda *_: calls.append(True))
    try:
        awq(model, lambda calibrated: calibrated(inputs, use_cache=False), algorithm, debug=True)
        projection = mtp.get_submodule(_PROJECTION)
        if algorithm != "awq_clip":
            assert projection.awq_lite.num_cache_steps > 0
            assert projection.awq_lite.num_search_steps > 0
            assert torch.isfinite(projection.input_quantizer.pre_quant_scale).all()
        if algorithm != "awq_lite":
            assert projection.awq_clip.num_tokens > 0
        assert torch.isfinite(projection.weight_quantizer.amax).all()
        if activation:
            assert projection.input_quantizer.amax is not None
            assert torch.isfinite(projection.input_quantizer.amax).all()
        assert projection.input_quantizer.is_enabled == activation
        assert calls
        assert not hasattr(projection, "_forward_no_awq")
        if not activation:
            calls.clear()
            model(inputs, use_cache=False)
            assert not calls  # Retained debug helpers must not keep MTP execution enabled.
    finally:
        handle.remove()


@pytest.mark.parametrize("activation", ["disabled", "constant", "dynamic"])
@pytest.mark.parametrize("projection", [_PROJECTION, "layers.1.mixer.experts"])
def test_local_hessian_collects_mtp_inputs(tiny_checkpoint, activation, projection):
    _, model, _ = tiny_checkpoint
    language_model = getattr(model, "language_model", model)
    mtp = language_model.mtp
    cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    cfg["quant_cfg"].extend(
        [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": f"{projection}.*weight_quantizer", "enable": True},
        ]
    )
    if activation != "disabled":
        cfg["quant_cfg"].append(
            {
                "quantizer_name": f"{projection}.*input_quantizer",
                "enable": True,
                "cfg": {"constant_amax": 448.0}
                if activation == "constant"
                else {"type": "dynamic"},
            }
        )
    mtq.quantize(mtp, {**cfg, "algorithm": None})
    hf.prepare_model_for_calibration(model)
    weight_quantizers = {
        id(q): q
        for name, q in mtp.get_submodule(projection).named_modules()
        if isinstance(q, TensorQuantizer) and q.is_enabled and "weight_quantizer" in name
    }
    inputs = torch.tensor([[1, 2, 3, 4]])
    calls = []
    handle = mtp.register_forward_hook(lambda *_: calls.append(True))
    try:
        local_hessian_calibrate(
            model,
            lambda calibrated: calibrated(inputs, use_cache=False),
            fp8_scale_sweep=False,
            debug=True,
        )
        accumulators = model._local_hessian_accumulators
        assert accumulators and accumulators.keys() <= weight_quantizers.keys()
        for acc in accumulators.values():
            assert acc.num_samples > 0
            assert acc.hessian_per_block is not None
            assert torch.isfinite(acc.hessian_per_block).all()
        assert all(torch.isfinite(q.amax).all() for q in weight_quantizers.values())
        assert calls
        assert not any(module._forward_pre_hooks for module in mtp.modules())
        calls.clear()
        model(inputs, use_cache=False)
        assert not calls  # Retained debug accumulators must not keep MTP execution enabled.
    finally:
        handle.remove()


@pytest.mark.parametrize("activation", ["disabled", "constant", "dynamic"])
@torch.no_grad()
def test_gptq_collects_mtp_inputs_and_restores_forward(tiny_checkpoint, activation):
    _, model, _ = tiny_checkpoint
    mtp = getattr(model, "language_model", model).mtp
    cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    cfg["quant_cfg"].extend(
        [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": f"{_PROJECTION}.weight_quantizer", "enable": True},
        ]
    )
    if activation != "disabled":
        cfg["quant_cfg"].append(
            {
                "quantizer_name": f"{_PROJECTION}.input_quantizer",
                "cfg": {"constant_amax": 1.0} if activation == "constant" else {"type": "dynamic"},
            }
        )
    mtq.quantize(mtp, {**cfg, "algorithm": None})
    hf.prepare_model_for_calibration(model)
    inputs = torch.tensor([[1, 2, 3, 4]])
    max_calibrate(model, lambda m: m(inputs, use_cache=False))
    projection = mtp.get_submodule(_PROJECTION)
    original_forward = projection.forward
    # The real GPTQ collector and unfused weight update run on CPU without GPU-memory offload.
    helper = GPTQHelper(projection, _PROJECTION)
    helper.setup()
    try:
        with set_quantizer_by_cfg_context(
            model, [{"quantizer_name": "*weight_quantizer", "enable": False}]
        ):
            model(inputs, use_cache=False)
        assert helper.n_samples == inputs.numel()
        assert torch.isfinite(helper.hessian).all() and helper.hessian.abs().sum() > 0
        helper.update_weights(block_size=16, perc_damp=0.01)
        assert torch.isfinite(projection.weight).all()
    finally:
        helper.cleanup()
        helper.free()
    assert projection.forward == original_forward
    assert not hasattr(projection, GPTQHelper.CACHE_NAME)
    calls = []
    handle = mtp.register_forward_hook(lambda *_: calls.append(True))
    try:
        model(inputs, use_cache=False)
        assert not calls
    finally:
        handle.remove()


def test_lifecycle_dispatch_ignores_unsupported_models():
    """Unrelated model families need neither checkpoint inspection nor auxiliary forwards."""
    assert not hf.checkpoint_has_mtp("unsupported", "unused-checkpoint")
    with hf.prepare_model_for_loading("unsupported", "unused-checkpoint", False):
        pass
    model = torch.nn.Linear(2, 2)
    hf.prepare_model_for_calibration(model)


@pytest.mark.parametrize("failure", ["base", "mtp"])
def test_calibration_removes_capture_hook_on_error(tiny_checkpoint, failure):
    _, model, _ = tiny_checkpoint
    language_model = getattr(model, "language_model", model)
    language_model.mtp.input_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
    if failure == "mtp":
        language_model.mtp.layers[0].eh_proj = torch.nn.Linear(1, 32)
    assert adapter.prepare_for_calibration(model)
    with pytest.raises((ValueError, RuntimeError), match=r"exactly one|cannot be multiplied"):
        model(input_ids=None if failure == "base" else torch.tensor([[1, 2]]), use_cache=False)
    assert not language_model.model.norm_f._forward_pre_hooks


def test_puzzle_omni_loads_and_saves_its_checkpoint_names(tmp_path, monkeypatch):
    """A puzzle Omni checkpoint (Nemotron 3.5 Bolt) loads in its own code and keeps its names."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NemotronH_Omni_Reasoning_V3"],
                "model_type": "nemotron_h_omni",
                "llm_config": {"model_type": "nemotron_h_puzzle"},
            }
        )
    )
    if not hasattr(conversion_mapping.WeightRenaming("a", "b"), "scope_prefix"):
        # Its renames would apply to every key, and their reverse corrupt the export's names.
        with (
            pytest.raises(RuntimeError, match="limits a sub-model's weight renames"),
            hf.prepare_model_for_loading("nemotron_h_omni", tmp_path, True),
        ):
            pass
        return
    native = pytest.importorskip("transformers.models.nemotron_h.modeling_nemotron_h")
    if not hasattr(conversion_mapping, "USER_REGISTERED_MAPPINGS"):
        pytest.skip("this test resets USER_REGISTERED_MAPPINGS, which this transformers lacks")
    # Leave transformers' registry as the test found it.
    monkeypatch.setattr(conversion_mapping, "_checkpoint_conversion_mapping_cache", None)
    monkeypatch.setattr(conversion_mapping, "USER_REGISTERED_MAPPINGS", set())

    class PuzzleConfig(native.NemotronHConfig):
        model_type = "nemotron_h_puzzle"

    class PuzzleForCausalLM(native.NemotronHForCausalLM):
        config_class = PuzzleConfig

    class OmniConfig(PretrainedConfig):
        model_type = "nemotron_h_omni"

    class TinyOmni(PreTrainedModel):
        config_class = OmniConfig

        def __init__(self, config):
            super().__init__(config)
            self.language_model = PuzzleForCausalLM(config.llm_config)
            self.vision_projector = torch.nn.ModuleDict({"mlp1": torch.nn.LayerNorm(32)})
            self.post_init()

    # The class name transformers keys its own port's projector renames under.
    omni_class = type("NemotronH_Omni_Reasoning_V3", (TinyOmni,), {})
    # As the checkpoint's code does, for its root model type.
    conversion_mapping.register_checkpoint_conversion_mapping(
        "nemotron_h_omni",
        [conversion_mapping.WeightRenaming(r"^mlp1\.0\.", "vision_projector.mlp1.")],
        overwrite=True,
    )
    llm_config = PuzzleConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=32,
        layers_block_type=["attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        use_mamba_kernels=False,
    )
    config = OmniConfig(llm_config=llm_config)
    source = omni_class(config).eval()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.uniform_(-0.1, 0.1)
    # The checkpoint's names: the language model's base model under ``backbone``, and ``mlp1.0``.
    checkpoint = {
        name.replace("language_model.model.", "language_model.backbone.").replace(
            "vision_projector.mlp1.", "mlp1.0."
        ): tensor.clone()
        for name, tensor in source.state_dict().items()
    }
    save_file(checkpoint, tmp_path / "model.safetensors")

    def loaded_source_tensors(model):
        state = model.state_dict()
        return {
            name for name, tensor in source.state_dict().items() if torch.equal(state[name], tensor)
        }

    without_hook = loaded_source_tensors(omni_class.from_pretrained(tmp_path, config=config))
    assert not any(name.startswith("language_model.model.") for name in without_hook)
    # transformers 5.18 adds renames for its port of the Omni class, which the load must bypass.
    port_renames = conversion_mapping.get_checkpoint_conversion_mapping(omni_class.__name__)
    port_conversions = repr(conversion_mapping.get_model_conversion_mapping(source))
    in_other_thread = []
    with hf.prepare_model_for_loading("nemotron_h_omni", tmp_path, True):
        # A load in another thread, of transformers' port for instance, keeps transformers' renames.
        other = threading.Thread(
            target=lambda: in_other_thread.append(
                repr(conversion_mapping.get_model_conversion_mapping(source))
            )
        )
        other.start()
        other.join()
        loaded = omni_class.from_pretrained(tmp_path, config=config)
    assert in_other_thread == [port_conversions]
    # Nothing the load changed outlives it.
    assert conversion_mapping.get_checkpoint_conversion_mapping("nemotron_h_puzzle") is None
    assert "nemotron_h_puzzle" not in conversion_mapping.USER_REGISTERED_MAPPINGS
    assert loaded_source_tensors(loaded) == set(source.state_dict())
    # A recipe quantizes the whole VLM, layer by layer for sequential GPTQ.
    layers = LayerActivationCollector.get_decoder_layers(loaded)
    assert layers is loaded.language_model.model.layers
    restored = conversion_mapping.get_checkpoint_conversion_mapping(omni_class.__name__)
    assert repr(restored) == repr(port_renames)
    loaded.save_pretrained(tmp_path / "saved")
    assert set(load_file(tmp_path / "saved" / "model.safetensors")) == set(checkpoint)
    # The checkpoint's code, not transformers' port of its Omni class, can hold a puzzle model.
    remote = type("Config", (PretrainedConfig,), {"__module__": "transformers_modules.bolt.cfg"})
    assert hf.prefers_bundled_code("nemotron_h_omni", remote(llm_config=llm_config), tmp_path, {})
