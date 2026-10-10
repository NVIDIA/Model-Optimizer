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


"""Tests for ``examples/deepseek/deepseek_v4/ptq.py``: the DeepSeek-V4-Pro-0813 PTQ
recipe and its guard, and the V4-Pro / V4.1-Flash compatibility shims: FP8 block size
(128x128 vs 32x32), the tokenizer-aware ``Transformer`` constructor, and the tuple
returned by V4.1's ``forward``.

``examples/deepseek/deepseek_v4/ptq.py`` keeps ``_build_nvfp4_experts_cfg()`` as its
default, so the recipe and that builder can drift apart without anything failing --
the symptom would only show up as a difference between two amax dumps. Lives here rather
than under ``tests/examples/``: CI runs ``tests/unit`` on every change, while the example
lanes cover a fixed allowlist that has no deepseek entry.
"""

import copy
import fnmatch
import importlib.util
import json
import types
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "deepseek" / "deepseek_v4" / "ptq.py"
_SPEC = importlib.util.spec_from_file_location("deepseek_v4_ptq", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
dsv4_ptq = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dsv4_ptq)

# Names spanning every branch of the config: routed experts (enabled), and the
# groups that must stay untouched -- shared expert, attention, MTP, lm_head.
_PROBES = [
    "model.layers.3.ffn.experts.17.w1_weight_quantizer",
    "model.layers.3.ffn.experts.17.w2_input_quantizer",
    "model.layers.3.ffn.shared_experts.w1_weight_quantizer",
    "model.layers.3.attn.wq_weight_quantizer",
    "mtp.0.ffn.experts.2.w1_weight_quantizer",
    "lm_head_weight_quantizer",
]


def _resolve(quant_cfg, name):
    """Effective (enabled, numeric format) for ``name``; later rules win, as mtq applies them in order."""
    state = (False, None)
    for entry in quant_cfg:
        if not isinstance(entry, dict) or "quantizer_name" not in entry:
            continue
        if fnmatch.fnmatch(name, entry["quantizer_name"]):
            cfg = entry.get("cfg")
            fmt = None
            if cfg:
                # Compare only the fields PTQ acts on. The recipe additionally carries
                # effective_bits from configs/numerics/nvfp4, which is autoquant-only.
                block_sizes = cfg["block_sizes"]
                fmt = (
                    tuple(cfg["num_bits"]),
                    block_sizes[-1],
                    block_sizes["type"],
                    tuple(block_sizes["scale_bits"]),
                )
            state = (entry.get("enable", True), fmt)
    return state


def test_recipe_matches_the_builtin_quant_cfg():
    """Loads via ``_quant_cfg_from_recipe``, so the guard's own checks (max algorithm,
    experts-only scope, NVFP4 encoding) run against the shipped recipe too."""
    recipe_cfg = dsv4_ptq._quant_cfg_from_recipe(dsv4_ptq._PUBLISHED_RECIPE)
    builtin_cfg = dsv4_ptq._build_nvfp4_experts_cfg()

    for name in _PROBES:
        assert _resolve(recipe_cfg["quant_cfg"], name) == _resolve(
            builtin_cfg["quant_cfg"], name
        ), f"recipe and _build_nvfp4_experts_cfg() disagree for {name}"


_NVFP4 = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}


def _entry(name, **cfg_overrides):
    cfg = copy.deepcopy(_NVFP4)
    cfg["block_sizes"].update(cfg_overrides.pop("block_sizes", {}))
    cfg.update(cfg_overrides)
    return {"quantizer_name": name, "enable": True, "cfg": cfg}


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda c: c.update(algorithm="awq_lite"), "max"),
        (
            lambda c: c["quant_cfg"].append(_entry("*shared_experts*weight_quantizer")),
            "routed-expert",
        ),
        (
            lambda c: c["quant_cfg"].append(
                _entry("*ffn.experts.*.w*_weight_quantizer", num_bits=(4, 3))
            ),
            "block-16 NVFP4",
        ),
        (
            lambda c: c["quant_cfg"].append(
                _entry("*ffn.experts.*.w*_weight_quantizer", block_sizes={"type": "static"})
            ),
            "block-16 NVFP4",
        ),
        (
            lambda c: c["quant_cfg"].append(
                _entry("*ffn.experts.*.w*_weight_quantizer", block_sizes={"scale_bits": (8, 0)})
            ),
            "block-16 NVFP4",
        ),
        (
            lambda c: c["quant_cfg"].append(
                {
                    "quantizer_name": "*ffn.experts.*.w*_weight_quantizer",
                    "enable": True,
                    "cfg": [_NVFP4],
                }
            ),
            "non-dict 'cfg'",
        ),
        (
            lambda c: c["quant_cfg"].append(_entry("*mtp.*ffn.experts.*w*_weight_quantizer")),
            "MTP quantizers",
        ),
    ],
    ids=[
        "wrong-algorithm",
        "quantizer-outside-experts",
        "wrong-num-bits",
        "static-block-quantization",
        "wrong-scale-bits",
        "list-valued-cfg",
        "mtp-experts-enabled",
    ],
)
def test_guard_rejects_recipes_the_export_path_cannot_represent(monkeypatch, mutate, match):
    """The manifest is hardcoded to NVFP4_W4A4; a deviating recipe must fail loudly."""
    base = dsv4_ptq.load_recipe(dsv4_ptq._PUBLISHED_RECIPE).quantize.model_dump()
    mutate(base)

    class _Stub:
        quantize = type("Q", (), {"model_dump": staticmethod(lambda: base)})()

    monkeypatch.setattr(dsv4_ptq, "load_recipe", lambda _path: _Stub())
    with pytest.raises(ValueError, match=match):
        dsv4_ptq._quant_cfg_from_recipe("ignored")


def test_guard_rejects_a_non_ptq_recipe():
    """``--recipe`` takes any path; a speculative-decoding recipe has no ``quantize``."""
    with pytest.raises(ValueError, match="no 'quantize' section"):
        dsv4_ptq._quant_cfg_from_recipe("general/speculative_decoding/eagle3")


# --- FP8 block size: V4-Pro's ``block_size`` (128) vs V4.1-Flash's ``fp8_block_size`` (32) --


def _fp8_pair(m, n, block):
    weight = torch.zeros(m, n, dtype=torch.float8_e4m3fn)
    scale = torch.zeros(dsv4_ptq._fp8_scale_shape(weight, block), dtype=torch.uint8)
    return weight, scale


@pytest.mark.parametrize(
    ("shape", "block", "expected"),
    [
        ((256, 256), 128, (2, 2)),
        ((300, 70), 128, (3, 1)),
        ((100, 70), 32, (4, 3)),
        ((32, 32), 32, (1, 1)),
    ],
)
def test_fp8_scale_shape_rounds_up(shape, block, expected):
    """Matches ``Linear.__init__``: a partial trailing block still gets a scale."""
    assert dsv4_ptq._fp8_scale_shape(torch.empty(shape), block) == expected


@pytest.mark.parametrize(
    ("module_globals", "block", "expected"),
    [
        ({"block_size": 128}, 128, 128),  # V4-Pro
        ({"fp8_block_size": 32}, 32, 32),  # V4.1-Flash
        ({"fp8_block_size": 32, "block_size": 128}, 32, 32),
        ({"fp8_block_size": 32, "block_size": 128}, 128, 128),  # only the consistent global wins
    ],
    ids=["v4-pro", "v4.1-flash", "both-globals-32", "both-globals-128"],
)
def test_block_size_prefers_the_global_consistent_with_the_tensors(
    monkeypatch, module_globals, block, expected
):
    monkeypatch.setattr(dsv4_ptq, "deekseep_v4_model", types.SimpleNamespace(**module_globals))
    assert dsv4_ptq._ds_fp8_block_size(*_fp8_pair(256, 512, block)) == expected


@pytest.mark.parametrize(
    ("shape", "block"),
    [
        ((256, 512), 32),
        ((256, 512), 128),
        ((100, 70), 32),
        ((300, 70), 128),
    ],
)
def test_block_size_without_globals_picks_the_unique_known_size(monkeypatch, shape, block):
    monkeypatch.setattr(dsv4_ptq, "deekseep_v4_model", types.SimpleNamespace())
    assert dsv4_ptq._ds_fp8_block_size(*_fp8_pair(*shape, block)) == block


def test_block_size_rejects_shapes_no_known_block_explains(monkeypatch):
    monkeypatch.setattr(dsv4_ptq, "deekseep_v4_model", types.SimpleNamespace(block_size=128))
    weight = torch.zeros(256, 256, dtype=torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="cannot infer FP8 block size"):
        dsv4_ptq._ds_fp8_block_size(weight, torch.zeros(3, 5, dtype=torch.uint8))


def test_block_size_rejects_shapes_several_known_blocks_explain(monkeypatch):
    """A 32x32 weight has a (1, 1) scale at both 32 and 128; refuse to guess."""
    monkeypatch.setattr(dsv4_ptq, "deekseep_v4_model", types.SimpleNamespace())
    with pytest.raises(AssertionError, match=r"matching these shapes: \[32, 128\]"):
        dsv4_ptq._ds_fp8_block_size(*_fp8_pair(32, 32, 32))


def test_fp8_dequant_crops_partial_blocks():
    """Each element is scaled by its own block's UE8M0 exponent, including the partial
    trailing blocks that the ceil-div scale shape allocates."""
    m, n, block = 100, 70, 32
    gen = torch.Generator().manual_seed(0)
    weight = (torch.rand(m, n, generator=gen) * 4 - 2).to(torch.float8_e4m3fn)
    exp = torch.randint(-3, 4, dsv4_ptq._fp8_scale_shape(weight, block), generator=gen)
    scale = (exp + 127).to(torch.uint8)
    got = dsv4_ptq._fp8_ue8m0_blockwise_to_bf16(weight, scale, block=block)
    rows, cols = torch.arange(m)[:, None] // block, torch.arange(n)[None, :] // block
    want = (weight.to(torch.float32) * torch.pow(2.0, exp[rows, cols].to(torch.float32))).to(
        torch.bfloat16
    )
    assert got.shape == (m, n)
    assert torch.equal(got, want)


# --- Transformer constructor: V4-Pro's ``(args)`` vs V4.1-Flash's ``(args, tokenizer)`` --


class _ModelArgs:
    def __init__(self, **kwargs):
        self.max_batch_size = 1
        self.__dict__.update(kwargs)


class _LegacyTransformer:
    def __init__(self, args):
        self.args, self.tokenizer = args, None


class _TokenizerAwareTransformer:
    def __init__(self, args, tokenizer):
        self.args, self.tokenizer = args, tokenizer


@pytest.fixture
def load_with(monkeypatch, tmp_path):
    """Call ``load_deepseek_v4`` on CPU against a stub reference module."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"max_batch_size": 1}))
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    # Both are process-global; the real calls would leak bf16 / CUDA defaults into later tests.
    monkeypatch.setattr(torch, "set_default_dtype", lambda _dtype: None)
    monkeypatch.setattr(torch, "set_default_device", lambda _device: None)

    def _load(transformer_cls, tokenizer):
        monkeypatch.setattr(
            dsv4_ptq,
            "deekseep_v4_model",
            types.SimpleNamespace(ModelArgs=_ModelArgs, Transformer=transformer_cls),
        )
        return dsv4_ptq.load_deepseek_v4(
            str(config), "unused", batch_size=4, dummy_weights=True, tokenizer=tokenizer
        )

    return _load


def test_legacy_transformer_is_built_from_args_only(load_with):
    model = load_with(_LegacyTransformer, tokenizer=object())
    assert model.tokenizer is None
    assert model.args.max_batch_size == 4


def test_tokenizer_aware_transformer_receives_the_tokenizer(load_with):
    tokenizer = object()
    model = load_with(_TokenizerAwareTransformer, tokenizer=tokenizer)
    assert model.tokenizer is tokenizer


def test_tokenizer_aware_transformer_requires_a_tokenizer(load_with):
    with pytest.raises(AssertionError, match="requires a tokenizer"):
        load_with(_TokenizerAwareTransformer, tokenizer=None)


# --- forward output: V4-Pro's logits vs V4.1-Flash's ``(output_ids, logits, main_hidden)`` --

_VOCAB = 16


class _Tokenizer:
    def __init__(self):
        self.decoded = None

    def __call__(self, prompt, return_tensors):
        return types.SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, ids, skip_special_tokens):
        self.decoded = ids
        return ""


class _CountingModel:
    """Emits token ``10 + step``. In the tuple form, the other two entries point at a
    different token, so taking the wrong element changes the completion."""

    def __init__(self, returns_tuple):
        self.returns_tuple, self.step = returns_tuple, 0

    def forward(self, tokens, start_pos):
        logits = torch.nn.functional.one_hot(torch.tensor([10 + self.step]), _VOCAB).float()
        self.step += 1
        if not self.returns_tuple:
            return logits
        decoy = torch.nn.functional.one_hot(torch.tensor([0]), _VOCAB).float()
        return decoy, logits, decoy


@pytest.mark.parametrize("returns_tuple", [False, True], ids=["v4-pro-logits", "v4.1-tuple"])
def test_generate_takes_the_logits_from_either_forward_output(monkeypatch, returns_tuple):
    monkeypatch.setenv("RANK", "0")
    tokenizer = _Tokenizer()
    dsv4_ptq._run_quantized_generate(
        _CountingModel(returns_tuple), tokenizer, "prompt", max_new_tokens=3, device="cpu"
    )
    assert tokenizer.decoded == [10, 11, 12]
