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

import dataclasses
import importlib
import re
from types import SimpleNamespace

import pytest
import torch

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.ggml as ggml
import modelopt.torch.quantization.ggml.iq1_s as iq1_s_module
import modelopt.torch.quantization.ggml.iq2_xs as iq2_xs_module
import modelopt.torch.quantization.ggml.q8_0 as q8_0_module
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.backend import ggml_fake_quant
from modelopt.torch.quantization.ggml.common import narrow_to_float32
from modelopt.torch.quantization.nn import TensorQuantizer

# Every registered format, paired with its own module. The module -- not the registry -- supplies
# what a test expects, so a mis-wired registry entry cannot make both sides of an assertion agree.
FORMAT_NAMES = sorted(GGML_FORMAT_REGISTRY)
FORMAT_MODULES = [
    (name, importlib.import_module(f"modelopt.torch.quantization.ggml.{name}"))
    for name in FORMAT_NAMES
]


@pytest.mark.parametrize(
    ("num_bits", "transform"), [(name, None) for name in FORMAT_NAMES] + [("iq2_s", "awq")]
)
def test_native_ggml_scale_repair_keeps_codes_and_reload(
    num_bits, transform, tmp_path, monkeypatch
):
    import gc
    import weakref

    import modelopt.torch.opt as mto
    from modelopt.torch.distill import LocalDistillationConfig
    from modelopt.torch.quantization.ggml.common import pinned_packed_weight
    from modelopt.torch.quantization.repair import QuantizationRepairPolicy

    torch.manual_seed(12)
    layer = torch.nn.Linear(256, 2, bias=False)
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": num_bits, "backend": "ggml"},
            },
        ],
        "algorithm": "max",
    }
    mtq.quantize(layer, cfg)
    if transform == "awq":
        layer.weight_quantizer.pre_quant_scale = torch.linspace(0.5, 1.5, 256)
    inputs = torch.randn(3, 256)
    original = layer(inputs).detach()
    fmt = GGML_FORMAT_REGISTRY[num_bits]
    packed = layer.weight_quantizer._quantizer_cache.packed_weights.clone()
    original_levels, _ = fmt.unpack_int8(packed.reshape(-1, fmt.block_bytes))
    decoded = []
    if num_bits == "iq2_s" and transform is None:
        original_decode = type(fmt).decode_int8

        def track_decode(self, *args, **kwargs):
            levels, steps = original_decode(self, *args, **kwargs)
            storage_owner = levels
            while storage_owner._base is not None:
                storage_owner = storage_owner._base
            decoded.append(weakref.ref(storage_owner))
            return levels, steps

        monkeypatch.setattr(type(fmt), "decode_int8", track_decode)
    policy = QuantizationRepairPolicy()
    groups = policy.prepare(layer, LocalDistillationConfig(updates=1))
    torch.testing.assert_close(layer(inputs), original, rtol=0, atol=0)
    optimizer = torch.optim.Adagrad([{"params": g.params, "lr": g.lr} for g in groups])
    layer(inputs).square().mean().backward()
    assert layer.weight.grad is None
    optimizer.step()
    policy.project_()
    changed = layer(inputs).detach()
    assert not torch.equal(changed, original)
    if num_bits == "iq2_s" and transform is None:
        gc.collect()
        assert decoded and all(reference() is None for reference in decoded), (
            "Expanded expert codes must be released after replay; keep only native packed codes."
        )
    if num_bits == "q8_0":
        # A provider can persist its native fields without policy type dispatch.
        layer.weight_quantizer._local_repair_state.finalize_(layer.weight_quantizer, layer.weight)
        policy.close()
    else:
        policy.finalize_()
    if transform:
        assert layer.weight_quantizer.pre_quant_scale is None
    native = pinned_packed_weight(layer.weight_quantizer, num_bits)
    assert native is not None
    levels, _ = fmt.unpack_int8(native.reshape(-1, fmt.block_bytes))
    assert torch.equal(levels, original_levels)
    torch.testing.assert_close(layer(inputs), changed, rtol=0, atol=0)
    mto.save(layer, tmp_path / "repaired.pt")
    restored = mto.restore(torch.nn.Linear(256, 2, bias=False), tmp_path / "repaired.pt")
    torch.testing.assert_close(restored(inputs), changed, rtol=0, atol=0)
    assert torch.equal(pinned_packed_weight(restored.weight_quantizer, num_bits), native)


def _substitute(monkeypatch, name, **changes):
    """Replace parts of one registered format for the duration of a test.

    Dispatch reads GGML_FORMAT_REGISTRY, so that is where a substitute has to go: patching the
    format module's function would not reach it.
    """
    monkeypatch.setitem(
        GGML_FORMAT_REGISTRY, name, dataclasses.replace(GGML_FORMAT_REGISTRY[name], **changes)
    )


@pytest.mark.parametrize("num_bits", FORMAT_NAMES)
def test_ggml_backend_via_quantize(num_bits):
    torch.manual_seed(1234)
    model = torch.nn.Linear(256, 2, bias=False)
    inputs = torch.randn(2, 256)
    unquantized_output = model(inputs).detach()
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": num_bits, "backend": "ggml"},
                "enable": True,
            },
        ],
        "algorithm": "max",
    }

    mtq.quantize(model, config, forward_loop=lambda module: module(inputs))
    output = model(inputs)

    assert model.weight_quantizer.backend == "ggml"
    assert model.weight_quantizer.num_bits == num_bits
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
    assert not torch.equal(output, unquantized_output)


def test_ggml_backend_rejects_unknown_format():
    # Match the dispatcher's own wording: a bare "requires num_bits" would also accept the
    # per-format guard's message and so could pass on the wrong error.
    with pytest.raises(ValueError, match="requires num_bits in"):
        ggml_fake_quant(torch.ones(1, 256), SimpleNamespace(num_bits="unknown"))


def test_ggml_codecs_are_exported_from_quantization_package():
    assert mtq.quantize_iq1_s is iq1_s_module.quantize_iq1_s
    assert mtq.quantize_iq2_xs is iq2_xs_module.quantize_iq2_xs
    assert mtq.quantize_q8_0 is q8_0_module.quantize_q8_0


@pytest.mark.parametrize(
    "extra_args",
    [
        {"block_chunk_size": 17},
        {"decode_chunk_size": 19},
        {"block_chunk_size": 17, "decode_chunk_size": 19},
    ],
)
def test_ggml_backend_forwards_chunk_sizes(monkeypatch, extra_args):
    """Both chunk knobs are per-quantizer tunable; they bound different loops."""
    received = {}

    def fake_quant(inputs, _quantizer, **kwargs):
        received.update(kwargs)
        return inputs

    # The dispatcher resolves through the registry, so that is the seam to patch.
    monkeypatch.setitem(GGML_FORMAT_REGISTRY, "iq1_s", SimpleNamespace(fake_quant=fake_quant))
    inputs = torch.ones(1, 256)
    quantizer = SimpleNamespace(num_bits="iq1_s", backend_extra_args=extra_args)

    assert ggml_fake_quant(inputs, quantizer) is inputs
    assert received == extra_args


def test_ggml_backend_rejects_unknown_extra_arg():
    quantizer = SimpleNamespace(num_bits="iq1_s", backend_extra_args={"unknown": 1})

    with pytest.raises(ValueError, match="Unsupported ggml backend_extra_args"):
        ggml_fake_quant(torch.ones(1, 256), quantizer)


@pytest.mark.parametrize(("num_bits", "module"), FORMAT_MODULES)
def test_ggml_backend_caches_packed_weight_and_invalidates_on_change(monkeypatch, num_bits, module):
    block_size = GGML_FORMAT_REGISTRY[num_bits].block_size
    weight = torch.randn(1, block_size)
    quantizer = SimpleNamespace(num_bits=num_bits, _quantizer_cache=None)
    original_quantize = getattr(module, f"quantize_{num_bits}")
    call_count = 0

    def counted_quantize(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_quantize(*args, **kwargs)

    _substitute(monkeypatch, num_bits, quantize=counted_quantize)
    fake_quant = GGML_FORMAT_REGISTRY[num_bits].fake_quant

    fake_quant(weight, quantizer, block_chunk_size=1)
    fake_quant(weight, quantizer, block_chunk_size=1)
    assert call_count == 1

    fake_quant(weight, quantizer, block_chunk_size=2)
    assert call_count == 2

    with torch.no_grad():
        weight.add_(0.01)
    fake_quant(weight, quantizer, block_chunk_size=2)
    assert call_count == 3

    # Megatron's distributed optimizer writes updated weights through ``.data`` views of its
    # parameter buffer, which leaves their version counter as it was; the step retires the payload.
    trainable, frozen = (torch.nn.Parameter(weight.clone(), requires_grad=r) for r in (True, False))
    for param in (trainable, frozen):
        fake_quant(param, quantizer, block_chunk_size=2)
        param.data.add_(0.01)
        torch.optim.SGD([trainable], lr=0.0).step()
        fake_quant(param, quantizer, block_chunk_size=2)
    # Packed twice for the trainable weight; once for the frozen one, which no optimizer updates.
    assert call_count == 6

    # Unfreezing the cached weight in place, as a trainer configuring trainable parameters does,
    # lets the next optimizer step retire its payload too.
    frozen.requires_grad_(True)
    fake_quant(frozen, quantizer, block_chunk_size=2)
    frozen.data.add_(0.01)
    torch.optim.SGD([frozen], lr=0.0).step()
    fake_quant(frozen, quantizer, block_chunk_size=2)
    assert call_count == 7


def test_narrow_to_float32_matches_the_cuda_load_float_policy():
    """Non-finite elements become zero; finite out-of-range elements saturate."""
    largest = torch.finfo(torch.float32).max
    values = torch.tensor(
        [torch.nan, torch.inf, -torch.inf, 1e100, -1e100, 1.5], dtype=torch.float64
    )

    narrowed = narrow_to_float32(values)

    assert narrowed.dtype is torch.float32
    assert torch.equal(
        narrowed, torch.tensor([0.0, 0.0, 0.0, largest, -largest, 1.5], dtype=torch.float32)
    )


@pytest.mark.parametrize(("num_bits", "module"), FORMAT_MODULES)
def test_ggml_weight_is_packed_once_across_forwards(monkeypatch, num_bits, module):
    """The packed weight is reused across forwards rather than re-encoded each time.

    TensorQuantizer hands the backend a fresh view of the weight on every forward, so a cache
    that checked tensor identity never hit: the codebook search reran on every forward, roughly
    100x during a generate loop and over 90% of a PTQ run's wall clock.
    """
    packer = f"quantize_{num_bits}"
    original = getattr(module, packer)
    calls = []

    def counting(weight, **kwargs):
        calls.append(tuple(weight.shape))
        return original(weight, **kwargs)

    _substitute(monkeypatch, num_bits, quantize=counting)
    block_size = GGML_FORMAT_REGISTRY[num_bits].block_size
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=num_bits, block_sizes={-1: block_size}, backend="ggml")
    )
    weight = torch.randn(4, 256)

    with torch.inference_mode():  # what generate() runs under
        for _ in range(5):
            quantizer(weight)

    assert calls == [(weight.numel() // block_size, block_size)], (
        f"expected one pack, got {len(calls)}"
    )


@pytest.mark.parametrize(("num_bits", "module"), FORMAT_MODULES)
def test_ggml_decode_chunk_is_sized_independently_of_the_encode_chunk(
    monkeypatch, num_bits, module
):
    """The decode runs every forward; the encode runs once and holds the big temporaries.

    Sharing one constant between them is what made IQ2_XS four times slower end to end than
    IQ1_S, so pin that the decode gets its own, larger chunk.
    """
    seen = {}
    original = getattr(module, f"dequantize_{num_bits}")

    def recording(packed_weights, weight_shape, **kwargs):
        seen["block_chunk_size"] = kwargs["block_chunk_size"]
        return original(packed_weights, weight_shape, **kwargs)

    _substitute(monkeypatch, num_bits, dequantize=recording)
    block_size = GGML_FORMAT_REGISTRY[num_bits].block_size
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=num_bits, block_sizes={-1: block_size}, backend="ggml")
    )
    quantizer(torch.randn(4, 256))

    assert seen["block_chunk_size"] == module._DEFAULT_DECODE_CHUNK_SIZE
    assert module._DEFAULT_DECODE_CHUNK_SIZE > module._DEFAULT_BLOCK_CHUNK_SIZE


def test_registry_lists_every_exported_encoder():
    """An encoder the package exports but the registry omits would be unreachable by dispatch."""
    encoders = {
        name.removeprefix("quantize_") for name in ggml.__all__ if name.startswith("quantize_")
    }
    assert encoders == set(GGML_FORMAT_REGISTRY)


def test_iq_registry_remains_a_compatible_subset_of_ggml_registry():
    assert set(ggml.IQ_FORMAT_REGISTRY) == {
        name for name in GGML_FORMAT_REGISTRY if name.startswith("iq")
    }
    assert "q8_0" not in ggml.IQ_FORMAT_REGISTRY
    for name, record in ggml.IQ_FORMAT_REGISTRY.items():
        assert record is GGML_FORMAT_REGISTRY[name]


@pytest.mark.parametrize(("num_bits", "module"), FORMAT_MODULES)
def test_registry_record_is_wired_to_its_own_codec(num_bits, module):
    """Each record points at its own format's encoder, decoder, geometry and chunk defaults."""
    record = GGML_FORMAT_REGISTRY[num_bits]
    upper = num_bits.upper()

    assert isinstance(record, ggml.GGMLFormat)
    assert record.name == num_bits
    assert record.quantize is getattr(module, f"quantize_{num_bits}")
    assert record.dequantize is getattr(module, f"dequantize_{num_bits}")
    assert record.block_size == getattr(module, f"{upper}_BLOCK_SIZE")
    assert record.block_bytes == getattr(module, f"{upper}_BLOCK_BYTES")
    assert record.effective_bits == pytest.approx(getattr(module, f"{upper}_EFFECTIVE_BITS"))
    assert record.block_chunk_size == module._DEFAULT_BLOCK_CHUNK_SIZE
    assert record.decode_chunk_size == module._DEFAULT_DECODE_CHUNK_SIZE


@pytest.mark.parametrize(("num_bits", "module"), FORMAT_MODULES)
def test_public_fake_quant_is_the_registered_record(num_bits, module):
    """The per-format entry point and backend dispatch run the same code path."""
    assert getattr(module, f"{num_bits}_fake_quant").__self__ is GGML_FORMAT_REGISTRY[num_bits]


@pytest.mark.parametrize("num_bits", FORMAT_NAMES)
def test_format_fake_quant_rejects_another_formats_quantizer(num_bits):
    """Calling one format's fake quant with a quantizer configured for another is refused.

    Dispatch picks the record by num_bits, so it never reaches this guard; it protects direct
    callers of a record or of a per-format ``<fmt>_fake_quant`` alias.
    """
    other = next(name for name in FORMAT_NAMES if name != num_bits)
    expected = f"The ggml {num_bits.upper()} backend requires num_bits={num_bits!r}"
    block_size = GGML_FORMAT_REGISTRY[num_bits].block_size

    with pytest.raises(ValueError, match=re.escape(expected)):
        GGML_FORMAT_REGISTRY[num_bits].fake_quant(
            torch.ones(1, block_size), SimpleNamespace(num_bits=other)
        )
