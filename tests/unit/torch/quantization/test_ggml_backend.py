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
import io
import re
from types import SimpleNamespace

import pytest
import torch
from _test_utils.torch.quantization.iq_llama_cpp_vectors import expected_values, packed_blocks

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.ggml as ggml
import modelopt.torch.quantization.ggml.iq1_s as iq1_s_module
import modelopt.torch.quantization.ggml.iq2_xs as iq2_xs_module
import modelopt.torch.quantization.ggml.q8_0 as q8_0_module
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY, IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.backend import ggml_fake_quant
from modelopt.torch.quantization.ggml.common import (
    PIN_OVERRIDES_WEIGHT,
    narrow_to_float32,
    pin_packed_weight,
    pinned_importance,
    pinned_packed_weight,
)
from modelopt.torch.quantization.ggml.scale_fields import decode_scale_fields, replace_scale_fields
from modelopt.torch.quantization.nn import QuantModuleRegistry, TensorQuantizer
from modelopt.torch.quantization.plugins.custom import _ParallelLinear

# Every registered format, paired with its own module. The module -- not the registry -- supplies
# what a test expects, so a mis-wired registry entry cannot make both sides of an assertion agree.
FORMAT_NAMES = sorted(GGML_FORMAT_REGISTRY)
FORMAT_MODULES = [
    (name, importlib.import_module(f"modelopt.torch.quantization.ggml.{name}"))
    for name in FORMAT_NAMES
]


class _CPUParallelLinear(torch.nn.Linear):
    def forward(self, inputs):
        return torch.nn.functional.linear(inputs, self.weight, self.bias)


class _QuantCPUParallelLinear(_ParallelLinear):
    _functionals_to_replace = [(torch.nn.functional, "linear")]


@pytest.fixture
def parallel_linear():
    QuantModuleRegistry.register({_CPUParallelLinear: "test_qdq_parallel_linear"})(
        _QuantCPUParallelLinear
    )
    try:
        yield _CPUParallelLinear
    finally:
        QuantModuleRegistry.unregister(_CPUParallelLinear)


def _substitute(monkeypatch, name, **changes):
    """Replace parts of one registered format for the duration of a test.

    Dispatch reads GGML_FORMAT_REGISTRY, so that is where a substitute has to go: patching the
    format module's function would not reach it.
    """
    monkeypatch.setitem(
        GGML_FORMAT_REGISTRY, name, dataclasses.replace(GGML_FORMAT_REGISTRY[name], **changes)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_bits", FORMAT_NAMES)
@pytest.mark.parametrize("parallel", [False, True])
def test_ggml_backend_via_quantize(num_bits, dtype, parallel, record_property, parallel_linear):
    torch.manual_seed(1234)
    linear = parallel_linear if parallel else torch.nn.Linear
    model = linear(256, 2, bias=False, dtype=dtype)
    inputs = torch.randn(2, 256, dtype=dtype)
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
    assert output.shape == (2, 2) and torch.isfinite(output).all()
    assert not torch.equal(output, unquantized_output)
    parameter = model.weight
    quantizer = model.weight_quantizer
    codec = GGML_FORMAT_REGISTRY[num_bits]
    packed_before = codec.pack(parameter, quantizer).detach().clone()
    expected_weight = quantizer(parameter).detach().clone()
    buffers = {name: value.clone() for name, value in quantizer.named_buffers()}
    mtq.initialize_qdq_weights(model)
    assert model.weight is parameter and quantizer.is_enabled
    torch.testing.assert_close(model.weight, expected_weight, rtol=0, atol=0)
    payload = pinned_packed_weight(quantizer, num_bits)
    fixed_scales = None
    if codec.payload_scales is not None:
        fixed_scales = codec.payload_scales(packed_before.reshape(-1, codec.block_bytes)).clone()
    else:
        assert payload is None
    for name, value in buffers.items():
        torch.testing.assert_close(dict(quantizer.named_buffers())[name], value, rtol=0, atol=0)
    torch.testing.assert_close(model(inputs), output, rtol=0, atol=0)
    torch.testing.assert_close(quantizer(model.weight), expected_weight, rtol=0, atol=0)
    if payload is not None:
        for start in range(2):
            torch.testing.assert_close(
                codec.pack(model.weight[start : start + 1], quantizer),
                payload[start : start + 1],
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            quantizer(model.weight.float()).to(dtype), expected_weight, rtol=0, atol=0
        )
    record_property("initialization_forward_relative_l2", 0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    torch.testing.assert_close(model(inputs), output, rtol=0, atol=0)
    checkpoint_keys = set(model.state_dict())
    checkpoint = io.BytesIO()
    mto.save(model, checkpoint)
    checkpoint.seek(0)
    restored = mto.restore(linear(256, 2, bias=False, dtype=dtype), checkpoint)
    torch.testing.assert_close(restored(inputs), output, rtol=0, atol=0)
    with torch.no_grad():
        restored.weight.copy_(restored.weight.detach().clone())
    torch.testing.assert_close(restored(inputs), output, rtol=0, atol=0)

    # The first actual update must not switch back to estimating scales from decoded weights.
    with torch.no_grad():
        if dtype == torch.bfloat16:
            # Use one BF16 ULP so the update is not rounded away.
            model.weight.data.copy_(
                torch.nextafter(model.weight, torch.full_like(model.weight, float("inf")))
            )
        else:
            model.weight.data.add_(1e-6)
    optimizer.step()
    assert not torch.equal(model.weight, expected_weight)
    if fixed_scales is not None:
        torch.testing.assert_close(
            codec.pack(model.weight, quantizer), packed_before, rtol=0, atol=0
        )
        assert payload is not None
        torch.testing.assert_close(payload, packed_before, rtol=0, atol=0)
        assert payload.shape == (2, 256 // codec.block_size, codec.block_bytes)
        assert payload.dtype == torch.uint8
        assert pinned_importance(quantizer, num_bits) is None

    # A larger shared-buffer write changes codes while retaining the original super-block scales.
    with torch.no_grad():
        model.weight.data.add_(0.125)
    optimizer.step()
    kwargs = {} if fixed_scales is None else {"scales": fixed_scales}
    packed, shape = codec.quantize(model.weight, **kwargs)
    expected_after_update = codec.dequantize(packed, shape, dtype=dtype)
    torch.testing.assert_close(quantizer(model.weight), expected_after_update, rtol=0, atol=0)
    torch.testing.assert_close(
        model(inputs), torch.nn.functional.linear(inputs, expected_after_update), rtol=0, atol=0
    )
    assert not torch.equal(codec.pack(model.weight, quantizer), packed_before)
    assert set(model.state_dict()) == checkpoint_keys
    if payload is not None:
        torch.testing.assert_close(
            pinned_packed_weight(quantizer, num_bits), payload, rtol=0, atol=0
        )
        torch.testing.assert_close(
            codec.payload_scales(payload.reshape(-1, codec.block_bytes)),
            fixed_scales,
            rtol=0,
            atol=0,
        )
    after_update = io.BytesIO()
    mto.save(model, after_update)
    after_update.seek(0)
    resumed = mto.restore(linear(256, 2, bias=False, dtype=dtype), after_update)
    torch.testing.assert_close(
        resumed.weight_quantizer(resumed.weight), expected_after_update, rtol=0, atol=0
    )
    model.weight.grad = None
    model(inputs).float().square().mean().backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()
    assert torch.count_nonzero(model.weight.grad) > 0


@pytest.mark.parametrize("num_bits", FORMAT_NAMES)
def test_new_pin_replaces_qdq_payload(num_bits):
    torch.manual_seed(1234)
    model = torch.nn.Linear(256, 2, bias=False)
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": num_bits, "backend": "ggml"},
            },
        ],
        "algorithm": "max",
    }
    mtq.quantize(model, config)
    mtq.initialize_qdq_weights(model)
    quantizer = model.weight_quantizer
    codec = GGML_FORMAT_REGISTRY[num_bits]
    packed, shape = codec.quantize(model.weight.detach() * 2)
    expected = codec.dequantize(packed, shape, dtype=model.weight.dtype)
    assert not torch.equal(model.weight, expected)
    # AutoQuantize can install a pin while retaining the candidate's source weights.
    setattr(quantizer, PIN_OVERRIDES_WEIGHT, True)
    pin_packed_weight(quantizer, num_bits, packed)
    torch.testing.assert_close(quantizer(model.weight), expected, rtol=0, atol=0)
    torch.testing.assert_close(codec.pack(model.weight, quantizer), packed, rtol=0, atol=0)
    torch.testing.assert_close(pinned_packed_weight(quantizer, num_bits), packed, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("preparation", "num_bits", "width"),
    [
        *[(preparation, "iq2_xxs", 256) for preparation in ("ptq", "ptq_updated", "gptq")],
        *[
            (preparation, name, width)
            for name in FORMAT_NAMES
            for preparation, width in (("ptq", 128), ("ptq_updated", 300))
        ],
    ],
)
def test_parallel_ggml_checkpoint_restore(
    preparation, num_bits, width, dtype, parallel_linear, monkeypatch
):
    torch.manual_seed(1234)
    model = parallel_linear(width, 2, bias=False, dtype=dtype)
    inputs = torch.randn(32, width, dtype=dtype)
    codec = GGML_FORMAT_REGISTRY[num_bits]
    if width % codec.block_size:
        with pytest.raises(ValueError):
            codec.quantize(model.weight)
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {
                    "num_bits": num_bits,
                    "backend": "ggml",
                    "block_sizes": {-1: codec.block_size},
                },
            },
        ],
        "algorithm": {"method": "gptq", "block_size": 256, "importance_weighted": True}
        if preparation == "gptq"
        else "max",
    }
    mtq.quantize(model, config, lambda m: m(inputs))
    initial_output = model(inputs).detach()
    mtq.initialize_qdq_weights(model)
    torch.testing.assert_close(model(inputs), initial_output, rtol=0, atol=0)
    assert model.weight.shape == (2, width)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    torch.testing.assert_close(model(inputs), initial_output, rtol=0, atol=0)
    if preparation == "ptq_updated":
        with torch.no_grad():
            model.weight.add_(0.125)
    expected = model(inputs).detach()
    checkpoint = io.BytesIO()
    mto.save(model, checkpoint)
    checkpoint.seek(0)
    reset = TensorQuantizer._reset_pytorch_state_from_metadata

    def poison_placeholders(quantizer, metadata):
        reset(quantizer, metadata)
        for name, buffer in quantizer.named_buffers(recurse=False):
            if name.startswith("_ggml_pinned_"):
                buffer.zero_()

    monkeypatch.setattr(TensorQuantizer, "_reset_pytorch_state_from_metadata", poison_placeholders)
    restored = mto.restore(parallel_linear(width, 2, bias=False, dtype=dtype), checkpoint)
    torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(dict(restored.named_buffers())[name], buffer, rtol=0, atol=0)


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
    # parameter buffer, which leaves their version counter as it was; the step invalidates the cache.
    trainable, frozen = (torch.nn.Parameter(weight.clone(), requires_grad=r) for r in (True, False))
    for param in (trainable, frozen):
        fake_quant(param, quantizer, block_chunk_size=2)
        param.data.add_(0.01)
        torch.optim.SGD([trainable], lr=0.0).step()
        fake_quant(param, quantizer, block_chunk_size=2)
    # Packed twice for the trainable weight; once for the frozen one, which no optimizer updates.
    assert call_count == 6

    # Unfreezing the cached weight in place, as a trainer configuring trainable parameters does,
    # lets the next optimizer step invalidate its cache too.
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


@pytest.mark.parametrize("sharing", ["shared", "sequential"])
def test_ggml_initialization_refuses_ambiguous_ptq_quantizers_before_writing(sharing):
    model = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.Linear(256, 2))
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": "iq2_xxs", "backend": "ggml"},
                "enable": True,
            },
        ],
        "algorithm": "max",
    }
    if sharing == "sequential":
        config["quant_cfg"][1]["cfg"] = [
            {"num_bits": name, "backend": "ggml"} for name in ["iq2_xxs", "q8_0"]
        ]
    mtq.quantize(model, config, lambda m: m(torch.randn(2, 256)))
    if sharing == "shared":
        model[1].weight_quantizer = model[0].weight_quantizer
    before = [m.weight.detach().clone() for m in model]
    with pytest.raises(ValueError):
        mtq.initialize_qdq_weights(model)
    for m, expected in zip(model, before):
        torch.testing.assert_close(m.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", sorted(IQ_FORMAT_REGISTRY))
@pytest.mark.parametrize("payload_layout", ["native", "flat", "odd_offset"])
def test_native_iq_fields_preserve_codes_and_match_reference(name, payload_layout):
    """Edit captured native bytes and independently reconstruct the new scale geometry."""
    fmt = GGML_FORMAT_REGISTRY[name]
    packed = torch.from_numpy(packed_blocks(name).copy())
    expected = torch.from_numpy(expected_values(name).copy())
    if payload_layout != "native":
        packed, expected = packed[:1], expected[:1]
        if payload_layout == "flat":
            packed = packed.reshape(-1)
        else:
            storage = torch.empty(packed.numel() + 1, dtype=torch.uint8)
            storage[1:] = packed.reshape(-1)
            packed = storage[1:].reshape(1, 1, fmt.block_bytes)
    original = packed.clone()
    shape = torch.tensor(expected.shape)
    torch.testing.assert_close(
        fmt.dequantize(original, shape, dtype=torch.float32), expected, rtol=1e-6, atol=1e-8
    )
    spec = fmt.scale_spec
    global_scale, local, local_max = decode_scale_fields(packed, fmt)
    changed_global = global_scale * 2
    changed_local = (local + 1).clamp_max(local_max)
    edited = replace_scale_fields(packed, fmt, changed_global[:, None], changed_local)
    assert torch.equal(packed, original)
    assert torch.equal(replace_scale_fields(edited, fmt, global_scale, local), original)

    # Explicit bit masks come from GGML's layouts, independently of NativeScaleSpec.
    allowed = torch.zeros_like(packed).reshape(-1, fmt.block_bytes)
    if name == "iq1_m":
        allowed[:, 48:] = 255
    else:
        allowed[:, :2] = 255
        if name == "iq1_s":
            allowed[:, 35:50:2] = 112
        elif name == "iq2_xxs":
            allowed[:, 9::8] = 240
        elif name in {"iq2_xs", "iq2_s"}:
            allowed[:, 66 if name == "iq2_xs" else 74 :] = 255
    assert not ((edited.reshape_as(allowed) ^ original.reshape_as(allowed)) & ~allowed).any()
    levels, _ = fmt.decode_int8(original, shape)
    steps = changed_global[:, None]
    steps = steps * (spec.local_multiplier * changed_local + spec.local_bias) / spec.divisor
    reconstructed = levels.reshape(len(steps), -1, fmt.int8_group_size).float() * steps.unsqueeze(
        -1
    )
    torch.testing.assert_close(
        fmt.dequantize(edited, shape, dtype=torch.float32),
        reconstructed.reshape(expected.shape),
        rtol=1e-6,
        atol=1e-8,
    )
    if payload_layout == "native":
        for value in (float("nan"), float("inf"), -1, local_max + 1, 0.5):
            illegal_local = local.float().clone()
            illegal_local[0, 0] = value
            with pytest.raises(ValueError):
                replace_scale_fields(packed, fmt, changed_global, illegal_local)
        assert torch.equal(packed, original)
    for accessor in ("read_local", "write_local"):
        with pytest.raises(ValueError):
            dataclasses.replace(spec, **{accessor: None})
    with pytest.raises(ValueError):
        decode_scale_fields(packed, dataclasses.replace(fmt, payload_scales=None))
    with pytest.raises(ValueError):
        replace_scale_fields(
            packed,
            dataclasses.replace(fmt, scale_fit=None, replace_payload_scales=None),
            changed_global,
            changed_local,
        )
