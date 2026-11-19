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

"""MXFP4 experts kept packed through PTQ compute and export what their decompressed load does."""

import importlib.util
import json

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_gpt_oss
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, Mxfp4Config
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.utils import enable_weight_access_and_writeback
from modelopt.torch.quantization.utils.numeric_utils import dequantize_mxfp4
from modelopt.torch.quantization.utils.packed_experts import is_packed
from modelopt.torch.utils.plugins.model_load_utils import load_pretrained


def test_dequantize_mxfp4_matches_the_decoders_of_decompressing_loads():
    generator = torch.Generator().manual_seed(0)
    # Every byte value, and every scale up to where E2M1's 6 overflows bf16.
    blocks = torch.randint(0, 256, (2, 64, 4, 16), dtype=torch.uint8, generator=generator)
    blocks[0, :4] = torch.arange(256, dtype=torch.uint8).reshape(4, 4, 16)
    scales = torch.randint(1, 254, (2, 64, 4), dtype=torch.uint8, generator=generator)
    decoded = dequantize_mxfp4(blocks, scales)

    # transformers' (GPT-OSS's): the transposed layout of the expert parameters.
    expected = convert_moe_packed_tensors(blocks, scales).transpose(1, 2)
    assert torch.equal(decoded.view(torch.int16), expected.contiguous().view(torch.int16))
    if importlib.util.find_spec("compressed_tensors") is not None:
        # compressed-tensors' (Kimi-K3's), one expert's weight at a time.
        from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
        from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

        weights = QuantizationArgs(num_bits=4, type="float", strategy="group", group_size=32)
        scheme = QuantizationScheme(targets=["Linear"], weights=weights)
        for expert in range(2):
            stored = {"weight_packed": blocks[expert].flatten(1), "weight_scale": scales[expert]}
            weight = MXFP4PackedCompressor.decompress(stored, scheme)["weight"]
            assert torch.equal(decoded[expert].view(torch.int16), weight.view(torch.int16))


def _mxfp4_gpt_oss_checkpoint(path):
    """A tiny GPT-OSS checkpoint whose experts are stored in MXFP4, as openai/gpt-oss-* stores them."""
    get_tiny_gpt_oss(num_hidden_layers=2, num_local_experts=4).save_pretrained(path)
    state = load_file(path / "model.safetensors")
    generator = torch.Generator().manual_seed(1)
    for key in [k for k in state if k.endswith(("experts.gate_up_proj", "experts.down_proj"))]:
        experts, inp, out = state.pop(key).shape
        state[f"{key}_blocks"] = torch.randint(
            0, 256, (experts, out, inp // 32, 16), dtype=torch.uint8, generator=generator
        )
        state[f"{key}_scales"] = torch.randint(
            118, 128, (experts, out, inp // 32), dtype=torch.uint8, generator=generator
        )
    save_file(state, path / "model.safetensors", metadata={"format": "pt"})
    config = json.loads((path / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "mxfp4"}
    (path / "config.json").write_text(json.dumps(config))


def test_packed_experts_calibrate_and_export_as_decompressed_ones(tmp_path):
    _mxfp4_gpt_oss_checkpoint(tmp_path / "src")
    models = {
        keep: load_pretrained(
            str(tmp_path / "src"),
            AutoModelForCausalLM,
            quantization_config=Mxfp4Config(dequantize=True),
            keep_compressed=keep,
        )
        for keep in (False, True)
    }
    packed = [m for m in models[True].modules() if is_packed(m)]
    assert len(packed) == 2
    assert packed[0].gate_up_proj.dtype == torch.uint8

    inputs = torch.randint(0, 32, (2, 12), generator=torch.Generator().manual_seed(2))

    def forward_loop(model):
        model(inputs)

    logits = {}
    for keep, model in models.items():
        with torch.no_grad():
            before = model(inputs).logits
        mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=forward_loop)
        with torch.no_grad():
            logits[keep] = (before, model(inputs).logits)
        export_hf_checkpoint(model, export_dir=tmp_path / str(keep))
    assert not any(is_packed(m) for m in models[True].modules())
    for decompressed, kept in zip(logits[False], logits[True]):
        assert torch.equal(decompressed, kept)
    exported = {keep: load_file(tmp_path / str(keep) / "model.safetensors") for keep in models}
    assert exported[True].keys() == exported[False].keys()
    for key, tensor in exported[False].items():
        kept = exported[True][key]
        assert tensor.dtype == kept.dtype and tensor.shape == kept.shape, key
        assert torch.equal(tensor.reshape(-1).view(torch.uint8), kept.reshape(-1).view(torch.uint8))
    for name in ("config.json", "hf_quant_config.json"):
        assert (tmp_path / "True" / name).read_text() == (tmp_path / "False" / name).read_text()


def test_packed_experts_refuse_a_change_they_cannot_hold(tmp_path):
    _mxfp4_gpt_oss_checkpoint(tmp_path / "src")
    model = load_pretrained(
        str(tmp_path / "src"),
        AutoModelForCausalLM,
        quantization_config=Mxfp4Config(dequantize=True),
        keep_compressed=True,
    )
    mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=lambda m: m(torch.zeros(1, 4).long()))
    experts = model.model.layers[0].mlp.experts
    stored = experts.gate_up_proj
    # Weight calibration reads them decompressed, outside any window too.
    assert {w.dtype for w, _ in experts.iter_weights_for_calibration()} == {torch.bfloat16}
    with enable_weight_access_and_writeback(experts, model, writeback=False):
        experts.gate_up_proj.data.zero_()  # dropped, as a read-only window asks
    assert experts.gate_up_proj is stored
    with (
        pytest.raises(ValueError, match="cannot hold the change"),
        enable_weight_access_and_writeback(experts, model),
    ):
        experts.gate_up_proj.data.zero_()
    assert experts.gate_up_proj is stored
    with pytest.raises(ValueError, match="cannot hold folded weights"):
        mtq.fold_weight(model)
    get_tiny_gpt_oss(num_hidden_layers=1).save_pretrained(tmp_path / "bf16")
    with pytest.raises(ValueError, match="kept no experts packed"):
        load_pretrained(str(tmp_path / "bf16"), AutoModelForCausalLM, keep_compressed=True)
