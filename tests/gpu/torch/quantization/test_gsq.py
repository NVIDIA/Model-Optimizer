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

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama
from safetensors.torch import load_file

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_utils import preprocess_linear_fusion
from modelopt.torch.export.unified_export_hf import export_hf_checkpoint
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from modelopt.torch.quantization.utils.shared_input import SHARED_PATTERNS, find_shared_input_groups


@pytest.mark.parametrize("initializer", ["max", "gptq"])
@pytest.mark.parametrize("nvfp4", [False, True])
def test_gsq_region_reconstruction_and_restore(tmp_path, initializer, nvfp4):
    torch.manual_seed(17)
    model = get_tiny_llama().cuda().bfloat16().eval()
    model.config.architectures = ["LlamaForCausalLM"]
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight[0, :16].zero_()
    pristine = copy.deepcopy(model)
    inputs = torch.randint(0, model.config.vocab_size, (1, 16), device="cuda")
    numeric = (
        {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)}}
        if nvfp4
        else {"num_bits": (4, 3), "axis": None}
    )
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": numeric},
            {"quantizer_name": "*lm_head*", "enable": False},
            {"quantizer_name": "*", "parent_class": "nn.Embedding", "enable": False},
        ]
    }
    samples = {}
    handles = []
    for name in ("self_attn", "mlp"):

        def capture(m, args, kwargs, output, name=name):
            samples[name] = (
                args,
                kwargs,
                output if isinstance(output, torch.Tensor) else output[0],
            )

        handles.append(
            getattr(pristine.model.layers[0], name).register_forward_hook(capture, with_kwargs=True)
        )
    with torch.no_grad():
        pristine(inputs, use_cache=False)
    for handle in handles:
        handle.remove()
    baseline = copy.deepcopy(pristine)
    cfg["algorithm"] = {"method": initializer, "layerwise": {"enable": True}}
    mtq.quantize(baseline, cfg, lambda m: m(inputs, use_cache=False))
    for _, members in find_shared_input_groups(baseline, SHARED_PATTERNS):
        preprocess_linear_fusion(members)
    cfg["algorithm"] = {"method": "gsq", "initializer": initializer, "steps": 2}
    mtq.quantize(model, cfg, lambda m: m(inputs, use_cache=False))
    with torch.no_grad():
        for name, (args, kwargs, target) in samples.items():
            errors = []
            for candidate in (baseline, model):
                output = getattr(candidate.model.layers[0], name)(*args, **kwargs)
                output = output if isinstance(output, torch.Tensor) else output[0]
                errors.append((output.float() - target.float()).square().mean())
            assert errors[1] <= errors[0] * 1.01 + 1e-8
    result = model(inputs).logits
    assert torch.isfinite(result).all()
    for layer in model.model.layers:
        assert layer.gsq["self_attn"]["best_loss"] <= layer.gsq["self_attn"]["initial_loss"]
        assert layer.gsq["mlp"]["best_loss"] <= layer.gsq["mlp"]["initial_loss"]
        assert not layer.self_attn.q_proj.input_quantizer.is_enabled
    assert torch.equal(model.lm_head.weight, pristine.lm_head.weight)
    assert torch.equal(model.model.embed_tokens.weight, pristine.model.embed_tokens.weight)
    mto.save(model, tmp_path / "model.pt")
    restored = mto.restore(pristine, tmp_path / "model.pt")
    torch.testing.assert_close(restored(inputs).logits, result, rtol=0, atol=0)

    projection = restored.model.layers[0].self_attn.q_proj
    weight = projection.weight.detach().clone()
    shape, dtype = weight.shape, weight.dtype
    qdq = projection.weight_quantizer(weight).detach()
    export_hf_checkpoint(restored, export_dir=tmp_path / "export")
    exported = load_file(str(tmp_path / "export" / "model.safetensors"), device="cuda")
    prefix = "model.layers.0.self_attn.q_proj."
    if nvfp4:
        packed = NVFP4QTensor(shape, dtype, exported[prefix + "weight"])
        decoded = packed.dequantize(
            dtype=dtype,
            scale=exported[prefix + "weight_scale"],
            double_scale=exported[prefix + "weight_scale_2"],
            block_sizes={-1: 16},
        )
    else:
        decoded = (
            exported[prefix + "weight"].float() * exported[prefix + "weight_scale"].float()
        ).to(dtype)
    torch.testing.assert_close(decoded, qdq, rtol=0, atol=0)
    assert torch.count_nonzero(decoded[0, :16]) == 0
