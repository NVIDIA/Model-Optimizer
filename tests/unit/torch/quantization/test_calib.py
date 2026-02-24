# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unittests for AWQ and SVDQuant"""

from functools import partial

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.quantization.quantize_common import get_awq_config

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.model_calib import (
    apply_pre_quant_scale_and_smooth,
    disable_pre_quant_scale_and_resmooth,
    sequential_calibrate,
)
from modelopt.torch.quantization.nn import TensorQuantizer


class _SimpleMLP(nn.Module):
    """Simple toy model."""

    def __init__(self, fi=16, f1=32, f2=16, fo=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fi, f1, bias=False),
            nn.ReLU(),
            nn.Linear(f1, f2, bias=False),
            nn.ReLU(),
            nn.Linear(f2, fo, bias=False),
        )

    def forward(self, x):
        for mod in self.net:
            if hasattr(mod, "_input_div"):
                x.div_(mod._input_div)
            x = mod(x)
        return x


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


# copied with modifications from https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py
@torch.no_grad()
def awq_lite_manual(module, x, quantizer):
    # w: co, ci
    # x: n, ci
    weight = module.weight
    w_max = get_weight_scale(weight, q_group_size=quantizer.block_sizes[-1])

    with torch.no_grad():
        org_out = module(x)

    x_max = get_act_scale(x)

    best_error = float("inf")
    best_ratio = -1
    best_scales = None

    n_grid = 20
    history = []

    org_weight = weight.clone()

    for ratio in range(n_grid):
        ratio = ratio * 1 / n_grid
        scales = (x_max.pow(ratio) / w_max.pow(1 - ratio)).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        module.weight.mul_(scales.view(1, -1))
        module.weight.data.copy_(quantizer(module.weight.data) / (scales.view(1, -1)))
        out = module(x)

        loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
        history.append(loss)
        is_best = loss < best_error
        if is_best:
            best_error = loss
            best_ratio = ratio
            best_scales = scales
        module.weight.data.copy_(org_weight)
    if best_ratio == -1:
        print(history)
        raise Exception
    best_scales = best_scales.view(-1)
    module.x_max = x_max
    module.w_max = w_max
    assert torch.isnan(best_scales).sum() == 0, best_scales
    return best_scales.detach()


# copied with modifications from https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_clip.py
@torch.no_grad()
def awq_clip_manual(module, x, quantizer, n_grid=20, max_shrink=0.5, n_sample_token=64):
    w = module.weight
    assert w.dim() == 2
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # x  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = quantizer.block_sizes[-1]
    x = x.view(-1, x.shape[-1])
    x = x.reshape(1, x.shape[0], -1, group_size)
    x = x[:, 0 :: x.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = w.shape[0]  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        x = x.to(w.device)
        org_out = (x * w).sum(dim=-1)  # co, n_token, n_group

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            # cur_w = torch.clamp(w, min_val, max_val)
            quantizer.amax = max_val.view(-1, 1)
            q_w = quantizer(w)
            cur_out = (x * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del q_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)

    del x
    del org_out
    torch.cuda.empty_cache()
    return best_max_val.view(-1, 1)


def forward_loop(model, dataloader):
    """Forward loop for calibration."""
    for data in dataloader:
        model(data)


def get_test_args(config):
    torch.manual_seed(0)
    model = _SimpleMLP()
    states = model.state_dict()
    model_ref = _SimpleMLP()
    model_ref.load_state_dict(states)

    dataloader = [torch.randn(2, 64, 16)]

    model = mtq.quantize(model, config, partial(forward_loop, dataloader=dataloader))

    x = dataloader[0].clone()
    ref_data = {}
    for i, mod in enumerate(model_ref.net):
        ref_data[i] = x
        x = mod(x)

    return model, model_ref, dataloader, ref_data


def _awq_lite_tester(model, model_ref, data):
    scale_dict = {}
    for i in [0, 2, 4]:
        scale_ref = awq_lite_manual(
            model_ref.net[i],
            data[i],
            TensorQuantizer(QuantizerAttributeConfig(num_bits=4, block_sizes={-1: 8})),
        )
        assert torch.allclose(model.net[i].awq_lite.best_scale, scale_ref)
        scale_dict[i] = scale_ref

    with torch.no_grad():
        model(data[0])

    return scale_dict


def apply_clip(model, clip_list):
    for module, clip in clip_list:
        module.weight_quantizer.amax = clip.view(-1, 1)


def _awq_clip_tester(model, model_ref, data, config):
    clip_list = []
    for i in [0, 2, 4]:
        clip = awq_clip_manual(
            model_ref.net[i],
            data[i],
            TensorQuantizer(QuantizerAttributeConfig(num_bits=4, block_sizes={-1: 8})),
        )
        assert torch.allclose(model.net[i].awq_clip.best_clip_val, clip)
        clip_list.append((model_ref.net[i], clip))

    mtq.replace_quant_module(model_ref)
    mtq.set_quantizer_by_cfg(model_ref, config["quant_cfg"])
    apply_clip(model_ref, clip_list)

    for i in [0, 2, 4]:
        assert torch.allclose(
            model.net[i].weight_quantizer(model.net[i].weight),
            model_ref.net[i].weight_quantizer(model_ref.net[i].weight),
        )

    with torch.no_grad():
        model(data[0])


def test_awq_lite():
    """Test awq_lite."""
    model, model_ref, dataloader, ref_data = get_test_args(get_awq_config("awq_lite"))
    _awq_lite_tester(model, model_ref, ref_data)


def test_awq_clip():
    """Test awq_clip."""
    config = get_awq_config("awq_clip")
    model, model_ref, dataloader, ref_data = get_test_args(config)
    _awq_clip_tester(model, model_ref, ref_data, config)


def test_awq_full():
    """Test awq."""
    config = get_awq_config("awq_full")
    model, model_ref, dataloader, ref_data = get_test_args(config)

    scale_dict = _awq_lite_tester(model, model_ref, ref_data)

    x = dataloader[0].clone()
    for i, mod in enumerate(model_ref.net):
        if i in scale_dict:
            with torch.no_grad():
                mod.weight.mul_(scale_dict[i].view(1, -1))
                x.div_(scale_dict[i])
                mod._input_div = scale_dict[i]
        ref_data[i] = x
        x = mod(x)

    _awq_clip_tester(model, model_ref, ref_data, config)

    out = model(dataloader[0])
    out_ref = model_ref(dataloader[0])
    assert torch.allclose(out, out_ref, atol=1e-6)


def test_awq_multi_batch():
    """Test awq multibatch."""
    # awq test
    model = _SimpleMLP()
    states = model.state_dict()

    # Multi batch test
    config = get_awq_config("awq_full")

    dataloader = [torch.randn(2, 16, 16) for _ in range(4)]
    dataloader_sb = [torch.cat(dataloader, dim=0)]

    model_s = _SimpleMLP()
    model_s.load_state_dict(states)
    model_s = mtq.quantize(model_s, config, partial(forward_loop, dataloader=dataloader_sb))

    model_m = _SimpleMLP()
    model_m.load_state_dict(states)
    model_m = mtq.quantize(model_m, config, partial(forward_loop, dataloader=dataloader))

    assert torch.allclose(model_s.net[0].awq_lite.best_scale, model_m.net[0].awq_lite.best_scale)
    assert torch.allclose(model_s.net[2].awq_lite.best_scale, model_m.net[2].awq_lite.best_scale)
    assert torch.allclose(model_s.net[4].awq_lite.best_scale, model_m.net[4].awq_lite.best_scale)


def test_padded_awq():
    """Test awq when padding is needed."""
    model = _SimpleMLP(f1=13, f2=19)
    config = get_awq_config("awq_full")
    model = mtq.quantize(model, config, partial(forward_loop, dataloader=[torch.randn(2, 16, 16)]))
    model(torch.randn(2, 16, 16))


def test_smoothquant_enable_disable():
    torch.manual_seed(1234)
    model = _SimpleMLP()
    cal_data = [torch.randn(2, 16, 16)]
    input = torch.randn(2, 16, 16)

    model_no_sq = _SimpleMLP()
    model_no_sq.load_state_dict(model.state_dict())

    model = mtq.quantize(model, mtq.INT8_SMOOTHQUANT_CFG, partial(forward_loop, model, cal_data))
    model_no_sq = mtq.quantize(
        model_no_sq, mtq.INT8_DEFAULT_CFG, partial(forward_loop, model_no_sq, cal_data)
    )

    out_sq = model(input)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            disable_pre_quant_scale_and_resmooth(module)
            assert module.input_quantizer.pre_quant_scale is None

    out_no_sq = model_no_sq(input)
    out = model(input)
    assert torch.allclose(out, out_no_sq)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            apply_pre_quant_scale_and_smooth(module)
            assert module.input_quantizer.pre_quant_scale is not None

    out = model(input)
    assert torch.allclose(out, out_sq)


def test_postprocess_amax():
    model = _SimpleMLP()

    mtq.quantize(
        model, mtq.INT8_DEFAULT_CFG, partial(forward_loop, dataloader=[torch.randn(2, 16, 16)])
    )

    model = mtq.postprocess_amax(model, "*input_quantizer", lambda amax: torch.clamp(amax, max=0.5))

    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and name.endswith("input_quantizer"):
            assert module.amax.amax() <= 0.5


def test_svdquant_lora_weights():
    model = _SimpleMLP(64, 64, 64, 64)

    quant_config = mtq.INT8_SMOOTHQUANT_CFG.copy()
    quant_config["algorithm"] = "svdquant"

    mtq.quantize(model, quant_config, partial(forward_loop, dataloader=[torch.randn(2, 64, 64)]))

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            assert module.weight_quantizer.svdquant_lora_a is not None
            assert module.weight_quantizer.svdquant_lora_b is not None

            lora_residual = (
                module.weight_quantizer.svdquant_lora_b @ module.weight_quantizer.svdquant_lora_a
            )
            assert lora_residual.shape == module.weight.shape


def test_sequential_calibrate_support_gate():
    class _UnsupportedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.linear(x)

    model = _UnsupportedModel()

    with (
        torch.no_grad(),
        pytest.raises(ValueError, match="Sequential calibration requires a model"),
    ):
        sequential_calibrate(
            model,
            forward_loop=lambda m: m(torch.randn(2, 4)),
            calib_func=lambda layer, loop: loop(layer),
        )


def test_sequential_calibrate_propagates_inputs_without_replaying_full_model(monkeypatch):
    from modelopt.torch.quantization.utils import LayerActivationCollector

    class _ToyLayer(nn.Module):
        def __init__(self, scale: float, bias: float):
            super().__init__()
            self.scale = scale
            self.bias = bias

        def forward(self, hidden_states):
            return hidden_states * self.scale + self.bias

    class _ToyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    _ToyLayer(scale=2.0, bias=1.0),
                    _ToyLayer(scale=0.5, bias=3.0),
                    _ToyLayer(scale=1.0, bias=-2.0),
                ]
            )

        def forward(self, hidden_states):
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            return hidden_states

    model = _ToyDecoder()
    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(lambda m: hasattr(m, "layers"), lambda m: m.layers)],
    )
    batches = [
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
    ]

    forward_loop_calls = 0

    def _forward_loop(m):
        nonlocal forward_loop_calls
        forward_loop_calls += 1
        for batch in batches:
            m(batch)

    observed_layer_inputs = []

    def _calib_func(layer, layer_forward_loop):
        captured = []

        def _pre_hook(_module, args):
            captured.append(args[0].clone())

        handle = layer.register_forward_pre_hook(_pre_hook)
        try:
            layer_forward_loop(layer)
        finally:
            handle.remove()
        observed_layer_inputs.append(captured)

    sequential_calibrate(model, _forward_loop, _calib_func)

    assert forward_loop_calls == len(model.layers)
    assert len(observed_layer_inputs) == len(model.layers)
    for layer_inputs in observed_layer_inputs:
        assert len(layer_inputs) == len(batches)

    expected_layer_0 = batches
    expected_layer_1 = [model.layers[0](batch) for batch in batches]
    expected_layer_2 = [model.layers[1](batch) for batch in expected_layer_1]

    for observed, expected in zip(observed_layer_inputs[0], expected_layer_0):
        assert torch.allclose(observed, expected)
    for observed, expected in zip(observed_layer_inputs[1], expected_layer_1):
        assert torch.allclose(observed, expected)
    for observed, expected in zip(observed_layer_inputs[2], expected_layer_2):
        assert torch.allclose(observed, expected)


def test_sequential_calibrate_uses_next_layer_hook_without_replaying_full_model(monkeypatch):
    from modelopt.torch.quantization.utils import LayerActivationCollector

    class _ToyLayer(nn.Module):
        def __init__(self, scale: float, bias: float):
            super().__init__()
            self.scale = scale
            self.bias = bias

        def forward(self, hidden_states):
            return hidden_states * self.scale + self.bias

    class _ToyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    _ToyLayer(scale=2.0, bias=1.0),
                    _ToyLayer(scale=0.5, bias=3.0),
                    _ToyLayer(scale=1.0, bias=-2.0),
                ]
            )

        def forward(self, hidden_states):
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            return hidden_states

    model = _ToyDecoder()
    batches = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]
    forward_loop_calls = 0

    def _forward_loop(m):
        nonlocal forward_loop_calls
        forward_loop_calls += 1
        for batch in batches:
            m(batch)

    def _supported(_model):
        return True

    def _build_hook(_model):
        def _hook(prev_layer, cached_inputs):
            next_inputs = []
            for args, kwargs in cached_inputs:
                hidden_states = prev_layer(*args, **kwargs)
                next_inputs.append(((hidden_states, *args[1:]), kwargs))
            return next_inputs

        return _hook

    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(lambda m: hasattr(m, "layers"), lambda m: m.layers)],
    )
    monkeypatch.setattr(
        LayerActivationCollector,
        "_next_layer_input_support",
        [(_supported, _build_hook)],
    )

    observed_layer_inputs = []

    def _calib_func(layer, layer_forward_loop):
        captured = []

        def _pre_hook(_module, args):
            captured.append(args[0].clone())

        handle = layer.register_forward_pre_hook(_pre_hook)
        try:
            layer_forward_loop(layer)
        finally:
            handle.remove()
        observed_layer_inputs.append(captured)

    sequential_calibrate(model, _forward_loop, _calib_func)

    assert forward_loop_calls == 1
    assert len(observed_layer_inputs) == len(model.layers)
