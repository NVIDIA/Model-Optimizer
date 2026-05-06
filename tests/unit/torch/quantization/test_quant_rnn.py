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

"""Tests of QuantRNN module."""

import copy

import pytest
import torch
import torch.nn as nn

from modelopt.torch.quantization import tensor_quant
from modelopt.torch.quantization.conversion import set_quantizer_attributes_partial
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.nn.modules.quant_rnn import VFRNNForward

SEQ_LEN = 3
BATCH_SIZE = 2
NUM_LAYERS = 2
HIDDEN_SIZE = 8
INPUT_SHAPE = [SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE]


class TestQuantRNN:
    @pytest.mark.parametrize(
        ("original_cls", "bidirectional", "bias"),
        [
            (nn.RNN, True, True),
            (nn.RNN, False, False),
            (nn.LSTM, True, True),
            (nn.LSTM, False, False),
            (nn.GRU, True, True),
            (nn.GRU, False, False),
        ],
    )
    def test_no_quant(self, original_cls, bidirectional, bias):
        rnn_object = original_cls(
            HIDDEN_SIZE, HIDDEN_SIZE, NUM_LAYERS, bidirectional=bidirectional, bias=bias
        )
        rnn_object_original = copy.deepcopy(rnn_object)
        quant_rnn_object = QuantModuleRegistry.convert(rnn_object)
        rnn_object.eval()
        rnn_object_original.eval()
        set_quantizer_attributes_partial(quant_rnn_object, lambda name: True, {"enable": False})

        assert torch.allclose(
            quant_rnn_object.weight_ih_l0, rnn_object_original.weight_ih_l0, atol=1e-6
        )
        assert torch.allclose(
            quant_rnn_object.weight_hh_l0, rnn_object_original.weight_hh_l0, atol=1e-6
        )

        test_input = torch.randn(INPUT_SHAPE)

        out1 = quant_rnn_object(test_input)[0]
        out2 = rnn_object_original(test_input)[0]
        assert torch.allclose(out1, out2, atol=1e-6)

    @pytest.mark.parametrize(
        ("original_cls", "bidirectional", "bias"),
        [
            (nn.RNN, True, True),
            (nn.RNN, False, False),
            (nn.LSTM, True, True),
            (nn.LSTM, False, False),
            (nn.GRU, True, True),
            (nn.GRU, False, False),
        ],
    )
    def test_no_quant_packed_sequence(self, original_cls, bidirectional, bias):
        rnn_object = original_cls(
            HIDDEN_SIZE, HIDDEN_SIZE, NUM_LAYERS, bidirectional=bidirectional, bias=bias
        )
        rnn_object_original = copy.deepcopy(rnn_object)
        quant_rnn_object = QuantModuleRegistry.convert(rnn_object)
        rnn_object.eval()
        rnn_object_original.eval()
        set_quantizer_attributes_partial(quant_rnn_object, lambda name: True, {"enable": False})

        assert torch.allclose(
            quant_rnn_object.weight_ih_l0, rnn_object_original.weight_ih_l0, atol=1e-6
        )
        assert torch.allclose(
            quant_rnn_object.weight_hh_l0, rnn_object_original.weight_hh_l0, atol=1e-6
        )

        test_input = [
            torch.randn([INPUT_SHAPE[0] - 1, INPUT_SHAPE[2]]),
            torch.randn([INPUT_SHAPE[0], INPUT_SHAPE[2]]),
        ]
        test_input = torch.nn.utils.rnn.pack_sequence(test_input, enforce_sorted=False)

        out1 = quant_rnn_object(test_input)[0]
        out2 = rnn_object_original(test_input)[0]
        assert torch.allclose(out1[0], out2[0], atol=1e-6)

    @pytest.mark.parametrize(
        ("original_cls", "bidirectional", "bias"),
        [
            (nn.LSTM, True, True),
            (nn.LSTM, False, False),
        ],
    )
    def test_no_quant_proj(self, original_cls, bidirectional, bias):
        rnn_object = original_cls(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            NUM_LAYERS,
            bidirectional=bidirectional,
            bias=bias,
            proj_size=4,
        )
        rnn_object_original = copy.deepcopy(rnn_object)
        quant_rnn_object = QuantModuleRegistry.convert(rnn_object)

        set_quantizer_attributes_partial(quant_rnn_object, lambda name: True, {"enable": False})

        test_input = torch.randn(INPUT_SHAPE)

        out1 = quant_rnn_object(test_input)[0]
        out2 = rnn_object_original(test_input)[0]
        assert torch.allclose(out1, out2, atol=1e-6)

    @pytest.mark.parametrize(
        ("original_cls", "bidirectional"),
        [
            (nn.RNN, True),
            (nn.RNN, False),
            (nn.LSTM, True),
            (nn.LSTM, False),
            (nn.GRU, True),
            (nn.GRU, False),
        ],
    )
    def test_no_quant_batch_first(self, original_cls, bidirectional):
        rnn_object = original_cls(
            HIDDEN_SIZE, HIDDEN_SIZE, NUM_LAYERS, bidirectional=bidirectional, batch_first=True
        )
        rnn_object_original = copy.deepcopy(rnn_object)
        quant_rnn_object = QuantModuleRegistry.convert(rnn_object)

        set_quantizer_attributes_partial(quant_rnn_object, lambda name: True, {"enable": False})

        test_input = torch.randn([INPUT_SHAPE[1], INPUT_SHAPE[0], INPUT_SHAPE[2]])

        out1 = quant_rnn_object(test_input)[0]
        out2 = rnn_object_original(test_input)[0]
        assert torch.allclose(out1, out2, atol=1e-6)

    @pytest.mark.parametrize(
        ("original_cls", "bidirectional"),
        [
            (nn.RNN, True),
            (nn.RNN, False),
            (nn.LSTM, True),
            (nn.LSTM, False),
            (nn.GRU, True),
            (nn.GRU, False),
        ],
    )
    @torch.no_grad()
    def test_fake_quant_per_tensor(self, original_cls, bidirectional):
        rnn_object = original_cls(
            HIDDEN_SIZE, HIDDEN_SIZE, NUM_LAYERS, bidirectional=bidirectional, bias=True
        )
        rnn_object_original = copy.deepcopy(rnn_object)
        quant_rnn_object = QuantModuleRegistry.convert(rnn_object)
        set_quantizer_attributes_partial(quant_rnn_object, lambda name: True, {"axis": None})
        quant_rnn_object._disable_input_quantizers()

        for name, weight in rnn_object_original.named_parameters():
            if name.startswith("weight"):
                quant_weight = tensor_quant.fake_tensor_quant(weight, torch.amax(torch.abs(weight)))
                weight.copy_(quant_weight)

        test_input = torch.randn(INPUT_SHAPE)

        out1 = quant_rnn_object(test_input)[0]
        out2 = rnn_object_original(test_input)[0]
        assert torch.allclose(out1, out2, atol=1e-6)

    @pytest.mark.parametrize(
        ("original_cls", "bidirectional"),
        [
            (nn.RNN, True),
            (nn.RNN, False),
            (nn.LSTM, True),
            (nn.LSTM, False),
            (nn.GRU, True),
            (nn.GRU, False),
        ],
    )
    def test_fake_quant_per_channel(self, original_cls, bidirectional):
        rnn_object = original_cls(HIDDEN_SIZE, HIDDEN_SIZE, NUM_LAYERS, bidirectional=bidirectional)
        rnn_object_original = copy.deepcopy(rnn_object)
        quant_rnn_object = QuantModuleRegistry.convert(rnn_object)
        set_quantizer_attributes_partial(quant_rnn_object, lambda name: True, {"axis": (0)})
        quant_rnn_object._disable_input_quantizers()

        for name, weight in rnn_object_original.named_parameters():
            if name.startswith("weight"):
                quant_weight = tensor_quant.fake_tensor_quant(
                    weight, weight.abs().amax(dim=1, keepdim=True)
                )
                setattr(rnn_object_original, name, nn.Parameter(quant_weight))

        test_input = torch.randn(INPUT_SHAPE)

        out1 = quant_rnn_object(test_input)[0]
        out2 = rnn_object_original(test_input)[0]
        assert torch.allclose(out1, out2, atol=1e-6)

    @pytest.mark.parametrize(
        ("original_cls", "bidirectional"),
        [
            (nn.LSTM, True),
            (nn.LSTM, False),
        ],
    )
    @torch.no_grad()
    def test_input_quant_per_tensor(self, original_cls, bidirectional):
        rnn_object = original_cls(
            HIDDEN_SIZE, HIDDEN_SIZE, NUM_LAYERS, bidirectional=bidirectional, bias=True
        )
        quant_rnn_object = QuantModuleRegistry.convert(rnn_object)
        set_quantizer_attributes_partial(quant_rnn_object, lambda name: True, {"axis": None})
        quant_rnn_object._disable_weight_quantizers()

        num_directions = 2 if bidirectional else 1
        input_quantizers = (
            [lambda x: tensor_quant.fake_tensor_quant(x, torch.amax(torch.abs(x)))]
            * num_directions
            * NUM_LAYERS
        )
        vfrnn = VFRNNForward(
            quant_rnn_object.mode, bidirectional, NUM_LAYERS, False, True, input_quantizers
        )

        test_input = torch.randn(INPUT_SHAPE)

        out1 = quant_rnn_object(test_input)[0]

        hidden = (
            torch.zeros(num_directions * NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE),
            torch.zeros(num_directions * NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE),
        )
        out2 = vfrnn(
            test_input,
            hidden,
            quant_rnn_object._flat_weights,
            True,
            NUM_LAYERS,
            0,
            True,
            bidirectional,
            False,
        )[0]
        assert torch.allclose(out1, out2, atol=1e-6)
