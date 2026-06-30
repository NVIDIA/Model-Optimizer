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

"""Quantized RNN."""

import contextlib
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import torch
import torch._VF as _VF
import torch.nn as nn
import torch.nn.functional as F

from ...tensor_quant import QUANT_DESC_8BIT_PER_TENSOR
from ...utils import is_torch_export_mode, multi_context, replace_function
from .quant_module import QuantModule, QuantModuleRegistry
from .tensor_quantizer import SequentialQuantizer, TensorQuantizer

_cell_call_map = {
    "RNN_TANH": _VF.rnn_tanh_cell,
    "RNN_RELU": _VF.rnn_relu_cell,
    "LSTM": _VF.lstm_cell,
    "GRU": _VF.gru_cell,
}
_layer_call_name_map = {
    "RNN_TANH": "rnn_tanh",
    "RNN_RELU": "rnn_relu",
    "LSTM": "lstm",
    "GRU": "gru",
}


class QuantRNNBase(QuantModule):
    """Base class for quantized RNN modules."""

    weight_quantizer: TensorQuantizer | SequentialQuantizer
    _enable_weight_quantization: bool
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    _functionals_to_replace: list[tuple[ModuleType, str, Callable]] = []

    @property
    def functionals_to_replace(self) -> Iterator[tuple[ModuleType, str, Callable]]:
        """Replace functions of packages on the fly."""
        return (
            (package, func_name, quantized_func)
            for package, func_name, quantized_func in self._functionals_to_replace
            if hasattr(package, func_name)
        )

    @property
    def all_input_quantizers_disabled(self):
        """Check if all input quantizer are disabled."""
        return all(not iq.is_enabled for iq in self._input_quantizers + self._proj_input_quantizers)

    @contextlib.contextmanager
    def quantize_weight(self):
        """Context in which ``self.weight`` is quantized."""
        self._enable_weight_quantization = True
        yield
        self._enable_weight_quantization = False

    @staticmethod
    def _get_quantized_weight_handler(weight_quantizer_name: str):
        def _get_quantized_weight(module: "QuantRNNBase", weight: torch.Tensor):
            if module._enable_weight_quantization:
                weight_quantizer = getattr(module, weight_quantizer_name)
                return weight_quantizer(weight)
            return weight

        return _get_quantized_weight

    def forward(self, input, *args, **kwargs):
        """Quantize the input and the weight before calling the original forward method."""
        contexts = (
            replace_function(package, func_name, quantized_func)
            for package, func_name, quantized_func in self.functionals_to_replace
        )
        if is_torch_export_mode():
            return super().forward(input, *args, **kwargs)
        with multi_context(
            self.quantize_weight(),
            *contexts,
        ):
            return super().forward(input, *args, **kwargs)

    def _setup(self):
        for name, _ in self.named_parameters():
            if name.startswith("weight"):
                # to be compatible with our current config, the name is some what weird
                # it would be weight_xxx_weight_quantizer
                weight_quantizer_name = name + "_weight_quantizer"
                self._register_temp_attribute(
                    weight_quantizer_name, TensorQuantizer(self.default_quant_desc_weight)
                )
                self._register_dynamic_attribute(
                    name, self._get_quantized_weight_handler(weight_quantizer_name)
                )
        # for cells
        self._register_temp_attribute("_input_quantizers", [])
        num_directions = 2 if self.bidirectional else 1
        # for projection layer if exists
        self._register_temp_attribute("_proj_input_quantizers", [])

        # the input quantizer is per cell (or per layer) based
        for layer in range(self.num_layers * num_directions):
            for direction in range(num_directions):
                suffix = "_reverse" if direction == 1 else ""
                name = f"layer_{layer}{suffix}_input_quantizer"
                quantizer = TensorQuantizer(self.default_quant_desc_input)
                self._register_temp_attribute(name, quantizer)
                self._input_quantizers.append(quantizer)
                if self.proj_size > 0:
                    name = f"proj_{layer}{suffix}_input_quantizer"
                    quantizer = TensorQuantizer(self.default_quant_desc_input)
                    self._register_temp_attribute(name, quantizer)
                    self._proj_input_quantizers.append(quantizer)
        self._register_temp_attribute("_enable_weight_quantization", False)


class QuantRNNFullBase(QuantRNNBase):
    """Quantized RNN with input quantizer."""

    def _disable_input_quantizers(self):
        for iq in self._input_quantizers + self._proj_input_quantizers:
            iq.disable()

    def _enable_input_quantizers(self):
        for iq in self._input_quantizers + self._proj_input_quantizers:
            iq.enable()

    def _disable_weight_quantizers(self):
        for name, module in self.named_modules():
            if name.endswith("weight_quantizer"):
                module.disable()

    def _enable_weight_quantizer(self):
        for name, module in self.named_modules():
            if name.endswith("weight_quantizer"):
                module.enable()

    def _setup(self):
        super()._setup()
        vf_rnn = VFRNNForward(
            self.mode,
            self.bidirectional,
            self.num_layers,
            self.proj_size > 0,
            self.bias,
            self._input_quantizers,
            self._proj_input_quantizers if self.proj_size > 0 else None,
            self.batch_first,
        )
        self._functionals_to_replace = [(_VF, _layer_call_name_map[self.mode], vf_rnn)]


class VFRNNForward:
    """Reimplement the _VF rnn calls with python to enable input quantizers.

    It's less efficient compared to original calls.
    """

    def __init__(
        self,
        mode: str,
        bidirectional: bool,
        num_layers: int,
        has_proj: bool,
        has_bias: bool,
        input_quantizers: list[TensorQuantizer],
        proj_input_quantizers: list[TensorQuantizer] | None = None,
        batch_first: bool | None = False,
    ):
        """Pre-construct necessary parameters for vf calls to reduce overhead.

        Refer to torch RNN modules for parameter information.
        """
        cell = _cell_call_map[mode]
        if has_proj:
            cell = lstm_cell_with_proj
        self.layer_forwards_const = (RNNLayerForward(cell, variable_len=False),)
        self.layer_forwards_variable = (RNNLayerForward(cell, variable_len=True),)
        if bidirectional:
            self.layer_forwards_const += (RNNLayerForward(cell, reverse=True, variable_len=False),)
            self.layer_forwards_variable += (
                RNNLayerForward(cell, reverse=True, variable_len=True),
            )
        self.num_directions = 2 if bidirectional else 1
        self.total_layers = num_layers * self.num_directions
        self.num_layers = num_layers
        self.is_lstm = mode == "LSTM"
        self.has_proj = has_proj

        self.num_weights_per_layer = self.num_cell_weights_per_layer = 4 if has_bias else 2
        if has_proj:
            self.num_weights_per_layer += 1

        self.input_quantizers = input_quantizers

        self.proj_input_quantizers = proj_input_quantizers
        self.batch_first = batch_first

    def forward(
        self,
        layer_forwards: tuple[Callable],
        input: torch.Tensor,
        flat_weights: list[torch.Tensor],
        hidden: torch.Tensor | tuple[torch.Tensor],
        dropout: float | None = 0,
        training: bool | None = True,
        batch_sizes: torch.Tensor | None = None,
    ):
        """This this the core implementation of vf rnn calls."""
        all_hiddens = []

        if self.is_lstm:
            hidden = list(zip(*hidden))
        if self.batch_first and batch_sizes is None:
            input = input.transpose(0, 1)

        for i in range(self.num_layers):
            all_output = []
            for j, layer_forward in enumerate(layer_forwards):
                l_i = i * self.num_directions + j
                # the flat_weights is a list of tensors, we need extract weights of each layer
                w_start = l_i * self.num_weights_per_layer
                w_end = w_start + self.num_weights_per_layer
                proj_input_quantizer = (
                    self.proj_input_quantizers[l_i] if self.proj_input_quantizers else None
                )

                hx, output = layer_forward(
                    input,
                    hidden[l_i],
                    flat_weights[w_start:w_end],
                    self.input_quantizers[l_i],
                    batch_sizes,
                    proj_input_quantizer=proj_input_quantizer,
                )

                all_hiddens.append(hx)
                all_output.append(output)

            input = torch.cat(all_output, -1)

            if dropout != 0 and i < self.num_layers - 1:
                input = F.dropout(input, p=dropout, training=training, inplace=False)

        if self.batch_first and batch_sizes is None:
            input = input.transpose(0, 1)

        if self.is_lstm:
            hn, cn = zip(*all_hiddens)
            all_hiddens = (torch.stack(hn, 0), torch.stack(cn, 0))
            output_tuple = (input, *all_hiddens)
        else:
            all_hiddens = torch.stack(all_hiddens, 0)
            output_tuple = (input, all_hiddens)

        return output_tuple

    def __call__(self, *args) -> tuple[torch.Tensor, torch.Tensor]:
        """Entry of vf calls.

        Original vf funcs are overloaded cpp funcs. Each has two different signatures,
        one accepts inputs with variable length (packed sequence), the other accepts constant length.
        The difference is that the fourth arg is a bool for constant length.
        """
        if isinstance(args[3], bool):
            # constant
            (
                input,
                hidden,
                flat_weights,
                bias,
                num_layers,
                dropout,
                training,
                bidirectional,
                batch_first,
            ) = args
            batch_sizes = None
            layer_forwards = self.layer_forwards_const
        else:
            # variable
            (
                input,
                batch_sizes,
                hidden,
                flat_weights,
                bias,
                num_layers,
                dropout,
                training,
                bidirectional,
            ) = args
            layer_forwards = self.layer_forwards_variable

        return self.forward(
            layer_forwards,
            input,
            flat_weights,
            hidden,
            dropout,
            training,
            batch_sizes,
        )


class RNNLayerForward:
    """A single layer of rnn modules."""

    def __init__(self, cell, reverse=False, variable_len=False):
        """Init the layer forward for different cells, directions, and inputs."""
        self.cell = cell
        self.reverse = reverse
        self.variable_len = variable_len
        if variable_len:
            if reverse:
                self.forward = get_quantized_rnn_layer_variable_len_reverse_forward(cell)
            else:
                self.forward = get_quantized_rnn_layer_variable_len_forward(cell)
        else:
            self.forward = get_quantized_rnn_layer_forward(cell, reverse=reverse)

    def __call__(
        self, input, hidden, weights, input_quantizer, batch_sizes=None, proj_input_quantizer=None
    ) -> Any:
        """Layer forward."""
        return self.forward(
            input,
            hidden,
            weights,
            input_quantizer,
            batch_sizes=batch_sizes,
            proj_input_quantizer=proj_input_quantizer,
        )


def lstm_cell_with_proj(input, hidden, *weights, proj_input_quantizer=None):
    """Currently the _VF.lstm_cell doesn't accept projected inputs. i.e. h_n and c_n must be same shape.

    This implementation is not optimized for cuda compared to _VF.lstm_cell, so we only use it when projection exists.
    """
    if len(weights) == 3:
        weight_ih, weight_hh, weight_hr = weights
        bias_ih = bias_hh = None
    else:
        weight_ih, weight_hh, bias_ih, bias_hh, weight_hr = weights
    hn, cn = hidden

    gates = F.linear(input, weight_ih, bias_ih) + F.linear(hn, weight_hh, bias_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cn) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    if proj_input_quantizer is not None:
        hy = proj_input_quantizer(hy)
    hy = F.linear(hy, weight_hr)
    return hy, cy


def quantized_cell_forward(
    cell, input, hidden, weights, input_quantizer, proj_input_quantizer=None
):
    """Call input quantizer before calling cell."""
    quant_input = input_quantizer(input)
    quant_hidden = (
        (input_quantizer(hidden[0]), input_quantizer(hidden[1]))
        if isinstance(hidden, tuple)
        else input_quantizer(hidden)
    )
    kwargs = {} if proj_input_quantizer is None else {"proj_input_quantizer": proj_input_quantizer}
    hidden = cell(quant_input, quant_hidden, *weights, **kwargs)
    return hidden


def get_quantized_rnn_layer_forward(cell, reverse=False):
    """Construct the forward call for different rnn cells.

    Note that batch_sizes is here for keeping a consistent signature with the forward of variable length.
    """

    def forward(
        input, hidden, weights, input_quantizer, batch_sizes=None, proj_input_quantizer=None
    ):
        seq_len = input.shape[0]
        steps = reversed(range(seq_len)) if reverse else range(seq_len)

        output = []
        for i in steps:
            hidden = quantized_cell_forward(
                cell,
                input[i],
                hidden,
                weights,
                input_quantizer,
                proj_input_quantizer=proj_input_quantizer,
            )
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.stack(output, 0)

        return hidden, output

    return forward


def get_quantized_rnn_layer_variable_len_forward(cell):
    """Construct the forward call for packed sequence."""

    def forward(input, hidden, weights, input_quantizer, batch_sizes, proj_input_quantizer=None):
        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
        for batch_size in batch_sizes:
            step_input = input[input_offset : input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)
            last_batch_size = batch_size
            hx = hidden[0] if flat_hidden else hidden
            hidden = quantized_cell_forward(
                cell,
                step_input,
                hx,
                weights,
                input_quantizer,
                proj_input_quantizer=proj_input_quantizer,
            )
            if flat_hidden:
                hidden = (hidden,)

            output.append(hidden[0])
        hiddens.append(hidden)
        hiddens.reverse()

        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].shape[0] == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)

        return hidden, output

    return forward


def get_quantized_rnn_layer_variable_len_reverse_forward(cell):
    """Construct the forward call for packed sequence in the reversed direction."""

    def forward(input, hidden, weights, input_quantizer, batch_sizes, proj_input_quantizer=None):
        output = []
        input_offset = input.shape[0]
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        hidden = tuple(h[: batch_sizes[-1]] for h in hidden)
        for i in reversed(range(len(batch_sizes))):
            batch_size = batch_sizes[i]
            inc = batch_size - last_batch_size
            if inc > 0:
                hidden = tuple(
                    torch.cat((h, ih[last_batch_size:batch_size]), 0)
                    for h, ih in zip(hidden, initial_hidden)
                )
            last_batch_size = batch_size
            step_input = input[input_offset - batch_size : input_offset]
            input_offset -= batch_size

            hx = hidden[0] if flat_hidden else hidden
            hidden = quantized_cell_forward(
                cell,
                step_input,
                hx,
                weights,
                input_quantizer,
                proj_input_quantizer=proj_input_quantizer,
            )
            if flat_hidden:
                hidden = (hidden,)
            output.append(hidden[0])

        output.reverse()
        output = torch.cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]
        return hidden, output

    return forward


QuantModuleRegistry.register({nn.RNN: "nn.RNN"})(QuantRNNFullBase)
QuantModuleRegistry.register({nn.LSTM: "nn.LSTM"})(QuantRNNFullBase)
QuantModuleRegistry.register({nn.GRU: "nn.GRU"})(QuantRNNFullBase)
