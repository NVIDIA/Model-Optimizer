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

"""Quantize a model with quantizers placed on its exported graph by the ONNX path's placement rules.

Module-based quantization only quantizes the inputs of quantizable modules, so functional ops between
modules (residual adds, squeeze-and-excitation multiplies, pooling) stay unquantized and TensorRT
cannot fuse them. Here the model is exported with :func:`torch.export.export`, the ONNX quantization
path decides the Q/DQ placement on its ONNX export, and :class:`TensorQuantizer` modules are inserted
into the exported graph at the same places, one quantizer per quantized tensor.
"""

from __future__ import annotations

import ast
import types
from typing import TYPE_CHECKING, Any

import torch
from torch import fx

from .model_quant import quantize
from .nn import TensorQuantizer

if TYPE_CHECKING:
    import onnx

    from modelopt.onnx.quantization.placement import QdqPlacement
    from modelopt.torch.opt.searcher import ForwardLoop

__all__ = ["quantize_graph"]

_SUPPORTED_ALGORITHMS = (None, "max")


def quantize_graph(
    model: torch.nn.Module,
    config: dict[str, Any],
    args: tuple,
    forward_loop: ForwardLoop | None = None,
    quantize_mode: str = "int8",
    **placement_kwargs,
) -> fx.GraphModule:
    """Quantize ``model`` with quantizers placed on its graph as ModelOpt's ONNX path places Q/DQ.

    The model is captured with ``torch.export`` in its current train/eval mode and with the shapes of
    ``args``. Quantizers are named so that ``config`` patterns apply as for module-based quantization:
    ``<module>.weight_quantizer`` for weights, ``<module>.input_quantizer`` or
    ``<module>.<tensor>_input_quantizer`` for activations.

    Example:

    .. code-block:: python

        gm = mtq.quantize_graph(model, mtq.INT8_DEFAULT_CFG, (images,), forward_loop)

    Args:
        config: A quantization config as for :func:`quantize`. Only ``"max"`` calibration is supported.
        args: Example positional inputs for ``torch.export``.
        quantize_mode: Placement rules to use: ``"int8"`` or ``"fp8"``.
        placement_kwargs: Extra placement options for :func:`modelopt.onnx.quantization.quantize`,
            e.g. ``nodes_to_exclude`` (ONNX node names).

    Returns:
        The exported graph module with quantizers inserted and calibrated.
    """
    # The ONNX path is an optional dependency of the torch path.
    from modelopt.onnx.quantization.placement import find_qdq_placement

    algorithm = config.get("algorithm")
    method = algorithm.get("method") if isinstance(algorithm, dict) else algorithm
    if method not in _SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Graph placement supports calibration {_SUPPORTED_ALGORITHMS}, not {method!r}."
        )

    exported = torch.export.export(model, args, strict=False)
    # The exporter decomposes the graph further before translating it; its decomposed program is
    # the one whose node names the ONNX model refers to.
    program = torch.onnx.export(exported, dynamo=True, verbose=False)
    onnx_model = program.model_proto
    placement = find_qdq_placement(onnx_model, quantize_mode, **placement_kwargs)
    graph_module = program.exported_program.module()
    _fix_train_mode(graph_module, model.training)
    _insert_quantizers(graph_module, onnx_model, placement)
    return quantize(graph_module, config, forward_loop)


def _fix_train_mode(graph_module: fx.GraphModule, training: bool) -> None:
    # Train/eval behaviour (batch norm, dropout) is baked into the exported graph; allow setting
    # only the mode it was exported in, which calibration does.
    def train(self, mode: bool = True):
        if mode != training:
            raise NotImplementedError(
                f"The graph was exported in {'train' if training else 'eval'} mode and cannot change mode; "
                "export the model again in the other mode."
            )
        return torch.nn.Module.train(self, mode)

    graph_module.train = types.MethodType(train, graph_module)
    graph_module.eval = types.MethodType(torch.nn.Module.eval, graph_module)
    graph_module.train(training)


def _insert_quantizers(
    graph_module: fx.GraphModule, onnx_model: onnx.ModelProto, placement: QdqPlacement
) -> None:
    graph = graph_module.graph
    fx_nodes = {node.name: node for node in graph.nodes}
    params = {node.target: node for node in graph.nodes if node.op == "get_attr"}
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    # Graph inputs are named after the forward arguments in ONNX but may be renamed in FX.
    fx_nodes.update(
        zip((value.name for value in onnx_model.graph.input), placeholders, strict=True)
    )
    onnx_nodes = {node.name: node for node in onnx_model.graph.node}
    initializer_numel = {
        init.name: int(torch.Size(init.dims).numel()) for init in onnx_model.graph.initializer
    }

    for tensor in placement.tensors.values():
        consumers = list(
            dict.fromkeys(fx_nodes[_fx_node_name(onnx_nodes[name])] for name, _ in tensor.consumers)
        )
        if tensor.name in fx_nodes:
            source = fx_nodes[tensor.name]
        elif tensor.name in params:
            source = params[tensor.name]
        elif tensor.is_constant:
            source = _find_folded_param(tensor.name, consumers, initializer_numel[tensor.name])
        else:
            raise ValueError(
                f"Quantized tensor '{tensor.name}' is internal to the ONNX decomposition of "
                f"{[c.name for c in consumers]} and has no counterpart in the torch graph."
            )

        consumers = list(dict.fromkeys(_reader_of(source, consumer) for consumer in consumers))
        name = _quantizer_name(graph_module, source, consumers[0], tensor.axis is not None)
        graph_module.add_submodule(name, TensorQuantizer())
        with graph.inserting_after(placeholders[-1] if source.op == "placeholder" else source):
            quantized = graph.call_module(name, (source,))
        for consumer in consumers:
            consumer.replace_input_with(source, quantized)

    graph.lint()
    graph_module.recompile()


def _fx_node_name(onnx_node: onnx.NodeProto) -> str:
    # The exporter records the FX node an ONNX node was translated from as the last name scope.
    metadata = {prop.key: prop.value for prop in onnx_node.metadata_props}
    return ast.literal_eval(metadata["pkg.torch.onnx.name_scopes"])[-1]


def _reader_of(source: fx.Node, consumer: fx.Node) -> fx.Node:
    # An ONNX node fused from several FX nodes (e.g. Conv with a folded BatchNorm) records the last
    # of them; the FX node reading ``source`` is then one of its ancestors.
    pending, seen = [consumer], set()
    while pending:
        node = pending.pop()
        if source in node.all_input_nodes:
            return node
        seen.add(node)
        pending.extend(n for n in node.all_input_nodes if n not in seen and n.op == "call_function")
    raise ValueError(f"No FX node at or above '{consumer.name}' reads '{source.name}'.")


def _find_folded_param(tensor_name: str, consumers: list[fx.Node], numel: int) -> fx.Node:
    # The exporter folds parameter transforms (e.g. the transposed weight of a batched linear) into
    # new initializers; the parameter is the consumer's only parameter input of the same size.
    candidates = {
        arg
        for consumer in consumers
        for arg in consumer.all_input_nodes
        if arg.op == "get_attr" and arg.meta["val"].numel() == numel
    }
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot map quantized ONNX constant '{tensor_name}' to a parameter of "
            f"{[c.name for c in consumers]}: candidates {[c.target for c in candidates]}."
        )
    return candidates.pop()


def _quantizer_name(
    graph_module: fx.GraphModule, source: fx.Node, first_consumer: fx.Node, per_channel: bool
) -> str:
    if source.op == "get_attr":
        owner, _, attr = source.target.rpartition(".")
        suffix = "weight_quantizer" if per_channel else "input_quantizer"
        leaf = suffix if attr == "weight" and per_channel else f"{attr}_{suffix}"
    else:
        stack = first_consumer.meta.get("nn_module_stack", {})
        owner = next(reversed(stack.values()))[0] if stack else ""
        # The activation input of a module's own op (e.g. a Conv2d's conv) is its input_quantizer.
        own_op = any(
            arg.op == "get_attr" and arg.target.startswith(f"{owner}.")
            for arg in first_consumer.all_input_nodes
        )
        leaf = "input_quantizer" if own_op else f"{source.name}_input_quantizer"
        if own_op and _has_submodule(graph_module, _join(owner, leaf)):
            leaf = f"{source.name}_input_quantizer"
    return _join(owner, leaf)


def _join(owner: str, leaf: str) -> str:
    return f"{owner}.{leaf}" if owner else leaf


def _has_submodule(module: torch.nn.Module, name: str) -> bool:
    try:
        module.get_submodule(name)
    except AttributeError:
        return False
    return True
