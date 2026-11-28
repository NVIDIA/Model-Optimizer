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
import math
import operator
import types
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F
from torch import fx

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.opt.conversion import ModeloptStateManager, apply_mode
from modelopt.torch.opt.mode import ModeDescriptor

from .mode import QuantizeModeRegistry
from .model_quant import quantize
from .nn import TensorQuantizer
from .utils import replace_function

if TYPE_CHECKING:
    import onnx

    from modelopt.onnx.quantization.placement import QdqPlacement
    from modelopt.torch.opt.searcher import ForwardLoop

__all__ = ["GraphPlacementConfig", "quantize_graph"]

_SUPPORTED_ALGORITHMS = (None, "max")
_MODE = "quantize_graph_placement"

aten = torch.ops.aten
# Ops whose train/eval behaviour is an argument in the exported graph: op -> index of the flag.
_TRAINING_ARG = {
    aten.batch_norm.default: 5,
    aten.dropout.default: 2,
    aten.feature_dropout.default: 2,
    aten.alpha_dropout.default: 2,
    aten.feature_alpha_dropout.default: 2,
}


class GraphPlacementConfig(ModeloptBaseConfig):
    """Configuration of the graph placement mode."""

    quantize_mode: Literal["int8", "fp8"] = ModeloptField(
        default="int8",
        title="Placement rules",
        description="The ONNX path's quantize mode whose Q/DQ placement rules are applied.",
    )
    placement_kwargs: dict[str, Any] = ModeloptField(
        default={},
        title="Placement options",
        description="Extra options for modelopt.onnx.quantization.quantize, e.g. nodes_to_exclude.",
    )
    dynamic_batch: bool = ModeloptField(
        default=False,
        title="Dynamic batch",
        description="Export with a dynamic first dimension of every input.",
    )


def quantize_graph(
    model: torch.nn.Module,
    config: dict[str, Any],
    args: tuple[torch.Tensor, ...],
    forward_loop: ForwardLoop | None = None,
    quantize_mode: str = "int8",
    dynamic_batch: bool = False,
    **placement_kwargs,
) -> fx.GraphModule:
    """Quantize ``model`` with quantizers placed on its graph as ModelOpt's ONNX path places Q/DQ.

    The model is captured with ``torch.export`` in eval mode and with the shapes of ``args``.
    Quantizers are named so that ``config`` patterns apply as for module-based quantization:
    ``<module>.weight_quantizer`` for weights, ``<module>.input_quantizer`` or
    ``<module>.<tensor>_input_quantizer`` for activations.

    The returned graph module shares the parameters of ``model``. Its ``train()`` and ``eval()``
    switch batch norm and dropout; other train/eval differences in Python code, e.g. stochastic
    depth or attention dropout, keep their eval behaviour. Save and restore it with
    :meth:`mto.save<modelopt.torch.opt.conversion.save>` and
    :meth:`mto.restore<modelopt.torch.opt.conversion.restore>` on the original model; restore
    exports the model again.

    Example:

    .. code-block:: python

        gm = mtq.quantize_graph(model, mtq.INT8_DEFAULT_CFG, (images,), forward_loop)

    Args:
        config: A quantization config as for :func:`quantize`. Only ``"max"`` calibration is supported.
        args: Example positional tensor inputs for ``torch.export``.
        quantize_mode: Placement rules to use: ``"int8"`` or ``"fp8"``.
        dynamic_batch: Export with a dynamic first dimension of every input, so the graph runs any
            batch size; the example batch size must be at least 2. The ONNX path then places Q/DQ
            as for a dynamic-batch ONNX model.
        placement_kwargs: Extra placement options for :func:`modelopt.onnx.quantization.quantize`,
            e.g. ``nodes_to_exclude`` (ONNX node names).

    Returns:
        The exported graph module with quantizers inserted and calibrated.
    """
    algorithm = config.get("algorithm")
    method = algorithm.get("method") if isinstance(algorithm, dict) else algorithm
    if method not in _SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Graph placement supports calibration {_SUPPORTED_ALGORITHMS}, not {method!r}."
        )
    if not all(isinstance(arg, torch.Tensor) for arg in args):
        raise ValueError("Graph placement supports only positional tensor inputs.")

    placement_config = {
        "quantize_mode": quantize_mode,
        "placement_kwargs": placement_kwargs,
        "dynamic_batch": dynamic_batch,
    }
    graph_module = apply_mode(
        model,
        [(_MODE, placement_config)],
        registry=QuantizeModeRegistry,
        mode_kwargs={"args": args},
    )
    return quantize(graph_module, config, forward_loop)


@QuantizeModeRegistry.register_mode
class GraphPlacementModeDescriptor(ModeDescriptor):
    """Mode that exports the model and inserts quantizers where the ONNX path places Q/DQ."""

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return _MODE

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return GraphPlacementConfig

    @property
    def convert(self):
        """The mode's entrypoint for converting a model."""
        return _convert

    @property
    def restore(self):
        """The mode's entrypoint for restoring a model."""
        return _restore


def _convert(
    model: torch.nn.Module, config: GraphPlacementConfig, args: tuple[torch.Tensor, ...]
) -> tuple[fx.GraphModule, dict[str, Any]]:
    # The ONNX path is an optional dependency of the torch path.
    from modelopt.onnx.quantization.placement import find_qdq_placement

    exported = _export(model, args, config.dynamic_batch)
    # The exporter decomposes the graph further before translating it; the ONNX model refers to
    # the nodes of its decomposed program.
    program = torch.onnx.export(exported, dynamo=True, verbose=False)
    placement = find_qdq_placement(
        program.model_proto, config.quantize_mode, **config.placement_kwargs
    )
    graph_module = exported.module()
    insertions = _map_placement(
        graph_module,
        id(exported.graph),
        program.exported_program.graph,
        program.model_proto,
        placement,
    )
    _insert_quantizers(graph_module, insertions)
    metadata = {
        "inputs": [(tuple(arg.shape), str(arg.dtype).removeprefix("torch.")) for arg in args],
        "insertions": insertions,
    }
    return _replace_model(model, graph_module), metadata


def _restore(
    model: torch.nn.Module, config: GraphPlacementConfig, metadata: dict[str, Any]
) -> fx.GraphModule:
    device = next(model.parameters(), torch.empty(0)).device
    args = tuple(
        torch.zeros(shape, dtype=getattr(torch, dtype), device=device)
        for shape, dtype in metadata["inputs"]
    )
    graph_module = _export(model, args, config.dynamic_batch).module()
    _insert_quantizers(graph_module, metadata["insertions"])
    return _replace_model(model, graph_module)


def _export(
    model: torch.nn.Module, args: tuple, dynamic_batch: bool
) -> torch.export.ExportedProgram:
    # Place on the deployed (eval) graph. The ONNX path quantizes the matmuls inside attention, so
    # attention is traced from its math form.
    modes = {module: module.training for module in model.modules()}
    # Non-strict export puts copies of plain attributes back on the model; keep the modelopt state
    # that the mode's state manager is recording into.
    state_keys = (ModeloptStateManager._state_key, ModeloptStateManager._state_version_key)
    state = {key: vars(model)[key] for key in state_keys if key in vars(model)}
    model.eval()
    try:
        with replace_function(F, "scaled_dot_product_attention", _sdpa_math):
            batch = torch.export.Dim.DYNAMIC
            dynamic_shapes = tuple({0: batch} for _ in args) if dynamic_batch else None
            return torch.export.export(model, args, dynamic_shapes=dynamic_shapes, strict=False)
    finally:
        for module, training in modes.items():
            module.training = training
        vars(model).update(state)


def _sdpa_math(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
):
    # The reference implementation of F.scaled_dot_product_attention from its documentation.
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)
    scale = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    scores = query @ key.transpose(-2, -1) * scale
    if is_causal:
        attn_mask = torch.ones(
            query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
        ).tril()
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        scores = scores.masked_fill(~attn_mask, float("-inf"))
    elif attn_mask is not None:
        scores = scores + attn_mask
    weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0:
        weights = torch.dropout(weights, dropout_p, train=True)
    return weights @ value


def _replace_model(model: torch.nn.Module, graph_module: fx.GraphModule) -> fx.GraphModule:
    # The graph module replaces the model, so it carries the modelopt state from here on.
    if ModeloptStateManager.is_converted(model, is_root=True):
        ModeloptStateManager.transfer_state_dict(model, graph_module)
    graph_module.train = types.MethodType(_train, graph_module)
    graph_module.eval = types.MethodType(torch.nn.Module.eval, graph_module)
    graph_module.train(model.training)
    return graph_module


def _train(self: fx.GraphModule, mode: bool = True) -> fx.GraphModule:
    for node in self.graph.nodes:
        index = _TRAINING_ARG.get(node.target)
        if index is None:
            continue
        if node.target == aten.batch_norm.default:
            if node.args[3] is None:  # no running stats: batch statistics in both modes
                continue
            if mode and node.args[6] == 0.0:
                raise NotImplementedError(
                    f"'{node.name}' was exported without momentum (cumulative moving average), "
                    "which the exported graph cannot train."
                )
        node.update_arg(index, mode)
    self.recompile()
    return torch.nn.Module.train(self, mode)


def _map_placement(
    graph_module: fx.GraphModule,
    graph_id: int,
    decomposed: fx.Graph,
    onnx_model: onnx.ModelProto,
    placement: QdqPlacement,
) -> list[dict[str, Any]]:
    """Map the ONNX placement to quantizer insertions on ``graph_module``, by node name."""
    nodes = {node.name: node for node in graph_module.graph.nodes}
    decomposed_nodes = {node.name: node for node in decomposed.nodes}

    def original(decomposed_node: fx.Node) -> fx.Node | None:
        # The exporter's decomposition records which node of the exported graph each node is from.
        sources = decomposed_node.meta.get("from_node")
        while sources:
            if sources[-1].graph_id == graph_id:
                return nodes.get(sources[-1].name)
            sources = sources[-1].from_node
        return None

    def traced(decomposed_node: fx.Node) -> fx.Node:
        node = original(decomposed_node)
        if node is None:
            raise ValueError(
                f"Cannot trace decomposed node '{decomposed_node.name}' to the export."
            )
        return node

    # Nodes the decomposition dropped as identities in eval mode, e.g. dropout.
    kept = {original(node) for node in decomposed.nodes}
    elided = {node for node in graph_module.graph.nodes if node.op == "call_function"} - kept

    params = {node.target: node for node in graph_module.graph.nodes if node.op == "get_attr"}
    placeholders = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
    # Graph inputs are named after the forward arguments in ONNX but may be renamed in FX.
    inputs = dict(zip((value.name for value in onnx_model.graph.input), placeholders, strict=True))
    onnx_nodes = {node.name: node for node in onnx_model.graph.node}
    producers = {output: node for node in onnx_model.graph.node for output in node.output}
    initializer_numel = {
        init.name: int(torch.Size(init.dims).numel()) for init in onnx_model.graph.initializer
    }

    insertions = []
    for tensor in placement.tensors.values():
        consumers = list(
            dict.fromkeys(
                traced(decomposed_nodes[_fx_node_name(onnx_nodes[name])])
                for name, _ in tensor.consumers
            )
        )
        if tensor.name in inputs:
            source = inputs[tensor.name]
        elif tensor.name in params:
            source = params[tensor.name]
        elif tensor.name in decomposed_nodes:
            source = traced(decomposed_nodes[tensor.name])
            # The tensor must be the output of the exported node, not internal to its decomposition.
            if not any(
                user.op == "output" or original(user) is not source
                for user in decomposed_nodes[tensor.name].users
            ):
                raise ValueError(
                    f"Quantized tensor '{tensor.name}' is internal to the decomposition of "
                    f"'{source.name}' and has no counterpart in the exported graph."
                )
        elif (getitem := _split_output(tensor.name, producers, decomposed_nodes)) is not None:
            source = traced(getitem)
        elif tensor.is_constant:
            source = _find_folded_param(tensor.name, consumers, initializer_numel[tensor.name])
        else:
            raise ValueError(
                f"Quantized tensor '{tensor.name}' is internal to the ONNX translation of "
                f"{[c.name for c in consumers]} and has no counterpart in the exported graph."
            )

        edges = dict.fromkeys(_reader_of(source, consumer, elided) for consumer in consumers)
        if len({input_ for _, input_ in edges}) != 1:
            raise ValueError(
                f"The readers of quantized tensor '{tensor.name}' read it through different "
                "nodes in the exported graph, so one quantizer cannot serve them."
            )
        consumers = [reader for reader, _ in edges]
        source = next(iter(edges))[1]
        name = _quantizer_name(source, consumers[0], tensor.axis is not None, insertions)
        insertions.append(
            {"name": name, "source": source.name, "consumers": [c.name for c in consumers]}
        )
    return insertions


def _insert_quantizers(graph_module: fx.GraphModule, insertions: list[dict[str, Any]]) -> None:
    graph = graph_module.graph
    nodes = {node.name: node for node in graph.nodes}
    missing = {i["source"] for i in insertions} | {c for i in insertions for c in i["consumers"]}
    missing -= nodes.keys()
    if missing:
        raise ValueError(
            f"The exported graph has no nodes {sorted(missing)} to place quantizers on."
        )
    last_placeholder = [node for node in graph.nodes if node.op == "placeholder"][-1]
    for insertion in insertions:
        source = nodes[insertion["source"]]
        graph_module.add_submodule(insertion["name"], TensorQuantizer())
        with graph.inserting_after(last_placeholder if source.op == "placeholder" else source):
            quantized = graph.call_module(insertion["name"], (source,))
        for consumer in insertion["consumers"]:
            nodes[consumer].replace_input_with(source, quantized)
    graph.lint()
    graph_module.recompile()


def _fx_node_name(onnx_node: onnx.NodeProto) -> str:
    # The exporter records the FX node an ONNX node was translated from as the last name scope.
    metadata = {prop.key: prop.value for prop in onnx_node.metadata_props}
    return ast.literal_eval(metadata["pkg.torch.onnx.name_scopes"])[-1]


def _reader_of(source: fx.Node, consumer: fx.Node, elided: set[fx.Node]) -> tuple[fx.Node, fx.Node]:
    """Return the FX node reading ``source`` for ``consumer`` and the input it reads it through.

    An ONNX node fused from several FX nodes (e.g. Conv with a folded BatchNorm) records the last
    of them; the reader is then one of its ancestors. The reader may read ``source`` through
    ``elided`` nodes (eval-mode identities such as dropout); the quantizer goes after them so they
    still apply in train mode.
    """
    pending, seen = [consumer], set()
    while pending:
        node = pending.pop()
        for input_ in node.all_input_nodes:
            through = input_
            while through in elided and through.args and isinstance(through.args[0], fx.Node):
                through = through.args[0]
            if through is source:
                return node, input_
        seen.add(node)
        pending.extend(n for n in node.all_input_nodes if n not in seen and n.op == "call_function")
    raise ValueError(f"No FX node at or above '{consumer.name}' reads '{source.name}'.")


def _split_output(
    tensor_name: str, producers: dict[str, onnx.NodeProto], decomposed_nodes: dict[str, fx.Node]
) -> fx.Node | None:
    """Return the ``getitem`` of an ``aten.split`` that the exporter folded into an ONNX Split."""
    # The folded Split has no FX metadata, so output k is matched to getitem k of the only split
    # of the same input.
    producer = producers.get(tensor_name)
    if producer is None or producer.op_type != "Split" or producer.input[0] not in decomposed_nodes:
        return None
    index = list(producer.output).index(tensor_name)
    splits = [
        user
        for user in decomposed_nodes[producer.input[0]].users
        if user.target in (torch.ops.aten.split.Tensor, torch.ops.aten.split_with_sizes.default)
    ]
    getitems = [
        user
        for split in splits
        for user in split.users
        if user.target is operator.getitem and user.args[1] == index
    ]
    return getitems[0] if len(splits) == 1 and len(getitems) == 1 else None


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
    source: fx.Node, first_consumer: fx.Node, per_channel: bool, insertions: list[dict[str, Any]]
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
        if own_op and any(i["name"] == _join(owner, leaf) for i in insertions):
            leaf = f"{source.name}_input_quantizer"
    return _join(owner, leaf)


def _join(owner: str, leaf: str) -> str:
    return f"{owner}.{leaf}" if owner else leaf
