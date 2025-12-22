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

"""Module for supporting tracing of concat operations.

Note that this implementation assumes that the concat operation/symbol is the _only_ searchable
symbol within dependent symbols. This enables us to simplify partial dependencies that would
otherwise arise from having multiple concats linked together.

There is one small exception to this: whenever one concat depends on another concat, we can disable
the independent concat and simplify the representation this way.

However, truly linked concats cannot be handled, e.g.,
``torch.cat([x1,x2], dim=1) + torch.cat([y1, y2], dim=1)``,
as there is no way of disabling one but not the other.
"""

import copy
import inspect

import torch
import torch.nn as nn
from torch.fx import Node
from torch.overrides import get_testing_overrides

from modelopt.torch.utils.graph import NodeTarget

from ..analyzer import GraphDependencyProcessor, NodeProcessor
from ..symbols import Symbol, SymInfo

__all__ = ["ConcatNodeProcessor", "ConcatSymbol"]


class ConcatSymbol(Symbol):
    """Symbol storing an ordered list of linked symbolic inputs to concat."""

    class Input(Symbol):
        """Special Symbol to represent an input to a ConcatSymbol.

        This symbol is an augmented version of the regular Symbol to handle the interaction with the
        concat operation and is used to monkey-patch the original symbol.
        """

        _concat_sym: "ConcatSymbol | None"  # concat symbol that this is linked to
        _original_class: Symbol | None  # original class of the symbol

        def __init__(self, *args, **kwargs):
            """Constructor."""
            raise NotImplementedError("ConcatSymbol.Input cannot be instantiated; only converted.")

        def link_to(self, other: Symbol) -> None:
            """Link self to other symbol by simply disabling both.

            ConcatSymbol.Input can never be linked to another symbol, but other symbols can be
            linked to it without disabling it!
            """
            ConcatSymbol._strict_link_to(self, other)

        @property
        def concat_sym(self) -> "ConcatSymbol":
            """Return concat symbl."""
            assert self._concat_sym is not None, "Concat symbol not set for ConcatSymbol.Input."
            return self._concat_sym

        @concat_sym.setter
        def concat_sym(self, concat_sym: "ConcatSymbol") -> None:
            """Set concat symbol."""
            self._concat_sym = concat_sym

        @staticmethod
        def convert(orig_sym: Symbol, cat_dim: int) -> "ConcatSymbol.Input | ConcatSymbol":
            """Modify and convert the sym in-place to a valid ConcatSymbol.Input."""
            # if symbol is already a concat symbol, we need to disable it but can still use it.
            if isinstance(orig_sym, (ConcatSymbol, ConcatSymbol.Input)):
                orig_sym.disable()
                return orig_sym

            # check compatibility of symbol with cat_dim (disable if not)
            if cat_dim not in orig_sym.elastic_dims:
                orig_sym.disable()

            # here we are going to monkey-patch the original symbol by changing its class.
            current_class = orig_sym.__class__
            orig_sym.__class__ = ConcatSymbol.Input
            assert isinstance(orig_sym, ConcatSymbol.Input)
            orig_sym._original_class = current_class

            return orig_sym

        def create_linked_copy(self) -> Symbol:
            """Get a linked deepcopy of self that is monkey-patched to the original symbol class."""
            orig_sym = copy.deepcopy(self)
            orig_sym.__class__ = self._original_class
            del orig_sym._concat_sym
            del orig_sym._original_class
            orig_sym._reset_state()
            orig_sym.link_to(self)

            return orig_sym

    def __init__(
        self,
        symbols: list[Symbol],
        cl_type: Symbol.CLType = Symbol.CLType.NONE,
        elastic_dims: set[int] | None = None,
    ):
        """Initializes Symbol from input symbols."""
        super().__init__(cl_type=cl_type, elastic_dims=elastic_dims)
        assert len(symbols) > 0, "ConcatSymbol must have at least one input."
        assert all(isinstance(sym, (ConcatSymbol.Input, ConcatSymbol)) for sym in symbols), (
            "All syms must be ConcatSymbol.Inputs."
        )

        # assign reference of self to each input symbol and store list of input symbols
        for sym in symbols:
            if isinstance(sym, ConcatSymbol.Input):
                sym.concat_sym = self
        self._input_syms: list[ConcatSymbol.Input | ConcatSymbol] = symbols

    def disable(self, _memo: set[Symbol] | None = None) -> None:
        """Disable all symbols including input symbols.

        We handle input symbols by fake adding them to the dependency list. Note that the dependency
        list is cleared at the end anyway - so this is fine.
        """
        self._dependencies.extend(self.input_syms)
        super().disable(_memo)

    @staticmethod
    def _strict_link_to(self_sym: "ConcatSymbol | ConcatSymbol.Input", other: Symbol) -> None:
        msg = "Linking concat symbol to regular symbol not supported. Use the other way around!"
        assert isinstance(other, (ConcatSymbol, ConcatSymbol.Input)), msg

        # Currently we don't support linking two such symbols together.
        # This is really tricky as the concat symbols might have totally different compositions.
        # Then how do you later set the correct choices for each???
        self_sym.disable()
        other.disable()

    def link_to(self, other: Symbol) -> None:
        """Link self to other symbol."""
        self._strict_link_to(self, other)

    @property
    def input_syms(self) -> list["ConcatSymbol.Input | ConcatSymbol"]:
        """Return symbols."""
        return self._input_syms

    @property
    def is_searchable(self) -> bool:
        """Return indicator whether symbol is searchable.

        Unlike a regular symbol where this is determined based on a manually set flag, is_searchable
        for concat is set according to whether ANY input symbols are searchable.
        """
        return any(sym.is_searchable for sym in self.input_syms)

    @property
    def is_constant(self) -> bool:
        """Return indicator whether symbol is constant.

        Unlike a regular symbol where this is determined based on a manually set flag, is_constant
        for concat is set according to whether ALL input symbols are constant.
        """
        return all(sym.is_constant for sym in self.input_syms)

    def _check_sortable(self, _memo: set[Symbol] | None = None) -> bool:
        # if we just start the recursion we also check input syms in separate calls/recursions
        _is_sortable_input = bool(_memo) or any(sym._check_sortable() for sym in self.input_syms)

        # All dependencies must be sortable and any of the input symbols must be sortable.
        # Then we can at least do partial sorting...
        return _is_sortable_input and super()._check_sortable(_memo)


class ConcatModule(nn.Module):
    """Fake DynamicConcat module to support torch.cat for conv inputs in tracing.

    We wrap the torch.cat operation in a module to support tracing for more control. This way we can
    use the "call_module" node type in the trace graph instead "call_method", "call_function".
    """

    def __init__(self, out_sym: ConcatSymbol):
        """Init."""
        super().__init__()
        self.out_sym = out_sym

    def get_sym_info(self) -> SymInfo:
        # we are giving the out-going symbol a generic name here as we don't know the exact name
        # but it shouldn't matter as this is only used for tracing and the tracing does not depend
        # on the name of the symbol.
        return SymInfo(is_shape_preserving=False, out_sym=self.out_sym)


@GraphDependencyProcessor.register_node_processor
class ConcatNodeProcessor(NodeProcessor):
    """Node for handling concat specific tracing logic."""

    def __init__(self, *args, **kwargs) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        self._processed_nodes: set[Node] = set()  # processed concat nodes with temp concat module
        self._concat_sym_to_mod_name: dict[ConcatSymbol, str] = {}  # look-up for temp ConcatModules

    def reset(self) -> None:
        """Reset state."""
        self._processed_nodes.clear()
        self._concat_sym_to_mod_name.clear()

    def is_special_node(self, node: Node, target: NodeTarget) -> bool:
        """Return whether node is a concat node."""
        aliases = [torch.cat, torch.concat, torch.concatenate]
        return self._is_from_node_list(target, aliases, self._gc)

    def process(self, node: Node, id: int, input_nodes: list[Node]) -> None:
        """Make all inputs for the concat searchable."""
        # extract root nodes for all input nodes
        input_nodes = self._get_root_nodes(input_nodes)

        # get concat dim from either kwargs, args, or default value
        if "dim" in node.kwargs:
            cat_dim = node.kwargs["dim"]
        elif len(node.args) > 1:
            cat_dim = node.args[1]
        else:
            # torch.cat does not contain a signature --> we have to use the fake target that is
            # retrievable from get_testing_overrides (it's a torch utility for exactly this purpose!)
            cat_target_fake = get_testing_overrides()[node.target]
            cat_dim = inspect.signature(cat_target_fake).parameters["dim"].default
        assert isinstance(cat_dim, int), "Concat dim must be an integer."

        # collect out-going symbols from all input nodes
        syms: list[Symbol] = []
        for in_node in input_nodes:
            in_target = self._get_root_target(in_node)
            syms_out = [sym for _, sym in self.named_out_symbols(in_target)]
            if len(syms_out) == 0:
                # TODO: we could create a fake symbol but then there is no module associated with it
                # --> handling this would be tough...
                # --> see below for previous solution...
                return self._process_boundary(node, id, input_nodes)

                # double-check that in_target is not in sym_map (otherwise we should find a sym...)
                assert in_target not in self._sym_map, "Input node must not be in sym_map."
                # let's create a fake, disabled symbol for this input node
                sym_fake = Symbol(cl_type=Symbol.CLType.OUTGOING, elastic_dims={cat_dim})
                sym_fake.disable()
                syms_out.append(sym_fake)
            elif len(syms_out) > 1:
                # we cannot handle multiple out symbols for concat nodes
                raise RuntimeError("Concat node with multiple out symbols not supported.")
            syms.append(syms_out[0])

        # check if there are *any* symbols that are searchable and compatible with cat_dim
        if not any(sym.is_searchable and cat_dim in sym.elastic_dims for sym in syms):
            # if not, we process the node as boundary instead --> there is no point in processing
            # it as concat node at this point...
            return self._process_boundary(node, id, input_nodes)

        # change op to call_module
        node.op = "call_module"

        # check for similar concat nodes (:=exact same roots but separate node)
        similar_node = None
        for processed_node in self._processed_nodes:
            input_nodes_ = self._get_root_nodes(self._identify_in_out_nodes(processed_node)[0])
            if input_nodes_ == input_nodes:
                similar_node = processed_node
                break

        if similar_node:  # example use case: OCRNet :)
            # reuse already created concat module otherwise the node will get treated as boundary
            node.target = similar_node.target
        else:
            # input_nodes might not contain duplicates, however, we need duplicates for syms!
            # Let's check args/kwargs of the node directly for that
            tensors = node.kwargs["tensors"] if "tensors" in node.kwargs else node.args[0]
            tensors = self._get_root_nodes(tensors)
            assert set(tensors) == set(input_nodes)

            # duplicate syms according to the tensors arg/kwarg
            syms = [syms[input_nodes.index(t)] for t in tensors]

            # modify and patch all symbols to valid input symbols
            for sym in set(syms):  # ignoring duplicate inputs to the symbol (this is fine...)
                ConcatSymbol.Input.convert(sym, cat_dim)

            # initialize the fake concat module with the symbols from the input nodes
            concat_sym = ConcatSymbol(syms, cl_type=Symbol.CLType.OUTGOING, elastic_dims={cat_dim})

            # modify node to be a module call to the fake concat module
            node.name = f"{node.name}-{id}"  # to avoid name conflicts
            node.target = node.name

            # setup the fake concat module and register it correctly within the model and sym_map
            mod_concat = ConcatModule(concat_sym)
            self._model.add_module(node.name, mod_concat)
            self._sym_map.add_sym_info(mod_concat, mod_concat.get_sym_info())

            # record created node
            self._processed_nodes.add(node)
            self._concat_sym_to_mod_name[concat_sym] = node.name

        self._create_root(node, id, True, priority=1)  # higher priority when synchronizing

    @classmethod
    def _flatten_nested_cats(cls, sym: ConcatSymbol, memo: set[ConcatSymbol]):
        """Flatten the inputs to a nested concat symbol."""
        in_flattened = []
        for sym_in in sym.input_syms:
            if not isinstance(sym_in, ConcatSymbol):
                in_flattened.append(sym_in)
                continue
            assert sym_in.is_constant
            memo.add(sym_in)
            in_flattened.extend(cls._flatten_nested_cats(sym_in, memo))
        return in_flattened

    def post_process(self) -> None:
        """Revert back to original symbols."""
        # re-assign ConcatSymbol as incoming symbol to relevant modules and remove the modules'
        # original incoming symbols. Note that we are doing reassignments, so we need to iterate
        # over a copy of the list!
        memo_flattened: set[ConcatSymbol] = set()
        for mod in list(self._model.modules()):
            for name, sym in list(self.named_in_symbols(mod)):
                # check if the symbol's parent points to a concat symbol
                parent = sym.parent
                if parent not in self._concat_sym_to_mod_name:
                    continue
                assert isinstance(parent, ConcatSymbol)
                concat_mod_name = self._concat_sym_to_mod_name.pop(parent)

                # transfer relevant properties to the parent and store as the new module symbol
                parent._cl_type = Symbol.CLType.INCOMING
                self._sym_map.set_symbol(mod, name, parent)

                # clean-up
                concat_mod = getattr(self._model, concat_mod_name)
                self._sym_map.pop(concat_mod)
                delattr(self._model, concat_mod_name)
                parent._dependencies.remove(sym)
                del sym

                # flatten potentially nested concats next
                parent._input_syms = self._flatten_nested_cats(parent, memo_flattened)

        # Hide the Input symbols of the concat modules from the visible symbols.
        for mod in self._model.modules():
            for name, sym in list(self._sym_map.named_symbols(mod)) if mod in self._sym_map else []:
                if not isinstance(sym, ConcatSymbol.Input):
                    continue
                self._sym_map.set_symbol(mod, name, sym.create_linked_copy())

        # Remove concats that were already used during flattening
        for concat_sym in memo_flattened:
            if concat_sym in self._concat_sym_to_mod_name:
                delattr(self._model, self._concat_sym_to_mod_name.pop(concat_sym))

        # Disable any concat modules that still present: This may be possible if
        # a concat module is not used in the output or it was disabled during the tracing.
        for concat_sym, name in self._concat_sym_to_mod_name.items():
            if not concat_sym.is_dynamic:
                concat_sym.disable()
            delattr(self._model, name)

        for mod in self._model.modules():
            assert not isinstance(mod, ConcatModule), "ConcatModule still present in model."
