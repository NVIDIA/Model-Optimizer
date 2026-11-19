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

"""The session a ModelOpt workflow runs in: one :class:`ModelOptSession` for the whole workflow.

An application runs its workflow inside ``with ModelOptSession():``. That makes it the process's
active session for the block (:meth:`ModelOptSession.current`), and every ModelOpt call in the block
runs in it, those the library makes included. The workflow binds the models it works on in it by
name (:data:`ModelName`, :meth:`ModelOptSession.bind`): ``"model"``, the model the workflow
optimizes and exports, and ``"teacher"``, the model a distillation learns from. A call in the block
reads what it needs to know of a model, such as its type, from the model's binding.

Outside any block, calls run in the process's default session (:meth:`ModelOptSession.active`),
which binds nothing: a call there reads what it needs off the model itself, so code that opens no
session runs as it did before ModelOptSession.

ModelOpt supports one model optimization workflow at a time in a process: different workflows run
at once from several threads, such as two models quantized in two threads, are not supported. The
one workflow's session is shared by all the process's threads, and holds one export in progress
at a time, a layerwise calibration's included.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

import torch

from modelopt.torch.models import hf_model_type

if TYPE_CHECKING:
    # From its defining module: mypy reads diffusers' top-level name as the placeholder class it
    # defines for installs without torch.
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

    from modelopt.torch.export.layerwise_export import LayerwiseExporter
    from modelopt.torch.export.registry import ExportContext
    from modelopt.torch.utils.distributed import DistributionPlan

__all__ = ["ModelBinding", "ModelName", "ModelOptSession", "RunState"]

# What a session binds: a model, or a diffusers pipeline, which its export binds whole.
_Model: TypeAlias = "torch.nn.Module | DiffusionPipeline"

# The names a session binds models under: ``"model"`` is the model the workflow optimizes and
# exports, ``"teacher"`` the model a distillation learns from.
ModelName: TypeAlias = Literal["model", "teacher"]


@dataclass(eq=False)
class ModelBinding:
    """A model a session describes, under its name, and what binding read off it.

    ``model_type`` is the model's Hugging Face model type (``config.model_type``), ``None`` for a
    model without one. The export resolves the model's spec in :mod:`modelopt.torch.models` by it,
    also for the decoder layers and experts it hands its handlers, which carry no type of their own.
    """

    name: ModelName
    model_type: str | None
    model: "_Model" = field(repr=False)


@dataclass
class RunState:
    """The live objects one call of a workflow, or one step of it, creates and a later one uses.

    ``layerwise_exporter`` is the :class:`~modelopt.torch.export.layerwise_export.LayerwiseExporter`
    a layerwise calibration wrote every layer's shard through, for the export to finalize.

    ``export_context`` is the :class:`~modelopt.torch.export.registry.ExportContext` of the export
    in progress. The export keeps it here from its setup until it ends, and each of its steps and
    handlers gets it here. A layerwise export's is here while it writes a layer, or the tail.
    """

    layerwise_exporter: "LayerwiseExporter | None" = None
    export_context: "ExportContext | None" = None


@dataclass(eq=False)
class ModelOptSession:
    """What one ModelOpt workflow decided, as its calls take it, and the models it works on.

    Each field belongs to the calls that read it, and a call works out a field left unset as it
    would without a session. A call's own argument for the same thing wins over the field.

    * ``distribution``: the :class:`~modelopt.torch.utils.distributed.DistributionPlan` to load the
      model by. ``load_pretrained`` loads by it where its ``distribution=`` is not passed, and
      records here the plan the load used, or, while none is held, the plan a load was given.

    ``runtime`` (:class:`RunState`) holds the live objects one call hands to a later one, such as
    the layerwise exporter calibration writes through and the export finalizes, and those one step
    of a call hands to the next, such as the context of the export in progress.

    ``with ModelOptSession():`` makes it the process's active session for the block, and every
    ModelOpt call in the block runs in it. Opening another while one is active raises: a workflow
    has one session.

    The workflow binds each of its models under a name (:meth:`bind`), and a name keeps the model
    it is first bound to. A call on that model or on a part of it, such as the language model a
    VLM is calibrated through, reads the model's binding (:meth:`binding`), and a call that reads
    the binding of a model the session does not bind raises. The session holds its models until
    its block ends. A copy of a session, pickled or deep-copied, is unbound and holds no live
    objects, as is a session whose block has ended.

    The process's default session, which calls outside any block run in, binds nothing: a call
    there reads what it needs off the model it is given.
    """

    distribution: "DistributionPlan | None" = None
    runtime: RunState = field(default_factory=RunState, init=False, repr=False)
    _bindings: dict[ModelName, ModelBinding] = field(default_factory=dict, init=False, repr=False)
    _is_default: bool = field(default=False, init=False, repr=False)

    @staticmethod
    def current() -> "ModelOptSession | None":
        """The session a ``with ModelOptSession():`` block opened, or ``None`` outside any."""
        return _active

    @staticmethod
    def active() -> "ModelOptSession":
        """The session ModelOpt's calls run in: the one a block opened, else the default session."""
        return _active if _active is not None else _default

    def __enter__(self) -> "ModelOptSession":
        global _active
        if self._is_default:
            raise RuntimeError(
                "The process's default ModelOptSession is never opened with `with`: open a new "
                "ModelOptSession for a workflow of its own."
            )
        if _active is not None:
            raise RuntimeError(
                "A ModelOptSession is already active: a ModelOpt workflow has one session, and a "
                "process runs one workflow at a time."
            )
        _active = self
        return self

    def __exit__(self, *exc_info: object) -> None:
        global _active
        _active = None
        # The workflow ends with its block: the session lets go of its models and live objects.
        self._bindings.clear()
        self.runtime = RunState()

    @property
    def model(self) -> "_Model | None":
        """The model bound as ``"model"``, or ``None`` until the workflow binds one."""
        binding = self._bindings.get("model")
        return binding.model if binding is not None else None

    @property
    def model_type(self) -> str | None:
        """The type of the model bound as ``"model"`` (:attr:`ModelBinding.model_type`)."""
        binding = self._bindings.get("model")
        return binding.model_type if binding is not None else None

    def bind(self, model: _Model, name: ModelName | None = None) -> "ModelOptSession":
        """Bind ``model`` under ``name`` in this session, and return the session.

        Without a name, ``model`` is bound under the name whose model it is or is a part of, the
        innermost where bound models nest, as a teacher inside its student is; a model no name
        covers is bound as ``"model"``. A name keeps the model it is first bound to: binding it
        again to that model or to a part of it changes nothing, and any other model raises, one
        that contains the bound model included, such as a compiled or PEFT wrapper of it. A workflow
        that calls on a part of a model binds the whole model first. Binding reads the model's type
        off the model it binds. The process's default session binds nothing, so binding in it
        raises.
        """
        if self._is_default:
            raise RuntimeError(
                "The process's default ModelOptSession binds no model: bind the workflow's models in "
                "the session a `with ModelOptSession():` block opens."
            )
        if name is None:
            if self._covering(model) is not None:
                return self
            name = "model"
        held = self._bindings.get(name)
        if held is not None:
            if _contains(held.model, model):
                return self
            raise RuntimeError(
                f"This ModelOptSession describes a workflow on another model as {name!r}, a "
                f"{type(held.model).__name__}, so it cannot be used for this "
                f"{type(model).__name__}: a session describes one model under each name, and a call "
                "takes that model or a part of it. Bind the session to the whole model before any "
                "call on a part of it."
            )
        self._bindings[name] = ModelBinding(name, hf_model_type(model), model)
        return self

    def binding(self, model: _Model) -> ModelBinding:
        """The binding of the model ``model`` is or is a part of, the innermost where they nest.

        A session a ``with`` block opened raises for a model it does not bind, except a component
        of a bound pipeline that the pipeline keeps where the session cannot see it, which is taken
        as a part of the pipeline. The default session binds nothing, so its binding of a model is
        read off the model itself.
        """
        if self._is_default:
            return ModelBinding("model", hf_model_type(model), model)
        covering = self._covering(model)
        if covering is None and not isinstance(self.model, (torch.nn.Module, type(None))):
            covering = self._bindings["model"]
        if covering is None:
            raise RuntimeError(
                f"The active ModelOptSession binds no model this {type(model).__name__} is, or is a "
                "part of: bind the workflow's model in it first, with session.bind(model)."
            )
        return covering

    def _covering(self, model: _Model) -> ModelBinding | None:
        innermost = None
        for binding in self._bindings.values():
            if not _contains(binding.model, model):
                continue
            if innermost is None or _contains(innermost.model, binding.model):
                innermost = binding
        return innermost

    def __getstate__(self) -> dict[str, object]:
        # The bindings and the live objects belong to this process and to this session; a copy
        # describes the same workflow without them.
        state = self.__dict__.copy()
        state["_bindings"] = {}
        state["_is_default"] = False
        state["runtime"] = RunState()
        return state


# The session a ``with ModelOptSession():`` block opened, for the block.
_active: ModelOptSession | None = None
# The process's default session, which calls outside any block run in. It binds nothing.
_default = ModelOptSession()
_default._is_default = True


def _contains(root: _Model, module: _Model) -> bool:
    """Whether ``module`` is ``root`` or one of its parts: a submodule, or a pipeline's component."""
    if root is module:
        return True
    if isinstance(root, torch.nn.Module):
        return any(m is module for m in root.modules())
    # A diffusers pipeline, which a workflow can quantize component by component and export whole.
    components = getattr(root, "components", None)
    return isinstance(components, dict) and any(
        isinstance(component, torch.nn.Module) and _contains(component, module)
        for component in components.values()
    )
