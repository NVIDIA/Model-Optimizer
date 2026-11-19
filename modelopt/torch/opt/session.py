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
runs in it, those the library makes included. Outside any block, calls run in the process's default
session, which is never closed (:meth:`ModelOptSession.active`), so code that opens no session runs
as it did before ModelOptSession.

A session holds its workflow's models by name (:data:`ModelName`): ``"model"``, the model the
workflow optimizes and exports, which a public call binds by itself, and ``"teacher"``, the model a
distillation learns from, which the workflow binds (:meth:`ModelOptSession.bind`).

ModelOpt supports one model optimization workflow at a time in a process: different workflows run
at once from several threads, such as two models quantized in two threads, are not supported. The
one workflow's session is shared by all the process's threads.
"""

import contextlib
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

import torch

from modelopt.torch.models import hf_model_type

if TYPE_CHECKING:
    # From its defining module: mypy reads diffusers' top-level name as the placeholder class it
    # defines for installs without torch.
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

__all__ = ["ModelBinding", "ModelName", "ModelOptSession"]

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
    _held: Callable[[], "_Model | None"] = field(repr=False)

    @property
    def model(self) -> "_Model | None":
        """The bound model, or ``None`` once the default session's weakly held model is collected."""
        return self._held()


@dataclass(eq=False)
class ModelOptSession:
    """What one ModelOpt workflow decided, as its calls take it, and the models it works on.

    Each field belongs to the calls that read it, and a call works out a field left unset as it
    would without a session. A call's own argument for the same thing wins over the field.

    ``with ModelOptSession():`` makes it the process's active session for the block, and every
    ModelOpt call in the block runs in it. Opening another while one is active raises: a workflow
    has one session.

    A session binds each model of its workflow under a name (:meth:`bind`), and a name keeps the
    model it is first bound to: a call on that model or on a part of it, such as the language model
    a VLM is calibrated through, runs under that name, and binding the name to any other model
    raises, one that contains the bound model included. The default session alone follows the
    model that code opening no session works on: a public call on a model no name covers rebinds
    ``"model"`` to it, dropping what the session held for the model before. A session a ``with``
    block opened holds its models until the block ends; the default session holds them weakly, so
    code that drops a model frees it. A copy of a session, pickled or deep-copied, is unbound, as is
    a session whose block has ended.
    """

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
        # The workflow ends with its block: the session lets go of its models.
        self._bindings.clear()

    @property
    def model(self) -> "_Model | None":
        """The model bound as ``"model"``, or ``None`` until a call binds one."""
        binding = self._bindings.get("model")
        return binding.model if binding is not None else None

    @property
    def model_type(self) -> str | None:
        """The type of the model bound as ``"model"`` (:attr:`ModelBinding.model_type`)."""
        binding = self._bindings.get("model")
        return binding.model_type if binding is not None and binding.model is not None else None

    def bind(self, model: _Model, name: ModelName | None = None) -> "ModelOptSession":
        """Bind ``model`` under ``name`` in this session, and return the session.

        Without a name, ``model`` runs under the name whose model it is or is a part of, the
        innermost where bound models nest, as a teacher inside its student does; a model no name
        covers is bound as ``"model"``. A name keeps the model it is first bound to: binding it
        again to that model or to a part of it changes nothing, and any other model raises, one
        that contains the bound model included, such as a compiled or PEFT wrapper of it. A workflow
        that calls on a part of a model binds the whole model first. In the default session, a model
        no name covers rebinds ``"model"`` instead, as code that opens no session moves on to
        another model. Binding reads the model's type off the model it binds.
        """
        self._bind(model, name)
        return self

    def _bind(self, model: _Model, name: ModelName | None) -> ModelBinding:
        follows = False
        if name is None:
            covering = self._covering(model)
            if covering is not None:
                return covering
            name, follows = "model", self._is_default
        held = self._bindings.get(name)
        bound = held.model if held is not None else None
        if held is not None and bound is not None:
            if _contains(bound, model):
                return held
            if not follows:
                raise RuntimeError(
                    f"This ModelOptSession describes a workflow on another model as {name!r}, a "
                    f"{type(bound).__name__}, so it cannot be used for this {type(model).__name__}: "
                    "a session describes one model under each name, and a call takes that model or "
                    "a part of it. Bind the session to the whole model before any call on a part of "
                    "it."
                )
        binding = ModelBinding(name, hf_model_type(model), self._hold(model))
        self._bindings[name] = binding
        return binding

    def binding(self, model: _Model) -> ModelBinding:
        """The binding of the model ``model`` is or is a part of, the innermost where they nest.

        A model no binding reaches, such as a component a pipeline keeps where the session cannot
        see it, is taken as a part of the model bound as ``"model"``.
        """
        covering = self._covering(model) or self._bindings.get("model")
        if covering is None or covering.model is None:
            raise RuntimeError(
                f"The active ModelOptSession binds no model this {type(model).__name__} is, or is a "
                "part of: a public ModelOpt call binds its model, so code that calls ModelOpt's "
                "internal functions binds it first (ensure_modelopt_session_binding)."
            )
        return covering

    def _covering(self, model: _Model) -> ModelBinding | None:
        innermost, innermost_model = None, None
        for binding in self._bindings.values():
            bound = binding.model
            if bound is None or not _contains(bound, model):
                continue
            if innermost_model is None or _contains(innermost_model, bound):
                innermost, innermost_model = binding, bound
        return innermost

    def _hold(self, model: _Model) -> Callable[[], "_Model | None"]:
        if self._is_default:
            # Weakly, so code that drops a model frees it; an object without weak references, such
            # as a stand-in pipeline, is held as a session a block opened holds it.
            with contextlib.suppress(TypeError):
                return weakref.ref(model)
        return lambda: model

    def __getstate__(self) -> dict[str, object]:
        # The bindings belong to this process and to this session; a copy describes the same
        # workflow without them.
        state = self.__dict__.copy()
        state["_bindings"] = {}
        state["_is_default"] = False
        return state


# The session a ``with ModelOptSession():`` block opened, for the block.
_active: ModelOptSession | None = None
# The process's default session, which calls outside any block run in. It is never closed.
_default = ModelOptSession()
_default._is_default = True


def ensure_modelopt_session_binding(model: _Model) -> ModelBinding:
    """Bind ``model``, the model a public ModelOpt call takes, in the session the call runs in.

    That is the session a ``with ModelOptSession():`` block opened, else the process's default
    session (:meth:`ModelOptSession.active`). Each public entry point calls this first, so the calls
    of a workflow share its session, and the internal functions they call find the model's binding.

    The model has to be the root model the ModelOpt workflow works on, such as a whole VLM rather
    than its language model: the session reads the model type off it, and checks it whole, as the
    export will write it. A call may take a part of the root once the session binds the root, which
    binding the part leaves in place, as when the language model of a VLM is calibrated in a session
    bound to the VLM.
    """
    return ModelOptSession.active()._bind(model, None)


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
