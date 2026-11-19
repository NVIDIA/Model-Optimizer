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
active session (:meth:`ModelOptSession.current`), and every ModelOpt call in the block that takes
part, those the library makes included, runs in it. A public call made outside any session runs in
one set up for that call alone, so a call made without a session gets what the call always did.

ModelOpt supports one model optimization workflow at a time in a process: different workflows run
at once from several threads, such as two models quantized in two threads, are not supported. The
one workflow's session is shared by all the process's threads.
"""

import contextlib
import functools
import inspect
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ParamSpec, TypeAlias, TypeVar, cast

import torch

from modelopt.torch.models import hf_model_type

if TYPE_CHECKING:
    # From its defining module: mypy reads diffusers' top-level name as the placeholder class it
    # defines for installs without torch.
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

__all__ = ["ModelOptSession"]

_P = ParamSpec("_P")
_R = TypeVar("_R")

# What a session is bound to: a model, or a diffusers pipeline, which its export binds whole.
_Model: TypeAlias = "torch.nn.Module | DiffusionPipeline"


@dataclass(eq=False)
class ModelOptSession:
    """What one ModelOpt workflow decided, as its calls take it, and the model it works on.

    Each field belongs to the calls that read it, and a call works out a field left unset as it
    would without a session. A call's own argument for the same thing wins over the field.

    * ``model_type``: the model's Hugging Face model type (``config.model_type``), which binding
      reads off the model (:meth:`bind`), ``None`` for a model without one. The export resolves
      the model's spec in :mod:`modelopt.torch.models` by it, also for the decoder layers and
      experts it hands its handlers, which carry no type of their own.

    ``with ModelOptSession():`` makes it the process's active session for the block, and every
    ModelOpt call in the block runs in it. Opening another while one is active raises: a workflow
    has one session. A session describes one model, which it holds until its block ends. The first
    call binds it to that model (:meth:`bind`), and a call on another model raises; the model's
    submodules, and the model containing it, are the same model, since a VLM is calibrated through
    its language model and exported whole. A copy of a session, pickled or deep-copied, is unbound,
    as is a session whose block has ended.
    """

    model_type: str | None = field(default=None, init=False)
    _model: "_Model | None" = field(default=None, init=False, repr=False)

    @staticmethod
    def current() -> "ModelOptSession | None":
        """The process's active session, or ``None`` outside any."""
        return _active

    @staticmethod
    def active() -> "ModelOptSession":
        """The process's active session, for the library's own functions, which run inside one.

        They run inside a public ModelOpt call or a ``with ModelOptSession():`` block, so outside
        any session this raises.
        """
        if _active is None:
            raise RuntimeError(
                "No ModelOptSession is active: ModelOpt's internal functions run inside a public "
                "ModelOpt call, or inside `with ModelOptSession():`."
            )
        return _active

    @classmethod
    @contextlib.contextmanager
    def ensure(cls, model: _Model) -> Generator["ModelOptSession", None, None]:
        """The session a public call on ``model`` runs in, for the call's duration.

        That is the active one, bound to ``model``, else one set up for this call alone.
        """
        if _active is not None:
            yield _active.bind(model)
            return
        with cls().bind(model) as session:
            yield session

    def __enter__(self) -> "ModelOptSession":
        global _active
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
        # The workflow ends with its block: the session lets go of its model.
        self._model = None

    @property
    def model(self) -> "_Model | None":
        """The model this session is bound to, or ``None`` until a call binds it."""
        return self._model

    def bind(self, model: _Model) -> "ModelOptSession":
        """Bind this session to ``model``, the model its workflow works on, and return it.

        Binding it again to that model or to one of its submodules changes nothing; binding it to
        the model that contains it, such as a compiled or PEFT wrapper, moves the binding there.
        Any other model raises. The session holds the model it is bound to, so one the workflow
        drops stays bound until the session's block ends, and no garbage collection, which each
        rank runs at its own time, changes what a call binds to. Binding reads the model's type off
        the model it binds to.
        """
        bound = self._model
        if bound is None or (model is not bound and _contains(model, bound)):
            self._model = model
            self.model_type = hf_model_type(model)
        elif not _contains(bound, model):
            raise ValueError(
                f"This ModelOptSession describes a workflow on another model, a "
                f"{type(bound).__name__}, so it cannot be used for this {type(model).__name__}: a "
                "session describes one model."
            )
        return self

    def __getstate__(self) -> dict[str, object]:
        # The binding belongs to this process; a copy describes the same workflow without it.
        state = self.__dict__.copy()
        state["_model"] = None
        return state


# The process's active session: ``with ModelOptSession():`` sets it for the block.
_active: ModelOptSession | None = None


def ensure_modelopt_session(call: Callable[_P, _R]) -> Callable[_P, _R]:
    """Make ``call``, a public ModelOpt call on the model it takes first, run in a session.

    It runs in the active session, bound to that model, else in one set up for the call alone
    (:meth:`ModelOptSession.ensure`).

    This is for old code, which establishes no ModelOptSession for its pipeline: each of its calls
    gets a session of its own, so it behaves as it did before ModelOptSession. A pipeline that
    establishes one (``with ModelOptSession():``) runs every call in that session, which the call
    binds to its model.

    That model has to be the root model the ModelOpt workflow works on, such as a whole VLM rather
    than its language model: the session reads the model type off it, and checks it whole, as the
    export will write it. A call may take a part of the root only in a session already bound to
    the root, which binding the part leaves in place, as when the language model of a VLM is
    calibrated in a session bound to the VLM. Outside a session, a call on a part gets a session
    bound to the part alone, which reads the part's own model type.
    """
    first = next(iter(inspect.signature(call).parameters))

    @functools.wraps(call)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if not args and first not in kwargs:
            return call(*args, **kwargs)  # which raises the TypeError of a missing argument
        with ModelOptSession.ensure(cast("_Model", args[0] if args else kwargs[first])):
            return call(*args, **kwargs)

    return run


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
