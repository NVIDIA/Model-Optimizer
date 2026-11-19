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

"""A ModelOpt workflow runs in one ModelOptSession, which describes one model."""

import copy
import gc
import pickle
import weakref
from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.opt import ModelOptSession
from modelopt.torch.opt.session import ensure_modelopt_session


def test_a_session_describes_one_model():
    """A session binds to the first model a call gives it, and holds it. That model's submodules and
    the model containing it are the same model, as a VLM's language model and the VLM are; any other
    model is refused, also once the workflow drops that model. A copy of the session is unbound and
    holds no live objects."""
    vlm = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    language_model = vlm[0]

    session = ModelOptSession().bind(language_model)
    assert session.bind(vlm) is session and session.model is vlm
    assert session.bind(language_model).model is vlm
    with pytest.raises(ValueError, match="a session describes one model"):
        session.bind(torch.nn.Linear(2, 2))
    session.runtime.export_context = object()
    for copied in (copy.deepcopy(session), pickle.loads(pickle.dumps(session))):
        assert copied.model is None and copied.runtime.export_context is None
        copied.bind(torch.nn.Linear(2, 2))
    held = weakref.ref(vlm)
    del vlm, language_model
    gc.collect()
    assert held() is not None and session.model is held()
    with pytest.raises(ValueError, match="a session describes one model"):
        session.bind(torch.nn.Linear(2, 2))


@pytest.mark.parametrize("collected", [False, True])
def test_a_wrapper_the_workflow_drops_stays_the_model_bound(collected):
    """A session bound to a wrapper of its model, such as a compiled one, holds the wrapper: once
    the workflow drops it, a call on the model or a part of it is a call on a part of the wrapper,
    and any other model, a new wrapper included, is refused. Whether a garbage collection, which
    each rank runs at its own time, has run decides nothing. A copy binds afresh."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    wrapper = torch.nn.Sequential(model)
    wrapper.config = SimpleNamespace(model_type="compiled")
    session = ModelOptSession().bind(wrapper)
    del wrapper
    if collected:
        gc.collect()

    for part in (model, model[0], model[1]):
        assert session.bind(part).model_type == "compiled"
    for other in (torch.nn.Sequential(model), torch.nn.Linear(2, 2)):
        with pytest.raises(ValueError, match="a session describes one model"):
            session.bind(other)
    for copied in (copy.deepcopy(session), pickle.loads(pickle.dumps(session))):
        assert copied.bind(model).model is model and copied.model_type is None


def test_a_pipeline_is_the_model_its_components_are_parts_of():
    """A diffusers pipeline is quantized component by component and exported whole, so a session
    bound to a component moves up to the pipeline, and binding a component leaves it there."""
    transformer = torch.nn.Linear(2, 2)
    pipe = SimpleNamespace(components={"transformer": transformer, "scheduler": object()})

    session = ModelOptSession().bind(transformer)
    assert session.bind(pipe).model is pipe
    assert session.bind(transformer).model is pipe
    with pytest.raises(ValueError, match="a session describes one model"):
        session.bind(torch.nn.Linear(2, 2))


def test_binding_reads_the_models_type():
    """The model's type is read off the model the session binds to, and only there: the outer
    model's once the binding moves up to it, unchanged by a submodule, None for a model without."""
    vlm = torch.nn.Module()
    vlm.config = SimpleNamespace(model_type="qwen3_vl")
    vlm.language_model = torch.nn.Linear(2, 2)
    vlm.language_model.config = SimpleNamespace(model_type="qwen3")

    session = ModelOptSession().bind(vlm.language_model)
    assert session.model_type == "qwen3"
    session.bind(vlm)
    assert session.model_type == "qwen3_vl"
    session.bind(vlm.language_model)
    assert session.model_type == "qwen3_vl"
    assert ModelOptSession().bind(torch.nn.Linear(2, 2)).model_type is None


def test_a_workflow_runs_in_one_session():
    """A public call runs in the active session, bound to its model, so the calls of a workflow,
    nested ones too, share one; outside any, each call runs in one of its own, which ends with it.
    A second session opened while one is active is refused, and an exception ends the active one."""

    @ensure_modelopt_session
    def call(model, inner=None):
        session = ModelOptSession.active()
        return session, inner(model) if inner else None

    model = torch.nn.Linear(2, 2)
    with ModelOptSession() as workflow:
        assert ModelOptSession.current() is workflow
        assert call(model, inner=call) == (workflow, (workflow, None))
        assert workflow.model is model
        with pytest.raises(RuntimeError, match="already active"), ModelOptSession():
            pass
        assert ModelOptSession.current() is workflow
    assert ModelOptSession.current() is None
    # The workflow's block ended, and the session let go of its model.
    assert workflow.model is None

    with pytest.raises(TypeError):
        call()  # the call's own error for its missing model, not one from looking the model up
    first, (nested, _) = call(model, inner=call)
    assert nested is first and first is not workflow
    assert call(model)[0] is not first
    assert ModelOptSession.current() is None
    with pytest.raises(RuntimeError, match="No ModelOptSession is active"):
        ModelOptSession.active()

    with pytest.raises(ZeroDivisionError), ModelOptSession():
        1 / 0
    assert ModelOptSession.current() is None
