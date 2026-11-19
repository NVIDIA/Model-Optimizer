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

"""A ModelOpt workflow runs in one ModelOptSession, which binds each of its models by name."""

import copy
import gc
import pickle
import weakref
from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.opt import ModelOptSession
from modelopt.torch.opt.session import ensure_modelopt_session_binding


def test_a_session_describes_one_model():
    """A session binds to the first model a call gives it, its workflow's root, and holds it. A
    call on a part of the root, as a VLM's language model is, runs in it; any other model is
    refused, one containing the root included, also once the workflow drops the root. A copy of the
    session is unbound."""
    vlm = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    language_model = vlm[0]

    session = ModelOptSession().bind(vlm)
    assert session.bind(language_model) is session and session.model is vlm
    for other in (torch.nn.Sequential(vlm), torch.nn.Linear(2, 2)):
        with pytest.raises(RuntimeError, match="a session describes one model"):
            session.bind(other)
    with pytest.raises(RuntimeError, match="a session describes one model"):
        ModelOptSession().bind(language_model).bind(vlm)
    for copied in (copy.deepcopy(session), pickle.loads(pickle.dumps(session))):
        assert copied.model is None
        copied.bind(torch.nn.Linear(2, 2))
    held = weakref.ref(vlm)
    del vlm, language_model
    gc.collect()
    assert held() is not None and session.model is held()
    with pytest.raises(RuntimeError, match="a session describes one model"):
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
        with pytest.raises(RuntimeError, match="a session describes one model"):
            session.bind(other)
    for copied in (copy.deepcopy(session), pickle.loads(pickle.dumps(session))):
        assert copied.bind(model).model is model and copied.model_type is None


def test_a_pipeline_is_the_model_its_components_are_parts_of():
    """A diffusers pipeline is quantized component by component and exported whole, so a session
    bound to the pipeline takes its components, and one bound to a component refuses the pipeline."""
    transformer = torch.nn.Linear(2, 2)
    pipe = SimpleNamespace(components={"transformer": transformer, "scheduler": object()})

    session = ModelOptSession().bind(pipe)
    assert session.bind(transformer).model is pipe
    with pytest.raises(RuntimeError, match="a session describes one model"):
        session.bind(torch.nn.Linear(2, 2))
    with pytest.raises(RuntimeError, match="a session describes one model"):
        ModelOptSession().bind(transformer).bind(pipe)


def test_binding_reads_the_models_type():
    """The model's type is read off the model the session binds to, its root, and only there: a
    part bound later leaves it, and a model without one reads None."""
    vlm = torch.nn.Module()
    vlm.config = SimpleNamespace(model_type="qwen3_vl")
    vlm.language_model = torch.nn.Linear(2, 2)
    vlm.language_model.config = SimpleNamespace(model_type="qwen3")

    session = ModelOptSession().bind(vlm)
    assert session.model_type == "qwen3_vl"
    session.bind(vlm.language_model)
    assert session.model_type == "qwen3_vl"
    assert ModelOptSession().bind(vlm.language_model).model_type == "qwen3"
    assert ModelOptSession().bind(torch.nn.Linear(2, 2)).model_type is None


def test_a_workflow_runs_in_one_session():
    """A public call runs in the session a block opened, bound to its model, so the calls of a
    workflow share one. A session opened while one is active is refused, as is opening the default
    one, and an exception ends the active one."""

    def call(model):
        ensure_modelopt_session_binding(model)
        return ModelOptSession.active()

    model = torch.nn.Linear(2, 2)
    with ModelOptSession() as workflow:
        assert ModelOptSession.current() is workflow
        assert call(model) is workflow and call(model) is workflow
        assert workflow.model is model
        with pytest.raises(RuntimeError, match="already active"), ModelOptSession():
            pass
        assert ModelOptSession.current() is workflow
    assert ModelOptSession.current() is None
    # The workflow's block ended, and the session let go of its model.
    assert workflow.model is None

    with pytest.raises(RuntimeError, match="never opened"), ModelOptSession.active():
        pass
    with pytest.raises(ZeroDivisionError), ModelOptSession():
        1 / 0
    assert ModelOptSession.current() is None


def test_code_that_opens_no_session_runs_in_the_default_one():
    """Outside any block, calls run in the process's default session, which is never closed: it
    follows the model the code moves on to, a model containing it included, keeps it for a call on
    a part, and holds it weakly, so code that drops the model frees it. A name given explicitly
    stays strict."""
    default = ModelOptSession.active()
    assert ModelOptSession.current() is None and ModelOptSession.active() is default
    vlm = torch.nn.Sequential(torch.nn.Linear(2, 2))
    assert ensure_modelopt_session_binding(vlm[0]).model is vlm[0]
    assert ensure_modelopt_session_binding(vlm).model is vlm and default.model is vlm
    assert ensure_modelopt_session_binding(vlm[0]).model is vlm

    other = torch.nn.Linear(2, 2)
    assert ensure_modelopt_session_binding(other).model is other and default.model is other
    with pytest.raises(RuntimeError, match="describes one model"):
        default.bind(vlm, name="model")
    held = weakref.ref(other)
    del other
    gc.collect()
    assert held() is None and default.model is None


def test_a_session_binds_each_model_of_its_workflow_by_name():
    """A distillation binds its teacher by name. A call on the teacher, or on a part of it, runs
    under that name, the innermost where the student holds its teacher, and the name refuses
    another model, in the default session too."""
    student, teacher = torch.nn.Module(), torch.nn.Module()
    student.config = SimpleNamespace(model_type="student")
    teacher.config = SimpleNamespace(model_type="teacher")
    student.layer, teacher.layer = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    student.teacher = teacher  # as ModelOpt's distillation holds its teacher

    with ModelOptSession() as session:
        session.bind(student).bind(teacher, name="teacher")
        assert session.binding(student.layer).name == "model"
        assert ensure_modelopt_session_binding(teacher.layer).name == "teacher"
        assert session.binding(teacher).model_type == "teacher" and session.model_type == "student"
        with pytest.raises(RuntimeError, match="describes one model"):
            session.bind(torch.nn.Linear(2, 2), name="teacher")
    with pytest.raises(RuntimeError, match="binds no model"):
        ModelOptSession().binding(teacher)
    ModelOptSession.active().bind(teacher, name="teacher")
    with pytest.raises(RuntimeError, match="describes one model"):
        ModelOptSession.active().bind(torch.nn.Linear(2, 2), name="teacher")
