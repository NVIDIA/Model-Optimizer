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

"""A ModelOpt workflow runs in one ModelOptSession, which describes the models it binds by name."""

import copy
import pickle
from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.opt import ModelOptSession


def test_a_session_describes_one_model():
    """A session binds to the first model the workflow gives it, its root. Binding a part of the
    root, as a VLM's language model is, keeps the root; any other model is refused, one containing
    the root included. A copy of the session is unbound and holds no live objects."""
    vlm = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    language_model = vlm[0]

    session = ModelOptSession().bind(vlm)
    assert session.bind(language_model) is session and session.model is vlm
    for other in (torch.nn.Sequential(vlm), torch.nn.Linear(2, 2)):
        with pytest.raises(RuntimeError, match="a session describes one model"):
            session.bind(other)
    with pytest.raises(RuntimeError, match="a session describes one model"):
        ModelOptSession().bind(language_model).bind(vlm)
    session.runtime.export_context = object()
    for copied in (copy.deepcopy(session), pickle.loads(pickle.dumps(session))):
        assert copied.model is None and copied.runtime.export_context is None
        copied.bind(torch.nn.Linear(2, 2))


def test_a_pipeline_is_the_model_its_components_are_parts_of():
    """A diffusers pipeline is quantized component by component and exported whole, so a session
    bound to the pipeline takes its components, also one the pipeline keeps where the session cannot
    see it, as LTX-2 keeps its transformer; a session bound to a component refuses the pipeline."""
    transformer = torch.nn.Linear(2, 2)
    pipe = SimpleNamespace(components={"transformer": transformer, "scheduler": object()})

    session = ModelOptSession().bind(pipe)
    assert session.bind(transformer).model is pipe
    assert session.binding(torch.nn.Linear(2, 2)).model is pipe
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
    assert session.bind(vlm.language_model).binding(vlm.language_model).model_type == "qwen3_vl"
    assert ModelOptSession().bind(vlm.language_model).model_type == "qwen3"
    assert ModelOptSession().bind(torch.nn.Linear(2, 2)).model_type is None


def test_a_workflow_runs_in_one_session():
    """The calls in a block run in the session it opened, which describes the models the workflow
    binds and refuses any other. A session opened while one is active is refused, as is opening the
    default one, and an exception ends the active one."""
    model = torch.nn.Linear(2, 2)
    with ModelOptSession().bind(model) as workflow:
        assert ModelOptSession.current() is workflow and ModelOptSession.active() is workflow
        assert workflow.binding(model).model is model
        with pytest.raises(RuntimeError, match="binds no model"):
            workflow.binding(torch.nn.Linear(2, 2))
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
    """Outside any block, calls run in the process's default session, which binds nothing: a call
    reads what it needs off the model it is given, a part's own type included, as before
    ModelOptSession."""
    default = ModelOptSession.active()
    assert ModelOptSession.current() is None and ModelOptSession.active() is default
    vlm = torch.nn.Module()
    vlm.config = SimpleNamespace(model_type="qwen3_vl")
    vlm.language_model = torch.nn.Linear(2, 2)
    vlm.language_model.config = SimpleNamespace(model_type="qwen3")

    assert default.binding(vlm).model_type == "qwen3_vl"
    assert default.binding(vlm.language_model).model_type == "qwen3"
    with pytest.raises(RuntimeError, match="binds no model"):
        default.bind(vlm)
    assert default.model is None


def test_a_session_binds_each_model_of_its_workflow_by_name():
    """A distillation binds its teacher by name. A call on the teacher, or on a part of it, reads the
    teacher's binding, the innermost where the student holds its teacher, and the name refuses
    another model."""
    student, teacher = torch.nn.Module(), torch.nn.Module()
    student.config = SimpleNamespace(model_type="student")
    teacher.config = SimpleNamespace(model_type="teacher")
    student.layer, teacher.layer = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    student.teacher = teacher  # as ModelOpt's distillation holds its teacher

    with ModelOptSession() as session:
        session.bind(student).bind(teacher, name="teacher")
        assert session.binding(student.layer).name == "model"
        assert session.binding(teacher.layer).name == "teacher"
        assert session.binding(teacher).model_type == "teacher" and session.model_type == "student"
        with pytest.raises(RuntimeError, match="describes one model"):
            session.bind(torch.nn.Linear(2, 2), name="teacher")
    with pytest.raises(RuntimeError, match="binds no model"):
        ModelOptSession().binding(teacher)
