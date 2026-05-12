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

"""Regression test for ``FunctionTarget`` kwargs dispatch.

The bypass-distillation factory stitches teacher and student block outputs into
a per-block loss function using ``InputArgs(target=...)`` and ``InputArgs(input=...)``
adapters (see ``stitched_model_factory.py:~545``). The loss function is then
invoked by ``StitchedModule.forward`` at ``core.py:600`` as
``node.target.function(*input_args.args, **input_args.kwargs)`` — i.e. with
**named kwargs**.

If sewing_kit ever switched to positional dispatch in stitch-declaration order,
asymmetric losses (KL divergence, relative-L2, anything where ``f(a, b) != f(b, a)``)
would silently swap their arguments. MSE-shaped losses would hide the regression
because they're symmetric. This test pins the contract.
"""

import torch

from modelopt.torch.puzzletron.sewing_kit.core import ExternalTarget, FunctionTarget, Needle
from modelopt.torch.puzzletron.sewing_kit.passage import InputArgs


def test_function_target_invoked_with_kwargs_not_positional():
    """The function callable must receive only kwargs (no positional args)."""
    received: dict[str, object] = {}

    def record_call(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = dict(kwargs)
        # The output stitch needs *something* to carry — return a sentinel scalar.
        return torch.tensor(0.0)

    loss_target = FunctionTarget("loss_fn", record_call)
    teacher_value = torch.full((2, 3), 7.0)
    student_value = torch.full((2, 3), 11.0)

    # Stitch order is intentionally reversed from the real factory: declare
    # student-first, teacher-second. If dispatch were positional-in-declaration-
    # order, ``input`` would receive the teacher value and ``target`` the student
    # value — which the assertions below would catch.
    stitched = (
        Needle()
        .stitch(
            ExternalTarget().output(
                name="student_act",
                adapter=lambda v: InputArgs(input=v),
            ),
            loss_target.input(),
        )
        .stitch(
            ExternalTarget().output(
                name="teacher_act",
                adapter=lambda v: InputArgs(target=v),
            ),
            loss_target.input(),
        )
        .stitch(
            loss_target.output(),
            ExternalTarget().output(name="loss"),
        )
        .knot()
    )

    stitched(
        {},
        {"student_act": student_value, "teacher_act": teacher_value},
    )

    assert received["args"] == (), (
        f"FunctionTarget called with positional args {received['args']!r}. "
        f"Sewing-kit must dispatch with kwargs only; positional dispatch would "
        f"silently swap input/target for asymmetric losses."
    )
    assert set(received["kwargs"].keys()) == {"input", "target"}
    assert torch.equal(received["kwargs"]["input"], student_value)
    assert torch.equal(received["kwargs"]["target"], teacher_value)


def test_function_target_kwargs_independent_of_stitch_order():
    """Same as the test above, but with the *real factory's* stitch order
    (teacher first, student second). Both orders must produce identical kwargs
    — the InputArgs.__add__ kwargs merge is order-independent for distinct
    keys."""
    received: dict[str, object] = {}

    def record_call(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = dict(kwargs)
        return torch.tensor(0.0)

    loss_target = FunctionTarget("loss_fn", record_call)
    teacher_value = torch.full((2, 3), 13.0)
    student_value = torch.full((2, 3), 17.0)

    stitched = (
        Needle()
        .stitch(
            ExternalTarget().output(
                name="teacher_act",
                adapter=lambda v: InputArgs(target=v),
            ),
            loss_target.input(),
        )
        .stitch(
            ExternalTarget().output(
                name="student_act",
                adapter=lambda v: InputArgs(input=v),
            ),
            loss_target.input(),
        )
        .stitch(
            loss_target.output(),
            ExternalTarget().output(name="loss"),
        )
        .knot()
    )

    stitched(
        {},
        {"teacher_act": teacher_value, "student_act": student_value},
    )

    assert received["args"] == ()
    assert set(received["kwargs"].keys()) == {"input", "target"}
    assert torch.equal(received["kwargs"]["input"], student_value)
    assert torch.equal(received["kwargs"]["target"], teacher_value)
