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

import copy

import torch
import torch.nn.functional as F
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

from modelopt.torch.puzzletron.distillation.flash_kld import TrainingFlashKLD
from modelopt.torch.puzzletron.distillation.global_kd_recipe import (
    _align_dtensor_to_module_mesh,
    _distillation_lm_head,
    _install_distillation_head_passthrough,
    _project_teacher_hidden_on_reference_mesh,
    _refresh_pp_hidden_output_meta,
    _WeightedObjectiveMixin,
)
from modelopt.torch.puzzletron.distillation.loss import ChunkedCrossEntropy, KDLoss


class _CountingLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)
        self.forward_count = 0

    def forward(self, value):
        self.forward_count += 1
        return super().forward(value)


def _naive_losses(
    student_hidden,
    student_head,
    teacher_hidden,
    teacher_head,
    labels,
    *,
    temperature,
):
    valid = labels != -100
    denominator = valid.sum()
    student_logits = student_head(student_hidden)
    with torch.no_grad():
        teacher_logits = teacher_head(teacher_hidden)
    ce = F.cross_entropy(student_logits.float(), labels, ignore_index=-100, reduction="sum")
    ce = ce / denominator
    student_logprob = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logprob = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    kd_per_token = F.kl_div(
        student_logprob,
        teacher_logprob,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    kd = kd_per_token.masked_select(valid).sum() / denominator * temperature**2
    return ce, kd


def test_combined_flash_kld_matches_full_logits_and_gradients():
    torch.manual_seed(7)
    labels = torch.tensor([1, 4, -100, 2, 0])
    student_hidden = torch.randn(5, 3, requires_grad=True)
    teacher_hidden = torch.randn(5, 3)
    student_head = _CountingLinear(3, 7)
    teacher_head = _CountingLinear(3, 7)
    reference_hidden = student_hidden.detach().clone().requires_grad_(True)
    reference_student_head = copy.deepcopy(student_head)
    reference_teacher_head = copy.deepcopy(teacher_head)
    temperature = 1.7

    loss = TrainingFlashKLD(
        token_chunk_size=2,
        temperature=temperature,
        checkpoint_chunks=False,
    )
    ce, kd = loss(
        student_hidden,
        student_head,
        labels,
        teacher_hidden=teacher_hidden,
        teacher_project=lambda hidden, _reference: teacher_head(hidden),
        compute_ce=True,
        compute_kd=True,
    )
    (0.3 * ce + 0.7 * kd).backward()

    reference_ce, reference_kd = _naive_losses(
        reference_hidden,
        reference_student_head,
        teacher_hidden,
        reference_teacher_head,
        labels,
        temperature=temperature,
    )
    (0.3 * reference_ce + 0.7 * reference_kd).backward()

    torch.testing.assert_close(ce, reference_ce)
    torch.testing.assert_close(kd, reference_kd)
    torch.testing.assert_close(student_hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(student_head.weight.grad, reference_student_head.weight.grad)
    assert student_head.forward_count == 3
    assert teacher_head.forward_count == 3


def test_ce_only_does_not_project_teacher():
    torch.manual_seed(11)
    student_hidden = torch.randn(4, 3, requires_grad=True)
    student_head = _CountingLinear(3, 5)
    labels = torch.tensor([0, 1, 2, -100])
    loss = TrainingFlashKLD(token_chunk_size=2, checkpoint_chunks=False)

    ce, kd = loss(
        student_hidden,
        student_head,
        labels,
        teacher_hidden=torch.randn_like(student_hidden),
        teacher_project=lambda *_args: (_ for _ in ()).throw(
            AssertionError("teacher projection must be skipped")
        ),
        compute_ce=True,
        compute_kd=False,
    )
    ce.backward()

    reference = (
        F.cross_entropy(
            student_head(student_hidden.detach()),
            labels,
            ignore_index=-100,
            reduction="sum",
        )
        / (labels != -100).sum()
    )
    torch.testing.assert_close(ce.detach(), reference.detach())
    assert kd.item() == 0.0
    assert student_head.forward_count == 3


def test_kd_only_skips_ce_but_respects_label_mask():
    torch.manual_seed(13)
    student_hidden = torch.randn(3, 2, requires_grad=True)
    teacher_hidden = torch.randn(3, 2)
    student_head = _CountingLinear(2, 4)
    teacher_head = _CountingLinear(2, 4)
    labels = torch.tensor([-100, 1, 2])
    loss = TrainingFlashKLD(token_chunk_size=1, checkpoint_chunks=False)

    ce, kd = loss(
        student_hidden,
        student_head,
        labels,
        teacher_hidden=teacher_hidden,
        teacher_project=lambda hidden, _reference: teacher_head(hidden),
        compute_ce=False,
        compute_kd=True,
    )

    assert ce.item() == 0.0
    assert kd.item() >= 0.0
    kd.backward()
    assert student_hidden.grad is not None


def test_distillation_passthrough_retains_trainable_lm_head():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(3, 7, bias=False)

    model = Model()
    original_head = model.lm_head
    original_weight = original_head.weight
    hidden = torch.randn(2, 3)

    assert _install_distillation_head_passthrough([model]) == 1

    torch.testing.assert_close(model.lm_head(hidden), hidden)
    assert model.lm_head is original_head
    assert _distillation_lm_head(model) is original_head
    assert any(parameter is original_weight for parameter in model.parameters())
    assert list(model.state_dict()) == ["lm_head.weight"]
    assert original_weight.requires_grad
    assert _install_distillation_head_passthrough([model]) == 0


def test_pp_hidden_output_metadata_uses_hidden_width():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(3, 7, bias=False)

    model = Model()
    _install_distillation_head_passthrough([model])
    stage = type(
        "Stage",
        (),
        {
            "submod": model,
            "is_last": True,
            "inputs_meta": (torch.empty(1, 4, 3, device="meta"),),
            "_outputs_meta": (
                torch.empty(1, 4, 7, device="meta"),
                torch.empty(1, 4, 2, device="meta"),
            ),
        },
    )()
    pp = type("PP", (), {"info": type("Info", (), {"stages": [stage]})()})()

    assert _refresh_pp_hidden_output_meta(pp) == 1
    assert tuple(stage._outputs_meta[0].shape) == (1, 4, 3)
    assert tuple(stage._outputs_meta[1].shape) == (1, 4, 2)


def test_weighted_objectives_share_hidden_flash_kld_for_main_and_mtp():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 5, bias=False)
            self.mtp_outputs_are_logits = False

    torch.manual_seed(19)
    student = Model()
    teacher = Model()
    _install_distillation_head_passthrough([student, teacher])
    main_student = torch.randn(1, 4, 2, requires_grad=True)
    mtp_student = torch.randn(1, 4, 2, requires_grad=True)
    main_teacher = torch.randn(1, 4, 2)
    mtp_teacher = torch.randn(1, 4, 2)
    labels = torch.tensor([[0, 1, 2, 3]])
    engine = TrainingFlashKLD(token_chunk_size=2, checkpoint_chunks=False)

    recipe = object.__new__(_WeightedObjectiveMixin)
    recipe.objective = {
        "main_ce": 0.25,
        "main_kd": 0.25,
        "mtp_ce": 0.25,
        "mtp_kd": 0.25,
    }
    recipe.main_flash_kld = engine
    recipe.mtp_flash_kld = engine
    recipe.loss_fn = ChunkedCrossEntropy(chunk_size=2)
    recipe.main_kd_loss_fn = KDLoss(chunk_size=2)
    recipe.mtp_kd_loss_fn = KDLoss(chunk_size=2)
    recipe._objective_buffers = {name: [] for name in recipe.objective}
    recipe._loss_topology_logged = True
    recipe.needs_teacher = True
    recipe.teacher_pp = None
    recipe.teacher_model = teacher

    total, terms = recipe._objective_loss(
        (main_student, mtp_student),
        (main_teacher, mtp_teacher),
        labels,
        student,
        num_label_tokens=4,
    )

    expected_main_ce, expected_main_kd = engine(
        main_student,
        student.lm_head,
        labels,
        teacher_hidden=main_teacher,
        teacher_project=lambda hidden, _reference: teacher.lm_head._puzzletron_projection_forward(
            hidden
        ),
    )
    mtp_labels = torch.roll(labels, shifts=-1, dims=-1)
    mtp_labels[..., -1] = -100
    expected_mtp_ce, expected_mtp_kd = engine(
        mtp_student,
        student.lm_head,
        mtp_labels,
        teacher_hidden=mtp_teacher,
        teacher_project=lambda hidden, _reference: teacher.lm_head._puzzletron_projection_forward(
            hidden
        ),
        num_label_tokens=4,
    )

    torch.testing.assert_close(terms["main_ce"], expected_main_ce)
    torch.testing.assert_close(terms["main_kd"], expected_main_kd)
    torch.testing.assert_close(terms["mtp_ce"], expected_mtp_ce)
    torch.testing.assert_close(terms["mtp_kd"], expected_mtp_kd)
    torch.testing.assert_close(total, 0.25 * sum(terms.values()))


def _tp_teacher_projection_job(_rank: int, _size: int) -> None:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(4, 8, bias=False)

    torch.manual_seed(23)
    teacher_mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
    student_mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
    teacher = Model()
    full_weight = teacher.lm_head.weight.detach().clone()
    teacher.lm_head = parallelize_module(
        teacher.lm_head,
        teacher_mesh,
        ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(-1),
            use_local_output=False,
        ),
    )
    assert _install_distillation_head_passthrough([teacher]) == 1
    full_hidden = torch.randn(3, 4)
    reference_logits = distribute_tensor(torch.empty(3, 8), student_mesh, (Shard(-1),))

    student_hidden = full_hidden.clone().requires_grad_()
    aligned_hidden = _align_dtensor_to_module_mesh(student_hidden, teacher.lm_head)
    assert aligned_hidden.placements == (Replicate(),)
    student_logits = teacher.lm_head._puzzletron_projection_forward(aligned_hidden)
    torch.testing.assert_close(student_logits.full_tensor(), F.linear(full_hidden, full_weight))
    student_logits.full_tensor().sum().backward()
    assert student_hidden.grad is not None

    actual = _project_teacher_hidden_on_reference_mesh(
        full_hidden.clone(),
        teacher.lm_head,
        reference_logits,
    )

    assert actual.device_mesh is student_mesh
    assert actual.placements == reference_logits.placements
    torch.testing.assert_close(actual.full_tensor(), F.linear(full_hidden, full_weight))

    replicated_reference_logits = distribute_tensor(torch.empty(3, 8), student_mesh, (Replicate(),))
    replicated_actual = _project_teacher_hidden_on_reference_mesh(
        full_hidden.clone(),
        teacher.lm_head,
        replicated_reference_logits,
    )
    assert replicated_actual.device_mesh is student_mesh
    assert replicated_actual.placements == (Replicate(),)
    torch.testing.assert_close(replicated_actual.full_tensor(), F.linear(full_hidden, full_weight))

    sharded_hidden = distribute_tensor(full_hidden, teacher_mesh, (Shard(-1),))
    sharded_actual = _project_teacher_hidden_on_reference_mesh(
        sharded_hidden,
        teacher.lm_head,
        reference_logits,
    )
    torch.testing.assert_close(sharded_actual.full_tensor(), F.linear(full_hidden, full_weight))


def test_tp_teacher_projection_redistributes_hidden_and_rewraps_logits():
    spawn_multiprocess_job(size=2, job=_tp_teacher_projection_job, backend="gloo")
