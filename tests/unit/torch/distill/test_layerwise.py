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
import warnings
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.vision_models import get_tiny_mobilenet_and_input
from torch.distributed.device_mesh import DeviceMesh

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
from modelopt.torch.distill.plugins.transformers.adapter import (
    TransformersLocalDistillationAdapter,
    _TransformersRuntime,
)


def _tp_statistics_require_every_boundary_job(rank, size):
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=3))
    try:
        mesh = DeviceMesh.from_group(group, "cpu", mesh_dim_names=("tp",))
        adapter = TransformersLocalDistillationAdapter(mesh)
        adapter.context.process_group = group
        runtime = _TransformersRuntime(adapter.context, mtd.LocalDistillationConfig())
        runtime.unit_names = ["projection"]
        runtime.objective_tp_units["projection"] = mesh
        totals = {"projection": torch.ones(4)} if rank == 0 else {}
        with pytest.raises(ValueError):
            runtime.reduce_statistics(totals)

        totals = {"projection": torch.tensor([rank + 1.0, 2.0, 3.0, 1.0])}
        reduced = runtime.reduce_statistics(totals)
        torch.testing.assert_close(reduced["projection"], torch.tensor([3.0, 4.0, 6.0, 1.0]))

        from examples.local_quantization_distillation.evaluate import evaluate_pair

        model = torch.nn.Linear(1, 1)
        batches = [] if rank == 0 else [{}]
        with pytest.raises(ValueError, match="microbatch schedules"):
            evaluate_pair(model, model, batches, adapter=adapter)

        class LogitsOnly(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scores = torch.nn.Parameter(torch.tensor([-1.0, 1.0]))

            def forward(self, input_ids):
                return self.scores.expand(*input_ids.shape, 2)

        logits_model = LogitsOnly()
        labels = torch.tensor([[2 if rank == 0 else 0]])
        invalid = {"input_ids": torch.ones(1, 1, dtype=torch.long), "labels": labels}
        with pytest.raises(ValueError, match="outside the output vocabulary"):
            evaluate_pair(logits_model, logits_model, [invalid], adapter=adapter)

        from modelopt.torch.distill.plugins.transformers import TransformersFamilyAdapter

        class EmptyRefusal(TransformersFamilyAdapter):
            def resolve_units(self, student, teacher, config):
                if rank == 0:
                    raise ValueError()
                return super().resolve_units(student, teacher, config)

        refusing = TransformersLocalDistillationAdapter(mesh, family_adapter=EmptyRefusal())
        refusing.context.process_group = group
        projection = torch.nn.Sequential(torch.nn.Linear(1, 1))
        with pytest.raises(ValueError):
            refusing.resolve_units(projection, projection, mtd.LocalDistillationConfig(units=["0"]))
        remaining = torch.ones(1)
        dist.all_reduce(remaining, group=group)
        torch.testing.assert_close(remaining, torch.tensor([2.0]))
    finally:
        dist.destroy_process_group(group)


def test_tp_statistics_reject_missing_boundary_before_reduction(cpu_dist_workers):
    cpu_dist_workers.run(_tp_statistics_require_every_boundary_job)


def test_local_diagnostics_full_vocabulary_and_document_targets():
    from examples.local_quantization_distillation.evaluate import diagnostic_statistics

    torch.manual_seed(5)
    teacher = torch.randn(2, 7, 13)
    candidate = teacher + torch.randn_like(teacher) * 0.3
    tokens = torch.randint(0, 13, (2, 7))
    documents = torch.tensor([[1, 1, 1, 2, 2, 2, 0], [1, 1, 2, 2, 2, 0, 0]])
    labels = tokens.roll(-1, dims=1)
    labels[:, -1] = -100
    labels[:, :-1][documents[:, 1:] != documents[:, :-1]] = -100
    labels[documents == 0] = -100
    stats = diagnostic_statistics(
        teacher, candidate, labels, documents > 0, token_chunk=3, vocabulary_chunk=5
    )
    valid = documents > 0
    expected_kl = (
        torch.nn.functional.kl_div(candidate.log_softmax(-1), teacher.softmax(-1), reduction="none")
        .sum(-1)[valid]
        .double()
        .sum()
    )
    expected_nll = torch.nn.functional.cross_entropy(
        candidate.reshape(-1, 13), labels.reshape(-1), reduction="sum"
    )
    torch.testing.assert_close(stats[0], expected_nll.double(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(stats[2], expected_kl, rtol=1e-5, atol=1e-6)
    assert int(stats[3]) == int(valid.sum()) > int(stats[1])
    identical = diagnostic_statistics(teacher, teacher, labels, valid, token_chunk=3)
    torch.testing.assert_close(identical[2], torch.zeros_like(identical[2]), atol=1e-6, rtol=0)

    from examples.local_quantization_distillation.evaluate import evaluate_pair

    class SharpModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.tensor([-1000.0, 0.0]))

        def forward(self, input_ids):
            return self.scores.expand(*input_ids.shape, 2)

    sharp = SharpModel()
    batch = {
        "input_ids": torch.zeros(1, 3, dtype=torch.long),
        "labels": torch.zeros(1, 3, dtype=torch.long),
        "valid_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    report = evaluate_pair(sharp, sharp, [batch])
    assert report["kld_teacher_candidate"] == 0
    assert report["ppl"] is None and report["teacher_ppl"] is None
    assert report["mean_cross_entropy"] == 1000

    class DocumentModel(SharpModel):
        def forward(self, input_ids, *, logits_to_keep=1, **kwargs):
            assert logits_to_keep == 0  # Diagnostics need every token; repair needs only one logit.
            prefix = input_ids.cumsum(-1).float()
            return self.scores + torch.stack((prefix, -prefix), dim=-1)

    model = DocumentModel().train()
    adapter = TransformersLocalDistillationAdapter()
    packed = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 0]]),
        "_packed_seq_ids": torch.tensor([[1, 1, 2, 2, 0]]),
        "labels": torch.tensor([[1, -100, 1, -100, -100]]),
    }
    packed_report = evaluate_pair(model, model, [packed], adapter=adapter)
    independent = [
        {
            "input_ids": packed["input_ids"][:, start : start + 2],
            "labels": packed["labels"][:, start : start + 2],
        }
        for start in (0, 2)
    ]
    separate_report = evaluate_pair(model, model, independent)
    for key in ("kld_teacher_candidate", "mean_cross_entropy", "kl_tokens", "next_token_targets"):
        assert packed_report[key] == separate_report[key]
    assert packed_report["kl_tokens"] == 4 and packed_report["next_token_targets"] == 2
    assert model.training

    def broken_loader():
        yield packed
        raise RuntimeError()  # Empty messages must still fail collectively, not truncate metrics.

    with pytest.raises(ValueError, match="RuntimeError"):
        evaluate_pair(model, model, broken_loader(), adapter=adapter)


@pytest.mark.parametrize(
    "family_name",
    ["llama", "llama_hooked", "llama_quantized", "llama_biased", "muse_spark", "mistral_large4"],
)
def test_transformers_head_chunks_match_full_logits_and_restore(family_name):
    from _test_utils.transformers_families import build_model, cases

    from examples.local_quantization_distillation.evaluate import (
        diagnostic_statistics,
        evaluate_pair,
    )

    torch.manual_seed(19)
    base_family = "llama" if family_name.startswith("llama") else family_name
    pytest.importorskip(f"transformers.models.{base_family}")
    record = next(
        record for record in cases(families=(base_family,)) if record["family"] == base_family
    )
    teacher = build_model(record, "cpu").float().eval()
    candidate = copy.deepcopy(teacher)
    if family_name.endswith("_hooked"):
        for model in (teacher, candidate):
            model.get_output_embeddings().register_forward_pre_hook(lambda _, args: (args[0] * 2,))
            model.get_output_embeddings().register_forward_hook(lambda _, args, out: out * 1.3)
    head = candidate.get_output_embeddings()
    with torch.no_grad():
        head.weight.add_(torch.randn_like(head.weight) * 0.03)
    tokens = torch.randint(3, 30, (1, 9))
    if family_name == "llama_biased":
        head.bias = torch.nn.Parameter(torch.randn(head.weight.shape[0]) * 0.1)
    if family_name == "llama_quantized":
        import modelopt.torch.quantization as mtq

        expected_amax = []
        handle = head.register_forward_pre_hook(
            lambda _, args: expected_amax.append(args[0].detach().abs().amax())
        )
        try:
            with torch.no_grad():
                candidate(input_ids=tokens)
        finally:
            handle.remove()
        calibration_adapter = TransformersLocalDistillationAdapter()
        calibration_lengths = []
        handle = head.register_forward_pre_hook(
            lambda _, args: calibration_lengths.append(args[0].shape[-2])
        )
        try:
            mtq.quantize(
                candidate,
                {
                    "quant_cfg": [
                        {"quantizer_name": "*", "enable": False},
                        {
                            "quantizer_name": "lm_head.weight_quantizer",
                            "enable": True,
                            "cfg": {"num_bits": 8, "axis": 0},
                        },
                        {
                            "quantizer_name": "lm_head.input_quantizer",
                            "enable": True,
                            "cfg": {"num_bits": 8, "axis": None},
                        },
                    ],
                    "algorithm": {"method": "gptq", "block_size": 16, "fused": False},
                },
                forward_loop=lambda model: calibration_adapter.teacher_microbatch(
                    model, [], {"input_ids": tokens}, lambda *_: None
                ),
            )
        finally:
            handle.remove()
        assert calibration_lengths and set(calibration_lengths) == {tokens.shape[1]}
        torch.testing.assert_close(head.input_quantizer.amax, expected_amax[0], rtol=0, atol=0)
    labels = tokens.roll(-1, -1)
    labels[:, -1] = -100
    valid = torch.ones_like(tokens, dtype=torch.bool)
    with torch.no_grad():
        target, prediction = teacher(input_ids=tokens).logits, candidate(input_ids=tokens).logits
        expected = diagnostic_statistics(target, prediction, labels, valid)
    original = head.forward
    report = evaluate_pair(
        teacher,
        candidate,
        [{"input_ids": tokens, "labels": labels, "valid_mask": valid}],
        adapter=TransformersLocalDistillationAdapter(),
        token_chunk=3,
        vocabulary_chunk=7,
    )
    torch.testing.assert_close(
        report["kld_teacher_candidate"], (expected[2] / expected[3]).item(), rtol=2e-5, atol=1e-7
    )
    torch.testing.assert_close(
        report["mean_cross_entropy"], (expected[0] / expected[1]).item(), rtol=2e-6, atol=1e-7
    )
    assert report["projection_modes"] == [
        "chunked_head" if family_name in ("llama_hooked", "llama_quantized") else "chunked_linear"
    ]
    assert head.forward == original
    assert report["kl_tokens"] == 9 and report["next_token_targets"] == 8

    if family_name == "llama_hooked":
        handle = torch.nn.modules.module.register_module_forward_pre_hook(lambda module, args: None)
        try:
            with pytest.raises(ValueError):
                evaluate_pair(
                    teacher,
                    candidate,
                    [{"input_ids": tokens}],
                    adapter=TransformersLocalDistillationAdapter(),
                )
        finally:
            handle.remove()
        assert head.forward == original


@pytest.mark.parametrize("selected_head", [False, True])
def test_transformers_repair_skips_only_unselected_head(selected_head):
    from transformers import LlamaConfig, LlamaForCausalLM

    import modelopt.torch.quantization as mtq

    torch.manual_seed(7)
    teacher = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            use_cache=False,
        )
    ).eval()
    student = mtq.quantize(
        copy.deepcopy(teacher),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "enable": True,
                    "cfg": {"num_bits": 8, "axis": 0},
                },
            ],
            "algorithm": "max",
        },
    )
    outputs = []
    handle = teacher.lm_head.register_forward_hook(
        lambda _, args, output: outputs.append(output.shape)
    )
    original = teacher.lm_head.forward
    batch = {
        "input_ids": torch.randint(1, 32, (1, 7)),
        "valid_mask": torch.ones(1, 7, dtype=torch.bool),
    }
    try:
        result = mtd.local_distill(
            student,
            teacher,
            [batch],
            {
                "units": ["lm_head" if selected_head else "model.layers.0.mlp"],
                "updates": 1,
                "global_batch_size": 1,
            },
            adapter=TransformersLocalDistillationAdapter(),
        )
    finally:
        handle.remove()
    assert teacher.lm_head.forward == original
    assert outputs == [torch.Size([1, 7 if selected_head else 1, 32 if selected_head else 1])]
    unit = next(iter(result.metrics["history"][0]["units"].values()))
    assert unit["valid_tokens"] == 7


def test_local_quantization_zero_updates_preserves_ptq():
    import modelopt.torch.quantization as mtq

    teacher = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).eval()
    student = mtq.quantize(copy.deepcopy(teacher), mtq.INT8_WEIGHT_ONLY_CFG)
    example = torch.randn(2, 8)
    expected = student(example).detach()
    flags = [parameter.requires_grad for parameter in student.parameters()]

    result = mtd.local_distill(student, teacher, [], {"units": ["0", "1"], "updates": 0})

    assert result.student is student
    torch.testing.assert_close(result.student(example), expected, rtol=0, atol=0)
    assert flags == [parameter.requires_grad for parameter in student.parameters()]


@pytest.mark.parametrize(
    "invalid_selection",
    ["overlap", "shared", "unsupported_backend", "transformed", "missing_teacher"],
)
def test_local_quantization_preflight_is_transactional(invalid_selection, monkeypatch):
    """Invalid repair ownership must leave an existing PTQ model untouched."""
    import modelopt.torch.quantization as mtq

    teacher = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).eval()
    student = mtq.quantize(copy.deepcopy(teacher), mtq.INT8_WEIGHT_ONLY_CFG)
    if invalid_selection == "shared":
        student[1].weight = student[0].weight
        units = ["0", "1"]
    elif invalid_selection == "overlap":
        units = ["", "0"]
    elif invalid_selection == "transformed":
        from modelopt.torch.quantization.config import RotateConfig

        student[0].weight_quantizer._rotate = RotateConfig(enable=True, mode="rotate_back")
        units = ["0", "1"]
    elif invalid_selection == "missing_teacher":
        del teacher[1]
        units = ["0", "1"]
    else:
        from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
            _QUANT_FUNCTIONAL_BACKENDS,
        )

        monkeypatch.setitem(
            _QUANT_FUNCTIONAL_BACKENDS, "no_repair_adapter", lambda inputs, _: inputs.square()
        )
        student[0].weight_quantizer.backend = "no_repair_adapter"
        units = ["0", "1"]
    before = {name: tensor.clone() for name, tensor in student.state_dict().items()}
    flags = [parameter.requires_grad for parameter in student.parameters()]
    hooks = [
        (len(module._forward_hooks), len(module._forward_pre_hooks)) for module in student.modules()
    ]

    with pytest.raises(ValueError):
        mtd.local_distill(student, teacher, [torch.randn(8, 8)], {"units": units, "updates": 1})

    assert before.keys() == student.state_dict().keys()
    for name, tensor in student.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert flags == [parameter.requires_grad for parameter in student.parameters()]
    assert hooks == [
        (len(module._forward_hooks), len(module._forward_pre_hooks)) for module in student.modules()
    ]


def test_local_repair_rejects_late_nonfinite_gradient_before_any_update(monkeypatch):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.repair import QuantizationRepairPolicy

    torch.manual_seed(12)
    teacher = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).eval()
    student = mtq.quantize(copy.deepcopy(teacher), mtq.INT8_WEIGHT_ONLY_CFG)
    prepared = []
    original = QuantizationRepairPolicy._prepare

    def prepare(policy, *args, **kwargs):
        groups = original(policy, *args, **kwargs)
        fields = [parameter for group in groups for parameter in group.params]
        prepared.append([(parameter, parameter.detach().clone()) for parameter in fields])
        if len(prepared) == 2:
            fields[0].register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        return groups

    monkeypatch.setattr(QuantizationRepairPolicy, "_prepare", prepare)
    with pytest.raises((RuntimeError, FloatingPointError), match=r"[Nn]onfinite|non-finite"):
        mtd.local_distill(
            student,
            teacher,
            [torch.randn(8, 3, 8)],
            {"units": ["0", "1"], "updates": 1, "warmup_fraction": 0, "scheduler": "constant"},
        )
    assert len(prepared) == 2
    for unit in prepared:
        for parameter, initial in unit:
            torch.testing.assert_close(parameter, initial, rtol=0, atol=0)


@pytest.mark.parametrize("batch_kind", ["dict", "tensor"])
@pytest.mark.parametrize("optimizer_name", ["adagrad", "adamw"])
def test_local_streaming_update_matches_independent_reference(batch_kind, optimizer_name):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.repair import QuantizationRepairPolicy

    class TinyHeadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
                    for _ in range(2)
                ]
            )
            self.lm_head = torch.nn.Linear(8, 32)
            self.head_lengths = []

        def forward(self, input, logits_to_keep=0):
            for layer in self.layers:
                input = layer(input)
            hidden = input[:, -logits_to_keep:]
            self.head_lengths.append(hidden.shape[1])
            return self.lm_head(hidden)

    torch.manual_seed(8)
    teacher = TinyHeadModel().eval()
    student = copy.deepcopy(teacher)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.mul_(0.9)
    student = mtq.quantize(student, mtq.INT8_WEIGHT_ONLY_CFG)
    reference = copy.deepcopy(student)
    inputs = torch.randn(8, 3, 8)
    valid = torch.tensor([[True, True, index % 2 == 0] for index in range(8)])
    if batch_kind == "tensor":
        valid = torch.ones_like(valid)
        batch = inputs
    else:
        batch = {"input": inputs, "valid_mask": valid, "labels": torch.full((8, 3), -100)}
    config = mtd.LocalDistillationConfig(
        units=["layers.0", "layers.1"],
        optimizer=optimizer_name,
        updates=1,
        scheduler="constant",
        warmup_fraction=0,
        micro_batch_size=1,
    )
    current = inputs
    for module, target_module in zip(reference.layers, teacher.layers):
        with torch.no_grad():
            target = target_module(current)
        policy = QuantizationRepairPolicy()
        groups = policy.prepare(module, config)
        optimizer_type = torch.optim.Adagrad if optimizer_name == "adagrad" else torch.optim.AdamW
        optimizer = optimizer_type(
            [{"params": group.params, "lr": group.lr} for group in groups], weight_decay=0
        )
        prediction = module(current.detach())
        (prediction.float() - target.float()).square()[valid].mean().backward()
        torch.nn.utils.clip_grad_norm_([p for group in groups for p in group.params], 1.0)
        optimizer.step()
        policy.project_()
        policy.finalize_()
        current = target
    result = mtd.local_distill(student, teacher, [batch], config, validation_dataloader=[batch])
    assert teacher.head_lengths and set(teacher.head_lengths) == {1}
    assert result.student is student
    torch.testing.assert_close(student(inputs), reference(inputs), rtol=1e-6, atol=1e-7)
    assert result.metrics["updates"] == 1
    row = result.metrics["history"][0]["units"]["layers.0"]
    assert row["learning_rates"] == [config.scale_lr]
    assert row["valid_tokens"] == int(valid.sum())
    assert result.metrics["phase_timings"]["teacher_forward_host_seconds"] >= 0
    assert result.metrics["phase_timings"]["student_replay_backward_host_seconds"] >= 0
    assert result.metrics["units"]["layers.0"]["validation_mse"] >= 0
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert not any(
        module._forward_hooks or module._forward_pre_hooks for module in teacher.modules()
    )
    assert not any("_local_repair" in name for name, _ in student.named_modules())


@pytest.mark.parametrize(
    ("mode", "hf_model"), [("scale_only", False), ("joint", False), ("scale_only", True)]
)
def test_local_repair_transactional_resume_matches_uninterrupted(tmp_path, mode, hf_model):
    import json

    import modelopt.torch.quantization as mtq

    torch.manual_seed(19)
    teacher = torch.nn.Sequential(
        torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    ).eval()
    if hf_model:
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

        try:
            from transformers.distributed import configuration_utils as distributed_configuration
        except ImportError:
            distributed_configuration = None
        distributed_config_type = getattr(distributed_configuration, "DistributedConfig", None)

        teacher = GPTNeoXForCausalLM(
            GPTNeoXConfig(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                max_position_embeddings=16,
            )
        ).eval()
        if distributed_config_type is not None:
            teacher.config.distributed_config = distributed_config_type()
    unit = "gpt_neox.layers.0.mlp" if hf_model else "0"
    unquantized = copy.deepcopy(teacher)
    student = mtq.quantize(copy.deepcopy(teacher), mtq.INT8_WEIGHT_ONLY_CFG)
    reference = copy.deepcopy(student)
    batches = [
        {"input_ids": torch.randint(16, (1, 4))} if hf_model else torch.randn(8, 8)
        for _ in range(2)
    ]
    config = {
        "units": [unit],
        "mode": mode,
        "updates": 2,
        "global_batch_size": 1 if hf_model else 8,
        "scheduler": "constant",
        "warmup_fraction": 0,
    }
    expected = mtd.local_distill(reference, teacher, batches, config)
    directory = tmp_path / "resume"
    checkpoint_config = {**config, "checkpoint_dir": str(directory), "checkpoint_interval": 1}
    with pytest.raises(ValueError, match="Repair data exhausted"):
        mtd.local_distill(student, teacher, batches[:1], checkpoint_config)
    first = (directory / "latest").read_text().strip()
    manifest = json.loads((directory / first / "COMMITTED.json").read_text())
    assert manifest["step"] == 1
    saved = torch.load(directory / first / "rank-0.pt", weights_only=False)
    groups = saved["optimizers"][unit]["param_groups"]
    assert len(groups) == (4 if mode == "joint" else 2)
    assert all(len(group["params"]) == 1 for group in groups)
    if hf_model:
        # Same tensors and shapes, different teacher math: this is not a valid resume.
        changes = [("layer_norm_eps", teacher.config.layer_norm_eps * 2)]
        for attribute in ("_attn_implementation", "_experts_implementation"):
            if hasattr(teacher.config, attribute):
                changed = "eager" if getattr(teacher.config, attribute) != "eager" else "sdpa"
                changes.append((attribute, changed))
        if distributed_config_type is not None and "fsdp_mixed_precision" in (
            distributed_config_type.__dataclass_fields__
        ):
            changes.append(
                ("distributed_config", distributed_config_type(fsdp_mixed_precision=True))
            )
        for attribute, changed in changes:
            previous = getattr(teacher.config, attribute)
            setattr(teacher.config, attribute, changed)
            with pytest.raises(ValueError, match="identity differs"):
                mtd.local_distill(
                    student, teacher, batches, {**checkpoint_config, "resume_from": str(directory)}
                )
            setattr(teacher.config, attribute, previous)
        for model in (student, teacher):
            model.config._name_or_path = "/relocated/input"
            model.config._commit_hash = "different-checkout"
            model.config.transformers_version = "different-version"
    changed_batch = (
        {"input_ids": (batches[0]["input_ids"] + 1) % 16} if hf_model else batches[0] + 1
    )
    with pytest.raises(ValueError, match="sample order"):
        mtd.local_distill(
            student,
            teacher,
            [changed_batch, *batches[1:]],
            {
                **checkpoint_config,
                "resume_from": str(directory),
            },
        )
    result = mtd.local_distill(
        student,
        teacher,
        batches,
        {
            **checkpoint_config,
            "resume_from": str(directory),
            "output_dir": str(tmp_path / "final"),
        },
    )
    example = torch.randint(16, (1, 4)) if hf_model else torch.randn(2, 8)
    actual_output, expected_output = result.student(example), expected.student(example)
    if hf_model:
        actual_output, expected_output = actual_output.logits, expected_output.logits
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    assert result.metrics["updates"] == 2
    assert len(result.metrics["history"][-1]["units"][unit]["learning_rates"]) == (
        2 if mode == "joint" else 1
    )
    assert not (directory / first).exists()
    assert len(list(directory.glob("*/COMMITTED.json"))) == 1
    restored = mto.restore(
        unquantized,
        result.report_paths["checkpoint"],
    )
    restored_output = restored(example)
    if hf_model:
        restored_output = restored_output.logits
    torch.testing.assert_close(restored_output, actual_output, rtol=0, atol=0)


def get_input_tensor():
    """Dummy input tensor."""
    return torch.rand(2, 3, 112, 112)


def tiny_mobilenet():
    return get_tiny_mobilenet_and_input()[0]


@pytest.fixture
def layerwise_distillation_model():
    student = tiny_mobilenet().train()
    config = {
        "teacher_model": tiny_mobilenet(),
        "criterion": {
            ("features.2", "features.2"): torch.nn.MSELoss(),
        },
        "loss_balancer": mtd.StaticLossBalancer(),
    }
    layerwise_model = mtd.convert(student, mode=[("layerwise_kd", config)])

    return layerwise_model


def test_layerwise_hooks_registration(layerwise_distillation_model):
    """Test that layerwise-specific hooks are registered correctly."""
    # Check that student layers have _teacher_layer attribute
    for student_layer, teacher_layer in layerwise_distillation_model._layers_to_loss:
        assert hasattr(student_layer, "_teacher_layer")
        assert student_layer._teacher_layer[0] is teacher_layer
        assert hasattr(student_layer, "_intermediate_output")

        # Check that teacher layers have both input and output capture attributes
        assert hasattr(teacher_layer, "_intermediate_input")
        assert hasattr(teacher_layer, "_intermediate_output")


def test_layerwise_forward_pass(layerwise_distillation_model):
    """Test that forward pass works and captures both teacher inputs and outputs."""
    layerwise_distillation_model.train()
    input_tensor = get_input_tensor()

    layerwise_distillation_model(input_tensor)

    # Check that teacher intermediate inputs and outputs are captured
    for student_layer, teacher_layer in layerwise_distillation_model._layers_to_loss:
        assert teacher_layer._intermediate_input is None
        assert teacher_layer._intermediate_output is not None
        assert student_layer._intermediate_output is not None


def test_layerwise_input_injection(layerwise_distillation_model):
    """Test that teacher inputs are injected into student layers during layerwise distillation."""
    layerwise_distillation_model.train()
    input_tensor = get_input_tensor()

    # Perform forward pass
    layerwise_distillation_model(input_tensor)

    # Verify that teacher inputs were captured (they should be reset after injection)
    # After forward, teacher inputs should have been consumed by student layers
    for student_layer, teacher_layer in layerwise_distillation_model._layers_to_loss:
        # After full forward pass, teacher_layer._intermediate_input should be None
        # because it gets consumed by the student layerwise hook
        assert teacher_layer._intermediate_input is None


def test_layerwise_loss_computation(layerwise_distillation_model):
    """Test that loss computation works with layerwise distillation."""
    layerwise_distillation_model.train()
    input_tensor = get_input_tensor()

    output = layerwise_distillation_model(input_tensor)
    loss = layerwise_distillation_model.compute_kd_loss(student_loss=output.mean())

    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1
    assert loss.requires_grad


def test_layerwise_only_student_forward(layerwise_distillation_model):
    """Test that only_student_forward context manager works with layerwise distillation."""
    layerwise_distillation_model.train()
    input_tensor = get_input_tensor()

    # When using only_student_forward, teacher inputs should not be captured
    with warnings.catch_warnings(record=True) as w:
        with layerwise_distillation_model.only_student_forward():
            layerwise_distillation_model(input_tensor)

        # Should get warning about missing teacher input
        warning_messages = [str(warning.message) for warning in w]
        assert any("has no intermediate input stored" in msg for msg in warning_messages)

    # Verify teacher didn't run
    for student_layer, teacher_layer in layerwise_distillation_model._layers_to_loss:
        assert teacher_layer._intermediate_input is None
        assert teacher_layer._intermediate_output is None
        assert student_layer._intermediate_output is not None


def test_layerwise_only_teacher_forward(layerwise_distillation_model):
    """Test that only_teacher_forward context manager works with layerwise distillation."""
    layerwise_distillation_model.train()
    input_tensor = get_input_tensor()

    with layerwise_distillation_model.only_teacher_forward():
        layerwise_distillation_model(input_tensor)

    # Verify teacher ran and student didn't
    for student_layer, teacher_layer in layerwise_distillation_model._layers_to_loss:
        assert teacher_layer._intermediate_input is not None
        assert teacher_layer._intermediate_output is not None
        assert student_layer._intermediate_output is None


def test_layerwise_export(layerwise_distillation_model):
    """Test that export correctly cleans up layerwise-specific attributes."""
    # Check that _teacher_layer exists before export
    for student_layer, _ in layerwise_distillation_model._layers_to_loss:
        assert hasattr(student_layer, "_teacher_layer")

    # Export the model
    exported_model = mtd.export(layerwise_distillation_model)

    # Check that _teacher_layer is removed after export
    for student_layer in exported_model.modules():
        assert not hasattr(student_layer, "_teacher_layer")

    assert not hasattr(exported_model, "_teacher_model")
    assert not isinstance(exported_model, mtd.LayerwiseDistillationModel)


def test_layerwise_save_restore(layerwise_distillation_model, tmp_path):
    """Test that save/restore works correctly with layerwise distillation."""
    mto.save(layerwise_distillation_model, tmp_path / "ckpt.pt")

    new_student = tiny_mobilenet()
    restored_model = mto.restore(new_student, tmp_path / "ckpt.pt")

    # Ensure state is not actually restored (expected behavior from test_distill.py)
    manager = mto.ModeloptStateManager(restored_model)
    assert not manager.has_state
    assert isinstance(restored_model, type(new_student))


def test_layerwise_multiloss():
    """Test layerwise distillation with multiple loss functions."""
    student = tiny_mobilenet().train()
    config = {
        "teacher_model": tiny_mobilenet(),
        "criterion": {
            ("features.1", "features.1"): torch.nn.MSELoss(),
            ("features.3", "features.3"): torch.nn.MSELoss(),
        },
        "loss_balancer": mtd.StaticLossBalancer([0.5, 0.5]),
    }
    layerwise_model = mtd.convert(student, mode=[("layerwise_kd", config)])

    # Verify hooks are registered for all layers
    assert len(layerwise_model._layers_to_loss) == 2

    # Test forward pass
    output = layerwise_model(get_input_tensor())
    loss = layerwise_model.compute_kd_loss(student_loss=output.mean())

    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1


def test_layerwise_gradient_flow():
    """Test that gradients flow correctly through layerwise distillation."""
    student = tiny_mobilenet().train()
    config = {
        "teacher_model": tiny_mobilenet(),
        "criterion": {
            ("features.2", "features.2"): torch.nn.MSELoss(),
        },
        "loss_balancer": None,
    }
    layerwise_model = mtd.convert(student, mode=[("layerwise_kd", config)])

    # Save param snapshots by module
    param_snapshots = {
        name: p.clone() for name, p in layerwise_model.named_parameters() if p.requires_grad
    }

    # Forward and backward
    optimizer = torch.optim.SGD(layerwise_model.parameters(), lr=0.5)
    optimizer.zero_grad()
    layerwise_model(get_input_tensor())
    loss = layerwise_model.compute_kd_loss()
    loss.backward()
    optimizer.step()

    # Check: parameters in only the target layer(s) are changed
    updated_any = False
    for name, param in layerwise_model.named_parameters():
        if not param.requires_grad:
            continue
        changed = not torch.allclose(param, param_snapshots[name])
        if "features.2" in name:
            assert changed, f"'{name}' parameters did not change!"
            updated_any = True
        else:
            assert not changed, f"Parameters in unrelated layer '{name}' changed!"
    assert updated_any, (
        "No parameters were updated in 'features.2' or related layers during training"
    )


def test_layerwise_stale_output_warning_is_bounded(layerwise_distillation_model):
    """Both layerwise hooks must share the parent's latch instead of failing or repeating."""
    layerwise_distillation_model.train()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in range(3):
            layerwise_distillation_model(get_input_tensor())
        stale = [x for x in w if "already has an intermediate output stored" in str(x.message)]
        assert len(stale) == 2  # teacher's input/output hook and the student's output hook
