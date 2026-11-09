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

"""Semantic repair selection on real Transformers architectures without weights."""

import copy

import pytest
import torch
import transformers
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

NemotronHConfig = getattr(transformers, "NemotronHConfig", None)
NemotronHForCausalLM = getattr(transformers, "NemotronHForCausalLM", None)
Qwen3_5ForCausalLM = getattr(transformers, "Qwen3_5ForCausalLM", None)
Qwen3_5TextConfig = getattr(transformers, "Qwen3_5TextConfig", None)

# Separate dense/expert mesh management is optional on older supported HF releases.
try:
    from transformers.distributed.utils import TransformersDeviceMesh
except ImportError:
    TransformersDeviceMesh = None

import modelopt.torch.distill as mtd
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.local_units import RepairUnit
from modelopt.torch.distill.plugins.transformers import (
    TransformersLocalDistillationAdapter,
    transformers_family,
)
from modelopt.torch.distill.plugins.transformers.adapter import _TransformersRuntime
from modelopt.torch.quantization import repair_partition
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.repair import QuantizationRepairPolicy
from modelopt.torch.quantization.repair_partition import (
    NativeWeightPartition,
    activation_partition,
    native_weight_partition,
    prepare_packed_weights_for_tp,
)


@pytest.mark.parametrize("family", ["gpt2", "gpt_neox", "nemotron_h"])
def test_transformers_family_semantic_repair_units(family):
    if family == "gpt2":
        model_class = GPT2LMHeadModel
        config = GPT2Config(vocab_size=32, n_embd=32, n_layer=2, n_head=4, n_positions=32)
        roles = {"mlp": ["transformer.h.0.mlp", "transformer.h.1.mlp"]}
    elif family == "gpt_neox":
        model_class = GPTNeoXForCausalLM
        config = GPTNeoXConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=32,
        )
        roles = {
            "mlp": ["gpt_neox.layers.0.mlp", "gpt_neox.layers.1.mlp"],
            "attention": ["gpt_neox.layers.0.attention", "gpt_neox.layers.1.attention"],
        }
    else:
        if NemotronHConfig is None or NemotronHForCausalLM is None:
            pytest.skip("Requires native Transformers Nemotron H modeling")
        model_class = NemotronHForCausalLM
        config = NemotronHConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            layers_block_type=["linear_attention", "full_attention", "mlp", "moe"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            mamba_num_heads=4,
            mamba_head_dim=16,
            n_groups=1,
            ssm_state_size=8,
            n_routed_experts=4,
            moe_intermediate_size=64,
            moe_shared_expert_intermediate_size=64,
        )
        roles = {
            "mlp": ["model.layers.2.mixer"],
            "moe": ["model.layers.3.mixer"],
            "attention": ["model.layers.1.mixer"],
            "mamba": ["model.layers.0.mixer"],
        }
    with torch.device("meta"):
        teacher = model_class(config).eval()
    student = mtq.quantize(copy.deepcopy(teacher), {**mtq.INT8_WEIGHT_ONLY_CFG, "algorithm": None})
    adapter = TransformersLocalDistillationAdapter()
    for role, expected in roles.items():
        result = mtd.local_distill(
            student, teacher, [], {"unit_types": (role,), "updates": 0}, adapter=adapter
        )
        assert result.metrics["units"] == expected


@pytest.mark.parametrize("mrope", [False, True])
def test_transformers_hybrid_packing_matches_independent_documents(mrope):
    if Qwen3_5TextConfig is None or Qwen3_5ForCausalLM is None:
        pytest.skip("Requires native Transformers Qwen3.5 modeling")
    torch.manual_seed(31)
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        attn_implementation="eager",
        use_cache=False,
    )
    teacher = Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
    adapter = TransformersLocalDistillationAdapter()
    unit = RepairUnit("mlp", "model.layers.1.mlp", "model.layers.1.mlp")
    ids = torch.randint(1, 32, (2, 16))
    documents = torch.tensor([[1] * 5 + [2] * 7 + [0] * 4]).expand(2, -1)
    batch = {
        "input_ids": ids,
        "_packed_seq_ids": documents,
        "valid_mask": documents > 0,
        "position_ids": torch.tensor([list(range(5)) + list(range(7)) + [0] * 4]).expand(2, -1),
    }
    if mrope:
        batch["position_ids"] = batch["position_ids"][None].expand(4, -1, -1)
    if not mrope:
        batch["attention_mask"] = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    expected = []
    for row, start, end in ((0, 0, 5), (0, 5, 12), (1, 0, 5), (1, 5, 12)):
        document = {
            "input_ids": ids[row : row + 1, start:end],
            "valid_mask": torch.ones(1, end - start, dtype=torch.bool),
            "position_ids": torch.arange(end - start)[None],
        }
        adapter.teacher_microbatch(
            teacher, [unit], document, lambda _, record: expected.append(record.target)
        )

    def capture_batches():
        records, forwards = [], []
        handle = teacher.register_forward_pre_hook(lambda *args: forwards.append(None))
        try:
            for microbatch in adapter.microbatches(batch, 1) if mrope else [batch]:
                adapter.teacher_microbatch(
                    teacher, [unit], microbatch, lambda _, record: records.append(record)
                )
        finally:
            handle.remove()
        return records, len(forwards)

    actual, forwards = capture_batches()
    assert forwards < 4 if mrope else forwards == 4
    torch.testing.assert_close(
        torch.cat([record.target[record.valid_mask][None] for record in actual], dim=1),
        torch.cat(expected, dim=1),
        rtol=1e-5,
        atol=1e-6,
    )
    assert sum(int(record.valid_mask.sum()) for record in actual) == 24
    if mrope:
        adapter.student = mtq.quantize(
            copy.deepcopy(teacher),
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    {
                        "quantizer_name": "*input_quantizer",
                        "cfg": {"num_bits": 8, "axis": None, "type": "dynamic"},
                    },
                ],
                "algorithm": None,
            },
        )
        guarded, forwards = capture_batches()
        assert forwards == 4
        torch.testing.assert_close(
            torch.cat([record.target[record.valid_mask][None] for record in guarded], dim=1),
            torch.cat(expected, dim=1),
            rtol=1e-5,
            atol=1e-6,
        )
        assert sum(int(record.valid_mask.sum()) for record in guarded) == 24


def test_transformers_standalone_moe_repair():
    torch.manual_seed(37)
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
    )
    config._experts_implementation = "eager"
    teacher = Qwen3MoeForCausalLM(config).model.layers[0].mlp.eval().requires_grad_(False)
    selection = mtd.LocalDistillationConfig(units=[""], updates=1, global_batch_size=1)
    patterns = transformers_family(teacher).weight_patterns(teacher, selection)
    student = mtq.quantize(
        copy.deepcopy(teacher),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                *[
                    {"quantizer_name": pattern, "cfg": {"num_bits": 4, "axis": 0}}
                    for pattern in patterns
                ],
            ],
            "algorithm": "max",
        },
    )
    inputs = torch.randn(1, 8, 32)
    expected_teacher = teacher(inputs).detach().clone()
    starting_point = student(inputs).detach().clone()
    result = mtd.local_distill(
        student,
        teacher,
        [inputs],
        selection,
        adapter=TransformersLocalDistillationAdapter(),
    )
    assert result.metrics["history"][0]["units"][""]["valid_tokens"] == 8
    assert result.metrics["history"][0]["units"][""]["gradient_norm"] > 0
    assert not torch.equal(student(inputs), starting_point)
    torch.testing.assert_close(teacher(inputs), expected_teacher, rtol=0, atol=0)


def _realized_contexts_preserve_unique_data_repair_job(rank, size):
    torch.manual_seed(53)
    teacher = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False)).eval().requires_grad_(False)
    student = mtq.quantize(copy.deepcopy(teacher), mtq.INT8_WEIGHT_ONLY_CFG)
    inputs = torch.randn(size, 3, 8)
    config = {
        "units": ["0"],
        "updates": 1,
        "global_batch_size": size,
        "micro_batch_size": 1,
        "warmup_fraction": 0,
        "scheduler": "constant",
    }
    reference = copy.deepcopy(student)
    singleton = DeviceMesh("cpu", [rank], mesh_dim_names=("dp",), _init_backend=False)
    expected = mtd.local_distill(
        reference,
        teacher,
        [inputs],
        config,
        adapter=TransformersLocalDistillationAdapter(singleton),
    )
    raw = init_device_mesh("cpu", (1, size), mesh_dim_names=("dp_replicate", "dp_shard"))
    dense = init_device_mesh("cpu", (1, size, 1), mesh_dim_names=("pp", "fsdp", "tp"))
    expert = init_device_mesh("cpu", (1, 1, size), mesh_dim_names=("pp", "efsdp", "ep"))
    topologies = [raw]
    if TransformersDeviceMesh is not None:
        topologies.append(TransformersDeviceMesh(dense, expert))
    for topology in topologies:
        actual = copy.deepcopy(student)
        result = mtd.local_distill(
            actual,
            teacher,
            [inputs[rank : rank + 1]],
            config,
            adapter=TransformersLocalDistillationAdapter(topology),
        )
        torch.testing.assert_close(actual[0].weight, reference[0].weight, rtol=1e-6, atol=1e-7)
        row = result.metrics["history"][0]["units"]["0"]
        expected_row = expected.metrics["history"][0]["units"]["0"]
        assert row["valid_tokens"] == expected_row["valid_tokens"] == size * 3
        torch.testing.assert_close(row["mse"], expected_row["mse"], rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(
            row["gradient_norm"], expected_row["gradient_norm"], rtol=1e-6, atol=1e-8
        )

    # Conditional experts must match serial optimization even when previously
    # active fields become absent or participate only on another replica.
    runtime = _TransformersRuntime(
        TransformersLocalDistillationAdapter(raw).context, mtd.LocalDistillationConfig()
    )
    for optimizer_type in (torch.optim.Adagrad, torch.optim.AdamW):
        fields = [torch.nn.Parameter(torch.tensor([1.0])) for _ in range(3)]
        reference_fields = [torch.nn.Parameter(field.detach().clone()) for field in fields]
        runtime.parameter_meshes = {id(field): [runtime.dp_cp] for field in fields}
        actual_optimizer = optimizer_type(fields, lr=0.01, weight_decay=0)
        reference_optimizer = optimizer_type(reference_fields, lr=0.01, weight_decay=0)
        for step in range(2):
            actual_optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            local = fields[0] if rank == 0 else fields[2] if step == 0 else None
            if local is not None:
                local.square().sum().backward()
            objective = reference_fields[0].square().sum()
            if step == 0:
                objective = objective + reference_fields[2].square().sum()
            objective.backward()
            runtime.reduce_gradients(fields)
            for field, expected_field in zip(fields, reference_fields):
                assert (field.grad is None) == (expected_field.grad is None)
                if field.grad is not None:
                    torch.testing.assert_close(field.grad, expected_field.grad)
            runtime.clip_gradients(fields, 1.0)
            torch.nn.utils.clip_grad_norm_(reference_fields, 1.0)
            actual_optimizer.step()
            reference_optimizer.step()
            for field, expected_field in zip(fields, reference_fields):
                torch.testing.assert_close(field, expected_field, rtol=0, atol=0)
                actual_state = actual_optimizer.state[field]
                expected_state = reference_optimizer.state[expected_field]
                assert actual_state.keys() == expected_state.keys()
                for key in actual_state:
                    torch.testing.assert_close(
                        actual_state[key], expected_state[key], rtol=0, atol=0
                    )


def test_realized_contexts_preserve_unique_data_repair(cpu_dist_workers):
    cpu_dist_workers.run(_realized_contexts_preserve_unique_data_repair_job)


@pytest.mark.parametrize("native_format", ["int4", "nf4"])
def test_packed_weight_staging_preserves_native_repair(native_format):
    torch.manual_seed(43)
    model = torch.nn.Sequential(torch.nn.Linear(128, 8, bias=False)).eval()
    blocks = {-1: 64}
    if native_format == "nf4":
        blocks.update(scale_bits=8, scale_block_sizes={-1: 2})
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": 4, "block_sizes": blocks},
                },
            ],
            "algorithm": "max",
        },
    )
    mtq.compress(model)
    inputs = torch.randn(2, 128)
    expected = model(inputs).detach().clone()
    prepare_packed_weights_for_tp(model)
    codes = model[0].weight_quantizer._native_weight_codes.clone()
    for _ in range(2):
        policy = QuantizationRepairPolicy()
        groups = policy.prepare(model, mtd.LocalDistillationConfig())
        torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
        optimizer = torch.optim.Adagrad(
            [{"params": group.params, "lr": group.lr} for group in groups]
        )
        model(inputs).square().mean().backward()
        optimizer.step()
        policy.project_()
        expected = model(inputs).detach().clone()
        policy.finalize_()
        torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            model[0].weight_quantizer._native_weight_codes, codes, rtol=0, atol=0
        )


def test_rowwise_tp_rejects_cut_dynamic_activation_blocks_without_calibration():
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=8, block_sizes={-1: 32, "type": "dynamic", "scale_bits": (8, 0)}
        )
    )
    with pytest.raises(ValueError, match="complete native quantization blocks"):
        activation_partition(quantizer, NativeWeightPartition((32, 96), 1, 0, 2))
    assert activation_partition(quantizer, NativeWeightPartition((32, 128), 1, 0, 2)) == {}
    rotating = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, enable=False, rotate=True))
    with pytest.raises(ValueError, match="complete native quantization blocks"):
        activation_partition(rotating, NativeWeightPartition((32, 128), 1, 0, 2))
    rotating.set_from_attribute_config({"rotate": {"enable": True, "block_size": 32}})
    assert activation_partition(rotating, NativeWeightPartition((32, 128), 1, 0, 2)) == {}


def test_rowwise_tp_rejects_local_dynamic_tensor_statistics():
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=None))
    values = torch.linspace(0.01, 1, 96).reshape(1, 96)
    values[:, 48:] *= 64
    full = quantizer(values)
    local = torch.cat((quantizer(values[:, :48]), quantizer(values[:, 48:])), dim=-1)
    assert not torch.equal(full, local)
    with pytest.raises(ValueError, match="activation statistics"):
        activation_partition(quantizer, NativeWeightPartition((32, 96), 1, 0, 2))
    quantizer._set_buffer("_amax", torch.tensor(64.0))
    before = quantizer._amax
    activation_partition(quantizer, NativeWeightPartition((32, 96), 1, 0, 2))
    assert quantizer._amax is before
    channel = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=-1))
    rows = torch.cat((values, values * 0.37), dim=0)
    torch.testing.assert_close(
        channel(rows), torch.cat((channel(rows[:, :48]), channel(rows[:, 48:])), -1), rtol=0, atol=0
    )
    assert activation_partition(channel, NativeWeightPartition((32, 96), 1, 0, 2)) == {}
    token = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, block_sizes={-1: None}))
    token(rows)
    with pytest.raises(ValueError, match="activation statistics"):
        activation_partition(token, NativeWeightPartition((32, 96), 1, 0, 2))


def test_rowwise_tp_retains_full_dynamic_block_calibration():
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(2, 1), block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
        )
    )
    calibrated = torch.linspace(1, 64, 96).reshape(1, 96)
    quantizer._set_buffer("_amax", calibrated)
    fields = activation_partition(quantizer, NativeWeightPartition((32, 96), 1, 0, 2))
    assert "_amax" not in fields
    assert quantizer._amax is calibrated
    actual = quantizer._get_amax(torch.zeros(2, 48))
    assert actual.shape == calibrated.shape
    torch.testing.assert_close(actual.amax(), calibrated.amax(), rtol=0, atol=0)


def test_rowwise_tp_rejects_quantization_of_partial_outputs():
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=4))
    quantizer._set_buffer("_amax", torch.ones(()))
    left, right = torch.tensor([0.07]), torch.tensor([0.07])
    assert not torch.equal(quantizer(left + right), quantizer(left) + quantizer(right))
    columns = NativeWeightPartition((128, 32), 0, 0, 2)
    repair_partition.validate_tp_output_quantizer(quantizer, columns)
    values = torch.linspace(0.01, 1, 128).reshape(1, 128)
    torch.testing.assert_close(
        quantizer(values),
        torch.cat((quantizer(values[:, :64]), quantizer(values[:, 64:])), -1),
        rtol=0,
        atol=0,
    )
    mx = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=8, block_sizes={-1: 32, "type": "dynamic", "scale_bits": (8, 0)}
        )
    )
    repair_partition.validate_tp_output_quantizer(mx, columns)
    with pytest.raises(ValueError, match="after the native reduction"):
        repair_partition.validate_tp_output_quantizer(
            quantizer, NativeWeightPartition((32, 96), 1, 0, 2)
        )


def test_rowwise_tp_rejects_dynamic_bias_with_fixed_amax():
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=8, bias={-1: None, "type": "dynamic", "method": "mean"})
    )
    quantizer._set_buffer("_amax", torch.tensor(64.0))
    values = torch.linspace(0.01, 1, 96).reshape(1, 96)
    values[:, 48:] *= 64
    assert not torch.equal(
        quantizer(values), torch.cat((quantizer(values[:, :48]), quantizer(values[:, 48:])), -1)
    )
    with pytest.raises(ValueError, match=r"dynamic.*bias"):
        activation_partition(quantizer, NativeWeightPartition((32, 96), 1, 0, 2))


@pytest.mark.parametrize(
    "case", ["dynamic", "channel_field", "mx_cut", "pre_scale", "bias_field", "rotation"]
)
def test_colwise_tp_output_quantization_requires_local_fields_and_statistics(case):
    cfg = {"num_bits": 8, "axis": None}
    if case == "mx_cut":
        cfg = {"num_bits": 8, "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)}}
    quantizer = TensorQuantizer(QuantizerAttributeConfig(**cfg))
    if case == "channel_field":
        quantizer._set_buffer("_amax", torch.ones(1, 96))
    if case in ("pre_scale", "bias_field", "rotation"):
        quantizer._set_buffer("_amax", torch.ones(()))
        if case == "pre_scale":
            quantizer.pre_quant_scale = torch.ones(96)
        elif case == "bias_field":
            quantizer._set_buffer("_bias_value", torch.ones(96))
        else:
            quantizer.set_from_attribute_config({"rotate": True})
    with pytest.raises(ValueError, match="Colwise TP"):
        repair_partition.validate_tp_output_quantizer(
            quantizer, NativeWeightPartition((96, 32), 0, 0, 2)
        )


def test_native_partition_distinguishes_fake_compression_recipe_from_native_scale_groups():
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=4, block_sizes={-1: 64, "scale_block_sizes": {-1: 256}})
    )
    quantizer._set_buffer("_amax", torch.ones(512, 1))
    rows = NativeWeightPartition((128, 256), 0, 0, 2)
    fields, _, _ = native_weight_partition(quantizer, rows)
    assert fields["_amax"].numel() == 256
    assert "_double_scale" not in fields
    quantizer._set_buffer("_scale", torch.ones(2, 256))
    quantizer._set_buffer("_double_scale", torch.ones(2))
    fields, _, _ = native_weight_partition(quantizer, rows)
    assert fields["_scale"].shape == (1, 256)
    assert fields["_double_scale"].numel() == 1
    with pytest.raises(ValueError, match="double-quantized"):
        native_weight_partition(quantizer, NativeWeightPartition((128, 256), 1, 0, 2))
    nvfp4 = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(2, 1), block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
        )
    )
    nvfp4._set_buffer("_scale", torch.ones(128, 8))
    nvfp4._set_buffer("_double_scale", torch.ones(()))
    _, layouts, _ = native_weight_partition(nvfp4, NativeWeightPartition((128, 128), 0, 0, 2))
    assert layouts["deltas.double_scale"] is False
