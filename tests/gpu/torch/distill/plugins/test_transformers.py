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

"""Real Transformers MoE repair, packing and parallel ownership checks."""

import copy
import os
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.transformers import TransformersLocalDistillationAdapter
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml.common import pinned_packed_weight
from modelopt.torch.quantization.ggml.registry import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.qtensor import FP8QTensor, QTensorWrapper
from modelopt.torch.quantization.repair import QuantizationRepairPolicy
from modelopt.torch.quantization.repair_partition import prepare_packed_weights_for_tp


def _model(device, implementation="eager", moe_intermediate_size=256):
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=moe_intermediate_size,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        dtype="bfloat16",
        attn_implementation="sdpa",
        use_cache=False,
        tie_word_embeddings=False,
    )
    config._experts_implementation = implementation
    return Qwen3MoeForCausalLM(config).to(device=device, dtype=torch.bfloat16).eval()


def _batch(device, packed=True):
    ids = torch.randint(3, 120, (1, 32), device=device)
    docs = torch.ones_like(ids)
    if packed:
        docs[:, 16:] = 2
    return {
        "input_ids": ids,
        "attention_mask": docs,
        "_packed_seq_ids": docs,
        "position_ids": torch.arange(16 if packed else 32, device=device).repeat(
            1, 2 if packed else 1
        ),
        "valid_mask": docs.bool(),
    }


def _ptq(model, batch, adapter, bits):
    spec = (
        {"num_bits": "iq2_s", "backend": "ggml"}
        if bits == "iq2_s"
        else {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
    )
    cfg = [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*mlp.experts.*weight_quantizer*", "cfg": spec},
    ]
    if bits == "fp8":
        spec.update(num_bits=(4, 3), block_sizes=None, axis=None)
    if bits == "nvfp4":
        cfg.append({"quantizer_name": "*mlp.experts.*input_quantizer", "cfg": spec})
    mtq.quantize(
        model,
        {
            "quant_cfg": cfg,
            "algorithm": "max"
            if bits == "fp8"
            else {"method": "gptq", "block_size": 256, "fused": bits == "nvfp4"},
        },
        lambda m: adapter.teacher_microbatch(m, [], batch, lambda *_: None),
    )


def _assert_native_tp_codes(actual, expected, tp):
    for name, module in actual.named_modules():
        reference_module = expected.get_submodule(name)
        if hasattr(module, "_first_proj_weight_quantizers_attr"):
            bindings = [
                (module._first_proj_attr, module._first_proj_weight_quantizers_attr),
                ("down_proj", "down_proj_weight_quantizers"),
            ]
        else:
            bindings = [("weight", "weight_quantizer")]
        for weight_name, quantizers_name in bindings:
            weight = module._parameters.get(weight_name)
            if not isinstance(weight, DTensor) or "tp" not in weight.device_mesh.mesh_dim_names:
                continue
            placement = weight.placements[weight.device_mesh.mesh_dim_names.index("tp")]
            if not isinstance(placement, (Shard, _StridedShard)):
                continue
            quantizers = getattr(module, quantizers_name, [])
            reference_quantizers = getattr(reference_module, quantizers_name, [])
            if isinstance(quantizers, TensorQuantizer):
                quantizers, reference_quantizers = [quantizers], [reference_quantizers]
            for quantizer, reference_quantizer in zip(
                quantizers, reference_quantizers, strict=True
            ):
                if getattr(reference_quantizer, "_native_weight_meta", None) is None:
                    continue
                shape = reference_quantizer._native_weight_meta["shape"]
                codes = reference_quantizer._native_weight_codes.reshape(shape[0], -1)
                if weight.ndim == 3:
                    codes = codes.unsqueeze(0)
                wanted = placement._split_tensor(codes, tp.size(), with_padding=False)[0][
                    tp.get_local_rank()
                ]
                torch.testing.assert_close(
                    quantizer._native_weight_codes.flatten(),
                    wanted.flatten(),
                    rtol=0,
                    atol=0,
                )


@pytest.mark.parametrize("shared_quantizer", [True, False])
def test_native_tp_staging_preserves_tied_packed_parameters(shared_quantizer):
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    q = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3), fake_quant=False))
    native, scale = FP8QTensor.quantize(
        torch.arange(64, device="cuda", dtype=torch.float32).reshape(8, 8) + 1
    )
    q._set_buffer("_scale", scale)
    model = torch.nn.ModuleList(
        [torch.nn.Linear(8, 8, bias=False), torch.nn.Linear(8, 8, bias=False)]
    ).cuda()
    original = QTensorWrapper(native)
    model[0].weight = model[1].weight = original
    model[0].weight_quantizer = q
    model[1].weight_quantizer = q if shared_quantizer else copy.deepcopy(q)
    if not shared_quantizer:
        with pytest.raises(ValueError):
            prepare_packed_weights_for_tp(model)
        assert model[0].weight is original and model[1].weight is original
        assert not q.fake_quant
        return
    metadata = prepare_packed_weights_for_tp(model)
    assert model[0].weight is model[1].weight
    assert not isinstance(model[0].weight, QTensorWrapper)
    assert metadata["0"]["fake_quant"] is False and metadata["1"]["fake_quant"] is False
    torch.testing.assert_close(
        q._native_weight_codes, native._quantized_data.view(torch.uint8), rtol=0, atol=0
    )


@pytest.mark.parametrize("bits", ["iq2_s", "nvfp4", "fp8"])
@pytest.mark.parametrize("implementation", ["eager", "grouped_mm"])
def test_transformers_moe_ptq_repair_packing_and_reload(bits, implementation, tmp_path):
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        pytest.skip("Run the independent roundtrip separately")
    torch.manual_seed(23)
    device = torch.device("cuda")
    teacher = _model(device, implementation).requires_grad_(False)
    student = copy.deepcopy(teacher)
    adapter = TransformersLocalDistillationAdapter()
    batch = _batch(device)
    with torch.no_grad():
        packed = teacher(**adapter.prepare_batch(teacher, batch)).logits
        for start in (0, 16):
            separate = teacher(
                input_ids=batch["input_ids"][:, start : start + 16], use_cache=False
            ).logits
            torch.testing.assert_close(
                packed[:, start : start + 16], separate, atol=0.01, rtol=0.01
            )
    _ptq(student, batch, adapter, bits)
    result = mtd.local_distill(
        student,
        teacher,
        [batch, batch],
        {
            "updates": 2,
            "global_batch_size": 1,
            "micro_batch_size": 1,
            "warmup_fraction": 0,
            "scheduler": "constant",
        },
        adapter=adapter,
    )
    with torch.no_grad():
        for block in student.model.layers:
            experts = block.mlp.experts
            for weight_name, quantizers_name in (
                (experts._first_proj_attr, experts._first_proj_weight_quantizers_attr),
                ("down_proj", "down_proj_weight_quantizers"),
            ):
                weight = getattr(experts, weight_name)
                quantizers = getattr(experts, quantizers_name)
                actual = experts._quantize_grouped_weight(weight, quantizers)
                expected = torch.stack([q(w) for q, w in zip(quantizers, weight, strict=True)])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(result.metrics["units"]) == 2
    for unit in result.metrics["units"].values():
        assert sum(unit["expert_coverage"]["experts"]["routed_assignments"]) == 128
    assert all(
        u["gradient_norm"] > 0 and u["valid_tokens"] == 32
        for row in result.metrics["history"]
        for u in row["units"].values()
    )
    assert not any(name.startswith("nemo_automodel") for name in sys.modules)
    path = tmp_path / "model.pt"
    mto.save(student, path)
    restored = mto.restore(copy.deepcopy(teacher), path)
    with torch.no_grad():
        torch.testing.assert_close(
            restored(**adapter.prepare_batch(restored, batch)).logits,
            student(**adapter.prepare_batch(student, batch)).logits,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "mode",
    [
        "cp",
        "pp",
        pytest.param("pp_nonfinite", marks=pytest.mark.timeout(40)),
        "cp_pp",
        "tp",
        "tp_iq2_xxs",
        "tp_attention",
        "cp_attention",
        "tp_fp8",
        "tp_staticint",
        "tp_mxfp4",
        "tp_mxfp8",
        "tp_mxfp6",
        "tp_mxint8",
        "tp_packed_fp8_attention",
        "tp_packed_nvfp4_attention",
        "tp_staticint_activation",
        "tp_staticblock_attention",
        "tp_linear",
    ],
)
@pytest.mark.timeout(180)
def test_transformers_parallel_repair_matches_reference(mode, monkeypatch):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size != 4:
        pytest.skip("Launch on four ranks")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    pp, cp = (2 if "pp" in mode else 1), (2 if "cp" in mode else 1)
    tp = 2 if mode.startswith("tp") else 1
    dp = world_size // (pp * cp * tp)
    mesh = init_device_mesh("cuda", (pp, dp, cp, tp), mesh_dim_names=("pp", "dp", "cp", "tp"))
    torch.manual_seed(27)
    teacher = _model(
        torch.device("cuda", rank), moe_intermediate_size=512 if mode == "tp_iq2_xxs" else 256
    ).requires_grad_(False)
    if mode in (
        "tp_attention",
        "tp_fp8",
        "tp_packed_fp8_attention",
        "tp_packed_nvfp4_attention",
        "tp_staticblock_attention",
    ):
        # BF16 reduction rounding can flip nearly cancelled scalar-scale gradients.
        teacher = teacher.float()
    student = copy.deepcopy(teacher)
    batch = _batch(torch.device("cuda", rank), packed=False)
    spec = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*mlp.experts.*weight_quantizer*",
                "cfg": {"num_bits": 4, "axis": 0},
            },
        ],
        "algorithm": "max",
    }
    if tp > 1:
        spec["quant_cfg"][1]["cfg"] = {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        }
    if mode == "tp_iq2_xxs":
        spec["quant_cfg"][1]["cfg"] = {"num_bits": "iq2_xxs", "backend": "ggml"}
    if mode in ("tp_fp8", "tp_staticint", "tp_staticint_activation"):
        spec["quant_cfg"][1]["cfg"] = {
            "num_bits": (4, 3) if mode == "tp_fp8" else 4,
            "axis": None if mode == "tp_fp8" else 0,
        }
    if mode.startswith("tp_mx"):
        spec["quant_cfg"][1]["cfg"] = {
            "num_bits": {
                "tp_mxfp4": (2, 1),
                "tp_mxfp8": (4, 3),
                "tp_mxfp6": (3, 2),
                "tp_mxint8": 8,
            }[mode],
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
        }
    if mode == "tp_packed_fp8_attention":
        spec["quant_cfg"][1]["cfg"] = {
            "num_bits": (4, 3),
            "axis": None,
        }
    if mode == "tp_staticint_activation":
        spec["quant_cfg"].append(
            {"quantizer_name": "*mlp.experts.*input_quantizer", "cfg": {"num_bits": 8, "axis": -1}}
        )
    if mode == "tp_staticblock_attention":
        spec["quant_cfg"].append(
            {
                "quantizer_name": "*self_attn*input_quantizer",
                "cfg": {"num_bits": 8, "block_sizes": {-1: 16}},
            }
        )
    if "attention" in mode:
        spec["quant_cfg"][1]["quantizer_name"] = "*self_attn*weight_quantizer"
    if mode == "tp_linear":
        spec["quant_cfg"][1]["quantizer_name"] = "*self_attn.q_proj.weight_quantizer"
    if mode == "tp_iq2_xxs":
        spec["algorithm"] = {"method": "gptq", "block_size": 256}
    mtq.quantize(
        student,
        spec,
        (lambda model: model(**TransformersLocalDistillationAdapter().prepare_batch(model, batch)))
        if mode in ("tp_iq2_xxs", "tp_staticint_activation", "tp_staticblock_attention")
        else None,
    )
    reference, reference_teacher = copy.deepcopy(student), copy.deepcopy(teacher)
    if mode in ("tp_packed_fp8_attention", "tp_packed_nvfp4_attention"):
        mtq.compress(student)
        mtq.compress(reference)
    config = {
        "updates": 2,
        "global_batch_size": 1,
        "micro_batch_size": 1,
        "warmup_fraction": 0,
        "scheduler": "constant",
        "gradient_clip_norm": 1.0,
        "unit_types": ("attention",) if "attention" in mode else ("mlp", "moe"),
    }
    if mode == "tp_linear":
        config["units"] = ["model.layers.0.self_attn.q_proj"]
    reference_result = mtd.local_distill(
        reference,
        reference_teacher,
        [batch, batch],
        config,
        adapter=TransformersLocalDistillationAdapter(
            DeviceMesh("cuda", [rank], mesh_dim_names=("dp",), _init_backend=False)
        ),
    )
    adapter = TransformersLocalDistillationAdapter(mesh)
    adapter.parallelize(teacher)
    adapter.parallelize(student)
    if cp > 1 or (mode == "tp_iq2_xxs" and dp > 1):
        shard_mesh = mesh["cp"] if cp > 1 else mesh["dp"]
        for model in (teacher, student):
            for block in model.model.layers:
                if any(True for _ in block.parameters()):
                    fully_shard(
                        block,
                        mesh=shard_mesh,
                        mp_policy=MixedPrecisionPolicy(
                            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                        ),
                    )
            fully_shard(
                model,
                mesh=shard_mesh,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
            )
    if mode == "pp_nonfinite":
        fields = []
        prepare = QuantizationRepairPolicy._prepare

        def prepare_with_fault(policy, *args, **kwargs):
            groups = prepare(policy, *args, **kwargs)
            for group in groups:
                for parameter in group.params:
                    fields.append((parameter, parameter.detach().clone()))
            if mesh["pp"].get_local_rank() == 1:
                groups[0].params[0].register_hook(
                    lambda gradient: torch.full_like(gradient, float("nan"))
                )
            return groups

        monkeypatch.setattr(QuantizationRepairPolicy, "_prepare", prepare_with_fault)
        with pytest.raises((ValueError, RuntimeError), match=r"[Nn]onfinite|non-finite"):
            mtd.local_distill(
                student,
                teacher,
                [batch, batch],
                {**config, "global_batch_size": dp},
                adapter=adapter,
            )
        assert fields
        for parameter, initial in fields:
            torch.testing.assert_close(parameter, initial, rtol=0, atol=0)
        dist.barrier()
        return
    result = mtd.local_distill(
        student, teacher, [batch, batch], {**config, "global_batch_size": dp}, adapter=adapter
    )
    for path in result.metrics["units"]:
        for observed, reference_row in zip(
            result.metrics["history"], reference_result.metrics["history"], strict=True
        ):
            torch.testing.assert_close(
                observed["units"][path]["mse"],
                reference_row["units"][path]["mse"],
                rtol=0.03,
                atol=1e-7,
            )
        actual = student.get_submodule(path)
        expected = reference.get_submodule(path)
        if tp > 1:
            _assert_native_tp_codes(actual, expected, mesh["tp"])
        actual_values = dict(actual.named_parameters()) if tp > 1 else actual.state_dict()
        expected_values = dict(expected.named_parameters()) if tp > 1 else expected.state_dict()
        for name, value in actual_values.items():
            # Fingerprints track rounded source weights, whose values are compared below.
            if name.rsplit(".", 1)[-1].startswith("_native_weight_source_"):
                continue
            other = expected_values[name]
            if isinstance(value, torch.Tensor):
                actual_value = value.full_tensor() if isinstance(value, DTensor) else value
                if isinstance(other, QTensorWrapper):
                    quantizer = expected.get_submodule(name.rsplit(".", 1)[0]).weight_quantizer
                    other = other.get_qtensor().dequantize(
                        dtype=actual_value.dtype,
                        scale=quantizer._scale,
                        double_scale=getattr(quantizer, "_double_scale", None),
                        block_sizes=quantizer.block_sizes,
                    )
                if not actual_value.is_floating_point():
                    torch.testing.assert_close(actual_value, other, rtol=0, atol=0, msg=name)
                    continue
                if actual_value.element_size() == 1:
                    # Float8 assert_close only supports exact tolerances; compare scale values in FP32.
                    assert actual_value.dtype == other.dtype, name
                    actual_value, other = actual_value.float(), other.float()
                torch.testing.assert_close(
                    actual_value,
                    other,
                    rtol=0.025 if cp > 1 or tp > 1 else 0.01,
                    atol=0.0002,
                    msg=name,
                )
                # Ring SDPA changes BF16 rounding; bound aggregate error as well.
                relative_error = (
                    actual_value.float() - other.float()
                ).norm() / other.float().norm().clamp_min(1e-12)
                assert relative_error < 0.003, name
        if mode == "tp_iq2_xxs":
            fmt = GGML_FORMAT_REGISTRY["iq2_xxs"]
            for projection in ("gate_up_proj", "down_proj"):
                weight = actual.experts._parameters[projection]
                placement = weight.placements[weight.device_mesh.mesh_dim_names.index("tp")]
                actual_quantizers = getattr(actual.experts, projection + "_weight_quantizers")
                expected_quantizers = getattr(expected.experts, projection + "_weight_quantizers")
                for quantizer, reference_quantizer in zip(
                    actual_quantizers, expected_quantizers, strict=True
                ):
                    expected_levels, _ = fmt.decode_int8(
                        pinned_packed_weight(reference_quantizer, fmt.name),
                        torch.tensor(weight.shape[1:]),
                    )
                    expected_levels = placement._split_tensor(
                        expected_levels.unsqueeze(0), tp, with_padding=False
                    )[0][mesh["tp"].get_local_rank()].squeeze(0)
                    actual_levels, _ = fmt.decode_int8(
                        pinned_packed_weight(quantizer, fmt.name),
                        torch.tensor(expected_levels.shape),
                    )
                    torch.testing.assert_close(actual_levels, expected_levels, rtol=0, atol=0)
        if mode in ("tp", "tp_attention", "tp_packed_nvfp4_attention"):
            fields = torch.stack(
                [
                    q._double_scale.reshape(())
                    for q in actual.modules()
                    if isinstance(q, TensorQuantizer) and q.is_enabled
                ]
            )
            replicas = [torch.empty_like(fields) for _ in range(tp)]
            dist.all_gather(replicas, fields, group=mesh["tp"].get_group())
            for replica in replicas:
                torch.testing.assert_close(fields, replica, rtol=0, atol=0)
            actual_modules = dict(actual.named_modules())
            for name, quantizer in expected.named_modules():
                if isinstance(quantizer, TensorQuantizer) and quantizer.is_enabled:
                    torch.testing.assert_close(
                        actual_modules[name]._double_scale,
                        quantizer._double_scale,
                        rtol=0.01,
                        atol=0,
                    )
        if tp > 1:
            for observed, reference_row in zip(
                result.metrics["history"], reference_result.metrics["history"], strict=True
            ):
                torch.testing.assert_close(
                    observed["units"][path]["gradient_norm"],
                    reference_row["units"][path]["gradient_norm"],
                    rtol=0.03,
                    atol=1e-7,
                )
    if "cp" in mode:
        original_backend = teacher.config._attn_implementation
        try:
            if rank == 0:
                teacher.config._attn_implementation = "eager"
            with pytest.raises(ValueError):
                adapter.teacher_microbatch(teacher, [], batch, lambda *_: None)
        finally:
            teacher.config._attn_implementation = original_backend
        with pytest.raises(ValueError):
            adapter.teacher_microbatch(
                teacher, [], _batch(torch.device("cuda", rank), packed=rank == 0), lambda *_: None
            )
    dist.barrier()
