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

"""Short native-format and AutoModel expert-parallel repair integration checks."""

import copy
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.repair import QuantizationRepairPolicy


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_iq_scale_kernel_matches_native_torch_gradients(dtype, monkeypatch):
    """Fused scale application retains projected values and local scale gradients."""
    from modelopt.torch.kernels.quantization.repair import apply_block_scales, decode_iq_levels
    from modelopt.torch.quantization.ggml.registry import GGML_FORMAT_REGISTRY
    from modelopt.torch.quantization.repair_formats import native_scale_execution

    for name in ("iq1_s", "iq1_m", "iq2_s", "iq2_xs"):
        fmt = GGML_FORMAT_REGISTRY[name]
        packed = torch.randint(0, 256, (47, fmt.block_bytes), device="cuda", dtype=torch.uint8)
        expected_levels, _ = fmt.decode_int8(packed, torch.tensor([47, 256]))
        actual_levels = decode_iq_levels(packed, name, (47, 256))
        torch.testing.assert_close(actual_levels, expected_levels, rtol=0, atol=0)
        layer = nn.Linear(256, 2, bias=False).to(device="cuda", dtype=dtype)
        mtq.quantize(
            layer,
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    {
                        "quantizer_name": "*weight_quantizer",
                        "cfg": {"num_bits": name, "backend": "ggml"},
                    },
                ],
                "algorithm": "max",
            },
        )
        policy = QuantizationRepairPolicy()
        groups = policy.prepare(layer, mtd.LocalDistillationConfig())
        parameters = [p for group in groups for p in group.params]
        with torch.no_grad():
            for parameter in parameters:
                parameter.uniform_(-0.1, 0.1)
        inputs = torch.randn(3, 256, device="cuda", dtype=dtype)
        reference = layer(inputs)
        reference.float().square().sum().backward()
        gradients = [parameter.grad.clone() for parameter in parameters]
        for parameter in parameters:
            parameter.grad = None

        def reference_unpack_forbidden(*args, **kwargs):
            raise AssertionError("Native IQ replay must avoid the reference unpacker.")

        with monkeypatch.context() as patch, native_scale_execution():
            patch.setattr(type(fmt), "decode_int8", reference_unpack_forbidden)
            actual = layer(inputs)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            actual.float().square().sum().backward()
        for parameter, expected in zip(parameters, gradients, strict=True):
            torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=1e-6)
        policy.finalize_()
        torch.testing.assert_close(layer(inputs), reference, rtol=0, atol=0)
    torch.manual_seed(43)
    levels = torch.randint(-9, 10, (513, 16), device="cuda", dtype=torch.int8)
    scales = torch.randn(513, device="cuda", requires_grad=True)
    gradient = torch.randn_like(levels, dtype=dtype)
    expected = (levels.float() * scales[:, None]).to(dtype)
    expected.backward(gradient)
    expected_gradient = scales.grad.clone()
    scales.grad = None
    actual = apply_block_scales(levels, scales, dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward(gradient)
    torch.testing.assert_close(scales.grad, expected_gradient, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compressed_nf4_native_rounding(dtype, tmp_path):
    """Native CUDA BF16 decode rounding survives scale repair and reload."""
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(256, 16, bias=False)).to(device="cuda", dtype=dtype)
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {
                        "num_bits": 4,
                        "block_sizes": {-1: 64, "scale_bits": 8, "scale_block_sizes": {-1: 16}},
                    },
                },
            ],
            "algorithm": "max",
        },
    )
    mtq.compress(model)
    inputs = torch.randn(2, 256, device="cuda", dtype=dtype)
    expected = model(inputs).detach()
    codes = model[0].weight.detach().view(torch.uint8).clone()
    policy = QuantizationRepairPolicy()
    groups = policy.prepare(model, mtd.LocalDistillationConfig(updates=1))
    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
    optimizer = torch.optim.Adagrad([{"params": g.params, "lr": g.lr} for g in groups])
    model(inputs).float().square().mean().backward()
    optimizer.step()
    policy.project_()
    repaired = model(inputs).detach()
    policy.finalize_()
    torch.testing.assert_close(model(inputs), repaired, rtol=0, atol=0)
    assert torch.equal(model[0].weight.view(torch.uint8), codes)
    mto.save(model, tmp_path / "nf4.pt")
    restored = mto.restore(
        nn.Sequential(nn.Linear(256, 16, bias=False)).to("cuda", dtype),
        tmp_path / "nf4.pt",
    ).cuda()
    restored_quantizer = restored[0].weight_quantizer
    assert all(
        getattr(restored_quantizer, field).device == inputs.device
        for field in ("_scale", "_double_scale", "_scale_zeros")
    )
    torch.testing.assert_close(restored(inputs), repaired, rtol=0, atol=0)


@pytest.mark.timeout(600)
def test_native_repair_cuda(tmp_path):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    workers = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cases = [
        ("int8", {"num_bits": 8, "axis": 0}),
        (
            "nvfp4",
            {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}},
        ),
        (
            "mxfp4",
            {"num_bits": (2, 1), "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)}},
        ),
        ("iq2_s", {"num_bits": "iq2_s", "backend": "ggml"}),
        *[
            (
                name,
                {
                    "num_bits": bits,
                    "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (4, 3)},
                },
            )
            for name, bits in [("nvfp6_e3m2", (3, 2)), ("nvfp8_e5m2", (5, 2))]
        ],
        *[
            (
                name,
                {
                    "num_bits": bits,
                    "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
                },
            )
            for name, bits in [
                ("mxfp6_e3m2", (3, 2)),
                ("mxint8", 8),
                ("mxfp8_e5m2", (5, 2)),
                ("mxfp6_e2m3", (2, 3)),
                ("mx4_e1m2", (1, 2)),
            ]
        ],
    ]
    for name, weight_config in cases[rank::workers]:
        torch.manual_seed(13)
        teacher = (
            nn.Sequential(nn.Linear(256, 32, bias=False), nn.SiLU(), nn.Linear(32, 256, bias=False))
            .to(device, torch.bfloat16)
            .eval()
        )
        student = copy.deepcopy(teacher)
        batches = [torch.randn(8, 16, 256, device=device, dtype=torch.bfloat16) for _ in range(2)]
        quantizers = [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "0.weight_quantizer" if name == "iq2_s" else "*weight_quantizer",
                "cfg": weight_config,
            },
        ]
        if name == "nvfp4":
            quantizers.append({"quantizer_name": "*input_quantizer", "cfg": weight_config})
        mtq.quantize(
            student, {"quant_cfg": quantizers, "algorithm": "max"}, lambda model: model(batches[0])
        )
        initial_mse = (
            (student(batches[0]).float() - teacher(batches[0]).float()).square().mean().item()
        )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        result = mtd.local_distill(
            student,
            teacher,
            batches,
            {
                "units": [""],
                "updates": 2,
                "global_batch_size": 8,
                "micro_batch_size": 2,
                "scheduler": "constant",
                "warmup_fraction": 0,
            },
            validation_dataloader=batches[:1],
        )
        torch.cuda.synchronize(device)
        if name.startswith(("nvfp6", "nvfp8")):
            torch.testing.assert_close(
                result.metrics["units"][""]["baseline_validation_mse"],
                initial_mse,
                rtol=2.5e-2,
                atol=1e-7,
            )
        assert torch.isfinite(result.student(batches[0])).all()
        assert all(row["units"][""]["gradient_norm"] > 0 for row in result.metrics["history"])
        path = tmp_path / (name + ".pt")
        mto.save(result.student, path)
        restored = mto.restore(copy.deepcopy(teacher), path)
        torch.testing.assert_close(restored(batches[0]), result.student(batches[0]), rtol=0, atol=0)
        print(
            json.dumps(
                {
                    "format": name,
                    "seconds": time.perf_counter() - started,
                    "peak_bytes": torch.cuda.max_memory_allocated(device),
                    "mse": result.metrics["units"][""]["validation_mse"],
                }
            ),
            flush=True,
        )


@pytest.mark.timeout(240)
def test_native_mamba_repair_preserves_packed_resets(tmp_path):
    from nemo_automodel.components.distributed.mesh import MeshContext
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM
    from torch.distributed.device_mesh import DeviceMesh
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    from modelopt.torch.distill.plugins.automodel import (
        AutoModelLocalDistillationAdapter,
        prepare_packed_batch,
    )

    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    torch.manual_seed(28)
    config = NemotronHConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        hybrid_override_pattern="M-",
        num_attention_heads=4,
        num_key_value_heads=2,
        mamba_num_heads=4,
        mamba_head_dim=16,
        ssm_state_size=16,
        conv_kernel=4,
        n_groups=1,
        chunk_size=16,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )
    teacher = (
        NemotronHForCausalLM(
            config,
            backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch"),
        )
        .to(rank)
        .eval()
    )
    teacher.initialize_weights(torch.device("cuda", rank), dtype=torch.bfloat16)
    tokens = torch.randint(1, 63, (2, 32), device=rank)
    documents = torch.ones_like(tokens)
    documents[:, 16:] = 2
    batch = {
        "input_ids": tokens,
        "attention_mask": documents,
        "_packed_seq_ids": documents,
        "valid_mask": documents.ne(0),
        "position_ids": torch.arange(16, device=rank).repeat(2, 2),
    }
    batch = prepare_packed_batch(teacher, batch)
    kwargs = {key: value for key, value in batch.items() if key != "valid_mask"}
    kwargs["use_cache"] = False
    with torch.no_grad():
        original = teacher(**kwargs).logits
        separate = torch.cat(
            [
                teacher(input_ids=tokens[:, start : start + 16], use_cache=False).logits
                for start in (0, 16)
            ],
            dim=1,
        )
    torch.testing.assert_close(original, separate, rtol=2e-2, atol=2e-3)
    student = mtq.quantize(
        copy.deepcopy(teacher),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*layers.0.mixer*weight_quantizer",
                    "cfg": {"num_bits": 4, "axis": 0},
                },
                {
                    "quantizer_name": "*layers.0.mixer*input_quantizer",
                    "cfg": {"num_bits": 4, "axis": None},
                },
            ],
            "algorithm": "max",
        },
        lambda model: model(**kwargs),
    )
    activations = {
        name: module.amax
        for name, module in student.named_modules()
        if name.startswith("model.layers.0.mixer") and name.endswith("input_quantizer")
    }
    assert all(value is not None for value in activations.values()), (
        f"Missing calibration: {activations}"
    )
    activations = {name: value.clone() for name, value in activations.items()}
    used_fields = set()
    handles = []
    for name, module in student.named_modules():
        if name.startswith("model.layers.0.mixer") and name.endswith("weight_quantizer"):
            handles.append(
                module.register_forward_hook(
                    lambda module, args, output, name=name: (
                        used_fields.add(name) if output.requires_grad else None
                    )
                )
            )
    singleton = DeviceMesh("cuda", [rank], mesh_dim_names=("dp",), _init_backend=False)
    result = mtd.local_distill(
        student,
        teacher,
        [batch, {**batch, "input_ids": tokens.roll(1, dims=0)}],
        {
            "unit_types": ["mamba"],
            "updates": 2,
            "global_batch_size": 2,
            "micro_batch_size": 1,
            "scheduler": "constant",
            "warmup_fraction": 0,
        },
        validation_dataloader=[batch],
        adapter=AutoModelLocalDistillationAdapter(MeshContext.from_meshes(singleton)),
    )
    for handle in handles:
        handle.remove()
    assert used_fields == {
        "model.layers.0.mixer." + name + ".weight_quantizer"
        for name in ("in_proj", "conv1d", "out_proj")
    }
    for name, expected in activations.items():
        torch.testing.assert_close(student.get_submodule(name).amax, expected, rtol=0, atol=0)
    assert all(
        row["units"]["model.layers.0.mixer"]["gradient_norm"] > 0
        and row["units"]["model.layers.0.mixer"]["valid_tokens"] == tokens.numel()
        for row in result.metrics["history"]
    )
    with torch.no_grad():
        torch.testing.assert_close(teacher(**kwargs).logits, original, rtol=0, atol=0)
    mto.save(result.student, tmp_path / "mamba.pt")
    restored = mto.restore(copy.deepcopy(teacher), tmp_path / "mamba.pt")
    with torch.no_grad():
        torch.testing.assert_close(
            restored(**kwargs).logits, result.student(**kwargs).logits, rtol=0, atol=0
        )


@pytest.mark.parametrize("runtime", ["native", "runner"])
def test_native_automodel_ep_gradient_and_finalize(runtime):
    if int(os.environ.get("WORLD_SIZE", 1)) < 2:
        pytest.skip("Launch with torchrun using at least two CUDA ranks")
    experts_module = pytest.importorskip("nemo_automodel.components.moe.experts")
    from nemo_automodel.components.moe.config import MoEConfig

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    size = dist.get_world_size()
    mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("ep",))
    config = MoEConfig(
        n_routed_experts=size,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0,
        aux_loss_coeff=0,
        score_func="softmax",
        route_scale=1,
        dim=32,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=True,
        dtype=torch.float32,
    )
    torch.manual_seed(31)
    teacher = experts_module.GroupedExperts(config).cuda(rank)
    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.normal_(std=0.05)
    student = mtq.quantize(
        copy.deepcopy(teacher),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*gate_and_up_projs_weight_quantizers.*",
                    "cfg": {"num_bits": 4, "axis": 0},
                },
                {
                    "quantizer_name": "*down_projs_weight_quantizers.*",
                    "cfg": {
                        "num_bits": (2, 1),
                        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                    },
                },
            ],
            "algorithm": "max",
        },
    )
    reference = copy.deepcopy(student)
    generator = torch.Generator(device="cuda").manual_seed(7)
    full_inputs = torch.randn(size * 2, 32, device="cuda", generator=generator)
    all_indices = (
        ((torch.arange(size, device="cuda") + 1) % size).repeat_interleave(2).unsqueeze(-1)
    )
    if runtime == "runner":
        # One expert stays globally unobserved; others include remote-only inputs.
        all_indices.remainder_(size - 1)
    all_probs = torch.ones(size * 2, 1, device="cuda")
    all_mask = torch.ones(size * 2, device="cuda", dtype=torch.bool)
    if runtime == "runner":
        all_mask[-1] = False
        all_indices[-1] = -1
    with torch.no_grad():
        target = teacher(full_inputs, all_mask, all_probs, all_indices)
    reference_policy = QuantizationRepairPolicy()
    reference_groups = reference_policy.prepare(reference, mtd.LocalDistillationConfig())
    reference(full_inputs, all_mask, all_probs, all_indices).sub(target).square()[
        all_mask
    ].mean().backward()
    for name in ("gate_and_up_projs", "down_projs"):
        student._parameters[name] = nn.Parameter(
            distribute_tensor(student._parameters[name].detach(), mesh, [Shard(0)]),
            requires_grad=False,
        )
    selection = slice(rank * 2, rank * 2 + 2)
    if runtime == "native":
        from modelopt.torch.quantization.model_calib import _register_weight_input_hooks
        from modelopt.torch.quantization.utils.core_utils import module_name_maps

        captured = {}
        handles = _register_weight_input_hooks(
            student,
            module_name_maps(student),
            lambda quantizer, weight, inputs: captured.setdefault(
                id(quantizer), inputs.detach().clone()
            ),
        )
        try:
            with torch.no_grad():
                student(
                    full_inputs[selection],
                    all_mask[selection],
                    all_probs[selection],
                    all_indices[selection],
                )
        finally:
            for handle in handles:
                handle.remove()
        # Every token for this expert comes from a remote EP rank. Calibration
        # must observe the actual post-gather routed inputs, not only local data.
        key = id(student.gate_and_up_projs_weight_quantizers[rank])
        assert key in captured, "Remote-only expert input was omitted during calibration"
        torch.testing.assert_close(
            captured[key], full_inputs[all_indices[:, 0] == rank], rtol=0, atol=0
        )
    if runtime == "runner":
        from nemo_automodel.components.distributed.mesh import MeshContext

        from modelopt.torch.distill.plugins.automodel import AutoModelLocalDistillationAdapter

        device_mesh = init_device_mesh(
            "cuda",
            (1, 1, size, 1, 1),
            mesh_dim_names=("pp", "dp_replicate", "dp_shard", "cp", "tp"),
        )
        device_mesh["dp_replicate", "dp_shard"]._flatten("dp")
        device_mesh["dp_replicate", "dp_shard", "cp"]._flatten("dp_cp")
        context = MeshContext.from_meshes(device_mesh, moe_mesh=mesh)
        batch = {
            "x": full_inputs[selection],
            "token_mask": all_mask[selection],
            "weights": all_probs[selection],
            "indices": all_indices[selection],
            "valid_mask": all_mask[selection],
        }

        class CoverageAdapter(AutoModelLocalDistillationAdapter):
            reductions = 0

            def reduce_coverage(self, counts):
                self.reductions += 1
                return super().reduce_coverage(counts)

        adapter = CoverageAdapter(context)
        result = mtd.local_distill(
            student,
            teacher,
            [batch],
            {
                "units": [""],
                "updates": 1,
                "global_batch_size": size * 2,
                "micro_batch_size": 1,
                "scheduler": "constant",
                "warmup_fraction": 0,
            },
            adapter=adapter,
        )
        reference_optimizer = torch.optim.Adagrad(
            [{"params": group.params, "lr": group.lr} for group in reference_groups]
        )
        reference_optimizer.step()
        reference_policy.project_()
        reference_policy.finalize_()
        expected = reference(full_inputs, all_mask, all_probs, all_indices)[selection]
        actual = result.student(
            full_inputs[selection],
            all_mask[selection],
            all_probs[selection],
            all_indices[selection],
        )
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=1e-7)
        assert result.metrics["history"][0]["units"][""]["elements"] == target[all_mask].numel()
        coverage = result.metrics["units"][""]["expert_coverage"][""]
        expected_counts = torch.bincount(all_indices[all_mask].flatten(), minlength=size).tolist()
        assert coverage["routed_assignments"] == expected_counts
        assert coverage["unobserved_experts"] == [size - 1]
        assert adapter.reductions == 1
        dist.barrier()
        return
    policy = QuantizationRepairPolicy()
    groups = policy.prepare(student, mtd.LocalDistillationConfig())
    output = student(
        full_inputs[selection], all_mask[selection], all_probs[selection], all_indices[selection]
    )
    output.sub(target[selection]).square().sum().backward()
    for group in groups:
        for parameter in group.params:
            parameter.grad.div_(target.numel())
    actual = {
        name: parameter for name, parameter in student.named_parameters() if parameter.requires_grad
    }
    expected = dict(reference.named_parameters())
    for name, parameter in actual.items():
        torch.testing.assert_close(parameter.grad, expected[name].grad, rtol=2e-3, atol=1e-8)
    optimizer = torch.optim.Adagrad([{"params": group.params, "lr": group.lr} for group in groups])
    optimizer.step()
    policy.project_()
    changed = student(
        full_inputs[selection], all_mask[selection], all_probs[selection], all_indices[selection]
    ).detach()
    policy.finalize_()
    torch.testing.assert_close(
        student(
            full_inputs[selection],
            all_mask[selection],
            all_probs[selection],
            all_indices[selection],
        ),
        changed,
        rtol=0,
        atol=0,
    )
    reference_policy.close()
    dist.barrier()


def _tiny_native_glm(rank, *, hidden_size=64, sparse=False, hybrid=False, grouped=False):
    from nemo_automodel.components.models.common.utils import BackendConfig
    from nemo_automodel.components.models.glm5_next.config import (
        Glm5NextConfig,
        Glm5NextTextConfig,
        Glm5NextVisionConfig,
    )
    from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration

    torch.manual_seed(17)
    text = Glm5NextTextConfig(
        vocab_size=128,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        moe_intermediate_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=0,
        qk_nope_head_dim=16,
        v_head_dim=16,
        index_topk=16,
        index_head_dim=16,
        index_n_heads=4,
        index_kpool=4,
        linear_head_dim=16,
        linear_num_heads=4,
        linear_conv_kernel_dim=4,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        mlp_layer_types=["sparse" if sparse else "dense"] * 2,
        layer_types=(
            ["linear_attention", "deepseek_sparse_attention"]
            if hybrid
            else ["deepseek_sparse_attention"] * 2
        ),
        indexer_types=["full"] * 2,
        pad_token_id=0,
        torch_dtype="bfloat16",
    )
    vision = Glm5NextVisionConfig(
        depth=1,
        hidden_size=16,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=hidden_size,
        intermediate_size=32,
        projection_intermediate_size=64,
        torch_dtype="bfloat16",
    )
    teacher = (
        Glm5NextForConditionalGeneration(
            Glm5NextConfig(
                text_config=text, vision_config=vision, image_token_id=127, pad_token_id=0
            ),
            backend=BackendConfig(
                attn="sdpa",
                linear="torch",
                rms_norm="torch_fp32",
                experts="torch_mm" if grouped else "torch",
                dispatcher="torch",
                rope_fusion=False,
                enable_hf_state_dict_adapter=False,
            ),
        )
        .to(rank)
        .eval()
    )
    teacher.initialize_weights(torch.device("cuda", rank), dtype=torch.bfloat16)
    return teacher


@pytest.mark.parametrize("distributed_ptq", [False, True])
def test_native_glm_gptq_repair_and_diagnostics(tmp_path, distributed_ptq):
    import yaml
    from nemo_automodel.components.distributed.mesh import MeshContext
    from torch.distributed.device_mesh import DeviceMesh

    from examples.local_quantization_distillation.evaluate import evaluate_pair
    from examples.local_quantization_distillation.repair import quantize_student
    from modelopt.torch.distill.plugins.automodel import AutoModelLocalDistillationAdapter

    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    teacher = _tiny_native_glm(rank, hidden_size=256, sparse=True, grouped=True).requires_grad_(
        False
    )
    student = copy.deepcopy(teacher)
    settings = yaml.safe_load(
        Path("examples/local_quantization_distillation/configs/glm53_flash_iq2_s.yaml").read_text()
    )
    tokens = torch.randint(1, 127, (8, 16), device=rank)
    documents = torch.ones_like(tokens)
    documents[:, 8:] = 2
    labels = tokens.roll(-1, dims=1)
    labels[:, 7] = labels[:, -1] = -100
    batch = {
        "input_ids": tokens,
        "attention_mask": documents,
        "_packed_seq_ids": documents,
        "position_ids": torch.arange(8, device=rank).repeat(8, 2),
        "labels": labels,
        "valid_mask": documents.ne(0),
    }
    singleton = DeviceMesh("cuda", [rank], mesh_dim_names=("dp",), _init_backend=False)
    context = MeshContext.from_meshes(singleton)
    if distributed_ptq:
        from nemo_automodel.components.distributed.config import FSDP2Config, MoEParallelizerConfig
        from nemo_automodel.components.distributed.mesh import ParallelismSizes
        from nemo_automodel.components.distributed.model_parallelizer import parallelize_model

        context = MeshContext.build(
            FSDP2Config(),
            ParallelismSizes(cp_size=2, ep_size=2),
            moe_parallel_config=MoEParallelizerConfig(),
        )
        teacher = parallelize_model(teacher, context)
        student = parallelize_model(student, context)
        lane = context.device_mesh["dp"].get_local_rank()
        batch = {name: value[lane * 4 : lane * 4 + 4] for name, value in batch.items()}
    adapter = AutoModelLocalDistillationAdapter(context)
    quantize_student(student, [batch], adapter, settings)
    owners = [module for module in student.modules() if hasattr(module, "use_torch_mm")]
    assert owners and all(module.use_torch_mm for module in owners)
    # Ordinary observers must not switch execution into calibration capture.
    observer = owners[0].gate_and_up_projs_input_quantizer.register_forward_hook(
        lambda module, inputs, output: None
    )
    if distributed_ptq:
        from modelopt.torch.quantization.plugins.automodel import _QuantAutoModelExperts

        owners = [
            module for module in student.modules() if isinstance(module, _QuantAutoModelExperts)
        ]
        assert len(owners) == 2
        assert all(
            module.parallel_state.data_parallel_group.world_size()
            == context.moe_mesh["ep_shard"].size()
            for module in owners
        )
    baseline = evaluate_pair(teacher, student, [batch], mesh_context=context)
    observer.remove()
    output = (
        os.path.join(os.environ["REPAIR_GPU_TEST_ROOT"], "glm-native") if distributed_ptq else None
    )
    result = mtd.local_distill(
        student,
        teacher,
        [batch, batch],
        {"updates": 2, "global_batch_size": 8, "micro_batch_size": 2, "output_dir": output},
        validation_dataloader=[batch],
        adapter=adapter,
    )
    if distributed_ptq:
        fresh = parallelize_model(
            _tiny_native_glm(rank, hidden_size=256, sparse=True, grouped=True).requires_grad_(
                False
            ),
            context,
        )
        restored = adapter.restore(fresh, output)
    else:
        path = tmp_path / "glm-repaired.pt"
        mto.save(result.student, path)
        restored = mto.restore(copy.deepcopy(teacher), path)
    actual = evaluate_pair(teacher, restored, [batch], mesh_context=context)
    assert actual["kl_tokens"] == 128 and actual["next_token_targets"] == 112
    assert all(
        torch.isfinite(torch.tensor(actual[key])) for key in ("ppl", "kld_teacher_candidate")
    )
    assert all(row["units"] for row in result.metrics["history"])
    print(json.dumps({"glm_ptq": baseline, "glm_repaired_reloaded": actual}), flush=True)
    dist.barrier()


@pytest.mark.timeout(240)
@pytest.mark.parametrize("boundary", ["mlp", "self_attn", "decoder"])
def test_native_packed_cp_repair_matches_unsharded_reference(boundary):
    if int(os.environ.get("WORLD_SIZE", 1)) < 4:
        pytest.skip("Launch with torchrun using four CUDA ranks")
    from nemo_automodel.components.distributed.config import FSDP2Config
    from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
    from nemo_automodel.components.distributed.model_parallelizer import parallelize_model
    from torch.distributed.device_mesh import DeviceMesh

    from modelopt.torch.distill.plugins.automodel import AutoModelLocalDistillationAdapter

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    torch.manual_seed(17)
    teacher = _tiny_native_glm(rank, hybrid=boundary != "mlp")
    reference_teacher = copy.deepcopy(teacher)
    student = mtq.quantize(
        copy.deepcopy(teacher),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": (
                        "*language_model*weight_quantizer"
                        if boundary == "decoder"
                        else f"*{boundary}*weight_quantizer"
                    ),
                    "cfg": {"num_bits": 4, "axis": 0},
                },
            ],
            "algorithm": "max",
        },
    )
    reference = copy.deepcopy(student)
    tokens = torch.randint(1, 127, (8, 16), device=rank)
    docs = torch.ones_like(tokens)
    docs[:, 8:] = 2
    docs[0, 12:] = 0
    batch = {
        "input_ids": tokens,
        "attention_mask": docs,
        "_packed_seq_ids": docs,
        "position_ids": torch.arange(8, device=rank).repeat(8, 2),
        "valid_mask": docs.ne(0),
    }
    suffix = "" if boundary == "decoder" else "." + boundary
    config = {
        "units": [f"model.language_model.layers.{index}{suffix}" for index in range(2)],
        "updates": 2,
        "global_batch_size": 8,
        "scheduler": "constant",
        "warmup_fraction": 0,
    }
    singleton = DeviceMesh("cuda", [rank], mesh_dim_names=("dp",), _init_backend=False)
    expected = mtd.local_distill(
        reference,
        reference_teacher,
        [batch, {**batch, "input_ids": tokens.roll(1, dims=0)}],
        config,
        adapter=AutoModelLocalDistillationAdapter(MeshContext.from_meshes(singleton)),
    )
    context = MeshContext.build(FSDP2Config(), ParallelismSizes(cp_size=2))
    from nemo_automodel.components.moe.parallelizer import apply_cp

    for model in (student, teacher):
        apply_cp(model, context.device_mesh["cp"])
    student.requires_grad_(False)
    teacher.requires_grad_(False)
    student = parallelize_model(student, context)
    teacher = parallelize_model(teacher, context)
    frozen_fp32 = {
        name: value.to_local().detach().clone()
        for name, value in student.named_parameters()
        if value.dtype == torch.float32
    }
    lane = context.device_mesh["dp"].get_local_rank()
    local = {key: value[lane * 4 : lane * 4 + 4] for key, value in batch.items()}
    checkpoint_dir = os.path.join(os.environ["REPAIR_GPU_TEST_ROOT"], "cp-resume-" + boundary)
    resume_config = {**config, "checkpoint_dir": checkpoint_dir, "checkpoint_interval": 1}
    with pytest.raises(ValueError, match="Repair data exhausted"):
        mtd.local_distill(
            student,
            teacher,
            [local],
            resume_config,
            adapter=AutoModelLocalDistillationAdapter(context),
        )
    root = Path(checkpoint_dir)
    first_generation = root / (root / "latest").read_text().strip()
    committed = json.loads((first_generation / "COMMITTED.json").read_text())
    assert committed["step"] == 1
    assert set(committed["shards"]) == {f"rank-{rank}.pt" for rank in range(dist.get_world_size())}
    local_second = {**local, "input_ids": tokens.roll(1, dims=0)[lane * 4 : lane * 4 + 4]}
    result = mtd.local_distill(
        student,
        teacher,
        [local, local_second],
        {**resume_config, "resume_from": checkpoint_dir},
        adapter=AutoModelLocalDistillationAdapter(context),
    )
    assert not first_generation.exists()
    assert len(list(root.glob("*/COMMITTED.json"))) == 1
    # Compare projected native fields, rather than making an unsharded CP forward.
    from modelopt.torch.quantization.utils.core_utils import (
        enable_weight_access_and_writeback,
        module_name_maps,
    )

    for name, value in student.named_parameters():
        if name in frozen_fp32:
            torch.testing.assert_close(value.to_local(), frozen_fp32[name], rtol=0, atol=0)
    names = module_name_maps(student)
    for path in config["units"]:
        actual = student.get_submodule(path)
        target = expected.student.get_submodule(path)
        with enable_weight_access_and_writeback(actual, student, names, writeback=False):
            for (name, parameter), (_, reference_parameter) in zip(
                actual.named_parameters(), target.named_parameters(), strict=True
            ):
                torch.testing.assert_close(
                    parameter.float(), reference_parameter.float(), rtol=2e-2, atol=3e-3, msg=name
                )
    assert result.metrics["history"][0]["units"][config["units"][0]]["elements"] == int(
        docs.ne(0).sum()
    ) * 64 * (2 if boundary == "decoder" else 1)
    assert result.metrics["history"][0]["units"][config["units"][0]]["valid_tokens"] == int(
        docs.ne(0).sum()
    )
    dist.barrier()


@pytest.mark.parametrize(("tp_size", "formats"), [(1, "int4"), (2, "int4"), (2, "mixed")])
def test_fsdp2_subblock_repair_matches_unsharded_reference(tp_size, formats):
    if int(os.environ.get("WORLD_SIZE", 1)) < 2:
        pytest.skip("Launch with torchrun using at least two CUDA ranks")
    from nemo_automodel.components.distributed.mesh import MeshContext
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from modelopt.torch.distill.plugins.automodel import AutoModelLocalDistillationAdapter

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    size = dist.get_world_size()
    mesh = init_device_mesh(
        "cuda",
        (1, 1, size // tp_size, 1, tp_size),
        mesh_dim_names=("pp", "dp_replicate", "dp_shard", "cp", "tp"),
    )
    dp = mesh["dp_replicate", "dp_shard"]._flatten("dp")
    mesh["dp_replicate", "dp_shard", "cp"]._flatten("dp_cp")
    singleton = DeviceMesh("cuda", [rank], mesh_dim_names=("dp",), _init_backend=False)
    hidden = 512 if formats == "mixed" else 16
    # Isolate replicated-field reduction math from BF16 cancellation. The
    # native format/reload cases above separately exercise BF16 execution.
    dtype = torch.float32 if formats == "mixed" else torch.bfloat16
    if formats == "mixed":
        torch.set_float32_matmul_precision("highest")

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(hidden)
            self.mlp = nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden)
            )

        def forward(self, input):
            return input + self.mlp(self.norm(input))

    torch.manual_seed(12)
    teacher = nn.Sequential(Block(), Block()).to(rank, dtype).eval()
    reference_teacher = copy.deepcopy(teacher)
    student = mtq.quantize(
        copy.deepcopy(teacher),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "axis": 0}},
                *(
                    [
                        {
                            "quantizer_name": "*.mlp.0.weight_quantizer",
                            "cfg": {
                                "num_bits": (2, 1),
                                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                            },
                        },
                        {
                            "quantizer_name": "*.mlp.2.weight_quantizer",
                            "cfg": {"num_bits": "iq2_s", "backend": "ggml"},
                        },
                    ]
                    if formats == "mixed"
                    else []
                ),
            ],
            "algorithm": "max",
        },
    )
    reference = copy.deepcopy(student)
    inputs = torch.randn(size * 2, 4, hidden, device=rank, dtype=dtype)
    valid = torch.ones(size * 2, 4, device=rank, dtype=torch.bool)
    valid[0] = False
    batch = {"input": inputs, "valid_mask": valid}
    config = {
        "units": ["0.mlp", "1.mlp"],
        "updates": 1,
        "global_batch_size": size * 2,
        "scheduler": "constant",
        "warmup_fraction": 0,
    }

    def observe_scales(target, path):
        def observe(module, args):
            state = getattr(module[0].weight_quantizer, "_local_repair_state", None)
            if state is not None and hasattr(state, "deltas") and "double_scale" in state.deltas:
                target[path] = state.deltas["double_scale"]

        return observe

    reference_scales, actual_scales = {}, {}
    reference_handles = (
        [
            reference.get_submodule(path).register_forward_pre_hook(
                observe_scales(reference_scales, path)
            )
            for path in config["units"]
        ]
        if formats == "mixed"
        else []
    )
    expected = mtd.local_distill(
        reference,
        reference_teacher,
        [batch],
        config,
        adapter=AutoModelLocalDistillationAdapter(MeshContext.from_meshes(singleton)),
    )
    for handle in reference_handles:
        handle.remove()
    student.requires_grad_(False)
    teacher.requires_grad_(False)
    mp = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
    from modelopt.torch.distill.plugins.automodel.parallel import _quantizers_detached

    def wrap(model):
        with _quantizers_detached(model):
            if tp_size > 1:
                from torch.distributed.tensor.parallel import (
                    ColwiseParallel,
                    RowwiseParallel,
                    parallelize_module,
                )

                for block in model:
                    parallelize_module(
                        block.mlp, mesh["tp"], {"0": ColwiseParallel(), "2": RowwiseParallel()}
                    )
            for block in model:
                fully_shard(block, mesh=dp, mp_policy=mp)
            fully_shard(model, mesh=dp, mp_policy=mp)
        return model

    for model in (student, teacher):
        wrap(model)
    observations = []
    handles = [
        block.mlp.register_forward_pre_hook(
            lambda module, args: observations.append(
                all(
                    parameter.dtype == torch.float32
                    for parameter in module.parameters()
                    if parameter.requires_grad
                )
            )
        )
        for block in student
    ]
    if formats == "mixed":
        handles.extend(
            student.get_submodule(path).register_forward_pre_hook(
                observe_scales(actual_scales, path)
            )
            for path in config["units"]
        )
    local_batch = size * 2 // dp.size()
    lane = dp.get_local_rank()
    selection = slice(lane * local_batch, (lane + 1) * local_batch)
    output = os.path.join(
        os.environ["REPAIR_GPU_TEST_ROOT"], f"native-output-tp{tp_size}-{formats}"
    )
    try:
        result = mtd.local_distill(
            student,
            teacher,
            [{"input": inputs[selection], "valid_mask": valid[selection]}],
            {**config, "output_dir": output},
            adapter=AutoModelLocalDistillationAdapter(MeshContext.from_meshes(mesh)),
        )
    finally:
        for handle in handles:
            handle.remove()
    assert observations and all(observations)
    with torch.no_grad():
        actual = result.student(inputs[selection])
        predicted = expected.student(inputs)[selection]
    if formats == "mixed":
        weights = {}
        reference_parameters = dict(expected.student.named_parameters())
        for name, parameter in result.student.named_parameters():
            value = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
            weights[name] = float((value - reference_parameters[name]).detach().abs().max())
        print(
            json.dumps(
                {
                    "mixed_tp_max_weight_errors": weights,
                    "mixed_tp_max_output_error": float((actual - predicted).abs().max()),
                    "mixed_tp_scale_gradients": {
                        path: [float(actual_scales[path].grad), float(reference_scales[path].grad)]
                        for path in config["units"]
                    },
                }
            ),
            flush=True,
        )
        torch.testing.assert_close(actual, predicted, rtol=1e-4, atol=2e-6)
        assert actual_scales.keys() == reference_scales.keys() == set(config["units"])
        for path in config["units"]:
            torch.testing.assert_close(
                actual_scales[path].grad, reference_scales[path].grad, rtol=1e-3, atol=1e-7
            )
    else:
        torch.testing.assert_close(actual, predicted, rtol=2e-2, atol=3e-3)
    assert result.metrics["history"][0]["units"]["0.mlp"]["elements"] == int(valid.sum()) * hidden
    for path in config["units"]:
        actual = result.metrics["history"][0]["units"][path]["gradient_norm"]
        target = expected.metrics["history"][0]["units"][path]["gradient_norm"]
        torch.testing.assert_close(actual, target, rtol=2e-2, atol=1e-7)
    restored = wrap(copy.deepcopy(reference_teacher).requires_grad_(False))
    restored = AutoModelLocalDistillationAdapter(MeshContext.from_meshes(mesh)).restore(
        restored, output
    )
    for name, expected_value in result.student.state_dict().items():
        actual_value = restored.state_dict()[name]
        if hasattr(expected_value, "to_local"):
            actual_value, expected_value = actual_value.to_local(), expected_value.to_local()
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    if formats == "mixed":
        # An empty data lane must stop every owner before FSDP/TP execution.
        with pytest.raises(ValueError, match="empty"):
            mtd.local_distill(
                result.student,
                teacher,
                [] if rank == 0 else [{"input": inputs[selection], "valid_mask": valid[selection]}],
                config,
                adapter=AutoModelLocalDistillationAdapter(MeshContext.from_meshes(mesh)),
            )
        local = {"input": inputs[selection], "valid_mask": valid[selection]}
        for mismatch in ("validation", "training"):
            training = [local]
            if mismatch == "training" and mesh["dp"].get_local_rank() == 0:
                training = [
                    {key: value[:3] for key, value in local.items()},
                    {key: value[3:] for key, value in local.items()},
                ]
            with pytest.raises(ValueError, match="microbatch schedules"):
                mtd.local_distill(
                    result.student,
                    teacher,
                    training,
                    {**config, "micro_batch_size": 2},
                    validation_dataloader=([] if rank == 0 else [local])
                    if mismatch == "validation"
                    else None,
                    adapter=AutoModelLocalDistillationAdapter(MeshContext.from_meshes(mesh)),
                )
    dist.barrier()


@pytest.mark.parametrize("embedding_only", [False, True])
def test_native_pipeline_local_repair_matches_reference(embedding_only):
    if int(os.environ.get("WORLD_SIZE", 1)) != 4:
        pytest.skip("Launch with torchrun using four CUDA ranks")
    from nemo_automodel.components.distributed.mesh import MeshContext
    from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.qwen3.model import Qwen3ForCausalLM
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from transformers import Qwen3Config

    from modelopt.torch.distill.plugins.automodel import AutoModelLocalDistillationAdapter
    from modelopt.torch.quantization.utils.core_utils import (
        enable_weight_access_and_writeback,
        module_name_maps,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    mesh = init_device_mesh(
        "cuda", (2, 1, 2, 1, 1), mesh_dim_names=("pp", "dp_replicate", "dp_shard", "cp", "tp")
    )
    dp = mesh["dp_replicate", "dp_shard"]._flatten("dp")
    mesh["dp_replicate", "dp_shard", "cp"]._flatten("dp_cp")
    torch.manual_seed(43)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        dtype="bfloat16",
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    teacher = (
        Qwen3ForCausalLM(config, BackendConfig(attn="sdpa", linear="torch", rms_norm="torch"))
        .to(rank)
        .eval()
    )
    reference_teacher = copy.deepcopy(teacher)
    student = mtq.quantize(
        copy.deepcopy(teacher),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*mlp*weight_quantizer", "cfg": {"num_bits": 4, "axis": 0}},
            ],
            "algorithm": "max",
        },
    )
    reference = copy.deepcopy(student)
    tokens = torch.randint(0, 128, (8, 8), device=rank)
    valid = torch.ones_like(tokens, dtype=torch.bool)
    valid[0] = False
    batch = {"input_ids": tokens, "valid_mask": valid}
    recipe = {
        "updates": 1,
        "global_batch_size": 8,
        "micro_batch_size": 2,
        "scheduler": "constant",
        "warmup_fraction": 0,
    }
    singleton = DeviceMesh("cuda", [rank], mesh_dim_names=("dp",), _init_backend=False)
    expected = mtd.local_distill(
        reference,
        reference_teacher,
        [batch],
        recipe,
        adapter=AutoModelLocalDistillationAdapter(MeshContext.from_meshes(singleton)),
    )
    split = (
        [
            ["model.embed_tokens", "model.rotary_emb"],
            [
                "model.layers.0",
                "model.layers.1",
                "model.layers.2",
                "model.rotary_emb",
                "model.norm",
                "lm_head",
            ],
        ]
        if embedding_only
        else None
    )
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    def parallelize(part, **kwargs):
        for block in part.model.layers.values():
            fully_shard(block, mesh=dp, mp_policy=mp)
        fully_shard(part, mesh=dp, mp_policy=mp)

    pipelines = []
    for model in (student, teacher):
        model.requires_grad_(False)
        pipeline = AutoPipeline(
            world_mesh=mesh,
            pp_microbatch_size=2,
            pp_batch_size=4,
            module_fqns_per_model_part=split,
            pp_seq_len=8,
            device=torch.device("cuda", rank),
            dtype=torch.bfloat16,
        )
        pipeline.build(model, loss_fn=nn.MSELoss(), parallelize_fn=parallelize)
        pipelines.append(pipeline)
    student_part, teacher_part = (pipeline.parts[0] for pipeline in pipelines)
    lane = dp.get_local_rank()
    local = {key: value[lane * 4 : lane * 4 + 4] for key, value in batch.items()}
    checkpoint = os.path.join(os.environ["REPAIR_GPU_TEST_ROOT"], f"pp-{embedding_only}")
    active, observed = [False], [False]
    step = pipelines[1].step

    def teacher_step(*args, **kwargs):
        active[0] = True
        try:
            return step(*args, **kwargs)
        finally:
            active[0] = False

    def observe_replay(module, inputs):
        observed[0] |= active[0] and torch.is_grad_enabled()

    pipelines[1].step = teacher_step
    head_lengths = []
    if getattr(teacher_part, "lm_head", None) is not None:
        head_handle = teacher_part.lm_head.register_forward_pre_hook(
            lambda module, args: head_lengths.append(args[0].shape[-2])
        )
    handles = [
        module.register_forward_pre_hook(observe_replay)
        for name, module in student_part.named_modules()
        if name.endswith("mlp")
    ]
    result = mtd.local_distill(
        student_part,
        teacher_part,
        [local],
        {**recipe, "checkpoint_dir": checkpoint, "checkpoint_interval": 1},
        adapter=AutoModelLocalDistillationAdapter(
            MeshContext.from_meshes(mesh), teacher_pipeline=pipelines[1]
        ),
    )
    for handle in handles:
        handle.remove()
    pipelines[1].step = step
    if getattr(teacher_part, "lm_head", None) is not None:
        head_handle.remove()
        assert head_lengths and set(head_lengths) == {1}
    during_schedule = torch.tensor(int(observed[0]), device=rank)
    dist.all_reduce(during_schedule, op=dist.ReduceOp.MAX)
    assert not during_schedule.item(), "Student replay ran inside the active teacher PP schedule"
    if embedding_only and pipelines[0].info.has_first_stage:
        assert not result.metrics["units"]
    names = module_name_maps(student_part)
    for path, metrics in result.metrics["history"][0]["units"].items():
        unit = student_part.get_submodule(path)
        target = expected.student.get_submodule(path)
        with enable_weight_access_and_writeback(unit, student_part, names, writeback=False):
            for name, parameter in unit.named_parameters():
                torch.testing.assert_close(
                    parameter, target.get_parameter(name), rtol=2e-2, atol=1e-3
                )
        assert metrics["elements"] == int(valid.sum()) * 32
    root = Path(checkpoint)
    manifest = json.loads(
        (root / (root / "latest").read_text().strip() / "COMMITTED.json").read_text()
    )
    assert len(manifest["shards"]) == dist.get_world_size()
    dist.barrier()


@pytest.mark.timeout(240)
def test_native_flashnext_packed_cp_decoder_repair():
    """Replay real CP contexts without copying handles or losing packed boundaries."""
    if int(os.environ.get("WORLD_SIZE", 1)) != 4:
        pytest.skip("Launch with torchrun using four CUDA ranks")
    from _test_utils.automodel_families import build_model, cases
    from nemo_automodel.components.distributed.config import FSDP2Config, MoEParallelizerConfig
    from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
    from nemo_automodel.components.distributed.model_parallelizer import parallelize_model
    from nemo_automodel.components.moe.parallelizer import apply_cp

    from modelopt.torch.distill.plugins.automodel import AutoModelLocalDistillationAdapter
    from modelopt.torch.distill.plugins.automodel.families import native_family

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    record = next(row for row in cases() if row["model_id"] == "Qwen/Qwen3.8-Flash-Next")
    teacher = build_model(record, torch.device("cuda", rank)).requires_grad_(False)
    student = build_model(record, torch.device("cuda", rank))
    student.load_state_dict(teacher.state_dict())
    mtq.quantize(
        student,
        {
            "quant_cfg": [{"quantizer_name": "*", "enable": False}]
            + [
                {"quantizer_name": pattern, "cfg": {"num_bits": 4, "axis": 0}}
                for pattern in native_family(student).mlp_weight_patterns(student)
            ],
            "algorithm": "max",
        },
    )
    # Native PLE ownership is prepared by AutoModel's EP parallelizer.
    context = MeshContext.build(
        FSDP2Config(),
        ParallelismSizes(cp_size=2, ep_size=2),
        moe_parallel_config=MoEParallelizerConfig(),
    )
    for model in (student, teacher):
        apply_cp(model, context.device_mesh["cp"])
    adapter = AutoModelLocalDistillationAdapter(context)
    teacher = parallelize_model(teacher, context)
    student.requires_grad_(False)
    student = adapter.parallelize(student)
    torch.manual_seed(29 + context.device_mesh["dp"].get_local_rank())
    tokens = torch.randint(3, 127, (1, 32), device=rank)
    documents = torch.ones_like(tokens)
    documents[:, 16:24] = 2
    documents[:, 24:] = 0
    batch = {
        "input_ids": tokens,
        "attention_mask": documents,
        "_packed_seq_ids": documents,
        "position_ids": torch.arange(16, device=rank).repeat(1, 2),
        "valid_mask": documents.ne(0),
    }
    result = mtd.local_distill(
        student,
        teacher,
        [batch, batch],
        {
            "unit_types": ["decoder"],
            "updates": 2,
            "global_batch_size": 2,
            "micro_batch_size": 1,
            "scheduler": "constant",
            "warmup_fraction": 0,
        },
        adapter=adapter,
    )
    assert len(result.metrics["units"]) == record["minimum_pattern_prefix"]
    for update in result.metrics["history"]:
        assert all(
            unit["valid_tokens"] == 48 and unit["gradient_norm"] > 0
            for unit in update["units"].values()
        )
    dist.barrier()


def test_iq_scale_kernels_large_expert_bank():
    """Native decode, application and gradients must address banks beyond 2^31 elements."""
    from modelopt.torch.kernels.quantization.repair import apply_block_scales, decode_iq_levels
    from modelopt.torch.quantization.ggml.registry import GGML_FORMAT_REGISTRY

    elements = 2**31 + 256
    packed = torch.zeros((elements // 256, 50), device="cuda", dtype=torch.uint8)
    reference, _ = GGML_FORMAT_REGISTRY["iq1_s"].decode_int8(packed[:1], torch.tensor([1, 256]))
    levels = decode_iq_levels(packed, "iq1_s", (elements // 256, 256)).flatten()
    indices = torch.cat(
        [
            torch.arange(256, device="cuda"),
            torch.arange(2**31 - 256, 2**31 + 256, device="cuda"),
        ]
    )
    expected = reference.flatten().repeat(3)
    torch.testing.assert_close(levels[indices], expected, rtol=0, atol=0)
    scales = torch.ones(elements // 32, device="cuda", requires_grad=True)
    output = apply_block_scales(levels, scales, torch.bfloat16)
    torch.testing.assert_close(output[indices], expected.bfloat16(), rtol=0, atol=0)
    output.backward(torch.ones_like(output))
    groups = torch.cat(
        [
            torch.arange(8, device="cuda"),
            torch.arange(2**31 // 32 - 8, elements // 32, device="cuda"),
        ]
    )
    expected_gradient = reference.reshape(8, 32).float().sum(-1).repeat(3)
    torch.testing.assert_close(scales.grad[groups], expected_gradient, rtol=0, atol=0)
