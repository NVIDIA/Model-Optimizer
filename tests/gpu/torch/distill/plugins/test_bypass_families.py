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

"""Real reduced release models: packed text, IQ updates, frozen teacher and native reload."""

import copy

import pytest
import torch

pytest.importorskip("transformers", minversion="5.23")

from _test_utils.transformers_families import build_model, cases, hf_torch_kernels, release_cases

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter
from modelopt.torch.models import get_spec
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pinned_packed_weight
from modelopt.torch.quantization.ggml.scale_fields import decode_scale_fields, replace_scale_fields
from modelopt.torch.quantization.nn import QuantModule

_PORTED = {"deepseek_v41", "kimi_k31", "step5", "mimo_v2"}


@pytest.mark.parametrize("record", release_cases(), ids=lambda r: r["model_id"])
@pytest.mark.timeout(300)
def test_release_text_bypass_and_native_reload(record, tmp_path):
    assert record["revision"]
    torch.manual_seed(149)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = build_model(record, device)
    if record["family"] in _PORTED:
        spec = get_spec(record["family"])
        assert spec is not None
        layout = spec.moe_spec
        assert layout is not None and any(layout.matches(module) for module in teacher.modules())
    teacher_before = {n: p.detach().clone() for n, p in teacher.state_dict().items()}
    ids = torch.randint(3, 100, (2, 8), device=device)
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "position_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], device=device).expand(2, -1),
        "_packed_seq_ids": torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]], device=device).expand(2, -1),
    }
    forward = {k: v for k, v in batch.items() if k != "_packed_seq_ids"}
    adapter = TransformersBypassAdapter()
    with hf_torch_kernels(teacher):
        student = mtq.quantize(
            copy.deepcopy(teacher),
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    *[
                        {
                            "quantizer_name": pattern,
                            "cfg": {"num_bits": "iq2_xxs", "backend": "ggml"},
                        }
                        for pattern in adapter.weight_patterns(teacher)
                    ],
                ],
                "algorithm": "max",
            },
            lambda model: model(**forward),
        )
        if record["family"] == "step5":
            shared = [
                q
                for n, q in student.named_modules()
                if ("share_expert" in n or "shared_experts" in n) and n.endswith("weight_quantizer")
            ]
            assert shared and all(q.is_enabled and q.num_bits == "iq2_xxs" for q in shared)
        original = {}
        fmt = IQ_FORMAT_REGISTRY["iq2_xxs"]
        for module in student.modules():
            if isinstance(module, QuantModule):
                for weight, quantizer in module.iter_weights_for_calibration():
                    if quantizer.is_enabled and quantizer.backend == "ggml":
                        original[quantizer] = fmt.pack(weight, quantizer).clone()
        frozen = {
            n: p.detach().clone()
            for n, p in student.named_parameters()
            if ".attn." in n or ".self_attn." in n or ".gate." in n or "norm" in n
        }
        assert original
        assert (
            mtd.bypass(
                student,
                teacher,
                [batch, batch],
                {"num_steps": 2, "global_batch_size": 2, "sequence_length": 8},
            )
            is student
        )
        changed = False
        for quantizer, before in original.items():
            after = pinned_packed_weight(quantizer, "iq2_xxs")
            global_scale, local_scale, _ = decode_scale_fields(before, fmt)
            assert torch.equal(replace_scale_fields(after, fmt, global_scale, local_scale), before)
            changed |= not torch.equal(before, after)
        assert changed
        assert all(torch.equal(student.get_parameter(n), p) for n, p in frozen.items())
        assert all(torch.equal(teacher.state_dict()[n], p) for n, p in teacher_before.items())
        assert not any("repair" in n for n, _ in student.named_modules())
        with torch.no_grad():
            expected = student(**forward).logits
        mto.save(student, tmp_path / "student.pt")
        restored = mto.restore(build_model(record, device), tmp_path / "student.pt")
        with torch.no_grad(), hf_torch_kernels(restored):
            torch.testing.assert_close(restored(**forward).logits, expected, rtol=0, atol=0)


def _native_port_worker(rank, size, record):
    from torch.distributed.device_mesh import init_device_mesh
    from transformers.distributed.utils import TransformersDeviceMesh

    assert size == 4
    torch.cuda.set_device(rank)
    torch.manual_seed(151)
    device = torch.device("cuda", rank)
    teacher = build_model(record, device)
    # Equal router scores deliberately leave two of four expert owners empty.
    with torch.no_grad():
        for name, module in teacher.named_modules():
            if name.endswith(".gate") and isinstance(getattr(module, "weight", None), torch.Tensor):
                module.weight.zero_()
    batches = [{"input_ids": torch.randint(3, 100, (1, 8), device=device)} for _ in range(8)]
    adapter = TransformersBypassAdapter()
    with hf_torch_kernels(teacher):
        student = mtq.quantize(
            copy.deepcopy(teacher),
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    *[
                        {"quantizer_name": p, "cfg": {"num_bits": "iq2_xxs", "backend": "ggml"}}
                        for p in adapter.weight_patterns(teacher)
                    ],
                ],
                "algorithm": "max",
            },
            lambda m: m(**batches[0]),
        )
        recipe = {"num_steps": 2, "global_batch_size": 4, "sequence_length": 8}
        single = init_device_mesh("cuda", (size, 1), mesh_dim_names=("owners", "dp"))["dp"]
        reference = copy.deepcopy(student)
        mtd.bypass(reference, copy.deepcopy(teacher), batches, recipe, device_mesh=single)
        with torch.no_grad():
            expected = reference(**batches[0]).logits.detach().cpu()
        del reference
        manager = TransformersDeviceMesh(
            init_device_mesh(
                "cuda", (1, 1, 4, 1), mesh_dim_names=("pp", "dp_replicate", "fsdp", "tp")
            ),
            init_device_mesh(
                "cuda", (1, 1, 1, 4), mesh_dim_names=("pp", "dp_replicate", "efsdp", "ep")
            ),
        )
        adapter = TransformersBypassAdapter(manager)
        adapter.parallelize(student)
        adapter.parallelize(teacher)
        before = {
            n: p.to_local().clone() if hasattr(p, "to_local") else p.clone()
            for n, p in teacher.state_dict().items()
        }
        local = [batches[step * 4 + rank] for step in range(2)]
        assert mtd.bypass(student, teacher, local, recipe, device_mesh=manager) is student
        assert all(
            len(ids) == 1
            for ids in adapter.context.parallel_metadata[id(student)]["expert_global_ids"].values()
        )
        for name, value in teacher.state_dict().items():
            value = value.to_local() if hasattr(value, "to_local") else value
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        with torch.no_grad():
            actual = student(**batches[0]).logits
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.025, atol=0.03)


@pytest.mark.parametrize(
    "record",
    list({r["family"]: r for r in release_cases() if r["family"] in _PORTED}.values())
    + cases(["qwen3_next"]),
    ids=lambda r: r["family"],
)
@pytest.mark.timeout(300)
def test_native_port_updates_and_empty_expert_owners(native_workers, record):
    native_workers.run(_native_port_worker, record)
