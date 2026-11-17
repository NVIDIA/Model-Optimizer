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


"""Offline tutorial and full-vocabulary diagnostics through actual public model/head paths."""

import copy
import json
import math

import pytest
import torch

from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter


def test_local_diagnostics_full_vocabulary_and_document_targets():
    from examples.bypass.evaluate import diagnostic_statistics

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

    from examples.bypass.evaluate import evaluate_pair

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
    packed = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 0]]),
        "_packed_seq_ids": torch.tensor([[1, 1, 2, 2, 0]]),
        "labels": torch.tensor([[1, -100, 1, -100, -100]]),
    }
    packed_report = evaluate_pair(model, model, [packed], device_mesh=None)
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
        evaluate_pair(model, model, broken_loader(), device_mesh=None)


@pytest.mark.parametrize(
    "family_name",
    ["llama", "llama_hooked", "llama_quantized", "llama_biased", "muse_spark", "mistral_large4"],
)
def test_transformers_head_chunks_match_full_logits_and_restore(family_name):
    from _test_utils.transformers_families import build_model, cases

    from examples.bypass.evaluate import diagnostic_statistics, evaluate_pair

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
        calibration_adapter = TransformersBypassAdapter()
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
        device_mesh=None,
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
                    device_mesh=None,
                )
        finally:
            handle.remove()
        assert head.forward == original


def test_public_bypass_tutorial_roundtrip(tmp_path):
    from _test_utils.torch.transformers_models import create_tiny_llama_dir
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import modelopt.torch.opt as mto
    from examples.bypass.bypass import main
    from examples.bypass.prepare_data import PackedBatchSequence, prepare_packed_corpus

    teacher_path = create_tiny_llama_dir(
        tmp_path / "teacher",
        with_tokenizer=True,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=2,
    )
    tokenizer = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True)

    def rows():
        for i in range(512):
            yield (
                str(i),
                tokenizer.encode("a test sentence " * (2 + i % 7), add_special_tokens=False),
            )

    data = tmp_path / "data"
    prepare_packed_corpus(
        rows(),
        data,
        sequence_length=16,
        pack_counts={"train": 4, "calibration": 2, "validation": 2, "diagnostic": 2},
    )
    output = tmp_path / "output"
    result = main(
        [
            "--model",
            str(teacher_path),
            "--data",
            str(data),
            "--output",
            str(output),
            "--num-steps",
            "2",
            "--global-batch-size",
            "2",
            "--sequence-length",
            "16",
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ]
    )
    report = json.loads((output / "diagnostics.json").read_text())
    assert math.isfinite(report["kld_teacher_candidate"])
    assert report["kl_tokens"] == 32 and report["next_token_targets"] > 0
    restored = mto.restore(
        AutoModelForCausalLM.from_pretrained(teacher_path, dtype=torch.float32),
        output / "student.pt",
    )
    batch = PackedBatchSequence(data, "diagnostic", global_batch_size=1)[0]
    kwargs = {
        k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "position_ids")
    }
    with torch.no_grad():
        torch.testing.assert_close(
            result(**kwargs).logits, restored(**kwargs).logits, rtol=0, atol=0
        )
    assert not any("repair" in name for name, _ in restored.named_modules())
    assert sorted(p.name for p in output.iterdir()) == ["diagnostics.json", "student.pt"]


@pytest.mark.parametrize("frozen_format", ["fp8", "nvfp4"])
@pytest.mark.timeout(300)
def test_public_mixed_shared_and_routed_bypass_roundtrip(frozen_format, tmp_path):
    from _test_utils.transformers_families import build_model, cases, hf_torch_kernels

    import modelopt.torch.distill as mtd
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.ggml.common import pinned_packed_weight

    if frozen_format == "nvfp4" and not torch.cuda.is_available():
        pytest.skip("Real frozen NVFP4 mixed-format integration requires CUDA")
    torch.manual_seed(157)
    record = cases(["qwen3_next"])[0]
    device = "cuda" if frozen_format == "nvfp4" else "cpu"
    teacher = build_model(record, device).float()
    batch = {"input_ids": torch.randint(3, 100, (2, 8), device=device)}
    frozen_cfg = (
        {"num_bits": (4, 3), "axis": None}
        if frozen_format == "fp8"
        else {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
    )
    with hf_torch_kernels(teacher):
        student = mtq.quantize(
            copy.deepcopy(teacher),
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    {
                        "quantizer_name": "*mlp.shared_expert.*weight_quantizer",
                        "cfg": {"num_bits": "iq2_s", "backend": "ggml"},
                    },
                    {
                        "quantizer_name": "*mlp.experts.gate_up_proj_weight_quantizers*",
                        "cfg": {"num_bits": "iq1_m", "backend": "ggml"},
                    },
                    {
                        "quantizer_name": "*mlp.experts.down_proj_weight_quantizers*",
                        "cfg": frozen_cfg,
                    },
                ],
                "algorithm": "max",
            },
            lambda m: m(**batch),
        )
        frozen = {
            n: p.detach().clone()
            for n, p in student.state_dict().items()
            if "experts.down_proj" in n or ".mlp.gate." in n
        }
        before = student.model.layers[0].mlp.shared_expert.up_proj.weight.detach().clone()
        assert (
            mtd.bypass(
                student,
                teacher,
                [batch],
                {"num_steps": 1, "global_batch_size": 2, "sequence_length": 8},
            )
            is student
        )
        assert not torch.equal(student.model.layers[0].mlp.shared_expert.up_proj.weight, before)
        for layer in student.model.layers:
            assert all(
                pinned_packed_weight(q, "iq1_m") is not None
                for q in layer.mlp.experts.gate_up_proj_weight_quantizers
            )
        assert all(torch.equal(student.state_dict()[n], p) for n, p in frozen.items())
        with torch.no_grad():
            expected = student(**batch).logits
        mto.save(student, tmp_path / "mixed.pt")
        restored = mto.restore(copy.deepcopy(teacher), tmp_path / "mixed.pt")
        with torch.no_grad():
            torch.testing.assert_close(restored(**batch).logits, expected, rtol=0, atol=0)
