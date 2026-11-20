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

"""Offline correctness cases for the KL example; no checkpoints or datasets are downloaded."""

import copy
import importlib
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoConfig,
    BloomConfig,
    BloomForCausalLM,
    GenerationConfig,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.fixture
def kl_eval(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples" / "llm_eval"))
    return importlib.import_module("kl_eval")


def _tiny_model(vocab_size=16, n_positions=16, tuple_default=False):
    tokens = {"vocab_size": vocab_size, "bos_token_id": 1, "eos_token_id": 0, "pad_token_id": 0}
    if tuple_default:
        return BloomForCausalLM(
            BloomConfig(hidden_size=8, n_layer=1, n_head=1, return_dict=False, **tokens)
        ).eval()
    return GPT2LMHeadModel(
        GPT2Config(n_positions=n_positions, n_embd=8, n_layer=1, n_head=1, **tokens)
    ).eval()


def _expected_kl(reference, candidate, top_k, positions=slice(None)):
    """Independent unchunked FP64 oracle using probability ratios and sorted token IDs."""
    p, q = reference[positions].double(), candidate[positions].double()
    ids = p.argsort(dim=-1, descending=True)[:, :top_k]
    p, q = p.gather(-1, ids), q.gather(-1, ids)
    p, q = p.softmax(-1), q.softmax(-1)
    return (p * (p / q).log()).sum(-1).mean().item()


@pytest.mark.parametrize(
    ("count", "expected"),
    [(1, [0]), (3, [0, 2, 5]), (6, list(range(6))), (8192, list(range(6)))],
)
def test_position_grid_includes_endpoints_and_caps_at_response_length(kl_eval, count, expected):
    assert kl_eval.spaced_positions(6, count) == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_reduction_direction_shared_support_chunks_and_invalid_logits(kl_eval, dtype):
    p = torch.tensor([[0.5, 0.3, 0.2]]).log().to(dtype).repeat(33, 1)
    q = torch.tensor([[0.25, 0.25, 0.5]]).log().to(dtype).repeat(33, 1)
    ids = torch.tensor([[0, 1]] * 33, dtype=torch.int32)
    log_p = p[:, :2].float().log_softmax(-1)
    assert kl_eval._cached_kl(log_p, ids, p) == pytest.approx(0, abs=1e-7)
    expected = _expected_kl(p, q, 2)  # Candidate favorite is outside reference support.
    assert kl_eval._cached_kl(log_p, ids, q) == pytest.approx(expected, abs=1e-6)
    assert kl_eval._cached_kl(torch.zeros(33, 1), ids[:, :1], q) == pytest.approx(0, abs=1e-7)
    q[:, -1] = float("nan")
    assert kl_eval._cached_kl(log_p, ids, q) == pytest.approx(expected, abs=1e-6)
    q[:, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite KL"):
        kl_eval._cached_kl(log_p, ids, q)


@pytest.mark.parametrize(
    "projection", ["tensor", "tuple_default", "integer", "full", "sharded", "quantized"]
)
@pytest.mark.parametrize(
    ("eos_token_id", "valid_lengths"),
    [(None, [4, 4, 4]), ([0, 8], [2, 1, 4]), (None, [1024] * 3)],
)
@torch.inference_mode()
def test_scores_match_unpadded_predictions_through_eos(
    kl_eval, monkeypatch, tmp_path, eos_token_id, valid_lengths, projection
):
    torch.manual_seed(0)
    length = max(valid_lengths)
    vocab_size = 2049 if length == 1024 else 16
    reference = _tiny_model(vocab_size, max(16, length + 4), projection == "tuple_default")
    candidate = copy.deepcopy(reference)
    candidate.lm_head.weight[:, 0].add_(torch.linspace(-3, 3, vocab_size))
    if projection == "quantized":
        mtq.quantize(
            candidate,
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    {
                        "quantizer_name": "lm_head.input_quantizer",
                        "enable": True,
                        "cfg": {"num_bits": 4, "axis": None, "type": "dynamic"},
                    },
                ],
                "algorithm": None,
            },
        )

        def prompt_outlier(_module, _args, hidden):
            hidden = hidden.clone()
            hidden[:, 0] *= 100  # An omitted prompt row must still affect the head's scale.
            return hidden

        candidate.transformer.ln_f.register_forward_hook(prompt_outlier)
    for model in (reference, candidate):
        # Legacy interfaces still run real Transformers forwards, with their original projection.
        original = model.forward
        if projection == "sharded":
            model.hf_device_map = {"model": "cuda:0", "lm_head": "cuda:1"}
        if projection == "integer":

            def forward(input_ids, logits_to_keep: int = 0, _original=original, **kwargs):
                return _original(input_ids=input_ids, logits_to_keep=logits_to_keep, **kwargs)

            monkeypatch.setattr(model, "forward", forward)
        elif projection == "full":

            def forward(input_ids, _original=original, **kwargs):
                return _original(input_ids=input_ids, **kwargs)

            monkeypatch.setattr(model, "forward", forward)
    prompts = [[0, 5, 6], [0, 5], [0, 5, 6, 7]]
    responses = [row * (length // 4) for row in [[7, 8, 0, 0], [8, 0, 0, 0], [7, 6, 5, 4]]]
    sequences = torch.tensor([[0] * (4 - len(p)) + p + r for p, r in zip(prompts, responses)])
    monkeypatch.setattr(reference, "generate", lambda **_kwargs: sequences)
    if projection == "tuple_default":
        assert isinstance(reference(torch.tensor([prompts[0]]), use_cache=False), tuple)
    for score_tokens in (0, 512 if vocab_size == 2049 else 2):
        top_k = 2048 if vocab_size == 2049 else 4
        expected = []
        for prompt, response, length in zip(prompts, responses, valid_lengths):
            inputs = torch.tensor([prompt + response[: length - 1]])
            p, q = [
                model(inputs, use_cache=False, return_dict=True).logits[0, len(prompt) - 1 :]
                for model in (reference, candidate)
            ]
            selected = slice(None)
            if score_tokens and length > score_tokens:
                selected = [i * (length - 1) // (score_tokens - 1) for i in range(score_tokens)]
            if projection == "quantized":
                positions = (
                    list(range(length))[selected] if isinstance(selected, slice) else selected
                )
                rows = kl_eval._scored_logits(candidate, prompt, response[:length], positions)
                torch.testing.assert_close(rows, q[selected], rtol=1e-6, atol=1e-6)
            expected.append(_expected_kl(p, q, top_k, selected))
        assert min(expected) > 1e-6
        directory = tmp_path / f"reference-{score_tokens}"
        generation = GenerationConfig(
            max_new_tokens=max(valid_lengths), eos_token_id=eos_token_id, pad_token_id=0
        )
        projected = []
        hook = reference.lm_head.register_forward_pre_hook(
            lambda _m, inputs: projected.append(inputs[0].shape[:2])
        )
        try:
            options = (
                {}
                if (top_k, score_tokens) == (2048, 512)
                else {"top_k": top_k, "score_tokens": score_tokens}
            )
            kl_eval._prepare_reference(reference, prompts, generation, directory, {}, **options)
        finally:
            hook.remove()
        assert all(batch == 1 for batch, _ in projected)
        if score_tokens and projection == "tensor":
            assert [rows for _, rows in projected] == [min(score_tokens, n) for n in valid_lengths]
        if projection == "sharded":
            assert [rows for _, rows in projected] == valid_lengths
        actual = kl_eval.evaluate(candidate, directory)
        assert isinstance(actual, float)
        assert actual == pytest.approx(math.fsum(expected) / len(prompts), rel=1e-6, abs=1e-6)


@pytest.fixture
def cached_evaluation(kl_eval, tmp_path):
    torch.manual_seed(0)
    reference = _tiny_model()
    directory = tmp_path / "reference"
    prompts = [[1, 3], [1, 4, 3]]
    generation = GenerationConfig(max_new_tokens=4, eos_token_id=0, pad_token_id=0)
    kl_eval._prepare_reference(
        reference, prompts, generation, directory, {}, top_k=4, score_tokens=2
    )
    return SimpleNamespace(
        reference=reference,
        candidate=copy.deepcopy(reference),
        directory=directory,
        prompts=prompts,
        generation=generation,
    )


@torch.inference_mode()
def test_reference_rescores_candidates_and_refuses_overwrite(kl_eval, cached_evaluation):
    case = cached_evaluation
    original = (case.directory / "manifest.json").read_bytes()
    fresh = kl_eval.evaluate(case.candidate, case.directory)
    assert fresh == pytest.approx(0, abs=1e-7)
    case.candidate.lm_head.weight[:, 0].add_(torch.linspace(-3, 3, 16))
    assert kl_eval.evaluate(case.candidate, case.directory) > fresh + 1e-6
    with pytest.raises(FileExistsError):
        kl_eval._prepare_reference(
            case.reference, case.prompts, case.generation, case.directory, {}, top_k=4
        )
    assert (case.directory / "manifest.json").read_bytes() == original


@pytest.mark.parametrize("corruption", ["positions", "statistics"])
def test_reference_rejects_mismatched_positions_and_statistics(
    kl_eval, cached_evaluation, corruption
):
    case = cached_evaluation
    if corruption == "positions":
        path = case.directory / "manifest.json"
        manifest = json.loads(path.read_text())
        positions = manifest["records"][0]["positions"]
        manifest["records"][0]["positions"] = [999] * len(positions)
        path.write_text(json.dumps(manifest))
    else:
        shard = case.directory / "0.safetensors"
        save_file({name: t[:1] for name, t in load_file(str(shard)).items()}, str(shard))
    with pytest.raises(ValueError, match="Invalid reference"):
        kl_eval.evaluate(case.candidate, case.directory)


@pytest.mark.parametrize(
    ("model", "backend", "layerwise", "error"),
    [
        ("other", None, [], "--model differs"),
        ("base", "eager", [], "Attention backend differs"),
        ("base", None, [{"export_dir": "out"}], "Layerwise"),
        ("base", None, [{"checkpoint_dir": "resume"}], "Layerwise"),
    ],
)
def test_evaluation_refuses_inputs_that_differ_from_the_reference(
    kl_eval, model, backend, layerwise, error
):
    manifest = {"metadata": {"model": "base", "attn_implementation": "sdpa"}}
    args = SimpleNamespace(model=model, attn_implementation=backend)
    with pytest.raises(ValueError, match=error):
        kl_eval._check_reference_match(args, manifest, layerwise)
    assert args.attn_implementation == backend


def test_model_guards_require_bf16_base_and_fake_quant_candidate(kl_eval):
    with pytest.raises(ValueError, match="BF16"):
        kl_eval._require_bf16_base(_tiny_model())
    kl_eval._require_bf16_base(_tiny_model().to(torch.bfloat16))
    quantizers = torch.nn.ModuleList(
        TensorQuantizer(QuantizerAttributeConfig(num_bits=8, fake_quant=fake))
        for fake in (True, False)
    )
    kl_eval._require_fake_quant(quantizers[:1])
    for candidate in (
        quantizers,
        torch.nn.ModuleList([TensorQuantizer(QuantizerAttributeConfig(enable=False))]),
    ):
        with pytest.raises(ValueError, match="fake quantization"):
            kl_eval._require_fake_quant(candidate)


@pytest.fixture
def blended_fixture(kl_eval, monkeypatch):
    backend = Tokenizer(WordLevel({"[UNK]": 0, "user": 1, "assistant": 2, "alpha": 3, "beta": 4}))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
        "{% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
    )
    dataset = Dataset.from_list(
        [
            {
                "cohort_rank": index,
                "messages": [{"role": "user", "content": content}],
                "gold_answer": "must not enter prompt",
            }
            for index, content in enumerate(["alpha beta", "beta", "alpha", "alpha " * 128])
        ]
    )
    # Replace only the external Hub read; exercise real dataset selection and chat tokenization.
    monkeypatch.setattr("datasets.load_dataset", lambda *_a, **_kw: dataset)
    return tokenizer


def test_blended_prompts_keep_order_and_refuse_truncation(kl_eval, blended_fixture):
    prompts = kl_eval._blended_prompts(blended_fixture, 3, "revision", 256, 4)
    assert prompts == [[1, 3, 4, 2], [1, 4, 2], [1, 3, 2]]
    with pytest.raises(ValueError, match="not truncated"):
        kl_eval._blended_prompts(blended_fixture, 4, "revision", 133, 4)


def test_cli_prepares_once_and_scores_repeated_candidates_identically(
    kl_eval, blended_fixture, tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(kl_eval.__file__).parents[1] / "hf_ptq"))
    hf_ptq = importlib.import_module("hf_ptq")
    checkpoint = tmp_path / "base"
    _tiny_model().to(torch.bfloat16).save_pretrained(checkpoint)
    tokenizer = blended_fixture
    tokenizer.pad_token = tokenizer.eos_token = "[UNK]"
    tokenizer.save_pretrained(checkpoint)
    recipe = tmp_path / "int8.yaml"
    recipe.write_text(
        "# modelopt-schema: modelopt.recipe.config.ModelOptPTQRecipe\n"
        + json.dumps({"metadata": {}, "quantize": hf_ptq.mtq.INT8_DEFAULT_CFG})
    )
    loads, generation_batches = [], []

    def load(*_args, **kwargs):
        loads.append(kwargs.get("attn_implementation"))
        return GPT2LMHeadModel.from_pretrained(
            checkpoint, dtype=torch.bfloat16, attn_implementation=kwargs.get("attn_implementation")
        )

    generate = GPT2LMHeadModel.generate

    def record_generation(model, *args, **kwargs):
        generation_batches.append(kwargs["input_ids"].shape[0])
        return generate(model, *args, **kwargs)

    # Exercise the real loader and calibration-argument handling; replace only placement and data.
    monkeypatch.setattr(hf_ptq, "get_model", load)
    monkeypatch.setattr(hf_ptq, "get_tokenizer", lambda *_a, **_kw: tokenizer)
    monkeypatch.setattr(
        hf_ptq,
        "resolve_distribution",
        lambda _args: (
            SimpleNamespace(is_expert_parallel=False, is_fsdp_dense=False),
            AutoConfig.from_pretrained(checkpoint),
        ),
    )
    monkeypatch.setattr(
        hf_ptq, "distributed_run", lambda **_kw: nullcontext(SimpleNamespace(device="cpu"))
    )
    monkeypatch.setattr(
        hf_ptq,
        "make_calib_dataloader",
        lambda *_a, **_kw: ([{"input_ids": torch.randint(0, 16, (1, 8))}], None),
    )
    monkeypatch.setattr(GPT2LMHeadModel, "generate", record_generation)
    # Treat the fixture as a Hub snapshot whose directory name is its commit.
    monkeypatch.setattr(hf_ptq, "ensure_local_checkpoint", lambda path: ("org/model", path))
    common = ["--model", str(checkpoint), "--reference", str(tmp_path / "reference")]
    prepare = ["prepare-reference", *common, "--num_examples", "3", "--max_new_tokens", "4"]
    prepare += ["--generation_batch_size", "2", "--top_k", "4", "--score_tokens", "2"]
    monkeypatch.setattr(sys, "argv", ["kl_eval.py", *prepare, "--attn_implementation", "eager"])
    config_path = checkpoint / "config.json"
    original_config = config_path.read_text()
    config_path.write_text(
        json.dumps({**json.loads(original_config), "quantization_config": {"format": "packed"}})
    )
    with pytest.raises(ValueError, match="unquantized BF16"):
        kl_eval.main()
    assert not loads
    config_path.write_text(original_config)
    assert kl_eval.main() is None
    assert generation_batches == [2, 1]

    def unexpected_generation(*_args, **_kwargs):
        pytest.fail("Candidate evaluation must not generate or load the dataset")

    monkeypatch.setattr(GPT2LMHeadModel, "generate", unexpected_generation)
    monkeypatch.setattr(kl_eval, "_blended_prompts", unexpected_generation)
    evaluate = ["evaluate", *common, "--recipe", str(recipe), "--batch_size", "1"]
    monkeypatch.setattr(sys, "argv", ["kl_eval.py", *evaluate, "--calib_size", "1"])
    first, second = kl_eval.main(), kl_eval.main()
    assert loads == ["eager"] * 3  # Evaluation inherits the reference's attention backend.
    assert math.isfinite(first) and first > 0
    assert first == pytest.approx(second, abs=1e-7)
    moved = tmp_path / "newer-revision"
    monkeypatch.setattr(hf_ptq, "ensure_local_checkpoint", lambda _path: ("org/model", moved))
    with pytest.raises(ValueError, match="Hub model revision differs"):
        kl_eval.main()
    assert len(loads) == 3
