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
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
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
    LlavaConfig,
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


@pytest.mark.parametrize(
    "corruption",
    ["positions", "float_positions", "top_k", "missing", "truncated", "oversized", "replaced"],
)
def test_reference_rejects_corrupted_records_and_shards(
    kl_eval, cached_evaluation, monkeypatch, corruption
):
    case = cached_evaluation
    path = case.directory / "manifest.json"
    manifest = json.loads(path.read_text())
    positions = manifest["records"][0]["positions"]
    shard = case.directory / "0.safetensors"
    if corruption == "positions":
        manifest["records"][0]["positions"] = [999] * len(positions)
    elif corruption == "float_positions":
        manifest["records"][0]["positions"] = [float(p) for p in positions]
    elif corruption == "top_k":
        manifest["top_k"] = 8
    elif corruption == "missing":
        (case.directory / "1.safetensors").unlink()
    elif corruption == "truncated":
        shard.write_bytes(shard.read_bytes()[:-8])
    elif corruption == "oversized":  # E.g. a model weight shard copied by mistake.
        shard.write_bytes(shard.read_bytes() + bytes(kl_eval.SHARD_HEADER_BYTES))
        monkeypatch.setattr(Path, "read_bytes", lambda _path: pytest.fail("read before size check"))
    else:  # A valid shard belonging to another record.
        shard.write_bytes((case.directory / "1.safetensors").read_bytes())
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid reference"):
        kl_eval.evaluate(case.candidate, case.directory)


@pytest.mark.parametrize(
    ("model", "backend", "error"),
    [("other", None, "--model differs"), ("base", "eager", "Attention backend differs")],
)
def test_evaluation_refuses_inputs_that_differ_from_the_reference(kl_eval, model, backend, error):
    manifest = {"metadata": {"model": "base", "attn_implementation": "sdpa"}}
    args = SimpleNamespace(model=model, attn_implementation=backend)
    with pytest.raises(ValueError, match=error):
        kl_eval._check_reference_match(args, manifest)
    assert args.attn_implementation == backend


def test_checkpoint_identity_tracks_hub_revision_config_and_weight_files(kl_eval, tmp_path):
    config, weights = tmp_path / "config.json", tmp_path / "model.safetensors"
    config.write_text("{}")
    weights.write_bytes(b"weights")
    local = kl_eval._checkpoint_identity(None, tmp_path)
    (tmp_path / "kl-reference").mkdir()  # A reference stored inside the checkpoint.
    (tmp_path / "kl-reference" / "0.safetensors").write_bytes(b"reference shard")
    assert kl_eval._checkpoint_identity(None, tmp_path) == local
    for change in (
        lambda: config.write_text('{"activation_function": "relu"}'),
        lambda: os.utime(weights, ns=(1, 1)),
        lambda: weights.write_bytes(b"new weights"),
    ):
        change()
        assert kl_eval._checkpoint_identity(None, tmp_path) != local
        local = kl_eval._checkpoint_identity(None, tmp_path)
    assert kl_eval._checkpoint_identity("org/model", tmp_path / "commit") == "commit"


def test_checkpoint_identity_follows_index_to_nested_shards(kl_eval, tmp_path):
    shard = tmp_path / "weights" / "model-00001-of-00001.safetensors"
    shard.parent.mkdir()
    shard.write_bytes(b"weights")
    index = {"weight_map": {"lm_head.weight": "weights/model-00001-of-00001.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    local = kl_eval._checkpoint_identity(None, tmp_path)
    (tmp_path / "kl-reference").mkdir()  # Not named by the index.
    (tmp_path / "kl-reference" / "0.safetensors").write_bytes(b"reference shard")
    assert kl_eval._checkpoint_identity(None, tmp_path) == local
    shard.write_bytes(b"new weights")
    assert kl_eval._checkpoint_identity(None, tmp_path) != local


def test_config_guard_reads_quantization_metadata_on_nested_text_config(kl_eval):
    kl_eval._require_unquantized_config(LlavaConfig())
    packed = {"model_type": "llama", "quantization_config": {"format": "packed"}}
    with pytest.raises(ValueError, match="unquantized BF16"):
        kl_eval._require_unquantized_config(LlavaConfig(text_config=packed))


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


@pytest.fixture
def cli(kl_eval, blended_fixture, tmp_path, monkeypatch):
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
    loads, generation_batches, during_load = [], [], []

    def load(path, *_args, **kwargs):
        loads.append(kwargs.get("attn_implementation"))
        for hook in during_load:
            hook()
        return GPT2LMHeadModel.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation=kwargs.get("attn_implementation")
        )

    generate = GPT2LMHeadModel.generate

    def record_generation(model, *args, **kwargs):
        generation_batches.append(kwargs["input_ids"].shape[0])
        return generate(model, *args, **kwargs)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["kl_eval.py", *argv])
        return kl_eval.main()

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
    prepare += ["--attn_implementation", "eager"]
    evaluate = ["evaluate", *common, "--batch_size", "1", "--calib_size", "1"]
    return SimpleNamespace(
        hf_ptq=hf_ptq,
        checkpoint=checkpoint,
        reference=tmp_path / "reference",
        recipe=recipe,
        during_load=during_load,
        loads=loads,
        generation_batches=generation_batches,
        run=run,
        prepare=prepare,
        evaluate=[*evaluate, "--recipe", str(recipe)],
        evaluate_without_recipe=evaluate,
    )


def test_cli_prepares_once_and_scores_repeated_candidates_identically(cli, kl_eval, monkeypatch):
    assert cli.run(*cli.prepare) is None
    assert cli.generation_batches == [2, 1]

    def unexpected_generation(*_args, **_kwargs):
        pytest.fail("Candidate evaluation must not generate or load the dataset")

    monkeypatch.setattr(GPT2LMHeadModel, "generate", unexpected_generation)
    monkeypatch.setattr(kl_eval, "_blended_prompts", unexpected_generation)
    # Editing the recipe file while the model loads must not change what is calibrated.
    original_recipe = cli.recipe.read_text()
    cli.during_load.append(lambda: cli.recipe.write_text("not a recipe: ["))
    first = cli.run(*cli.evaluate)
    cli.during_load.clear()
    cli.recipe.write_text(original_recipe)
    second = cli.run(*cli.evaluate)
    assert cli.loads == ["eager"] * 3  # Evaluation inherits the reference's attention backend.
    assert math.isfinite(first) and first > 0
    assert first == pytest.approx(second, abs=1e-7)


def test_cli_refuses_unusable_checkpoints_recipes_and_launches(
    cli, kl_eval, tmp_path, monkeypatch
):
    config_path = cli.checkpoint / "config.json"
    original_config = config_path.read_text()
    config_path.write_text(
        json.dumps({**json.loads(original_config), "quantization_config": {"format": "packed"}})
    )
    with pytest.raises(ValueError, match="unquantized BF16"):
        cli.run(*cli.prepare)
    assert not cli.loads
    config_path.write_text(original_config)
    # A checkpoint restored with ModelOpt state, which hf_ptq would not calibrate.
    restored = tmp_path / "restored" / cli.checkpoint.name  # Same snapshot commit.
    model = _tiny_model().to(torch.bfloat16)
    mtq.quantize(model, {"quant_cfg": [{"quantizer_name": "*", "enable": False}]})
    model.save_pretrained(restored)
    assert (restored / "modelopt_state.pth").is_file()

    def use_checkpoint(path):
        monkeypatch.setattr(
            cli.hf_ptq, "ensure_local_checkpoint", lambda _path: ("org/model", path)
        )

    use_checkpoint(restored)
    with pytest.raises(ValueError, match="already carries ModelOpt quantizers"):
        cli.run(*cli.prepare)
    assert not cli.reference.exists()
    use_checkpoint(cli.checkpoint)
    cli.run(*cli.prepare)
    use_checkpoint(restored)
    with pytest.raises(ValueError, match="already carries ModelOpt quantizers"):
        cli.run(*cli.evaluate)
    assert len(cli.loads) == 3
    # The remaining refusals happen before any model loads.
    use_checkpoint(tmp_path / "newer-revision")
    with pytest.raises(ValueError, match="Checkpoint differs"):
        cli.run(*cli.evaluate)
    use_checkpoint(cli.checkpoint)
    layerwise = json.loads(json.dumps(cli.hf_ptq.mtq.INT8_DEFAULT_CFG))
    layerwise["algorithm"] = {"method": "max", "layerwise": {"enable": True, "checkpoint_dir": "x"}}
    resume = tmp_path / "resume.yaml"
    resume.write_text(
        "# modelopt-schema: modelopt.recipe.config.ModelOptPTQRecipe\n"
        + json.dumps({"metadata": {}, "quantize": layerwise})
    )
    with pytest.raises(ValueError, match="checkpoint_dir"):
        cli.run(*cli.evaluate_without_recipe, "--recipe", str(resume))
    last = sorted(cli.reference.glob("*.safetensors"), key=lambda p: int(p.stem))[-1]
    original_shard = last.read_bytes()
    for corrupt in (original_shard[:-8], original_shard + bytes(kl_eval.SHARD_HEADER_BYTES)):
        last.write_bytes(corrupt)
        with pytest.raises(ValueError, match="Invalid reference"):
            cli.run(*cli.evaluate)
    last.write_bytes(original_shard)
    auto_quantize = "general/auto_quantize/nvfp4_fp8_at_5p4bits"
    with pytest.raises(ValueError, match="fixed PTQ recipe"):
        cli.run(*cli.evaluate_without_recipe, "--recipe", auto_quantize)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="one process"):
        cli.run(*cli.evaluate)
    assert len(cli.loads) == 3
