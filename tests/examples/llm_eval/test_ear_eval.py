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

"""Exercise frozen EAR scoring against ordinary, unpadded causal-model forwards."""

import copy
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)


@pytest.fixture
def ear_eval(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples" / "llm_eval"))
    return importlib.import_module("ear_eval")


def _models(context=32, layers=1):
    torch.manual_seed(7)
    reference = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=16,
            n_positions=context,
            n_embd=8,
            n_layer=layers,
            n_head=1,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            bos_token_id=1,
            eos_token_id=0,
            pad_token_id=0,
        )
    ).eval()
    candidate = copy.deepcopy(reference)
    with torch.no_grad():
        candidate.lm_head.weight[3, 0].add_(1)
    return reference, candidate


def _ordinary_scores(reference, candidate, prompt, continuation):
    inputs = torch.tensor([prompt + continuation[:-1]])
    with torch.inference_mode():
        p = reference(inputs, use_cache=False).logits[0, len(prompt) - 1 :].double().softmax(-1)
        q = candidate(inputs, use_cache=False).logits[0, len(prompt) - 1 :].double().softmax(-1)
        ids = p.topk(2, dim=-1).indices
        return {
            "full": torch.minimum(p, q).sum(-1).tolist(),
            "2": torch.minimum(p.gather(-1, ids), q.gather(-1, ids)).sum(-1).tolist(),
            "reference_mass_2": p.gather(-1, ids).sum(-1).tolist(),
        }


def test_frozen_scores_match_ordinary_forwards_through_first_eos(ear_eval):
    reference, candidate = _models()
    rows = [
        {"id": "long", "prompt_ids": [4, 5, 6], "generated_ids": [7, 8, 0, 10, 0]},
        {"id": "eos", "prompt_ids": [4], "generated_ids": [0, 0, 0]},
        {"id": "open", "prompt_ids": [5, 6], "generated_ids": [9, 10, 11, 12]},
    ]
    original = copy.deepcopy(rows)
    scored = ear_eval.score_frozen(
        reference,
        candidate,
        rows,
        eos_token_ids=(token for token in [0]),
        top_ks=(k for k in [2]),
        batch_size=3,
        chunk_size=2,
    )
    assert rows == original
    assert [row["id"] for row in scored] == ["long", "eos", "open"]
    continuations = [[7, 8, 0], [0], [9, 10, 11, 12]]
    for source, actual, continuation in zip(rows, scored, continuations):
        assert actual["generated_ids"] == continuation
        expected = _ordinary_scores(reference, candidate, source["prompt_ids"], continuation)
        assert set(actual["values"]) == set(expected)
        for name, values in expected.items():
            assert actual["values"][name] == pytest.approx(values, abs=1e-7)
    assert any(value < 0.999 for row in scored for value in row["values"]["full"])
    qualification = ear_eval.qualify_batch(
        reference, candidate, rows, [0], [1, 2, 3], top_ks=[2], chunk_size=2, atol=1e-7
    )
    assert [item["batch_size"] for item in qualification["batches"]] == [1, 2, 3]
    assert all(item["passed"] for item in qualification["batches"])
    assert all(item["positions_per_second"] > 0 for item in qualification["batches"])


def test_full_length_continuation_and_padding_are_not_clipped(ear_eval):
    # A real zero-block decoder keeps the long-position alignment case inexpensive on CPU.
    reference, candidate = _models(context=8196, layers=0)
    rows = [
        {"id": "horizon", "prompt_ids": [4, 5], "generated_ids": [6] * 8192 + [7]},
        {"id": "short", "prompt_ids": [8], "generated_ids": [9, 0, 10]},
    ]
    scored = ear_eval.score_frozen(
        reference,
        candidate,
        rows,
        eos_token_ids=[0],
        top_ks=[2],
        max_tokens=8192,
        batch_size=2,
        chunk_size=257,
    )
    assert [len(row["generated_ids"]) for row in scored] == [8192, 2]
    for source, actual in zip(rows, scored):
        expected = _ordinary_scores(
            reference, candidate, source["prompt_ids"], actual["generated_ids"]
        )
        for name, values in expected.items():
            assert len(actual["values"][name]) == len(actual["generated_ids"])
            assert actual["values"][name] == pytest.approx(values, abs=1e-7)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("final_logit_softcapping", 1),
        ("logit_scale", 0),
        ("atol", float("nan")),
        ("atol", float("inf")),
    ],
)
def test_unsupported_transforms_and_invalid_tolerances_fail(ear_eval, setting, value):
    reference, candidate = _models()
    rows = [{"id": "fixture", "prompt_ids": [4], "generated_ids": [5, 0]}]
    with pytest.raises(ValueError):
        if setting == "atol":
            ear_eval.qualify_batch(reference, candidate, rows, [0], [1], atol=value)
        else:
            setattr(reference.config, setting, value)
            ear_eval.score_frozen(reference, candidate, rows, [0])


@pytest.mark.parametrize("corruption", ["unexpected_quantizer", "missing_weight"])
def test_checkpoint_loader_refuses_ignored_or_missing_weights(ear_eval, tmp_path, corruption):
    original, _ = _models()
    original.save_pretrained(tmp_path, safe_serialization=True)
    loaded, info = ear_eval._load_checkpoint(AutoModelForCausalLM, tmp_path, local_files_only=True)
    assert isinstance(info, dict)
    assert json.loads(json.dumps(info, allow_nan=False)) == info
    assert all(
        not info.get(key)
        for key in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
            "conversion_errors",
        )
    )
    assert torch.equal(
        loaded.transformer.h[0].mlp.c_fc.weight, original.transformer.h[0].mlp.c_fc.weight
    )
    checkpoint = tmp_path / "model.safetensors"
    with safe_open(checkpoint, framework="pt") as stream:
        metadata = stream.metadata()
    state = load_file(checkpoint)
    if corruption == "unexpected_quantizer":
        state["unrestored_quantizer._amax"] = torch.tensor(1.0)
    else:
        del state["transformer.h.0.mlp.c_fc.weight"]
    save_file(state, checkpoint, metadata=metadata)
    with pytest.raises(ValueError):
        ear_eval._load_checkpoint(AutoModelForCausalLM, tmp_path, local_files_only=True)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native checkpoint integration requires CUDA."
)
def test_cli_restores_native_quantization_in_a_fresh_process(ear_eval, tmp_path):
    # ModelOpt calibration imports require the optional CUDA integration dependencies.
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq

    torch.manual_seed(11)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attn_implementation="eager",
    )
    reference = LlamaForCausalLM(config).to(device="cuda:0", dtype=torch.bfloat16).eval()
    candidate = copy.deepcopy(reference)
    mto.enable_huggingface_checkpointing()
    calibration = torch.tensor(
        [[1, 4, 5, 6, 7, 8, 9, 2], [1, 10, 11, 12, 13, 14, 15, 2]], device="cuda:0"
    )
    with torch.no_grad():
        mtq.quantize(
            candidate,
            copy.deepcopy(mtq.INT8_DEFAULT_CFG),
            lambda model: model(calibration, use_cache=False),
        )
    paths = [tmp_path / "reference", tmp_path / "candidate"]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({f"token{i}": i for i in range(32)}, unk_token="token3")
        ),
        eos_token="token2",
        pad_token="token0",
        bos_token="token1",
        unk_token="token3",
    )
    for model, path in zip((reference, candidate), paths, strict=True):
        model.save_pretrained(path, safe_serialization=True)
        tokenizer.save_pretrained(path)
    assert (paths[1] / "modelopt_state.pth").is_file()
    rows = [
        {"id": "first", "prompt_ids": [1, 4, 5], "generated_ids": [7, 8, 2, 0, 0]},
        {"id": "second", "prompt_ids": [1, 6], "generated_ids": [9, 2, 0]},
    ]
    expected = ear_eval.score_frozen(reference, candidate, rows, [2], pad_token_id=0, top_ks=[10])
    assert any(value < 1 - 1e-8 for row in expected for value in row["values"]["full"])
    trajectories, output = tmp_path / "trajectories.json", tmp_path / "scores.json"
    trajectories.write_text(json.dumps(rows))
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": os.pathsep.join(
            [str(Path(ear_eval.__file__).resolve().parents[2]), os.environ.get("PYTHONPATH", "")]
        ),
    }
    # A fresh interpreter cannot inherit the parent's globally enabled restoration hook.
    completed = subprocess.run(
        [
            sys.executable,
            ear_eval.__file__,
            "--reference",
            str(paths[0]),
            "--candidate",
            str(paths[1]),
            "--trajectories",
            str(trajectories),
            "--output",
            str(output),
            "--eos_token_ids",
            "2",
            "--top_ks",
            "10",
            "--device_map",
            "cuda:0",
            "--attn_implementation",
            "eager",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = json.loads(output.read_text())
    assert any(value < 1 - 1e-8 for row in actual["rows"] for value in row["values"]["full"])
    for observed, prepared in zip(actual["rows"], expected, strict=True):
        assert observed["id"] == prepared["id"]
        assert observed["generated_ids"] == prepared["generated_ids"]
        assert set(observed["values"]) == {"full", "10", "reference_mass_10"}
        for key in prepared["values"]:
            assert observed["values"][key] == pytest.approx(
                prepared["values"][key], abs=1e-7, rel=0
            )
    assert [len(row["generated_ids"]) for row in actual["rows"]] == [3, 2]
    assert all(
        not info.get(key)
        for info in actual["settings"]["checkpoint_loading_info"]
        for key in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
            "conversion_errors",
        )
    )
