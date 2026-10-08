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

"""Exercise weight and KV AutoQuant with real FSDP2 and the Hugging Face example."""

import argparse
import gc
import importlib
import json
import math
import os
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import DistributedWorkerPool
from _test_utils.torch.transformers_models import create_tiny_llama_dir, get_tiny_llama

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.kv_cache_auto_quant import AutoQuantizeKVSearcher
from modelopt.torch.utils import distributed as dist_utils
from modelopt.torch.utils.plugins.model_load_utils import parallel_load_and_prepare_fsdp2

pytestmark = [pytest.mark.usefixtures("need_2_gpus"), pytest.mark.timeout(300)]


def test_kl_memory_probe_reserves_logit_working_memory(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples" / "hf_ptq"))
    autoquant_utils = importlib.import_module("autoquant_utils")
    recipe = load_recipe("general/auto_quantize/kv_fp8_nvfp4_cast_kl_div_at_5p4bits")
    model = get_tiny_llama(vocab_size=32768).to(device="cuda", dtype=torch.bfloat16).eval()
    model.config.use_cache = False
    tokens = torch.ones((2, 32), dtype=torch.long, device="cuda")
    probe = autoquant_utils._get_autoquant_memory_probe(recipe)
    peaks = []
    with torch.no_grad():
        model(tokens)
        for workload in (lambda: model(tokens), lambda: probe(model, tokens)):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            workload()
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() - baseline)
    assert peaks[1] > peaks[0] + tokens.numel() * model.config.vocab_size * 4


def _calibration_tokens(rank, size, device, global_samples):
    assert global_samples > 0 and global_samples % size == 0
    tokens = torch.randint(
        0, 128, (global_samples, 16), generator=torch.Generator().manual_seed(1234)
    )
    return [sample.unsqueeze(0).to(device) for sample in tokens[rank::size]]


def _quantized_snapshot(model, device):
    quantizer_state = {
        name: {key: value.detach().cpu().clone() for key, value in buffers.items()}
        for name, buffers in mtq.utils.get_quantizer_state_dict(model).items()
    }
    assert any(
        value.numel() > 0 for buffers in quantizer_state.values() for value in buffers.values()
    ), "Expected calibrated quantizer buffers"
    model.eval()
    probe = torch.arange(16, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=probe, use_cache=False).logits.detach().cpu().clone()
    assert torch.isfinite(logits).all()
    return quantizer_state, logits


def _run_gradient_autoquant(
    rank, size, checkpoint_dir, search_dir, local_rank=None, global_samples=None
):
    local_rank = rank if local_rank is None else local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    global_samples = 2 * size if global_samples is None else global_samples
    batches = _calibration_tokens(rank, size, device, global_samples)
    original_result = None
    for resume in (False, True):
        model = parallel_load_and_prepare_fsdp2(checkpoint_dir, device, rank, size)
        forward_calls = 0

        def forward_step(model, tokens):
            nonlocal forward_calls
            forward_calls += 1
            return model(input_ids=tokens, labels=tokens, use_cache=False)

        model, state = mtq.auto_quantize(
            model,
            constraints={"effective_bits": 8.0},
            quantization_formats=["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG"],
            data_loader=batches,
            forward_step=forward_step,
            loss_func=lambda output, _: output.loss,
            num_calib_steps=len(batches),
            num_score_steps=len(batches),
            method="gradient",
            checkpoint=search_dir,
        )
        scores = [v for stat in state["candidate_stats"].values() for v in stat["raw_scores"]]
        assert all(math.isfinite(value) and value >= 0 for value in scores)
        assert any(value > 0 for value in scores)
        assert state["best"]["is_satisfied"]
        assert state["best"]["constraints"]["effective_bits"] <= 8.0 + 1e-6
        named_scores = {
            name: list(stat["raw_scores"]) for name, stat in state["candidate_stats"].items()
        }
        candidate_names = {
            name: [str(candidate) for candidate in stat["formats"]]
            for name, stat in state["candidate_stats"].items()
        }
        recipe = {name: str(candidate) for name, candidate in state["best"]["recipe"].items()}
        results = [None] * size
        dist.all_gather_object(results, (named_scores, candidate_names, recipe))
        assert all(item == results[0] for item in results)
        if resume:
            assert forward_calls == 0, "Resume must restore calibration and scores"
        else:
            assert forward_calls > 0

        quantizer_state, logits = _quantized_snapshot(model, device)
        result = {
            "named_scores": named_scores,
            "candidate_names": candidate_names,
            "recipe": recipe,
            "quantizer_state": quantizer_state,
            "logits": logits,
            "effective_bits": state["best"]["constraints"]["effective_bits"],
            "score_reduction": "sum",
            "global_samples": global_samples,
            "sample_ids": list(range(rank, global_samples, size)),
        }
        if resume:
            for key in ("named_scores", "candidate_names", "recipe", "effective_bits"):
                assert result[key] == original_result[key]
            torch.testing.assert_close(
                quantizer_state, original_result["quantizer_state"], rtol=0, atol=0
            )
            torch.testing.assert_close(logits, original_result["logits"], rtol=0, atol=0)
        else:
            original_result = result
        del model, state
        gc.collect()
        dist.barrier()
    return result


def test_fsdp2_gradient_autoquant(dist_workers, tmp_path):
    checkpoint = create_tiny_llama_dir(tmp_path, vocab_size=128, num_hidden_layers=2)
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    dist_workers.run(
        partial(_run_gradient_autoquant, checkpoint_dir=str(checkpoint), search_dir=str(search_dir))
    )


def _run_kv_autoquant_example(
    rank, size, checkpoint_dir, search_dir, composed, local_rank=None, global_samples=None
):
    local_rank = rank if local_rank is None else local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    global_samples = 2 * size if global_samples is None else global_samples
    assert AutoQuantizeKVSearcher()._collective_device == device
    recipe_name = "general/auto_quantize/" + (
        "nvfp4_fp8_gradient_then_kv_fp8_nvfp4_cast_kl_div_at_5p4bits"
        if composed
        else "kv_fp8_nvfp4_cast_kl_div_at_5p4bits"
    )
    args = SimpleNamespace(
        pyt_ckpt_path=str(checkpoint_dir),
        recipe=recipe_name,
        use_fsdp2=True,
        dist_state=SimpleNamespace(rank=rank, world_size=size, device=device),
        trust_remote_code=False,
        cpu_offload=False,
        attn_implementation="eager",
        calib_with_images=False,
        specdec_offline_dataset=None,
        low_memory_mode=False,
        dataset=["local-test-data"],
        calib_size=[global_samples],
        batch_size=1,
        inference_pipeline_parallel=1,
        kv_cache_qformat="none",
        auto_quantize_checkpoint=str(search_dir / "weights"),
        kv_auto_quantize_checkpoint=str(search_dir / "kv"),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples" / "hf_ptq"))
        hf_ptq = importlib.import_module("hf_ptq")
    recipe = load_recipe(recipe_name)
    for stage in (recipe.auto_quantize, recipe.kv_auto_quantize):
        if stage is not None:
            stage.score_size = global_samples
    kv_stage = recipe.kv_auto_quantize if composed else recipe.auto_quantize
    kv_stage.constraints.effective_bits = 6.25
    batches = [
        {"input_ids": tokens, "labels": tokens, "attention_mask": torch.ones_like(tokens)}
        for tokens in _calibration_tokens(rank, size, device, global_samples)
    ]

    model, language_model, model_type, calibration_only, *_ = hf_ptq.load_model(args)
    model.config.use_cache = False
    hf_ptq._run_auto_quantize_recipe(
        args, recipe, model, language_model, model_type, calibration_only, batches, False
    )
    state = torch.load(search_dir / "kv" / f"rank{rank}.pth", weights_only=True, map_location="cpu")
    assert state["num_scored_tokens"] == 16 * global_samples
    assert state["num_calib_steps"] == len(batches)
    assert state["num_score_steps"] == len(batches)
    assert state["best"]["is_satisfied"]
    assert state["best"]["constraints"]["effective_bits"] <= 6.25
    results = [None] * size
    dist.all_gather_object(results, (state["layers"], state["best"]))
    assert all(result == results[0] for result in results)
    if composed:
        assert (search_dir / "weights" / f"rank{rank}.pth").is_file()
    quantizer_state, logits = _quantized_snapshot(model, device)

    del language_model, model
    model, language_model, model_type, calibration_only, *_ = hf_ptq.load_model(args)
    model.config.use_cache = False
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model, "forward", lambda *_a, **_kw: pytest.fail("Resume ran a forward"))
        hf_ptq._run_auto_quantize_recipe(
            args, recipe, model, language_model, model_type, calibration_only, batches, False
        )
    resumed = torch.load(
        search_dir / "kv" / f"rank{rank}.pth", weights_only=True, map_location="cpu"
    )
    assert resumed["layers"] == state["layers"]
    assert resumed["best"] == state["best"]
    restored_quantizers, restored_logits = _quantized_snapshot(model, device)
    torch.testing.assert_close(restored_quantizers, quantizer_state, rtol=0, atol=0)
    torch.testing.assert_close(restored_logits, logits, rtol=0, atol=0)
    torch.testing.assert_close(resumed["quantizer_state"], state["quantizer_state"], rtol=0, atol=0)
    return {
        "named_scores": {
            name: list(layer["scores"].values()) for name, layer in state["layers"].items()
        },
        "candidate_names": {name: list(layer["scores"]) for name, layer in state["layers"].items()},
        "recipe": state["best"]["recipe"],
        "quantizer_state": quantizer_state,
        "logits": logits,
        "effective_bits": state["best"]["constraints"]["effective_bits"],
        "score_reduction": state["score_reduction"],
        "global_samples": global_samples,
        "sample_ids": list(range(rank, global_samples, size)),
        "source_state": state,
    }


@pytest.fixture(scope="module", params=["cpu:gloo,cuda:nccl", "cuda:nccl"])
def kv_dist_workers(request):
    pool = DistributedWorkerPool(world_size=2, backend=request.param)
    yield pool
    pool.shutdown()


@pytest.mark.parametrize("composed", [False, True], ids=["kv", "weight_then_kv"])
def test_fsdp2_kv_autoquant_example(kv_dist_workers, tmp_path, composed):
    checkpoint = create_tiny_llama_dir(
        tmp_path,
        with_tokenizer=True,
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
    )
    kv_dist_workers.run(
        partial(
            _run_kv_autoquant_example,
            checkpoint_dir=checkpoint,
            search_dir=tmp_path / "search",
            composed=composed,
        )
    )


def _main():
    parser = argparse.ArgumentParser(prog="fsdp2-autoquant", description=__doc__)
    parser.add_argument("--mode", choices=("weight", "kv", "weight_then_kv"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("cpu:gloo,cuda:nccl", "cuda:nccl"), default="cpu:gloo,cuda:nccl"
    )
    parser.add_argument("--global-samples", type=int, default=16)
    parser.add_argument(
        "--private-log-file",
        type=Path,
        help="Exclusive log prefix (adds .rank<N>.log); raw diagnostics may include local paths.",
    )
    args = parser.parse_args()
    rank, size, local_rank = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if not args.checkpoint_dir.is_dir():
        parser.error("--checkpoint-dir must be an offline local checkpoint directory")
    if args.global_samples <= 0 or args.global_samples % size:
        parser.error("--global-samples must be positive and divisible by WORLD_SIZE")
    receipt = {
        "rank": rank,
        "world_size": size,
        "mode": args.mode,
        "backend": args.backend,
        "global_samples": args.global_samples,
    }
    with ExitStack() as stack:
        quiet = stack.enter_context(open(os.devnull, "w"))
        try:
            if args.private_log_file is not None:
                quiet = stack.enter_context(
                    open(
                        f"{args.private_log_file}.rank{rank}.log",
                        "x",
                        opener=lambda path, flags: os.open(path, flags, 0o600),
                    )
                )
            with redirect_stdout(quiet), redirect_stderr(quiet):
                device = torch.device("cuda", local_rank)
                torch.cuda.set_device(device)
                dist.init_process_group(
                    args.backend, timeout=timedelta(seconds=180), device_id=device
                )
                search_dir = args.search_dir / f"rank{rank}"
                search_dir.mkdir(parents=True, exist_ok=False)
                kwargs = {"local_rank": local_rank, "global_samples": args.global_samples}
                if args.mode == "weight":
                    result = _run_gradient_autoquant(
                        rank, size, str(args.checkpoint_dir), str(search_dir), **kwargs
                    )
                else:
                    result = _run_kv_autoquant_example(
                        rank,
                        size,
                        args.checkpoint_dir,
                        search_dir,
                        args.mode == "weight_then_kv",
                        **kwargs,
                    )
                dist_utils.cleanup()
        except BaseException as error:
            print(
                json.dumps({**receipt, "passed": False, "error_type": type(error).__name__}),
                flush=True,
            )
            with redirect_stdout(quiet), redirect_stderr(quiet):
                dist_utils.abort()
    print(
        json.dumps(
            {
                **receipt,
                "passed": True,
                "effective_bits": result["effective_bits"],
                "score_groups": len(result["named_scores"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    _main()
