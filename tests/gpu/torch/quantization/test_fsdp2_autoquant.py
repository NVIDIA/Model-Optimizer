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

import importlib
import math
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


def _run_gradient_autoquant(rank, size, checkpoint_dir, search_dir):
    device = torch.device(f"cuda:{rank}")
    model = parallel_load_and_prepare_fsdp2(checkpoint_dir, device, rank, size)
    torch.manual_seed(rank)
    batches = [torch.randint(0, 128, (1, 16), device=device) for _ in range(2)]
    _, state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG"],
        data_loader=batches,
        forward_step=lambda m, x: m(input_ids=x, labels=x, use_cache=False),
        loss_func=lambda output, _: output.loss,
        num_calib_steps=2,
        num_score_steps=2,
        method="gradient",
        checkpoint=search_dir,
    )
    scores = [v for stat in state["candidate_stats"].values() for v in stat["raw_scores"]]
    assert all(math.isfinite(value) and value >= 0 for value in scores)
    assert any(value > 0 for value in scores)
    assert state["best"]["is_satisfied"]
    assert state["best"]["constraints"]["effective_bits"] <= 8.0 + 1e-6
    results = [None] * size
    dist.all_gather_object(
        results, (scores, {k: str(v) for k, v in state["best"]["recipe"].items()})
    )
    assert all(result == results[0] for result in results)


def test_fsdp2_gradient_autoquant(dist_workers, tmp_path):
    checkpoint = create_tiny_llama_dir(tmp_path, vocab_size=128, num_hidden_layers=2)
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    dist_workers.run(
        partial(_run_gradient_autoquant, checkpoint_dir=str(checkpoint), search_dir=str(search_dir))
    )


def _run_kv_autoquant_example(rank, size, checkpoint_dir, search_dir, composed):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
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
        calib_size=[2 * size],
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
            stage.score_size = 2 * size
    kv_stage = recipe.kv_auto_quantize if composed else recipe.auto_quantize
    kv_stage.constraints.effective_bits = 6.25
    torch.manual_seed(rank)
    batches = []
    for _ in range(2):
        tokens = torch.randint(0, 128, (1, 16), device=device)
        batches.append(
            {"input_ids": tokens, "labels": tokens, "attention_mask": torch.ones_like(tokens)}
        )

    model, language_model, model_type, calibration_only, *_ = hf_ptq.load_model(args)
    model.config.use_cache = False
    hf_ptq._run_auto_quantize_recipe(
        args, recipe, model, language_model, model_type, calibration_only, batches, False
    )
    state = torch.load(search_dir / "kv" / f"rank{rank}.pth", weights_only=True)
    assert state["num_scored_tokens"] == 32 * size
    assert state["best"]["is_satisfied"]
    assert state["best"]["constraints"]["effective_bits"] <= 6.25
    results = [None] * size
    dist.all_gather_object(results, (state["layers"], state["best"]))
    assert all(result == results[0] for result in results)
    if composed:
        assert (search_dir / "weights" / f"rank{rank}.pth").is_file()

    del language_model, model
    model, language_model, model_type, calibration_only, *_ = hf_ptq.load_model(args)
    model.config.use_cache = False
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model, "forward", lambda *_a, **_kw: pytest.fail("Resume ran a forward"))
        hf_ptq._run_auto_quantize_recipe(
            args, recipe, model, language_model, model_type, calibration_only, batches, False
        )
    resumed = torch.load(search_dir / "kv" / f"rank{rank}.pth", weights_only=True)
    assert resumed["layers"] == state["layers"]
    assert resumed["best"] == state["best"]


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
