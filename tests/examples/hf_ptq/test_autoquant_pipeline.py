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

"""Multi-process HF CLI correctness; the tiny fixture is not a performance study."""

import importlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
import yaml
from _test_utils.torch.transformers_models import (
    create_tiny_qwen3_5_moe_vl_dir,
    create_tiny_qwen3_moe_dir,
    get_tiny_tokenizer,
)
from safetensors.torch import load_file

from modelopt.recipe import ModelOptAutoQuantizeRecipe, load_recipe
from modelopt.torch.utils import safe_load


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Two GPUs required for independent workers"
)
@pytest.mark.timeout(1200)
@pytest.mark.usefixtures("exclusive_gpu")
@pytest.mark.parametrize(
    ("hybrid", "layerwise"),
    [(False, False), (True, False), (True, True)],
    ids=["qwen3", "qwen3_5", "qwen3_5_layerwise"],
)
def test_format_workers_match_serial_hf_cli(monkeypatch, tmp_path, hybrid, layerwise):
    repo = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(repo / "tools/launcher"))
    generate = importlib.import_module("autoquant").generate_auto_quantize_pipeline
    if hybrid:
        tokenizer = get_tiny_tokenizer()
        vision_tokens = {
            "image_token": "<|image_pad|>",
            "video_token": "<|video_pad|>",
            "vision_bos_token": "<|vision_start|>",
            "vision_eos_token": "<|vision_end|>",
        }
        tokenizer.add_special_tokens({"additional_special_tokens": list(vision_tokens.values())})
        tokenizer._set_model_specific_special_tokens(vision_tokens)
        source = create_tiny_qwen3_5_moe_vl_dir(
            tmp_path,
            tokenizer=tokenizer,
            hidden_size=256,
            moe_intermediate_size=256,
            shared_expert_intermediate_size=256,
        )
        tokenizer.save_pretrained(source)
    else:
        source = create_tiny_qwen3_moe_dir(
            tmp_path,
            with_tokenizer=True,
            hidden_size=256,
            moe_intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=64,
        )
    payload = load_recipe(
        "models/zai-org/GLM-5.3-Flash-BF16/auto_quantize/iq2_gptq_nvfp4_fp8_at_3p0bits"
    ).model_dump()
    search = payload["auto_quantize"]
    search["score_size"] = 4
    search["constraints"]["effective_bits"] = 4.0
    if not layerwise:
        for config in search["module_search_spaces"][0]["candidate_formats"][:3]:
            config["algorithm"] = {"method": "imatrix", "layerwise": {"enable": False}}
    if layerwise:
        formats = search["module_search_spaces"][0]["candidate_formats"]
        for index, config in enumerate(formats):
            if index in (0, 2):
                continue  # Preserve the recipe's complete GPTQ chain and group-size settings.
            config["algorithm"] = {
                "method": "imatrix" if index == 1 else "max",
                "layerwise": {"enable": True},
            }
    recipe = ModelOptAutoQuantizeRecipe(**payload)
    payload = recipe.model_dump()
    payload["metadata"]["recipe_type"] = recipe.recipe_type.value
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"text": f"Hello world {i}. " * 32}) for i in range(8)))
    allocated_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))
    resources = {
        "max_concurrent_formats": 2,
        "max_total_gpus": 2 * allocated_gpus,
        "allocated_gpus_per_worker": allocated_gpus,
        "gpu_bytes": 8 * 1024**3,
        "host_bytes": 32 * 1024**3,
        "input_bytes": 1024**2,
        "candidate_bytes": 16 * 1024**2,
        "checkpoint_bytes": 128 * 1024**2,
        "metadata_bytes": 1024**2,
        "export_bytes": 128 * 1024**2,
        "artifact_storage_bytes": 1024**3,
        "retries": 0,
    }
    if layerwise:
        resources.update(activation_bytes=4 * 1024**2, final_checkpoint_bytes=64 * 1024**2)
    pipeline = generate(
        str(recipe_path),
        str(source),
        str(data),
        str(tmp_path / "work"),
        str(tmp_path / "parallel"),
        {},
        resources,
        calib_size=8,
        calib_seq=16,
    )["pipeline"]
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")[:2]

    def run(label, gpu, flags, spec=None, expected=0):
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": devices[gpu],
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
        if spec is not None:
            env["MODELOPT_AUTOQUANT_PIPELINE"] = spec
        else:
            env.pop("MODELOPT_AUTOQUANT_PIPELINE", None)
        command = [sys.executable, str(repo / "examples/hf_ptq/hf_ptq.py"), *flags]
        if layerwise:
            prelude = """
import json,os,runpy,sys,weakref
from pathlib import Path
refs=[]
def profile(frame,event,arg):
    if frame.f_code.co_name == 'load_model' and frame.f_code.co_filename.endswith('/hf_ptq.py'):
        if event == 'call' and refs:
            assert refs[-1]() is None, 'Previous source model survived into the next load'
        if event == 'return' and arg is not None:
            refs.append(weakref.ref(arg[0]))
            Path(os.environ['SOURCE_LIFETIME_RECEIPT']).write_text(json.dumps({'loads':len(refs)}))
sys.setprofile(profile)
if os.environ.get('INTERRUPT_FINAL') == '1':
    from modelopt.torch.quantization._layerwise_checkpoint import _AssignmentCheckpoint
    original_save = _AssignmentCheckpoint.save
    def save(self,index,*args,**kwargs):
        result=original_save(self,index,*args,**kwargs)
        if index == 0: os._exit(73)
        return result
    _AssignmentCheckpoint.save=save
sys.path.insert(0,str(Path(sys.argv[1]).parent))
script=sys.argv.pop(1)
runpy.run_path(script,run_name='__main__')
"""
            env["SOURCE_LIFETIME_RECEIPT"] = str(tmp_path / f"{label}-lifetime.json")
            env["INTERRUPT_FINAL"] = "1" if label == "coordinator-interrupted" else "0"
            command = [
                sys.executable,
                "-c",
                prelude,
                str(repo / "examples/hf_ptq/hf_ptq.py"),
                *flags,
            ]
        log = tmp_path / f"{label}.log"
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=repo,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=300,
            )
        assert result.returncode == expected, log.read_text()[-8000:]

    def task(index, gpu, label, expected=0):
        run(
            label,
            gpu,
            ["--auto_quantize_pipeline"],
            pipeline["tasks"][index]["environment"][0]["MODELOPT_AUTOQUANT_PIPELINE"],
            expected,
        )

    def worker(gpu):
        for index in range(gpu, len(pipeline["tasks"]) - 1, 2):
            task(index, gpu, f"format-{index}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu) for gpu in range(2)]
        for future in futures:
            future.result()
    if layerwise:
        task(-1, 0, "coordinator-interrupted", expected=73)
        (checkpoint,) = (tmp_path / "work").glob("*/final")
        assert (checkpoint / "layer_0000").is_dir()
    task(-1, 0, "coordinator")
    if layerwise:
        assert not checkpoint.exists()
        assert (
            json.loads((tmp_path / "coordinator-interrupted-lifetime.json").read_text())["loads"]
            == 2
        )
        assert json.loads((tmp_path / "coordinator-lifetime.json").read_text())["loads"] == 1
        control = generate(
            str(recipe_path),
            str(source),
            str(data),
            str(tmp_path / "serial-work"),
            str(tmp_path / "serial"),
            {},
            {**resources, "max_concurrent_formats": 1},
            calib_size=8,
            calib_seq=16,
        )["pipeline"]
        for index, entry in enumerate(control["tasks"]):
            run(
                f"serial-{index}",
                0,
                ["--auto_quantize_pipeline"],
                entry["environment"][0]["MODELOPT_AUTOQUANT_PIPELINE"],
            )
        assert (
            json.loads(
                (tmp_path / f"serial-{len(control['tasks']) - 1}-lifetime.json").read_text()
            )["loads"]
            == 2
        )
    else:
        run(
            "serial",
            0,
            [
                "--pyt_ckpt_path",
                str(source),
                "--recipe",
                str(recipe_path),
                "--dataset",
                str(data),
                "--calib_size",
                "8",
                "--calib_seq",
                "16",
                "--batch_size",
                "1",
                "--skip_generate",
                "--no-verbose",
                "--kv_cache_qformat",
                "none",
                "--export_path",
                str(tmp_path / "serial"),
                "--auto_quantize_checkpoint",
                str(tmp_path / "serial-search.pth"),
            ],
        )
    (search,) = (tmp_path / "work").glob("*/search")
    serial_path = (
        next((tmp_path / "serial-work").glob("*/search/rank0.pth"))
        if layerwise
        else tmp_path / "serial-search.pth"
    )
    parallel, serial = safe_load(search / "rank0.pth"), safe_load(serial_path)

    def states(checkpoint):
        return {
            r.checkpoint_signature: value["state_dict"]
            for r, value in checkpoint["quantizer_states"].items()
        }

    torch.testing.assert_close(states(parallel), states(serial), rtol=0, atol=0)
    packed_formats = set()
    prefix = "model.language_model" if hybrid else "model"
    expected_experts = {
        f"{prefix}.layers.{layer}.mlp.experts.{projection}_weight_quantizers.{expert}"
        for layer in range(4 if hybrid else 1)
        for projection in ("gate_up_proj", "down_proj")
        for expert in range(4)
    }
    for entry in parallel["quantizer_states"].values():
        packed_modules = set()
        for name, metadata in entry["metadata"].items():
            if metadata.get("backend") != "ggml" or metadata.get("_disabled", False):
                continue
            fmt = metadata["_num_bits"]
            packed = entry["state_dict"][name][f"_ggml_pinned_{fmt}"]
            assert packed.dtype == torch.uint8 and packed.numel() > 0
            packed_formats.add(fmt)
            packed_modules.add(name)
        if packed_modules:
            assert packed_modules == expected_experts
    assert packed_formats == {"iq2_xxs", "iq2_xs", "iq2_s"}

    def tensors(directory):
        return {
            key: tensor
            for file in directory.glob("*.safetensors")
            for key, tensor in load_file(file).items()
        }

    torch.testing.assert_close(
        tensors(tmp_path / "parallel"), tensors(tmp_path / "serial"), rtol=0, atol=0
    )
    before = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "parallel").iterdir()}
    task(0, 0, "worker-resume")
    task(-1, 0, "coordinator-resume")
    assert before == {p.name: p.stat().st_mtime_ns for p in (tmp_path / "parallel").iterdir()}
