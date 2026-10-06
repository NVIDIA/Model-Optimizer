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

"""Reduced-scale inputs for the real QAT README commands."""

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from _test_utils.torch.transformers_models import create_tiny_llama_dir, create_tiny_qwen3_dir
from datasets import Dataset
from safetensors import safe_open

__all__ = ["prepare_llm_qat_workspace", "verify_llm_qat_workspace"]


def prepare_llm_qat_workspace(repo: Path, tmp: Path, *, llama: bool = False) -> Path:
    """Copy scripts/configs, build local inputs and retain real NVFP4/FSDP2 execution."""
    root = tmp / "checkout"
    example = root / "examples" / "llm_qat"
    example.mkdir(parents=True)
    source = repo / "examples" / "llm_qat"
    for script in source.glob("*.py"):
        shutil.copy2(script, example / script.name)
    shutil.copytree(source / "configs", example / "configs")
    (root / "modelopt_recipes").symlink_to(repo / "modelopt_recipes", target_is_directory=True)
    model = (create_tiny_llama_dir if llama else create_tiny_qwen3_dir)(
        tmp / "model",
        with_tokenizer=True,
        hidden_size=512,
        intermediate_size=512,
        num_hidden_layers=2,
    )
    (example / "Qwen").mkdir()
    (example / "Qwen" / "Qwen3-8B").symlink_to(model, target_is_directory=True)
    (example / "meta-llama").mkdir()
    (example / "meta-llama/Llama-3.2-3B").symlink_to(model, target_is_directory=True)
    data = tmp / "data"
    Dataset.from_dict(
        {"text": ["The quick brown fox jumps over the lazy dog. " * 4] * 20}
    ).save_to_disk(str(data))
    blend = {
        "blend_size": 20,
        "splits": {"train": 0.8, "eval": 0.2},
        "sources": [
            {"hf_path": str(data), "split": "train", "ratio": 1, "apply_chat_template": False}
        ],
    }
    (example / "configs/dataset/blend.yaml").write_text(yaml.safe_dump(blend))
    for training_path in (example / "configs/train").glob("*_nvfp4.yaml"):
        training = yaml.safe_load(training_path.read_text())
        training.update(
            model_max_length=128,
            train_samples=16,
            eval_samples=4,
            max_steps=2,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            eval_steps=1,
            save_steps=1,
            logging_steps=1,
            num_proc=1,
            report_to="none",
        )
        training_path.write_text(yaml.safe_dump(training))
    devices = int(os.environ.get("MODELOPT_DOC_TEST_GPUS", "1"))
    if devices not in (1, 2):
        raise ValueError("MODELOPT_DOC_TEST_GPUS must be 1 or 2")
    for name in ("fsdp2", "ddp"):
        accelerate_path = example / f"configs/accelerate/{name}.yaml"
        accelerate = yaml.safe_load(accelerate_path.read_text())
        accelerate.update(num_processes=devices, main_process_port=0)
        accelerate_path.write_text(yaml.safe_dump(accelerate))
    print(
        f"QAT smoke inputs: 2 layers, hidden/intermediate=512, 20 local text samples; "
        f"PTQ length=8192, training length=128, max_steps=2, FSDP2 ranks={devices}; "
        "NVFP4 recipe, FlashAttention and Liger unchanged.",
        flush=True,
    )
    return root


def verify_llm_qat_workspace(root: Path, variant="qat"):
    """Require quantized state, finite training metrics, changed weights and NVFP4 export."""
    example = root / "examples/llm_qat"
    ptq = example / "qwen3-8b-quantized"
    trained = example / f"qwen3-8b-{variant}-nvfp4"
    exported = example / f"qwen3-8b-{variant}-deploy"
    for checkpoint in (ptq, trained):
        assert (checkpoint / "modelopt_state.pth").stat().st_size > 0
        assert (checkpoint / "model.safetensors").stat().st_size > 0
    state = json.loads((trained / "trainer_state.json").read_text())
    assert state["global_step"] == 2
    losses = [
        entry[key]
        for entry in state["log_history"]
        for key in ("loss", "eval_loss", "train_loss")
        if key in entry
    ]
    assert losses and all(math.isfinite(value) for value in losses), state
    assert any("train_loss" in entry for entry in state["log_history"]), state
    with (
        safe_open(ptq / "model.safetensors", framework="pt", device="cpu") as before,
        safe_open(trained / "model.safetensors", framework="pt", device="cpu") as after,
    ):
        before_keys, after_keys = set(before.keys()), set(after.keys())
        keys = [key for key in before_keys & after_keys if key.endswith("weight")]
        assert keys and any(
            not torch.equal(before.get_tensor(key), after.get_tensor(key)) for key in keys
        )
    config = json.loads((exported / "config.json").read_text())
    assert config["quantization_config"]["quant_algo"] == "NVFP4"
    assert (exported / "model.safetensors").stat().st_size > 0
    with safe_open(exported / "model.safetensors", framework="pt", device="cpu") as weights:
        exported_keys = weights.keys()
        assert any(weights.get_tensor(key).dtype == torch.uint8 for key in exported_keys)


def prepare_quantized_input(ctx, *, llama=False):
    """Produce a real prerequisite checkpoint for standalone training snippets."""
    root = prepare_llm_qat_workspace(ctx.repo, ctx.tmp, llama=llama)
    ctx.cwd = root / "examples/llm_qat"
    subprocess.run(
        [
            sys.executable,
            "quantize.py",
            "--model_name_or_path",
            "Qwen/Qwen3-8B",
            "--dataset_config",
            "configs/dataset/blend.yaml",
            "--recipe",
            "general/ptq/nvfp4_default-kv_fp8",
            "--output_dir",
            "qwen3-8b-quantized",
        ],
        cwd=ctx.cwd,
        env=ctx.env,
        check=True,
    )
    return root


def prepare_python_api(ctx, *, quantized):
    """Supply the model, data and Trainer inputs omitted by the API fragments."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        default_data_collator,
    )

    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    from modelopt.recipe import load_recipe
    from modelopt.torch.quantization.plugins.transformers_trainer import QATTrainer

    root = prepare_llm_qat_workspace(ctx.repo, ctx.tmp)
    ctx.cwd = root / "examples/llm_qat"
    mto.enable_huggingface_checkpointing()
    model = AutoModelForCausalLM.from_pretrained(
        ctx.cwd / "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(ctx.cwd / "Qwen/Qwen3-8B")
    tokens = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")

    def forward_loop(m):
        with torch.no_grad():
            m(**{key: value.cuda() for key, value in tokens.items()})

    if quantized:
        mtq.quantize(model, load_recipe("general/ptq/nvfp4_default-kv_fp8").quantize, forward_loop)
    record = {key: value[0].tolist() for key, value in tokens.items()}
    record["labels"] = record["input_ids"].copy()
    data_module = {
        "train_dataset": Dataset.from_list([record] * 4),
        "data_collator": default_data_collator,
    }
    training_args = TrainingArguments(
        output_dir=str(ctx.tmp / "api-trained"),
        max_steps=2,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=True,
    )
    trainer = (
        None
        if quantized
        else QATTrainer(model=model, processing_class=tokenizer, args=training_args, **data_module)
    )
    initial_weights = {
        name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
    }
    return {
        "initial_weights": initial_weights,
        "model": model,
        "tokenizer": tokenizer,
        "training_args": training_args,
        "data_module": data_module,
        "forward_loop": forward_loop,
        "trainer": trainer,
    }


def verify_python_api(trainer, initial_weights):
    """Assert optimization completed and checkpointing retained quantization state."""
    assert trainer.state.global_step == 2
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    assert losses and all(math.isfinite(value) for value in losses)
    assert any(
        not torch.equal(initial_weights[name], parameter.detach().cpu())
        for name, parameter in trainer.model.named_parameters()
        if name in initial_weights
    )
    output = Path(trainer.args.output_dir)
    assert (output / "modelopt_state.pth").is_file()
    assert (output / "model.safetensors").stat().st_size > 0
