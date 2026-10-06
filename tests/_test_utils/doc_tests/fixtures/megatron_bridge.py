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

"""Local inputs and artifact checks for real Megatron Bridge README commands."""

import json
import shutil
import subprocess
from pathlib import Path


def prepare_bridge(ctx):
    """Keep command flags visible; reduce only inputs through documented variables."""
    from _test_utils.torch.transformers_models import create_tiny_qwen3_dir

    ctx.cwd = ctx.tmp / "megatron_bridge"
    shutil.copytree(ctx.repo / "examples/megatron_bridge", ctx.cwd)
    model = create_tiny_qwen3_dir(
        ctx.tmp / "model",
        with_tokenizer=True,
        hidden_size=512,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
    )
    data = ctx.tmp / "calibration.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"text": "The quick brown fox jumps over the lazy dog. " * 8})
            for _ in range(8)
        )
    )
    ctx.env.update(
        MODEL=str(model),
        CALIB_DATA=str(data),
        CALIB_SAMPLES="4",
        SEQ_LENGTH="32",
        TRAIN_ITERS="2",
        EVAL_INTERVAL="1",
        EVAL_ITERS="1",
        GPUS="2",
        GLOBAL_BATCH="4",
        WARMUP_ITERS="1",
        OUTPUT_ROOT=str(ctx.tmp / "outputs"),
        PRUNE_CONFIG=json.dumps({"ffn_hidden_size": 256}),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    print(
        "Bridge smoke: local 2-layer Qwen3, hidden/intermediate=512, 4 calibration records, "
        "sequence=32, two training steps, two real torchrun ranks.",
        flush=True,
    )


def verify_quantized(ctx):
    from _test_utils.torch.megatron.modelopt_state import assert_has_modelopt_state

    output = Path(ctx.env["OUTPUT_ROOT"])
    assert_has_modelopt_state(output / "Qwen3-8B-NVFP4-megatron")
    exported = output / "Qwen3-8B-NVFP4-hf"
    assert (exported / "config.json").is_file()
    assert list(exported.glob("*.safetensors"))
    config = json.loads((exported / "hf_quant_config.json").read_text())
    assert config["quantization"]["quant_algo"] == "NVFP4"


def verify_distilled(ctx):
    import torch
    from transformers import AutoModelForCausalLM

    output = Path(ctx.env["OUTPUT_ROOT"])
    checkpoints = output / "test_distill/checkpoints"
    assert (checkpoints / "iter_0000002").is_dir()
    for iteration in (1, 2):
        model = AutoModelForCausalLM.from_pretrained(output / f"hf_validation/iter_{iteration:07d}")
        assert model.config.hidden_size == 512
        weights = model.state_dict()
        assert all(torch.isfinite(value).all() for value in weights.values())


def verify_pruned(ctx):
    from transformers import AutoModelForCausalLM

    output = Path(ctx.env["OUTPUT_ROOT"]) / "Qwen3-8B-Pruned-6B-manual"
    model = AutoModelForCausalLM.from_pretrained(output)
    assert model.config.intermediate_size == 256
    assert model.config.hidden_size == 512


def prepare_full_scale(ctx):
    """Opt-in integration runs use real remote models and user-provided data."""
    ctx.cwd = ctx.repo / "examples/megatron_bridge"
    ctx.env["OUTPUT_ROOT"] = str(ctx.tmp / "outputs")
    Path(ctx.env["OUTPUT_ROOT"]).mkdir()


def run_distill_prerequisite(ctx):
    """Build real checkpoints using the README's training fence, in the same process group."""
    from _test_utils.doc_tests.parser import parse_markdown

    scenario = next(
        s
        for s in parse_markdown(ctx.repo / "examples/megatron_bridge/README.md")
        if s.id == "bridge-distill"
    )
    code = "\n".join(step.code for step in scenario.steps if step.kind == "run")
    subprocess.run(["bash", "-euo", "pipefail", "-c", code], cwd=ctx.cwd, env=ctx.env, check=True)


def verify_full_export(path):
    """Require a parseable architecture and nonempty saved weights in a full-scale export."""
    assert json.loads((path / "config.json").read_text())["model_type"]
    weights = list(path.glob("*.safetensors"))
    assert weights and all(weight.stat().st_size > 0 for weight in weights)
