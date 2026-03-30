# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""PTQ: Load an HF model, quantize weights + activations with FP4 algorithms, save."""

import argparse
import copy
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from utils import make_supervised_data_module

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

mto.enable_huggingface_checkpointing()

# ── Quantizer attribute building blocks ───────────────────────────────────────

_NVFP4_ACT_DYNAMIC = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    "axis": None,
    "enable": True,
}
_NVFP4_WEIGHT_DYNAMIC = {**_NVFP4_ACT_DYNAMIC}
_NVFP4_WEIGHT_STATIC = {
    **_NVFP4_ACT_DYNAMIC,
    "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
}

_DISABLED = {
    "*lm_head*": {"enable": False},
    "*output_quantizer": {"enable": False},
    "default": {"enable": False},
}


def _cfg(w, algo):
    """Full quant config: given weight attr + algorithm, activation is always dynamic NVFP4."""
    return {
        "quant_cfg": {
            "*weight_quantizer": copy.deepcopy(w),
            "*input_quantizer": copy.deepcopy(_NVFP4_ACT_DYNAMIC),
            **_DISABLED,
        },
        "algorithm": copy.deepcopy(algo),
    }


def _scale_algo(init):
    """Build scale_algorithm dict with optional FP8 sweep."""
    d: dict = {"method": init}
    if init != "max":
        d["fp8_scale_sweep"] = True
    return d


def _smooth_lsq(init):
    """SmoothLSQ algorithm with given init method."""
    return {"method": "smooth_lsq", "scale_algorithm": _scale_algo(init)}


def _lsq(init):
    """LSQ algorithm with given init method."""
    return {"method": "lsq", "scale_algorithm": _scale_algo(init)}


def _laq(init):
    """LAQ algorithm with given init method."""
    return {"method": "laq", "scale_algorithm": _scale_algo(init)}


def _smooth_laq(init):
    """SmoothLAQ algorithm with given init method."""
    return {"method": "smooth_laq", "scale_algorithm": _scale_algo(init)}


def _adaround(init, learnable="smooth_lsq"):
    """adaround algorithm with given init method and learnable-scale mode."""
    return {
        "method": "adaround",
        "init_algorithm": {"method": learnable, "scale_algorithm": _scale_algo(init)},
    }


# ── Algorithm registry ────────────────────────────────────────────────────────

W_DYN = _NVFP4_WEIGHT_DYNAMIC
W_STA = _NVFP4_WEIGHT_STATIC

ALGORITHM_MAP = {
    "nvfp4_default": _cfg(W_DYN, "max"),
    "nvfp4_weight_mse": _cfg(W_STA, {"method": "mse", "fp8_scale_sweep": True}),
    "nvfp4_weight_local_hessian": _cfg(W_STA, {"method": "local_hessian", "fp8_scale_sweep": True}),
    "nvfp4_smooth_lsq_mse_init": _cfg(W_STA, _smooth_lsq("mse")),
    "nvfp4_smooth_lsq_max_init": _cfg(W_STA, _smooth_lsq("max")),
    "nvfp4_smooth_lsq_local_hessian_init": _cfg(W_STA, _smooth_lsq("local_hessian")),
    "nvfp4_adaround_mse_init": _cfg(W_STA, _adaround("mse")),
    "nvfp4_adaround_max_init": _cfg(W_STA, _adaround("max")),
    "nvfp4_adaround_local_hessian_init": _cfg(W_STA, _adaround("local_hessian")),
    "nvfp4_lsq_max_init": _cfg(W_STA, _lsq("max")),
    "nvfp4_lsq_mse_init": _cfg(W_STA, _lsq("mse")),
    "nvfp4_lsq_local_hessian_init": _cfg(W_STA, _lsq("local_hessian")),
    "nvfp4_laq_max_init": _cfg(W_STA, _laq("max")),
    "nvfp4_laq_mse_init": _cfg(W_STA, _laq("mse")),
    "nvfp4_laq_local_hessian_init": _cfg(W_STA, _laq("local_hessian")),
    "nvfp4_smooth_laq_max_init": _cfg(W_STA, _smooth_laq("max")),
    "nvfp4_smooth_laq_mse_init": _cfg(W_STA, _smooth_laq("mse")),
    "nvfp4_smooth_laq_local_hessian_init": _cfg(W_STA, _smooth_laq("local_hessian")),
}


def parse_args():
    p = argparse.ArgumentParser(description="FP4 PTQ for HF models.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--algorithm", type=str, required=True, choices=list(ALGORITHM_MAP))
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--calib_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument(
        "--dataset",
        type=str,
        default="/home/scratch.akuriparambi_coreai/datasets/qat_blend_sft/blend_sft.jsonl",
    )
    p.add_argument("--eval_size", type=int, default=0)
    p.add_argument("--train_size", type=int, default=0)
    p.add_argument("--dataset_cache_path", type=str, default="dataset_cache")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or os.path.join(
        "checkpoints_ptq", f"{model_name}_{args.algorithm}"
    )

    print(f"PTQ | model={args.model} | algo={args.algorithm} | output={output_dir}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = args.max_seq_length

    # Same train/test split & seed as QAT, cached to disk for reuse across runs
    data_module = make_supervised_data_module(
        dataset=args.dataset,
        tokenizer=tokenizer,
        train_size=args.train_size,
        eval_size=args.eval_size,
        dataset_cache_path=args.dataset_cache_path,
    )
    eval_dataset = data_module["eval_dataset"]
    calib_size = min(args.calib_size, len(eval_dataset))
    calib_dataset = eval_dataset.select(range(calib_size))
    print(f"Calibration: {calib_size}/{len(eval_dataset)} eval samples")

    device = next(model.parameters()).device
    calib_dataloader = DataLoader(
        calib_dataset, batch_size=args.batch_size, collate_fn=default_data_collator
    )

    def forward_loop(model):
        for batch in tqdm(calib_dataloader, desc="Calibrating"):
            model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )

    quant_cfg = copy.deepcopy(ALGORITHM_MAP[args.algorithm])
    model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    mtq.print_quant_summary(model)

    # Quick generation sanity check
    prompt = "How is the weather in New York today?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10)
    print(f"Prompt:   {prompt}")
    print(f"Response: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
