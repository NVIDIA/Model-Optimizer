#!/usr/bin/env python3
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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantize AlpamayoR1 and export as an HF-style checkpoint.

Usage:
    python quantize.py --ckpt nvidia/Alpamayo-R1-10B --output-dir ./alpamayo-r1-fp8 --quantize fp8
    python quantize.py --ckpt nvidia/Alpamayo-R1-10B --output-dir ./alpamayo-r1-nvfp4 --quantize nvfp4 --real-quant
"""

import argparse
import collections.abc
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

import einops
import pandas as pd
import torch
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.token_utils import to_special_token
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.export.quant_utils import get_quant_config
from modelopt.torch.opt.plugins.huggingface import (
    _LIBRARY_CLASSES_FOR_PATCHING,
    _PATCHED_CLASSES,
    patch_pretrained_methods,
)
from modelopt.torch.utils.dataset_utils import create_forward_loop, get_dataset_dataloader

logger = logging.getLogger(__name__)

try:
    assert torch.ops.tensorrt.quantize_op.default
except Exception:
    logger.warning("Unable to import quantization op. Please install modelopt library")

MIN_PIXELS = 163840
MAX_PIXELS = 196608
BASE_PROCESSOR_NAME = "Qwen/Qwen3-VL-2B-Instruct"


def create_message(frames: torch.Tensor):
    """Construct the message using images and cot."""
    assert frames.ndim == 4, f"{frames.ndim=}, expected (N, C, H, W)"

    # NOTE: we expand the padding tokens to match training, so we can directly apply native processor from VLM.
    num_traj_token = 48
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": frame} for frame in frames]
            + [
                {
                    "type": "text",
                    "text": f"{hist_traj_placeholder}output the chain-of-thought reasoning of the \
                    driving process, then output the future trajectory.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "<|cot_start|>",
                }
            ],
        },
    ]


def get_processor(tokenizer: AutoTokenizer) -> AutoProcessor:
    """Get the processor for the Qwen3-VL-2B-Instruct model."""
    processor_kwargs = {
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
    }

    processor = AutoProcessor.from_pretrained(BASE_PROCESSOR_NAME, **processor_kwargs)
    processor.tokenizer = tokenizer
    return processor


def to_device(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Recursively cast data into the specified device, dtype."""
    if isinstance(data, torch.Tensor):
        data = data.to(
            device=device,
            dtype=dtype,
        )
        return data
    elif isinstance(data, collections.abc.Mapping):
        return {key: to_device(data[key], device=device, dtype=dtype) for key in data}
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return [to_device(elem, device=device, dtype=dtype) for elem in data]
    else:
        return data


def enable_huggingface_checkpointing_patch() -> None:
    """Patch PreTrainedModel.from_pretrained / save_pretrained to save/restore ModelOpt state.

    Must be called before AlpamayoR1.from_pretrained() when loading a quantized (FP8/NVFP4)
    checkpoint so that modelopt_state.pth is restored and _amax scaling factors are applied.
    """
    for name, (classes, methods_list) in _LIBRARY_CLASSES_FOR_PATCHING.items():
        for cls, patch_methods in zip(classes, methods_list):
            if cls in _PATCHED_CLASSES:
                continue
            patch_methods = [m for m in patch_methods if m[0] != "_from_config"]
            patch_pretrained_methods(cls, patch_methods)
            _PATCHED_CLASSES.add(cls)
        print(f"ModelOpt save/restore enabled for `{name}` library.")


enable_huggingface_checkpointing_patch()


def _teacher_forced_flow_loss_forward(
    self,
    data: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Differentiable forward that returns the flow-matching training targets.

    Bypasses autoregressive reasoning generation and diffusion sampling.
    The VLM runs in a single non-sampling forward pass (with ``<traj_future_start>``
    appended to the prompt) to build the prompt KV cache; the expert then runs once
    on a linearly-interpolated noisy action and returns the predicted velocity field.

    Args:
        data: dict with ``tokenized_data`` (input_ids + other processor outputs),
            ``ego_history_xyz``, ``ego_history_rot``, ``ego_future_xyz``,
            ``ego_future_rot``.

    Returns:
        dict with keys ``v_pred`` and ``v_target``, both shape
        ``(b,n_diffusion_tokens, action_dim)``. Callers compute MSE between them.
    """
    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]
    ego_future_xyz = data["ego_future_xyz"]
    ego_future_rot = data["ego_future_rot"]
    b, n_traj_group, _, _ = ego_history_xyz.shape
    assert n_traj_group == 1, "Only one trajectory group is supported."

    tokenized_data = dict(data["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")
    traj_data_vlm = {
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
    device = input_ids.device

    # Append <traj_future_start> so the expert attends through the full prompt
    # that inference would have generated up to the action block.
    traj_future_start_id = self.tokenizer.convert_tokens_to_ids(
        to_special_token("traj_future_start")
    )
    start_col = torch.full(
        (input_ids.shape[0], 1),
        traj_future_start_id,
        dtype=input_ids.dtype,
        device=device,
    )
    input_ids = torch.cat([input_ids, start_col], dim=1)
    if "attention_mask" in tokenized_data and tokenized_data["attention_mask"] is not None:
        am = tokenized_data["attention_mask"]
        tokenized_data["attention_mask"] = torch.cat(
            [am, torch.ones((am.shape[0], 1), dtype=am.dtype, device=am.device)], dim=1
        )

    vlm_outputs = self.vlm(
        input_ids=input_ids,
        use_cache=True,
        return_dict=True,
        **tokenized_data,
    )
    prompt_cache = vlm_outputs.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    rope_deltas = self.vlm.model.rope_deltas

    n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
    offset = torch.full((b,), prefill_seq_len, device=device, dtype=torch.long)

    position_ids = torch.arange(n_diffusion_tokens, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b).clone()
    delta = rope_deltas + offset[:, None]
    position_ids += delta.to(position_ids.device)

    # No padding between prompt cache and action block: full attention mask.
    attention_mask = torch.zeros(
        (b, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
        dtype=torch.float32,
        device=device,
    )

    forward_kwargs = {}
    if self.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    # Build flow-matching target: x_1 = GT action, x_0 ~ N(0, I).
    x_1 = self.action_space.traj_to_action(
        traj_history_xyz=ego_history_xyz[:, 0],
        traj_history_rot=ego_history_rot[:, 0],
        traj_future_xyz=ego_future_xyz[:, 0],
        traj_future_rot=ego_future_rot[:, 0],
    )  # (b,n_diffusion_tokens, 2)
    x_1 = x_1.to(device=device, dtype=torch.float32)

    x_0 = torch.randn_like(x_1)
    t = torch.rand(b, 1, 1, device=device, dtype=x_1.dtype)
    x_t = (1.0 - t) * x_0 + t * x_1
    v_target = x_1 - x_0

    # Cast to action-module dtype to match action_in_proj / expert weights.
    proj_dtype = next(self.action_in_proj.parameters()).dtype
    x_t_cast = x_t.to(dtype=proj_dtype)
    t_cast = t.to(dtype=proj_dtype)

    future_token_embeds = self.action_in_proj(x_t_cast, t_cast)
    if future_token_embeds.dim() == 2:
        future_token_embeds = future_token_embeds.view(b, n_diffusion_tokens, -1)

    expert_out = self.expert(
        inputs_embeds=future_token_embeds,
        position_ids=position_ids,
        past_key_values=prompt_cache,
        attention_mask=attention_mask,
        use_cache=True,
        **forward_kwargs,
    )
    prompt_cache.crop(prefill_seq_len)
    last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
    v_pred = self.action_out_proj(last_hidden).view(b, *self.action_space.get_action_space_dims())

    return {"v_pred": v_pred.to(torch.float32), "v_target": v_target}


def patch_teacher_forced_flow_loss_forward() -> None:
    """Attach teacher_forced_flow_loss_forward to AlpamayoR1 if missing.

    The public OSS AlpamayoR1 (github.com/nvlabs/alpamayo) does not define this
    method; it exists only on the internal training fork. The body ported above
    is the calibration path used by auto_quantize_model.
    """
    if not hasattr(AlpamayoR1, "teacher_forced_flow_loss_forward"):
        AlpamayoR1.teacher_forced_flow_loss_forward = _teacher_forced_flow_loss_forward


patch_teacher_forced_flow_loss_forward()


def make_joint_calibration_forward_loop(
    *,
    clip_ids: list[str],
    processor,
    t0_us: int,
    top_p: float,
    temperature: float,
    max_generation_length: int,
    calibration_traj_samples: int,
    device: str,
):
    """
    Build a calibration loop that exercises both VLM generation and diffusion.

    This avoids text-only calibration and ensures quantizers in the rollout path
    (vlm/expert/diffusion-related modules) observe representative activations.
    """

    def _calibration_loop(runtime_model):
        runtime_model.eval()
        with torch.no_grad():
            for clip_id in tqdm(clip_ids, desc="Calibration"):
                data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
                messages = create_message(data["image_frames"].flatten(0, 1))
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                model_inputs = {
                    "tokenized_data": inputs,
                    "ego_history_xyz": data["ego_history_xyz"],
                    "ego_history_rot": data["ego_history_rot"],
                }
                model_inputs = to_device(model_inputs, device)

                with torch.autocast("cuda", dtype=torch.float16):
                    runtime_model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=top_p,
                        temperature=temperature,
                        num_traj_samples=calibration_traj_samples,
                        max_generation_length=max_generation_length,
                    )

    return _calibration_loop


def read_clip_ids_from_parquet(parquet_path: str) -> list[str]:
    """
    Reads clip_ids from parquet. Tries common column names; falls back to index if needed.
    Returns clip_ids as a list of strings (unique, preserving first occurrence order).
    """
    parquet_path = str(parquet_path)
    df = pd.read_parquet(parquet_path)
    cols_lower = {c.lower(): c for c in df.columns}
    clip_ids = df[cols_lower["key"]].astype(str).tolist()

    seen = set()
    uniq = []
    for cid in clip_ids:
        if cid not in seen:
            seen.add(cid)
            uniq.append(cid)
    return uniq


def quantize_model(model, args, tokenizer=None, calibration_forward_loop=None):
    """
    Quantize a PyTorch model using ModelOpt post-training quantization (PTQ).

    This function applies quantization to reduce model precision for faster inference
    while maintaining acceptable accuracy. It uses calibration data generated from
    the provided tokenizer to determine optimal quantization parameters.

    Supported quantization formats:
        - fp8: 8-bit floating point quantization
        - nvfp4: 4-bit NVIDIA floating point quantization
    Args:
        model: PyTorch model to quantize. Must be in evaluation mode.
        args: Command line arguments containing quant_format and debug.
        tokenizer: Hugging Face tokenizer for creating calibration data.
            Required only when `calibration_forward_loop` is not provided.
        calibration_forward_loop: Optional callable taking `model` and running
            calibration forward passes. Use this for non-text modules whose
            forward signature is not compatible with dataset_utils batches.

    Returns:
        Quantized model
    """
    # Create calibration forward loop. For standard text models we can build
    # it from tokenizer-based data, but vision modules often need custom args.
    if calibration_forward_loop is None:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when calibration_forward_loop is None")
        calib_dataloader = get_dataset_dataloader(
            tokenizer=tokenizer,
            batch_size=32,
            num_samples=512,
            device="cuda:0",
        )
        calibrate_loop = create_forward_loop(dataloader=calib_dataloader)
    else:
        calibrate_loop = calibration_forward_loop

    if args.quant_format == "fp8":
        quant_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    elif args.quant_format == "nvfp4":
        quant_cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    else:
        raise RuntimeError("Unsupported quantization format")
    quant_cfg["quant_cfg"].append({"quantizer_name": "*vlm.model.visual*", "enable": False})

    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    if args.debug:
        print("================== quantize_model summary ==================")
        mtq.print_quant_summary(model)

    return model


def auto_quantize_model(
    model,
    args,
    *,
    clip_ids,
    processor,
    t0_us: int,
    top_p: float,
    temperature: float,
    max_generation_length: int,
    calibration_traj_samples: int,
    device: str,
):
    """
    Quantize a PyTorch model using ModelOpt's AutoQuantize API.

    Searches per-layer across [NVFP4_DEFAULT_CFG, FP8_DEFAULT_CFG] under the
    effective-bits budget in args.auto_quantize_bits. Calibration data is built
    from the same joint VLM + diffusion rollout used by
    alpamayo_r1.eval.make_joint_calibration_forward_loop.

    Args:
        model: PyTorch model to quantize. Must be in eval mode.
        args: Namespace with `auto_quantize_bits` (float) and `debug` (bool).
        clip_ids: Iterable of clip_ids for calibration.
        processor: HF processor used for chat-template tokenization.
        t0_us, top_p, temperature, max_generation_length, calibration_traj_samples,
        device: Same semantics as make_joint_calibration_forward_loop.

    Returns:
        Quantized model (the search_state from mtq.auto_quantize is discarded).
    """

    def _one_epoch():
        for clip_id in clip_ids:
            data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
            messages = create_message(data["image_frames"].flatten(0, 1))
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = {
                "tokenized_data": inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
                "ego_future_xyz": data["ego_future_xyz"],
                "ego_future_rot": data["ego_future_rot"],
            }
            yield to_device(model_inputs, device)

    class _ReusableLoader:
        """Re-iterable wrapper so modelopt can run calibration + scoring passes."""

        def __iter__(self):
            return _one_epoch()

    data_loader = _ReusableLoader()

    def forward_step(runtime_model, data):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = runtime_model.teacher_forced_flow_loss_forward(data=data)
        v_pred, v_target = out["v_pred"], out["v_target"]
        print(
            f"[autoquant-fwd] v_pred: finite={torch.isfinite(v_pred).all().item()} "
            f"min={v_pred.min().item():.4g} max={v_pred.max().item():.4g} "
            f"abs_mean={v_pred.abs().mean().item():.4g} | "
            f"v_target: finite={torch.isfinite(v_target).all().item()} "
            f"min={v_target.min().item():.4g} max={v_target.max().item():.4g}"
        )
        return out

    def loss_func(output, batch):
        loss = torch.nn.functional.mse_loss(output["v_pred"], output["v_target"])
        print(f"[autoquant-loss] loss={loss.item():.6g} finite={torch.isfinite(loss).item()}")
        return loss

    # try:
    model, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": args.auto_quantize_bits},
        quantization_formats=["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG"],
        data_loader=data_loader,
        forward_step=forward_step,
        loss_func=loss_func,
        disabled_layers="*lm_head*",
        verbose=True,
    )

    print("================== auto_quantize search_state ==================")
    print(search_state)

    if args.debug:
        print("================== auto_quantize_model summary ==================")
        mtq.print_quant_summary(model)

    return model


def main():
    ap = argparse.ArgumentParser(description="Quantize AlpamayoR1 and export as HF checkpoint")
    ap.add_argument(
        "--ckpt",
        type=str,
        default="nvidia/Alpamayo-R1-10B",
        help="HF hub id or local path of the input checkpoint",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the quantized HF checkpoint",
    )
    ap.add_argument(
        "--qformat",
        type=str,
        required=True,
        choices=["fp8", "nvfp4", "auto"],
        help="Quantization format",
    )
    ap.add_argument(
        "--auto_quantize_bits",
        type=float,
        default=4.8,
        help="Effective-bits budget for AutoQuantize (only used when --quantize auto)",
    )
    ap.add_argument(
        "--parquet",
        type=str,
        default="1005_7cam_gold_eval_metadb_public.parquet",
        help="Parquet file with clip_ids for calibration",
    )
    ap.add_argument("--t0_us", type=int, default=5_100_000)
    ap.add_argument("--top_p", type=float, default=0.98)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_generation_length", type=int, default=256)
    ap.add_argument("--num_traj_samples", type=int, default=6)
    ap.add_argument(
        "--limit", type=int, default=644, help="How many clip_ids to use for calibration"
    )
    ap.add_argument(
        "--real-quant",
        action="store_true",
        help="Export packed real-quantized weights (fp8 / NVFP4) via "
        "modelopt.torch.export.export_hf_checkpoint instead of "
        "saving fake-quant fp16 weights with quantizer state.",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    parquet_path = (script_dir / args.parquet).resolve()

    clip_ids = read_clip_ids_from_parquet(str(parquet_path))
    if args.limit is not None and args.limit > 0:
        clip_ids = clip_ids[: args.limit]
    print(f"Loaded {len(clip_ids)} clip_ids from: {parquet_path}")

    device = "cuda"
    print(f"Loading model from {args.ckpt!r} ...")
    model = AlpamayoR1.from_pretrained(args.ckpt, dtype=torch.float16).to(
        device=device, dtype=torch.float16
    )
    model.eval()

    processor = get_processor(model.tokenizer)

    # Quantize using existing recipe
    print(f"Quantizing model ({args.qformat}) ...")
    quantization_args = argparse.Namespace(
        quant_format=args.qformat,
        quant_algo="max",
        weight_only=False,
        debug=True,
        auto_quantize_bits=args.auto_quantize_bits,
    )
    if args.qformat == "auto":
        model = auto_quantize_model(
            model,
            quantization_args,
            clip_ids=clip_ids,
            processor=processor,
            t0_us=args.t0_us,
            top_p=args.top_p,
            temperature=args.temperature,
            max_generation_length=args.max_generation_length,
            calibration_traj_samples=args.num_traj_samples,
            device=device,
        )
    else:
        # Build calibration loop
        calibration_forward_loop = make_joint_calibration_forward_loop(
            clip_ids=clip_ids,
            processor=processor,
            t0_us=args.t0_us,
            top_p=args.top_p,
            temperature=args.temperature,
            max_generation_length=args.max_generation_length,
            calibration_traj_samples=args.num_traj_samples,
            device=device,
        )
        model = quantize_model(
            model,
            quantization_args,
            calibration_forward_loop=calibration_forward_loop,
        )
    model.eval()

    # Save as HF-style checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving quantized checkpoint to {args.output_dir!r} ...")

    if args.real_quant:
        # Persist processor + composite config first so export_hf_checkpoint's
        # injected `quantization_config` in config.json is the one that wins.
        processor.save_pretrained(args.output_dir)
        model.config.save_pretrained(args.output_dir)
        with torch.inference_mode():
            export_hf_checkpoint(
                model,
                dtype=torch.float16,
                export_dir=args.output_dir,
            )
    else:
        with torch.inference_mode():
            model.save_pretrained(args.output_dir)

        processor.save_pretrained(args.output_dir)
        model.config.save_pretrained(args.output_dir)

        quant_cfg = get_quant_config(model)
        with open(os.path.join(args.output_dir, "hf_quant_config.json"), "w") as f:
            json.dump(quant_cfg, f)

    print(f"Quantized checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
