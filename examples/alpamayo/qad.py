#!/usr/bin/env python3
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

"""Quantization-aware distillation of a quantized AlpamayoR1 VLM.

Takes the quantized checkpoint written by ``quantize.py`` and distills its VLM against the
FP16 VLM of the original checkpoint, recovering accuracy lost to quantization. Only the VLM
is trained; the action expert stays frozen.

Usage:
    python quantize.py --ckpt nvidia/Alpamayo-R1-10B --output-dir ./alpamayo-auto --quantize auto
    torchrun --standalone --nproc_per_node 8 qad.py \
        --student_ckpt ./alpamayo-auto --output_dir ./alpamayo-auto-qad \
        --parquet ./train_clips.parquet \
        --max_steps 500 --fsdp2 --grad_ckpt --export
"""

import argparse
import gc
import glob
import os

import physical_ai_av
import torch
import transformers
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

# quantize.py owns the Alpamayo input pipeline (prompt construction, processor, clip-id
# reading); import it rather than restating it so the two stages cannot drift apart.
from quantize import create_message, get_processor, read_clip_ids_from_parquet
from safetensors.torch import load_file
from torch.utils.data import Dataset

import modelopt.torch.opt as mto
from modelopt.torch.quantization.plugins.transformers_trainer import QADTrainer

# The VLM's decoder and vision blocks are the FSDP2 auto-wrap unit.
FSDP_WRAP_CLASSES = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]


class ClipVLMDataset(Dataset):
    """Maps a PhysicalAI-AV clip id to the input dict the Alpamayo VLM consumes.

    The prompt ends at ``<|cot_start|>``, so every position in the sequence is a prompt
    position and the distillation loss covers all of them.
    """

    def __init__(self, clip_ids, processor, model_for_fusion, *, t0_us, fuse_traj, avdi=None):
        self.clip_ids = list(clip_ids)
        self.processor = processor
        self.model_for_fusion = model_for_fusion
        self.t0_us = t0_us
        self.fuse_traj = fuse_traj
        self.avdi = avdi

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, idx: int) -> dict:
        data = load_physical_aiavdataset(self.clip_ids[idx], t0_us=self.t0_us, avdi=self.avdi)
        messages = create_message(data["image_frames"].flatten(0, 1))
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        if self.fuse_traj:
            # Replace the trajectory-history placeholders with real history tokens. Length
            # preserving, and applied identically for student and teacher.
            input_ids = self.model_for_fusion.fuse_traj_tokens(
                input_ids,
                {
                    "ego_history_xyz": data["ego_history_xyz"],
                    "ego_history_rot": data["ego_history_rot"],
                },
            )

        attention_mask = inputs.get("attention_mask")
        # Batch size is 1, so nothing is ever padded and no position is masked out. ``labels``
        # is still required: KDTrainer's eval branch runs a cross-entropy forward to report
        # ``eval_ce_loss`` alongside the KD loss.
        item = {"input_ids": input_ids, "labels": input_ids.clone()}
        if attention_mask is not None:
            item["attention_mask"] = attention_mask
        for key in ("pixel_values", "image_grid_thw"):
            if key in inputs:
                item[key] = inputs[key]
        return item


class VLMCollator:
    """Batch-of-one collator.

    Each dataset item is already a batch of one from the Qwen processor, with its own leading
    dims on ``pixel_values``/``image_grid_thw``. Use ``--grad_accum`` for a larger effective
    batch; padding several multimodal samples together is not implemented.
    """

    def __call__(self, samples: list[dict]) -> dict:
        if len(samples) != 1:
            raise NotImplementedError(
                f"batch size must be 1 (got {len(samples)}); use --grad_accum instead."
            )
        return samples[0]


class SyncStateCadenceCallback(transformers.TrainerCallback):
    """Re-assert ``--save_steps``/``--eval_steps``/``--logging_steps`` after a resume.

    ``DefaultFlowCallback`` reads the cadence from ``TrainerState``, not ``TrainingArguments``,
    and ``TrainerState.load_from_json`` overwrites it with the resumed checkpoint's values, so a
    resumed run silently keeps its original cadence. ``on_train_begin`` runs after the state is
    restored, which is where this can still take effect.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        for field in ("save_steps", "eval_steps", "logging_steps"):
            wanted = getattr(args, field)
            if getattr(state, field, None) != wanted:
                print(f"[qad] state.{field} -> {wanted} (from args)", flush=True)
                setattr(state, field, wanted)


def is_main_rank() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def build_avdi(revision: str):
    """Build one dataset interface shared across clip loads, pinned to a dataset revision.

    Pinning skips a per-clip ref-lookup call to the Hub, and is required to read a pre-warmed
    cache with ``HF_HUB_OFFLINE=1``. Without a revision each clip is loaded on its own.
    """
    if not revision:
        return None
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(revision=revision)
    print(f"[qad] PhysicalAI-AV interface pinned to revision {avdi.revision}", flush=True)
    return avdi


def clip_slice(parquet: str, offset: int, limit: int) -> list[str]:
    clip_ids = read_clip_ids_from_parquet(parquet)[offset:]
    return clip_ids[:limit]


def export_full_model(student_ckpt: str, trained_vlm_dir: str, export_dir: str, dtype) -> None:
    """Reassemble a full AlpamayoR1 from the trained VLM weights and save it.

    ModelOpt tracks its state on the top-level model, so training the VLM submodule alone
    produces trained weights with no usable ``modelopt_state.pth``. Reloading the quantized
    checkpoint restores the quantizer structure; the trained tensors are then loaded into its
    VLM and the whole model re-saved, which is what makes the result loadable again through
    ``AlpamayoR1.from_pretrained``.
    """
    print(f"[qad] export: quant structure from {student_ckpt}", flush=True)
    full = AlpamayoR1.from_pretrained(student_ckpt, dtype=dtype)

    shards = sorted(glob.glob(os.path.join(trained_vlm_dir, "model*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no model*.safetensors found in {trained_vlm_dir}")
    state_dict = {}
    for shard in shards:
        state_dict.update(load_file(shard))

    missing, unexpected = full.vlm.load_state_dict(state_dict, strict=False)
    if missing:
        raise ValueError(
            f"[qad] export: missing keys in trained VLM state dict "
            f"(showing first 8): {list(missing)[:8]}. "
            f"The trained checkpoint is incomplete; check that --output_dir "
            f"contains the full trained VLM shards."
        )
    if unexpected:
        raise ValueError(
            f"[qad] export: unexpected keys in trained VLM state dict "
            f"(showing first 8): {list(unexpected)[:8]}. "
            f"The trained checkpoint does not match the quantized student; "
            f"check --student_ckpt and --output_dir point to compatible checkpoints."
        )
    print(
        f"[qad] export: loaded {len(state_dict)} trained tensors",
        flush=True,
    )

    full.save_pretrained(export_dir)
    get_processor(full.tokenizer).save_pretrained(export_dir)
    full.config.save_pretrained(export_dir)
    print(f"[qad] export: wrote full model to {export_dir}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--student_ckpt",
        required=True,
        help="quantized checkpoint from quantize.py; its VLM is the student",
    )
    ap.add_argument(
        "--teacher_ckpt",
        default="nvidia/Alpamayo-R1-10B",
        help="unquantized checkpoint whose VLM is the distillation teacher",
    )
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--parquet", required=True, help="clip ids, read from the 'key' column")
    ap.add_argument("--train_offset", type=int, default=0)
    ap.add_argument("--limit_train", type=int, required=True, help="train clips to use")
    ap.add_argument("--val_offset", type=int, default=12)
    ap.add_argument("--limit_val", type=int, default=4)
    ap.add_argument("--t0_us", type=int, default=5_100_000)
    ap.add_argument(
        "--pai_revision",
        default=os.environ.get("PAI_REVISION", ""),
        help="pin the PhysicalAI-AV dataset revision; required to read a cache offline",
    )
    ap.add_argument("--no_fuse_traj", dest="fuse_traj", action="store_false", default=True)

    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--logging_steps", type=int, default=1)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=0, help=">0 evaluates the val KD loss")
    ap.add_argument("--grad_ckpt", action="store_true", help="required to fit the 10B model")
    ap.add_argument("--freeze_vision", action="store_true", help="train the text backbone only")
    ap.add_argument("--fsdp2", action="store_true", help="shard student and teacher with FSDP2")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--resume_from_checkpoint", default="")
    ap.add_argument(
        "--ignore_data_skip", action="store_true", help="skip replaying consumed batches"
    )

    ap.add_argument(
        "--export", action="store_true", help="reassemble the full model after training"
    )
    ap.add_argument("--export_dir", default="", help="defaults to <output_dir>-full")
    ap.add_argument(
        "--trained_vlm",
        default="",
        help="export from this already-trained VLM directory and skip training",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        dtype = getattr(torch, args.dtype)
    except AttributeError:
        raise ValueError(
            f"Invalid dtype '{args.dtype}'. Must be a valid torch dtype "
            f"(e.g., 'float16', 'bfloat16', 'float32')"
        )
    export_dir = args.export_dir or f"{args.output_dir.rstrip('/')}-full"

    # Validate train/val offsets and limits are non-negative
    if args.train_offset < 0:
        raise ValueError(f"--train_offset must be non-negative, got {args.train_offset}")
    if args.limit_train < 0:
        raise ValueError(f"--limit_train must be non-negative, got {args.limit_train}")
    if args.val_offset < 0:
        raise ValueError(f"--val_offset must be non-negative, got {args.val_offset}")
    if args.limit_val < 0:
        raise ValueError(f"--limit_val must be non-negative, got {args.limit_val}")

    # Validate train/val split does not overlap
    train_end = args.train_offset + args.limit_train
    val_end = args.val_offset + args.limit_val
    if not (train_end <= args.val_offset or val_end <= args.train_offset):
        raise ValueError(
            f"train and validation clip ranges overlap: "
            f"train=[{args.train_offset}:{train_end}], "
            f"val=[{args.val_offset}:{val_end}]. "
            f"Use non-overlapping offsets and limits."
        )

    # Restores the quantizer state and _amax buffers when loading the quantized checkpoint.
    mto.enable_huggingface_checkpointing()

    if args.trained_vlm:
        if is_main_rank():
            export_full_model(args.student_ckpt, args.trained_vlm, export_dir, dtype)
        return

    print(
        f"[qad] student (quantized) {args.student_ckpt} | teacher (FP) {args.teacher_ckpt}",
        flush=True,
    )
    student_full = AlpamayoR1.from_pretrained(args.student_ckpt, dtype=dtype)
    teacher_full = AlpamayoR1.from_pretrained(args.teacher_ckpt, dtype=dtype)
    student_vlm, teacher_vlm = student_full.vlm, teacher_full.vlm
    teacher_vlm.requires_grad_(False)

    if args.freeze_vision:
        frozen = 0
        for name, param in student_vlm.named_parameters():
            if "visual" in name:
                param.requires_grad_(False)
                frozen += 1
        print(f"[qad] froze {frozen} vision-tower parameters", flush=True)

    train_ids = clip_slice(args.parquet, args.train_offset, args.limit_train)
    val_ids = clip_slice(args.parquet, args.val_offset, args.limit_val)
    print(f"[qad] train clips={len(train_ids)} val clips={len(val_ids)}", flush=True)

    processor = get_processor(student_full.tokenizer)
    dataset_kwargs = {
        "processor": processor,
        "model_for_fusion": student_full,
        "t0_us": args.t0_us,
        "fuse_traj": args.fuse_traj,
        "avdi": build_avdi(args.pai_revision),
    }

    training_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "eval_strategy": "steps" if args.eval_steps > 0 else "no",
        "eval_steps": args.eval_steps if args.eval_steps > 0 else None,
        # Measure the untrained student before the first optimizer step, so the val curve has a
        # step-0 anchor rather than starting at the first eval interval.
        "eval_on_start": args.eval_steps > 0,
        "bf16": dtype == torch.bfloat16,
        "fp16": dtype == torch.float16,
        "ignore_data_skip": args.ignore_data_skip,
        "gradient_checkpointing": args.grad_ckpt,
        # When gradient checkpointing is enabled with FSDP2, use non-reentrant checkpointing:
        # the reentrant autograd.Function cannot see all-gathered DTensor parameters as graph
        # inputs, so the recomputed forward would receive no gradient.
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if args.grad_ckpt else None,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "report_to": "none",
    }
    if args.fsdp2:
        # transformers derives the other FSDP_* variables from fsdp_config but never sets
        # FSDP_VERSION, so accelerate would build an FSDP1 plugin, which KDTrainer rejects.
        os.environ["FSDP_VERSION"] = "2"
        # Likewise unset under torchrun. Without resharding after forward the FSDP2 parameter
        # group has no post-forward mesh, which ModelOpt's weight-access path requires.
        os.environ["FSDP_RESHARD_AFTER_FORWARD"] = "true"
        training_kwargs["fsdp"] = "full_shard auto_wrap"
        training_kwargs["fsdp_config"] = {
            "fsdp_version": 2,
            "transformer_layer_cls_to_wrap": FSDP_WRAP_CLASSES,
        }

    # No quant_args: the student is already quantized, so the trainer's own quantization step
    # must not run. QATTrainer's modelopt_state_train.pth holds only quantizer buffers, since
    # ModeloptStateManager lives on the AlpamayoR1 root, above the VLM.
    trainer = QADTrainer(
        model=student_vlm,
        args=transformers.TrainingArguments(**training_kwargs),
        processing_class=student_full.tokenizer,
        distill_args={"teacher_model": teacher_vlm, "temperature": args.temperature},
        train_dataset=ClipVLMDataset(train_ids, **dataset_kwargs),
        eval_dataset=ClipVLMDataset(val_ids, **dataset_kwargs),
        data_collator=VLMCollator(),
    )
    trainer.add_callback(SyncStateCadenceCallback())

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    if trainer.state.log_history:
        final_log = trainer.state.log_history[-1]
        print(
            f"[qad] final train loss: {final_log.get('loss', 'N/A'):.6f} | "
            f"eval loss: {final_log.get('eval_loss', 'N/A')}",
            flush=True,
        )
    trainer.save_model(args.output_dir)
    print(f"[qad] trained VLM saved to {args.output_dir}", flush=True)

    if args.export:
        # Drop every reference before loading a third full model. dataset_kwargs holds
        # student_full as model_for_fusion and student_full holds the VLM, so clearing it
        # is what actually frees the student — the del alone frees only the teacher.
        dataset_kwargs.clear()
        del trainer, student_full, teacher_full, student_vlm, teacher_vlm
        gc.collect()
        torch.cuda.empty_cache()
        if is_main_rank():
            export_full_model(args.student_ckpt, args.output_dir, export_dir, dtype)


if __name__ == "__main__":
    main()
