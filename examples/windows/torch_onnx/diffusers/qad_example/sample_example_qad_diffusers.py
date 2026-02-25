#!/usr/bin/env python3
"""
QAD (Quantization-Aware Distillation) for LTX-2 using the native LTX training loop + ModelOpt.

Uses:
- LtxvTrainer: training loop, dataset, strategies (masked loss, audio/video split)
- ModelOpt: mtq.quantize for calibration, mtd.convert for distillation

Usage:
    # Training
    accelerate launch --config_file configs/accelerate/fsdp.yaml ltx2_qad_ltx_pipeline.py \
        train --config configs/ltx2_full_finetune.yaml \
        --calib-size 512 \
        --kd-loss-weight 0.5

    # Create inference checkpoint from existing trained weights
    python ltx2_qad_ltx_pipeline.py create-inference \
        --trained path/to/model_weights_step_02200.safetensors \
        --base path/to/ltx-video-2b-v0.9.5.safetensors \
        --output path/to/inference.safetensors
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import struct
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# LTX imports
from ltx_core.model_loader import load_transformer
from ltxv_trainer.config import LtxvTrainerConfig
from ltxv_trainer.datasets import PrecomputedDataset
from ltxv_trainer.timestep_samplers import SAMPLERS
from ltxv_trainer.trainer import LtxvTrainer
from ltxv_trainer.training_strategies import get_training_strategy

# ModelOpt imports
import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.distillation_model import DistillationModel
from modelopt.torch.quantization.config import NVFP4_DEFAULT_CFG
from modelopt.torch.quantization.nn import TensorQuantizer

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

QUANTIZER_KEYWORDS = [
    "_amax", "_zero_point",
    "input_quantizer", "weight_quantizer", "output_quantizer",
]
TEACHER_KEYWORDS = ["_teacher_model"]
LOSS_KEYWORDS = ["_loss_modules"]

NON_TRANSFORMER_PREFIXES = [
    "vae.", "audio_vae.", "vocoder.", "text_embedding_projection.",
    "text_encoders.", "first_stage_model.", "cond_stage_model.", "conditioner.",
]
STRIP_PREFIXES = ["diffusion_model.", "transformer.", "_orig_mod.", "model."]
CORRECT_PREFIX = "model.diffusion_model."

SENSITIVE_LAYER_PATTERNS = [
    "*patchify_proj*",
    "*adaln_single*",
    "*caption_projection*",
    "*proj_out*",
    "*audio_patchify_proj*",
    "*audio_adaln_single*",
    "*audio_caption_projection*",
    "*audio_proj_out*",
    "*av_ca_video_scale_shift_adaln_single*",
    "*av_ca_a2v_gate_adaln_single*",
    "*av_ca_audio_scale_shift_adaln_single*",
    "*av_ca_v2a_gate_adaln_single*",
]


# ─── Multi-node safety ───────────────────────────────────────────────────────

def is_global_rank0() -> bool:
    """Global rank 0 check — safe for multi-node shared filesystem writes."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return os.environ.get("RANK", "0") == "0"


# ─── Format detection and loading ─────────────────────────────────────────────

def detect_format(path: str) -> str:
    """Detect whether file is safetensors or torch pickle."""
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"PK" or magic[:1] == b"\x80":
        return "torch"
    return "safetensors"


def load_state_dict_any_format(path: str, label: str = "") -> tuple[dict, dict | None]:
    """Load state dict from either torch pickle or safetensors."""
    fmt = detect_format(path)
    logger.info(f"[{label}] Detected format: {fmt} for {path}")

    if fmt == "torch":
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "state_dict" in raw:
            return raw["state_dict"], None
        return raw, None
    else:
        try:
            from safetensors.torch import load_file, safe_open
            with safe_open(path, framework="pt", device="cpu") as f:
                metadata = f.metadata() or {}
            return load_file(path, device="cpu"), metadata
        except Exception as e:
            logger.warning(f"safe_open failed ({e}), trying manual parse...")
            return _load_safetensors_manual(path)


def _load_safetensors_manual(path: str) -> tuple[dict, dict]:
    """Manual safetensors parser for files with oversized headers."""
    DTYPE_MAP = {
        "F64": torch.float64, "F32": torch.float32,
        "F16": torch.float16, "BF16": torch.bfloat16,
        "I64": torch.int64, "I32": torch.int32,
        "I16": torch.int16, "I8": torch.int8,
        "U8": torch.uint8, "BOOL": torch.bool,
    }
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    metadata = header.pop("__metadata__", {})
    data_start = 8 + header_size

    state_dict = {}
    with open(path, "rb") as f:
        for k, info in header.items():
            torch_dtype = DTYPE_MAP[info["dtype"]]
            start, end = info["data_offsets"]
            f.seek(data_start + start)
            tensor = torch.frombuffer(bytearray(f.read(end - start)), dtype=torch_dtype)
            if info["shape"]:
                tensor = tensor.reshape(info["shape"])
            state_dict[k] = tensor
    return state_dict, metadata


# ─── Helpers ──────────────────────────────────────────────────────────────────

def is_removable_key(k: str) -> str | None:
    """Return removal reason if key should be removed, else None."""
    if any(kw in k for kw in QUANTIZER_KEYWORDS):
        return "quantizer"
    if any(kw in k for kw in TEACHER_KEYWORDS):
        return "teacher"
    if any(kw in k for kw in LOSS_KEYWORDS):
        return "loss"
    return None


def is_non_transformer(k: str) -> bool:
    return any(k.startswith(p) for p in NON_TRANSFORMER_PREFIXES)


def extract_amax_values(state_dict: dict) -> dict:
    """Extract all amax tensors into a JSON-serializable dict."""
    amax_dict = {}
    for k, v in state_dict.items():
        if "_amax" in k:
            amax_dict[k] = float(v.item()) if v.numel() == 1 else v.cpu().float().tolist()
    return amax_dict


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Recursively move batch tensors to device."""
    result = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            result[k] = {
                ik: iv.to(device) if isinstance(iv, torch.Tensor) else iv
                for ik, iv in v.items()
            }
        elif isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        else:
            result[k] = v
    return result


def apply_connector(training_batch, connector, conditioning_mode: str):
    """Apply Gemma connector to prompt embeddings."""
    device = training_batch.prompt_embeds.device
    connector.to(device)

    prompt_embeds_v, prompt_attention_mask = connector.preprocess_prompt_embeds(
        training_batch.prompt_embeds,
        training_batch.prompt_attention_mask,
        is_audio=False,
    )

    if conditioning_mode == "audio_video":
        prompt_embeds_a, _ = connector.preprocess_prompt_embeds(
            training_batch.prompt_embeds,
            training_batch.prompt_attention_mask,
            is_audio=True,
        )
        final_prompt_embeds = torch.cat([prompt_embeds_v, prompt_embeds_a], dim=-1)
    else:
        final_prompt_embeds = prompt_embeds_v

    training_batch = training_batch.model_copy(update={
        "prompt_embeds": final_prompt_embeds,
        "prompt_attention_mask": prompt_attention_mask,
    })
    return training_batch


# ─── Quantization config builder ─────────────────────────────────────────────

def build_quant_config(
    exclude_blocks: list[int] | None = None,
) -> dict:
    """Build the NVFP4 quantization config with sensitive layers excluded.

    Args:
        exclude_blocks: Transformer block indices to exclude from quantization.
            Defaults to [0, 1, 46, 47] (first 2 and last 2).
    """
    if exclude_blocks is None:
        exclude_blocks = [0, 1, 46, 47]

    quant_cfg = {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
    }

    for pattern in SENSITIVE_LAYER_PATTERNS:
        quant_cfg[pattern] = {"enable": False}

    for block_idx in exclude_blocks:
        quant_cfg[f"*transformer_blocks.{block_idx}.*"] = {"enable": False}

    return {
        "quant_cfg": quant_cfg,
        "algorithm": NVFP4_DEFAULT_CFG["algorithm"],
    }


# ─── Distillation loss ───────────────────────────────────────────────────────

class DiffusionMSELoss(torch.nn.modules.loss._Loss):
    """MSE loss between student and teacher outputs for distillation."""

    def forward(self, student_output, teacher_output):
        print(f"Student shape: {student_output.shape}, Teacher shape: {teacher_output.shape}")
        return torch.nn.functional.mse_loss(
            student_output.float(), teacher_output.float()
        )


# ─── QAD Trainer ──────────────────────────────────────────────────────────────

class LtxvQADTrainer(LtxvTrainer):
    """Extends LtxvTrainer with ModelOpt quantization and distillation.

    Execution order:
        1. super().__init__() loads models, sets up accelerator
        2. _prepare_models_for_training() is OVERRIDDEN to:
           a. Quantize the raw model (PTQ calibration)
           b. Wrap with distillation (teacher + student)
           c. THEN do accelerator.prepare() (FSDP wrapping)
        3. train() runs normal LTX training loop
        4. _training_step() is OVERRIDDEN to add KD loss
        5. _save_checkpoint() is OVERRIDDEN for clean safetensors output
    """

    def __init__(
        self,
        trainer_config: LtxvTrainerConfig,
        quant_cfg: dict,
        calib_size: int = 512,
        kd_loss_weight: float = 0.5,
    ):
        self._quant_cfg = quant_cfg
        self._calib_size = calib_size
        self._kd_loss_weight = kd_loss_weight
        super().__init__(trainer_config)

    # ── Model preparation ─────────────────────────────────────────────────

    def _prepare_models_for_training(self):
        """Override: quantize + distill BEFORE FSDP wrapping."""
        self._transformer.set_gradient_checkpointing(
            self._config.optimization.enable_gradient_checkpointing
        )

        self._run_calibration()
        self._setup_distillation()

        self._vae = self._vae.to("cpu")
        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder = self._text_encoder.to("cpu")

        self._transformer.to(torch.bfloat16)
        self._transformer = self._accelerator.prepare(self._transformer)

        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory after model preparation: {vram_gb:.2f} GB")

    # ── Calibration ───────────────────────────────────────────────────────

    def _run_calibration(self):
        """Run PTQ calibration using LTX's own dataset and training strategy."""
        logger.info("Running PTQ calibration...")

        if not hasattr(self, "_training_strategy") or self._training_strategy is None:
            self._training_strategy = get_training_strategy(self._config.conditioning)

        data_sources = self._training_strategy.get_data_sources()
        dataset = PrecomputedDataset(
            self._config.data.preprocessed_data_root,
            data_sources=data_sources,
        )
        torch.manual_seed(42)
        calib_loader = DataLoader(
            dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )

        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

        calib_steps = min(self._calib_size, len(dataset))
        strategy = self._training_strategy
        device = self._accelerator.device
        connector = self._connector
        conditioning_mode = self._config.conditioning.mode

        self._transformer.to(device)
        if connector is not None:
            connector.to(device)

        def calibration_forward_loop(model):
            model.eval()
            data_iter = iter(calib_loader)
            failures = 0
            with torch.no_grad():
                for i in range(calib_steps):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(calib_loader)
                        batch = next(data_iter)

                    batch = move_batch_to_device(batch, device)

                    try:
                        training_batch = strategy.prepare_batch(batch, timestep_sampler)

                        if connector is not None:
                            training_batch = apply_connector(
                                training_batch, connector, conditioning_mode
                            )

                        model_inputs = strategy.prepare_model_inputs(training_batch)
                        model_dtype = next(model.parameters()).dtype
                        for k, v in model_inputs.items():
                            if isinstance(v, torch.Tensor) and v.is_floating_point():
                                model_inputs[k] = v.to(dtype=model_dtype)
                        model(**model_inputs)

                    except Exception as e:
                        failures += 1
                        if failures == 1:
                            import traceback
                            logger.warning(
                                f"Calibration batch {i} failed:\n{traceback.format_exc()}"
                            )
                        elif failures <= 5:
                            logger.warning(f"Calibration batch {i} failed: {e}")
                        if failures > calib_steps * 0.5:
                            logger.error(
                                f"Too many calibration failures ({failures}/{i+1}), aborting"
                            )
                            return
                        continue

                    if (i + 1) % 50 == 0 or (i + 1) == calib_steps:
                        logger.info(f"Calibrated {i + 1}/{calib_steps} batches")

            if failures > 0:
                logger.warning(
                    f"Calibration completed with {failures}/{calib_steps} failed batches"
                )

        mtq.quantize(self._transformer, self._quant_cfg, calibration_forward_loop)
        logger.info("PTQ calibration complete")
        if is_global_rank0():
            mtq.print_quant_summary(self._transformer)

    # ── Distillation setup ────────────────────────────────────────────────

    def _setup_distillation(self):
        """Load teacher from same checkpoint and wrap with DistillationModel."""
        logger.info("Setting up distillation...")

        checkpoint_path = self._config.model.model_source

        teacher = load_transformer(
            checkpoint_or_state=checkpoint_path,
            device="cpu",
            dtype=torch.bfloat16,
        )

        distill_config = {
            "teacher_model": (lambda: teacher, (), {}),
            "criterion": DiffusionMSELoss(),
            "loss_balancer": mtd.StaticLossBalancer(kd_loss_weight=self._kd_loss_weight),
            "expose_minimal_state_dict": False,
        }

        mtd.convert(self._transformer, mode=[("kd_loss", distill_config)])
        logger.info(f"Distillation model created (kd_loss_weight={self._kd_loss_weight})")

    # ── Training step ─────────────────────────────────────────────────────

    def _training_step(self, batch):
        """Override: use strategy's loss + add distillation loss."""
        training_batch = self._training_strategy.prepare_batch(batch, self._timestep_sampler)

        if self._connector is not None:
            training_batch = apply_connector(
                training_batch, self._connector, self._config.conditioning.mode
            )

        model_inputs = self._training_strategy.prepare_model_inputs(training_batch)

        model_dtype = next(self._transformer.parameters()).dtype
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                model_inputs[k] = v.to(dtype=model_dtype)
            elif isinstance(v, list):
                model_inputs[k] = [
                    t.to(dtype=model_dtype)
                    if isinstance(t, torch.Tensor) and t.is_floating_point() else t
                    for t in v
                ]

        model_pred = self._transformer(**model_inputs)
        hard_loss = self._training_strategy.compute_loss(model_pred, training_batch)

        unwrapped = self._accelerator.unwrap_model(self._transformer)
        if isinstance(unwrapped, DistillationModel) and unwrapped.training:
            return unwrapped.compute_kd_loss(student_loss=hard_loss)

        return hard_loss

    # ── Checkpoint saving ─────────────────────────────────────────────────

    def _save_checkpoint(self) -> Path:
        """Override: save clean student weights as real safetensors + modelopt state.

        Fixes vs original:
        - Uses safetensors.save_file() directly (no silent fallback to pickle)
        - Atomic save (write to .tmp, rename on success)
        - Multi-node safe (global rank 0 only for writes)
        - Extracts and saves amax values as separate JSON
        """
        from safetensors.torch import save_file

        self._accelerator.wait_for_everyone()
        save_dir = Path(self._config.output_dir) / "checkpoints"

        # FSDP collective — all ranks must call this
        state_dict = self._accelerator.get_state_dict(self._transformer)

        prefix = "model" if self._config.model.training_mode == "full" else "lora"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename

        if is_global_rank0() and state_dict is not None:
            save_dir.mkdir(exist_ok=True, parents=True)
            total_keys = len(state_dict)

            # 1. Extract amax values BEFORE filtering
            amax_dict = extract_amax_values(state_dict)
            if amax_dict:
                amax_path = save_dir / f"amax_step_{self._global_step:05d}.json"
                with open(amax_path, "w") as f:
                    json.dump(
                        {"total_amax_keys": len(amax_dict), "amax_values": amax_dict},
                        f, indent=2, sort_keys=True,
                    )
                logger.info(f"Saved {len(amax_dict)} amax values to {amax_path}")

            # 2. Filter out teacher, loss, and quantizer keys
            clean_state = {}
            removed = {"teacher": 0, "loss": 0, "quantizer": 0}
            for k, v in state_dict.items():
                reason = is_removable_key(k)
                if reason:
                    removed[reason] += 1
                else:
                    clean_state[k] = v
            del state_dict

            logger.info(
                f"Filtered: kept {len(clean_state)} keys, "
                f"removed {sum(removed.values())} "
                f"(teacher={removed['teacher']}, loss={removed['loss']}, "
                f"quantizer={removed['quantizer']})"
            )

            # 3. Match dtypes with base model
            try:
                from safetensors.torch import load_file as _load_base
                base_state = _load_base(self._config.model.model_source)
                dtype_fixed = 0
                for k in clean_state:
                    base_key = f"{CORRECT_PREFIX}{k}"
                    if base_key in base_state:
                        ref_dtype = base_state[base_key].dtype
                    elif k in base_state:
                        ref_dtype = base_state[k].dtype
                    else:
                        ref_dtype = (
                            torch.bfloat16
                            if clean_state[k].dtype == torch.float32 else None
                        )

                    if ref_dtype is not None and clean_state[k].dtype != ref_dtype:
                        clean_state[k] = clean_state[k].to(ref_dtype)
                        dtype_fixed += 1
                del base_state
                if dtype_fixed:
                    logger.info(f"Fixed {dtype_fixed} tensor dtypes to match base model")
            except Exception as e:
                logger.warning(f"Could not load base model for dtype matching: {e}")

            # 4. Save as safetensors (atomic: write to .tmp, then rename)
            save_size_gb = sum(
                v.numel() * v.element_size() for v in clean_state.values()
            ) / (1024**3)
            logger.info(
                f"Saving checkpoint: {len(clean_state)} keys, {save_size_gb:.2f} GB"
            )

            tmp_path = saved_weights_path.with_suffix(".safetensors.tmp")
            save_file(clean_state, str(tmp_path))
            tmp_path.rename(saved_weights_path)
            del clean_state

            # 5. Save modelopt state
            try:
                unwrapped = self._accelerator.unwrap_model(self._transformer)
                modelopt_state = mto.modelopt_state(unwrapped)
                from modelopt.torch.quantization.utils import get_quantizer_state_dict
                modelopt_state["modelopt_state_weights"] = get_quantizer_state_dict(
                    unwrapped
                )
                modelopt_path = (
                    save_dir / f"modelopt_state_step_{self._global_step:05d}.pth"
                )
                torch.save(modelopt_state, str(modelopt_path))
                logger.info(f"Saved modelopt state to {modelopt_path}")
            except Exception as e:
                logger.warning(f"Failed to save modelopt state: {e}")

        self._accelerator.wait_for_everyone()
        self._checkpoint_paths.append(saved_weights_path)
        self._cleanup_checkpoints()
        return saved_weights_path


# ─── Standalone inference checkpoint creation ─────────────────────────────────

def create_inference_checkpoint(
    trained_path: str,
    base_path: str,
    output_path: str,
):
    """Create inference checkpoint by merging base and trained weights.

    Handles both torch pickle and safetensors input formats.
    Always outputs real safetensors format with atomic save.

    Strategy:
    1. Load trained checkpoint (any format)
    2. Extract amax values, then strip teacher/loss/quantizer keys
    3. Load base checkpoint, match dtypes
    4. Add 'model.diffusion_model.' prefix (ComfyUI compatibility)
    5. Merge: base non-transformer + base embeddings_connectors + trained transformer
    6. Save as safetensors with base model metadata
    """
    from safetensors.torch import save_file

    trained_path = Path(trained_path)
    base_path = Path(base_path)
    output_path = Path(output_path)

    for p, label in [(trained_path, "Trained"), (base_path, "Base")]:
        if not p.exists():
            print(f"ERROR: {label} checkpoint not found: {p}")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("Creating Inference Checkpoint")
    print("=" * 80)

    # ── Step 1: Load trained checkpoint ──
    print(f"\n[1/7] Loading trained checkpoint: {trained_path}")
    trained_state, _ = load_state_dict_any_format(str(trained_path), label="trained")
    print(f"  Loaded {len(trained_state)} keys")

    # ── Step 2: Extract amax values ──
    print(f"\n[2/7] Extracting amax values...")
    amax_dict = extract_amax_values(trained_state)
    if amax_dict:
        amax_path = output_path.parent / (output_path.stem + "_amax.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(amax_path, "w") as f:
            json.dump(
                {"total_amax_keys": len(amax_dict), "amax_values": amax_dict},
                f, indent=2, sort_keys=True,
            )
        print(f"  Saved {len(amax_dict)} amax values to: {amax_path}")
    else:
        print("  No amax values found")

    # ── Step 3: Remove teacher / loss / quantizer keys ──
    print(f"\n[3/7] Cleaning: removing teacher, loss, and quantizer keys...")
    removal_counts = {"quantizer": 0, "teacher": 0, "loss": 0}
    cleaned = {}
    for k, v in trained_state.items():
        reason = is_removable_key(k)
        if reason:
            removal_counts[reason] += 1
        else:
            cleaned[k] = v
    del trained_state

    for reason, cnt in removal_counts.items():
        if cnt > 0:
            print(f"  Removed {cnt} {reason} keys")
    print(f"  Remaining: {len(cleaned)} keys")

    # ── Step 4: Load base checkpoint ──
    print(f"\n[4/7] Loading base checkpoint: {base_path}")
    base_state, base_metadata = load_state_dict_any_format(str(base_path), label="base")
    if base_metadata is None:
        base_metadata = {}
    print(f"  Loaded {len(base_state)} keys")

    # ── Step 5: Match dtypes with base ──
    print(f"\n[5/7] Matching dtypes with base model...")
    dtype_fixed = 0
    dtype_mismatches = []
    for k in cleaned:
        base_key = f"{CORRECT_PREFIX}{k}"
        if base_key in base_state:
            ref_dtype = base_state[base_key].dtype
            match_source = base_key
        elif k in base_state:
            ref_dtype = base_state[k].dtype
            match_source = k
        else:
            ref_dtype = torch.bfloat16 if cleaned[k].dtype == torch.float32 else None
            match_source = "fallback (fp32->bf16)"

        if ref_dtype is not None and cleaned[k].dtype != ref_dtype:
            dtype_mismatches.append({
                "key": k,
                "trained": str(cleaned[k].dtype),
                "base": str(ref_dtype),
                "source": match_source,
            })
            cleaned[k] = cleaned[k].to(ref_dtype)
            dtype_fixed += 1

    print(f"  Fixed {dtype_fixed} tensor dtypes")
    if dtype_mismatches:
        conversion_counter = Counter()
        for m in dtype_mismatches:
            conversion_counter[f"{m['trained']} -> {m['base']}"] += 1
        for conv, cnt in conversion_counter.most_common():
            print(f"    {conv}: {cnt} tensors")

        dtype_log_path = output_path.parent / (output_path.stem + "_dtype_fixes.json")
        with open(dtype_log_path, "w") as f:
            json.dump({"total": dtype_fixed, "fixes": dtype_mismatches}, f, indent=2)
        print(f"  Dtype fix log saved to: {dtype_log_path}")

    # ── Step 6: Add prefix ──
    print(f"\n[6/7] Adding '{CORRECT_PREFIX}' prefix to transformer keys...")
    prefixed = {}
    stats = {"already_correct": 0, "non_transformer_skipped": 0, "fixed": 0}

    for k, v in cleaned.items():
        if is_non_transformer(k):
            stats["non_transformer_skipped"] += 1
            continue
        elif k.startswith(CORRECT_PREFIX):
            prefixed[k] = v
            stats["already_correct"] += 1
        else:
            clean_k = k
            for pfx in STRIP_PREFIXES:
                if clean_k.startswith(pfx):
                    clean_k = clean_k[len(pfx):]
                    break
            prefixed[f"{CORRECT_PREFIX}{clean_k}"] = v
            stats["fixed"] += 1

    del cleaned
    print(f"  Already correct: {stats['already_correct']}")
    print(f"  Prefix added:    {stats['fixed']}")
    print(f"  Non-transformer: {stats['non_transformer_skipped']} (skipped)")

    # ── Step 7: Merge with base ──
    print(f"\n[7/7] Merging with base model...")

    base_non_transformer = {
        k: v for k, v in base_state.items() if is_non_transformer(k)
    }
    base_connectors = {
        k: v for k, v in base_state.items()
        if "embeddings_connector" in k and k.startswith(CORRECT_PREFIX)
    }
    del base_state

    print(f"  Base non-transformer:       {len(base_non_transformer)} keys")
    print(f"  Base embeddings_connectors: {len(base_connectors)} keys (NOT trained)")
    print(f"  Trained transformer:        {len(prefixed)} keys")

    merged = {}
    merged.update(base_non_transformer)
    merged.update(base_connectors)
    merged.update(prefixed)
    del base_non_transformer, base_connectors, prefixed

    total_params = sum(v.numel() for v in merged.values())
    total_gb = sum(v.numel() * v.element_size() for v in merged.values()) / (1024 ** 3)
    print(f"\n  Final: {len(merged)} keys, {total_params:,} params, {total_gb:.2f} GB")

    # ── Save (atomic) ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".safetensors.tmp")
    print(f"\n  Saving as safetensors...")
    save_file(merged, str(tmp_path), metadata=base_metadata)
    tmp_path.rename(output_path)

    file_size_gb = output_path.stat().st_size / (1024 ** 3)

    print("\n" + "=" * 80)
    print("Inference Checkpoint Created!")
    print("=" * 80)
    print(f"  Path:   {output_path}")
    print(f"  Format: safetensors")
    print(f"  Size:   {file_size_gb:.2f} GB")
    print(f"  Keys:   {len(merged)}")
    if amax_dict:
        print(f"  Amax:   {output_path.stem}_amax.json ({len(amax_dict)} values)")
    print("=" * 80)

    del merged
    gc.collect()


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="QAD for LTX-2 (Native Trainer + ModelOpt)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── Train command ──
    train_parser = subparsers.add_parser("train", help="Run QAD training")
    train_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to LTX training config YAML",
    )
    train_parser.add_argument(
        "--calib-size", type=int, default=512,
        help="Number of calibration batches for PTQ",
    )
    train_parser.add_argument(
        "--kd-loss-weight", type=float, default=0.5,
        help="KD loss weight (0=pure hard loss, 1=pure KD loss)",
    )
    train_parser.add_argument(
        "--exclude-blocks", type=int, nargs="*", default=[0, 1, 46, 47],
        help="Transformer block indices to exclude from quantization",
    )
    train_parser.add_argument(
        "--skip-inference-ckpt", action="store_true",
        help="Skip creating inference checkpoint after training",
    )

    # ── Create inference checkpoint command ──
    infer_parser = subparsers.add_parser(
        "create-inference",
        help="Create inference checkpoint from trained weights",
    )
    infer_parser.add_argument(
        "--trained", type=str, required=True,
        help="Path to trained checkpoint (any format)",
    )
    infer_parser.add_argument(
        "--base", type=str, required=True,
        help="Path to base model checkpoint",
    )
    infer_parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for inference .safetensors",
    )

    # Backward compatibility: if no subcommand, treat as train
    args, remaining = parser.parse_known_args()
    if args.command is None:
        if "--config" in sys.argv:
            args = train_parser.parse_args(sys.argv[1:])
            args.command = "train"
        elif "--trained" in sys.argv:
            old_parser = argparse.ArgumentParser()
            old_parser.add_argument("--create-inference", action="store_true")
            old_parser.add_argument("--trained-checkpoint", "--trained", type=str)
            old_parser.add_argument("--base-checkpoint", "--base", type=str)
            old_parser.add_argument("--output-checkpoint", "--output", type=str)
            args = old_parser.parse_args()
            args.command = "create-inference"
            args.trained = args.trained_checkpoint
            args.base = args.base_checkpoint
            args.output = args.output_checkpoint
        else:
            parser.print_help()
            sys.exit(1)

    return args


def main():
    # Only rank 0 gets INFO logging; other ranks get WARNING to reduce noise
    log_level = logging.INFO if is_global_rank0() else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress verbose logging from ModelOpt / third-party libs on all ranks
    for noisy_logger in ["modelopt", "torch.distributed", "accelerate"]:
        logging.getLogger(noisy_logger).setLevel(
            logging.WARNING if is_global_rank0() else logging.ERROR
        )

    args = parse_args()

    if args.command == "create-inference":
        create_inference_checkpoint(
            trained_path=args.trained,
            base_path=args.base,
            output_path=args.output,
        )
        return

    # ── Train ──
    import yaml

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    # Extract QAD-specific config (not part of LtxvTrainerConfig)
    qad_config = config_dict.pop("qad", {})

    config = LtxvTrainerConfig(**config_dict)

    # Resolve QAD params: CLI args override YAML values, YAML overrides defaults
    calib_size = args.calib_size if args.calib_size != 512 else qad_config.get("calib_size", 512)
    kd_loss_weight = (
        args.kd_loss_weight if args.kd_loss_weight != 0.5
        else qad_config.get("kd_loss_weight", 0.5)
    )
    exclude_blocks = (
        args.exclude_blocks if args.exclude_blocks != [0, 1, 46, 47]
        else qad_config.get("exclude_blocks", [0, 1, 46, 47])
    )
    skip_inference_ckpt = args.skip_inference_ckpt or qad_config.get("skip_inference_ckpt", False)

    quant_cfg = build_quant_config(exclude_blocks=exclude_blocks)

    logger.info("=" * 80)
    logger.info("QAD for LTX-2 (Native LTX Trainer + ModelOpt)")
    logger.info("=" * 80)
    logger.info(f"Config:          {args.config}")
    logger.info(f"Model:           {config.model.model_source}")
    logger.info(f"Data:            {config.data.preprocessed_data_root}")
    logger.info(f"Output:          {config.output_dir}")
    logger.info(f"Calib size:      {calib_size}")
    logger.info(f"KD loss weight:  {kd_loss_weight}")
    logger.info(f"Excluded blocks: {exclude_blocks}")

    trainer = LtxvQADTrainer(
        trainer_config=config,
        quant_cfg=quant_cfg,
        calib_size=calib_size,
        kd_loss_weight=kd_loss_weight,
    )

    saved_path, stats = trainer.train()

    logger.info(f"Training complete! Checkpoint: {saved_path}")
    logger.info(
        f"Steps/sec: {stats.steps_per_second:.2f}, "
        f"Peak GPU: {stats.peak_gpu_memory_gb:.2f} GB"
    )

    if (
        not skip_inference_ckpt
        and is_global_rank0()
        and saved_path is not None
    ):
        create_inference_checkpoint(
            trained_path=str(saved_path),
            base_path=config.model.model_source,
            output_path=str(
                Path(config.output_dir) / "ltx2_qad_inference.safetensors"
            ),
        )


if __name__ == "__main__":
    main()
