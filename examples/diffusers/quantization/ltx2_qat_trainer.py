#!/usr/bin/env python3
"""
QAT for LTX-2 using QATTrainer (HuggingFace Trainer API)
Uses the same path setup and dataset as qat_test11_safetensor.py
"""

import argparse
import torch
from pathlib import Path
import sys

# Add Training/src to path - ROBUST PATH DETECTION
SCRIPT_DIR = Path(__file__).parent.resolve()

# Try multiple possible locations
possible_paths = [
    SCRIPT_DIR / "Training" / "src",  # Original: script_dir/Training/src
    SCRIPT_DIR / "training" / "src",  # Lowercase: script_dir/training/src
    SCRIPT_DIR.parent / "training" / "src",  # One level up: ../training/src
    Path("/lustre/fsw/portfolios/adlr/projects/adlr_psx_numerics/users/ynankani/ltx-2/training/src"),  # Absolute
]

TRAINING_SRC = None
for path in possible_paths:
    if (path / "ltx_core").exists():
        TRAINING_SRC = path
        print(f"✓ Found ltx_core at: {TRAINING_SRC}")
        break

if TRAINING_SRC is None:
    print("ERROR: Could not find ltx_core module in any of these locations:")
    for path in possible_paths:
        print(f"  - {path}")
    print(f"\nCurrent script location: {SCRIPT_DIR}")
    print(f"Current working directory: {Path.cwd()}")
    sys.exit(1)

sys.path.insert(0, str(TRAINING_SRC))

# ModelOpt imports
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import INT8_DEFAULT_CFG
from modelopt.torch.quantization.plugins.transformers_trainer import (
    QATTrainer,
    QuantizationArgumentsWithConfig,
)

# Training folder imports
from ltx_core.model_loader import load_transformer

# HuggingFace imports
from transformers import TrainingArguments

# SafeTensors imports
try:
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available. Install with: pip install safetensors")


# ============================================================
# DATASET LOADER (from qat_test11_safetensor.py)
# ============================================================

class SimplePrecomputedDataset(torch.utils.data.Dataset):
    """Load .precomputed data with video latents and text conditions."""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.latents_dir = self.data_root / "latents"
        self.conditions_dir = self.data_root / "conditions"

        self.latent_files = sorted(self.latents_dir.glob("*.pt"))

        if len(self.latent_files) == 0:
            raise ValueError(f"No .pt files found in {self.latents_dir}")

        print(f"Found {len(self.latent_files)} samples")

    def __len__(self):
        return len(self.latent_files)

    def _infer_shape_from_seq_len(self, seq_len: int):
        """Infer (T, H, W) from sequence length, trying common video dimensions."""
        for T in [16, 20, 23, 25, 32, 40, 46, 50]:
            remaining = seq_len / T
            if remaining == int(remaining):
                HW = int(remaining)
                H = int(HW ** 0.5)
                if H * H == HW:
                    return (T, H, H)
                for h in range(max(1, H - 2), H + 3):
                    w = HW // h
                    if h * w == HW:
                        return (T, h, w)

        HW_sqrt = int((seq_len / 16) ** 0.5)
        if HW_sqrt * HW_sqrt * 16 == seq_len:
            return (16, HW_sqrt, HW_sqrt)

        approx_dim = int(seq_len ** (1 / 3))
        return (approx_dim, approx_dim, approx_dim)

    def __getitem__(self, idx):
        latent_file = self.latent_files[idx]

        if latent_file.stem.startswith("latent_"):
            condition_name = f"condition_{latent_file.stem[7:]}.pt"
        else:
            condition_name = latent_file.name.replace("latent", "condition")

        condition_file = self.conditions_dir / condition_name

        latent_data = torch.load(latent_file, map_location="cpu", weights_only=False)
        condition_data = torch.load(condition_file, map_location="cpu", weights_only=False)

        # Extract video latents
        if isinstance(latent_data, dict):
            video_latents = latent_data.get("latents")
            if video_latents is None:
                video_latents = latent_data.get("video_latents")
            if video_latents is None:
                for v in latent_data.values():
                    if isinstance(v, torch.Tensor):
                        video_latents = v
                        break
        else:
            video_latents = latent_data

        # Extract text embeddings
        if isinstance(condition_data, dict):
            text_embeddings = condition_data.get("prompt_embeds")
            if text_embeddings is None:
                text_embeddings = condition_data.get("embeds")
            if text_embeddings is None:
                for v in condition_data.values():
                    if isinstance(v, torch.Tensor):
                        text_embeddings = v
                        break
        else:
            text_embeddings = condition_data

        # Ensure video_latents are UNBATCHED [C, T, H, W]
        latent_shape = None

        if video_latents.dim() == 5:
            if video_latents.shape[0] != 1:
                raise ValueError(f"Expected per-file batch size 1, got {video_latents.shape}")
            video_latents = video_latents.squeeze(0)
            C, T, H, W = video_latents.shape
            latent_shape = (T, H, W)
        elif video_latents.dim() == 4:
            C, T, H, W = video_latents.shape
            latent_shape = (T, H, W)
        elif video_latents.dim() == 3:
            C, T, HW = video_latents.shape
            H = W = int(HW ** 0.5)
            if H * W != HW:
                raise ValueError(f"Cannot infer H,W from video_latents shape {video_latents.shape}")
            video_latents = video_latents.view(C, T, H, W)
            C, T, H, W = video_latents.shape
            latent_shape = (T, H, W)
        elif video_latents.dim() == 2:
            seq_len, C_feat = video_latents.shape
            T, H, W = self._infer_shape_from_seq_len(seq_len)
            if T * H * W != seq_len:
                raise ValueError(f"Could not map seq_len={seq_len} to T,H,W")
            video_latents = video_latents.view(T, H, W, C_feat).permute(3, 0, 1, 2)
            C, T, H, W = video_latents.shape
            latent_shape = (T, H, W)
        else:
            raise ValueError(f"Unsupported video_latents shape: {video_latents.shape}")

        if latent_shape is None:
            raise ValueError(f"Could not determine latent_shape")

        # Ensure text embeddings are UNBATCHED [L, D]
        if text_embeddings.dim() == 3:
            if text_embeddings.shape[0] == 1:
                text_embeddings = text_embeddings.squeeze(0)
            else:
                raise ValueError(f"Unexpected text_embeddings shape {text_embeddings.shape}")
        elif text_embeddings.dim() == 2:
            pass
        else:
            raise ValueError(f"Unsupported text_embeddings shape: {text_embeddings.shape}")

        # Flow matching
        t = torch.rand(1).item()
        timesteps = torch.tensor([t * 1000.0])

        noise = torch.randn_like(video_latents)
        noisy_latents = (1 - t) * video_latents + t * noise
        targets = noise - video_latents

        return {
            "video_latents": noisy_latents,
            "prompt_embeds": text_embeddings,
            "timesteps": timesteps,
            "targets": targets,
            "latent_shape": latent_shape,
        }


# ============================================================
# MAIN
# ============================================================

def main(args):
    print("=" * 80)
    print("QAT for LTX-2 using QATTrainer (HuggingFace Trainer API)")
    print("=" * 80)
    print(f"\nCheckpoint:   {args.checkpoint}")
    print(f"Data:         {args.data_root}")
    print(f"Output:       {args.output_dir}\n")

    print("[1/3] Loading LTX-2 model...")
    
    model = load_transformer(
        checkpoint_or_state=args.checkpoint,
        device="cpu",
        dtype=torch.bfloat16,
    )
    
    print(f"✓ Loaded: {type(model).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B\n")

    print("[2/3] Loading dataset...")
    
    dataset = SimplePrecomputedDataset(args.data_root)
    
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Eval samples:  {len(eval_dataset)}\n")

    print("[3/3] Setting up QAT training...")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.qat_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.qat_steps // 3,
        save_strategy="steps",
        save_steps=args.qat_steps // 3,
        save_total_limit=2,
        logging_steps=args.log_interval,
        logging_dir=f"{args.output_dir}/logs",
        report_to="tensorboard",
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        load_best_model_at_end=False,
        warmup_steps=50,
        max_grad_norm=args.max_grad_norm,
    )
    
    quant_args = QuantizationArgumentsWithConfig(
        quant_cfg=INT8_DEFAULT_CFG,
        calib_size=args.calib_steps,
        compress=False,
    )
    
    print("✓ Training arguments configured")
    print(f"  Max steps: {args.qat_steps}")
    print(f"  Calibration size: {args.calib_steps}\n")

    print("Creating QATTrainer...")
    trainer = QATTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        quant_args=quant_args,
    )
    
    print("\n" + "=" * 80)
    print("Starting QAT Training...")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    print("\n" + "=" * 80)
    print("Saving final quantized model...")
    print("=" * 80)
    
    trainer.save_model(f"{args.output_dir}/final_model")
    
    print(f"\n✅ Training completed! Model saved to: {args.output_dir}/final_model")


def parse_args():
    parser = argparse.ArgumentParser(description="QAT for LTX-2 using QATTrainer")
    
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./qat_output")
    parser.add_argument("--calib-steps", type=int, default=128)
    parser.add_argument("--qat-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
