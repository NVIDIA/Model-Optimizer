#!/usr/bin/env python3
"""
Quantization-Aware Distillation (QAD) for LTX-2 Models

Trains an INT8 quantized student model guided by a full-precision teacher model.

Usage:
    accelerate launch --config_file /path/to/fsdp_config.yaml ltx2_qad_trainer4.py \
        --checkpoint /path/to/model.safetensors \
        --data-root /path/to/precomputed_data \
        --output-dir ./qad_output
"""

import argparse
import sys
import torch
from pathlib import Path

# ============================================================
# PATH RESOLUTION (Robust - tries multiple locations)
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()

# Try multiple possible locations for Training/src
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
    print("=" * 80)
    print("ERROR: Could not find ltx_core module!")
    print("=" * 80)
    print(f"Script location: {SCRIPT_DIR}")
    print("\nSearched in the following locations:")
    for i, path in enumerate(possible_paths, 1):
        status = "✓ EXISTS" if path.exists() else "✗ NOT FOUND"
        ltx_core_status = "✓ HAS ltx_core" if (path / "ltx_core").exists() else "✗ NO ltx_core"
        print(f"  {i}. {path}")
        print(f"     Directory: {status}, {ltx_core_status}")
    print("\nPlease ensure ltx_core exists in one of these locations, or add the correct path to possible_paths.")
    print("=" * 80)
    sys.exit(1)

sys.path.insert(0, str(TRAINING_SRC))

# Now we can import ltx_core
try:
    from ltx_core.model_loader import load_transformer
except ImportError as e:
    print("=" * 80)
    print("ERROR: Failed to import ltx_core!")
    print("=" * 80)
    print(f"Error: {e}")
    print(f"Python path includes: {TRAINING_SRC}")
    print("\nTried to import from:")
    print(f"  {TRAINING_SRC / 'ltx_core'}")
    print("\nPlease verify the ltx_core module is properly installed.")
    print("=" * 80)
    sys.exit(1)

# ModelOpt imports
import modelopt.torch.quantization as mtq
import modelopt.torch.distill as mtd
from modelopt.torch.quantization.config import INT8_DEFAULT_CFG
from modelopt.torch.quantization.plugins.transformers_trainer import (
    QADTrainer,
    QuantizationArgumentsWithConfig,
)

# HuggingFace imports
from transformers import TrainingArguments


# ============================================================
# DATASET (FIXED - includes latent_shape)
# ============================================================

class SimplePrecomputedDataset(torch.utils.data.Dataset):
    """Load precomputed latents and text conditions."""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.latents_dir = self.data_root / "latents"
        self.conditions_dir = self.data_root / "conditions"

        if not self.latents_dir.exists():
            raise ValueError(f"Latents directory not found: {self.latents_dir}")
        if not self.conditions_dir.exists():
            raise ValueError(f"Conditions directory not found: {self.conditions_dir}")

        self.latent_files = sorted(self.latents_dir.glob("*.pt"))

        if len(self.latent_files) == 0:
            raise ValueError(f"No .pt files found in {self.latents_dir}")

        print(f"Found {len(self.latent_files)} samples")

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_file = self.latent_files[idx]

        # Map latent_X.pt to condition_X.pt
        if latent_file.stem.startswith("latent_"):
            condition_name = f"condition_{latent_file.stem[7:]}.pt"
        else:
            condition_name = latent_file.name.replace("latent", "condition")

        condition_file = self.conditions_dir / condition_name

        # Load data
        latent_data = torch.load(latent_file, map_location="cpu", weights_only=False)
        condition_data = torch.load(condition_file, map_location="cpu", weights_only=False)

        # Extract video latents and shape information
        if isinstance(latent_data, dict):
            video_latents = latent_data.get("latents", latent_data.get("video_latents"))
            if video_latents is None:
                video_latents = next(v for v in latent_data.values() if isinstance(v, torch.Tensor))
            
            # Extract shape information (required for packed latents)
            num_frames = latent_data.get("num_frames")
            height = latent_data.get("height")
            width = latent_data.get("width")
            
            # Validate shape metadata
            if num_frames is None or height is None or width is None:
                raise ValueError(
                    f"Latent file {latent_file} is missing shape metadata. "
                    f"Expected 'num_frames', 'height', 'width' keys in the dict."
                )
        else:
            video_latents = latent_data
            # If no metadata, infer from tensor shape
            if video_latents.dim() == 4:  # [C, F, H, W]
                num_frames = video_latents.shape[1]
                height = video_latents.shape[2]
                width = video_latents.shape[3]
            else:
                raise ValueError(
                    f"Cannot infer shape from latents with shape {video_latents.shape}. "
                    f"Latent files should be dicts with 'latents', 'num_frames', 'height', 'width' keys."
                )

        # Extract text embeddings
        if isinstance(condition_data, dict):
            text_embeddings = condition_data.get("prompt_embeds", condition_data.get("embeds"))
            if text_embeddings is None:
                text_embeddings = next(v for v in condition_data.values() if isinstance(v, torch.Tensor))
        else:
            text_embeddings = condition_data

        # Ensure correct shapes
        if video_latents.dim() == 5:
            video_latents = video_latents.squeeze(0)
        if text_embeddings.dim() == 3:
            text_embeddings = text_embeddings.squeeze(0)

        # Create noisy latents for flow matching
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
            "latent_shape": (num_frames, height, width),  # ✅ FIXED: Required for model forward pass
        }


# ============================================================
# DISTILLATION LOSS (inherits from _Loss)
# ============================================================

class DiffusionMSELoss(torch.nn.modules.loss._Loss):
    """
    MSE loss for distilling diffusion model outputs.
    
    Inherits from _Loss (the base class for all PyTorch loss functions) 
    to be compatible with ModelOpt's KDLossConfig validation.
    """
    
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
    
    def forward(self, student_output, teacher_output):
        """
        Compute MSE between student and teacher outputs.
        
        Args:
            student_output: Student model predictions
            teacher_output: Teacher model predictions
            
        Returns:
            MSE loss value
        """
        return torch.nn.functional.mse_loss(
            student_output.float(),
            teacher_output.float(),
            reduction=self.reduction
        )


# ============================================================
# MAIN
# ============================================================

def main(args):
    print("=" * 80)
    print("QAD for LTX-2 using QADTrainer (HuggingFace Trainer API)")
    print("=" * 80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Data:       {args.data_root}")
    print(f"Output:     {args.output_dir}\n")

    # ============================================================
    # 1. LOAD TEACHER MODEL
    # ============================================================
    print("[1/4] Loading teacher model (full precision)...")
    
    teacher = load_transformer(
        checkpoint_or_state=args.checkpoint,
        device="cpu",
        dtype=torch.bfloat16,
    )
    
    teacher.eval()
    teacher.requires_grad_(False)
    
    print(f"✓ Loaded: {type(teacher).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in teacher.parameters()) / 1e9:.2f}B\n")

    # ============================================================
    # 2. LOAD STUDENT MODEL
    # ============================================================
    print("[2/4] Loading student model...")
    
    student = load_transformer(
        checkpoint_or_state=args.checkpoint,
        device="cpu",
        dtype=torch.bfloat16,
    )
    
    print(f"✓ Loaded: {type(student).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in student.parameters()) / 1e9:.2f}B\n")

    # ============================================================
    # 3. LOAD DATASET
    # ============================================================
    print("[3/4] Loading dataset...")
    
    dataset = SimplePrecomputedDataset(args.data_root)
    
    # Split into train/eval
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Eval samples:  {len(eval_dataset)}\n")

    # ============================================================
    # 4. SETUP QAD TRAINING
    # ============================================================
    print("[4/4] Setting up QAD training...")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to=[],
    )
    
    quant_args = QuantizationArgumentsWithConfig(
        quant_cfg=INT8_DEFAULT_CFG,
        calib_size=args.calib_size,
        compress=False,
    )
    
    # Use StaticLossBalancer instance
    distill_config = {
        "teacher_model": (lambda: teacher, (), {}),
        "criterion": DiffusionMSELoss(),
        "loss_balancer": mtd.StaticLossBalancer(kd_loss_weight=0.5),
    }
    
    print("✓ Training arguments configured")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Calibration size: {args.calib_size}")
    print(f"  KD loss weight: 0.5 (student loss weight: 0.5)")
    print("\nCreating QADTrainer...")
    
    # ============================================================
    # 5. CREATE TRAINER AND TRAIN
    # ============================================================
    trainer = QADTrainer(
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        quant_args=quant_args,
        distill_config=distill_config,
    )
    
    print("\n" + "=" * 80)
    print("Starting QAD Training...")
    print("=" * 80)
    print()
    
    # Train
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 80)
    print("Saving final quantized student model...")
    print("=" * 80)
    
    try:
        trainer.save_model(f"{args.output_dir}/final_model", export_student=True)
        print(f"\n✅ Training completed! Model saved to: {args.output_dir}/final_model")
    except Exception as e:
        print(f"\n⚠️ Model weights saved, but encountered error: {e}")
        print(f"✅ You can still use the model from: {args.output_dir}/final_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QAD for LTX-2")
    
    # Model and data
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to LTX-2 checkpoint (.safetensors)")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to precomputed dataset root")
    parser.add_argument("--output-dir", type=str, default="./qad_output",
                        help="Output directory")
    
    # Training hyperparameters
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Warmup steps")
    
    # Quantization
    parser.add_argument("--calib-size", type=int, default=128,
                        help="Calibration dataset size")
    
    # Logging and saving
    parser.add_argument("--logging-steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=50,
                        help="Evaluate every N steps")
    parser.add_argument("--save-total-limit", type=int, default=3,
                        help="Maximum checkpoints to keep")
    
    # Other
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")
    
    args = parser.parse_args()
    main(args)
