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

"""Training entry point for Qwen3 model with DMS.

Usage:
    # First run prepares the dataset (single process):
    python -m models.qwen3.train --config configs/qwen3_8b.yaml --prepare-dataset-only

    # Then launch distributed training:
    accelerate launch -m models.qwen3.train --config configs/qwen3_8b.yaml
"""

import argparse
import functools
import json
import os
import shutil
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from dms.logging import get_logger
from dms.training.data import ConfiguredTokenizer, DataBlend, DataBlendElement
from dms.training.engine import (
    CombinedModel,
    DistillationModelArguments,
    DMSTrainerState,
    ModelArguments,
    ModifiedTrainer,
    distillation_forward,
    dms_schedule,
    get_student_model,
    get_teacher_model,
    get_tokenizer,
)
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    Qwen3ForCausalLM,
    TrainingArguments,
)

from .modeling_qwen3_dms import Qwen3ForCausalLMDMS

logger = get_logger("Train")


# =============================================================================
# Config loading
# =============================================================================


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, output_dir: str) -> None:
    """Save the configuration to the output directory for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved config to {config_path}")


def resolve_checkpoint(cfg: dict) -> str | None:
    """Resolve the checkpoint path for resume, supporting 'auto' detection."""
    resume = cfg["hf_trainer"].get("resume_from_checkpoint")
    if resume == "auto":
        output_dir = Path(cfg["hf_trainer"]["output_dir"])
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        if checkpoints:
            logger.info(f"Auto-detected latest checkpoint: {checkpoints[-1]}")
            return str(checkpoints[-1])
        else:
            logger.info("No checkpoints found, starting fresh.")
            return None
    return resume


# =============================================================================
# Dataset parsing
# =============================================================================


def _parse_data_blend_elements(blend_string: str) -> list[DataBlendElement]:
    """Parse data blend elements from a comma-separated string.

    Args:
        blend_string: Comma-separated string of "dataset_name:weight" pairs.

    Returns:
        List of DataBlendElement objects.
    """
    import dms.training.data as rf_datasets

    elements = []
    for entry in blend_string.split(","):
        dataset_name, weight_str = entry.split(":")
        dataset = getattr(rf_datasets, dataset_name)
        weight = float(weight_str)
        logger.info(f"Adding dataset {dataset_name} with weight {weight}")
        elements.append(DataBlendElement(dataset=dataset, weight=weight))
    return elements


def _create_train_dataset(
    data_cfg: dict,
    configured_tokenizer: ConfiguredTokenizer,
) -> Dataset:
    """Create the training dataset from data blend configuration."""
    data_blend_elements = _parse_data_blend_elements(data_cfg["blend"])

    data_blend = DataBlend(
        data_blend_elements=data_blend_elements,
        configured_tokenizer=configured_tokenizer,
        train_samples=data_cfg["train_samples"],
        concat_up_to=data_cfg["max_length"],
        concat_always_start_new=data_cfg.get("concat_always_start_new", True),
    )

    return Dataset.from_generator(lambda: (data_blend[i] for i in range(len(data_blend))))


# =============================================================================
# Model building
# =============================================================================


def build_combined_model(
    model_args: DistillationModelArguments,
    training_args: TrainingArguments,
    dms_cfg: dict,
    data_cfg: dict,
    tokenizer: PreTrainedTokenizer,
    trainer_state: DMSTrainerState,
) -> CombinedModel:
    """Build the combined student-teacher model for distillation."""
    dms_kwargs = {
        f"dms_{k}" if not k.startswith("dms_") else k: v
        for k, v in dms_cfg.items()
        if k not in ("initial_cr", "final_cr", "final_step")
    }

    student_model = get_student_model(
        model_args,
        zero_out_proj_alpha=True,
        model_constructor=Qwen3ForCausalLMDMS,
        dms_kwargs=dms_kwargs,
    )

    student_is_teacher = (
        model_args.student.model_name_or_path == model_args.teacher.model_name_or_path
        and model_args.student.dtype == model_args.teacher.dtype
    )
    if student_is_teacher:
        logger.info("Student and teacher are the same model - optimization enabled")
        teacher_model = student_model
    else:
        logger.info("Student and teacher are different models")
        teacher_model = get_teacher_model(model_args, model_constructor=Qwen3ForCausalLM)

    return CombinedModel(
        student_model=student_model,
        teacher_model=teacher_model,
        trainer_state=trainer_state,
        dms_schedule=functools.partial(
            dms_schedule,
            training_args=training_args,
            dms_initial_cr=dms_cfg["initial_cr"],
            dms_final_cr=dms_cfg["final_cr"],
            dms_final_step=dms_cfg.get("final_step"),
        ),
        forward_fn=distillation_forward,
        student_is_teacher=student_is_teacher,
        tokenizer=tokenizer,
        process_vocab_using_chunk=data_cfg.get("process_vocab_using_chunk", 4096),
        forward_fn_kwargs_student=model_args.student.forward_fn_kwargs,
        forward_fn_kwargs_teacher=model_args.teacher.forward_fn_kwargs,
    )


# =============================================================================
# Student model extraction
# =============================================================================

AUTO_MAP_CONFIG = {
    "AutoConfig": "configuration_qwen3_dms.Qwen3ConfigDMS",
    "AutoModel": "modeling_qwen3_dms.Qwen3ModelDMS",
    "AutoModelForCausalLM": "modeling_qwen3_dms.Qwen3ForCausalLMDMS",
    "AutoModelForQuestionAnswering": "modeling_qwen3_dms.Qwen3ForQuestionAnsweringDMS",
    "AutoModelForSequenceClassification": "modeling_qwen3_dms.Qwen3ForSequenceClassificationDMS",
    "AutoModelForTokenClassification": "modeling_qwen3_dms.Qwen3ForTokenClassificationDMS",
}


def extract_student_model(
    combined_model: CombinedModel,
    tokenizer: PreTrainedTokenizer,
    save_path: str,
) -> None:
    """Extract the student model from a CombinedModel and save it for inference.

    The saved model includes:
    - Model weights in bfloat16
    - Config with auto_map for trust_remote_code
    - Model implementation files (config.py, model.py)
    - Tokenizer

    Note: The saved model imports from the `dms` package. Make sure `dms` is
    installed (pip install -e .) in any environment where you load this model.
    """
    student_model = combined_model.student_model
    logger.info(f"Extracting student model to: {save_path}")

    student_model.to(torch.bfloat16)
    student_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Update config.json with auto_map
    config_path = Path(save_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    config.pop("architectures", None)
    config["auto_map"] = AUTO_MAP_CONFIG
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Copy model implementation files for trust_remote_code
    model_dir = Path(__file__).parent
    for src_name in ["configuration_qwen3_dms.py", "modeling_qwen3_dms.py"]:
        shutil.copy(model_dir / src_name, Path(save_path) / src_name)

    logger.info(f"Successfully saved student model to: {save_path}")


# =============================================================================
# Main entry point
# =============================================================================


def main() -> None:
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="Train DMS adapter for Qwen3")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--prepare-dataset-only",
        action="store_true",
        help="Only prepare the dataset, then exit (run with single process first)",
    )
    args, _unknown = parser.parse_known_args()

    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    dms_cfg = cfg["dms"]
    data_cfg = cfg["data"]
    hf_trainer_cfg = cfg["hf_trainer"]

    # Build model arguments
    model_args = DistillationModelArguments(
        student=ModelArguments(
            model_name_or_path=model_cfg["name"],
            dtype=model_cfg.get("dtype", "float32"),
            forward_fn_kwargs=model_cfg.get("forward_fn_kwargs", {}),
        ),
        teacher=ModelArguments(
            model_name_or_path=model_cfg.get("teacher_name", model_cfg["name"]),
            dtype=model_cfg.get("teacher_dtype", model_cfg.get("dtype", "float32")),
            forward_fn_kwargs=model_cfg.get(
                "teacher_forward_fn_kwargs", model_cfg.get("forward_fn_kwargs", {})
            ),
        ),
    )

    # Resolve checkpoint resume
    checkpoint_path = resolve_checkpoint(cfg)
    if checkpoint_path:
        hf_trainer_cfg["resume_from_checkpoint"] = checkpoint_path

    training_args = TrainingArguments(**hf_trainer_cfg)

    logger.info(f"\n--- Config ---\n{yaml.dump(cfg, default_flow_style=False)}")

    # Tokenizer
    tokenizer = get_tokenizer(model_args.student)
    tokenizer_kwargs = data_cfg.get("tokenizer_kwargs", {})
    configured_tokenizer = ConfiguredTokenizer(
        tokenizer=tokenizer,
        apply_chat_template_kwargs=tokenizer_kwargs,
        encode_kwargs={},
    )

    # Dataset
    train_dataset = _create_train_dataset(data_cfg, configured_tokenizer)

    if args.prepare_dataset_only:
        logger.info("Dataset preparation complete. Exiting (--prepare-dataset-only).")
        return

    # Save config for reproducibility
    save_config(cfg, training_args.output_dir)

    # Build model
    trainer_state = DMSTrainerState()
    combined_model = build_combined_model(
        model_args=model_args,
        training_args=training_args,
        dms_cfg=dms_cfg,
        data_cfg=data_cfg,
        tokenizer=tokenizer,
        trainer_state=trainer_state,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=data_cfg["max_length"],
        return_tensors="pt",
    )

    trainer = ModifiedTrainer(
        trainer_state=trainer_state,
        model=combined_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer_state.set_trainer(trainer)

    # Train
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()

    # Auto-save student model at end of training
    student_model_path = os.path.join(training_args.output_dir, "student_model")
    extract_student_model(combined_model, tokenizer, student_model_path)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
