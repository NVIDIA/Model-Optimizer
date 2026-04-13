# Adapted from https://github.com/tatsu-lab/stanford_alpaca/blob/3783d18/train.py

#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

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

import argparse
import os
from dataclasses import dataclass, field
from typing import Literal

import torch
import transformers
from accelerate import ParallelismConfig
from eagle_utils import (
    EagleTrainerWithAccLog,
    EagleTrainingPlot,
    make_eagle_supervised_data_module,
    patch_ring_attention_for_ttt,
)
from omegaconf import OmegaConf
from transformers.trainer_utils import get_last_checkpoint

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import EagleConfig
from modelopt.torch.speculative.utils import load_vlm_or_llm, patch_transformers5_params_loading
from modelopt.torch.utils import print_rank_0

torch.manual_seed(0)
mto.enable_huggingface_checkpointing()


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={"help": "HuggingFace model ID or local path to the base model."},
    )
    use_fake_base_for_offline: bool = field(
        default=False,
        metadata={
            "help": "Load model architecture without real base weights. Offline training only."
        },
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code when loading model."}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the online training data."},
    )
    offline_data_path: str = field(
        default=None,
        metadata={
            "help": "Path to offline training data directory (.pt files). This argument enables offline mode.",
        },
    )
    lazy_preprocess: bool = True
    draft_vocab_cache: str | None = field(
        default=None,
        metadata={"help": "Path to draft vocabulary cache file."},
    )
    vlm_img_dir: str = field(default=None, metadata={"help": "Path to the VLM image directory."})
    vlm_processor: str = field(default=None, metadata={"help": "Path to the VLM processor."})
    sample_size: int = field(
        default=-1,
        metadata={"help": "Number of samples to use for training. Use -1 to use all samples."},
    )

    def __post_init__(self):
        if self.sample_size == 0 or self.sample_size < -1:
            raise ValueError("sample_size must be -1 (use all samples) or a positive integer")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    training_seq_len: int = field(
        default=2048,
        metadata={
            "help": (
                "Training sequence length. Sequences will be right padded or truncated to this length."
            )
        },
    )
    mode: Literal["eagle3", "medusa"] = "eagle3"
    estimate_ar: bool = field(
        default=False, metadata={"help": "Whether to estimate AR using training accuracy to log."}
    )
    ar_validate_steps: int = field(default=1000, metadata={"help": "AR validation interval."})
    cp_size: int = field(default=1, metadata={"help": "Context parallelism size."})
    dp_shard_size: int | None = field(
        default=None,
        metadata={"help": "Data parallelism shard size. None = auto (total_gpu / cp_size)."},
    )


@dataclass
class MedusaArguments:
    medusa_num_heads: int | None = field(default=1)
    medusa_num_layers: int | None = field(default=1)


def _parse_cli() -> tuple[str, list[str]]:
    """Parse --config (required) from argv; return remaining args as config overrides.

    Extra arguments use OmegaConf dotlist syntax, e.g.
    ``model.model_name_or_path=meta-llama/Llama-3.2-1B training.output_dir=ckpts/test``.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", required=True, help="Path to the YAML config file.")
    args, overrides = p.parse_known_args()
    return args.config, overrides


def _load_config(config_path: str, overrides: list[str] = ()) -> tuple[dict, dict]:
    """Load training config from a YAML file with sections: model, data, training, eagle.

    *overrides* are OmegaConf dotlist entries (e.g. ``["model.model_name_or_path=xxx"]``)
    applied on top of the YAML.

    Returns:
        hf_cfg: Flat dict from model/data/training sections, for HfArgumentParser.parse_dict()
        eagle_cfg: Eagle section dict (EagleConfig fields), passed directly to mtsp.convert()
    """
    merged = OmegaConf.load(config_path)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    cfg = OmegaConf.to_container(merged, resolve=True)

    # Eagle section maps directly to EagleConfig fields — no field enumeration needed.
    # eagle_architecture_config is a nested dict and is included as-is.
    eagle_cfg = cfg.get("eagle", {})

    hf_cfg = {
        **cfg.get("model", {}),
        **cfg.get("data", {}),
        **cfg.get("training", {}),
    }

    if hf_cfg.get("dp_shard_size") is None:
        cp_size = hf_cfg.get("cp_size", 1)
        hf_cfg["dp_shard_size"] = torch.cuda.device_count() // cp_size

    return hf_cfg, eagle_cfg


def train():
    config_path, overrides = _parse_cli()
    hf_cfg, eagle_cfg = _load_config(config_path, overrides)

    parser = transformers.HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            MedusaArguments,
        )
    )
    model_args, data_args, training_args, medusa_args = parser.parse_dict(
        hf_cfg, allow_extra_keys=True
    )

    if not data_args.data_path and not data_args.offline_data_path:
        raise ValueError(
            "Either data.data_path or data.offline_data_path must be set in the config."
        )
    if training_args.cp_size > 1 or training_args.dp_shard_size > 1:
        training_args.parallelism_config = ParallelismConfig(
            cp_size=training_args.cp_size, dp_shard_size=training_args.dp_shard_size
        )
    if training_args.cp_size > 1:
        patch_ring_attention_for_ttt()
        # Specific patch to accelerate 1.12.0. Removable after move to 1.13.0
        training_args.parallelism_config.sp_backend = None
    print_rank_0(f"arguments: {model_args}, {training_args}, {medusa_args}, eagle_cfg={eagle_cfg}")

    # Detect checkpoint to resume from
    last_checkpoint = (
        get_last_checkpoint(training_args.output_dir)
        if os.path.isdir(training_args.output_dir)
        else None
    )
    if last_checkpoint:
        print_rank_0(f"Last checkpoint detected: {last_checkpoint}")

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint

    use_offline_training = data_args.offline_data_path is not None

    if checkpoint:
        with patch_transformers5_params_loading():
            model = load_vlm_or_llm(
                checkpoint, dtype="auto", trust_remote_code=model_args.trust_remote_code
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=model_args.trust_remote_code
        )
    else:
        # To avoid OOM for large models, we load and convert model on CPU first.
        # Model will be moved to GPU during HF trainer.init().
        if use_offline_training:
            # Load config first to preserve original num_hidden_layers before
            # load_vlm_or_llm may reduce layers for offline space savings.
            model_config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
            )
        model = load_vlm_or_llm(
            model_args.model_name_or_path,
            use_fake_base=model_args.use_fake_base_for_offline,
            use_offline_training=use_offline_training,
            dtype="auto",
            device_map="cpu",
            trust_remote_code=model_args.trust_remote_code,
        )
        if use_offline_training:
            # When doing offline training, we need to set num_hidden_layers
            # since we override it when loading the model for space savings.
            # Some models (e.g. Kimi-K2.5) use non-standard config attributes,
            # so fall back to the model's own config if the attribute is missing.
            model.config.num_orig_hidden_layers = getattr(
                model_config, "num_hidden_layers", model.config.num_hidden_layers
            )
            if hasattr(model.config, "layer_types"):
                del (
                    model.config.layer_types
                )  # remove layer_types to avoid mismatch with the modified model
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.training_seq_len,
            trust_remote_code=model_args.trust_remote_code,
        )
        if training_args.mode == "medusa":
            config = {
                "medusa_num_heads": medusa_args.medusa_num_heads,
                "medusa_num_layers": medusa_args.medusa_num_layers,
            }
            mtsp.convert(model, [("medusa", config)])
        elif training_args.mode == "eagle3":
            # Validate and rewrite eagle config fields
            eagle_cfg = EagleConfig.model_validate(
                eagle_cfg,
                context={"training_args": training_args, "data_args": data_args},
            ).model_dump()
            mtsp.convert(model, [("eagle", eagle_cfg)])

            # Load draft vocab cache if the draft model uses a compressed vocabulary
            if model.eagle_config.draft_vocab_size < model.eagle_config.vocab_size:
                if not os.path.isfile(data_args.draft_vocab_cache):
                    raise FileNotFoundError(
                        f"Draft vocab cache provided but not found: {data_args.draft_vocab_cache}"
                    )
                model.eagle_module.d2t = torch.load(data_args.draft_vocab_cache, weights_only=True)
                print_rank_0(f"Loaded draft vocab cache from {data_args.draft_vocab_cache}.")
        else:
            raise Exception(f"{training_args.mode} is not supported!")

    print_rank_0("Loading dataset...")
    if training_args.mode == "eagle3":
        data_module = make_eagle_supervised_data_module(
            tokenizer, data_args, train_len=training_args.training_seq_len
        )

    trainer = EagleTrainerWithAccLog(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[EagleTrainingPlot(training_args.ar_validate_steps, training_args.estimate_ar)],
        **data_module,
    )

    # Manually enable this to return loss in eval
    trainer.can_return_loss = True
    # Make sure label_smoother is None
    assert trainer.label_smoother is None, (
        "label_smoother is not supported in speculative decoding!"
    )

    print_rank_0("Start training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
