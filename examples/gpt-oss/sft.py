# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copied and Adapted from https://github.com/huggingface/gpt-oss-recipes/blob/main/sft.py
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
accelerate launch \
    --config_file configs/zero3.yaml \
    sft.py \
    --config configs/sft_full.yaml \
    --model_name_or_path openai/gpt-oss-20b \
    --packing true packing_strategy wrapped \
    --run_name 20b-full-qat \
    --attn_implementation kernels-community/vllm-flash-attn3
    --quant_cfg MXFP4_MLP_WEIGHT_ONLY_CFG
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    # SFTTrainer, Use ModelOpt's version instead
    TrlParser,
)
from utils import (
    get_original_huggingface_quant_method,
    get_peft_config_for_moe,
    is_distributed_job,
    load_dataset_from_hub_or_local,
)

import modelopt.torch.opt as mto

# import ModelOpt's QATSFTTrainer instead of Huggingface TRL's SFTTrainer
from modelopt.torch.quantization.plugins import QATSFTTrainer, QuantizationArguments

# Enable automatic save/load of modelopt state huggingface checkpointing
mto.enable_huggingface_checkpointing()


def main(script_args, training_args, model_args, quant_args):
    # ------------------------
    # Load model & tokenizer
    # ------------------------
    model_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": getattr(model_args, "dtype", "float32"),
        "use_cache": not training_args.gradient_checkpointing,
    }

    if get_original_huggingface_quant_method(model_args.model_name_or_path) == "mxfp4":
        model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

    if not is_distributed_job():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    # --------------
    # Load dataset
    # --------------
    dataset = load_dataset_from_hub_or_local(script_args, training_args)

    # -------------
    # Train model
    # -------------
    # Use ModelOpt's QATSFTTrainer instead of Huggingface TRL's SFTTrainer
    trainer = QATSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        processing_class=tokenizer,
        peft_config=get_peft_config_for_moe(model, model_args),
        quant_args=quant_args,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, QuantizationArguments))
    script_args, training_args, model_args, quant_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, quant_args)
