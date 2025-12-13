# Adapted from https://github.com/declare-lab/instruct-eval/blob/720e66f627369266ed1cfd74426666ec37e524bc/mmlu.py

# MIT License
#
# Copyright (c) 2020 Dan Hendrycks
# Copyright (c) 2023 Deep Cognition and Language Research (DeCLaRe) Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

import os
import random
import warnings
from argparse import Namespace

import numpy as np
import pandas as pd
from fire import Fire
from modeling import EvalModel, select_model
from quantization_utils import MAX_SEQ_LEN, get_tokenizer, quantize_model
from sparse_attention_utils import sparsify_model
from tqdm import tqdm

try:
    from modelopt.deploy.llm import LLM
except ImportError:
    LLM = None  # type: ignore[misc]
import modelopt.torch.opt as mto
from modelopt.torch.quantization.utils import is_quantized
from modelopt.torch.sparsity.attention_sparsity.conversion import is_attn_sparsified

os.environ["TOKENIZERS_PARALLELISM"] = "false"

RAND_SEED = 1234


def get_choices():
    return ["A", "B", "C", "D"]


def get_subcategories():
    return {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }


def get_categories():
    return {
        "STEM": [
            "physics",
            "chemistry",
            "biology",
            "computer science",
            "math",
            "engineering",
        ],
        "humanities": ["history", "philosophy", "law"],
        "social sciences": [
            "politics",
            "culture",
            "economics",
            "geography",
            "psychology",
        ],
        "other (business, health, misc.)": ["other", "business", "health"],
    }


def format_subject(subject):
    line = subject.split("_")
    s = ""
    for entry in line:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += f"\n{get_choices()[j]}. {df.iloc[idx, j + 1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.iloc[idx, k + 1]}\n\n"
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def evaluate(args, subject, model: EvalModel | LLM, dev_df, test_df):
    cors = []
    all_probs = []
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        def check_valid_length(model, prompt):
            if isinstance(model, EvalModel):
                return model.check_valid_length(prompt)
            else:
                return len(model.tokenizer.encode(prompt)) < model.max_seq_len

        while not check_valid_length(model, prompt) and k > 0:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]
        if isinstance(model, EvalModel):
            pred = model.run(prompt)
        elif isinstance(model, LLM):
            pred = model.generate_text([prompt], 2)[0]

        probs = [0 for _ in get_choices()]
        cor = pred.strip().startswith(label)
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print(f"Average accuracy {acc:.3f} - {subject}")

    return cors, acc, all_probs


def main(
    data_dir: str = "data/mmlu",
    ntrain: int = 5,
    quant_cfg: str | None = None,
    auto_quantize_bits: float | None = None,
    batch_size: int = 0,
    calib_size: int = 512,
    dtype: str = "bfloat16",
    auto_quantize_method: str = "gradient",
    auto_quantize_score_size: int = 128,
    auto_quantize_checkpoint: str | None = None,
    sparse_cfg: str | None = None,
    **kwargs,
):
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    args = Namespace(**locals())
    print(locals())

    data_fullpath = os.path.join(data_dir, "test")

    subjects = sorted(
        [f.split("_test.csv")[0] for f in os.listdir(data_fullpath) if "_test.csv" in f]
    )

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in get_subcategories().values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in get_categories()}

    # Model Optimizer modification
    # Enable automatic save/load of modelopt state huggingface checkpointing
    mto.enable_huggingface_checkpointing()
    model_path = kwargs["model_path"]
    tokenizer = get_tokenizer(model_path, trust_remote_code=kwargs.get("trust_remote_code", False))
    if kwargs.get("checkpoint_dir"):
        assert LLM is not None, "tensorrt_llm APIs could not be imported."
        medusa_choices = kwargs.get("medusa_choices")
        model = LLM(
            checkpoint_dir=kwargs["checkpoint_dir"],
            tokenizer=tokenizer,
            medusa_choices=medusa_choices,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=1,
        )
    else:
        # Force eager attention if sparse attention is requested
        if sparse_cfg:
            kwargs["attn_implementation"] = "eager"
            warnings.warn(
                "Sparse attention requires attn_implementation='eager'. "
                "Forcing eager attention implementation."
            )

        model = select_model(
            max_input_length=MAX_SEQ_LEN, max_output_length=2, dtype=dtype, **kwargs
        )
        assert isinstance(model, EvalModel)
        if quant_cfg:
            model.load()

            if is_quantized(model.model):
                # Do not quantize the model if model is already quantized
                warnings.warn("Skipping quantization: model is already quantized.")
            else:
                quantize_model(
                    model=model,
                    quant_cfg=quant_cfg,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    calib_size=calib_size,
                    auto_quantize_bits=auto_quantize_bits,
                    auto_quantize_method=auto_quantize_method,
                    auto_quantize_score_size=auto_quantize_score_size,
                    auto_quantize_checkpoint=auto_quantize_checkpoint,
                )

        # Apply sparse attention if requested
        if sparse_cfg:
            model.load()

            if is_attn_sparsified(model.model):
                warnings.warn(
                    "Skipping sparse attention: model already has sparse attention applied."
                )
            else:
                sparsify_model(
                    model=model,
                    sparse_cfg=sparse_cfg,
                )

    for subject in tqdm(subjects):
        dev_df = pd.read_csv(os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None)[
            :ntrain
        ]
        test_df = pd.read_csv(os.path.join(data_dir, "test", subject + "_test.csv"), header=None)

        cors, acc, probs = evaluate(args, subject, model, dev_df, test_df)
        subcats = get_subcategories()[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in get_categories():
                if subcat in get_categories()[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print(f"Average accuracy {subcat_acc:.3f} - {subcat}")

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print(f"Average accuracy {cat_acc:.3f} - {cat}")

    weighted_acc = np.mean(np.concatenate(all_cors))
    print(f"Average accuracy: {weighted_acc:.3f}")
    return weighted_acc


if __name__ == "__main__":
    Fire(main)
