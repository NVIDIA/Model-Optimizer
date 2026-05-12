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

import gc
import json
import os
import random
import time
from argparse import Namespace

import numpy as np
import onnxruntime as rt
import onnxruntime_genai as og
import pandas as pd
from fire import Fire
from modeling import EvalModel, TrtllmPipeline, select_model
from quantization_utils import MAX_SEQ_LEN, get_tokenizer, quantize_model
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import tensorrt_llm
    from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner
    from trtllm_utils import load_tokenizer, read_model_name

    if PYTHON_BINDINGS:
        from tensorrt_llm.runtime import ModelRunnerCpp
except ImportError:
    tensorrt_llm = None
    PYTHON_BINDINGS = None
    ModelRunner = None
    ModelRunnerCpp = None

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


def evaluate_trtllm(args, subject, pipeline, dev_df, test_df):
    rank = tensorrt_llm.mpi_rank()
    cors = []
    all_probs = []
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while not pipeline.check_valid_length(prompt) and k > 0:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]
        pred = pipeline(prompt)

        if rank == 0:
            probs = [0 for _ in get_choices()]
            cor = pred.strip().startswith(label)
            cors.append(cor)
            all_probs.append(probs)

    if rank == 0:
        acc = np.mean(cors)
        cors = np.array(cors)

        all_probs = np.array(all_probs)
        print(f"Average accuracy {acc:.3f} - {subject}")

        return cors, acc, all_probs
    else:
        return None, 0, None


def evaluate_genai_dml(args, subject, model, dev_df, test_df, model_path):
    cors = []
    all_probs = []
    tokenizer = og.Tokenizer(model)
    search_options = {}

    # disable sampling
    search_options["do_sample"] = False
    search_options["temperature"] = 0.0

    for i in range(test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_tokens = tokenizer.encode(prompt)
        params = og.GeneratorParams(model)

        # params.input_ids = input_tokens

        if len(input_tokens) + 2 > args.max_seq_length:
            print(
                f"Warning: Seq length {len(input_tokens) + 2} exceeds max_seq_length {args.max_seq_length}"
            )
            print(f"Skipping example {i}")
            continue

        # print(f"Processing example {i} with seq length {len(input_tokens) + 2}")
        search_options["max_length"] = args.max_seq_length
        search_options["min_length"] = len(input_tokens) + 2

        params.set_search_options(**search_options)
        generator = og.Generator(model, params)

        generator.append_tokens(input_tokens)

        new_tokens = []
        # generator.compute_logits()
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        new_tokens.append(new_token)
        if not generator.is_done():
            # generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            new_tokens.append(new_token)

        pred = tokenizer.decode(new_tokens)
        del params
        del generator

        label = test_df.iloc[i, test_df.shape[1] - 1]

        probs = [0 for _ in get_choices()]
        cor = pred.strip().startswith(label)
        cors.append(cor)
        all_probs.append(probs)

        gc.collect()  # Collect garbage after inference

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print(f"Average accuracy {acc:.3f} - {subject}")

    return cors, acc, all_probs


def evaluate_pt(args, subject, model: EvalModel, dev_df, test_df):
    cors = []
    all_probs = []

    for i in range(test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        def check_valid_length(model, prompt):
            if isinstance(model, EvalModel):
                return model.check_valid_length(prompt)
            else:
                return len(model.tokenizer.encode(prompt)) <= model.max_input_len

        while not check_valid_length(model, prompt) and k > 0:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]
        if isinstance(model, EvalModel):
            pred = model.run(prompt)

        probs = [0 for _ in get_choices()]
        cor = pred.strip().startswith(label)
        cors.append(cor)
        all_probs.append(probs)

        gc.collect()

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    print(f"Average accuracy {acc:.3f} - {subject}")

    return cors, acc, all_probs


def forward_ort_native(sess, input_feed, max_length, tokenizer, num_layers):
    # Record the length of the original input
    original_input_length = input_feed["input_ids"].shape[1]

    # Initialize the list to store generated IDs
    generated_ids = input_feed["input_ids"].tolist()[0]
    # max_length = 30
    for step in range(max_length):
        try:
            outputs = sess.run(None, input_feed)
        except Exception as e:
            print(f"Error during sess.run(): {e}")
            return None

        next_token_logits = outputs[0][:, -1, :]
        next_token_id = int(np.argmax(next_token_logits, axis=-1).item())
        generated_ids.append(next_token_id)

        input_feed["input_ids"] = np.array([[next_token_id]], dtype=np.int64)
        input_feed["position_ids"] = np.array([[len(generated_ids) - 1]], dtype=np.int64)
        input_feed["attention_mask"] = np.ones((1, len(generated_ids)), dtype=np.int64)

        for i in range(num_layers):
            input_feed[f"past_key_values.{i}.key"] = outputs[i * 2 + 1].astype(np.float32)
            input_feed[f"past_key_values.{i}.value"] = outputs[i * 2 + 2].astype(np.float32)

        if next_token_id == tokenizer.eos_token_id:
            break

    # Only decode the tokens that were generated after the original input
    new_generated_ids = generated_ids[original_input_length:]
    generated_text = tokenizer.decode(new_generated_ids, skip_special_tokens=True)
    return generated_text


def evaluate_ort_native(args, subject, sess, tokenizer, dev_df, test_df, config):
    cors = []
    all_probs = []

    num_layers = config["num_hidden_layers"]

    for i in range(test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        input_ids = tokenizer.encode(prompt, return_tensors="np").astype(np.int64)
        # print(f"Context length {input_ids.shape[1]}")
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[np.newaxis, :]

        hidden_size = config["hidden_size"]
        num_heads = config["num_key_value_heads"]
        head_dim = hidden_size // config["num_attention_heads"]

        past_key_values = {
            f"past_key_values.{i}.key": np.zeros((1, num_heads, 0, head_dim), dtype=np.float32)
            for i in range(num_layers)
        }
        past_key_values.update(
            {
                f"past_key_values.{i}.value": np.zeros(
                    (1, num_heads, 0, head_dim), dtype=np.float32
                )
                for i in range(num_layers)
            }
        )

        input_feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **past_key_values,
        }

        pred = forward_ort_native(sess, input_feed, 2, tokenizer, num_layers)
        if pred is None:
            continue

        label = test_df.iloc[i, test_df.shape[1] - 1]

        probs = [0 for _ in get_choices()]
        cor = pred.strip().startswith(label)
        cors.append(cor)
        all_probs.append(probs)

        gc.collect()

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    print(f"Average accuracy {acc:.3f} - {subject}")

    return cors, acc, all_probs


def save_results_to_json(results, output_file="results.json"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


def main(
    data_dir: str = "data/mmlu",
    ntrain: int = 0,
    quant_cfg: str | None = None,
    batch_size: int = 0,
    calib_size: int = 512,
    dtype: str = "bfloat16",
    output_file: str = "results/results.json",
    subject: str | tuple | None = None,
    ep: str = "pt",
    max_seq_length: int = 4096,
    trust_remote_code: bool = False,
    **kwargs,
):
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    args = Namespace(**locals())
    print(locals())

    data_fullpath = os.path.join(data_dir, "test")

    if subject is None:
        subjects = sorted(
            [f.split("_test.csv")[0] for f in os.listdir(data_fullpath) if "_test.csv" in f]
        )
    else:
        if isinstance(subject, tuple):
            subject = ",".join(subject)
        subjects = subject.split(",")

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in get_subcategories().values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in get_categories()}

    results = {
        "subject_accuracies": {},
        "subcategory_accuracies": {},
        "category_accuracies": {},
        "subject_details": {},
    }

    if ep == "genai_dml":
        # ORT DML Model Loading
        model = og.Model(kwargs["model_path"])

        def evaluate_func(args, subject, dev_df, test_df):
            return evaluate_genai_dml(
                args, subject, model, dev_df, test_df, model_path=kwargs["model_path"]
            )

    elif ep.startswith("ort_"):
        backend = ep.split("_")[1]

        if backend not in ["cpu", "dml", "cuda"]:
            raise ValueError(
                f"Invalid backend '{backend}'. Allowed values are 'cpu', 'dml', or 'cuda'."
            )

        # Set the execution provider based on the backend
        if backend == "cpu":
            providers = ["CPUExecutionProvider"]
        elif backend == "dml":
            providers = ["DmlExecutionProvider"]
        elif backend == "cuda":
            providers = ["CUDAExecutionProvider"]

        onnx_model_path = kwargs["model_path"]

        # Create the InferenceSession with the selected provider
        sess = rt.InferenceSession(os.path.join(onnx_model_path, "model.onnx"), providers=providers)

        with open(os.path.join(onnx_model_path, "config.json")) as config_file:
            config = json.load(config_file)

        tokenizer = AutoTokenizer.from_pretrained(onnx_model_path, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token

        def evaluate_func(args, subject, dev_df, test_df):
            return evaluate_ort_native(args, subject, sess, tokenizer, dev_df, test_df, config)

    elif ep == "pt":
        model_ckpt_path = kwargs["model_path"]
        tokenizer = get_tokenizer(model_ckpt_path, trust_remote_code=trust_remote_code)

        model = select_model(
            max_input_length=MAX_SEQ_LEN,
            max_output_length=2,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        assert isinstance(model, EvalModel)
        if quant_cfg:
            model.load()
            quantize_model(
                model=model,
                quant_cfg=quant_cfg,
                tokenizer=tokenizer,
                batch_size=batch_size,
                calib_size=calib_size,
            )

        # Pass all required arguments to evaluate_pt
        def evaluate_func(args, subject, dev_df, test_df):
            return evaluate_pt(args, subject, model, dev_df, test_df)

    # evaluate trtllm models
    elif ep == "trt-llm":  # for trt-llm
        engine_dir = kwargs["engine_dir"]
        tokenizer_dir = kwargs["hf_model_dir"]
        runtime_rank = tensorrt_llm.mpi_rank()
        model_name, model_version = read_model_name(engine_dir)

        tokenizer, pad_id, end_id = load_tokenizer(
            tokenizer_dir=tokenizer_dir,
            vocab_file=None,
            model_name=model_name,
            model_version=model_version,
            trust_remote_code=trust_remote_code,
        )

        runner_cls = ModelRunner if not PYTHON_BINDINGS else ModelRunnerCpp
        runner_kwargs = {}
        if PYTHON_BINDINGS:
            runner_kwargs.update(max_beam_width=1)

        # run with default settings for now
        runner_kwargs.update(
            max_tokens_in_paged_kv_cache=None,  # args.max_tokens_in_paged_kv_cache,
            kv_cache_enable_block_reuse=False,  # args.kv_cache_enable_block_reuse,
            kv_cache_free_gpu_memory_fraction=0.9,  # args.kv_cache_free_gpu_memory_fraction,
            enable_chunked_context=False,  # args.enable_chunked_context,
            multi_block_mode=False,
        )  # args.multi_block_mode)
        model = runner_cls.from_dir(engine_dir, rank=runtime_rank, **runner_kwargs)

        max_attention_window_size = None  # Run with default settings
        pipeline = TrtllmPipeline(
            tokenizer, model, model_name, pad_id, end_id, max_attention_window_size
        )

        def evaluate_func(args, subject, dev_df, test_df):
            return evaluate_trtllm(args, subject, pipeline, dev_df, test_df)

    for subject in tqdm(subjects):
        start_time = time.time()  # Start timing
        dev_df = pd.read_csv(os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None)[
            :ntrain
        ]
        test_df = pd.read_csv(os.path.join(data_dir, "test", subject + "_test.csv"), header=None)

        cors, acc, probs = evaluate_func(args, subject, dev_df, test_df)

        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Time taken for {subject}: {elapsed_time:.2f} seconds")

        results["subject_accuracies"][subject] = acc
        results["subject_details"][subject] = {
            "accuracy": acc,
            "correctness": cors.tolist(),
            "time_taken_seconds": elapsed_time,  # Add time taken to results
        }

        subcats = get_subcategories()[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in get_categories():
                if subcat in get_categories()[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

    for subcat in subcat_cors:
        if subcat_cors[subcat]:
            subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
            results["subcategory_accuracies"][subcat] = subcat_acc

    for cat in cat_cors:
        if cat_cors[cat]:
            cat_acc = np.mean(np.concatenate(cat_cors[cat]))
            results["category_accuracies"][cat] = cat_acc

    if all_cors:
        weighted_acc = np.mean(np.concatenate(all_cors))
        print(f"Overall average accuracy: {weighted_acc:.3f}")
    else:
        weighted_acc = None

    ordered_results = {
        "subject_accuracies": results["subject_accuracies"],
        "overall_accuracy": weighted_acc,
        "category_accuracies": results["category_accuracies"],
        "subcategory_accuracies": results["subcategory_accuracies"],
        "subject_details": results["subject_details"],
    }
    save_results_to_json(ordered_results, output_file)
    return weighted_acc


if __name__ == "__main__":
    Fire(main)
