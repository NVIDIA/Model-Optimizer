# Tokenizing for Megatron Frameworks

| **Section** | **Description** | **Link** |
| :---: | :---: | :---: |
| From JSONL files | Tokenize local JSONL files | \[[Link](#from-jsonl-files)\] |
| From Hugging Face Hub | Stream or download HF datasets and tokenize | \[[Link](#from-hugging-face-hub)\] |
| `reasoning_content` for Post-Training v3 | Control how chain-of-thought traces are handled | \[[Link](#reasoning_content-for-post-training-v3-datasets)\] |
| Nemotron Pre/Post-Training Datasets | Ready-to-run commands for all Nemotron datasets | \[[Link](#ready-to-run-tokenization-commands)\] |

The distillation and pre-training scripts in Megatron-Bridge or Megatron-LM expect data pre-tokenized in Megatron's binary indexed format (`.bin` / `.idx`).
Use the `megatron_preprocess_data` utility to tokenize any JSONL or Hugging Face dataset.
The tokenization scripts below print the list of output prefixes (e.g. `tokenized_qwen3/data1_text`) that you can use for the `data_paths` argument (with relative weights on different files) in Megatron training scripts.

**Important Notes:**

- For Pretraining / raw-text data (`text` key) — use `--append_eod` so Megatron can tell where documents end when concatenating them into long sequences.
- For Post-training chat data (`messages` key) — omit `--append_eod`; the chat template already appends EOS at the end of each conversation.
- Set `--max_sequence_length 256_000` to avoid rare OOM errors if some text is very long.

## From JSONL files

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --jsonl_paths /path/to/data1.jsonl /path/to/data2.jsonl ... \
    --json_keys text \
    --tokenizer Qwen/Qwen3-0.6B \
    --output_dir tokenized_qwen3 \
    --workers 32 \
    --append_eod
```

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --jsonl_paths /path/to/sft_data.jsonl \
    --json_keys messages \
    --tokenizer Qwen/Qwen3-0.6B \
    --output_dir tokenized_qwen3 \
    --workers 32
```

Instead of `--jsonl_paths`, pass `--input_dir /path/to/dir` to tokenize all JSONL files in a directory (`.jsonl` and `.jsonl.gz` are both supported).

## From Hugging Face Hub

To tokenize a dataset directly from Hugging Face Hub:

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --hf_dataset nvidia/Nemotron-Pretraining-SFT-v1 \
    --hf_name Nemotron-SFT-Code \
    --hf_split train \
    --hf_max_samples_per_split 10_000_000 \
    --json_keys text \
    --tokenizer Qwen/Qwen3-0.6B \
    --output_dir tokenized_qwen3 \
    --workers 32 \
    --append_eod
```

Omit `--hf_name` to process all subsets, `--hf_split` for all splits, or `--hf_max_samples_per_split` for all samples.
To quickly test, use [nvidia/Nemotron-Pretraining-Dataset-sample](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample).

For very large datasets (tens of millions of documents), or datasets with complex nested message schemas (e.g. `tool_calls`, `function_call` fields) that cause Arrow type-cast errors in non-streaming mode, add `--hf_streaming` to avoid downloading the full dataset — only the rows actually consumed are fetched. Optionally pair with `--hf_max_samples_per_split <num_samples>` to cap the row count; without it streaming still works but re-downloads on every run with no disk cache.

> **Performance note:** Non-streaming mode downloads all Parquet shards once and caches them as Arrow files on disk.
> Re-runs read from cache and are much faster.
> Streaming re-downloads on every run with no cache, so it is slower for full-dataset processing.

## `reasoning_content` for Post-Training v3 Datasets

v3 datasets include a `reasoning_content` field in assistant messages (chain-of-thought separate from
the final answer). Use `--reasoning_content` to control how it is handled:

| Value | Behaviour |
| --- | --- |
| `strip` (default) | Field is discarded before `apply_chat_template`. Safe for any tokenizer. |
| `inline` | Wrapped as `<think>…</think>` and prepended to `content`. Preserves reasoning in a tokenizer-agnostic way. |
| `native` | Passed unchanged. Requires the tokenizer's chat template to handle the field (e.g. Qwen3). |

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --hf_dataset nvidia/Nemotron-Math-v2 \
    --hf_split high_part00 \
    --json_keys messages \
    --tokenizer nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --output_dir tokenized_nemotron_v2 \
    --workers 32 \
    --reasoning_content inline
```

---

## Ready-to-run tokenization commands

Tokenization commands for all Nemotron Pre-Training and Post-Training datasets used in Megatron-Bridge distillation experiments.

Two parameters vary by model — set them before running the commands below:

```bash
TOKENIZER=nvidia/NVIDIA-Nemotron-Nano-9B-v2        # HuggingFace tokenizer (or local path)
OUTPUT_DIR=tokenized_nemotron_v2                   # Output directory for tokenized files
```

> [!TIP]
> Token count for a `.bin` file = file size in bytes ÷ 4. This is also printed by the tokenization script on completion.

> [!NOTE]
> Tokenizing each of the datasets below will take anywhere between 10 minutes to few hours. You can tokenize all in parallel to speed up the process.
>
> You may tokenize more datasets or skip some datasets depending on your needs.

### Nemotron Pretraining dataset

**[nvidia/Nemotron-Pretraining-SFT-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1)** — raw text; omitting `--hf_name` tokenizes all 3 subsets (Code, General, MATH) in one command, producing a separate output file per subset named after each:

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
  --hf_dataset nvidia/Nemotron-Pretraining-SFT-v1 \
  --hf_split train \
  --hf_streaming \
  --hf_max_samples_per_split 10_000_000 \
  --json_keys text \
  --tokenizer ${TOKENIZER} \
  --output_dir ${OUTPUT_DIR} \
  --workers 96 \
  --max_sequence_length 256_000 \
  --append_eod \
  --strip_newlines
```

---

### Nemotron Post-training v1 dataset

**[nvidia/Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1)** — STEM subset, capped at 5M samples. v1 data does not contain reasoning traces:

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
  --hf_dataset nvidia/Nemotron-Post-Training-Dataset-v1 \
  --hf_name default \
  --hf_split stem \
  --hf_streaming \
  --hf_max_samples_per_split 5_000_000 \
  --json_keys messages \
  --tokenizer ${TOKENIZER} \
  --output_dir ${OUTPUT_DIR} \
  --workers 96 \
  --max_sequence_length 256_000
```

---

### Nemotron Post-training v3 collection

Datasets below are from the [Nemotron Post-Training v3 collection](https://huggingface.co/collections/nvidia/nemotron-post-training-v3). All use `--reasoning_content inline` to preserve `<think>…</think>` traces. The collection contains many more datasets — if you care about benchmarks not covered here (e.g. multilingual, agentic/tool use, SWE, safety), pick the relevant datasets from the collection and tokenize them the same way.

**[nvidia/Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2)** — tokenize `high_part00` and `high_part01` separately:

```bash
for SPLIT in high_part00 high_part01; do
  python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --hf_dataset nvidia/Nemotron-Math-v2 \
    --hf_split ${SPLIT} \
    --json_keys messages \
    --tokenizer ${TOKENIZER} \
    --output_dir ${OUTPUT_DIR} \
    --workers 96 \
    --max_sequence_length 256_000 \
    --reasoning_content inline
done
```

**[nvidia/Nemotron-SFT-Competitive-Programming-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2)** — stored as raw JSONL on HuggingFace, download before tokenizing:

```bash
hf download nvidia/Nemotron-SFT-Competitive-Programming-v2 \
    --repo-type dataset \
    --local-dir datasets/Nemotron-SFT-Competitive-Programming-v2/
for FILE in competitive_programming_python_00 competitive_programming_cpp_00; do
  python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --jsonl_paths datasets/Nemotron-SFT-Competitive-Programming-v2/data/${FILE}.jsonl \
    --json_keys messages \
    --tokenizer ${TOKENIZER} \
    --output_dir ${OUTPUT_DIR} \
    --workers 96 \
    --max_sequence_length 256_000 \
    --reasoning_content inline
done
```

**[nvidia/Nemotron-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1)** — stored as raw JSONL on HuggingFace, download before tokenizing:

```bash
hf download nvidia/Nemotron-Science-v1 \
    --repo-type dataset \
    --local-dir datasets/Nemotron-Science-v1/
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --input_dir datasets/Nemotron-Science-v1/data/ \
    --json_keys messages \
    --tokenizer ${TOKENIZER} \
    --output_dir ${OUTPUT_DIR} \
    --workers 96 \
    --max_sequence_length 256_000 \
    --reasoning_content inline
```

**[nvidia/Nemotron-SFT-Instruction-Following-Chat-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2)** — stored as raw JSONL on HuggingFace, download before tokenizing:

```bash
hf download nvidia/Nemotron-SFT-Instruction-Following-Chat-v2 \
    --repo-type dataset \
    --local-dir datasets/Nemotron-SFT-Instruction-Following-Chat-v2/
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --input_dir datasets/Nemotron-SFT-Instruction-Following-Chat-v2/data/ \
    --json_keys messages \
    --tokenizer ${TOKENIZER} \
    --output_dir ${OUTPUT_DIR} \
    --workers 96 \
    --max_sequence_length 256_000 \
    --reasoning_content inline
```

---

### Expected output

After running all commands above, `${OUTPUT_DIR}/` should contain the following `.bin` / `.idx` file pairs:

```text
nvidia--Nemotron-Pretraining-SFT-v1_Nemotron-SFT-Code_train_text_max10000000.{bin,idx}
nvidia--Nemotron-Pretraining-SFT-v1_Nemotron-SFT-General_train_text_max10000000.{bin,idx}
nvidia--Nemotron-Pretraining-SFT-v1_Nemotron-SFT-MATH_train_text_max10000000.{bin,idx}
nvidia--Nemotron-Post-Training-Dataset-v1_default_stem_messages_max5000000.{bin,idx}
nvidia--Nemotron-Math-v2_default_high_part00_messages.{bin,idx}
nvidia--Nemotron-Math-v2_default_high_part01_messages.{bin,idx}
competitive_programming_python_00_messages.{bin,idx}
competitive_programming_cpp_00_messages.{bin,idx}
MCQ_messages.{bin,idx}
RQA_messages.{bin,idx}
reasoning_off_messages.{bin,idx}
reasoning_on_messages.{bin,idx}
```
