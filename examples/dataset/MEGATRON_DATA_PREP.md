# Data Preparation

Tokenization commands for the Nemotron Pre-Training and Post-Training dataset collections used in megatron-bridge distillation experiments.

Two parameters vary by model — set them before running the commands below:

```bash
TOKENIZER=nvidia/NVIDIA-Nemotron-Nano-9B-v2   # HuggingFace tokenizer (or local path)
OUTPUT_DIR=tokenized_nano_v2                   # Output directory for tokenized files
```

Output files are written in Megatron binary format (`.bin` / `.idx`). See [examples/dataset/README.md](../dataset/README.md) for full tokenization documentation.

> [!TIP]
> Token count for a `.bin` file = file size in bytes ÷ 4. This is also printed by the tokenization script on completion.

> Tokenizing each of the datasets below will take anywhere between 10 minutes to 1 hour. You can tokenize all in parallel to speed up the process.

---

## Nemotron Pretraining dataset

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

## Nemotron Post-training v1 dataset

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

## Nemotron Post-training v3 collection

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

**[nvidia/Nemotron-SFT-Math-v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v3)**:

```bash
python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --hf_dataset nvidia/Nemotron-SFT-Math-v3 \
    --hf_name default \
    --hf_split train \
    --json_keys messages \
    --tokenizer ${TOKENIZER} \
    --output_dir ${OUTPUT_DIR} \
    --workers 96 \
    --max_sequence_length 256_000 \
    --reasoning_content inline
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

## Expected output

After running all commands above, `${OUTPUT_DIR}/` should contain the following `.bin` / `.idx` file pairs:

```text
nvidia--Nemotron-Pretraining-SFT-v1_Nemotron-SFT-Code_train_text_max10000000.{bin,idx}
nvidia--Nemotron-Pretraining-SFT-v1_Nemotron-SFT-General_train_text_max10000000.{bin,idx}
nvidia--Nemotron-Pretraining-SFT-v1_Nemotron-SFT-MATH_train_text_max10000000.{bin,idx}
nvidia--Nemotron-Post-Training-Dataset-v1_default_stem_messages_max5000000.{bin,idx}
nvidia--Nemotron-Math-v2_default_high_part00_messages.{bin,idx}
nvidia--Nemotron-Math-v2_default_high_part01_messages.{bin,idx}
nvidia--Nemotron-SFT-Math-v3_default_train_messages.{bin,idx}
competitive_programming_python_00_messages.{bin,idx}
competitive_programming_cpp_00_messages.{bin,idx}
MCQ_messages.{bin,idx}
RQA_messages.{bin,idx}
reasoning_off_messages.{bin,idx}
reasoning_on_messages.{bin,idx}
```
