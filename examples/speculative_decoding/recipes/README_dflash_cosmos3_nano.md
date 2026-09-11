<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Training a DFlash Draft Model for Cosmos3 Nano

An end-to-end multimodal DFlash recipe: synthesize training data from image and
video prompts, train the draft model, export it, and smoke-test it under vLLM.

The executable pipeline is a ModelOpt Launcher example:

```bash
cd tools/launcher
uv run launch.py --yaml examples/nvidia/Cosmos3-Nano/hf_online_dflash_multimodal.yaml --yes
```

This document covers the reasoning behind that pipeline. For per-step
configuration, read the YAML — every task is commented inline.

## Why synthesize the training data

DFlash trains a draft model to predict what the *target* model will say. It
benefits far more from the target model's own completions than from a large
collection of human-written answers, because the draft's job is to match the
target's distribution, not to be independently correct.

So the pipeline does not consume an off-the-shelf SFT dataset. It takes prompts
from several sources, replays every prompt through the target model, and trains
on the target's responses. Each prompt is generated at several temperatures,
which widens coverage of the target's output distribution; the merge step then
removes near-duplicate completions.

## The data sources

Four complementary sources, three of which the pipeline builds itself:

1. **PAI-Understanding** — representative video usage.
2. **VQA v2** — image visual reasoning.
3. **Multilingual prompts** — high-quality text prompts
   (`nvidia/Speculative-Decoding-Multilingual-Prompt-v2`).
4. **Curated text** — optional, supplied by you.

The PAI and VQA sources keep their media; the other two are text-only. Mixing
text into a multimodal draft matters: a draft trained only on media prompts
degrades on the plain-text turns that dominate real conversations.

### Curated text (optional fourth source)

To build the Nemotron Chat component, follow
`recipes/train_eagle_head_cosmos_reason2.ipynb`: accept the Hugging Face licence
for `nvidia/Nemotron-Post-Training-Dataset-v2`, then run

```bash
python ../prepare_input_conversations/add_nemotron_chat.py --mapping-file nemotron_mapping.bin
```

which writes `input_conversations/nemotron-chat.jsonl` (89,511 conversations).
Point a `--source curated_text=<path>` at that file, or at any privacy-reviewed
JSONL of assistant-completed conversations in the same `messages`/`conversations`
format. To use both, concatenate them into one valid JSONL first.

## The merge step

Merging is not just concatenation. It does two things the training job depends on:

- **Resolves every image/video reference to an absolute path.** This is why
  training runs with `data.vlm_img_dir=/`.
- **Deduplicates conservatively.** Records are grouped by identical prompt and
  media, and only near-identical completions within a group are dropped. The
  temperature sweep intentionally produces varied answers to the same prompt;
  the goal is to remove redundancy, not diversity.

## Training constraints worth knowing

- `training_seq_len` must be divisible by `dflash_block_size`. The example uses
  `16384` and `8`.
- The `VLM_*` environment variables cap text and visual token growth *before*
  tokenization. They matter because visual token count depends on the actual
  resolution and frame count of each sample: without caps, one high-resolution
  video can expand past `training_seq_len`, which the collator rejects rather
  than silently truncating (a silent truncation would corrupt the DFlash labels).
- Setting `data.vlm_processor` is what selects the multimodal collator. Leave it
  unset and training uses the text-only path, even for a VLM target.

## Deployment

The pipeline's last task runs a vLLM smoke test against the exported draft.

Serve the original Cosmos3 Nano target with the exported draft. A DFlash block
size of eight yields seven speculative tokens — the remaining position is the
context/bonus token. The command below is a single-GPU smoke test: it binds to
localhost and caps the context at 4096 tokens to keep startup and memory
bounded. Raise `--max-model-len` only after sizing the context and concurrency
you need. Add `--trust-remote-code` only for a checkpoint you trust.

```bash
SERVED_MODEL_NAME=cosmos3-nano-dflash
vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 \
  --served-model-name "$SERVED_MODEL_NAME" \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$EXPORT_PATH\",\"num_speculative_tokens\":7}"

# In another terminal:
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: DFlash deployment smoke test passed.\"}],\"temperature\":0,\"max_tokens\":16}"
```
