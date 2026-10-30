# Offline AR validation for DFlash drafts

Measures a DFlash draft's acceptance length (AR) without running the target. The target's
greedy replies and hidden states are dumped once; `offline_ar_validate.py` then replays greedy
speculative decoding on them, so every draft checkpoint is evaluated in minutes on one GPU and
all of them on the same references.

Run the commands from `examples/speculative_decoding/`.

## 1. Generate the target's greedy replies

Serve the target with vLLM (or any OpenAI-compatible server) and generate at temperature 0:

```bash
python scripts/server_generate.py --data_path prompts.jsonl --output_path replies.jsonl \
    --url http://localhost:8000/v1 --model <served-name> --temperature 0 --max_tokens 1024 \
    --extra_body '{"skip_special_tokens": false}'
```

Keep special tokens: step 2 re-tokenizes the replies, and a reply stripped of its special
tokens (e.g. a reasoning channel's markers) no longer matches the tokens the target produced.
Add the chat-template arguments the deployment uses, e.g.
`"chat_template_kwargs": {"enable_thinking": true}`.

## 2. Dump the target's hidden states on them

```bash
python collect_hidden_states/compute_hidden_states_hf.py --model <target> \
    --input-data replies.jsonl --output-dir dumps/ --answer-only-loss \
    --aux-layers <the draft's target_layer_ids, comma-separated>
```

`--aux-layers` must list the draft's `target_layer_ids` in order. `--answer-only-loss` marks the
replies; pass `--chat-template` if the target's template has no `{% generation %}` tags. The
vLLM and TensorRT-LLM dumpers in `collect_hidden_states/` write the same format.

## 3. Replay

```bash
python offline_ar/offline_ar_validate.py --model_path <DFlash checkpoint> --dump_dir dumps/
```

It prints the AR, defined as vLLM's 1 + accepted / drafts pooled over all steps, and the target
greedy agreement: the share of reply tokens that are the target's own greedy pick, read from the
dumped final hidden states. Faithful greedy references give about 0.99 (vLLM and the dumper
disagree on near-ties); below 0.98 the script warns that the AR is biased. `--output` writes
per-reply step counts as JSON, and `--num_speculative_tokens` verifies fewer drafts per step.

Only the DFlash block is replayed; drafts that refine it further (e.g. DSpark) are refused.

## Validation

Nemotron-3.5-Bolt-E4B with its DFlash draft (block 8), first turns of MT-Bench (80 prompts,
thinking on, up to 1024 tokens), one RTX PRO 6000:

| References | AR | Greedy agreement | Time |
|---|---|---|---|
| vLLM engine, online | 3.746 | | |
| Exact greedy token ids | 3.741 | 0.993 | 2 min |
| Steps 1-2 | 3.718 | 0.994 | 2 min |
| Steps 1-2 with special tokens stripped | 3.689 | 0.967 | 2 min |

Steps 1-2 produce different but equally greedy replies, since vLLM's greedy decoding depends on
the batch; the AR moves with the replies, not with the replay.
