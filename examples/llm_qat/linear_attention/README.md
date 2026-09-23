# Linear-attention decode study

Run `quality_study.py` with a pinned checkpoint, tokenizer, and disjoint tokenized
train/evaluation data. `decode_study_plan.json` records the existing public KDA
model and dataset protocol. Use `--prefill-tokens` to select the exact prefix;
training and evaluation score the suffix. These examples require the dependencies
in `requirements.txt` and a CUDA device.

The `configs/kda_decode_state_int8.json` and
`configs/kda_decode_replay_int8.json` recipes enable signed narrow-range INT8
state or replay anchors. Replay factors retain their separate FP8 setting.
Existing FP8 recipes remain available. No INT8 quality result is claimed.

`benchmark_decode.py` measures matched-policy training overhead.
`compare_quality.py` compares completed paired receipts with confidence bounds.
See [the decode guide](../../../docs/linear_attention_decode.md) for the numerical
contract, phase context, checkpoint behavior, and current limitations.

Delivery order is decode, prefill operand matmuls, then approximate inverse.
This branch contains exact prefix support only; serving through vLLM is pending.
