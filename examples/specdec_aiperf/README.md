# Speculative Decoding Benchmark with AIPerf

`bench.py` measures the acceptance length and throughput of a speculative-decoding model served by vLLM,
using [AIPerf](https://github.com/ai-dynamo/aiperf) as the client. One running server can be measured at
several temperatures and concurrencies without restarting it, which makes temperature sweeps and
throughput-vs-concurrency curves cheap.

Results use the file names, fields and acceptance-length definition of [specdec_bench](../specdec_bench/), so
the two can be compared directly:

- `acceptance_rate.json`: `Average_AL`, `Category_AL`, `Request_AL`, and the acceptance-length histogram
  with conditional and joint acceptance rates.
- `timing.json`: `Output TPS`, `Output TPS/gpu`, time to first token, per-request decode speed and latency.
- `mtbench_responses.jsonl` or `specbench_responses.jsonl`: the generated replies.
- `configuration.json`: the arguments, served model, versions and the AIPerf command.
- `aiperf/`: AIPerf's own exports, including its per-request and token-weighted acceptance lengths.

Acceptance length comes from the server's per-request speculative-decoding statistics, which needs vLLM
v0.29.0 or later started with `--per-request-spec-decode-metrics summary`. As in specdec_bench, a request
counts the prefill token as one step, pools the steps of all its turns, and `Average_AL` is the mean over
requests.

## Install

Install AIPerf and this script's dependencies in their own virtual environment:

```bash
python3 -m venv aiperf-venv
aiperf-venv/bin/pip install -r requirements.txt
```

## Run

Start the server with per-request spec-decode metrics, for example a DFlash drafter:

```bash
vllm serve <target> --tensor-parallel-size 4 --per-request-spec-decode-metrics summary \
    --speculative-config '{"method": "dflash", "model": "<draft>", "num_speculative_tokens": 7}'
```

Then benchmark it. Each temperature × concurrency is written to `<save_dir>/t<T>_c<C>/`:

```bash
aiperf-venv/bin/python bench.py --url http://localhost:8000 \
    --mtbench question.jsonl --runtime_params runtime.yaml \
    --temperature 0 1 --concurrency 1 8 32 --output_length 4096 --num_gpus 4 \
    --save_dir results/mtbench
```

`--dataset speed --dataset_path <split>` reads a SPEED-Bench split prepared by specdec_bench's
`prepare_data.py`, and `--dataset specbench` reads Spec-Bench questions. Every point uses the same
requests, so acceptance lengths across concurrencies are comparable. `--runtime_params` takes the same
YAML as specdec_bench: its `sampling_kwargs` and `chat_template_args` are sent with every request.

Before the first run, `bench.py` sends one short request and stops if the response carries no
spec-decode stats. A point that fails is reported and skipped, the remaining points still run, and the
script exits non-zero.

To match a specdec_bench run:

- Sampling (`temperature`, `top_p`, `top_k`) is always sent explicitly, because `vllm serve` otherwise
  fills unset fields from the model's `generation_config.json`.
- Turn 2 replays the turn-1 reply with special tokens kept, as specdec_bench does.
- For temperature > 0 with DFlash or DSpark, add `"draft_sample_method": "probabilistic"` to the
  speculative config.
- `vllm serve` chooses its own default `--max-num-batched-tokens`, which can differ from specdec_bench's
  in-process engine and changes the KV cache size. This moves time to first token and throughput, not
  acceptance length; set it explicitly when comparing throughput.
- For DFlash, keep `--max-num-seqs` × (1 + speculative tokens) within `--max-num-batched-tokens`, or
  vLLM fails at startup.

`tools/launcher/common/specdec_aiperf/run.sh` runs the server and the benchmark as one launcher task; see
`tools/launcher/examples/Qwen/Qwen3-8B/specdec_aiperf_dflash.yaml`. It installs AIPerf into `AIPERF_VENV`
when that is missing; point it at a shared directory so the installation happens once.

To add a dataset, write a reader that returns its rows and register it in `DATASETS` in `bench.py`.
