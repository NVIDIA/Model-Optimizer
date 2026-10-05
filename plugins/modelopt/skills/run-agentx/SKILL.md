---
name: run-agentx
description: Run the AgentX agentic serving benchmark with the SemiAnalysis harness. Use for AgentX setup, dataset downloads, concurrency sweeps, or serving latency and throughput comparisons.
license: Apache-2.0
---

# Run AgentX

## Setup

Resolve the endpoint URL, served model name, matching tokenizer, and requested
concurrency from the task. Reuse the model checkpoint. If serving is needed, follow
[deployment](../deployment/SKILL.md) to download missing weights and launch it.
Enable prefix caching, streaming usage, and cached-token reporting. For vLLM, use
`--enable-prefix-caching --enable-prompt-tokens-details`. Verify the context limit
fits the traces with the model's tokenizer; never silently truncate requests.

Install the [AgentX fork](https://github.com/SemiAnalysisAI/agentx-harness) in a
separate client environment. The pinned revision supplies the scenario and dataset
loader used below; installing the upstream `aiperf` package is insufficient.

```bash
set -euo pipefail
export AGENTX_WORKDIR="${AGENTX_WORKDIR:-$PWD/agentx-work}"
python3.12 -m venv "$AGENTX_WORKDIR/venv"
source "$AGENTX_WORKDIR/venv/bin/activate"
python -m pip install 'aiperf @ git+https://github.com/SemiAnalysisAI/agentx-harness.git@56a0cf70f4c0359454ee4bd15a17770b541a3e3e'
```

Set `AGENTX_DATASET` to an accepted, date-pinned with-subagents alias from the
[pinned AgentX tutorial](https://github.com/SemiAnalysisAI/agentx-harness/blob/56a0cf70f4c0359454ee4bd15a17770b541a3e3e/docs/tutorials/agentx-mvp.md).
The public loader downloads and caches the traces and prompt reconstruction assets.
Record the resolved dataset revision; dated aliases can move. Prefer the deployed
checkpoint's local tokenizer. If it requires custom code, add
`--tokenizer-trust-remote-code` only after the user explicitly trusts its repository.

## Choose concurrency

Sweep concurrency upward until KV-cache usage nears capacity, then refine around
that point. Resweep when the model, hardware, or cache budget changes. Publish the
full sweep and repeat promising points before claiming an advantage.

## Run

In each shell, set `AGENTX_WORKDIR` to the setup directory and set `AGENTX_DATASET`,
`AGENTX_URL`, `AGENTX_MODEL`, `AGENTX_TOKENIZER`, `AGENTX_MAX_CONTEXT_LENGTH`,
and `AGENTX_CONCURRENCY` from the deployment and sweep. Use the server's base URL
and actual context limit. Fix `AGENTX_SEED` across comparisons.
First check endpoint health and repeat a long prompt to verify nonzero cached-token
usage. Keep the scenario's default trajectory window and duration. Run each
concurrency separately with a fresh artifact directory:

```bash
set -euo pipefail
source "${AGENTX_WORKDIR:?}/venv/bin/activate"
export HF_HOME="${HF_HOME:-$AGENTX_WORKDIR/hf-cache}"
export AIPERF_DATASET_MMAP_CACHE_DIR="$AGENTX_WORKDIR/dataset-cache"
mkdir -p "$AGENTX_WORKDIR/results"
AGENTX_RUN_DIR=$(mktemp -d "$AGENTX_WORKDIR/results/run-XXXXXX")
python -m pip freeze > "$AGENTX_RUN_DIR/client-packages.txt"
aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url "${AGENTX_URL:?}" --model "${AGENTX_MODEL:?}" \
  --tokenizer "${AGENTX_TOKENIZER:?}" --endpoint-type chat \
  --public-dataset "${AGENTX_DATASET:?}" \
  --max-context-length "${AGENTX_MAX_CONTEXT_LENGTH:?}" \
  --concurrency "${AGENTX_CONCURRENCY:?}" \
  --random-seed "${AGENTX_SEED:?}" \
  --streaming --use-server-token-count --extra-inputs ignore_eos:true \
  --artifact-dir "$AGENTX_RUN_DIR" --ui simple \
  2>&1 | tee "$AGENTX_RUN_DIR/client.log"
```

## Compare and report

Use the [benchmarking guide](../deployment/references/benchmarking.md#3-run-a-sweep)
for sweep isolation, performance metrics, and comparison controls. For AgentX:

- Concurrency counts session trees. Report actual overlapping HTTP requests
  separately because child sessions can overlap.
- Hold KV-cache bytes fixed when comparing KV formats. Keep the dataset, seed,
  context filter, and warmup fixed, and start a fresh server per point.
- Report token-weighted cache hits from server-reported usage, cache usage, and
  preemptions. Missing usage means cache hits are unknown.
- Report `metadata.submission_valid` and any `metadata.submission_invalid_reasons`
  from the result export. Flag invalid or missing validity instead of treating the
  run as a valid comparison.
- Include request errors, unfinished requests, and warmup failures.
  Synthetic prompts measure serving performance, not model quality; shortened
  smoke runs are not benchmark results.
