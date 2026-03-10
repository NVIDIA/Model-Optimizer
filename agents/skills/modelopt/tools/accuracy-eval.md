# Tool: Accuracy Evaluation with nel

## Inputs (provided by caller)

- `MODEL_PATH`: Absolute path to the quantized model checkpoint
- `MODEL_NAME`: Short name for the model (e.g., `Qwen3-0.6B_nvfp4`)
- `GPU_IDS`: Comma-separated GPU indices
- `PORT`: vLLM server port (default: `8199`)
- `EVAL_TASKS`: Comma-separated eval tasks (default: `mmlu`)
- `VLLM_ALREADY_RUNNING`: Whether vLLM is already serving on `PORT` (default: `false`)

## Procedure

### 1. Ensure vLLM server is running

If `VLLM_ALREADY_RUNNING` is `true`, verify:

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health
```

If 200, skip to step 2. If not, start the server:

```bash
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --port $PORT \
  --max-model-len 4096 \
  --trust-remote-code \
  2>&1 | tee /tmp/vllm_server_eval.log &
VLLM_PID=$!
```

Wait for readiness (poll `http://localhost:$PORT/health` every 3s, timeout 5 min).

### 2. Check nel is installed

```bash
nel --version
```

If not found: `pip install nemo-evaluator-launcher`

### 3. Generate nel config

```bash
EVAL_DIR=$(mktemp -d)
mkdir -p "$EVAL_DIR/results"
```

Write `$EVAL_DIR/eval_config.yaml` with the following content (substitute actual values):

```yaml
target:
  api_endpoint:
    url: http://localhost:PORT/v1
    model_id: MODEL_NAME
    api_key_name: DUMMY_API_KEY

evaluation:
  nemo_evaluator_config:
    config:
      params:
        request_timeout: 3600
        parallelism: 16
        extra:
          tokenizer: MODEL_PATH
          tokenizer_backend: huggingface
  tasks:
    - name: lm-evaluation-harness.mmlu

execution:
  output_dir: EVAL_DIR/results
```

For each task in `EVAL_TASKS` beyond `mmlu`, add additional task entries:

- `arc_challenge` -> `lm-evaluation-harness.arc_challenge`
- `hellaswag` -> `lm-evaluation-harness.hellaswag`
- `gpqa` -> `lm-evaluation-harness.gpqa`
- `commonsense_qa` -> `lm-evaluation-harness.commonsense_qa`

Replace `PORT`, `MODEL_NAME`, `MODEL_PATH`, and `EVAL_DIR` with the actual values (not shell variables -- write literal values into the YAML).

### 4. Run evaluation

```bash
export DUMMY_API_KEY=dummy
nel run --config "$EVAL_DIR/eval_config.yaml" 2>&1 | tee "$EVAL_DIR/nel_output.log"
```

This may take 10-60+ minutes depending on tasks and model size. Monitor output for progress.

### 5. Parse results

```bash
python -c "
import json, glob, os

results_dir = '$EVAL_DIR/results'
scores = {}

for f in glob.glob(os.path.join(results_dir, '**', '*.json'), recursive=True):
    try:
        data = json.load(open(f))
        if isinstance(data, dict) and 'results' in data:
            for task, metrics in data['results'].items():
                acc = metrics.get('acc,none') or metrics.get('acc_norm,none')
                if acc is not None:
                    scores[task] = round(acc, 4)
        elif isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, (int, float)) and 0 <= val <= 1:
                    scores[key] = round(val, 4)
    except:
        pass

if scores:
    print('========================================')
    print('  ACCURACY RESULTS')
    print('========================================')
    for task, score in sorted(scores.items()):
        print(f'  {task:<25} {score:.4f}')
    print('========================================')
else:
    print('No accuracy scores found.')
    print('Check nel output: $EVAL_DIR/nel_output.log')
"
```

### 6. Cleanup

Kill the vLLM server if we started it:

```bash
if [ -n "$VLLM_PID" ]; then
  kill $VLLM_PID 2>/dev/null
  wait $VLLM_PID 2>/dev/null
  echo "vLLM server stopped"
fi
```

## Error Recovery

| Error | Fix |
|-------|-----|
| `nel: command not found` | `pip install nemo-evaluator-launcher` |
| `Connection refused` | vLLM server is not running. Start it (step 1). |
| `Timeout` on requests | Increase `request_timeout` to 7200 or reduce `parallelism` to 8 |
| `Task not found` | Run `nel ls tasks` to see available task names |
| nel hangs | Check vLLM server health. Kill nel, restart. |

## Output

- Accuracy scores per task
- Report results to caller for combined presentation
