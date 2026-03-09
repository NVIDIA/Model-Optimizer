# Tool: vLLM Deploy + Speed Benchmark

## Inputs (provided by caller)

- `MODEL_PATH`: Absolute path to the quantized model checkpoint
- `MODEL_NAME`: Short name for the model (e.g., `Qwen3-0.6B_nvfp4`)
- `GPU_IDS`: Comma-separated GPU indices (e.g., `0`)
- `PORT`: vLLM server port (default: `8199`)
- `TP_SIZE`: Tensor parallel size (default: `1`)
- `KEEP_SERVER`: Whether to leave the server running after benchmark (default: `false`)

## Part A: Deploy and Verify

### 1. Start vLLM server

Run in the background:

```bash
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --port $PORT \
  --max-model-len 4096 \
  --trust-remote-code \
  --tensor-parallel-size $TP_SIZE \
  2>&1 | tee /tmp/vllm_server.log &
VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"
```

### 2. Wait for server readiness

Poll the health endpoint (timeout 5 minutes, poll every 3 seconds):

```bash
for i in $(seq 1 100); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null | grep -q "200"; then
    echo "vLLM server ready after $((i*3)) seconds"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM server process died. Check logs:"
    tail -30 /tmp/vllm_server.log
    break
  fi
  sleep 3
done
```

### 3. Verify model is loaded

```bash
curl -s http://localhost:$PORT/v1/models | python -c "
import json, sys
d = json.load(sys.stdin)
models = [m['id'] for m in d['data']]
print('Loaded models:', models)
"
```

If the server failed to start, check `/tmp/vllm_server.log` and apply fixes:

| Error Pattern | Fix |
|--------------|-----|
| `torch.cuda.OutOfMemoryError` | Add `--gpu-memory-utilization 0.8` or reduce `--max-model-len 2048` |
| `ValueError` mentioning quantization | vLLM may not support this quant format. `pip install --upgrade vllm` |
| `Address already in use` | Change `PORT` to 8200, or `kill $(lsof -t -i:$PORT)` |
| Model too large for TP | Increase `TP_SIZE` to match available GPUs |

## Part B: Speed Benchmark

### 4. Run benchmark

**Option A** -- vLLM built-in benchmark (preferred):

```bash
vllm bench serve \
  --model "$MODEL_NAME" \
  --base-url http://localhost:$PORT/v1 \
  --num-prompts 100 \
  --input-len 128 \
  --output-len 128 \
  2>&1 | tee /tmp/vllm_bench.log
```

**Option B** -- If `vllm bench` is not available, use a Python script:

```bash
python -c "
import requests, time, statistics

url = 'http://localhost:$PORT/v1/completions'
prompt = 'The quick brown fox ' * 30  # ~120 tokens
latencies, output_tokens = [], []

for i in range(20):
    start = time.time()
    resp = requests.post(url, json={
        'model': '$MODEL_NAME', 'prompt': prompt,
        'max_tokens': 128, 'stream': False,
    })
    elapsed = time.time() - start
    n = resp.json()['usage']['completion_tokens']
    latencies.append(elapsed)
    output_tokens.append(n)
    print(f'Request {i+1}: {elapsed:.2f}s, {n} tokens')

total_tok = sum(output_tokens)
total_t = sum(latencies)
sl = sorted(latencies)
print(f'\n--- SPEED RESULTS ---')
print(f'Throughput:    {total_tok/total_t:.1f} tok/s')
print(f'Avg latency:   {statistics.mean(latencies)*1000:.1f} ms')
print(f'P50 latency:   {sl[len(sl)//2]*1000:.1f} ms')
print(f'P99 latency:   {sl[int(len(sl)*0.99)]*1000:.1f} ms')
"
```

### 5. Record results

Capture from the benchmark output:

- **Throughput**: output tokens/s
- **TTFT** (time to first token): mean, p50, p99 (if available from `vllm bench`)
- **End-to-end latency**: mean, p50, p99

### 6. Server cleanup

If `KEEP_SERVER` is `true`, leave the server running and report `VLLM_PID` and `PORT` to the caller.

Otherwise:

```bash
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "vLLM server stopped"
```

## Output

- Speed benchmark results (throughput, latency metrics)
- Server status: running (with PID and PORT) or stopped
- Any deployment issues encountered
