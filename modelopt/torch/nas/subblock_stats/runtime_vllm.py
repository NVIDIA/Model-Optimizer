import json
import os
import subprocess
from pathlib import Path

from modelopt.torch.nas.subblock_stats.runtime_utils import RuntimeConfig


def run_vllm_latency_benchmark(model_path: Path, runtime_config: RuntimeConfig):
    """Run ``vllm bench latency`` and return the average latency in milliseconds."""
    output_json_path = model_path / "vllm_latency_benchmark.json"

    cmd = [
        "vllm",
        "bench",
        "latency",
        "--model",
        str(model_path),
        "--input-len",
        str(runtime_config.prefill_seq_len),
        "--output-len",
        str(runtime_config.generation_seq_len),
        "--batch-size",
        str(runtime_config.batch_size),
        "--output-json",
        str(output_json_path),
        "--max-model-len",
        str(runtime_config.prefill_seq_len + runtime_config.generation_seq_len),
        "--num-iters-warmup",
        str(runtime_config.num_warmup_iters),
        "--num-iters",
        str(runtime_config.num_iters),
        "--max-num-seqs",
        "1",
        "--distributed-executor-backend",
        "external_launcher",
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
    ]
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    subprocess.run(cmd)

    with open(output_json_path) as f:
        vllm_results = json.load(f)
    print(vllm_results)
    return vllm_results["avg_latency"] * 1000  # convert to milliseconds
