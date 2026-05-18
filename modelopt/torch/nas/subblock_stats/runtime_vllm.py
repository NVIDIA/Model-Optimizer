import argparse
import json
import os
from pathlib import Path

from vllm.benchmarks.latency import main as vllm_latency_main

from modelopt.torch.nas.subblock_stats.runtime_utils import RuntimeConfig


def run_vllm_latency_benchmark(model_path: Path, runtime_config: RuntimeConfig):
    """Run ``vllm bench latency`` and return the average latency in milliseconds."""
    output_json_path = model_path / "vllm_latency_benchmark.json"

    # Use vLLM latency benchmark as a library.

    # Create a mock argparse.Namespace similar to what is parsed by vllm.benchmarks.latency.main
    args_ns = argparse.Namespace()

    # Populate the Namespace with all required attributes
    args_ns.model = str(model_path)
    args_ns.input_len = runtime_config.prefill_seq_len
    args_ns.output_len = runtime_config.generation_seq_len
    args_ns.batch_size = 1
    args_ns.output_json = str(output_json_path)
    args_ns.max_model_len = runtime_config.prefill_seq_len + runtime_config.generation_seq_len
    args_ns.num_iters_warmup = runtime_config.num_warmup_iters
    args_ns.num_iters = runtime_config.num_iters
    args_ns.max_num_seqs = 1
    args_ns.distributed_executor_backend = (
        "external_launcher"  # Running vLLM with torchrun so need to indicate that.
    )
    args_ns.tensor_parallel_size = 1
    args_ns.pipeline_parallel_size = 1
    args_ns.optimization_level = 0  # This is required to make the stats accurate.
    args_ns.n = 1
    args_ns.disable_detokenize = False

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    vllm_latency_main(args_ns)

    with open(output_json_path) as f:
        vllm_results = json.load(f)
    print(vllm_results)
    return vllm_results["avg_latency"] * 1000  # convert to milliseconds
