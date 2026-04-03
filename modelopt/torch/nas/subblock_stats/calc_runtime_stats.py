import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from vllm.benchmarks.latency import add_cli_args
from vllm.benchmarks.latency import main as vllm_main

from modelopt.torch.puzzletron.anymodel.converter import Converter
from modelopt.torch.puzzletron.anymodel.models.llama import LlamaConverter, LlamaModelDescriptor
from modelopt.torch.puzzletron.anymodel.puzzformer import deci_x_patcher
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    SubblockConfig,
)
from modelopt.torch.puzzletron.tools.logger import mprint


def create_benchmark_model(
    vocab_size: int,
    hidden_size: int,
    num_attention_heads: int,
    prefill_seq_len: int,
    generation_seq_len: int,
    block_config: BlockConfig | None,
    repeat_block_n_times: int = 10,
) -> LlamaForCausalLM:

    block_configs = [
        BlockConfig(
            attention=AttentionConfig(no_op=False, num_key_value_heads=num_attention_heads),
            ffn=FFNConfig(
                no_op=False, intermediate_size=hidden_size, moe=None
            ),  # , hidden_act="silu"),
            parallel_blocks=None,
        )
    ]

    if block_config:
        block_configs.extend([block_config] * repeat_block_n_times)

    model_config = LlamaConfig(
        max_position_embeddings=prefill_seq_len + generation_seq_len,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=len(block_configs),
        head_dim=None,  # Compute from hidden_size // num_attention_heads instead of using default 128
        # this is required for trt-llm convertion to know which model classes to use to the checkpoint
        auto_map={
            "AutoConfig": "transformers.models.llama.configuration_llama.LlamaConfig",
            "AutoModelForCausalLM": "transformers.models.llama.modeling_llama.LlamaForCausalLM",
        },
    )

    block_configs = LlamaConverter.create_block_configs_from_main_config(model_config)
    model_config.block_configs = block_configs

    with deci_x_patcher(LlamaModelDescriptor, block_configs):
        model = AutoModelForCausalLM.from_config(model_config)

    model.config.architectures = ["AnyModel"]
    model.config.base_architecture = "LlamaForCausalLM"

    return model


def save_model_as_anymodel(model, output_dir: Path, descriptor, num_hidden_layers: int):

    # Save standard model checkpoint (as safetensors, HF format)
    model.save_pretrained(output_dir, safe_serialization=True)

    # Convert/slice weights into AnyModel subblock_safetensors format
    Converter.convert_model_weights(
        input_dir=output_dir,
        output_dir=output_dir,
        descriptor=descriptor,
        num_hidden_layers=num_hidden_layers,
    )
    # Load the model config.json, update "architectures" to ["AnyModel"], and write back to disk.

    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
        config_data["architectures"] = ["AnyModel"]
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)


def save_model(
    model: LlamaForCausalLM, tokenizer_path: Path, output_path: Path, num_hidden_layers: int
) -> None:

    model.to(dtype=torch.bfloat16).save_pretrained(output_path)
    save_model_as_anymodel(model, output_path, LlamaModelDescriptor, num_hidden_layers)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_path)


@dataclass(frozen=True)
class RuntimeConfig:
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    master_puzzle_dir: str
    tokenizer_path: str
    synth_dataset_num_requests: int
    repeat_block_n_times: int
    prefill_seq_len: int
    generation_seq_len: int
    batch_size: int


def run_vllm_latency_benchmark(model_path: Path, runtime_config: RuntimeConfig):

    output_json_path = model_path / "vllm_latency_benchmark.json"
    parser = argparse.ArgumentParser()
    add_cli_args(parser)

    args = parser.parse_args(
        [
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
        ]
    )
    vllm_main(args)
    with open(output_json_path) as f:
        vllm_results = json.load(f)
    return vllm_results["avg_latency"]


def calc_subblock_runtime(
    runtime_config: RuntimeConfig,
    subblock_config: SubblockConfig,
) -> float:

    if subblock_config is not None:
        if not isinstance(subblock_config, AttentionConfig) and not isinstance(
            subblock_config, FFNConfig
        ):
            raise Exception("Runtime stats: Not supported subblock type: {subblock_config}")

    model = create_benchmark_model(
        runtime_config.vocab_size,
        runtime_config.hidden_size,
        runtime_config.num_attention_heads,
        runtime_config.prefill_seq_len,
        runtime_config.generation_seq_len,
        block_config=subblock_config.to_blockconfig() if subblock_config is not None else None,
        repeat_block_n_times=runtime_config.repeat_block_n_times,
    )
    with tempfile.TemporaryDirectory() as model_tmpdir:
        save_model(
            model,
            Path(runtime_config.tokenizer_path),
            Path(model_tmpdir),
            num_hidden_layers=runtime_config.repeat_block_n_times + 1,
        )

        subblock_total_runtime_ms = run_vllm_latency_benchmark(Path(model_tmpdir), runtime_config)

    return subblock_total_runtime_ms


def calc_runtime_for_subblocks(
    subblock_config_set: set[SubblockConfig],
    runtime_stats_config: DictConfig,
    vocab_size: int,
    hidden_size: int,
    num_attention_heads: int,
    master_puzzle_dir: str,
    tokenizer_path: str,
    synth_dataset_num_requests: int,
    prefill_seq_len: int,
    generation_seq_len: int,
) -> tuple[dict[SubblockConfig, float], float]:

    repeat_block_n_times = 10
    runtime_config = RuntimeConfig(
        vocab_size,
        hidden_size,
        num_attention_heads,
        master_puzzle_dir,
        tokenizer_path,
        synth_dataset_num_requests,
        repeat_block_n_times,
        prefill_seq_len,
        generation_seq_len,
        runtime_stats_config.get("batch_size", 1),
    )

    runtime_by_subblock_dict = {}

    for subblock_config in tqdm(
        sorted(subblock_config_set),
        desc=(
            f"Computing runtime_by_subblock_dict [hidden_size={hidden_size}, "
            f"num_subblocks={len(subblock_config_set)}]"
        ),
    ):
        if subblock_config.no_op:
            total_runtime_ms = 0.0
        else:
            subblock_total_runtime_ms = calc_subblock_runtime(runtime_config, subblock_config)
            baseline_runtime_ms = calc_subblock_runtime(runtime_config, None)
            total_runtime_ms = subblock_total_runtime_ms - baseline_runtime_ms
            mprint(
                f"|||| {subblock_config=} {subblock_total_runtime_ms=} {baseline_runtime_ms=} {total_runtime_ms=}"
            )

        runtime_by_subblock_dict[subblock_config] = total_runtime_ms

    no_block_runtime_ms = 0.0  # TODO: Implement this

    return runtime_by_subblock_dict, no_block_runtime_ms
