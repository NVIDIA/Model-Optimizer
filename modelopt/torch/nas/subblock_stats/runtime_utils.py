import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from modelopt.torch.puzzletron.anymodel.converter import Converter
from modelopt.torch.puzzletron.anymodel.models.llama import LlamaModelDescriptor


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for a vLLM latency benchmark run."""

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
    num_iters: int
    num_warmup_iters: int


def save_model(
    model: LlamaForCausalLM, tokenizer_path: Path, output_path: Path, num_hidden_layers: int
) -> None:
    """Save model weights as AnyModel and copy the tokenizer to ``output_path``."""
    model.to(dtype=torch.bfloat16).save_pretrained(output_path)
    save_model_as_anymodel(model, output_path, LlamaModelDescriptor, num_hidden_layers)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_path)


def save_model_as_anymodel(model, output_dir: Path, descriptor, num_hidden_layers: int):
    """Save a model checkpoint in AnyModel subblock-safetensors format."""
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
