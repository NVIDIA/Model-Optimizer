# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Command-line interface for graph surgery operations.

This module provides CLI access to graph surgery tools:

Replace attention with GQA (for FP16/BF16 LLMs)::

    python -m modelopt.onnx.graph_surgery replace-gqa \
        --input model.onnx \
        --output model_gqa.onnx \
        --model-id meta-llama/Llama-2-7b-hf

Replace attention with GQA (for INT4/AWQ quantized LLMs)::

    python -m modelopt.onnx.graph_surgery replace-gqa \
        --input model.onnx \
        --output model_gqa.onnx \
        --model-id meta-llama/Llama-3.1-8B

Add cross-attention KV cache to encoder::

    python -m modelopt.onnx.graph_surgery add-cross-kv \
        --input encoder_model.onnx \
        --output encoder_with_kv.onnx \
        --model-id openai/whisper-large-v3-turbo

Convert FP16 to BF16::

    python -m modelopt.onnx.graph_surgery convert-bf16 \
        --input model_fp16.onnx \
        --output model_bf16.onnx

Transpose DequantizeLinear weights (column-major optimization)::

    python -m modelopt.onnx.graph_surgery transpose-dq \
        --input model_quantized.onnx \
        --output model_quantized_transposed.onnx

Analyze attention pattern::

    python -m modelopt.onnx.graph_surgery analyze \
        --input model.onnx \
        --layer 0
"""

import argparse
import sys


def main():
    """Main entry point for graph surgery CLI."""
    parser = argparse.ArgumentParser(
        description="ONNX Graph Surgery Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Replace attention with GQA (FP16/BF16 LLMs):
    python -m modelopt.onnx.graph_surgery replace-gqa -i model.onnx -o model_gqa.onnx -m meta-llama/Llama-2-7b-hf

  Replace attention with GQA (INT4/AWQ quantized LLMs):
    python -m modelopt.onnx.graph_surgery replace-gqa -i model.onnx -o model_gqa.onnx -m meta-llama/Llama-3.1-8B

  Add cross-attention KV to encoder:
    python -m modelopt.onnx.graph_surgery add-cross-kv \\
      -i encoder.onnx -o encoder_kv.onnx -m openai/whisper-large-v3-turbo

  Convert FP16 to BF16:
    python -m modelopt.onnx.graph_surgery convert-bf16 -i model_fp16.onnx -o model_bf16.onnx

  Transpose DequantizeLinear weights:
    python -m modelopt.onnx.graph_surgery transpose-dq -i model_quantized.onnx -o model_transposed.onnx

  Analyze attention pattern:
    python -m modelopt.onnx.graph_surgery analyze -i model.onnx --layer 0
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Replace GQA subcommand
    gqa_parser = subparsers.add_parser(
        "replace-gqa",
        help="Replace attention with GroupQueryAttention",
        description="Replace standard attention subgraphs with GroupQueryAttention (GQA).",
    )
    gqa_parser.add_argument("-i", "--input", required=True, help="Input ONNX model path")
    gqa_parser.add_argument("-o", "--output", required=True, help="Output ONNX model path")
    gqa_parser.add_argument(
        "-m", "--model-id", required=True, help="HuggingFace model ID for config"
    )
    gqa_parser.add_argument("--max-seq-len", type=int, default=4096, help="Maximum sequence length")
    gqa_parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="I/O data type",
    )
    gqa_parser.add_argument(
        "--no-external-data",
        action="store_true",
        help="Embed weights in the model file (disables external data)",
    )
    gqa_parser.add_argument(
        "--external-data-name",
        type=str,
        default=None,
        help="Name for external data file (default: model.onnx_data)",
    )
    gqa_parser.add_argument(
        "--ir-version",
        type=int,
        default=None,
        help="Set ONNX IR version for compatibility (e.g., 9 for older ORT versions)",
    )
    gqa_parser.add_argument(
        "--pack-qkv",
        action="store_true",
        help=(
            "For quantized models: concatenate Q/K/V outputs into a single packed"
            " QKV tensor for GQA input (default: separate Q/K/V inputs)"
        ),
    )
    gqa_parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress messages")
    gqa_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code in HuggingFace model config",
    )

    # Add cross-KV subcommand
    cross_kv_parser = subparsers.add_parser(
        "add-cross-kv",
        help="Add cross-attention KV cache outputs to encoder",
        description="Add cross-attention K/V projection outputs to encoder for GenAI compatibility.",
    )
    cross_kv_parser.add_argument(
        "-i", "--input", required=True, help="Input encoder ONNX model path"
    )
    cross_kv_parser.add_argument("-o", "--output", required=True, help="Output ONNX model path")
    cross_kv_parser.add_argument(
        "-m", "--model-id", required=True, help="HuggingFace model ID for cross-attention weights"
    )
    cross_kv_parser.add_argument(
        "--hidden-state-name",
        default="last_hidden_state",
        help="Name of encoder hidden state output",
    )
    cross_kv_parser.add_argument(
        "--no-rename-input",
        action="store_true",
        help="Don't rename input_features to audio_features",
    )
    cross_kv_parser.add_argument(
        "--no-external-data",
        action="store_true",
        help="Don't save weights as external data",
    )
    cross_kv_parser.add_argument(
        "--decoder-filename",
        default="decoder_with_past_model.onnx",
        help="Decoder ONNX filename for genai_config.json (default: decoder_with_past_model.onnx)",
    )
    cross_kv_parser.add_argument(
        "--no-genai-config",
        action="store_true",
        help="Don't generate genai_config.json",
    )
    cross_kv_parser.add_argument(
        "--provider",
        default="cuda",
        choices=["cuda", "cpu", "NvTensorRtRtx"],
        help="Execution provider for genai_config.json",
    )
    cross_kv_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress messages"
    )
    cross_kv_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code in HuggingFace model",
    )

    # Convert BF16 subcommand
    bf16_parser = subparsers.add_parser(
        "convert-bf16",
        help="Convert FP16 model to BF16",
        description="Convert an ONNX model from FP16 to BF16 precision.",
    )
    bf16_parser.add_argument("-i", "--input", required=True, help="Input FP16 ONNX model path")
    bf16_parser.add_argument("-o", "--output", required=True, help="Output BF16 ONNX model path")
    bf16_parser.add_argument(
        "--no-external-data",
        action="store_true",
        help="Don't save weights as external data",
    )
    bf16_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress messages"
    )

    # Transpose DQ subcommand
    transpose_parser = subparsers.add_parser(
        "transpose-dq",
        help="Transpose DequantizeLinear weights for column-major storage",
        description="Transpose weights in DequantizeLinear nodes for column-major storage optimization.",
    )
    transpose_parser.add_argument(
        "-i", "--input", required=True, help="Input quantized ONNX model path"
    )
    transpose_parser.add_argument("-o", "--output", required=True, help="Output ONNX model path")
    transpose_parser.add_argument(
        "--no-external-data",
        action="store_true",
        help="Don't save weights as external data",
    )
    transpose_parser.add_argument(
        "--external-data-name",
        type=str,
        default=None,
        help="Name for external data file",
    )
    transpose_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress messages"
    )

    # Analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze attention pattern in model",
        description="Analyze the attention pattern in an existing model for debugging.",
    )
    analyze_parser.add_argument("-i", "--input", required=True, help="Input ONNX model path")
    analyze_parser.add_argument("--layer", type=int, default=0, help="Layer to analyze")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "replace-gqa":
        from .gqa_replacement import replace_attention_with_gqa

        replace_attention_with_gqa(
            model_path=args.input,
            output_path=args.output,
            hf_model_id=args.model_id,
            max_seq_len=args.max_seq_len,
            io_dtype=args.dtype,
            use_external_data=not args.no_external_data,
            external_data_name=args.external_data_name,
            ir_version=args.ir_version,
            pack_qkv=args.pack_qkv,
            verbose=not args.quiet,
            trust_remote_code=args.trust_remote_code,
        )

    elif args.command == "add-cross-kv":
        from .encoder_cross_kv import add_cross_kv_to_encoder

        add_cross_kv_to_encoder(
            encoder_path=args.input,
            output_path=args.output,
            hf_model_id=args.model_id,
            hidden_state_output_name=args.hidden_state_name,
            rename_input_features=not args.no_rename_input,
            use_external_data=not args.no_external_data,
            decoder_filename=args.decoder_filename,
            generate_genai_config=not args.no_genai_config,
            provider=args.provider,
            verbose=not args.quiet,
            trust_remote_code=args.trust_remote_code,
        )

    elif args.command == "convert-bf16":
        from .utils.dtype_conversion import convert_fp16_to_bf16

        convert_fp16_to_bf16(
            input_path=args.input,
            output_path=args.output,
            external_data=not args.no_external_data,
            verbose=not args.quiet,
        )

    elif args.command == "transpose-dq":
        from .dq_transpose import transpose_dequantize_linear_weights

        transpose_dequantize_linear_weights(
            model_path=args.input,
            output_path=args.output,
            use_external_data=not args.no_external_data,
            external_data_name=args.external_data_name,
            verbose=not args.quiet,
        )

    elif args.command == "analyze":
        from .gqa_replacement import analyze_attention_pattern

        analyze_attention_pattern(args.input, args.layer)


if __name__ == "__main__":
    main()
