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

import argparse
import os
import sys

import pandas as pd

# Ensure this directory is on sys.path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from perplexity_metrics import calculate_perplexity_hf, perplexity_eval  # noqa: E402


def run_perplexity_on_models(
    model_dirs,
    output_file="perplexity_results.csv",
    i="1024",
    chunk_size=None,
    hf_model=None,
    hf_device="cuda",
    hf_dtype=None,
):
    """
    Run perplexity evaluation on multiple ONNX Runtime GenAI models and/or a HuggingFace model.

    This function evaluates one or more models at different input sequence lengths,
    saves results to a CSV file, and prints a summary report. Each model-length
    combination is evaluated independently, with errors handled gracefully.

    Args:
        model_dirs (list[str]): List of model directory paths to evaluate.
                               Each directory must contain a valid ONNX Runtime GenAI model.
        output_file (str, optional): Path for the output CSV file containing results.
                                    Defaults to "perplexity_results.csv".
        i (str or list, optional): Input sequence lengths to evaluate. Can be:
                                  - String: comma-separated values (e.g., "1024,2048,4096")
                                  - List/tuple: sequence of integers
                                  - Single int: one length to evaluate
                                  Defaults to "1024".
        chunk_size (int, optional): Prefill chunk size for KV cache chunking.
                                   Required for input lengths > 1024.
                                   Overrides chunk_size in model config if provided.
                                   Defaults to None.
        hf_model (str, optional): HuggingFace model name or path to evaluate.
                                 If provided, will download and evaluate this model.
                                 Defaults to None.
        hf_device (str, optional): Device to run HuggingFace model on.
                                  Defaults to "cuda".
        hf_dtype (str, optional): Data type for HuggingFace model.
                                 Options: "float16", "bfloat16", "float32".
                                 Defaults to None (uses model default).

    Returns:
        pd.DataFrame: DataFrame containing evaluation results with columns:
                     - Model Path: Full path to model directory
                     - Model Type: "ONNX" or "HuggingFace"
                     - Input Length: Sequence length used for evaluation
                     - Perplexity: Computed perplexity score (or "N/A" if failed)
                     - Status: "Success" or "Failed"
                     - Error: Error message if failed, "None" if successful

    """
    results = []

    # Parse input lengths
    if isinstance(i, str):
        i_list = [int(x.strip()) for x in i.split(",") if x.strip()]
    elif isinstance(i, (list, tuple)):
        i_list = [int(x) for x in i]
    else:
        i_list = [int(i)]

    # Evaluate HuggingFace model if provided
    if hf_model is not None:
        print(f"\n{'=' * 60}")
        print(f"Evaluating HuggingFace model: {hf_model}")
        print(f"{'=' * 60}")

        # Convert dtype string to torch dtype
        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(hf_dtype.lower()) if hf_dtype else torch.float16

        for input_len in i_list:
            try:
                print(f"  Evaluating with input length: {input_len}")
                if torch_dtype:
                    print(f"  Using dtype: {torch_dtype}")

                # Calculate stride (use chunk_size if provided, otherwise use half of input_len)
                stride = chunk_size if chunk_size is not None else input_len // 2

                if chunk_size is not None:
                    assert stride == chunk_size, (
                        f"For chunking case, stride must equal chunk_size. "
                        f"Got stride={stride}, chunk_size={chunk_size}"
                    )
                perplexity = calculate_perplexity_hf(
                    model_name_or_path=hf_model,
                    max_length=input_len,
                    stride=stride,
                    device=hf_device,
                    torch_dtype=torch_dtype,
                )

                results.append(
                    {
                        "Model Path": hf_model,
                        "Model Type": "HuggingFace",
                        "Input Length": int(input_len),
                        "Perplexity": float(perplexity),
                        "Status": "Success",
                        "Error": "None",
                    }
                )
            except Exception as e:  # noqa: PERF203
                print(f"  Error for input length {input_len}: {e!s}")
                results.append(
                    {
                        "Model Path": hf_model,
                        "Model Type": "HuggingFace",
                        "Input Length": int(input_len),
                        "Perplexity": "N/A",
                        "Status": "Failed",
                        "Error": str(e),
                    }
                )

        print(" HuggingFace perplexity evaluation completed")

        # Unload HuggingFace model from GPU memory before ONNX evaluation
        print("[CLEANUP] Unloading HuggingFace model from GPU memory...")
        import gc

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("[CLEANUP] GPU memory freed")

    # Evaluate ONNX models
    for model_dir in model_dirs:
        print(f"\n{'=' * 60}")
        print(f"Evaluating perplexity for: {model_dir}")
        print(f"{'=' * 60}")

        try:
            # Check if model directory exists
            if not os.path.exists(model_dir):
                print(f"Error: Model directory does not exist: {model_dir}")
                results.append(
                    {
                        "Model Path": model_dir,
                        "Perplexity": "N/A",
                        "Status": "Directory not found",
                        "Error": "Directory does not exist",
                    }
                )
                continue

            # Check if genai_config.json exists
            config_path = os.path.join(model_dir, "genai_config.json")
            if not os.path.exists(config_path):
                print(f"Error: genai_config.json not found in: {model_dir}")
                results.append(
                    {
                        "Model Path": model_dir,
                        "Model Type": "ONNX",
                        "Perplexity": "N/A",
                        "Status": "Invalid model format",
                        "Error": "genai_config.json not found",
                    }
                )
                continue

            # For each input length, run perplexity_eval and record results
            for input_len in i_list:
                try:
                    print(f"  Evaluating with input length: {input_len}")
                    if chunk_size is None:
                        print(
                            "  Note: input length is ignored unless chunk_size is set or "
                            "config.search.chunk_size is present."
                        )
                    if chunk_size is not None:
                        print(f"  Using chunk_size: {chunk_size}")
                        perplexity = perplexity_eval(model_dir, str(input_len), chunk_size)
                    else:
                        perplexity = perplexity_eval(model_dir, str(input_len))
                    results.append(
                        {
                            "Model Path": model_dir,
                            "Model Type": "ONNX",
                            "Input Length": int(input_len),
                            "Perplexity": float(perplexity),
                            "Status": "Success",
                            "Error": "None",
                        }
                    )
                except Exception as e:  # noqa: PERF203
                    print(f"  Error for input length {input_len}: {e!s}")
                    results.append(
                        {
                            "Model Path": model_dir,
                            "Model Type": "ONNX",
                            "Input Length": int(input_len),
                            "Perplexity": "N/A",
                            "Status": "Failed",
                            "Error": str(e),
                        }
                    )

            print(" Perplexity evaluation completed successfully")

        except Exception as e:
            print(f"Error during perplexity evaluation: {e!s}")
            results.append(
                {
                    "Model Path": model_dir,
                    "Model Type": "ONNX",
                    "Perplexity": "N/A",
                    "Status": "Failed",
                    "Error": str(e),
                }
            )

    # Create results DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 60}")

    # Print summary
    successful = df[df["Status"] == "Success"]
    failed = df[df["Status"] != "Success"]

    print("\nSummary:")
    print(f"  Successful evaluations: {len(successful)}")
    print(f"  Failed evaluations: {len(failed)}")

    if len(successful) > 0:
        print("\nPerplexity Results:")
        for _, row in successful.iterrows():
            print(
                f"  {os.path.basename(row['Model Path'])} [i={row.get('Input Length', '?')}]: "
                f"{row['Perplexity']:.4f}"
                if isinstance(row["Perplexity"], (int, float))
                else row["Perplexity"]
            )

    return df


def main():
    """
    Command-line entry point for perplexity evaluation.

    Parses command-line arguments and runs perplexity evaluation on specified
    ONNX Runtime GenAI models and/or HuggingFace models. Results are saved to a CSV file.

    Command-line Arguments:
        --models: One or more ONNX model directory paths (optional)
        --hf_model: HuggingFace model name or path (optional)
        --hf_device: Device for HuggingFace model (default: "cuda")
        --hf_dtype: Data type for HuggingFace model (default: None)
        --i: Comma-separated input sequence lengths (default: "1024")
        --output: Output CSV file path (default: "perplexity_results.csv")
        --chunk_size: Prefill chunk size for prefill chunking (optional)

    Examples:
        # Evaluate ONNX models
        $ python run_perplexity.py --models /path/to/model
        $ python run_perplexity.py --models /path/to/model1 /path/to/model2 \\
              --i 1024,2048,4096 --chunk_size 1024 --output results.csv

        # Evaluate HuggingFace model
        $ python run_perplexity.py --hf_model meta-llama/Llama-2-7b-hf --i 1024
        $ python run_perplexity.py --hf_model meta-llama/Llama-2-7b-hf \\
              --hf_dtype float16 --hf_device cuda --i 1024,2048

        # Evaluate both ONNX and HuggingFace models
        $ python run_perplexity.py --models /path/to/onnx_model \\
              --hf_model meta-llama/Llama-2-7b-hf --i 1024
    """
    parser = argparse.ArgumentParser(
        description="Run perplexity evaluation on ONNX Runtime GenAI and/or HuggingFace models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="List of ONNX model directory paths to evaluate (optional)",
    )
    parser.add_argument(
        "--i",
        default="1024",
        help="Comma-separated input seq lengths to be evaluated (e.g. 1024,2048) please enter number >= 1024",
    )
    parser.add_argument(
        "--output",
        default="perplexity_results.csv",
        help="Output CSV file name (default: perplexity_results.csv)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Chunk size for KV caching optimization (optional)",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        default=None,
        help="HuggingFace model name or path to evaluate (e.g., 'meta-llama/Llama-2-7b-hf')",
    )
    parser.add_argument(
        "--hf_device",
        type=str,
        default="cuda",
        help="Device to run HuggingFace model on (default: 'cuda')",
    )
    parser.add_argument(
        "--hf_dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"],
        help="Data type for HuggingFace model (default: None, uses model default)",
    )

    args = parser.parse_args()

    # Validate that at least one model source is provided
    if not args.models and not args.hf_model:
        print("Error: You must provide either --models or --hf_model (or both)")
        parser.print_help()
        return

    # Validate that all model directories exist
    valid_models = []
    for model_dir in args.models:
        if os.path.exists(model_dir):
            valid_models.append(model_dir)
        else:
            print(f"Warning: Model directory does not exist: {model_dir}")

    # Count total models to evaluate
    total_models = len(valid_models) + (1 if args.hf_model else 0)

    print(f"Running perplexity evaluation on {total_models} model(s)...")
    if args.chunk_size is not None:
        print(f"Using chunk_size: {args.chunk_size}")

    run_perplexity_on_models(
        valid_models,
        args.output,
        args.i,
        args.chunk_size,
        args.hf_model,
        args.hf_device,
        args.hf_dtype,
    )


if __name__ == "__main__":
    main()
