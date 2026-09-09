# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Quantize an HF embedding or reranking model and export it to ONNX.

The default recipe quantizes weights, inputs, and projection outputs to NVFP4 so
TensorRT can keep inter-layer activations in FP4. Embedding models export pooled,
normalized embeddings; reranking models export relevance logits.

This example is experimental because its accuracy has not been validated.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from onnx import numpy_helper
from torch.onnx import register_custom_op_symbolic, symbolic_helper
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers.integrations import sdpa_attention

import modelopt.torch._deploy.utils.torch_onnx as torch_onnx_utils
import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata
from modelopt.torch.quantization.export_onnx import scaled_dot_product_attention

__all__ = [
    "EmbeddingModel",
    "RerankModel",
    "install_static_extent_fix",
    "main",
    "register_bidirectional_sdpa",
]

DEFAULT_RECIPE = "huggingface/nemotron_llama/ptq/nvfp4_output_quant_proj"

# TODO: Add an accuracy evaluation pipeline for the embedding and reranking models.
CALIBRATION_TEXTS = [
    ("What is the capital of France?", "Paris is the capital and most populous city of France."),
    ("How do vaccines work?", "Vaccines train the immune system to recognize pathogens."),
    ("Who wrote Hamlet?", "Hamlet is a tragedy written by William Shakespeare around 1600."),
    ("What causes rain?", "Rain forms when water vapor condenses into droplets that fall."),
    (
        "What is photosynthesis?",
        "Plants convert light energy into chemical energy in chloroplasts.",
    ),
    ("How fast is light?", "Light travels at approximately 299,792 kilometers per second."),
    ("What is machine learning?", "Machine learning builds models that learn patterns from data."),
    ("Where is the Great Barrier Reef?", "The reef lies off the coast of Queensland, Australia."),
    ("What is inflation?", "Inflation is the rate at which prices for goods and services rise."),
    ("How do airplanes fly?", "Wings generate lift as air flows faster over their curved tops."),
    ("What is DNA?", "DNA carries genetic instructions for development and functioning."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa in the early 1500s."),
    ("What is quantum computing?", "Quantum computers use qubits that exist in superposition."),
    ("Why is the sky blue?", "Rayleigh scattering disperses shorter blue wavelengths of sunlight."),
    ("What is the tallest mountain?", "Mount Everest rises 8,849 meters above sea level."),
    ("How do batteries store energy?", "Batteries store energy in chemical form as electricity."),
]


class EmbeddingModel(torch.nn.Module):
    """Bidirectional encoder + mean pooling + L2 normalization."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, input_ids, attention_mask):
        hidden = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return F.normalize(pooled, p=2, dim=1)


class RerankModel(torch.nn.Module):
    """Sequence-classification reranker exported to its relevance logits."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, input_ids, attention_mask):
        return self.base(input_ids=input_ids, attention_mask=attention_mask).logits


def register_bidirectional_sdpa():
    """Export SDPA as a plain MatMul->Add->Softmax->MatMul pattern.

    torch's default sdpa symbolic emits IsNaN/Where guards for the dynamic
    is_causal flag traced by transformers, which prevents TensorRT attention
    fusion. This model is bidirectional, so is_causal is pinned to False.
    """

    def bidirectional_sdpa(
        g,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        if not isinstance(dropout_p, float):
            dropout_p = symbolic_helper._maybe_get_const(dropout_p, "f")
        # Keep the attention chain in a single precision: the symbolic emits the
        # scale and boolean-mask constants in f32 otherwise, upcasting softmax.
        q_dtype = query.type().scalarType()
        if scale is not None and not symbolic_helper._is_none(scale) and q_dtype is not None:
            scale = g.op("Cast", scale, to_i=symbolic_helper.cast_pytorch_to_onnx[q_dtype])
        if (
            attn_mask is not None
            and not symbolic_helper._is_none(attn_mask)
            and attn_mask.type().scalarType() == "Bool"
        ):
            mdtype = torch.float16 if q_dtype == "Half" else torch.float32
            zero = g.op("Constant", value_t=torch.tensor([0.0], dtype=mdtype))
            neg = g.op("Constant", value_t=torch.tensor([torch.finfo(mdtype).min], dtype=mdtype))
            attn_mask = g.op("Where", attn_mask, zero, neg)
        return scaled_dot_product_attention(
            g, query, key, value, attn_mask, dropout_p, False, scale, False
        )

    register_custom_op_symbolic("aten::scaled_dot_product_attention", bidirectional_sdpa, 14)

    # Repeat KV heads explicitly: the sdpa symbolic does not implement enable_gqa.
    sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False


def install_static_extent_fix(hidden_size):
    """Give Reshape -> TRT_FP4DynamicQuantize a static blocked-axis extent.

    torch traces the attention-output reshape as Reshape(target=Concat(B, S, -1));
    TensorRT's DynamicQuantize requires the blocked (last) axis extent to be known
    at build time, so the trailing -1 is rewired to the hidden size.
    """
    orig_quantize_weights = torch_onnx_utils.quantize_weights

    def fix_reshapes(onnx_graph):
        producers = {out: n for n in onnx_graph.graph.node for out in n.output}
        inits = {i.name: i for i in onnx_graph.graph.initializer}
        fixed = 0
        for node in onnx_graph.graph.node:
            if node.op_type != "TRT_FP4DynamicQuantize":
                continue
            src = producers.get(node.input[0])
            while src is not None and src.op_type in ("Cast", "Identity"):
                src = producers.get(src.input[0])
            if src is None or src.op_type != "Reshape":
                continue
            shape_src = producers.get(src.input[1])
            if shape_src is None or shape_src.op_type != "Concat":
                continue
            last = shape_src.input[-1]
            arr = None
            if last in inits:
                arr = numpy_helper.to_array(inits[last])
            else:
                cn = producers.get(last)
                if cn is not None and cn.op_type == "Constant":
                    arr = numpy_helper.to_array(cn.attribute[0].t)
            if arr is None or arr.size != 1 or int(arr.reshape(-1)[0]) != -1:
                continue
            new_init = numpy_helper.from_array(
                np.array([hidden_size], dtype=np.int64),
                name=f"{src.name or src.output[0]}_static_dim",
            )
            onnx_graph.graph.initializer.append(new_init)
            shape_src.input[len(shape_src.input) - 1] = new_init.name
            fixed += 1
        print(f"Static-extent fix applied to {fixed} Reshape->DynamicQuantize sites")
        return onnx_graph

    def quantize_weights_and_fix(model, onnx_graph):
        return fix_reshapes(orig_quantize_weights(model, onnx_graph))

    torch_onnx_utils.quantize_weights = quantize_weights_and_fix


def positive_int(value):
    """Parse a strictly positive integer argument."""
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def main():
    """Run recipe-driven quantization and ONNX export."""
    parser = argparse.ArgumentParser(
        description="Quantize an HF embedding model with a PTQ recipe and export to ONNX."
    )
    parser.add_argument(
        "--model_path",
        default="nvidia/llama-nemotron-embed-1b-v2",
        help="HF hub id or local path of the embedding model.",
    )
    parser.add_argument(
        "--recipe",
        default=DEFAULT_RECIPE,
        help="PTQ recipe: a path under modelopt_recipes/ or a recipe YAML file.",
    )
    parser.add_argument(
        "--onnx_save_path",
        required=True,
        help="The save path for the exported ONNX model.",
    )
    parser.add_argument(
        "--calibration_data_size",
        type=positive_int,
        default=64,
        help="Number of calibration samples.",
    )
    parser.add_argument(
        "--batch_size",
        type=positive_int,
        default=8,
        help="Batch size for calibration.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow model repositories to execute their custom Python code.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    is_reranker = any("ForSequenceClassification" in a for a in (config.architectures or []))
    if is_reranker:
        base = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, trust_remote_code=args.trust_remote_code, dtype=torch.float32
        )
        model = RerankModel(base).to(device).eval()
    else:
        base = AutoModel.from_pretrained(
            args.model_path, trust_remote_code=args.trust_remote_code, dtype=torch.float32
        )
        model = EmbeddingModel(base).to(device).eval()

    pairs = (CALIBRATION_TEXTS * (args.calibration_data_size // len(CALIBRATION_TEXTS) + 1))[
        : args.calibration_data_size
    ]

    def forward_loop(m):
        for i in range(0, len(pairs), args.batch_size):
            chunk = pairs[i : i + args.batch_size]
            if is_reranker:
                # Rerankers score (query, passage) pairs.
                batch = tokenizer(
                    [q for q, _ in chunk],
                    [p for _, p in chunk],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
            else:
                batch = tokenizer(
                    [f"question:{q} \n \n passage:{p}" for q, p in chunk],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
            with torch.no_grad():
                m(batch["input_ids"], batch["attention_mask"])

    recipe = load_recipe(args.recipe)
    quant_cfg = recipe.quantize.model_dump(exclude_unset=True)
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    mtq.print_quant_summary(model)

    # Unequal lengths so padding produces a real attention mask: transformers
    # otherwise drops the all-ones mask and switches sdpa to native GQA, which
    # the export symbolic does not support.
    example = tokenizer(
        ["example query one", "example query two"],
        ["example passage one", "a considerably longer example passage two with extra words"],
        padding=True,
        return_tensors="pt",
    ).to(device)

    register_bidirectional_sdpa()
    install_static_extent_fix(base.config.hidden_size)

    model_name = os.path.basename(args.onnx_save_path).replace(".onnx", "")
    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model=model,
        dummy_input={
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
        },
        model_name=model_name,
        weights_dtype="fp16",
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
        },
    )
    OnnxBytes.from_bytes(onnx_bytes).write_to_disk(
        os.path.dirname(args.onnx_save_path) or ".", clean_dir=False
    )
    print(f"Quantized ONNX model is saved to {args.onnx_save_path}")


if __name__ == "__main__":
    main()
