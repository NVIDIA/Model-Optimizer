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

"""Prepare a BF16 reference, then evaluate a calibrated fake-quant candidate's conditional KL."""

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load, save
from transformers import AutoConfig, GenerationConfig, set_seed

from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils import is_quantized
from modelopt.torch.utils.plugins.hf_checkpoint_utils import resolve_checkpoint_file

__all__ = ["evaluate", "spaced_positions"]

EVALUATION_DATASET = "nvidia/balanced-benchmark-prompts-short-1000"
EVALUATION_REVISION = "c592347fa6645b5e663e128e464850f6ccad4fcc"
SCHEMA = 4
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TOP_K = 2048
DEFAULT_SCORE_TOKENS = 512
DEFAULT_GENERATION_BATCH_SIZE = 64
SHARD_HEADER_BYTES = 64 * 1024  # Safetensors length prefix plus a generous JSON header.


def spaced_positions(length, count):
    """Select an endpoint-inclusive integer grid, matching EAR's position policy."""
    if length < 1 or count < 1:
        raise ValueError("Require positive response length and position count.")
    take = min(length, count)
    return [0] if take == 1 else [i * (length - 1) // (take - 1) for i in range(take)]


def _enabled_quantizers(module):
    return [m for m in module.modules() if isinstance(m, TensorQuantizer) and m.is_enabled]


def _continuations(reference, prompts, generation):
    device = reference.get_input_embeddings().weight.device
    width = max(len(prompt) for prompt in prompts)
    inputs = torch.full((len(prompts), width), generation.pad_token_id, device=device)
    attention_mask = torch.zeros_like(inputs)
    for index, prompt in enumerate(prompts):
        inputs[index, width - len(prompt) :] = torch.tensor(prompt, device=device)
        attention_mask[index, width - len(prompt) :] = 1
    original = reference.generation_config
    try:
        reference.generation_config = generation
        sequences = reference.generate(
            input_ids=inputs, attention_mask=attention_mask, generation_config=generation
        )
    finally:
        reference.generation_config = original
    eos = generation.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    result = []
    for tokens in sequences[:, width:].cpu().tolist():
        end = next((i + 1 for i, token in enumerate(tokens) if token in eos_ids), len(tokens))
        result.append(tokens[:end])
    return result


@torch.inference_mode()
def _scored_logits(model, prompt, continuation, positions):
    device = model.get_input_embeddings().weight.device
    inputs = torch.tensor([prompt + continuation[:-1]], device=device)
    columns = torch.tensor([len(prompt) - 1 + p for p in positions], device=device)
    kwargs = {}
    parameter = inspect.signature(model.forward).parameters.get("logits_to_keep")
    head = model.get_output_embeddings()
    # Activation quantizers may derive shared scales from rows that selection would omit.
    if any(
        _enabled_quantizers(quantizer)
        for name in ("input_quantizer", "output_quantizer")
        if (quantizer := getattr(head, name, None)) is not None
    ):
        parameter = None
    # Accelerate moves tensor kwargs to the input GPU; final hidden states can be elsewhere.
    sparse = (
        parameter is not None
        and "Tensor" in str(parameter.annotation)
        and len(set((getattr(model, "hf_device_map", None) or {}).values())) <= 1
    )
    if sparse:
        kwargs["logits_to_keep"] = columns
    elif parameter is not None:
        kwargs["logits_to_keep"] = len(continuation)
    logits = model(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        position_ids=torch.arange(inputs.shape[1], device=device).unsqueeze(0),
        use_cache=False,
        return_dict=True,
        **kwargs,
    ).logits[0]
    if sparse:
        return logits
    return logits[columns.to(logits.device) - (inputs.shape[1] - len(logits))]


@torch.inference_mode()
def _prepare_reference(
    reference,
    prompts,
    generation,
    directory,
    metadata,
    top_k=DEFAULT_TOP_K,
    score_tokens=DEFAULT_SCORE_TOKENS,
    batch_size=DEFAULT_GENERATION_BATCH_SIZE,
):
    if not prompts or batch_size < 1 or score_tokens < 0:
        raise ValueError(
            "Require prompts, positive generation batch size and nonnegative score_tokens."
        )
    text_config = getattr(reference.config, "text_config", reference.config)
    if not 1 <= top_k <= text_config.vocab_size:
        raise ValueError("top_k must be positive and fit the vocabulary.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    reference.eval()
    records = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        for prompt, tokens in zip(batch, _continuations(reference, batch, generation)):
            positions = spaced_positions(len(tokens), score_tokens or len(tokens))
            logits = _scored_logits(reference, prompt, tokens, positions)
            ids, log_p = [], []
            for offset in range(0, len(logits), 32):
                values, indices = logits[offset : offset + 32].float().topk(top_k, dim=-1)
                ids.append(indices.to(device="cpu", dtype=torch.int32))
                log_p.append(F.log_softmax(values, dim=-1).cpu())
            tensors = {"ids": torch.cat(ids), "log_p": torch.cat(log_p)}
            if not torch.isfinite(tensors["log_p"]).all():
                raise ValueError("Non-finite reference log-probabilities.")
            data = save(tensors)
            (directory / f"{len(records)}.safetensors").write_bytes(data)
            records.append(
                {
                    "prompt": prompt,
                    "continuation": tokens,
                    "positions": positions,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            del logits
    manifest = {
        "schema": SCHEMA,
        "vocab_size": text_config.vocab_size,
        "top_k": top_k,
        "score_tokens": score_tokens,
        "generation": generation.to_dict(),
        "generation_batch_size": batch_size,
        "metadata": {**metadata, "attn_implementation": text_config._attn_implementation},
        "records": records,
    }
    # The manifest is written last, so an interrupted preparation cannot be evaluated.
    (directory / "manifest.json").write_text(json.dumps(manifest))


def _load_reference(directory):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported reference schema; prepare a new reference.")
    for index, record in enumerate(manifest["records"]):
        positions, length = record["positions"], len(record["continuation"])
        # JSON floats compare equal to ints but cannot index logits.
        if any(type(p) is not int for p in positions) or positions != spaced_positions(
            length, manifest["score_tokens"] or length
        ):
            raise ValueError("Invalid reference positions.")
        _read_shard(directory, index, manifest)  # Check every shard before any model loads.
    return manifest


def _read_shard(directory, index, manifest):
    """Bytes of one shard, bounded by its record's size and checked against its digest."""
    record, path = manifest["records"][index], Path(directory) / f"{index}.safetensors"
    # int32 IDs plus float32 log-probabilities for each selected position.
    limit = len(record["positions"]) * manifest["top_k"] * 8 + SHARD_HEADER_BYTES
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"Invalid reference: shard {index} is missing or too large.")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"Invalid reference: shard {index} does not match its record.")
    return data


def _load_statistics(directory, index, manifest):
    # Decode the same bytes that were hashed, binding each shard to its own record.
    statistics = load(_read_shard(directory, index, manifest))
    ids, log_p = statistics["ids"], statistics["log_p"]
    expected = (len(manifest["records"][index]["positions"]), manifest["top_k"])
    if ids.shape != expected or log_p.shape != expected:
        raise ValueError("Invalid reference statistics shape.")
    return ids, log_p


def _cached_kl(log_p, ids, rows):
    """Reduce cached conditional KL over selected positions in bounded chunks."""
    score = torch.zeros((), dtype=torch.float32, device=rows.device)
    for offset in range(0, len(rows), 32):
        selected = ids[offset : offset + 32].to(device=rows.device, dtype=torch.long)
        log_q = F.log_softmax(rows[offset : offset + 32].gather(-1, selected).float(), dim=-1)
        score += F.kl_div(
            log_q,
            log_p[offset : offset + 32].to(rows.device),
            log_target=True,
            reduction="sum",
        )
    score = score / len(rows)
    if not torch.isfinite(score):
        raise ValueError("Non-finite KL: inspect the model logits instead of dropping positions.")
    return score.item()


def evaluate(candidate, reference_dir):
    """Return mean conditional KL over every prompt in a prepared reference."""
    return _score(candidate, Path(reference_dir), _load_reference(reference_dir))


@torch.inference_mode()
def _score(candidate, directory, manifest):
    text_config = getattr(candidate.config, "text_config", candidate.config)
    if text_config.vocab_size != manifest["vocab_size"]:
        raise ValueError("Candidate vocabulary differs from the reference.")
    if text_config._attn_implementation != manifest["metadata"]["attn_implementation"]:
        raise ValueError("Candidate attention backend differs from the reference.")
    candidate.eval()
    total = 0.0
    for index, record in enumerate(manifest["records"]):
        ids, log_p = _load_statistics(directory, index, manifest)
        rows = _scored_logits(
            candidate, record["prompt"], record["continuation"], record["positions"]
        )
        total += _cached_kl(log_p, ids, rows)
        del rows, ids, log_p
    return total / len(manifest["records"])


def _blended_prompts(tokenizer, num_examples, revision, context_limit, max_new_tokens):
    # Defer dataset loading so cached scoring does not require the dataset dependency.
    from datasets import load_dataset

    dataset = load_dataset(EVALUATION_DATASET, split="test", revision=revision, token=True)
    if not 1 <= num_examples <= len(dataset):
        raise ValueError(f"Require 1 <= num_examples <= {len(dataset)} dataset rows.")
    prompts = []
    for row in dataset.select(range(num_examples)):
        # Only messages enter the model; difficulty labels and other metadata stay outside it.
        input_ids = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            return_dict=False,
        )
        if not input_ids:
            raise ValueError(f"Empty prompt for dataset row {row['cohort_rank']}.")
        if context_limit and len(input_ids) + max_new_tokens > context_limit:
            raise ValueError(
                f"Dataset row {row['cohort_rank']} requires {len(input_ids)} prompt tokens "
                f"plus {max_new_tokens} continuation tokens, exceeding context {context_limit}. "
                "Prompts are not truncated; use a model with a sufficient context window."
            )
        prompts.append(input_ids)
    return prompts


def _check_reference_match(args, manifest):
    """Refuse evaluation inputs that cannot reproduce the reference, before any model loads."""
    if args.model != manifest["metadata"]["model"]:
        raise ValueError("--model differs from the prepared reference.")
    backend = manifest["metadata"]["attn_implementation"]
    if args.attn_implementation not in (None, backend):
        raise ValueError("Attention backend differs from the prepared reference.")
    args.attn_implementation = backend


def _checkpoint_identity(hub_model_id, checkpoint):
    """Hub snapshot commit, or a local checkpoint's config hashes and weight sizes and mtimes."""
    if hub_model_id:
        # A Hub ID resolves to its latest snapshot, whose directory is named by the commit.
        return checkpoint.name
    metadata, names = {}, set()
    for name in ("config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (checkpoint / name).exists():
            data = resolve_checkpoint_file(checkpoint, name).read_bytes()
            metadata[name] = hashlib.sha256(data).hexdigest()
            if name.endswith(".index.json"):
                names.update(json.loads(data)["weight_map"].values())
    # Without an index, weights sit at the top level; a reference inside the checkpoint is skipped.
    names = names or {
        path.name
        for path in checkpoint.iterdir()
        if path.is_file() and path.suffix in {".safetensors", ".bin"}
    }
    weights = {}
    for name in sorted(names):
        stat = resolve_checkpoint_file(checkpoint, name, max_bytes=None).stat()
        weights[name] = [stat.st_size, stat.st_mtime_ns]
    return {"metadata": metadata, "weights": weights}


def _require_unquantized_config(config):
    text_config = getattr(config, "text_config", config)
    if config.is_encoder_decoder or any(
        getattr(cfg, "quantization_config", None) for cfg in (config, text_config)
    ):
        raise ValueError("Supply an unquantized BF16 causal checkpoint.")


def _require_bf16_base(model):
    # Plain VLM loading attaches disabled quantizers outside the language model.
    if model.dtype != torch.bfloat16 or _enabled_quantizers(model):
        raise ValueError("Supply an unquantized BF16 base checkpoint.")


def _require_fake_quant(model):
    quantizers = _enabled_quantizers(model)
    if not quantizers or any(not quantizer.fake_quant for quantizer in quantizers):
        raise ValueError("The recipe must apply fake quantization; none or packed was found.")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True)
    common.add_argument(
        "--reference", type=Path, required=True, help="Prepared reference directory."
    )
    common.add_argument("--gpu_max_mem_percentage", type=float)
    common.add_argument("--attn_implementation")
    common.add_argument("--trust_remote_code", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-reference", parents=[common])
    prepare.add_argument("--num_examples", type=int, default=300)
    prepare.add_argument("--eval_revision", default=EVALUATION_REVISION)
    prepare.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    prepare.add_argument("--generation_batch_size", type=int, default=DEFAULT_GENERATION_BATCH_SIZE)
    prepare.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    prepare.add_argument(
        "--score_tokens",
        type=int,
        default=DEFAULT_SCORE_TOKENS,
        help="Spaced response positions; 0 scores every valid position.",
    )
    prepare.add_argument("--seed", type=int, default=0)
    score = commands.add_parser("evaluate", parents=[common])
    score.add_argument("--recipe", required=True, help="Built-in PTQ recipe name or YAML path.")
    # None preserves hf_ptq's calibration defaults.
    score.add_argument("--dataset")
    score.add_argument("--calib_size", type=int)
    score.add_argument("--calib_seq", type=int)
    score.add_argument(
        "--batch_size", type=int, help="Calibration batch size; 0 selects automatically."
    )
    return parser.parse_args()


def main():
    """Prepare a reference, or calibrate a candidate and return its scalar KL."""
    args = _parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run kl_eval.py in one process; device-map sharding uses every GPU.")
    directory = args.reference.expanduser().resolve()
    if args.command == "prepare-reference" and directory.exists():
        raise FileExistsError(f"Reference directory already exists: {directory}")
    # Defer hf_ptq's heavy export/calibration imports so cached scoring remains usable on CPU.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hf_ptq"))
    import hf_ptq

    manifest = recipe = None
    if args.command == "evaluate":
        manifest = _load_reference(directory)
        recipe = hf_ptq.load_recipe(args.recipe)
        if not isinstance(recipe, hf_ptq.ModelOptPTQRecipe):
            raise ValueError("--recipe must be a fixed PTQ recipe, not AutoQuantize.")
        hf_ptq.require_live_calibration(recipe)  # Before loading, not after.
        _check_reference_match(args, manifest)
    ptq_argv = ["--model", args.model, "--skip_generate"]
    for name in (
        "recipe",
        "dataset",
        "calib_size",
        "calib_seq",
        "batch_size",
        "gpu_max_mem_percentage",
        "attn_implementation",
    ):
        if (value := getattr(args, name, None)) is not None:
            ptq_argv.extend([f"--{name}", str(value)])
    if args.trust_remote_code:
        ptq_argv.append("--trust_remote_code")
    ptq_args = hf_ptq.parse_args(ptq_argv)
    ptq_args.dataset = ptq_args.dataset.split(",") if ptq_args.dataset else None
    ptq_args.calib_size = [int(ptq_args.calib_size)]
    set_seed(hf_ptq.RAND_SEED)
    torch.compiler.set_stance("force_eager")
    ptq_args.hub_model_id, ptq_args.local_checkpoint_path = hf_ptq.ensure_local_checkpoint(
        ptq_args.pyt_ckpt_path
    )
    checkpoint = Path(ptq_args.local_checkpoint_path).resolve()
    identity = _checkpoint_identity(ptq_args.hub_model_id, checkpoint)
    if manifest is not None and identity != manifest["metadata"]["checkpoint"]:
        raise ValueError("Checkpoint differs from the prepared reference.")
    config = AutoConfig.from_pretrained(
        checkpoint, trust_remote_code=args.trust_remote_code, local_files_only=True
    )
    _require_unquantized_config(config)
    text_config = getattr(config, "text_config", config)
    if args.command == "prepare-reference":
        tokenizer = hf_ptq.get_tokenizer(
            checkpoint, trust_remote_code=args.trust_remote_code, local_files_only=True
        )
        prompts = _blended_prompts(
            tokenizer,
            args.num_examples,
            args.eval_revision,
            getattr(text_config, "max_position_embeddings", None),
            args.max_new_tokens,
        )
        original = (
            GenerationConfig.from_pretrained(checkpoint, local_files_only=True)
            if (checkpoint / "generation_config.json").exists()
            else GenerationConfig.from_model_config(config)
        )
        generation = GenerationConfig(
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=original.eos_token_id
            if original.eos_token_id is not None
            else tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    with hf_ptq.distributed_run(device=ptq_args.device) as dist_state:
        ptq_args.dist_state = dist_state
        # Pass the checked recipe on, so an edit to its file while loading cannot change it.
        loaded = hf_ptq.load_model(ptq_args, resolved_recipe=recipe)
        model, language_model, calibration_only = loaded[0], loaded[1], loaded[3]
        _require_bf16_base(model)
        # hf_ptq skips calibration for a target restored with ModelOpt state.
        if is_quantized(language_model) and not calibration_only:
            raise ValueError("The checkpoint already carries ModelOpt quantizers.")
        if args.command == "prepare-reference":
            set_seed(args.seed)
            return _prepare_reference(
                model,
                prompts,
                generation,
                directory,
                {
                    "model": args.model,
                    "checkpoint": identity,
                    "dataset": EVALUATION_DATASET,
                    "revision": args.eval_revision,
                    "seed": args.seed,
                },
                args.top_k,
                args.score_tokens,
                args.generation_batch_size,
            )
        model = hf_ptq.quantize_main(ptq_args, *loaded, export=False, resolved_recipe=recipe)
        _require_fake_quant(model)
        return _score(model, directory, manifest)


if __name__ == "__main__":
    score = main()
    if score is not None:
        print(score)
