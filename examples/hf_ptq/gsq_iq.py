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

"""Paired IQ GPTQ/GSQ WikiText-2 experiment for Qwen3.8-27B."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import time
from pathlib import Path

import torch

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

__all__ = [
    "calibration_windows",
    "evaluate_perplexity",
    "prepare_data",
    "source_identity",
    "source_revision",
    "token_fingerprint",
]

MODEL_ID = "Qwen/Qwen3.8-27B"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
SEQUENCE_LENGTH = 2048
FORMATS = ("iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s")


def source_revision(repository: Path) -> str:
    """Read the checkout commit without requiring Git inside the evaluation container."""
    metadata = repository / ".git"
    if metadata.is_file():
        directive = metadata.read_text().strip()
        if not directive.startswith("gitdir: "):
            raise ValueError("Unsupported Git metadata file")
        metadata = (repository / directive.removeprefix("gitdir: ")).resolve()
    common = metadata
    if (metadata / "commondir").is_file():
        common = (metadata / (metadata / "commondir").read_text().strip()).resolve()
    revision = (metadata / "HEAD").read_text().strip()
    if revision.startswith("ref: "):
        reference = revision.removeprefix("ref: ")
        revision = ""
        for directory in (metadata, common):
            if (directory / reference).is_file():
                revision = (directory / reference).read_text().strip()
                break
        if not revision and (common / "packed-refs").is_file():
            for line in (common / "packed-refs").read_text().splitlines():
                candidate, _, name = line.partition(" ")
                if name == reference:
                    revision = candidate
                    break
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Git metadata does not resolve to a 40-character source commit")
    return revision


def source_identity(model_path: str | Path, model_revision: str) -> dict:
    """Verify the official cached snapshot and hash its actual weights and tokenizer artifacts."""
    snapshot = Path(model_path).resolve(strict=True)
    if tuple(snapshot.parts[-3:]) != ("models--Qwen--Qwen3.8-27B", "snapshots", model_revision):
        raise ValueError(
            "Model path must be the official Qwen3.8-27B cached snapshot at the declared revision"
        )
    index = snapshot / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values())
    if not shards or any(
        Path(name).name != name or not name.endswith(".safetensors") for name in shards
    ):
        raise ValueError("The model index must name weight shards inside its snapshot")
    required = ("config.json", "tokenizer.json", "tokenizer_config.json", index.name)
    if any(not (snapshot / name).is_file() for name in required):
        raise ValueError(
            "The official snapshot is missing a required configuration or tokenizer artifact"
        )
    artifacts = shards | {
        path.name
        for path in snapshot.iterdir()
        if path.is_file() and path.suffix in (".json", ".jinja", ".txt", ".model")
    }
    files = {}
    for name in sorted(artifacts):
        path = snapshot / name
        target = path.resolve(strict=True)
        if name in shards and (
            not path.is_symlink()
            or target.parent != snapshot.parents[1] / "blobs"
            or not re.fullmatch(r"[0-9a-f]{64}", target.name)
        ):
            raise ValueError(f"Weight shard {name} must reference an immutable SHA256 cache blob")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if name in shards and digest != target.name:
            raise ValueError(f"Weight shard {name} does not match its SHA256 cache blob")
        files[name] = {"sha256": digest, "bytes": target.stat().st_size}
    return {"model": MODEL_ID, "revision": model_revision, "files": files}


def token_fingerprint(tokens: torch.Tensor) -> str:
    """Hash canonical int64 token IDs for reproducible data accounting."""
    return hashlib.sha256(tokens.detach().cpu().to(torch.int64).numpy().tobytes()).hexdigest()


def calibration_windows(tokens: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Draw the fixed 128 full-length calibration sequences from training tokens."""
    if tokens.ndim != 1 or len(tokens) < SEQUENCE_LENGTH + 127:
        raise ValueError("Calibration needs a flat training token stream with 128 full windows")
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randperm(len(tokens) - SEQUENCE_LENGTH + 1, generator=generator)[:128]
    return torch.stack([tokens[start : start + SEQUENCE_LENGTH] for start in starts.tolist()])


def prepare_data(tokenizer, dataset_revision: str, seed: int = 42):
    """Tokenize raw WikiText-2 splits without chat templates or special tokens."""
    if load_dataset is None:
        raise ImportError("Install the datasets package to run the WikiText-2 experiment")
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, revision=dataset_revision)
    splits = {
        name: torch.tensor(
            tokenizer(
                "\n\n".join(dataset[name]["text"]),
                add_special_tokens=False,
                truncation=False,
            )["input_ids"],
            dtype=torch.long,
        )
        for name in ("train", "validation", "test")
    }
    calibration = calibration_windows(splits["train"], seed)
    metadata = {
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": dataset_revision,
        "seed": seed,
        "calibration_sequences": 128,
        "calibration_sequence_length": SEQUENCE_LENGTH,
        "calibration_tokens": calibration.numel(),
        "calibration_sha256": token_fingerprint(calibration),
        "splits": {
            name: {
                "dataset_fingerprint": dataset[name]._fingerprint,
                "tokens": tokens.numel(),
                "token_sha256": token_fingerprint(tokens),
            }
            for name, tokens in splits.items()
        },
    }
    return calibration, splits, metadata


@torch.inference_mode()
def evaluate_perplexity(model, tokens: torch.Tensor, device) -> dict:
    """Score every next-token target, including the tail, in windows up to 2,048 tokens."""
    if tokens.ndim != 1 or len(tokens) < 2:
        raise ValueError("Perplexity needs a flat token stream containing at least two tokens")
    model.eval()
    negative_log_likelihood = 0.0
    scored_tokens = 0
    window_count = 0
    for start in range(0, len(tokens) - 1, SEQUENCE_LENGTH - 1):
        window = tokens[start : start + SEQUENCE_LENGTH].unsqueeze(0).to(device)
        targets = window.numel() - 1
        loss = model(window, labels=window, use_cache=False).loss.double().item()
        if not math.isfinite(loss):
            raise RuntimeError("Nonfinite teacher-forced token loss")
        negative_log_likelihood += loss * targets
        scored_tokens += targets
        window_count += 1
    return {
        "perplexity": math.exp(negative_log_likelihood / scored_tokens),
        "negative_log_likelihood": negative_log_likelihood,
        "scored_tokens": scored_tokens,
        "windows": window_count,
        "token_sha256": token_fingerprint(tokens),
        "window_length": SEQUENCE_LENGTH,
        "window_stride": SEQUENCE_LENGTH - 1,
        "final_window_tokens": window.numel(),
        "generation_requests": 0,
        "autoregressive_repeats": 0,
    }


def _write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    """Run the fixed five-format comparison, selecting budgets on validation only."""
    # The experiment's optional model-loading dependencies are not needed by PPL accounting tests.
    from example_utils import get_model, get_tokenizer

    import modelopt.torch.quantization as mtq
    from modelopt.recipe import load_recipe
    from modelopt.torch.quantization.ggml.gsq import IQGSQResult, quantize_iq_gsq
    from modelopt.torch.quantization.utils import is_quantized_linear

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--num-steps", nargs="+", type=int, choices=(200, 500), default=[200])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-tolerance", type=float, default=1e-6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(
        not re.fullmatch(r"[0-9a-f]{40}", revision)
        for revision in (args.model_revision, args.dataset_revision)
    ):
        parser.error("Model and dataset revisions must be immutable 40-character commit hashes")
    if args.validation_tolerance < 0 or not math.isfinite(args.validation_tolerance):
        parser.error("Validation tolerance must be finite and nonnegative")
    budgets = sorted(set(args.num_steps))
    if budgets[0] != 200:
        parser.error("The 200-update budget is required; 500 is the only optional alternative")
    verified_source = source_identity(args.model_path, args.model_revision)
    source_config_bytes = (Path(args.model_path) / "config.json").read_bytes()
    source_config = json.loads(source_config_bytes)
    text_config = source_config.get("text_config", source_config)
    source_dtype = text_config.get(
        "dtype",
        text_config.get(
            "torch_dtype", source_config.get("dtype", source_config.get("torch_dtype"))
        ),
    )
    if (
        source_config.get("quantization_config") is not None
        or text_config.get("quantization_config") is not None
    ):
        parser.error("The source must be the unquantized BF16 Qwen3.8-27B checkpoint")
    if source_dtype not in ("bfloat16", "torch.bfloat16"):
        parser.error("The source configuration must declare BF16 weights")
    if source_config.get("model_type") != "qwen3_5" or not any(
        name in ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
        for name in source_config.get("architectures", [])
    ):
        parser.error("Use the official Qwen3.8-27B checkpoint with its Qwen3_5 architecture")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "results.json"
    tokenizer = get_tokenizer(args.model_path)
    calibration, splits, data_metadata = prepare_data(tokenizer, args.dataset_revision, args.seed)
    report = {
        "model": MODEL_ID,
        "model_path": args.model_path,
        "model_revision": args.model_revision,
        "source_config_sha256": hashlib.sha256(source_config_bytes).hexdigest(),
        "verified_source": verified_source,
        "code_commit": source_revision(Path(__file__).resolve().parents[2]),
        "hostname": platform.node(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets")
        },
        "data": data_metadata,
        "budgets": budgets,
        "validation_tolerance": args.validation_tolerance,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "formats": {},
        "generation_requests": 0,
        "autoregressive_repeats": 0,
    }
    identity = {
        "code_commit": report["code_commit"],
        "model_revision": args.model_revision,
        "verified_source": verified_source,
        "source_config_sha256": report["source_config_sha256"],
        "data": data_metadata,
        "budgets": budgets,
        "validation_tolerance": args.validation_tolerance,
        "float32_matmul_precision": report["float32_matmul_precision"],
        "source_files": {
            str(path.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[2] / "modelopt/torch/quantization/ggml/gptq.py",
                *sorted(
                    (Path(__file__).resolve().parents[2] / "modelopt/torch/quantization/ggml").glob(
                        "gsq*.py"
                    )
                ),
            )
        },
    }
    identity_path = args.output_dir / "run-identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError(
            "Existing experiment outputs belong to different data, settings, or source"
        )
    if report_path.exists():
        report["formats"] = json.loads(report_path.read_text())["formats"]
    _write_report(identity_path, identity)
    _write_report(report_path, report)
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Run the quantization experiment on a CUDA GPU allocation")
    report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    for format_name in dict.fromkeys(args.formats):
        if "improved" in report["formats"].get(format_name, {}):
            continue
        torch.manual_seed(args.seed)
        model = get_model(args.model_path, attn_implementation="eager").eval()
        if model.get_input_embeddings().weight.dtype != torch.bfloat16 or any(
            module.weight.dtype != torch.bfloat16
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and "mlp" in name
            and "vision" not in name
            and "visual" not in name
        ):
            raise ValueError("The loaded language model must have BF16 embeddings and MLP weights")
        device = model.get_input_embeddings().weight.device
        config = load_recipe(f"general/ptq/{format_name}").quantize.model_dump(exclude_unset=True)
        config["quant_cfg"].extend(
            [
                {"quantizer_name": "*visual*", "enable": False},
                {"quantizer_name": "*vision*", "enable": False},
            ]
        )

        @torch.no_grad()
        def forward_loop(current_model):
            for sequence in calibration:
                current_model(sequence.unsqueeze(0).to(device), use_cache=False)

        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
        started = time.perf_counter()
        checkpoint_dir = args.output_dir / format_name
        final_checkpoint = checkpoint_dir / "result.pt"
        if final_checkpoint.exists():
            result = IQGSQResult.load(final_checkpoint)
            mtq.quantize(model, {**config, "algorithm": None}, forward_loop=None)
            result.activate(model, 0)
        else:
            result = quantize_iq_gsq(
                model,
                config,
                forward_loop,
                num_steps=tuple(budgets),
                seed=args.seed,
                checkpoint_dir=checkpoint_dir,
            )
            result.save(final_checkpoint)
        quantized_names = [
            name
            for name, module in model.named_modules()
            if is_quantized_linear(module) and module.weight_quantizer.is_enabled
        ]
        if not quantized_names or any(
            "mlp" not in name or "visual" in name or "vision" in name for name in quantized_names
        ):
            raise RuntimeError("The experiment requires quantization of language MLPs only")
        entry = {
            "recipe": f"general/ptq/{format_name}",
            "quantization_config": config,
            "quantization_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": {
                str(i): torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
            },
            "quantized_linears": quantized_names,
            "model_dtype": str(next(model.parameters()).dtype),
            "local_reconstruction": result.metrics,
            "validation": {},
        }
        report["formats"][format_name] = entry
        _write_report(report_path, report)
        for budget in (0, *budgets):
            result.activate(model, budget)
            entry["validation"][str(budget)] = evaluate_perplexity(
                model, splits["validation"], device
            )
            _write_report(report_path, report)
        selected = budgets[0]
        for budget in budgets[1:]:
            if entry["validation"][str(budget)]["perplexity"] < (
                entry["validation"][str(selected)]["perplexity"] - args.validation_tolerance
            ):
                selected = budget
        entry["selected_updates"] = selected
        entry["test"] = {}
        for label, budget in (("gptq", 0), ("gsq", selected)):
            result.activate(model, budget)
            started = time.perf_counter()
            entry["test"][label] = evaluate_perplexity(model, splits["test"], device)
            entry["test"][label]["seconds"] = time.perf_counter() - started
            _write_report(report_path, report)
        baseline = entry["test"]["gptq"]["perplexity"]
        refined = entry["test"]["gsq"]["perplexity"]
        entry["perplexity_difference"] = refined - baseline
        entry["improved"] = refined < baseline
        _write_report(report_path, report)
        del result, model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
