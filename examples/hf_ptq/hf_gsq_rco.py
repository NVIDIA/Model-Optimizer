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

"""Weight-only Qwen AutoQuant comparisons with frozen data and exact cached-teacher KL."""

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import re
import time
from contextlib import ExitStack
from fnmatch import fnmatch
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from datasets import load_dataset
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.recipe.presets import QUANT_CFG_CHOICES
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.export.model_utils import get_language_model_from_vl
from modelopt.torch.quantization.config import (
    GPTQCalibConfig,
    GSQCalibConfig,
    MaxCalibConfig,
    QuantizeConfig,
)
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

__all__ = ["CachedTeacherKL", "freeze_samples"]

TARGETS = (
    4.511470881608226,
    4.899085228096201,
    5.286699574584175,
    5.674313921072151,
    6.061928267560125,
    6.4495426140481005,
    6.837156960536076,
    7.22477130702405,
    7.612385653512025,
    8.0,
)
VARIANTS = ("ordinary", "gsq", "rco", "gsq_rco")


def _digest(value):
    return hashlib.sha256(
        json.dumps(json.loads(json.dumps(value)), sort_keys=True).encode()
    ).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_samples(records, tokenizer, directory, identity, count=512, max_length=4096, seed=0):
    """Persist exact token/mask tensors, order and source provenance for reproducible reuse."""
    if count < 1 or not 2 <= max_length <= 4096:
        raise ValueError("Require positive sample count and sequence length in [2, 4096]")
    directory = Path(directory)
    tokenizer_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    if identity.get("tokenizer_sha256", tokenizer_hash) != tokenizer_hash:
        raise ValueError("Declared tokenizer content differs from the supplied tokenizer")
    settings = {
        **identity,
        "tokenizer_sha256": tokenizer_hash,
        "count": count,
        "max_length": max_length,
        "seed": seed,
        "preprocessing": "unformatted-text;add_special_tokens=False;random-document-crop-v1",
    }
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        # Exact tokenizer content/revision remains authoritative when its location changes.
        previous = {k: v for k, v in manifest["settings"].items() if k != "tokenizer"}
        current = {k: v for k, v in settings.items() if k != "tokenizer"}
        if previous != current or manifest["sha256"] != _file_hash(directory / "tokens.pt"):
            raise ValueError("Frozen data identity or content differs")
        return torch.load(directory / "tokens.pt", weights_only=True), manifest
    rng = random.Random(seed)
    batches, sources = [], []
    for row, record in enumerate(records):
        text = record["text"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < 2:
            continue
        start = rng.randrange(max(1, len(ids) - max_length + 1))
        tokens = torch.tensor([ids[start : start + max_length]], dtype=torch.long)
        batches.append(
            {
                "sample_id": len(batches),
                "input_ids": tokens,
                "attention_mask": torch.ones_like(tokens),
                "loss_mask": torch.ones_like(tokens),
            }
        )
        sources.append(
            {
                "row": row,
                "url": record.get("url"),
                "timestamp": record.get("timestamp"),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "token_start": start,
            }
        )
        if len(batches) == count:
            break
    if len(batches) != count:
        raise ValueError(f"Only found {len(batches)} usable documents, requested {count}")
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(batches, directory / "tokens.pt")
    manifest = {
        "settings": settings,
        "sources": sources,
        "sha256": _file_hash(directory / "tokens.pt"),
    }
    manifest["fingerprint"] = _digest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return batches, manifest


def _decoder(model):
    lineage = get_language_model_from_vl(model, strict=True)
    return lineage[-1] if lineage else model.base_model


def _inputs(batch, device):
    return {key: batch[key].to(device) for key in ("input_ids", "attention_mask")}


class CachedTeacherKL:
    """Existing forward_backward callback with exact token-chunked full-vocabulary KL."""

    def __init__(self, directory, fingerprint, chunk_tokens=32):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if manifest["fingerprint"] != fingerprint or chunk_tokens < 1:
            raise ValueError("Teacher cache identity or token chunk size is incompatible")
        if any(
            _file_hash(self.directory / name) != info["sha256"]
            for name, info in manifest["files"].items()
        ):
            raise ValueError("Teacher cache content differs from its manifest")
        self.chunk_tokens = chunk_tokens
        self.mean_valid_tokens = manifest["mean_valid_tokens"]
        self.mean_prediction_tokens = manifest["mean_prediction_tokens"]

    @classmethod
    def prepare(cls, model, batches, directory, fingerprint, chunk_tokens=32):
        """Cache original teacher head inputs in their exact dtype, or validate an existing cache."""
        directory = Path(directory)
        if not (directory / "manifest.json").exists():
            directory.mkdir(parents=True, exist_ok=True)
            device = next(model.parameters()).device
            with torch.no_grad():
                files = {}
                valid_tokens = []
                prediction_tokens = []
                for index, batch in enumerate(batches):
                    valid_tokens.append(int(batch.get("loss_mask", batch["attention_mask"]).sum()))
                    mask = batch.get("loss_mask", batch["attention_mask"])
                    prediction_tokens.append(
                        int((mask[:, 1:] * batch["attention_mask"][:, :-1]).sum())
                    )
                    hidden = _decoder(model)(
                        **_inputs(batch, device), use_cache=False
                    ).last_hidden_state
                    if index == 0:
                        direct = model(
                            **_inputs(batch, device), use_cache=False, logits_to_keep=1
                        ).logits
                        torch.testing.assert_close(
                            model.get_output_embeddings()(hidden[:, -1:]), direct
                        )
                    path = directory / f"{batch['sample_id']}.pt"
                    torch.save(hidden.cpu(), path)
                    files[path.name] = {
                        "sha256": _file_hash(path),
                        "dtype": str(hidden.dtype),
                        "shape": list(hidden.shape),
                    }
            (directory / "manifest.json").write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "files": files,
                        "mean_valid_tokens": sum(valid_tokens) / len(valid_tokens),
                        "mean_prediction_tokens": sum(prediction_tokens) / len(prediction_tokens),
                    }
                )
            )
        instance = cls(directory, fingerprint, chunk_tokens)
        model.get_output_embeddings().requires_grad_(False)
        return instance

    def __call__(self, model, batch):
        """Accumulate head gradients by token chunk, then backward through the decoder once."""
        return self.metric(model, batch, backward=True, normalizer=self.mean_valid_tokens)

    def cross_entropy(self, model, batch):
        """Score ordinary AutoQuant controls with causal CE and one decoder backward."""
        return self.metric(
            model, batch, backward=True, normalizer=self.mean_prediction_tokens, cross_entropy=True
        )

    def metric(self, model, batch, backward=False, normalizer=None, cross_entropy=False):
        """Evaluate the same full-vocabulary objective, optionally performing its backward."""
        device = next(model.parameters()).device
        with torch.set_grad_enabled(backward):
            student = _decoder(model)(**_inputs(batch, device), use_cache=False).last_hidden_state
        return self._metric_from_hidden(
            model,
            batch,
            student,
            backward=backward,
            normalizer=normalizer,
            cross_entropy=cross_entropy,
        )

    def _metric_from_hidden(
        self, model, batch, student, backward=False, normalizer=None, cross_entropy=False
    ):
        device = student.device
        mask = batch.get("loss_mask", batch["attention_mask"])
        if cross_entropy:
            student = student[:, :-1]
            labels = batch["input_ids"][:, 1:].to(device).flatten()
            mask = mask[:, 1:] * batch["attention_mask"][:, :-1]
        else:
            teacher = torch.load(
                self.directory / f"{batch['sample_id']}.pt", map_location="cpu", weights_only=True
            ).flatten(0, -2)
        leaf = student.detach().requires_grad_(backward)
        flat = leaf.flatten(0, -2)
        mask = mask.to(device).flatten()
        count = mask.sum()
        if not bool(count):
            raise ValueError("Objective batch has no valid tokens")
        metric = torch.zeros((), device=device)
        head = model.get_output_embeddings()
        for start in range(0, flat.shape[0], self.chunk_tokens):
            stop = start + self.chunk_tokens
            with torch.set_grad_enabled(backward):
                logits = head(flat[start:stop]).float()
                if cross_entropy:
                    loss = F.cross_entropy(logits, labels[start:stop], reduction="none")
                else:
                    with torch.no_grad():
                        teacher_logp = F.log_softmax(
                            head(teacher[start:stop].to(device)).float(), -1
                        )
                    loss = F.kl_div(
                        F.log_softmax(logits, -1), teacher_logp, log_target=True, reduction="none"
                    ).sum(-1)
                loss = (loss * mask[start:stop]).sum() / (
                    count if normalizer is None else normalizer
                )
                if backward:
                    loss.backward()
            metric += loss.detach()
        if backward and student.requires_grad:
            student.backward(leaf.grad)
        return metric.detach()


def _formats(initializer, gsq_options=None):
    formats = []
    for name in ("w4a16_nvfp4", "fp8"):
        config = copy.deepcopy(QUANT_CFG_CHOICES[name])
        config["quant_cfg"] = [
            entry for entry in config["quant_cfg"] if entry["quantizer_name"] != "*input_quantizer"
        ]
        if name == "w4a16_nvfp4":
            for entry in config["quant_cfg"]:
                if entry["quantizer_name"] == "*weight_quantizer":
                    entry["cfg"]["block_sizes"]["type"] = "static"
        config["algorithm"] = (
            {"method": "gsq", "initializer": initializer, **gsq_options}
            if gsq_options is not None
            else initializer
        )
        if gsq_options is not None and gsq_options.get("optimizer_checkpoint_dir"):
            path = Path(gsq_options["optimizer_checkpoint_dir"]) / name
            config["algorithm"]["optimizer_checkpoint_dir"] = str(path)
            config["algorithm"].setdefault("resume", (path / "manifest.pt").exists())
        algorithm_type = (
            GSQCalibConfig
            if gsq_options is not None
            else {"max": MaxCalibConfig, "gptq": GPTQCalibConfig}[initializer]
        )
        options = config["algorithm"] if gsq_options is not None else {"method": initializer}
        config["algorithm"] = algorithm_type(**options).model_dump()
        formats.append((QuantizeConfig(**config).model_dump(), name))
    return formats


def _candidate_identity(formats):
    formats = copy.deepcopy(formats)
    for config, _ in formats:
        algorithm = config["algorithm"]
        if isinstance(algorithm, dict) and algorithm.get("method") == "gsq":
            for key in ("resume", "optimizer_checkpoint_dir"):
                algorithm.pop(key, None)
            algorithm.get("layerwise", {}).pop("checkpoint_dir", None)
    return formats


def _disabled_layers(model):
    decoder_modules = set(_decoder(model).modules())
    protected = [
        entry["quantizer_name"]
        for entry in QUANT_CFG_CHOICES["fp8"]["quant_cfg"]
        if entry.get("enable") is False and entry["quantizer_name"] != "*"
    ]
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and (
            module not in decoder_modules
            or any(fnmatch(name + ".weight_quantizer", pattern) for pattern in protected)
        )
    ]


@torch.no_grad()
def _evaluate(model, batches, objective):
    device = next(model.parameters()).device
    kl, nll, valid_tokens, count = 0.0, 0.0, 0, 0
    for batch in batches:
        hidden = _decoder(model)(**_inputs(batch, device), use_cache=False).last_hidden_state
        mask = batch.get("loss_mask", batch["attention_mask"])
        valid = int(mask.sum())
        predictions = int((mask[:, 1:] * batch["attention_mask"][:, :-1]).sum())
        kl += float(objective._metric_from_hidden(model, batch, hidden)) * valid
        if predictions:
            nll += (
                float(objective._metric_from_hidden(model, batch, hidden, cross_entropy=True))
                * predictions
            )
        valid_tokens += valid
        count += predictions
    return {
        "perplexity": math.exp(nll / count),
        "teacher_kl": kl / valid_tokens,
        "prediction_tokens": count,
    }


def _load_model(args):
    # Both Auto classes are native HF loaders; the Qwen checkpoint is a VLM with a text decoder.
    loader = AutoModelForImageTextToText if args.multimodal else AutoModelForCausalLM
    model = loader.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    if args.pilot_layers:
        decoder = _decoder(model)
        decoder.layers = decoder.layers[: args.pilot_layers]
        decoder.config.num_hidden_layers = args.pilot_layers
        if hasattr(decoder.config, "layer_types"):
            decoder.config.layer_types = decoder.config.layer_types[: args.pilot_layers]
    return model


def _search(args, model, calibration, objective, formats, method, checkpoint, target):
    return mtq.auto_quantize(
        model,
        constraints={"effective_bits": target},
        quantization_formats=formats,
        data_loader=calibration,
        forward_step=lambda m, b: _decoder(m)(
            **_inputs(b, next(m.parameters()).device), use_cache=False
        ),
        forward_backward_step=objective.cross_entropy
        if method["method"] == "gradient"
        else objective,
        disabled_layers=_disabled_layers(model),
        num_calib_steps=len(calibration),
        num_score_steps=len(calibration),
        method=method,
        checkpoint=str(checkpoint),
        verbose=True,
    )


@torch.no_grad()
def _probe_logits(model, batch):
    hidden = _decoder(model)(
        **_inputs(batch, next(model.parameters()).device), use_cache=False
    ).last_hidden_state
    return model.get_output_embeddings()(hidden[:, -1:]).cpu()


def _reload_packed_for_validation(directory, *, multimodal=False, device="cuda:0"):
    """Decode published weight-only tensors into an ordinary model for parity validation."""
    directory = Path(directory)
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    quantization = getattr(config, "quantization_config", {})
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    loader = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    with init_empty_weights(include_buffers=False):
        model = loader.from_config(config, dtype=config.dtype, attn_implementation="sdpa")
    index_path = directory / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"] if index_path.exists() else None
    shards = set(weight_map.values()) if weight_map is not None else {"model.safetensors"}
    with ExitStack() as stack:
        readers = {
            name: stack.enter_context(
                safe_open(directory / name, framework="pt", device=str(device))
            )
            for name in shards
        }
        if weight_map is None:
            weight_map = dict.fromkeys(readers["model.safetensors"].keys(), "model.safetensors")
        expected = model.state_dict()
        missing = expected.keys() - weight_map.keys()
        if missing:
            raise ValueError(
                f"Published checkpoint is missing architecture tensors: {sorted(missing)}"
            )

        def read(name):
            return readers[weight_map[name]].get_tensor(name)

        for name, parameter in expected.items():
            value = read(name)
            if name.endswith(".weight") and value.dtype in {torch.uint8, torch.float8_e4m3fn}:
                prefix = name.removesuffix(".weight")
                scheme = quantization.get("quantized_layers", {}).get(prefix, quantization)
                if any(
                    prefix + suffix in weight_map for suffix in (".input_scale", ".pre_quant_scale")
                ):
                    raise ValueError(
                        "Packed validation requires weight-only tensors without input scaling"
                    )
                if value.dtype == torch.uint8 and scheme.get("quant_algo") == "W4A16_NVFP4":
                    value = NVFP4QTensor(parameter.shape, parameter.dtype, value).dequantize(
                        dtype=parameter.dtype,
                        scale=read(prefix + ".weight_scale"),
                        double_scale=read(prefix + ".weight_scale_2"),
                        block_sizes={-1: scheme["group_size"]},
                    )
                elif value.dtype == torch.float8_e4m3fn and scheme.get("quant_algo") == "FP8":
                    value = (value.float() * read(prefix + ".weight_scale").float()).to(
                        parameter.dtype
                    )
                else:
                    raise ValueError(f"Unsupported published weight scheme for {name}: {scheme}")
            set_module_tensor_to_device(
                model, name, device, value=value, dtype=value.dtype, clear_cache=False
            )
    return model.to(device).eval()


def _weight_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        for chunk in tensor.detach().reshape(-1).split(4 * 1024 * 1024):
            digest.update(chunk.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    """Freeze C4/WikiText and run matched weight-only variants at the exact historical ceilings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--c4-revision", required=True)
    parser.add_argument("--wikitext-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--initializer", choices=("max", "gptq"), default="max")
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument("--budget-index", type=int, action="append", choices=range(10))
    parser.add_argument("--gsq-options", type=json.loads, default={})
    parser.add_argument("--rco-options", type=json.loads, default={})
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-count", type=int, default=512)
    parser.add_argument("--heldout-count", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--head-chunk-tokens", type=int, default=32)
    parser.add_argument("--pilot-layers", type=int, default=0)
    parser.add_argument("--reload-atol", type=float, default=0.02)
    parser.add_argument("--reload-rtol", type=float, default=0.02)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        parser.error("This example requires exactly one visible CUDA GPU")
    if any(
        not re.fullmatch(r"[0-9a-f]{40,64}", revision)
        for revision in (
            args.model_revision,
            args.tokenizer_revision,
            args.c4_revision,
            args.wikitext_revision,
        )
    ):
        parser.error("Use exact model, tokenizer, and dataset commit revisions")
    preparation_start = time.perf_counter()
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {
        **vars(args),
        "output": str(args.output),
        "targets": TARGETS,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(torch.cuda.get_device_properties(0)),
        "control_objective": "valid-prediction-token-mean causal cross-entropy",
        "rco_objective": "valid-token-mean full-vocabulary teacher-to-student KL",
    }
    identity = _digest(
        {
            key: settings[key]
            for key in ("model", "model_revision", "tokenizer_revision", "pilot_layers")
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.tokenizer_revision)
    tokenizer_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    frozen = []
    for name, repo, config, split, revision, count in (
        ("calibration", "allenai/c4", "en", "train", args.c4_revision, args.calibration_count),
        (
            "heldout",
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            "test",
            args.wikitext_revision,
            args.heldout_count,
        ),
    ):
        source = {
            "dataset": repo,
            "config": config,
            "split": split,
            "revision": revision,
            "tokenizer": args.model,
            "tokenizer_revision": args.tokenizer_revision,
            "tokenizer_sha256": tokenizer_hash,
        }
        path = args.output / name
        records = (
            []
            if (path / "manifest.json").exists()
            else load_dataset(repo, config, split=split, revision=revision, streaming=True)
        )
        frozen.append(
            freeze_samples(records, tokenizer, path, source, count, args.max_length, args.seed)
        )
    (calibration, cal_manifest), (heldout, held_manifest) = frozen
    if {source["text_sha256"] for source in cal_manifest["sources"]} & {
        source["text_sha256"] for source in held_manifest["sources"]
    }:
        raise ValueError("Calibration and held-out documents overlap")
    model = _load_model(args)
    settings["original_weight_sha256"] = _weight_fingerprint(model)
    identity = _digest({"source": identity, "weights": settings["original_weight_sha256"]})
    objectives = [
        CachedTeacherKL.prepare(
            model,
            batches,
            args.output / f"{name}_teacher",
            _digest({"model": identity, "data": manifest["fingerprint"]}),
            args.head_chunk_tokens,
        )
        for name, (batches, manifest) in zip(("calibration", "heldout"), frozen, strict=True)
    ]
    settings["bf16"] = _evaluate(model, heldout, objectives[1])
    settings["preparation_seconds"] = time.perf_counter() - preparation_start
    settings["cache_bytes"] = sum(path.stat().st_size for path in args.output.rglob("*.pt"))
    (args.output / "manifest.json").write_text(json.dumps(settings, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if args.prepare_only:
        return
    allocations = {}
    for variant in args.variant or VARIANTS:
        gsq_options = {
            "seed": args.seed,
            "data_fingerprint": _digest({"model": identity, "data": cal_manifest["fingerprint"]}),
            "optimizer_checkpoint_dir": str(args.output / "gsq" / "optimizer"),
            **args.gsq_options,
        }
        formats = _formats(args.initializer, gsq_options if "gsq" in variant else None)
        family = "gsq" if "gsq" in variant else "ordinary"
        candidate_checkpoint = args.output / family / "candidates.pt"
        candidate_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = _digest(
            {
                "model": identity,
                "data": cal_manifest["fingerprint"],
                "formats": _candidate_identity(formats),
                "seed": args.seed,
                "scoring_objective": "shifted-token-ce-v1",
                "preserve_candidate_weights": True,
            }
        )
        preparation_path = candidate_checkpoint.with_suffix(".json")
        preparation = (
            json.loads(preparation_path.read_text()) if preparation_path.exists() else None
        )
        if preparation and preparation["data_fingerprint"] != fingerprint:
            raise ValueError("Candidate preparation model, data, objective or formats differ")
        if not candidate_checkpoint.exists() or preparation is None:
            model = _load_model(args)
            model.get_output_embeddings().requires_grad_(False)
            torch.cuda.reset_peak_memory_stats()
            candidate_start = time.perf_counter()
            _, candidate_state = _search(
                args,
                model,
                calibration,
                objectives[0],
                formats,
                {
                    "method": "gradient",
                    "data_fingerprint": fingerprint,
                    "preserve_candidate_weights": True,
                },
                candidate_checkpoint,
                16.0,
            )
            minimum = (
                16
                * sum(min(s["costs"]) for s in candidate_state["candidate_stats"].values())
                / candidate_state["cost_denominator"]
            )
            preparation_path.write_text(
                json.dumps(
                    {
                        "seconds": time.perf_counter() - candidate_start,
                        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                        "data_fingerprint": fingerprint,
                        "formats": formats,
                        "minimum_effective_bits": minimum,
                    },
                    indent=2,
                )
            )
            del candidate_state
            del model
            gc.collect()
            torch.cuda.empty_cache()
        for index in args.budget_index or range(10):
            directory = args.output / variant / f"budget_{index:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            result = {
                "variant": variant,
                "budget": f"budget_{index:02d}",
                "requested": TARGETS[index],
                "objective": settings["rco_objective" if "rco" in variant else "control_objective"],
            }
            metadata_path = candidate_checkpoint.with_suffix(".json")
            minimum = (
                json.loads(metadata_path.read_text()).get("minimum_effective_bits")
                if metadata_path.exists()
                else None
            )
            result["minimum_effective_bits"] = minimum
            if minimum is not None and TARGETS[index] < minimum - 1e-10:
                result.update({"status": "infeasible", "feasible": False})
                (directory / "result.json").write_text(json.dumps(result, indent=2))
                continue
            model = _load_model(args)
            model.get_output_embeddings().requires_grad_(False)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            method = (
                {
                    "method": "rco",
                    "seed": args.seed,
                    "data_fingerprint": fingerprint,
                    "candidate_checkpoint": str(candidate_checkpoint),
                    "objective_fingerprint": _digest(
                        {
                            "objective": "cached-fullvocab-head-kl-v1",
                            "model": identity,
                            "data": cal_manifest["fingerprint"],
                        }
                    ),
                    **args.rco_options,
                }
                if "rco" in variant
                else {
                    "method": "gradient",
                    "data_fingerprint": fingerprint,
                    "preserve_candidate_weights": True,
                }
            )
            try:
                checkpoint = directory / "search.pt" if "rco" in variant else candidate_checkpoint
                model, state = _search(
                    args,
                    model,
                    calibration,
                    objectives[0],
                    formats,
                    method,
                    checkpoint,
                    TARGETS[index],
                )
                torch.cuda.synchronize()
                result.update(
                    {
                        "optimization_seconds": time.perf_counter() - start,
                        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                        "achieved": state["best"]["constraints"]["effective_bits"],
                        "feasible": state["best"]["is_satisfied"],
                    }
                )
                result["resolved_method"] = method
                result["resolved_formats"] = formats
                if "rco" in variant:
                    result["continuous_expected_effective_bits"] = state["best"][
                        "continuous_expected_effective_bits"
                    ]
                    result["solver_calls"] = state["rco_state"]["solver_calls"]
                    result["effective_optimization_steps"] = state["rco_state"][
                        "effective_optimization_steps"
                    ]
                    result["rco_history"] = state["rco_state"]["history"]
                    result["resolved_schedule"] = state["rco_signature"]
                if not result["feasible"] or result["achieved"] > TARGETS[index] + 1e-10:
                    raise ValueError("Final discrete cost exceeds the requested ceiling")
                allocation = {name: str(recipe) for name, recipe in state["best"]["recipe"].items()}
                result["allocation"] = allocation
                result["allocation_sha256"] = _digest(allocation)
                result["duplicate_of"] = allocations.get((variant, result["allocation_sha256"]))
                allocations[(variant, result["allocation_sha256"])] = f"budget_{index:02d}"
                result["heldout"] = _evaluate(model, heldout, objectives[1])
                expected = _probe_logits(model, heldout[0])
                mto.save(model, directory / "modelopt.pt")
                export_hf_checkpoint(model, export_dir=directory / "hf")
                tokenizer.save_pretrained(directory / "hf")
                model = None
                gc.collect()
                torch.cuda.empty_cache()
                model = mto.restore(_load_model(args), directory / "modelopt.pt")
                restored = _probe_logits(model, heldout[0])
                torch.testing.assert_close(
                    restored, expected, atol=args.reload_atol, rtol=args.reload_rtol
                )
                result["modelopt_reload_max_abs_error"] = float((restored - expected).abs().max())
                model = None
                gc.collect()
                torch.cuda.empty_cache()
                model = _reload_packed_for_validation(directory / "hf", multimodal=args.multimodal)
                restored = _probe_logits(model, heldout[0])
                torch.testing.assert_close(
                    restored, expected, atol=args.reload_atol, rtol=args.reload_rtol
                )
                result["packed_reload_max_abs_error"] = float((restored - expected).abs().max())
                result["status"] = "completed_run"
                result["resume_verification"] = "not_run"
            except Exception as error:
                result.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            finally:
                result["total_seconds"] = time.perf_counter() - start
                result["total_peak_memory_bytes"] = torch.cuda.max_memory_allocated()
                (directory / "result.json").write_text(json.dumps(result, indent=2))
                model = None
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
