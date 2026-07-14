"""Verify and summarize the stage-wide cross-model conversion artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import load_model_config
from modelopt.torch.puzzletron.stages.convert import _register_automodel_config_aliases


_VISION_MARKERS = (
    ".visual.",
    ".vision_",
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
    "embed_vision",
)
_MTP_MARKERS = ("mtp.", ".mtp.", "nextn", "multi_token")


def _load_descriptor_config(checkpoint: Path, descriptor):
    _register_automodel_config_aliases()
    return load_model_config(
        checkpoint,
        trust_remote_code=descriptor.requires_trust_remote_code(),
    )


def _weight_map(path: Path) -> dict[str, str]:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if index_path.is_file():
            mapping = dict(json.loads(index_path.read_text()).get("weight_map") or {})
            if not mapping:
                raise ValueError(f"{index_path} has an empty weight_map")
            return mapping
    single = path / "model.safetensors"
    if single.is_file():
        from safetensors import safe_open

        with safe_open(single, framework="pt", device="cpu") as stream:
            return {key: single.name for key in stream.keys()}
    binary = path / "pytorch_model.bin"
    if binary.is_file():
        return {"<pytorch_model.bin>": binary.name}
    raise FileNotFoundError(f"no standard HF weight file or index under {path}")


def _language_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("text_config") or config.get("language_config") or config)


def verify_checkpoint(
    checkpoint: str | Path,
    *,
    model_id: str,
    multimodal: bool,
    mtp_expected: bool,
    output_embedding: str = "lm_head",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{model_id}: missing {config_path}")
    config = json.loads(config_path.read_text())
    blocks = list(config.get("block_configs") or [])
    expected_layers = int(_language_config(config).get("num_hidden_layers", 0))
    if not blocks or len(blocks) != expected_layers:
        raise ValueError(
            f"{model_id}: block_configs={len(blocks)} does not match layers={expected_layers}"
        )
    weights = _weight_map(checkpoint)
    for filename in sorted(set(weights.values())):
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError(f"{model_id}: indexed weight shard is missing: {filename}")
    if not any(
        (checkpoint / name).is_file()
        for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
    ):
        raise FileNotFoundError(f"{model_id}: converted checkpoint has no tokenizer assets")

    lowered = {key.lower() for key in weights}
    vision_keys = [
        key for key in lowered if any(marker in key for marker in _VISION_MARKERS)
    ]
    mtp_keys = [key for key in lowered if any(marker in key for marker in _MTP_MARKERS)]
    language_config = _language_config(config)
    nextn_layers = int(language_config.get("num_nextn_predict_layers", 0) or 0)
    if nextn_layers:
        base_layer = int(language_config["num_hidden_layers"])
        folded_prefixes = tuple(
            f"model.layers.{layer_idx}."
            for layer_idx in range(base_layer, base_layer + nextn_layers)
        )
        mtp_keys.extend(
            key for key in lowered if key.startswith(folded_prefixes)
        )
        mtp_keys = sorted(set(mtp_keys))
    if multimodal:
        if not vision_keys:
            raise ValueError(f"{model_id}: multimodal checkpoint has no vision weights")
        if not any(
            (checkpoint / name).is_file()
            for name in (
                "preprocessor_config.json",
                "processor_config.json",
                "image_processor_config.json",
            )
        ):
            raise FileNotFoundError(f"{model_id}: multimodal processor assets are missing")
    if mtp_expected and not mtp_keys:
        raise ValueError(f"{model_id}: preflight declared MTP/next-token assets but no weights exist")
    output_weight = f"{output_embedding}.weight"
    if config.get("tie_word_embeddings") is False and output_weight not in weights:
        raise ValueError(
            f"{model_id}: untied checkpoint is missing descriptor output {output_weight}"
        )
    return {
        "model_id": model_id,
        "checkpoint": str(checkpoint),
        "num_blocks": len(blocks),
        "weight_count": len(weights),
        "weight_shards": sorted(set(weights.values())),
        "vision_weight_count": len(vision_keys),
        "mtp_weight_count": len(mtp_keys),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "output_embedding": output_embedding,
    }


def _compare_source_keys(
    output: Path,
    *,
    hf_id: str,
    revision: str,
    key_rewrites: tuple[tuple[str, str], ...] = (),
    allowed_added: tuple[str, ...] = ("lm_head.weight",),
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from modelopt.torch.puzzletron.anymodel.converter.generic_decoder import (
        rewrite_checkpoint_key,
    )

    source = Path(snapshot_download(repo_id=hf_id, revision=revision))
    source_keys = {
        rewrite_checkpoint_key(key, key_rewrites) for key in _weight_map(source)
    }
    output_keys = set(_weight_map(output))
    missing = sorted(source_keys - output_keys)
    unexpected = sorted(output_keys - source_keys - set(allowed_added))
    if missing or unexpected:
        raise ValueError(
            f"{hf_id}: conversion key mismatch missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    return {
        "source_checkpoint": str(source),
        "source_weight_count": len(source_keys),
        "added_weights": sorted(output_keys - source_keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("puzzle_runs/clean/acceptance/2026-07-06-cross-model-stage-matrix"),
    )
    parser.add_argument("--skip-source-compare", action="store_true")
    args = parser.parse_args()
    preflight = json.loads((args.root / "campaign" / "preflight.json").read_text())
    from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptorFactory
    summary = {
        "version": 1,
        "campaign_fingerprint": preflight["campaign_fingerprint"],
        "models": [],
    }
    for record in preflight["models"]:
        model_id = record["model_id"]
        config = yaml.safe_load((args.root / "configs" / f"{model_id}.yaml").read_text())
        descriptor = ModelDescriptorFactory.get(config["descriptor"])
        model_config = _load_descriptor_config(
            args.root / "models" / model_id / "ckpts" / "teacher", descriptor
        )
        contract = descriptor.generic_decoder_contract(model_config)
        key_rewrites = contract.checkpoint_key_rewrites if contract is not None else ()
        output_embedding = descriptor.output_embedding_name()
        manifest_path = args.root / "models" / model_id / "manifests" / "convert.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{model_id}: conversion manifest is missing")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") not in {"success", "skipped"}:
            raise ValueError(f"{model_id}: conversion status is {manifest.get('status')!r}")
        checkpoint = args.root / "models" / model_id / "ckpts" / "teacher"
        result = verify_checkpoint(
            checkpoint,
            model_id=model_id,
            multimodal=config["data"]["modality"] == "multimodal",
            mtp_expected=bool(record.get("mtp_fields")),
            output_embedding=output_embedding,
        )
        if not args.skip_source_compare:
            result["source_comparison"] = _compare_source_keys(
                checkpoint,
                hf_id=record["hf_id"],
                revision=record["immutable_revision"],
                key_rewrites=key_rewrites,
                allowed_added=(f"{output_embedding}.weight",),
            )
        summary["models"].append(result)
    output = args.root / "campaign" / "conversion_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
