import json
from pathlib import Path

from examples.puzzletron.verify_cross_model_conversions import (
    _load_descriptor_config,
    verify_checkpoint,
)


def _checkpoint(root: Path, *, multimodal: bool, mtp: bool) -> Path:
    root.mkdir(parents=True)
    config = {
        "num_hidden_layers": 2,
        "block_configs": [
            {"subblock_configs": [{"kind": "ffn", "name": "ffn", "intermediate_size": 8}]},
            {"subblock_configs": [{"kind": "ffn", "name": "ffn", "intermediate_size": 8}]},
        ],
        "tie_word_embeddings": False,
    }
    (root / "config.json").write_text(json.dumps(config))
    keys = {
        "model.embed_tokens.weight": "model.safetensors",
        "lm_head.weight": "model.safetensors",
    }
    if multimodal:
        keys["model.visual.patch_embed.weight"] = "model.safetensors"
        (root / "preprocessor_config.json").write_text("{}\n")
    if mtp:
        keys["mtp.layers.0.mlp.down_proj.weight"] = "model.safetensors"
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": keys})
    )
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "tokenizer_config.json").write_text("{}\n")
    return root


def test_conversion_verifier_checks_blocks_vit_mtp_and_weight_shards(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "teacher", multimodal=True, mtp=True)

    result = verify_checkpoint(
        checkpoint,
        model_id="vlm",
        multimodal=True,
        mtp_expected=True,
    )

    assert result["num_blocks"] == 2
    assert result["vision_weight_count"] == 1
    assert result["mtp_weight_count"] == 1


def test_conversion_verifier_rejects_missing_index_shard(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "teacher", multimodal=False, mtp=False)
    (checkpoint / "model.safetensors").unlink()

    try:
        verify_checkpoint(
            checkpoint,
            model_id="text",
            multimodal=False,
            mtp_expected=False,
        )
    except FileNotFoundError as error:
        assert "model.safetensors" in str(error)
    else:
        raise AssertionError("a missing indexed weight shard must fail conversion verification")


def test_conversion_verifier_detects_nextn_as_extra_decoder_layer(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "teacher", multimodal=False, mtp=False)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config["num_nextn_predict_layers"] = 1
    config_path.write_text(json.dumps(config))
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["model.layers.2.shared_head.head.weight"] = "model.safetensors"
    index_path.write_text(json.dumps(index))

    result = verify_checkpoint(
        checkpoint,
        model_id="nextn",
        multimodal=False,
        mtp_expected=True,
    )

    assert result["mtp_weight_count"] == 1


def test_conversion_verifier_uses_descriptor_owned_output_embedding(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "teacher", multimodal=False, mtp=False)
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["language_model.lm_head.weight"] = index["weight_map"].pop(
        "lm_head.weight"
    )
    index_path.write_text(json.dumps(index))

    result = verify_checkpoint(
        checkpoint,
        model_id="namespaced",
        multimodal=False,
        mtp_expected=False,
        output_embedding="language_model.lm_head",
    )

    assert result["output_embedding"] == "language_model.lm_head"


def test_conversion_verifier_registers_automodel_aliases_before_loading(
    tmp_path: Path, monkeypatch
) -> None:
    events = []

    class Descriptor:
        @staticmethod
        def requires_trust_remote_code():
            return False

    monkeypatch.setattr(
        "examples.puzzletron.verify_cross_model_conversions._register_automodel_config_aliases",
        lambda: events.append("register"),
    )
    monkeypatch.setattr(
        "examples.puzzletron.verify_cross_model_conversions.load_model_config",
        lambda *args, **kwargs: events.append("load") or {"model_type": "qwen3_5"},
    )

    assert _load_descriptor_config(tmp_path, Descriptor)["model_type"] == "qwen3_5"
    assert events == ["register", "load"]
