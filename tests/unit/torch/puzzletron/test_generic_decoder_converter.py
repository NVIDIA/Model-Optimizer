import json
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from modelopt.torch.puzzletron.anymodel.converter.generic_decoder import (
    GenericDecoderConverter,
    rewrite_safetensor_checkpoint_keys,
)
from modelopt.torch.puzzletron.anymodel.models.qwen3_5.qwen3_5_converter import Qwen3P5Converter


def test_gpt_oss_converter_preserves_explicit_full_and_sliding_windows() -> None:
    config = SimpleNamespace(
        model_type="gpt_oss",
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=128,
        num_local_experts=4,
        intermediate_size=32,
        num_experts_per_tok=2,
    )
    from modelopt.torch.puzzletron.anymodel.models.gpt_oss.gpt_oss_model_descriptor import (
        GptOssModelDescriptor,
    )

    blocks = GenericDecoderConverter.create_block_configs(GptOssModelDescriptor, config)

    assert [
        block.require_subblock("attention").sliding_window_size for block in blocks
    ] == [128, "full"]


def test_qwen_moe_converter_uses_moe_instead_of_dense_ffn() -> None:
    text_config = SimpleNamespace(
        num_hidden_layers=2,
        layer_types=("linear_attention", "full_attention"),
        linear_key_head_dim=128,
        linear_num_value_heads=16,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_conv_kernel_dim=4,
        num_key_value_heads=2,
        num_attention_heads=16,
        num_experts=64,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=1024,
    )

    blocks = Qwen3P5Converter.create_block_configs_from_main_config(
        SimpleNamespace(text_config=text_config)
    )

    assert blocks[0].get_subblock("mamba") is not None
    assert blocks[1].get_subblock("attention") is not None
    assert all(block.get_subblock("ffn") is None for block in blocks)
    assert all(block.get_subblock("moe").num_experts == 64 for block in blocks)


def test_generic_converter_rewrites_legacy_checkpoint_keys_without_mutating_source(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    tensors = {
        "language_model.model.embed_tokens.weight": torch.arange(6).reshape(2, 3),
        "vision_tower.patch_conv.weight": torch.arange(4).reshape(2, 2),
    }
    save_file(tensors, source / "model.safetensors", metadata={"format": "pt"})
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 10},
                "weight_map": dict.fromkeys(tensors, "model.safetensors"),
            }
        )
    )
    GenericDecoderConverter.copy_checkpoint_files(source, output)

    rewritten = rewrite_safetensor_checkpoint_keys(
        output,
        ((r"^language_model\.model\.", "model.language_model."),),
    )

    assert rewritten == {"language_model.model.embed_tokens.weight": "model.language_model.embed_tokens.weight"}
    assert set(load_file(source / "model.safetensors")) == set(tensors)
    assert set(load_file(output / "model.safetensors")) == {
        "model.language_model.embed_tokens.weight",
        "vision_tower.patch_conv.weight",
    }
    output_index = json.loads((output / "model.safetensors.index.json").read_text())
    assert output_index["metadata"] == {"total_size": 10}
    assert set(output_index["weight_map"]) == set(load_file(output / "model.safetensors"))
