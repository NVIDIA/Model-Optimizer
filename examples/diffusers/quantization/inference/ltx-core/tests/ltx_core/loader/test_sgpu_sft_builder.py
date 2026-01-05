from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from ltx_core.loader.registry import StateDictRegistry
from ltx_core.loader.sd_keys_ops import LTXV_LORA_COMFY_RENAMING_MAP, LTXV_MODEL_COMFY_RENAMING_MAP
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator

MODEL_PATH = "packages/ltx-core/tests/ltx_core/loader/assets/model-transformer_block.7.attn1.to_v.sft"
LORA_PATH = "packages/ltx-core/tests/ltx_core/loader/assets/lora-transformer_block.7.attn1.to_v.sft"


# Fixtures for float values
@pytest.fixture
def tolerance() -> float:
    """Tolerance value for tensor comparisons."""
    return 1e-3


@pytest.fixture
def expected_tensor_values() -> dict[str, float]:
    """Expected tensor values at specific indices."""
    return {
        "vanilla_0_0": 0.0275,
        "vanilla_2048_2048": -0.0830,
        "single_lora_0_0": 0.0225,
        "single_lora_2048_2048": -0.0688,
        "double_lora_0_0": 0.0175,
        "double_lora_2048_2048": -0.0546,
    }


@pytest.fixture
def model_dimensions() -> dict[str, int]:
    """Model dimension constants."""
    return {
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "weight_shape_0": 4096,
        "weight_shape_1": 4096,
        "inner_dim": 32 * 128,
    }


@pytest.fixture
def lora_scale() -> float:
    """LoRA scale value."""
    return 1.0


# Fixtures for string values
@pytest.fixture
def device() -> torch.device:
    """Device for model operations."""
    return torch.device("cpu")


@pytest.fixture
def state_dict_keys() -> dict[str, str]:
    """State dict key names."""
    return {
        "vanilla_weight": "model.diffusion_model.transformer_blocks.7.attn1.to_v.weight",
        "renamed_weight": "transformer_blocks.7.attn1.to_v.weight",
        "model_weight": "transformer.blocks.7.attn1.to_v.weight",
        "lora_A": "transformer_blocks.7.attn1.to_v.lora_A.weight",
        "lora_B": "transformer_blocks.7.attn1.to_v.lora_B.weight",
    }


@pytest.fixture
def metadata_keys() -> dict[str, str]:
    """Metadata dictionary keys."""
    return {
        "transformer": "transformer",
        "num_attention_heads": "num_attention_heads",
        "attention_head_dim": "attention_head_dim",
    }


@pytest.fixture
def tensor_indices() -> dict[str, tuple[int, int]]:
    """Tensor index tuples for assertions."""
    return {
        "top_left": (0, 0),
        "center": (2048, 2048),
    }


def test_sft_metadata_loading(metadata_keys: dict[str, str], model_dimensions: dict[str, int]) -> None:
    builder = Builder(
        model_path=MODEL_PATH,
        model_class_configurator=LTXModelConfigurator,
    )
    metadata = builder.model_loader.metadata(MODEL_PATH)
    assert metadata_keys["transformer"] in metadata
    transformer_config = metadata[metadata_keys["transformer"]]
    assert transformer_config[metadata_keys["num_attention_heads"]] == model_dimensions["num_attention_heads"]
    assert transformer_config[metadata_keys["attention_head_dim"]] == model_dimensions["attention_head_dim"]


def test_metamodel_creation(model_dimensions: dict[str, int]) -> None:
    builder = Builder(
        model_path=MODEL_PATH,
        model_class_configurator=LTXModelConfigurator,
    )
    transformer_cfg = builder.model_config()
    meta_transformer = builder.meta_model(transformer_cfg)
    assert isinstance(meta_transformer, LTXModel)
    assert meta_transformer.inner_dim == model_dimensions["inner_dim"]


def test_model_state_dict_loading(
    device: torch.device,
    state_dict_keys: dict[str, str],
    model_dimensions: dict[str, int],
    tensor_indices: dict[str, tuple[int, int]],
    expected_tensor_values: dict[str, float],
    tolerance: float,
) -> None:
    registry = StateDictRegistry()
    builder = Builder(
        model_path=MODEL_PATH,
        model_class_configurator=LTXModelConfigurator,
        registry=registry,
    )
    model_sd = builder.load_sd(
        [MODEL_PATH],
        registry=builder.registry,
        device=device,
    )
    model_renamed_sd = builder.load_sd(
        [MODEL_PATH],
        sd_key_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        registry=builder.registry,
        device=device,
    )
    assert model_sd.sd is not None
    assert len(model_sd.sd) == 1
    assert state_dict_keys["vanilla_weight"] in model_sd.sd
    to_v = model_sd.sd[state_dict_keys["vanilla_weight"]]
    assert to_v.shape == (model_dimensions["weight_shape_0"], model_dimensions["weight_shape_1"])
    assert torch.isclose(
        to_v[tensor_indices["top_left"]].float(),
        torch.tensor(expected_tensor_values["vanilla_0_0"]),
        atol=tolerance,
    )
    assert torch.isclose(
        to_v[tensor_indices["center"]].float(),
        torch.tensor(expected_tensor_values["vanilla_2048_2048"]),
        atol=tolerance,
    )
    assert model_renamed_sd.sd is not None
    assert len(model_renamed_sd.sd) == 1
    assert state_dict_keys["renamed_weight"] in model_renamed_sd.sd
    to_v_renamed = model_renamed_sd.sd[state_dict_keys["renamed_weight"]]
    assert torch.allclose(to_v, to_v_renamed, atol=tolerance)
    vanilla_id = registry.get([MODEL_PATH], None)
    assert vanilla_id is not None
    renamed_id = registry.get([MODEL_PATH], LTXV_MODEL_COMFY_RENAMING_MAP)
    assert renamed_id is not None
    assert renamed_id != vanilla_id


def test_lora_state_dict_loading(device: torch.device, state_dict_keys: dict[str, str]) -> None:
    builder = Builder(
        model_path=MODEL_PATH,
        model_class_configurator=LTXModelConfigurator,
    )
    lora_sd = builder.load_sd(
        [LORA_PATH],
        registry=builder.registry,
        device=device,
        sd_key_ops=LTXV_LORA_COMFY_RENAMING_MAP,
    )
    assert lora_sd.sd is not None
    assert len(lora_sd.sd) == 2
    assert state_dict_keys["lora_A"] in lora_sd.sd
    assert state_dict_keys["lora_B"] in lora_sd.sd


def test_vanilla_model_building(
    device: torch.device, state_dict_keys: dict[str, str], model_dimensions: dict[str, int]
) -> None:
    model = Builder(
        model_path=MODEL_PATH,
        model_class_configurator=LTXModelConfigurator,
    ).build(device=device)
    assert isinstance(model, LTXModel)
    assert model.inner_dim == model_dimensions["inner_dim"]
    for name, param in model.named_parameters():
        if name != state_dict_keys["model_weight"]:
            continue
        assert str(param.device) == str(device)


def test_model_with_single_lora_building(
    device: torch.device,
    state_dict_keys: dict[str, str],
    tensor_indices: dict[str, tuple[int, int]],
    expected_tensor_values: dict[str, float],
    tolerance: float,
    lora_scale: float,
) -> None:
    builder = Builder(
        model_path=MODEL_PATH,
        model_sd_key_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        model_class_configurator=LTXModelConfigurator,
    )
    model_with_lora = builder.lora(LORA_PATH, lora_scale, LTXV_LORA_COMFY_RENAMING_MAP).build(device=device)
    for name, param in model_with_lora.named_parameters():
        if name != state_dict_keys["renamed_weight"]:
            continue
        assert torch.isclose(
            param[tensor_indices["top_left"]].float(),
            torch.tensor(expected_tensor_values["single_lora_0_0"]),
            atol=tolerance,
        )
        assert torch.isclose(
            param[tensor_indices["center"]].float(),
            torch.tensor(expected_tensor_values["single_lora_2048_2048"]),
            atol=tolerance,
        )


def test_model_and_registry_with_multiple_loras_building(
    device: torch.device,
    state_dict_keys: dict[str, str],
    tensor_indices: dict[str, tuple[int, int]],
    expected_tensor_values: dict[str, float],
    tolerance: float,
    lora_scale: float,
) -> None:
    registry = StateDictRegistry()
    builder = Builder(
        model_path=MODEL_PATH,
        model_sd_key_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        model_class_configurator=LTXModelConfigurator,
        registry=registry,
    )
    with TemporaryDirectory() as temp_dir:
        lora_path = str(Path(temp_dir) / "lora.sft")
        Path(lora_path).write_bytes(Path(LORA_PATH).read_bytes())
        single_lora_builder = builder.lora(LORA_PATH, lora_scale, LTXV_LORA_COMFY_RENAMING_MAP)
        dbl_lora_builder = builder.lora(LORA_PATH, lora_scale, LTXV_LORA_COMFY_RENAMING_MAP).lora(
            lora_path, lora_scale, LTXV_LORA_COMFY_RENAMING_MAP
        )
        single_lora_model = single_lora_builder.build(device=device)
        dbl_lora_model = dbl_lora_builder.build(device=device)
        for name, param in single_lora_model.named_parameters():
            if name != state_dict_keys["renamed_weight"]:
                continue
            assert torch.isclose(
                param[tensor_indices["top_left"]].float(),
                torch.tensor(expected_tensor_values["single_lora_0_0"]),
                atol=tolerance,
            )
            assert torch.isclose(
                param[tensor_indices["center"]].float(),
                torch.tensor(expected_tensor_values["single_lora_2048_2048"]),
                atol=tolerance,
            )
        for name, param in dbl_lora_model.named_parameters():
            if name != state_dict_keys["renamed_weight"]:
                continue
            assert torch.isclose(
                param[tensor_indices["top_left"]].float(),
                torch.tensor(expected_tensor_values["double_lora_0_0"]),
                atol=tolerance,
            )
            assert torch.isclose(
                param[tensor_indices["center"]].float(),
                torch.tensor(expected_tensor_values["double_lora_2048_2048"]),
                atol=tolerance,
            )
        assert len(registry._state_dicts) == 3
