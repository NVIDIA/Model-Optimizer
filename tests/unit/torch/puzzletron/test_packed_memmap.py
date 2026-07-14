import json

import numpy as np

from modelopt.torch.puzzletron.distillation.dataset import make_puzzletron_llm_dataset
from modelopt.torch.puzzletron.utils.data.dataloaders import create_train_dataloader
from modelopt.torch.puzzletron.utils.data.packed_memmap import PackedTokenMemmapDataset


def _write_cache(path, *, samples: int = 2, sequence_length: int = 8):
    tokens = np.arange(samples * (sequence_length + 1), dtype=np.uint32).reshape(
        samples, sequence_length + 1
    )
    tokens.tofile(path)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(
            {
                "status": "complete",
                "num_samples": samples,
                "seq_length": sequence_length,
            }
        )
    )
    return tokens


def test_packed_cache_can_return_a_configured_prefix_length(tmp_path):
    path = tmp_path / "tokens.bin"
    tokens = _write_cache(path)

    dataset = PackedTokenMemmapDataset(path, sequence_length=4)

    sample = dataset[0]
    assert sample["input_ids"].tolist() == tokens[0, :4].tolist()
    assert sample["targets"].tolist() == tokens[0, 1:5].tolist()


def test_train_dataloader_applies_block_size_to_packed_cache(tmp_path):
    path = tmp_path / "tokens.bin"
    _write_cache(path)

    dataloader = create_train_dataloader(
        seed=1,
        tokenizer=None,
        block_size=4,
        dataset_path="unused",
        content_field="unused",
        fim_rate=0.0,
        fim_spm_rate=0.0,
        micro_batch_size=2,
        packed_token_cache_path=path,
    )

    batch = next(iter(dataloader))
    assert batch["input_ids"].shape == (2, 4)
    assert batch["targets"].shape == (2, 4)


def test_global_kd_dataset_applies_sequence_length_to_packed_cache(tmp_path):
    path = tmp_path / "tokens.bin"
    _write_cache(path)

    dataset = make_puzzletron_llm_dataset(
        tokenizer=None,
        dataset_path="unused",
        num_samples=2,
        seq_length=4,
        packed_token_cache_path=str(path),
    )

    sample = next(iter(dataset))
    assert sample["input_ids"].shape == (4,)
    assert sample["labels"].shape == (4,)
