# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the vendored NeMo-AutoModel migration in the fastgen example.

Two tiers:

* **Environment-independent** tests (config repoint, no ``tools.*`` imports, and the
  removability coverage audit) run anywhere — they only read files in this example.
* **Dependency-guarded** tests (data builders, collate broadcast, checkpointer subclass,
  preprocessing registry) use ``pytest.importorskip`` so they skip where ``nemo_automodel`` /
  ``torch`` are absent (e.g. a CPU login node) and run in the training container.

The GPU/container end-to-end positive tests (smoke train, FSDP2 resume, preprocessing on real
images) are exercised via SLURM, not here; see ``round-*-summary`` / the run docs.
"""

from __future__ import annotations

import inspect
import pathlib
import re
import sys

import pytest

# Resolve the example dir (examples/diffusers/fastgen) from this test's location
# (tests/examples/diffusers/fastgen/) and put it on sys.path so ``fastgen_data`` /
# ``fastgen_checkpoint`` / ``preprocess`` import as top-level modules, exactly as
# dmd2_finetune.py / preprocess_qwen_image.py do.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))


# Maps each of the nine staged AutoModel files (the changes the user must be able to delete) to
# its disposition in modelopt: a target path (a vendored copy or a thin wrapper over stock,
# relative to this example dir) or ``None`` when the patch is intentionally excluded (unused on
# the DMD2 path, or not needed for real training).
STAGED_AUTOMODEL_DISPOSITION = {
    "components/checkpoint/checkpointing.py": "fastgen_checkpoint.py",  # subclass override
    "components/datasets/diffusion/__init__.py": "fastgen_data/__init__.py",
    "components/datasets/diffusion/collate_fns.py": "fastgen_data/collate_fns.py",  # thin wrapper
    "components/datasets/diffusion/mock_dataloader.py": None,  # excluded: mock smoke (not real training)
    "components/datasets/diffusion/text_to_image_dataset.py": "fastgen_data/text_to_image_dataset.py",
    "components/flow_matching/adapters/qwen_image.py": None,  # excluded: dead on the DMD2 path
    "tools/diffusion/preprocessing_multiprocess.py": "preprocess/preprocessing_multiprocess.py",
    "tools/diffusion/processors/qwen_image.py": "preprocess/processors/qwen_image.py",
    "tests/unit_tests/flow_matching/test_qwen_image_adapter.py": None,  # excluded: test of unused adapter
}


# --------------------------------------------------------------------------------------------- #
#  Environment-independent invariants
# --------------------------------------------------------------------------------------------- #


def test_all_configs_target_vendored_builders():
    """EVERY config in configs/ targets a fastgen_data.* dataloader, never the upstream registry.

    Enumerates all YAMLs so a newly added config cannot silently reintroduce the upstream
    dependence (which would break on stock nemo_automodel).
    """
    configs = sorted((_FASTGEN_DIR / "configs").glob("*.yaml"))
    assert configs, "no configs found under configs/"
    for cfg in configs:
        text = cfg.read_text()
        assert "nemo_automodel.components.datasets.diffusion.build_" not in text, (
            f"{cfg.name} still targets the upstream dataloader builder (breaks on stock upstream)"
        )
        assert "_target_: fastgen_data.build_" in text, (
            f"{cfg.name} does not target a vendored fastgen_data builder"
        )


def test_no_tools_star_imports_in_vendored_code():
    """Nothing under the vendored packages imports the un-packaged AutoModel ``tools/`` tree."""
    pat = re.compile(r"^\s*(?:from|import)\s+tools\.", re.MULTILINE)
    offenders = [
        str(py.relative_to(_FASTGEN_DIR))
        for sub in ("fastgen_data", "preprocess")
        for py in (_FASTGEN_DIR / sub).rglob("*.py")
        if pat.search(py.read_text())
    ]
    assert not offenders, f"tools.* imports found in vendored code: {offenders}"


def test_all_staged_automodel_files_are_removable():
    """Each of the nine staged AutoModel files is vendored or a documented exclusion."""
    assert len(STAGED_AUTOMODEL_DISPOSITION) == 9
    # Vendored targets must exist in the example tree.
    for src, target in STAGED_AUTOMODEL_DISPOSITION.items():
        if target is not None:
            assert (_FASTGEN_DIR / target).is_file(), f"{src}: missing vendored target {target}"
    # Excluded: the flow-matching adapter + its unit test (dead on the DMD2 path) and the mock
    # smoke loader (not needed for real training).
    excluded = {src for src, target in STAGED_AUTOMODEL_DISPOSITION.items() if target is None}
    assert excluded == {
        "components/datasets/diffusion/mock_dataloader.py",
        "components/flow_matching/adapters/qwen_image.py",
        "tests/unit_tests/flow_matching/test_qwen_image_adapter.py",
    }


# Files copied from AutoModel. Per review, these are NVIDIA-authored Apache-2.0 files, so they
# are treated as ordinary NVIDIA files: the insert-license hook manages the header (no pre-commit
# exclusion), and the per-file "Vendored from" provenance note + duplicated original-license block
# were removed — only the standard NVIDIA SPDX header remains.
FORMERLY_VENDORED = [
    "fastgen_data/text_to_image_dataset.py",
    "preprocess/preprocessing_multiprocess.py",
    "preprocess/processors/__init__.py",
    "preprocess/processors/base.py",
    "preprocess/processors/registry.py",
    "preprocess/processors/caption_loaders.py",
    "preprocess/processors/qwen_image.py",
]


def test_formerly_vendored_files_use_standard_nvidia_header():
    """They carry only the standard NVIDIA SPDX header — no provenance note, no duplicate license."""
    for target in FORMERLY_VENDORED:
        text = (_FASTGEN_DIR / target).read_text()
        assert text.startswith(
            "# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES"
        ), f"{target}: must start with the standard NVIDIA SPDX header"
        assert "SPDX-License-Identifier: Apache-2.0" in text, f"{target}: missing SPDX license id"
        assert "Vendored from" not in text, f"{target}: stray 'Vendored from' provenance note"
        assert text.count('Licensed under the Apache License, Version 2.0 (the "License")') == 1, (
            f"{target}: expected exactly one license block (no duplicate)"
        )


# --------------------------------------------------------------------------------------------- #
#  Dependency-guarded structural tests (run in the container)
# --------------------------------------------------------------------------------------------- #


def test_data_builders_importable_and_accept_negative_prompt_path():
    """The real-data builder exists and accepts ``negative_prompt_embedding_path``."""
    pytest.importorskip("nemo_automodel")
    pytest.importorskip("torch")

    import fastgen_data

    assert callable(fastgen_data.build_text_to_image_multiresolution_dataloader)
    sig = inspect.signature(fastgen_data.build_text_to_image_multiresolution_dataloader)
    assert "negative_prompt_embedding_path" in sig.parameters
    # Default None => CFG-less construction works without the negative embedding (it is optional).
    assert sig.parameters["negative_prompt_embedding_path"].default is None


def test_collate_emits_contract_keys_and_broadcasts_negative_prompt():
    """Collate maps prompt_embeds_mask -> text_embeddings_mask and broadcasts the neg embed."""
    pytest.importorskip("nemo_automodel")
    torch = pytest.importorskip("torch")

    from fastgen_data import collate_fn_text_to_image

    seq, dim, c, h, w = 5, 16, 4, 8, 8
    # A per-item sample matching what TextToImageDataset emits — collate_fn_production requires
    # crop_resolution / original_resolution / crop_offset / prompt / image_path / bucket_id /
    # aspect_ratio in addition to the latent + text embeds.
    sample = {
        "latent": torch.randn(c, h, w),
        "crop_resolution": torch.tensor([h, w]),
        "original_resolution": torch.tensor([h, w]),
        "crop_offset": torch.tensor([0, 0]),
        "prompt": "a test prompt",
        "image_path": "img.png",
        "bucket_id": 0,
        "aspect_ratio": 1.0,
        "prompt_embeds": torch.randn(seq, dim),
        "prompt_embeds_mask": torch.ones(seq, dtype=torch.long),
    }
    batch = [dict(sample), dict(sample)]
    neg = torch.randn(seq, dim)

    out = collate_fn_text_to_image(batch, negative_text_embeddings=neg)

    assert "image_latents" in out and out["image_latents"].shape[0] == len(batch)
    assert "text_embeddings" in out
    assert "text_embeddings_mask" in out  # mapped from prompt_embeds_mask
    assert out["negative_text_embeddings"].shape[0] == len(
        batch
    )  # broadcast [seq,dim]->[B,seq,dim]


def test_partial_load_checkpointer_overrides_only_load_optimizer():
    """The subclass relaxes only optimizer load; model-state load stays strict (inherited)."""
    pytest.importorskip("nemo_automodel")
    from fastgen_checkpoint import PartialLoadCheckpointer, make_optimizer_partial_load_tolerant
    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer

    assert issubclass(PartialLoadCheckpointer, Checkpointer)
    # Only load_optimizer is overridden on the subclass; load_model is inherited (strict).
    assert "load_optimizer" in PartialLoadCheckpointer.__dict__
    assert "load_model" not in PartialLoadCheckpointer.__dict__

    # The in-place upgrade re-blesses an existing instance without changing other behavior.
    obj = Checkpointer.__new__(Checkpointer)
    make_optimizer_partial_load_tolerant(obj)
    assert isinstance(obj, PartialLoadCheckpointer)
    make_optimizer_partial_load_tolerant(obj)  # idempotent
    assert isinstance(obj, PartialLoadCheckpointer)


def test_recipe_injects_partial_load_checkpointer_in_load_checkpoint():
    """The recipe upgrades self.checkpointer in load_checkpoint (before the parent restore)."""
    src = (_FASTGEN_DIR / "dmd2_recipe.py").read_text()
    assert "from fastgen_checkpoint import make_optimizer_partial_load_tolerant" in src
    assert "make_optimizer_partial_load_tolerant(self.checkpointer)" in src


def test_qwen_image_processor_registered():
    """Importing the vendored preprocessing package registers the qwen_image processor."""
    pytest.importorskip("torch")
    pytest.importorskip("nemo_automodel")
    from preprocess.processors import ProcessorRegistry

    assert ProcessorRegistry.is_registered("qwen_image")
