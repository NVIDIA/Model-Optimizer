# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import platform
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import init_process

import modelopt.torch.opt as mto


@pytest.fixture(scope="session")
def verbose(request):
    return request.config.getoption("verbose")


def pytest_addoption(parser):
    parser.addoption(
        "--run-manual",
        action="store_true",
        default=False,
        help="Run manual tests",
    )
    parser.addoption(
        "--run-release",
        action="store_true",
        default=False,
        help="Run release tests",
    )


# Default per-test `call` wall-clock cap (seconds) by top-level tests/ subdirectory
# Every collectible test group must be listed here else collection errors occur
# A test can override its cap by adding ``@pytest.mark.timeout(...)``
_DEFAULT_TIMEOUT = {
    "examples": 3600,
    "gpu": 120,
    "gpu_megatron": 120,
    "gpu_trtllm": 60,
    "gpu_vllm": 60,
    "regression": 180,
    "unit": 60,
}


def pytest_collection_modifyitems(config, items):
    """Skip flag-gated tests and apply a default per-test timeout based on the test directory."""
    skip_marks = [
        ("manual", "--run-manual"),
        ("release", "--run-release"),
    ]

    for mark_name, option_name in skip_marks:
        if not config.getoption(option_name):
            skipper = pytest.mark.skip(reason=f"Only run when {option_name} is given")
            for item in items:
                if mark_name in item.keywords:
                    item.add_marker(skipper)

    tests_root = Path(__file__).parent
    for item in items:
        if item.get_closest_marker("timeout") is not None or not item.path.is_relative_to(
            tests_root
        ):
            continue
        # First path component under tests/ is the group dir (unit, gpu, examples, ...).
        # Crash loudly (rather than silently skip) if a group has no configured default, so a
        # newly added tests/<group>/ must be given an explicit timeout in the mapping above.
        group = item.path.relative_to(tests_root).parts[0]
        if group not in _DEFAULT_TIMEOUT:
            raise pytest.UsageError(
                f"tests/{group}/ has no default timeout; add '{group}' to "
                "_DEFAULT_TIMEOUT in tests/conftest.py."
            )
        item.add_marker(pytest.mark.timeout(_DEFAULT_TIMEOUT[group]))


@pytest.fixture
def tiny_tokenizer():
    """Real tiny HF tokenizer (vocab=128) shared across unit and gpu test lanes."""
    # Lazy import: transformers_models.py runs ``pytest.importorskip("transformers")``
    # at module load, which we don't want to trigger at conftest import time.
    from _test_utils.torch.transformers_models import get_tiny_tokenizer

    return get_tiny_tokenizer()


@pytest.fixture
def skip_on_windows():
    if platform.system() == "Windows":
        pytest.skip("Skipping on Windows")


@pytest.fixture(scope="session")
def num_gpus():
    return torch.cuda.device_count()


@pytest.fixture(scope="session")
def cuda_capability():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.cuda.get_device_capability()


@pytest.fixture
def distributed_setup_size_1():
    init_process(rank=0, size=1, backend="nccl")
    yield
    dist.destroy_process_group()


@pytest.fixture
def need_2_gpus():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs to run this test")


@pytest.fixture
def need_4_gpus():
    if torch.cuda.device_count() < 4:
        pytest.skip("Need at least 4 GPUs to run this test")


@pytest.fixture
def need_8_gpus():
    if torch.cuda.device_count() < 8:
        pytest.skip("Need at least 8 GPUs to run this test")


@pytest.fixture(scope="module")
def set_torch_dtype(request):
    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    yield
    torch.set_default_dtype(orig_dtype)


@pytest.fixture(scope="session", autouse=True)
def enable_hf_checkpointing():
    mto.enable_huggingface_checkpointing()


@pytest.fixture(scope="session")
def project_root_path(request: pytest.FixtureRequest) -> Path:
    """Fixture providing the project root path for tests."""
    return Path(request.config.rootpath)
