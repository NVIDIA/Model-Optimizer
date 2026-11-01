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
"""Run example steps without a ``torchrun`` launch; see ``_test_utils.examples.megatron_example_runner``."""

import os
import shutil

import pytest
import transformers
import yaml
from _test_utils.examples.megatron_example_runner import (
    _load_example_module,
    reset_megatron_global_state,
    run_example_step,
)
from _test_utils.examples.run_command import set_in_process_runner


@pytest.fixture(autouse=True)
def _fast_example_runner():
    """Run example steps without shelling out to ``torchrun``.

    Per test, not per session: the hook is a module-global in ``run_command``, and this runner
    raises rather than falling back, so leaving it installed would break any other example suite
    collected later in the same session (``pytest tests/examples``).
    """
    set_in_process_runner(run_example_step)
    try:
        yield
    finally:
        set_in_process_runner(None)


@pytest.fixture(autouse=True)
def _isolate_megatron_global_state(request):
    """Reset shared state around every test so a failure cannot cascade into the next one.

    In-process steps share the interpreter. Besides Megatron's singletons, Transformer-Engine
    records its chosen attention backend in ``NVTE_*``, which failed a Mamba hybrid that ran after
    an attention model -- so the environment is restored wholesale rather than by naming variables.
    """
    # Only tests that take ``num_gpus`` run example steps; the gc-heavy reset is wasted on the rest.
    reset_state = "num_gpus" in request.fixturenames
    env_before = os.environ.copy()
    if reset_state:
        reset_megatron_global_state()
    try:
        yield
    finally:
        if reset_state:
            reset_megatron_global_state()
        os.environ.clear()
        os.environ.update(env_before)


class FakeHubCheckpoint:
    """Serves a local checkpoint to example scripts as if it were Hub model ``hub_model_id``.

    ``serve(checkpoint, *scripts, nproc=...)`` patches ``ensure_local_checkpoint`` on each script's
    module -- the one the in-process runner caches and runs -- to resolve ``hub_model_id`` (or
    ``checkpoint`` itself, for a script that must read its argument before resolving it) to a copy
    of ``checkpoint`` holding the file ``marker``, and returns ``hub_model_id`` to pass on the
    command line: a step that read it without resolving it fails, and ``marker`` reaching an
    export shows the export read the resolved copy. Like the Hugging Face cache would for a real Hub
    ID, ``AutoTokenizer.from_pretrained(hub_model_id)`` loads the copy's tokenizer: Megatron-Bridge
    loads the tokenizer a checkpoint records. Only a single-rank step runs in this process
    and sees the patch, so with ``nproc > 1`` nothing is patched, ``checkpoint`` is returned
    unchanged and ``active`` is False: check the marker and the Hub ID only when it is True.
    """

    hub_model_id = "org/tiny-model"
    marker = "from_the_resolved_copy.md"

    def __init__(self, monkeypatch, tmp_path):
        self._monkeypatch = monkeypatch
        self._resolved = tmp_path / "resolved_hub_checkpoint"
        self.active = False

    def serve(self, checkpoint, *scripts, nproc: int) -> str:
        self.active = nproc == 1
        if not self.active:
            return str(checkpoint)
        shutil.copytree(checkpoint, self._resolved)
        (self._resolved / self.marker).write_text("from the resolved copy\n")

        def ensure_local_checkpoint(model_name_or_path, group=None):
            assert str(model_name_or_path) in (self.hub_model_id, str(checkpoint))
            return self.hub_model_id, str(self._resolved)

        for script in scripts:
            module = _load_example_module(script, "megatron_bridge")
            self._monkeypatch.setattr(module, "ensure_local_checkpoint", ensure_local_checkpoint)

        load_tokenizer = transformers.AutoTokenizer.from_pretrained

        def from_pretrained(*args, **kwargs):
            # Megatron-LM passes the name as pretrained_model_name_or_path=.
            if args and str(args[0]) == self.hub_model_id:
                args = (str(self._resolved), *args[1:])
            if str(kwargs.get("pretrained_model_name_or_path")) == self.hub_model_id:
                kwargs["pretrained_model_name_or_path"] = str(self._resolved)
            return load_tokenizer(*args, **kwargs)

        self._monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", from_pretrained)
        return self.hub_model_id

    @staticmethod
    def saved_tokenizer_model(megatron_path) -> str:
        """The tokenizer a Megatron-Bridge checkpoint records in its ``run_config.yaml``."""
        (run_config,) = sorted(megatron_path.rglob("run_config.yaml"))
        return yaml.safe_load(run_config.read_text())["tokenizer"]["tokenizer_model"]


@pytest.fixture
def fake_hub_checkpoint(monkeypatch, tmp_path):
    """See :class:`FakeHubCheckpoint`."""
    return FakeHubCheckpoint(monkeypatch, tmp_path)
