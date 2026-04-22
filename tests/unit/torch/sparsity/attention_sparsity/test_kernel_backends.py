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

"""Unit tests for diffusers kernel backend registration and thread-local context.

The forward pass of ``_diffusers_triton_attention`` requires a GPU and is
exercised in ``tests/gpu/torch/sparsity/attention_sparsity/
test_diffusers_triton_attention.py``. These CPU tests cover backend
registration, thread-local config, and the conversion-time plumbing.
"""

import pytest
import torch.nn as nn

pytest.importorskip("diffusers")


# ---------------------------------------------------------------------------
# Thread-local skip-softmax context
# ---------------------------------------------------------------------------


class TestSkipSoftmaxContext:
    def test_default_is_false(self):
        from modelopt.torch.sparsity.attention_sparsity.kernels import get_skip_softmax_context

        assert get_skip_softmax_context() is False

    def test_set_and_get(self):
        from modelopt.torch.sparsity.attention_sparsity.kernels import (
            get_skip_softmax_context,
            set_skip_softmax_context,
        )

        set_skip_softmax_context(True)
        assert get_skip_softmax_context() is True
        set_skip_softmax_context(False)
        assert get_skip_softmax_context() is False


# ---------------------------------------------------------------------------
# Diffusers triton attention backend registration and config
# ---------------------------------------------------------------------------


class TestDiffusersTritonBackend:
    """Backend registration, thread-local config — no kernel execution."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        from modelopt.torch.sparsity.attention_sparsity.kernels import (
            diffusers_triton_attention as mod,
        )

        mod._BACKEND_REGISTERED = False
        mod.clear_triton_skip_softmax_config()
        yield
        mod.clear_triton_skip_softmax_config()

    def test_set_clear_config(self):
        from modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention import (
            clear_triton_skip_softmax_config,
            set_triton_skip_softmax_config,
        )

        set_triton_skip_softmax_config(threshold=0.1)
        clear_triton_skip_softmax_config()

    def test_register_idempotent(self):
        from modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention import (
            register_diffusers_triton_attention,
        )

        register_diffusers_triton_attention()
        register_diffusers_triton_attention()  # Should be a no-op

    def test_get_backend_before_register_raises(self):
        from modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention import (
            get_triton_attention_backend,
        )

        with pytest.raises(RuntimeError, match="not registered"):
            get_triton_attention_backend()


# ---------------------------------------------------------------------------
# conversion._register_diffusers_backends_if_needed
# ---------------------------------------------------------------------------


class TestRegisterDiffusersBackends:
    def test_no_diffusers_model_no_error(self):
        """Non-ModelMixin models pass through without registering."""
        from modelopt.torch.sparsity.attention_sparsity.conversion import (
            _register_diffusers_backends_if_needed,
        )

        _register_diffusers_backends_if_needed(nn.Linear(10, 10))

    def test_with_diffusers_model(self):
        """A ModelMixin subclass triggers diffusers backend registration."""
        from diffusers.models.modeling_utils import ModelMixin

        from modelopt.torch.sparsity.attention_sparsity.conversion import (
            _register_diffusers_backends_if_needed,
        )
        from modelopt.torch.sparsity.attention_sparsity.kernels import (
            diffusers_triton_attention as mod,
        )

        mod._BACKEND_REGISTERED = False

        class _TinyMixinModel(ModelMixin):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4)

        _register_diffusers_backends_if_needed(_TinyMixinModel())
        assert mod._BACKEND_REGISTERED is True
