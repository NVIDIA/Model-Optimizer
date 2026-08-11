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

"""Targeted unit tests for modelopt.torch.quantization.conversion.

Only exercises code paths not covered elsewhere in the unit suite: the SVDQuant
conversion entrypoint, quantizer-state restore error paths, parent-class lookup
failure, SequentialQuantizer edge branches, the temporary-config context
manager's module-type restoration, and the deprecated set_quantizer_attribute shim.
"""

import pytest
import torch.nn as nn
from _test_utils.torch.quantization.models import SimpleLinear

from modelopt.torch.opt.conversion import ApplyModeError, ModeloptStateManager
from modelopt.torch.quantization.config import QuantizeConfig, QuantizerAttributeConfig
from modelopt.torch.quantization.conversion import (
    convert_to_quantized_model_svdquant,
    quantizer_state,
    replace_quant_module,
    restore_quantizer_state,
    set_quantizer_attribute,
    set_quantizer_attributes_full,
    set_quantizer_attributes_partial,
    set_quantizer_by_cfg,
    set_quantizer_by_cfg_context,
    update_quantize_metadata,
)
from modelopt.torch.quantization.nn import SequentialQuantizer, SVDQuantLinear, TensorQuantizer

# SimpleLinear has three nn.Linear layers -> three weight quantizers after replacement.
SIMPLE_LINEAR_NUM_WEIGHT_QUANTIZERS = 3


def _replaced():
    """Return a fresh SimpleLinear with quantized modules inserted (no calibration)."""
    model = SimpleLinear()
    replace_quant_module(model)
    return model


def _weight_quantizers(model):
    """TensorQuantizer weight quantizers, guaranteed non-empty (guards vacuous loops)."""
    found = {
        n: m
        for n, m in model.named_modules()
        if isinstance(m, TensorQuantizer) and n.endswith("weight_quantizer")
    }
    assert len(found) == SIMPLE_LINEAR_NUM_WEIGHT_QUANTIZERS
    return found


def _seq_weight_quantizers(model):
    """SequentialQuantizer containers on the weight path (subclass nn.Sequential, not
    TensorQuantizer, so _weight_quantizers cannot see them)."""
    return {
        n: m
        for n, m in model.named_modules()
        if isinstance(m, SequentialQuantizer) and n.endswith("weight_quantizer")
    }


def _make_saved_metadata():
    """A configured model plus the metadata dict a checkpoint would carry."""
    model = _replaced()
    set_quantizer_by_cfg(
        model,
        [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "axis": 0}},
        ],
    )
    metadata = {}
    update_quantize_metadata(model, QuantizeConfig(), metadata)
    return model, metadata


class TestSvdquantConversion:
    def test_convert_replaces_quant_linears_and_records_metadata(self):
        model = SimpleLinear()
        ModeloptStateManager(model, init_state=True)
        replace_quant_module(model)  # SVDQuant conversion runs on an already-quantized model
        config = QuantizeConfig(
            quant_cfg=[{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4, "axis": 0}}]
        )
        converted, metadata = convert_to_quantized_model_svdquant(model, config)
        assert converted is model  # conversion is in-place
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == SIMPLE_LINEAR_NUM_WEIGHT_QUANTIZERS
        assert all(isinstance(m, SVDQuantLinear) for m in linears)
        # The quant_cfg is applied to the quantizers of the SVDQuant modules...
        for module in _weight_quantizers(model).values():
            assert module.num_bits == 4
            assert module.axis == 0
        # ...and the metadata records the resulting quantizer state.
        assert metadata["quantizer_state"] == quantizer_state(model)


class TestRestoreQuantizerStateErrors:
    # NOTE: the happy-path restore round-trip is already covered by existing unit tests;
    # only the key-mismatch error branches are pinned here.

    def test_unmatched_checkpoint_keys_raise(self):
        """Keys present in the checkpoint but missing from the model are 'unmatched'."""
        _, metadata = _make_saved_metadata()
        metadata["quantizer_state"]["bogus.weight_quantizer"] = {}
        with pytest.raises(ApplyModeError, match=r"Unmatched keys.*bogus\.weight_quantizer"):
            restore_quantizer_state(_replaced(), QuantizeConfig(), metadata)

    def test_extra_model_keys_raise(self):
        """Quantizers in the model that have no checkpoint entry are 'extra'."""
        _, metadata = _make_saved_metadata()
        del metadata["quantizer_state"]["net.0.weight_quantizer"]
        with pytest.raises(ApplyModeError, match=r"Extra keys.*net\.0\.weight_quantizer"):
            restore_quantizer_state(_replaced(), QuantizeConfig(), metadata)


def test_unknown_parent_class_raises():
    model = _replaced()
    with pytest.raises(ValueError, match="not found in QuantModuleRegistry"):
        set_quantizer_by_cfg(
            model,
            [{"parent_class": "nn.NoSuchModule", "quantizer_name": "*", "enable": False}],
        )


class TestSequentialQuantizerEdgeBranches:
    # Callable filters are used below so that only the SequentialQuantizer containers match;
    # a "*weight_quantizer" wildcard would also reach the containers' ".0/.1" children via
    # the fused-experts name normalization and hit different branches.

    def test_full_list_length_mismatch_warns_and_assigns_partially(self):
        model = _replaced()
        set_quantizer_attributes_full(
            model,
            lambda name: name.endswith("weight_quantizer"),
            [QuantizerAttributeConfig(num_bits=4), QuantizerAttributeConfig(num_bits=8)],
        )
        with pytest.warns(UserWarning, match="does not match the number"):
            set_quantizer_attributes_full(
                model,
                lambda name: name.endswith("weight_quantizer"),
                [QuantizerAttributeConfig(num_bits=2)],
            )
        containers = _seq_weight_quantizers(model)
        assert len(containers) == SIMPLE_LINEAR_NUM_WEIGHT_QUANTIZERS
        for module in containers.values():
            # Partial assignment: only the first sub-quantizer got the new config.
            assert [q.num_bits for q in module] == [2, 8]

    def test_partial_list_on_plain_tensor_quantizer_raises(self):
        model = _replaced()
        with pytest.raises(ValueError, match="not a SequentialQuantizer"):
            set_quantizer_attributes_partial(
                model, "*input_quantizer", [{"enable": False}, {"enable": True}]
            )

    def test_partial_list_applies_per_position_on_sequential(self):
        model = _replaced()
        set_quantizer_attributes_full(
            model,
            "*weight_quantizer",
            [QuantizerAttributeConfig(num_bits=4), QuantizerAttributeConfig(num_bits=8)],
        )
        set_quantizer_attributes_partial(
            model,
            lambda name: name.endswith("weight_quantizer"),
            [{"enable": False}, {"enable": True}],
        )
        containers = _seq_weight_quantizers(model)
        assert len(containers) == SIMPLE_LINEAR_NUM_WEIGHT_QUANTIZERS
        for module in containers.values():
            assert not module[0].is_enabled
            assert module[1].is_enabled
            assert [q.num_bits for q in module] == [4, 8]  # merged, not replaced


class TestSetQuantizerByCfgContextTypeRestore:
    # NOTE: plain attribute save/restore of the context manager is covered elsewhere;
    # these tests pin the module-type re-creation branches of the exit handler.

    def test_sequential_downgrade_is_reverted_on_exit(self):
        """A single-cfg entry downgrades a SequentialQuantizer; exit re-creates it."""
        model = _replaced()
        set_quantizer_by_cfg(
            model,
            [
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": [{"num_bits": 4}, {"num_bits": 8, "axis": 0}],
                }
            ],
        )
        with set_quantizer_by_cfg_context(
            model, [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}}]
        ):
            wq = model.net[0].weight_quantizer
            assert isinstance(wq, TensorQuantizer)
            assert not isinstance(wq, SequentialQuantizer)
            assert wq.num_bits == 8
        wq = model.net[0].weight_quantizer
        assert isinstance(wq, SequentialQuantizer)
        assert len(wq) == 2
        assert [q.num_bits for q in wq] == [4, 8]
        assert wq[1].axis == 0

    def test_manual_upgrade_inside_context_is_reverted(self):
        """Even a manual TensorQuantizer->SequentialQuantizer swap in the body is undone."""
        model = _replaced()
        with set_quantizer_by_cfg_context(model, []):
            set_quantizer_attributes_full(
                model,
                "*weight_quantizer",
                [QuantizerAttributeConfig(num_bits=4), QuantizerAttributeConfig(num_bits=8)],
            )
            assert isinstance(model.net[0].weight_quantizer, SequentialQuantizer)
        wq = model.net[0].weight_quantizer
        assert isinstance(wq, TensorQuantizer)
        assert not isinstance(wq, SequentialQuantizer)
        assert wq.num_bits == 8
        assert wq.axis == 0  # original default weight attributes restored


def test_deprecated_set_quantizer_attribute_warns_and_merges():
    model = _replaced()
    set_quantizer_attributes_full(
        model, "*weight_quantizer", QuantizerAttributeConfig(num_bits=8, axis=0)
    )
    with pytest.warns(DeprecationWarning, match="set_quantizer_attribute is deprecated"):
        set_quantizer_attribute(model, "*weight_quantizer", {"num_bits": 4})
    for module in _weight_quantizers(model).values():
        assert module.num_bits == 4
        assert module.axis == 0  # merged like set_quantizer_attributes_partial
