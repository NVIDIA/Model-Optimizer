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

import pytest
import torch
import torch.nn.functional as F
from _test_utils.torch.quantization.quant_utils import quant

from modelopt.torch.quantization import tensor_quant
from modelopt.torch.quantization import utils as quant_utils
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.model_calib import max_calibrate
from modelopt.torch.quantization.nn import QuantLinear, SequentialQuantizer, TensorQuantizer


class TensorQuantizerTester:
    device = None

    def test_simple_run(self):
        """Quantizer calls fake_tensor_quant by default"""
        x = torch.randn(3, 7).to(self.device)
        amax_x = torch.max(torch.abs(x))
        fn_quant_x = tensor_quant.fake_tensor_quant(x, amax_x, None)
        quantizer = TensorQuantizer()
        module_quant_x = quantizer(x)
        assert torch.allclose(fn_quant_x, module_quant_x)

    def test_per_tensor_scale(self):
        """Quantizer performs expected quantization"""
        x = torch.randn(31).to(self.device)

        quant_x = quant(x, torch.max(torch.abs(x)), fake=True)
        quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
        module_quant_x = quantizer(x)
        assert torch.allclose(module_quant_x, quant_x)

    def test_per_channel_scale(self):
        """Quantizer performs per channel scaling"""
        x = torch.randn(5, 5, 4, 8).to(self.device)

        # Pytorch filter layout seems to be KCRS, reduce max to shape [K, 1, 1, 1] to test per channel scale
        # Shrink max a little, so that clip behavior is tested
        amax_x = torch.amax(x.abs(), axis=(1, 2, 3), keepdims=True)

        quant_x_ref = quant(x, amax_x, fake=True)
        quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=(0)))
        quantizer.to(self.device)
        module_quant_x = quantizer(x)
        assert torch.allclose(module_quant_x, quant_x_ref, rtol=1e-2)

    def test_disable(self):
        x = torch.randn(3, 7).to(self.device)
        torch.max(torch.abs(x))
        quantizer = TensorQuantizer(QuantizerAttributeConfig(enable=False)).to(self.device)
        module_quant_x = quantizer(x)
        assert torch.allclose(x, module_quant_x)

    def test_state_loading(self):
        """Test quant_attr_cfg loading via state_dict"""
        amax = [3.14, 2.718]
        quant_attr_cfg1 = QuantizerAttributeConfig(num_bits=(4, 3))
        quantizer1 = TensorQuantizer(quant_attr_cfg1, amax=amax).to(self.device)

        # copy state
        quantizer1.load_state_dict(quantizer1.state_dict())
        assert torch.allclose(quantizer1.amax, torch.tensor(amax).to(self.device))

    def test_properties(self):
        quant_attr_cfg1 = QuantizerAttributeConfig()
        quantizer1 = TensorQuantizer(quant_attr_cfg1, amax=3.14).to(self.device)
        quantizer1.amax = 0.577

        assert quantizer1.amax == torch.tensor(0.577).to(self.device)
        assert quantizer1.step_size == 0.577 / 127.0

        quant_attr_cfg2 = QuantizerAttributeConfig()
        quantizer2 = TensorQuantizer(quant_attr_cfg2).to(self.device)
        amax = torch.tensor([3.142, 2.718]).to(self.device)
        quantizer2.amax = amax
        assert torch.allclose(quantizer2.amax, amax)

        quant_attr_cfg3 = QuantizerAttributeConfig()
        quantizer3 = TensorQuantizer(quant_attr_cfg3)
        assert quantizer3.amax is None

    def test_init_calib(self):
        quant_attr_cfg2 = QuantizerAttributeConfig(axis=(0, 1))
        quantizer2 = TensorQuantizer(quant_attr_cfg2, if_calib=True, if_quant=False).to(self.device)

        x_2 = torch.rand(7, 6, 7, 7).to(self.device)
        quantizer2(x_2)
        quantizer2.load_calib_amax()

        assert quantizer2.amax.numel() == 7 * 6

    def test_max_calib(self):
        axis = 0
        reduce_axis = (1, 2, 3)
        quant_attr_cfg1 = QuantizerAttributeConfig(axis=axis)
        quantizer1 = TensorQuantizer(quant_attr_cfg1).to(self.device)
        quantizer1.enable_calib()

        quant_attr_cfg1 = QuantizerAttributeConfig(axis=axis)
        quantizer1 = TensorQuantizer(quant_attr_cfg1).to(self.device)
        quantizer1.enable_calib()

        with pytest.raises(RuntimeError, match="Calibrator returned None"):
            quantizer1.load_calib_amax()

        x_1 = torch.rand(7, 6, 7, 7).to(self.device)
        x_2 = torch.rand(7, 6, 7, 7).to(self.device)
        quantizer1(x_1)
        quantizer1(x_2)
        quantizer1.disable_calib()

        global_amax = torch.max(
            quant_utils.reduce_amax(x_1, axis=reduce_axis, keepdims=True),
            quant_utils.reduce_amax(x_2, axis=reduce_axis, keepdims=True),
        )
        assert torch.allclose(
            quantizer1._calibrator.compute_amax(),
            global_amax,
            atol=0,
            rtol=0,
        )

        quantizer1.load_calib_amax()
        assert torch.allclose(
            quantizer1.amax,
            global_amax,
            atol=0,
            rtol=0,
        )

    @pytest.mark.manual(reason="slow test, run with --run-manual")
    def test_entropy_and_percentile_calib(self):
        """Don't really have a good way to test it."""
        quant_attr_cfg1 = QuantizerAttributeConfig(calib_method="histogram")
        quantizer1 = TensorQuantizer(quant_attr_cfg1, if_calib=True, if_quant=False).to(self.device)

        x_1 = torch.rand(3, 6, 7, 7).to(self.device)
        x_2 = torch.rand(3, 6, 7, 7).to(self.device)
        quantizer1(x_1)
        quantizer1(x_2)

        quantizer1.load_calib_amax("entropy")
        assert torch.allclose(
            quantizer1._calibrator.compute_amax("entropy"),
            quantizer1.amax,
            atol=0,
            rtol=0,
        )
        quantizer1._calibrator.reset()

        quantizer1(x_1)
        quantizer1(x_2)

        quantizer1.load_calib_amax("percentile", percentile=99.99)
        assert torch.allclose(
            quantizer1._calibrator.compute_amax("percentile", percentile=99.99),
            quantizer1.amax,
            atol=0,
            rtol=0,
        )

    def test_setters(self):
        quantizer = TensorQuantizer()
        quantizer.num_bits = 7
        quantizer.unsigned = True

        assert quantizer.num_bits == 7
        assert quantizer.unsigned

    def test_pre_quant_scale(self):
        quant_attr_cfg = QuantizerAttributeConfig(axis=1, num_bits=8)
        quantizer = TensorQuantizer(quant_attr_cfg, amax=127.0).to(self.device)
        quantizer2 = TensorQuantizer(quant_attr_cfg, amax=127.0).to(self.device)

        inputs = torch.Tensor([[0, 0.4, 1.1, 2.0]]).to(self.device)
        outputs_gt = torch.Tensor([[0, 0, 1, 2]]).to(self.device)
        assert torch.allclose(quantizer(inputs), outputs_gt)

        quantizer.pre_quant_scale = 2.0
        outputs_gt = torch.Tensor([[0, 1, 2, 4]]).to(self.device)
        assert torch.allclose(quantizer(inputs), outputs_gt)

        quantizer2.pre_quant_scale = torch.Tensor([[1.0, 2.0, 3.0, 4.0]]).to(self.device)
        outputs_gt = torch.Tensor([[0, 1, 3, 8]]).to(self.device)
        assert torch.allclose(quantizer2(inputs), outputs_gt)

    def test_set_from_attribute_config(self):
        tq = TensorQuantizer()
        tq.set_from_attribute_config({"num_bits": 4})
        assert tq.num_bits == 4
        tq.set_from_attribute_config({"axis": -1})
        assert tq.axis == -1
        tq.set_from_attribute_config({"enable": False})
        assert tq._disabled

    def test_modelopt_state(self):
        # Test loading of amax from ref to test
        tensor_quantizer_ref = TensorQuantizer(QuantizerAttributeConfig(num_bits=4), amax=10.0)
        tensor_quantizer_ref.to(self.device)
        tensor_quantizer_test = TensorQuantizer(QuantizerAttributeConfig())

        state_dict = tensor_quantizer_ref.get_modelopt_state()
        for k in tensor_quantizer_ref._get_properties_for_modelopt_state():
            assert k in state_dict
            assert state_dict[k] == getattr(tensor_quantizer_ref, k)

        assert "_pytorch_state_metadata" in state_dict
        assert (
            "_amax" not in state_dict
            and "_amax" in state_dict["_pytorch_state_metadata"]["buffers"]
        )

        tensor_quantizer_test.set_from_modelopt_state(state_dict)
        tensor_quantizer_test.load_state_dict(tensor_quantizer_ref.state_dict())
        tensor_quantizer_test.to(self.device)

        x = torch.randn(10, 10).to(self.device)
        assert torch.allclose(tensor_quantizer_ref(x), tensor_quantizer_test(x))

    def test_amax_export(self):
        a = torch.randn(4, 6, 8)

        # test block-quantization with amax flattened
        quant_attr_cfg = QuantizerAttributeConfig(num_bits=4, block_sizes={-1: 4})
        quantizer = TensorQuantizer(quant_attr_cfg)
        assert not hasattr(quantizer, "_amax_shape_for_export")
        quantizer(a)
        quantizer.enable_calib()
        quantizer(a)
        quantizer.load_calib_amax()
        amax = quantizer.export_amax()
        assert hasattr(
            quantizer, "_amax_shape_for_export"
        ) and quantizer._amax_shape_for_export == (4, 6, -1)
        assert amax.shape == (4, 6, 2), amax.shape

        # test a different block-quantization case
        quant_attr_cfg = QuantizerAttributeConfig(num_bits=4, block_sizes={-1: 8})
        quantizer = TensorQuantizer(quant_attr_cfg)
        assert not hasattr(quantizer, "_amax_shape_for_export")
        quantizer(a)
        quantizer.enable_calib()
        quantizer(a)
        quantizer.load_calib_amax()
        amax = quantizer.export_amax()
        assert hasattr(
            quantizer, "_amax_shape_for_export"
        ) and quantizer._amax_shape_for_export == (4, 6, -1)
        assert amax.shape == (4, 6, 1), amax.shape

        # test per-channel quantization
        quant_attr_cfg = QuantizerAttributeConfig(num_bits=4, axis=0)
        quantizer = TensorQuantizer(quant_attr_cfg)
        quantizer.enable_calib()
        quantizer(a)
        quantizer.load_calib_amax()
        amax = quantizer.export_amax()
        assert not hasattr(quantizer, "_amax_shape_for_export")
        assert amax.shape == (4,), amax.shape

        # test per-tensor quantization
        quant_attr_cfg = QuantizerAttributeConfig(num_bits=4)
        quantizer = TensorQuantizer(quant_attr_cfg)
        quantizer.enable_calib()
        quantizer(a)
        quantizer.load_calib_amax()
        amax = quantizer.export_amax()
        assert amax.shape == (1,)

    def test_save_restore(self):
        ref_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=4, axis=0))

        ref_quantizer.enable_calib()
        ref_quantizer(torch.randn(4, 6, 8))
        ref_quantizer(torch.randn(4, 6, 8))
        ref_quantizer.load_calib_amax()
        ref_quantizer.disable_calib()

        ref_quantizer_state = ref_quantizer.get_modelopt_state()

        test_quantizer = TensorQuantizer(QuantizerAttributeConfig())
        test_quantizer.set_from_modelopt_state(ref_quantizer_state)
        test_quantizer.load_state_dict(ref_quantizer.state_dict())
        test_quantizer.cpu()

        inp = torch.randn(4, 6, 8)
        out1 = ref_quantizer(inp)
        out2 = test_quantizer(inp)
        assert torch.allclose(out1, out2)

    def test_backward(self):
        x = torch.randn(3, 7, requires_grad=True).to(self.device)
        tq = TensorQuantizer()
        tq(x).sum().backward()

    def test_block_sizes_axis(self):
        # Initialize TensorQuantizer with block_sizes
        test_weight_quantizer = TensorQuantizer(
            QuantizerAttributeConfig(
                num_bits=8,
                block_sizes={1: None},
            )
        )
        ref_weight_quantizer = TensorQuantizer(
            QuantizerAttributeConfig(
                num_bits=8,
                axis=0,
            )
        )

        input_quantizer = TensorQuantizer(
            QuantizerAttributeConfig(
                num_bits=8,
                block_sizes={0: None, 1: None},
            )
        )
        ref_input_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=-1))

        # Create a dummy tensor
        input_tensor = torch.randn(1, 2, 3)
        weight_tensor = torch.randn(3, 3)

        # Test TensorQuantizer with block_sizes and axis
        assert torch.allclose(
            test_weight_quantizer(weight_tensor), ref_weight_quantizer(weight_tensor)
        )
        assert torch.allclose(input_quantizer(input_tensor), ref_input_quantizer(input_tensor))


class BlockQuantTester:
    device = None

    def test_simple(self):
        a = torch.randn(8, 4).to(self.device)
        amax_ref = a.view(-1, 2).abs().amax(dim=-1, keepdim=True)
        a_quant_ref = tensor_quant.fake_tensor_quant(
            a.view(-1, 2), amax_ref, None, 8, False, False
        ).view_as(a)

        block_desc = QuantizerAttributeConfig(num_bits=8, block_sizes={-1: 2})
        block_quantizer = TensorQuantizer(block_desc)

        block_quantizer.enable_calib()
        a_quant = block_quantizer(a)
        block_quantizer.load_calib_amax()

        assert torch.allclose(block_quantizer.amax, amax_ref)
        assert torch.allclose(a_quant, a_quant_ref)

        block_desc = QuantizerAttributeConfig(num_bits=8, block_sizes={1: 2})
        block_quantizer = TensorQuantizer(block_desc)

        block_quantizer.enable_calib()
        a_quant = block_quantizer(a)
        block_quantizer.load_calib_amax()

        assert torch.allclose(block_quantizer.amax, amax_ref)
        assert torch.allclose(a_quant, a_quant_ref)

    def test_complex(self):
        a = torch.randn(2, 6, 4).to(self.device)
        a_max = a.reshape(2, 2, 3, 2, 2).abs().amax(dim=(2, 4), keepdim=True)
        a_quant_ref = tensor_quant.fake_tensor_quant(
            a.reshape(2, 2, 3, 2, 2), a_max, None, 8, False, False
        ).view_as(a)

        block_desc = QuantizerAttributeConfig(num_bits=8, block_sizes={-1: 2, 1: 3})
        block_quantizer = TensorQuantizer(block_desc)

        a_quant = block_quantizer(a)
        assert torch.allclose(a_quant, a_quant_ref)

        # Try a different order for block_size and block_axis
        block_desc = QuantizerAttributeConfig(num_bits=8, block_sizes={1: 3, 2: 2})
        block_quantizer = TensorQuantizer(block_desc)

        a_quant = block_quantizer(a)
        assert torch.allclose(a_quant, a_quant_ref)

    def test_simple_padding(self):
        a = torch.randn(3, 4, 5).to(self.device)
        a_padded = F.pad(a, (0, 1), mode="constant", value=0)
        amax = a_padded.view(-1, 2).abs().amax(dim=-1, keepdim=True)
        a_quant_ref = tensor_quant.fake_tensor_quant(
            a_padded.view(-1, 2), amax, None, 8, False, False
        ).view_as(a_padded)[:, :, :5]

        block_desc = QuantizerAttributeConfig(num_bits=8, block_sizes={-1: 2})
        block_quantizer = TensorQuantizer(block_desc)

        block_quantizer.enable_calib()
        a_quant = block_quantizer(a)
        block_quantizer.load_calib_amax()

        assert torch.allclose(block_quantizer.amax, amax)
        assert torch.allclose(a_quant, a_quant_ref)

        # Try a different format for quantization axis
        block_desc = QuantizerAttributeConfig(num_bits=8, block_sizes={2: 2})
        block_quantizer = TensorQuantizer(block_desc)

        block_quantizer.enable_calib()
        a_quant = block_quantizer(a)
        block_quantizer.load_calib_amax()

        assert torch.allclose(block_quantizer.amax, amax)
        assert torch.allclose(a_quant, a_quant_ref)

    def test_complex_padding(self):
        a = torch.randn(2, 10, 5).to(self.device)
        a_padded = F.pad(a, (0, 1, 0, 2), mode="constant", value=0)
        amax = a_padded.view(2, 4, 3, 3, 2).abs().amax(dim=(2, 4), keepdim=True)
        a_quant_ref = tensor_quant.fake_tensor_quant(
            a_padded.view(2, 4, 3, 3, 2), amax, None, 8, False, False
        ).view_as(a_padded)[:, :10, :5]

        block_desc = QuantizerAttributeConfig(num_bits=8, block_sizes={-1: 2, 1: 3})
        block_quantizer = TensorQuantizer(block_desc)

        block_quantizer.enable_calib()
        a_quant = block_quantizer(a)
        block_quantizer.load_calib_amax()

        assert torch.allclose(amax, block_quantizer.amax)
        assert torch.allclose(a_quant, a_quant_ref)

        # Try a different order for block_size and block_axis
        block_desc = QuantizerAttributeConfig(num_bits=8, block_sizes={1: 3, 2: 2})
        block_quantizer = TensorQuantizer(block_desc)

        a_quant = block_quantizer(a)
        assert torch.allclose(a_quant, a_quant_ref)


class SequentialQuantizerTester:
    device = None

    def test_sequential_quantizer(self):
        nq = SequentialQuantizer(
            TensorQuantizer(QuantizerAttributeConfig()),
            TensorQuantizer(QuantizerAttributeConfig()),
            TensorQuantizer(QuantizerAttributeConfig()),
        )
        nq[0].set_from_attribute_config({"num_bits": 4})
        nq[1].set_from_attribute_config({"block_sizes": {-1: 4}})
        nq[2].set_from_attribute_config({"enable": False})

        tqs = [TensorQuantizer() for _ in range(3)]
        tqs[0].set_from_attribute_config({"num_bits": 4})
        tqs[1].set_from_attribute_config({"block_sizes": {-1: 4}})
        tqs[2].set_from_attribute_config({"enable": False})

        x = torch.randn(2, 16).to(self.device)
        x_quant = nq(x)

        x_ref = x
        for tq in tqs:
            x_ref = tq(x_ref)

        assert torch.allclose(x_ref, x_quant)

    def test_replace_sequential_quantizer(self):
        model = torch.nn.Module()
        model.model = torch.nn.Sequential(QuantLinear(4, 4), torch.nn.ReLU())
        model.model[0].weight_quantizer = SequentialQuantizer(
            TensorQuantizer(QuantizerAttributeConfig()), TensorQuantizer(QuantizerAttributeConfig())
        )
        sequential_quantizer = model.model[0].weight_quantizer

        with SequentialQuantizer.convert_to_single_quantizer(model):
            for name, module in model.model.named_modules():
                assert not isinstance(module, SequentialQuantizer)

        assert (
            isinstance(model.model[0].weight_quantizer, SequentialQuantizer)
            and model.model[0].weight_quantizer is sequential_quantizer
        )
        assert not hasattr(model.model[0], "_original_weight_quantizer")

    def test_sequential_quantizer_attribute(self):
        sq = SequentialQuantizer(
            TensorQuantizer(QuantizerAttributeConfig()),
            TensorQuantizer(QuantizerAttributeConfig()),
        )

        sq.disable()
        assert sq[0].is_enabled is False and sq[1].is_enabled is False
        sq.enable()
        assert sq[0].is_enabled is True and sq[1].is_enabled is True

        max_calibrate(sq, lambda x: x(torch.randn(1, 4)))
        sq.reset_amax()
        assert sq[0].amax is None and sq[1].amax is None

        assert sq.fake_quant is True
        sq[0]._fake_quant = False
        assert sq.fake_quant is False and sq[1]._fake_quant is True
