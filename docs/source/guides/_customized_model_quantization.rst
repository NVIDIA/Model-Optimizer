===================================================================================
Guides to quantize a customized model from Hugging Face for TensorRT-LLM deployment
===================================================================================

ModelOpt can usually quantize PyTorch models from the Hugging Face directly. By default, ModelOpt searches the PyTorch model and replaces the torch ``nn.Linear`` module with a quantized linear module.
Fused MoE experts that store their weights as 3-D ``gate_up_proj`` / ``down_proj`` parameters and call ``F.linear`` (e.g. Mixtral, Qwen3-MoE, DeepSeek-V3 in transformers 5.x) are also detected and quantized automatically.

If a model computes its linear ops some other way, a customized Hugging Face plugin is needed to insert the quantizers.

The following example shows how ModelOpt supports the `Llama 4 <https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct>`_ MoE.
Its `Llama4TextExperts <https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py>`_ module stores all experts in two ``nn.Parameter`` tensors, ``gate_up_proj`` of shape ``(num_experts, hidden_size, 2 * expert_dim)`` and ``down_proj`` of shape ``(num_experts, expert_dim, hidden_size)``, and applies them with ``torch.bmm``, so there is no ``nn.Linear`` for ModelOpt to replace.

The plugin works as follows:

#. Define a ``_QuantLlama4TextExperts`` subclass of ``QuantModule`` and create the input and weight ``TensorQuantizer`` modules in ``_setup``.
#. Re-implement ``forward`` with the same signature, quantizing the inputs and the weights of both ``bmm`` calls.
   ModelOpt's per-channel and per-block quantization expect the input dimension last, but these weights are ``(num_experts, in_dim, out_dim)``, so they are quantized transposed.
#. Register ``_QuantLlama4TextExperts`` to replace ``Llama4TextExperts`` from the ``transformers`` library.
#. Quantize the model after the plugin is registered, for example with the `hf_ptq example <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq>`_.
#. Export the quantized model with :meth:`export_hf_checkpoint <modelopt.torch.export.unified_export_hf.export_hf_checkpoint>`. A new module type may also need an export handler; see ``modelopt/torch/export/hf_export_handlers.py``.
   If the customized model is not supported by TensorRT-LLM, add support in its PyTorch backend. See the :doc:`unified HF export guide <../deployment/3_unified_hf>` or :doc:`contact us <../support/1_contact>` for help.

The following code snippet is simplified from the plugin in ``modelopt/torch/quantization/plugins/huggingface.py``, which also handles weight-only calibration.
For your own model, write a plugin like it in your own code and run it before quantizing.

.. code-block:: python

    import torch
    from transformers.models.llama4.modeling_llama4 import Llama4TextExperts

    from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry, TensorQuantizer


    class _TransposedQuantization(torch.autograd.Function):
        """Quantize a (num_experts, in_dim, out_dim) weight with in_dim last, using a straight-through gradient."""

        @staticmethod
        def forward(ctx, inputs, quantizer):
            return quantizer(inputs.transpose(-1, -2).contiguous()).transpose(-1, -2)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None


    _transposed_quantize = _TransposedQuantization.apply


    class _QuantLlama4TextExperts(QuantModule):
        def _setup(self):
            self.gate_up_proj_input_quantizer = TensorQuantizer()
            self.gate_up_proj_weight_quantizer = TensorQuantizer()
            self.down_proj_input_quantizer = TensorQuantizer()
            self.down_proj_weight_quantizer = TensorQuantizer()

        # Same as Llama4TextExperts.forward, with quantizers on both bmm inputs and weights
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(
                self.gate_up_proj_input_quantizer(hidden_states),
                _transposed_quantize(self.gate_up_proj, self.gate_up_proj_weight_quantizer),
            )
            gate, up = gate_up.chunk(2, dim=-1)
            next_states = torch.bmm(
                self.down_proj_input_quantizer(up * self.act_fn(gate)),
                _transposed_quantize(self.down_proj, self.down_proj_weight_quantizer),
            )
            return next_states.view(-1, self.hidden_size)


    if Llama4TextExperts not in QuantModuleRegistry:
        QuantModuleRegistry.register({Llama4TextExperts: "hf.Llama4TextExperts"})(
            _QuantLlama4TextExperts
        )
