===================================================================================
Guides to quantize a customized model from Hugging Face for TensorRT-LLM deployment
===================================================================================

ModelOpt can usually quantize PyTorch models from the Hugging Face directly. By default, ModelOpt searches the PyTorch model and replaces the torch ``nn.Linear`` module with a quantized linear module.

If the model happens not using the ``nn.Linear`` for the linear layers, a customized Hugging Face plugin needs to be implemented to convert the model to use ``nn.Linear`` instead.

The following is an example about how a customized Hugging Face model can be supported using modelopt:

The `DBRX model <https://huggingface.co/databricks/dbrx-instruct>`_ is an MoE model with customized MoE linear implementation. The MoE layer in DBRX is implemented as a `DbrxExperts <https://github.com/databricks/dbrx/blob/main/model/modeling_dbrx.py>`_ module, where the three linear ops (w1, v1 and v2) are represented as ``nn.Parameter``. The linear op is forwarded as a pure ``matmul`` op.

As ModelOpt cannot detect these linear ops out-of-the-box, a HugggingFace plugin is implemented as the following:

#. Define a customized ``_QuantDbrxExpertGLU`` as a ``DynamicModule`` with the same ``forward`` signature.
#. Rewrite the linear ops (w1, v1 and v2) as a standard ``nn.Linear`` op, and re-implement the ``forward`` method.
#. Register the new dynamic ``_QuantDbrxExperts`` to replace the ``DbrxExperts`` from the modeling_dbrx.py in the ``transformers`` library
#. Try quantize the DBRX model after the plugin is implemented, feel free to follow the `llm_ptq example <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq>`_.
#. TensorRT-LLM is open-sourced. If this customized model is not supported by TensorRT-LLM yet, please modify :meth:`export_tensorrt_llm_checkpoint <modelopt.torch.export.export_tensorrt_llm_checkpoint>` or :meth:`export_hf_checkpoint <modelopt.torch.export.export_hf_checkpoint>` to export the quantized model for deployment with a customized TensorRT-LLM modeling implementation. Feel free to :doc:`contact us <../support/1_contact>` if further support is needed.

The following code snippet is excerpted from ``modelopt/torch/quantization/plugins/huggingface.py``

.. code-block:: python

    from modelopt.torch.opt.dynamic import DynamicModule
    from modelopt.torch.quantization.nn import QuantModuleRegistry

    # For more information on DbrxExpert, see https://github.com/huggingface/transformers/blob/dcdda532/src/transformers/models/dbrx/modeling_dbrx.py#L756
    class _QuantDbrxExperts(DynamicModule):
        def _setup(self):
            """Modify the DbrxExpert."""
            # No setup is needed for DbrxExpert, we only need to update DbrxExpertGLU
            pass

        # forward method copied from the original dbrx repo - https://github.com/databricks/dbrx/blob/a3200393/model/modeling_dbrx.py#L795
        def forward(
            self,
            x: torch.Tensor,
            weights: torch.Tensor,
            top_weights: torch.Tensor,
            top_experts: torch.LongTensor,
        ) -> torch.Tensor:
            bsz, q_len, hidden_size = x.shape
            x = x.view(-1, hidden_size)
            out = torch.zeros_like(x)

            expert_mask = nn.functional.one_hot(top_experts, num_classes=self.moe_num_experts).permute(
                2, 1, 0
            )
            for expert_idx in range(0, self.moe_num_experts):
                topk_idx, token_idx = torch.where(expert_mask[expert_idx])
                if token_idx.shape[0] == 0:
                    continue

                token_list = token_idx.tolist()
                topk_list = topk_idx.tolist()

                expert_tokens = x[None, token_list].reshape(-1, hidden_size)
                expert_out = (
                    self.mlp(expert_tokens, expert_idx) * top_weights[token_list, topk_list, None]
                )

                out.index_add_(0, token_idx, expert_out)

            out = out.reshape(bsz, q_len, hidden_size)
            return out


    class _QuantDbrxExpertGLU(DynamicModule):
        def _setup(self):
            """Modify the DbrxExpertGLU by using nn.Linear layers."""
            dtype, device = self.w1.dtype, self.w1.device

            def _copy_weights(modules, weights):
                modules.to(dtype=dtype, device=device)
                for expert_idx, module in enumerate(modules):
                    with torch.no_grad():
                        module.weight.copy_(weights[expert_idx].detach())

            self.w1_linear = nn.ModuleList(
                [
                    nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=False)
                    for _ in range(self.moe_num_experts)
                ]
            )
            _copy_weights(
                self.w1_linear,
                self.w1.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size),
            )
            delattr(self, "w1")

            self.v1_linear = nn.ModuleList(
                [
                    nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=False)
                    for _ in range(self.moe_num_experts)
                ]
            )
            _copy_weights(
                self.v1_linear,
                self.v1.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size),
            )
            delattr(self, "v1")

            self.w2_linear = nn.ModuleList(
                [
                    nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=False)
                    for _ in range(self.moe_num_experts)
                ]
            )
            _copy_weights(
                self.w2_linear,
                self.w2.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size).transpose(
                    1, 2
                ),
            )
            delattr(self, "w2")

        def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
            x1 = self.w1_linear[expert_idx](x)
            x2 = self.v1_linear[expert_idx](x)
            x1 = self.activation_fn(x1)
            x1 = x1 * x2
            return self.w2_linear[expert_idx](x1)


    if transformers.models.dbrx.modeling_dbrx.DbrxExperts not in QuantModuleRegistry:
        QuantModuleRegistry.register(
            {transformers.models.dbrx.modeling_dbrx.DbrxExperts: "hf.DbrxExperts"}
        )(_QuantDbrxExperts)

    if transformers.models.dbrx.modeling_dbrx.DbrxExpertGLU not in QuantModuleRegistry:
        QuantModuleRegistry.register(
            {transformers.models.dbrx.modeling_dbrx.DbrxExpertGLU: "hf.DbrxExpertGLU"}
        )(_QuantDbrxExpertGLU)


    def register_dbrx_moe_on_the_fly(model):
        """Register DBRX MoE modules as QUANT_MODULE.

        The MoE class in DBRX is `transformers_modules.modeling_dbrx.DbrxExpertGLU`, which loads dynamically.
        """
        if type(model).__name__ in ["DbrxForCausalLM"]:
            moe_type = type(model.transformer.blocks[0].ffn.experts.mlp)
            # Create a QuantDbrxExpertGLU class on the fly
            if QuantModuleRegistry.get(moe_type) is None:
                QuantModuleRegistry.register({moe_type: moe_type.__name__})(_QuantDbrxExpertGLU)
