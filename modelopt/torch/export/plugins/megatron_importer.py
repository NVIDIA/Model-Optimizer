# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Code that export quantized Megatron Core models for deployment."""

import tempfile
from pathlib import Path

import torch
import torch.distributed
from huggingface_hub import snapshot_download
from tqdm import tqdm

from modelopt.torch.utils import import_plugin

from .mcore_common import all_mcore_hf_import_mapping
from .mcore_custom import (
    CustomModuleMapping,
    ParallelConfig,
    dequantize_mxfp4_to_bf16,
    get_safetensor,
)

with import_plugin("transformers", verbose=False):
    import transformers

has_mcore = False
with import_plugin("megatron"):
    from megatron.core.parallel_state import (
        get_expert_tensor_parallel_world_size,
        get_tensor_model_parallel_world_size,
    )
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.torch_norm import L2Norm
    from megatron.core.transformer.transformer_layer import TransformerLayer

    has_mcore = True


class GPTModelImporter:
    """Megatron Core GPTModel HuggingFace Importer.

    The Importer is created by `import_mcore_gpt_from_hf` to host attributes
    and methods that import a Megatron Core GPTModel from a supported Hugging
    Face model.

    Args:
        model: The Megatron Core GPTModel instance.
        pretrained_model_name_or_path: Can be either: the *model id* of a
            pretrained model hosted inside a model repo on huggingface.co; or
            a *directory* containing model weights saved using
            [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
        dtype: The weights data type to export the unquantized layers.
    """

    weight_scale_name: str = "weight_scale_inv"

    def __init__(
        self,
        model: torch.nn.Module,
        pretrained_model_name_or_path: str,
        workspace_dir: str | None = None,
        dtype=torch.bfloat16,
        dequantize: bool = True,
        trust_remote_code: bool = False,
        verbose: bool = False,
        moe_router_dtype: torch.dtype | None = None,
    ):
        """Create a GPTModel importer instance."""
        self._hf_config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
        self.moe_router_dtype = None
        if moe_router_dtype == "fp32":
            self.moe_router_dtype = torch.float32
        elif moe_router_dtype == "fp64":
            self.moe_router_dtype = torch.float64

        pretrained_model_path = Path(pretrained_model_name_or_path)
        if not pretrained_model_path.is_dir():
            if workspace_dir is None:
                workspace_dir = tempfile.gettempdir()
            pretrained_model_path = workspace_dir + "/" + pretrained_model_name_or_path
            if torch.distributed.get_rank() == 0:
                snapshot_download(
                    repo_id=pretrained_model_name_or_path,
                    local_dir=pretrained_model_path,
                )
            torch.distributed.barrier()
        self.arch = self._hf_config.architectures[0]
        self.all_rules = self._populate_rule_book()
        self.rules = self.all_rules[self.arch]
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        self.dtype = dtype
        self.dequantize = dequantize
        self.verbose = verbose
        self.disable_tqdm = torch.distributed.get_rank() > 0 or verbose

    def _populate_rule_book(self):
        """The rule book maps each state_dict key to a Callable."""
        all_rules = {}

        def _custom_mapping_to_lambda(mapping):
            method_map = {
                "name_remapping": self._name_remapping,
                "qkv_merging": self._qkv_merging,
                "gated_mlp_merging": self._gated_mlp_merging,
                "unpack_name_remapping": self._unpack_name_remapping,
                "unpack_name_remapping_gpt_oss": self._unpack_name_remapping_gpt_oss,
            }
            func = method_map[mapping.func_name]
            prefix = mapping.target_name_or_prefix
            func_kwargs = mapping.func_kwargs
            return lambda m, *args, **kwargs: func(
                m, prefix.format(*args), **{**func_kwargs, **kwargs}
            )

        for arch, mappings in all_mcore_hf_import_mapping.items():
            all_rules[arch] = {
                k: _custom_mapping_to_lambda(v) if isinstance(v, CustomModuleMapping) else v
                for (k, v) in mappings.items()
                if isinstance(v, (CustomModuleMapping, bool))
            }

        return all_rules

    def _get_safetensor(self, key, parallel_config: ParallelConfig | None = None):
        return get_safetensor(
            self.pretrained_model_path, key, parallel_config, dequantize=self.dequantize
        )

    def _name_remapping(
        self,
        module,
        prefix,
        mapping={},
        parallel_config: ParallelConfig | None = None,
        dtype: torch.dtype | None = None,
    ):
        if dtype is None:
            dtype = self.dtype
        if isinstance(module, torch.Tensor):
            tensor = self._get_safetensor(prefix, parallel_config=parallel_config)
            module.data.copy_(tensor)
            return

        weight = module.state_dict().get("weight", None)
        weight_scale = module.state_dict().get("weight_quantizer._scale", None)

        state_dict = {}

        if weight is None:
            raise ValueError(f"{module!s} does not contain weight!")
        else:
            tensor = self._get_safetensor(prefix + "weight", parallel_config=parallel_config)

        if weight_scale is not None:
            scale_name = prefix + self.weight_scale_name
            if weight_scale.ndim > 0:
                scale = self._get_safetensor(scale_name, parallel_config=parallel_config)
            else:
                scale = self._get_safetensor(scale_name)
            scale = scale.to(weight_scale.dtype).to(device=weight_scale.device)
            state_dict["weight_quantizer._scale"] = scale

            if tensor.shape != weight.shape:
                expanded_tensor = torch.zeros(weight.shape, dtype=tensor.dtype)
                expanded_tensor[: tensor.shape[0], : tensor.shape[1]] = tensor
                tensor = expanded_tensor
            state_dict["weight"] = tensor.view(dtype=weight.dtype).to(device=weight.device)
        else:
            state_dict["weight"] = tensor.to(dtype=self.dtype).to(device=weight.device)

        # Handle the rest of the state_dict.
        for key, val in module.state_dict().items():
            if key in {"weight", "weight_quantizer._scale"}:
                continue
            elif "extra_state" in key:
                state_dict[key] = val
            else:
                source_key = mapping.get(key, key)
                # For bias tensors in ROW_TP layers, don't use parallel config to avoid sharding
                # since bias should always be replicated, not sharded
                if (
                    key == "bias"
                    and parallel_config is not None
                    and parallel_config.sharding_dim == 1
                ):
                    tensor = self._get_safetensor(prefix + source_key, parallel_config=None)
                else:
                    tensor = self._get_safetensor(
                        prefix + source_key, parallel_config=parallel_config
                    )
                state_dict[key] = tensor.to(dtype=dtype).to(device=val.device)

        module.load_state_dict(state_dict)

    def _gated_mlp_merging(
        self,
        module,
        prefix,
        gate_proj_name="gate_proj",
        up_proj_name="up_proj",
        parallel_config: ParallelConfig | None = None,
    ):
        weight = module.state_dict().get("weight", None)
        weight_scale = module.state_dict().get("weight_quantizer._scale", None)

        state_dict = {}

        if weight is None:
            raise ValueError(f"{module!s} does not contain weight!")
        else:
            gate_proj = self._get_safetensor(
                prefix + gate_proj_name + ".weight", parallel_config=parallel_config
            )
            up_proj = self._get_safetensor(
                prefix + up_proj_name + ".weight", parallel_config=parallel_config
            )
            tensor = torch.cat((gate_proj, up_proj), dim=0)

        if weight_scale is not None:
            gate_scale_name = prefix + gate_proj_name + "." + self.weight_scale_name
            up_scale_name = prefix + up_proj_name + "." + self.weight_scale_name
            if weight_scale.ndim > 0:
                gate_scale = self._get_safetensor(gate_scale_name, parallel_config=parallel_config)
                up_scale = self._get_safetensor(up_scale_name, parallel_config=parallel_config)
                scale = torch.cat((gate_scale, up_scale), dim=0)
            else:
                scale = self._get_safetensor(gate_scale_name)
                # If source model is per tensor, compute a per tensor scale with max.
                if scale.ndim > 0:
                    scale = scale.max(dim=0).max(dim=0)
            state_dict["weight_quantizer._scale"] = scale.to(weight_scale.dtype).to(
                device=weight_scale.device
            )
            state_dict["weight"] = tensor.view(weight.dtype).to(device=weight.device)
        else:
            state_dict["weight"] = tensor.to(self.dtype).to(device=weight.device)

        module.load_state_dict(state_dict)

    def _qkv_merging(
        self,
        module,
        prefix,
        q_proj_name="q_proj",
        k_proj_name="k_proj",
        v_proj_name="v_proj",
        parallel_config: ParallelConfig | None = None,
    ):
        config = module.config
        hidden_size = config.hidden_size
        num_query_groups = config.num_query_groups
        head_num = config.num_attention_heads
        head_size = config.kv_channels

        if parallel_config is not None:
            tp_size = get_tensor_model_parallel_world_size()
            assert head_num % tp_size == 0
            assert num_query_groups % tp_size == 0
            head_num = head_num // tp_size
            num_query_groups = num_query_groups // tp_size

        heads_per_group = head_num // num_query_groups
        qkv_total_dim = head_num + 2 * num_query_groups
        q_slice = torch.cat(
            [
                torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                for i in range(num_query_groups)
            ]
        )
        k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
        v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

        state_dict = {}

        weight = module.state_dict().get("weight", None)
        weight_scale = module.state_dict().get("weight_quantizer._scale", None)

        if weight is None:
            raise ValueError(f"{module!s} does not contain weight!")

        if weight_scale is not None:
            q_scale_name = prefix + q_proj_name + "." + self.weight_scale_name
            k_scale_name = prefix + k_proj_name + "." + self.weight_scale_name
            v_scale_name = prefix + v_proj_name + "." + self.weight_scale_name

            if weight_scale.ndim > 0:
                q_scale = self._get_safetensor(q_scale_name, parallel_config=parallel_config)
                k_scale = self._get_safetensor(k_scale_name, parallel_config=parallel_config)
                v_scale = self._get_safetensor(v_scale_name, parallel_config=parallel_config)
                weight_scale[q_slice] = q_scale.to(weight_scale.dtype).to(
                    device=weight_scale.device
                )
                weight_scale[k_slice] = k_scale.to(weight_scale.dtype).to(
                    device=weight_scale.device
                )
                weight_scale[v_slice] = v_scale.to(weight_scale.dtype).to(
                    device=weight_scale.device
                )
            else:
                q_scale = self._get_safetensor(q_scale_name)
                weight_scale = q_scale.to(weight_scale.dtype).to(device=weight_scale.device)
            state_dict["weight_quantizer._scale"] = weight_scale

        q_proj = self._get_safetensor(
            prefix + q_proj_name + ".weight", parallel_config=parallel_config
        )
        k_proj = self._get_safetensor(
            prefix + k_proj_name + ".weight", parallel_config=parallel_config
        )
        v_proj = self._get_safetensor(
            prefix + v_proj_name + ".weight", parallel_config=parallel_config
        )
        q_proj = q_proj.reshape(-1, head_size, hidden_size)
        k_proj = k_proj.reshape(-1, head_size, hidden_size)
        v_proj = v_proj.reshape(-1, head_size, hidden_size)
        tensor = weight.detach().clone().reshape([qkv_total_dim, head_size, hidden_size])

        if weight_scale is not None:
            tensor[q_slice] = q_proj.view(dtype=tensor.dtype).to(device=tensor.device)
            tensor[k_slice] = k_proj.view(dtype=tensor.dtype).to(device=tensor.device)
            tensor[v_slice] = v_proj.view(dtype=tensor.dtype).to(device=tensor.device)
        else:
            tensor[q_slice] = q_proj.to(dtype=tensor.dtype).to(device=tensor.device)
            tensor[k_slice] = k_proj.to(dtype=tensor.dtype).to(device=tensor.device)
            tensor[v_slice] = v_proj.to(dtype=tensor.dtype).to(device=tensor.device)

        state_dict["weight"] = tensor.reshape(-1, hidden_size)

        # Handle bias merging
        bias = module.state_dict().get("bias", None)
        if bias is not None:
            q_bias = self._get_safetensor(
                prefix + q_proj_name + ".bias", parallel_config=parallel_config
            )
            k_bias = self._get_safetensor(
                prefix + k_proj_name + ".bias", parallel_config=parallel_config
            )
            v_bias = self._get_safetensor(
                prefix + v_proj_name + ".bias", parallel_config=parallel_config
            )

            # Reshape separate biases to match the head structure
            q_bias = q_bias.reshape(-1, head_size)
            k_bias = k_bias.reshape(-1, head_size)
            v_bias = v_bias.reshape(-1, head_size)

            # Create target bias tensor with the same structure as the fused QKV
            bias_tensor = bias.detach().clone().reshape([qkv_total_dim, head_size])

            # Merge biases using the same slicing logic as weights
            bias_tensor[q_slice] = q_bias.to(dtype=bias_tensor.dtype).to(device=bias_tensor.device)
            bias_tensor[k_slice] = k_bias.to(dtype=bias_tensor.dtype).to(device=bias_tensor.device)
            bias_tensor[v_slice] = v_bias.to(dtype=bias_tensor.dtype).to(device=bias_tensor.device)

            state_dict["bias"] = bias_tensor.reshape(-1)

        module.load_state_dict(state_dict)

    def _unpack_name_remapping(
        self,
        module,
        prefix,
        layer_type: str,
        parallel_config: ParallelConfig | None = None,
    ):
        tensor = self._get_safetensor(prefix, parallel_config=parallel_config)

        for idx, sub_module in enumerate(module.children()):
            state_dict = {}
            linear_module = getattr(sub_module, layer_type)
            weight = linear_module.state_dict().get("weight", None)
            sub_tensor = tensor[idx]
            if weight is None:
                raise ValueError(f"{linear_module!s} does not contain weight!")
                # TODO (yueshen): Handle weight_scale case
            else:
                # Transpose to match huggingface format with Mcore format
                sub_tensor = sub_tensor.transpose(-1, -2)
                state_dict["weight"] = sub_tensor.to(dtype=self.dtype).to(device=weight.device)

            for key, val in linear_module.state_dict().items():
                if key in {"weight", "weight_quantizer._scale"}:
                    continue
                elif "extra_state" in key:
                    state_dict[key] = val

            linear_module.load_state_dict(state_dict)

    def _unpack_name_remapping_gpt_oss(
        self,
        module,
        prefix,
        layer_type: str,
        parallel_config: ParallelConfig | None = None,
    ):
        tensor_blocks = self._get_safetensor(prefix + "_blocks", parallel_config=parallel_config)
        tensor_bias = self._get_safetensor(prefix + "_bias", parallel_config=parallel_config)
        tensor_scales = self._get_safetensor(prefix + "_scales", parallel_config=parallel_config)
        tensor = dequantize_mxfp4_to_bf16(tensor_blocks, tensor_scales, dtype=self.dtype)

        for idx, sub_module in enumerate(module.children()):
            state_dict = {}
            linear_module = getattr(sub_module, layer_type)
            weight = linear_module.state_dict().get("weight", None)
            sub_tensor = tensor[idx]
            if weight is None:
                raise ValueError(f"{linear_module!s} does not contain weight!")
                # TODO (yueshen): Handle weight_scale case
            else:
                if layer_type == "linear_fc1":
                    # HF checkpoint has interleaved weights, need to de-interleave
                    # Pattern: [0,2,4,...,5758] -> [0,1,2,...,2879] and [1,3,5,...,5759] -> [2880,2881,...,5759]
                    height, width = sub_tensor.shape
                    half_height = height // 2

                    # Create de-interleaved tensor
                    deinterleaved_tensor = torch.zeros_like(sub_tensor)
                    deinterleaved_tensor[:half_height] = sub_tensor[
                        ::2
                    ]  # Even indices -> first half
                    deinterleaved_tensor[half_height:] = sub_tensor[
                        1::2
                    ]  # Odd indices -> second half
                    sub_tensor = deinterleaved_tensor

                state_dict["weight"] = sub_tensor.to(dtype=self.dtype).to(device=weight.device)

            for key, val in linear_module.state_dict().items():
                if key in {"weight", "weight_quantizer._scale"}:
                    continue
                elif "extra_state" in key:
                    state_dict[key] = val
                elif "bias" in key:
                    sub_tensor_bias = tensor_bias[idx]

                    if layer_type == "linear_fc1":
                        # HF checkpoint has interleaved bias, need to de-interleave
                        bias_len = sub_tensor_bias.shape[0]
                        half_bias_len = bias_len // 2

                        # Create de-interleaved bias tensor
                        deinterleaved_bias = torch.zeros_like(sub_tensor_bias)
                        deinterleaved_bias[:half_bias_len] = sub_tensor_bias[
                            ::2
                        ]  # Even indices -> first half
                        deinterleaved_bias[half_bias_len:] = sub_tensor_bias[
                            1::2
                        ]  # Odd indices -> second half
                        sub_tensor_bias = deinterleaved_bias

                    state_dict["bias"] = sub_tensor_bias.to(dtype=self.dtype).to(device=val.device)

            linear_module.load_state_dict(state_dict)

    def _import_state_dict(self):
        model = self.model

        layer_pbar = tqdm(model.decoder.layers, disable=self.disable_tqdm)

        # Embedding
        if hasattr(model, "embedding"):
            layer_pbar.set_description("Importing word embedding")
            self.rules["word_embeddings"](model.embedding.word_embeddings)

        # Decoder layers
        for layer in layer_pbar:
            layer_id = layer.layer_number - 1

            if isinstance(layer, MambaLayer):
                if not isinstance(layer.norm, IdentityOp):
                    self.rules["norm"](layer.norm, layer_id)

                self.rules["mixer_norm"](layer.mixer.norm, layer_id)
                self.rules["A_log"](layer.mixer.A_log, layer_id)
                self.rules["D"](layer.mixer.D, layer_id)
                self.rules["dt_bias"](layer.mixer.dt_bias, layer_id)

                self.rules["conv1d"](layer.mixer.conv1d, layer_id)
                self.rules["in_proj"](layer.mixer.in_proj, layer_id)
                self.rules["out_proj"](layer.mixer.out_proj, layer_id)

            elif isinstance(layer, TransformerLayer):
                if not isinstance(layer.input_layernorm, IdentityOp):
                    self.rules["input_layernorm"](layer.input_layernorm, layer_id)

                attention = layer.self_attention
                if not isinstance(attention, IdentityOp):
                    if "MLASelfAttention" in str(type(attention)):
                        if hasattr(attention, "linear_q_proj"):
                            layer_pbar.set_description("Importing MLA (without q LoRA)")
                            self.rules["linear_q_proj"](attention.linear_q_proj, layer_id)
                        else:
                            layer_pbar.set_description("Importing MLA (with q LoRA)")
                            self.rules["linear_q_down_proj"](attention.linear_q_down_proj, layer_id)
                            self.rules["linear_q_layernorm"](attention.q_layernorm, layer_id)
                            self.rules["linear_q_up_proj"](attention.linear_q_up_proj, layer_id)
                        self.rules["linear_kv_down_proj"](attention.linear_kv_down_proj, layer_id)
                        self.rules["linear_kv_layernorm"](attention.kv_layernorm, layer_id)
                        self.rules["linear_kv_up_proj"](attention.linear_kv_up_proj, layer_id)
                        self.rules["linear_proj"](attention.linear_proj, layer_id)
                    else:
                        layer_pbar.set_description("Importing GQA/MHA")
                        if attention.q_layernorm is not None and not isinstance(
                            attention.q_layernorm, (IdentityOp, L2Norm)
                        ):
                            self.rules["q_layernorm"](attention.q_layernorm, layer_id)
                            self.rules["k_layernorm"](attention.k_layernorm, layer_id)
                        self.rules["linear_qkv"](attention.linear_qkv, layer_id)
                        self.rules["linear_proj"](attention.linear_proj, layer_id)
                        if getattr(attention.core_attention, "softmax_offset", None) is not None:
                            self.rules["softmax_offset"](
                                attention.core_attention.softmax_offset, layer_id
                            )

                if not isinstance(layer.pre_mlp_layernorm, IdentityOp):
                    self.rules["pre_mlp_layernorm"](layer.pre_mlp_layernorm, layer_id)

                if not isinstance(layer.mlp, IdentityOp):
                    if "MoE" in str(type(layer.mlp)):
                        layer_pbar.set_description("Importing MoE")
                        self.rules["router"](
                            layer.mlp.router, layer_id, dtype=self.moe_router_dtype
                        )
                        if (
                            hasattr(layer.mlp, "shared_experts")
                            and layer.mlp.shared_experts is not None
                        ):
                            layer_pbar.set_description("Importing MoE shared experts")
                            fc1 = layer.mlp.shared_experts.linear_fc1
                            fc2 = layer.mlp.shared_experts.linear_fc2
                            self.rules["shared_experts.linear_fc1"](fc1, layer_id)
                            self.rules["shared_experts.linear_fc2"](fc2, layer_id)
                        if not self.rules.get("use_packed_local_experts", False):
                            for local_expert_id, expert in tqdm(
                                enumerate(layer.mlp.experts.local_experts),
                                desc="Importing MoE local experts",
                                leave=False,
                                disable=self.disable_tqdm,
                            ):
                                expert_id = layer.mlp.local_expert_indices[local_expert_id]
                                fc1 = expert.linear_fc1
                                fc2 = expert.linear_fc2
                                self.rules["local_experts.linear_fc1"](fc1, layer_id, expert_id)
                                self.rules["local_experts.linear_fc2"](fc2, layer_id, expert_id)
                        # We only support either EP or ETP for now
                        elif get_expert_tensor_parallel_world_size() > 1:
                            # ETP supports for packed MoE
                            # ETP is not supported for gpt-oss model
                            if self.arch in ["GptOssForCausalLM"]:
                                raise ValueError("ETP is not supported for gpt-oss model")
                            self.rules["local_experts.linear_fc1_etp"](
                                layer.mlp.experts.local_experts, layer_id
                            )
                            self.rules["local_experts.linear_fc2_etp"](
                                layer.mlp.experts.local_experts, layer_id
                            )
                        else:
                            # EP supports for packed MoE
                            self.rules["local_experts.linear_fc1_ep"](
                                layer.mlp.experts.local_experts, layer_id
                            )
                            self.rules["local_experts.linear_fc2_ep"](
                                layer.mlp.experts.local_experts, layer_id
                            )
                    else:
                        layer_pbar.set_description("Importing MLP")
                        self.rules["linear_fc1"](layer.mlp.linear_fc1, layer_id)
                        self.rules["linear_fc2"](layer.mlp.linear_fc2, layer_id)

            if self.verbose:
                print(
                    "{:3}/{:3} completes importing layer {:3}.".format(
                        torch.distributed.get_rank(), torch.distributed.get_world_size(), layer_id
                    ),
                    flush=True,
                )

        # Final layernorm
        if hasattr(model.decoder, "final_layernorm") and model.decoder.final_layernorm:
            self.rules["final_layernorm"](model.decoder.final_layernorm)

        if hasattr(model.decoder, "final_norm") and model.decoder.final_norm:
            self.rules["final_norm"](model.decoder.final_norm)

        # Output layer
        if hasattr(model, "output_layer") and not model.share_embeddings_and_output_weights:
            self.rules["output_layer"](model.output_layer)
        # MTP
        if hasattr(model, "mtp"):
            # MTP is the last layer in DeepSeek V3/R1
            layer_id += 1
            for mtp in model.mtp:
                self.rules["mtp.fc"](mtp.fc, layer_id)
                self.rules["mtp.enorm"](mtp.enorm, layer_id)
                self.rules["mtp.hnorm"](mtp.hnorm, layer_id)
                self.rules["mtp.input_layernorm"](mtp.decoder.layers[0].input_layernorm, layer_id)
                if hasattr(mtp.decoder.layers[0].self_attention, "linear_q_proj"):
                    self.rules["mtp.linear_q_proj"](
                        mtp.decoder.layers[0].self_attention.linear_q_proj, layer_id
                    )
                else:
                    self.rules["mtp.linear_q_down_proj"](
                        mtp.decoder.layers[0].self_attention.linear_q_down_proj, layer_id
                    )
                    self.rules["mtp.linear_q_layernorm"](
                        mtp.decoder.layers[0].self_attention.q_layernorm, layer_id
                    )
                    self.rules["mtp.linear_q_up_proj"](
                        mtp.decoder.layers[0].self_attention.linear_q_up_proj, layer_id
                    )
                self.rules["mtp.linear_kv_down_proj"](
                    mtp.decoder.layers[0].self_attention.linear_kv_down_proj, layer_id
                )
                self.rules["mtp.linear_kv_layernorm"](
                    mtp.decoder.layers[0].self_attention.kv_layernorm, layer_id
                )
                self.rules["mtp.linear_kv_up_proj"](
                    mtp.decoder.layers[0].self_attention.linear_kv_up_proj, layer_id
                )
                self.rules["mtp.linear_proj"](
                    mtp.decoder.layers[0].self_attention.linear_proj, layer_id
                )
                self.rules["mtp.pre_mlp_layernorm"](
                    mtp.decoder.layers[0].pre_mlp_layernorm, layer_id
                )
                self.rules["mtp.router"](mtp.decoder.layers[0].mlp.router, layer_id)
                self.rules["mtp.shared_experts.linear_fc1"](
                    mtp.decoder.layers[0].mlp.shared_experts.linear_fc1, layer_id
                )
                self.rules["mtp.shared_experts.linear_fc2"](
                    mtp.decoder.layers[0].mlp.shared_experts.linear_fc2, layer_id
                )
                for expert_id, expert in tqdm(
                    enumerate(mtp.decoder.layers[0].mlp.experts.local_experts),
                    desc="Importing MoE local experts",
                    leave=False,
                    disable=self.disable_tqdm,
                ):
                    self.rules["mtp.local_experts.linear_fc1"](
                        expert.linear_fc1, layer_id, expert_id
                    )
                    self.rules["mtp.local_experts.linear_fc2"](
                        expert.linear_fc2, layer_id, expert_id
                    )
