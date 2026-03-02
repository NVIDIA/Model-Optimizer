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

# MIT License

# Copyright (c) 2025 sgl-project

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This file contains the wrapper for the SGL model.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from sglang.bench_one_batch import BenchArgs, _maybe_prepare_mlp_sync_batch, load_model
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import configure_logger
from transformers import AutoConfig

torch.manual_seed(42)


class LogitsProcessorForEAGLE3(torch.nn.Module):
    def __init__(self, logits_processor: LogitsProcessor, return_full_logits: bool = False):
        super().__init__()
        self.logits_processor = logits_processor
        self.return_full_logits = return_full_logits

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head,
        logits_metadata,
        aux_hidden_states: torch.Tensor | None = None,
    ) -> LogitsProcessorOutput:
        if self.return_full_logits:
            logits_metadata.forward_mode = ForwardMode.DECODE
        ret = self.logits_processor.forward(
            input_ids, hidden_states, lm_head, logits_metadata, aux_hidden_states
        )
        ret.last_hidden_states = hidden_states
        return ret


def wrap_logits_processors_in_module(module: nn.Module, return_full_logits: bool = False):
    for name, submodule in module.named_modules():
        if isinstance(submodule, LogitsProcessor):
            wrapped = LogitsProcessorForEAGLE3(submodule, return_full_logits)
            setattr(module, name, wrapped)
            print(f"wrapped {name} with LogitsProcessorForEAGLE3")


class SglangTargetModel(nn.Module):
    def __init__(
        self,
        args,
        tp_group,
        gpu_id,
        return_full_logits=False,
    ):
        super().__init__()
        self.return_full_logits = return_full_logits
        tp_rank = dist.get_rank(group=tp_group) if dist.is_initialized() else 0
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.args = args
        self.bench_args = BenchArgs()
        self.server_args = ServerArgs(model_path=args.model_path)
        self.server_args.enable_return_hidden_states = True
        self.server_args.context_length = args.max_length
        self.server_args.tp_size = args.tp_size
        self.server_args.ep_size = args.teacher_ep_size

        self.server_args.cuda_graph_max_bs = max(self.bench_args.batch_size)
        self.server_args.cuda_graph_bs = list(self.bench_args.batch_size)
        _set_envs_and_config(self.server_args)
        self.port_args = PortArgs.init_new(self.server_args)
        configure_logger(self.server_args, prefix=f" TP{tp_rank}")
        self.model_runner, _ = load_model(self.server_args, self.port_args, gpu_id, tp_rank)
        wrap_logits_processors_in_module(self.model_runner.model, return_full_logits)

    def set_aux_hidden_states_layers(self, aux_hidden_states_layers=None):
        config = AutoConfig.from_pretrained(
            self.server_args.model_path,
            trust_remote_code=self.server_args.trust_remote_code,
        )
        if aux_hidden_states_layers is None:
            if hasattr(config, "num_hidden_layers"):
                num_layers = config.num_hidden_layers
            elif hasattr(config, "text_config"):
                num_layers = config.text_config.num_hidden_layers
            else:
                raise ValueError(
                    f"config {config} does not have num_hidden_layers or text_config.num_hidden_layers"
                )
            # in sglang, when we do set_eagle3_layers_to_capture, we will add 1 to the layer index
            aux_hidden_states_layers = [
                2 - 1,
                num_layers // 2 - 1,
                num_layers - 3 - 1,
            ]
        self.aux_hidden_states_layers = aux_hidden_states_layers
        assert len(self.aux_hidden_states_layers) == 3, (
            "aux_hidden_states_layers is expected to be 3 layers"
        )
        print(f"Capturing Aux hidden states layers: {self.aux_hidden_states_layers}")

        if not hasattr(self.model_runner.model, "set_eagle3_layers_to_capture"):
            raise ValueError(
                f"model_runner.model {self.model_runner.model} does not have set_eagle3_layers_to_capture"
            )
        self.model_runner.model.set_eagle3_layers_to_capture(self.aux_hidden_states_layers)
        if hasattr(self.model_runner.model, "capture_aux_hidden_states"):
            assert self.model_runner.model.capture_aux_hidden_states, (
                "model_runner.model.capture_aux_hidden_states is expected to be True"
            )
        elif hasattr(self.model_runner.model.language_model, "capture_aux_hidden_states"):
            assert self.model_runner.model.language_model.capture_aux_hidden_states, (
                "model_runner.model.capture_aux_hidden_states is expected to be True"
            )
        else:
            raise ValueError(
                f"model_runner.model {self.model_runner.model} does not have capture_aux_hidden_states"
            )

    @torch.no_grad
    def extend(self, reqs: list[Req]):
        tree_cache = RadixCache(
            None,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.model_runner.server_args.page_size,
        )
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        _maybe_prepare_mlp_sync_batch(batch, self.model_runner)
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        logits_output, _ = self.model_runner.forward(forward_batch)
        input_lens = [len(req.origin_input_ids) for req in reqs]
        assert logits_output.last_hidden_states.shape[0] == sum(input_lens), (
            "the number of hidden states is not correct"
        )
        assert logits_output.hidden_states.shape[0] == sum(input_lens), (
            "the number of hidden states is not correct"
        )
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()
        return (
            logits_output.next_token_logits,
            logits_output.last_hidden_states,
            logits_output.hidden_states,
        )

    def forward(
        self,
        data_for_target: list[dict[str, torch.Tensor]],
    ):
        """
        arguments:
            data_for_target: List[Dict[str, torch.Tensor]] of target_batch_size
                - input_ids: (tp_size, seq_len)
                - attention_mask: (tp_size, seq_len)
                - loss_mask: (tp_size, seq_len)
        """
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs = []
        for idx, data in enumerate(data_for_target):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=data["input_ids"].view(-1).tolist(),
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            reqs.append(req)
        logits, hidden_states_list, aux_hidden_states_list = self.extend(reqs)
        return logits, hidden_states_list, aux_hidden_states_list
