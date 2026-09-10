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


import torch
import torch.nn as nn
from fire import Fire
from peft import PeftModel
from pydantic import BaseModel, ConfigDict
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

try:
    import tensorrt_llm
    from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

    if PYTHON_BINDINGS:
        from tensorrt_llm.runtime import ModelRunnerCpp
except ImportError:
    tensorrt_llm = None
    PYTHON_BINDINGS = None
    ModelRunner = None
    ModelRunnerCpp = None

try:
    from awq import AutoAWQForCausalLM
except ImportError:
    AutoAWQForCausalLM = None


class EvalModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_path: str
    trust_remote_code: bool = False
    max_input_length: int = 512
    max_output_length: int = 512
    dtype: str = "auto"

    def run(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def count_text_length(self, text: str) -> int:
        raise NotImplementedError

    def check_valid_length(self, text: str) -> bool:
        return self.count_text_length(text) <= self.max_input_length

    def load(self):
        raise NotImplementedError


class SeqToSeqModel(EvalModel):
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    lora_path: str = ""
    device: str = "cuda"
    load_8bit: bool = False

    def load(self):
        if self.model is None:
            args = {}
            if self.device == "cuda":
                args.update(device_map="auto")
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            if self.dtype != "auto":
                args.update(torch_dtype=getattr(torch, self.dtype))
            else:
                args.update(torch_dtype="auto")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code, **args
            )
            print_gpu_utilization()
            if self.lora_path:
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model.eval()
            if "device_map" not in args:
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        assert self.model is not None, "Model must be loaded before running."
        assert self.tokenizer is not None, "Tokenizer must be loaded before running."
        device = self.model.device if hasattr(self.model, "device") else self.device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_output_length,
            **kwargs,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def count_text_length(self, text: str) -> int:
        self.load()
        assert self.tokenizer is not None, "Tokenizer must be loaded to count text length."
        return len(self.tokenizer(text).input_ids)

    def get_choice(self, text: str, **kwargs) -> tuple[float, float]:
        self.load()
        assert self.model is not None, "Model must be loaded to get choices."
        assert self.tokenizer is not None, "Tokenizer must be loaded to get choices."
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        start_token = torch.tensor([[self.tokenizer.pad_token_id]], dtype=torch.long).to(
            self.device
        )
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                decoder_input_ids=start_token,
                **kwargs,
            ).logits[0, 0]
        a_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        b_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        a = float(predictions[a_index].cpu())
        b = float(predictions[b_index].cpu())
        return a, b


class CausalModel(SeqToSeqModel):
    def load(self):
        if self.model is None:
            args = {}
            if self.device == "cuda":
                args.update(device_map="auto")
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            args.update(torch_dtype=getattr(torch, self.dtype) if self.dtype != "auto" else "auto")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code, **args
            )
            self.model.eval()
            if "device_map" not in args:
                self.model.to(self.device)
            print_gpu_utilization()
            # Sampling with temperature will cause MMLU to drop
            self.model.generation_config.do_sample = False
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        assert self.model is not None, "Model must be loaded before running."
        assert self.tokenizer is not None, "Tokenizer must be loaded before running."
        device = self.model.device if hasattr(self.model, "device") else self.device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        if "RWForCausalLM" in str(type(self.model)) or "Falcon" in str(type(self.model)):
            # this key is used by falcon 180b, but not by falcon 40b
            inputs.pop("token_type_ids", None)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)

    def get_choice(self, text: str, **kwargs) -> tuple[float, float]:
        self.load()
        assert self.model is not None, "Model must be loaded to get choices."
        assert self.tokenizer is not None, "Tokenizer must be loaded to get choices."
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        a_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        b_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        a = float(predictions[a_index].cpu())
        b = float(predictions[b_index].cpu())
        return a, b


class AutoAWQCausalModel(SeqToSeqModel):
    def load(self):
        if self.model is None:
            args = {}
            if self.device == "cuda":
                args.update(device_map="auto")
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            args.update(torch_dtype=getattr(torch, self.dtype) if self.dtype != "auto" else "auto")
            self.model = AutoAWQForCausalLM.from_quantized(
                self.model_path, trust_remote_code=self.trust_remote_code, **args
            )
            self.model.eval()
            if "device_map" not in args:
                self.model.to(self.device)
            print_gpu_utilization()
            # Sampling with temperature will cause MMLU to drop
            self.model.config.do_sample = False
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        assert self.model is not None, "Model must be loaded before running."
        assert self.tokenizer is not None, "Tokenizer must be loaded before running."
        device = self.model.device if hasattr(self.model, "device") else self.device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        if "RWForCausalLM" in str(type(self.model)) or "Falcon" in str(type(self.model)):
            # this key is used by falcon 180b, but not by falcon 40b
            inputs.pop("token_type_ids", None)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)

    def get_choice(self, text: str, **kwargs) -> tuple[float, float]:
        self.load()
        assert self.model is not None, "Model must be loaded before running."
        assert self.tokenizer is not None, "Tokenizer must be loaded before running."
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        a_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        b_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        a = float(predictions[a_index].cpu())
        b = float(predictions[b_index].cpu())
        return a, b


def print_gpu_utilization():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1e9} GB")


def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = {
        "causal": CausalModel,
        "autoawq_causal": AutoAWQCausalModel,
    }
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
    return model_class(**kwargs)


class TrtllmPipeline:
    def __init__(self, tokenizer, model, model_name, pad_id, end_id, max_attention_window_size):
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
        self.pad_id = pad_id
        self.end_id = end_id
        self.max_attention_window_size = max_attention_window_size
        self.output_len = 2

    def __call__(self, prompt):
        rank = tensorrt_llm.mpi_rank()
        # Run the model in batch size 1 and beam size 1
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
        batch_input_ids: list[torch.Tensor] = [inputs]

        # For multi-choice tasks like MMLU, we don't need to adjust following parameters
        output_len = self.output_len
        top_k = 1
        top_p = 0.0

        input_lengths = [x.size(0) for x in batch_input_ids]

        with torch.no_grad():
            if isinstance(self.model, nn.Module):
                # Left padding for HF
                max_length = max(input_lengths)
                paddings = [
                    torch.ones(max_length - length, dtype=torch.int32) * self.pad_id
                    for length in input_lengths
                ]
                batch_input_ids = [torch.cat([pad, x]) for x, pad in zip(batch_input_ids, paddings)]
                batch_input_ids_tensor: torch.Tensor = torch.stack(batch_input_ids)
                batch_input_ids_tensor = batch_input_ids_tensor.cuda()
                with torch.no_grad():
                    # Use default temperature and top_k
                    outputs = self.model.generate(
                        batch_input_ids_tensor, max_new_tokens=output_len, top_k=top_k
                    )
                    output_ids = outputs[0, input_lengths[0] :]

            elif isinstance(self.model, (ModelRunnerCpp, ModelRunner)):
                outputs = self.model.generate(
                    batch_input_ids,
                    max_new_tokens=output_len,
                    max_attention_window_size=self.max_attention_window_size,
                    end_id=self.end_id,
                    pad_id=self.pad_id,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=1,
                )
                torch.cuda.synchronize()
                if rank == 0:
                    output_ids = outputs[0, 0, input_lengths[0] :]

        if rank == 0:
            return self.tokenizer.decode(output_ids, skip_special_tokens=True)
        else:
            return None

    def check_valid_length(self, prompt):
        if isinstance(self.model, nn.Module):
            return True
        input_len = len(self.tokenizer.encode(prompt))
        return (
            input_len <= self.model.max_input_len
            and input_len + self.output_len <= self.model.max_seq_len
        )


if __name__ == "__main__":
    Fire()
