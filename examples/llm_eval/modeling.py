# Adapted from https://github.com/declare-lab/instruct-eval/blob/720e66f627369266ed1cfd74426666ec37e524bc/modeling.py

# MIT License
#
# Copyright (c) 2023 Deep Cognition and Language Research (DeCLaRe) Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

# ruff: noqa: N806, PGH003
# type: ignore

"""NOTE: This file is only used to overwrite certain methods in the instruct-eval repo."""

import json
import signal
import time
from pathlib import Path

import openai
import rwkv
import rwkv.utils
import tiktoken
import torch
import torch.nn as nn
import transformers
from fire import Fire
from peft import PeftModel
from pydantic import BaseModel
from rwkv.model import RWKV
from torchvision.datasets.utils import download_url
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class EvalModel(BaseModel, arbitrary_types_allowed=True):
    model_path: str
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


class OpenAIModel(EvalModel):
    model_path: str
    engine: str = ""
    use_azure: bool = False
    tokenizer: tiktoken.Encoding | None
    api_endpoint: str = "https://research.openai.azure.com/"
    api_version: str = "2023-03-15-preview"
    timeout: int = 60
    temperature: float = 0.0

    def load(self):
        if self.tokenizer is None:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # chatgpt/gpt-4

        with open(self.model_path) as f:
            info = json.load(f)
            openai.api_key = info["key"]
            self.engine = info["engine"]

        if self.use_azure:
            openai.api_type = "azure"
            openai.api_base = self.api_endpoint
            openai.api_version = self.api_version

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        output = ""
        error_message = "The response was filtered"

        while not output:
            try:
                key = "engine" if self.use_azure else "model"
                kwargs = {key: self.engine}
                response = openai.ChatCompletion.create(
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout,
                    request_timeout=self.timeout,
                    temperature=0,  # this is the degree of randomness of the model's output
                    **kwargs,
                )
                if response.choices[0].finish_reason == "content_filter":
                    raise ValueError(error_message)
                output = response.choices[0].message.content
            except Exception as e:
                print(e)
                if error_message in str(e):
                    output = error_message

            if not output:
                print("OpenAIModel request failed, retrying.")

        return output

    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer.encode(text))

    def get_choice(self, prompt: str, **kwargs) -> str:
        self.load()

        def handler(signum, frame):
            raise Exception("Timeout")

        signal.signal(signal.SIGALRM, handler)

        for i in range(3):  # try 5 times
            signal.alarm(2)  # 5 seconds
            try:
                response = openai.ChatCompletion.create(
                    engine=self.model_path,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except Exception as e:
                if "content management policy" in str(e):
                    break
                else:
                    time.sleep(3)
        return "Z"


class SeqToSeqModel(EvalModel):
    model_path: str
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    lora_path: str = ""
    device: str = "cuda"
    load_8bit: bool = False
    attn_implementation: str | None = None

    def load(self):
        if self.model is None:
            args = {}
            if self.device == "cuda":
                args.update(device_map="auto")
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            args.update(torch_dtype=getattr(torch, self.dtype) if self.dtype != "auto" else "auto")
            if self.attn_implementation:
                args["attn_implementation"] = self.attn_implementation
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, **args)
            print_gpu_utilization()
            if self.lora_path:
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model.eval()
            if "device_map" not in args:
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
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
        return len(self.tokenizer(text).input_ids)

    def get_choice(self, text: str, **kwargs) -> tuple[float, float]:
        self.load()
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
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B


class CausalModel(SeqToSeqModel):
    def load(self):
        if self.model is None:
            args = {}
            if self.device == "cuda":
                args.update(device_map="auto")
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            args.update(torch_dtype=getattr(torch, self.dtype) if self.dtype != "auto" else "auto")
            if self.attn_implementation:
                args["attn_implementation"] = self.attn_implementation
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True, **args
            )
            self.model.eval()
            if "device_map" not in args:
                self.model.to(self.device)
            print_gpu_utilization()
            # Sampling with temperature will cause MMLU to drop
            self.model.generation_config.do_sample = False
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
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

    def run_batch(self, batch_input: transformers.BatchEncoding, **kwargs):
        # Run batched inference.
        self.load()
        outputs = self.model.generate(
            **batch_input,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
            **kwargs,
        )

        # Left padding, we need to remove the padding in the outputs
        _, length = batch_input.input_ids.shape
        output_ids = outputs[:, length:].cpu().tolist()
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def get_choice(self, text: str, **kwargs) -> tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B


class LlamaModel(SeqToSeqModel):
    use_template: bool = False
    """
    Not officially supported by AutoModelForCausalLM, so we need the specific class
    Optionally, we can use the prompt template from: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
    However, initial MMLU experiments indicate that the template is not useful for few-shot settings
    """

    def load(self):
        if self.tokenizer is None:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        if self.model is None:
            args = {}
            if self.device == "cuda":
                args.update(device_map="auto")
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            args.update(torch_dtype=getattr(torch, self.dtype) if self.dtype != "auto" else "auto")
            self.model = LlamaForCausalLM.from_pretrained(self.model_path, **args)
            print_gpu_utilization()
            if self.lora_path:
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model.eval()
            if "device_map" not in args:
                self.model.to(self.device)

    def run(self, prompt: str, **kwargs) -> str:
        if self.use_template:
            template = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            )
            text = template.format_map({"instruction": prompt})
        else:
            text = prompt

        self.load()
        device = self.model.device if hasattr(self.model, "device") else self.device
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        if "65b" in self.model_path.lower():
            self.max_input_length = 1024
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length,
            ).to(device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)

    def get_choice(self, text: str, **kwargs) -> tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B


def find_layers(module, layers=(nn.Conv2d, nn.Linear), name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1)
        )
    return res


def noop(*args, **kwargs):
    assert args is not None
    assert kwargs is not None


def load_quant(
    model,
    checkpoint,
    wbits,
    groupsize=-1,
    fused_mlp=True,
    warmup_autotune=True,
):
    import quant

    config = LlamaConfig.from_pretrained(model)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop
    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()

    layers = find_layers(model)
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]

    quant.make_quant_linear(model, layers, wbits, groupsize)
    del layers

    print("Loading model ...")
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load

        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)
    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)

    model.seqlen = 2048
    print("Done.")
    return model


def print_gpu_utilization():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1e9} GB")


class GPTQModel(LlamaModel):
    quantized_path: str
    model: LlamaForCausalLM | None = None
    tokenizer: LlamaTokenizer | None = None
    num_bits: int = 4
    group_size: int = 128

    def load(self):
        # https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/05781593c818d4dc8adc2d32c975e83d17d2b9a8/llama_inference.py
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not Path(self.quantized_path).exists():
            url = f"https://huggingface.co/{self.model_path}/resolve/main/{self.quantized_path}"
            download_url(url, root=".")

        if self.model is None:
            self.model = load_quant(
                model=self.model_path,
                checkpoint=self.quantized_path,
                wbits=self.num_bits,
                groupsize=self.group_size,
            )
            self.model.to(self.device)

        if self.tokenizer is None:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
            self.test_max_length()

    def test_max_length(self):
        # Detect any OOMs at the beginning
        text = " ".join(["test sentence for max length"] * 1000)
        self.run(text)


class ChatGLMModel(SeqToSeqModel):
    def load(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                self.model_path, trust_remote_code=True
            ).half()  # FP16 is required for ChatGLM
            self.model.eval()
            self.model.to(self.device)

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        response, history = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            **kwargs,
        )
        return response


class RWKVModel(EvalModel):
    tokenizer_path: str = "https://github.com/BlinkDL/ChatRWKV/raw/main/20B_tokenizer.json"
    download_root: str = "."
    model: rwkv.utils.PIPELINE | None = None

    def download(self, url: str) -> str:
        path = Path(self.download_root, Path(url).name)
        if not path.exists():
            download_url(url, root=self.download_root)
        return str(path)

    def load(self):
        model_path = self.download(self.model_path)
        tokenizer_path = self.download(self.tokenizer_path)

        if self.model is None:
            model = RWKV(model=model_path, strategy="cuda fp16")
            self.model = rwkv.utils.PIPELINE(model, tokenizer_path)

    def run(self, prompt: str, **kwargs) -> str:
        # Adapted from: https://github.com/BlinkDL/ChatRWKV/blob/main/v2/benchmark_more.py
        self.load()
        out_tokens = []
        out_last = 0
        out_str = ""
        occurrence = {}
        state = None
        token = None

        # ctx = f"Bob: {prompt.strip()}\n\nAlice:"
        ctx = prompt  # Special format has lower few-shot performance

        for i in range(self.max_output_length):
            tokens = self.model.encode(ctx) if i == 0 else [token]

            out, state = self.model.model.forward(tokens, state)
            for n in occurrence:
                out[n] -= 0.2 + occurrence[n] * 0.2

            token = self.model.sample_logits(out, temperature=1.0, top_p=0)
            if token == 0:
                break  # exit when 'endoftext'

            out_tokens += [token]
            occurrence[token] = 1 + occurrence.get(token, 0)

            tmp = self.model.decode(out_tokens[out_last:])
            if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
                # only print when the string is valid utf-8 and not end with \n
                out_str += tmp
                out_last = i + 1

            if "\n\n" in tmp:
                break  # exit when '\n\n'

        return out_str

    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.model.encode(text))


def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = {
        "seq_to_seq": SeqToSeqModel,
        "causal": CausalModel,
        "llama": LlamaModel,
        "chatglm": ChatGLMModel,
        "openai": OpenAIModel,
        "rwkv": RWKVModel,
        "gptq": GPTQModel,
    }
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
    return model_class(**kwargs)


def test_model(
    prompt: str = "Write an email about an alpaca that likes flan.",
    model_name: str = "seq_to_seq",
    model_path: str = "google/flan-t5-base",
    **kwargs,
):
    model = select_model(model_name, model_path=model_path, **kwargs)
    print(locals())
    print(model.run(prompt))


if __name__ == "__main__":
    Fire()
