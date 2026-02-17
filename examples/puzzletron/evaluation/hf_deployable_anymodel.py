# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
from typing import Any

import numpy as np
import torch
from nemo_deploy import ITritonDeployable
from nemo_deploy.utils import broadcast_list, cast_output, str_ndarray2list
from nemo_export_deploy_common.import_utils import (
    MISSING_TRITON_MSG,
    UnavailableError,
    null_decorator,
)
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    from pytriton.decorators import batch
    from pytriton.model_config import Tensor

    HAVE_TRITON = True
except (ImportError, ModuleNotFoundError):
    from unittest.mock import MagicMock

    HAVE_TRITON = False
    batch = MagicMock()
    Tensor = MagicMock()
    batch = null_decorator


LOGGER = logging.getLogger("NeMo")

SUPPORTED_TASKS = ["text-generation"]


class HuggingFaceLLMDeploy(ITritonDeployable):
    """A Triton inference server compatible wrapper for HuggingFace models.

    This class provides a standardized interface for deploying HuggingFace models
    in Triton inference server. It supports various NLP tasks and handles model
    loading, inference, and deployment configurations.

    Args:
        hf_model_id_path (Optional[str]): Path to the HuggingFace model or model identifier.
            Can be a local path or a model ID from HuggingFace Hub.
        hf_peft_model_id_path (Optional[str]): Path to the PEFT model or model identifier.
            Can be a local path or a model ID from HuggingFace Hub.
        tokenizer_id_path (Optional[str]): Path to the tokenizer or tokenizer identifier.
            If None, will use the same path as hf_model_id_path.
        model (Optional[AutoModel]): Pre-loaded HuggingFace model.
        tokenizer (Optional[AutoTokenizer]): Pre-loaded HuggingFace tokenizer.
        tokenizer_padding (bool): Whether to enable padding in tokenizer. Defaults to True.
        tokenizer_truncation (bool): Whether to enable truncation in tokenizer. Defaults to True.
        tokenizer_padding_side (str): Which side to pad on ('left' or 'right'). Defaults to 'left'.
        task (str): HuggingFace task type (e.g., "text-generation"). Defaults to "text-generation".
        **hf_kwargs: Additional keyword arguments to pass to HuggingFace model loading.
    """

    def __init__(
        self,
        hf_model_id_path: str | None = None,
        hf_peft_model_id_path: str | None = None,
        tokenizer_id_path: str | None = None,
        model: AutoModel | None = None,
        tokenizer: AutoTokenizer | None = None,
        tokenizer_padding=True,
        tokenizer_truncation=True,
        tokenizer_padding_side="left",
        task: str | None = "text-generation",
        torch_dtype: torch.dtype | None = "auto",
        device_map: str | None = "auto",
        **hf_kwargs,
    ):
        if not HAVE_TRITON:
            raise UnavailableError(MISSING_TRITON_MSG)

        if hf_model_id_path is None and model is None:
            raise ValueError("hf_model_id_path or model parameters has to be passed.")
        elif hf_model_id_path is not None and model is not None:
            LOGGER.warning(
                "hf_model_id_path will be ignored and the HuggingFace model set with model parameter will be used."
            )

        assert task in SUPPORTED_TASKS, "Task {} is not a support task.".format(task)

        self.hf_model_id_path = hf_model_id_path
        self.hf_peft_model_id_path = hf_peft_model_id_path
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_padding = tokenizer_padding
        self.tokenizer_truncation = tokenizer_truncation
        self.tokenizer_padding_side = tokenizer_padding_side

        if tokenizer_id_path is None:
            self.tokenizer_id_path = hf_model_id_path
        else:
            self.tokenizer_id_path = tokenizer_id_path

        if model is None:
            self._load(torch_dtype=torch_dtype, device_map=device_map, **hf_kwargs)

    def _load(
        self, torch_dtype: torch.dtype | None = "auto", device_map: str | None = "auto", **hf_kwargs
    ) -> None:
        """Load the HuggingFace pipeline with the specified model and task.

        This method initializes the HuggingFace AutoModel classes using the provided model
        configuration and task type. It handles the model and tokenizer loading
        process.

        Args:
            torch_dtype (torch.dtype): Data type for the model. Defaults to "auto".
            device_map (str): Device map for the model. Defaults to "auto".
            **hf_kwargs: Additional keyword arguments to pass to the HuggingFace model loading.

        Raises:
            AssertionError: If task is not specified.
        """
        assert self.task is not None, "A task has to be given for the generation task."

        if self.task == "text-generation":
            # =========================================================================
            # BEGIN ANYMODEL PATCH
            # Wraps model loading with deci_x_patcher for heterogeneous layer configs.
            # See: modelopt/torch/puzzletron/anymodel/puzzformer/utils.py
            # =========================================================================
            import os
            import sys

            modelopt_workdir = os.environ.get("MODELOPT_WORKDIR") or os.environ.get(
                "PUZZLE_WORKDIR"
            )
            if modelopt_workdir and modelopt_workdir not in sys.path:
                sys.path.insert(0, modelopt_workdir)
            from modelopt.torch.puzzletron.anymodel.models.llama import LlamaModelDescriptor
            from modelopt.torch.puzzletron.anymodel.puzzformer import deci_x_patcher

            with deci_x_patcher(model_descriptor=LlamaModelDescriptor):
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.hf_model_id_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    **hf_kwargs,
                )
            # =========================================================================
            # END ANYMODEL PATCH
            # =========================================================================

            if self.hf_peft_model_id_path is not None:
                self.model = PeftModel.from_pretrained(self.model, self.hf_peft_model_id_path)
        else:
            raise ValueError("Task {} is not supported.".format(self.task))
        num_gpus = torch.cuda.device_count()
        # If there is only one GPU, move the model to GPU. If you are using device_map as "auto" or "balanced",
        # the model will be moved to GPU automatically.
        if device_map is None and num_gpus >= 1 and self.model.device.type != "cuda":
            self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id_path,
            trust_remote_code=hf_kwargs.pop("trust_remote_code", False),
            padding=self.tokenizer_padding,
            truncation=self.tokenizer_truncation,
            padding_side=self.tokenizer_padding_side,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        **kwargs: Any,
    ) -> list[str]:
        """Generate text based on the provided input prompts.

        This method processes input prompts through the loaded pipeline and
        generates text according to the specified parameters.

        Args:
            **kwargs: Generation parameters including:
                - text_inputs: List of input prompts
                - max_length: Maximum number of tokens to generate
                - num_return_sequences: Number of sequences to generate per prompt
                - temperature: Sampling temperature
                - top_k: Number of highest probability tokens to consider
                - top_p: Cumulative probability threshold for token sampling
                - do_sample: Whether to use sampling, default is False for greedy decoding
                - echo: Whether to return prompt + generated text (True) or just generated text (False)
                - return_full_text: Whether to return full text or only generated part

        Returns:
            If output logits and output scores are False:
            List[str]: A list of generated texts, one for each input prompt.
            If output logits and output scores are True:
            Dict: A dictionary containing:
                - sentences: List of generated texts
                - logits: List of logits
                - scores: List of scores
                - input_lengths: List of input token lengths (for echo processing)

        Raises:
            RuntimeError: If the pipeline is not initialized.
        """
        if not self.model:
            raise RuntimeError("Model is not initialized")

        inputs = self.tokenizer(
            kwargs["text_inputs"],
            return_tensors="pt",
            padding=self.tokenizer_padding,
            truncation=self.tokenizer_truncation,
        )

        # Store input lengths to extract only generated tokens later
        input_lengths = [len(input_ids) for input_ids in inputs["input_ids"]]

        # Get echo parameter (default False - only return generated text)
        echo = kwargs.pop("echo", False)
        kwargs.pop("text_inputs")  # Remove text_inputs as it's already been tokenized

        kwargs = {**inputs, **kwargs}
        for key, val in kwargs.items():
            if torch.is_tensor(val):
                kwargs[key] = val.cuda()

        with torch.no_grad():
            generated_ids = self.model.generate(**kwargs)
        return_dict_in_generate = kwargs.get("return_dict_in_generate", False)
        if return_dict_in_generate:
            # Handle dict output (when logits/scores are requested)
            sequences = generated_ids["sequences"]
            output = {"sentences": [], "input_lengths": input_lengths, "sequences": sequences}

            if echo:
                # Return full text (prompt + generated).
                # HF model's generate returns the input/prompt tokens as well by default.
                for i, seq in enumerate(sequences):
                    full_text = self.tokenizer.decode(seq, skip_special_tokens=True)
                    output["sentences"].append(full_text)
            else:
                # Extract only the generated tokens (skip input tokens).
                # This is required as HF model's generate returns the input/prompt tokens
                # as well by default. (return_full_text is specific to some models)
                for i, seq in enumerate(sequences):
                    input_len = input_lengths[i] if i < len(input_lengths) else 0
                    generated_tokens = seq[input_len:]  # Skip input tokens
                    generated_text = self.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    output["sentences"].append(generated_text)

            if kwargs.get("output_logits", False):
                output["logits"] = generated_ids["logits"]
            if kwargs.get("output_scores", False):
                output["scores"] = generated_ids["scores"]
        else:
            # Handle list output (normal case)
            output = []
            if echo:
                # Return full text (prompt + generated), which is the default in case of HF model generate.
                for i, seq in enumerate(generated_ids):
                    full_text = self.tokenizer.decode(seq, skip_special_tokens=True)
                    output.append(full_text)
            else:
                # Extract only the generated tokens (skip input tokens) as the default
                # behavior returns the input/prompt tokens as well.
                for i, seq in enumerate(generated_ids):
                    input_len = input_lengths[i] if i < len(input_lengths) else 0
                    generated_tokens = seq[input_len:]  # Skip input tokens
                    generated_text = self.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    output.append(generated_text)

        return output

    def generate_other_ranks(self):
        """Generate function for ranks other than the rank 0."""
        while True:
            message = torch.empty(1, dtype=torch.long, device="cuda")
            torch.distributed.broadcast(message, src=0)
            if message == 0:
                prompts = broadcast_list(data=[None], src=0)
                (
                    temperature,
                    top_k,
                    top_p,
                    num_tokens_to_generate,
                    output_logits,
                    output_scores,
                ) = broadcast_list(data=[None], src=0)

                return_dict_in_generate = False
                if output_logits or output_scores:
                    return_dict_in_generate = True

                self.generate(
                    text_inputs=prompts,
                    do_sample=False,  # do_sample=False for greedy decoding
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    max_new_tokens=num_tokens_to_generate,
                    output_logits=output_logits,
                    output_scores=output_scores,
                    return_dict_in_generate=return_dict_in_generate,
                )
            else:
                return

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="max_length", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="max_batch_size", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="random_seed", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="max_length", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="output_logits", shape=(-1,), dtype=np.bool_, optional=True),
            Tensor(name="output_scores", shape=(-1,), dtype=np.bool_, optional=True),
        )
        return inputs

    @property
    def get_triton_output(self):
        return (
            Tensor(name="sentences", shape=(-1,), dtype=bytes),
            Tensor(name="logits", shape=(-1,), dtype=np.single),
            Tensor(name="scores", shape=(-1,), dtype=np.single),
        )

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        output_infer = {}

        try:
            prompts = str_ndarray2list(inputs.pop("prompts"))
            temperature = inputs.pop("temperature")[0][0] if "temperature" in inputs else 1.0
            top_k = int(inputs.pop("top_k")[0][0] if "top_k" in inputs else 1)
            top_p = inputs.pop("top_p")[0][0] if "top_p" in inputs else 0
            num_tokens_to_generate = (
                inputs.pop("max_length")[0][0] if "max_length" in inputs else 256
            )
            output_logits = (
                inputs.pop("output_logits")[0][0] if "output_logits" in inputs else False
            )
            output_scores = (
                inputs.pop("output_scores")[0][0] if "output_scores" in inputs else False
            )
            return_dict_in_generate = False
            if output_logits or output_scores:
                return_dict_in_generate = True

            if torch.distributed.is_initialized():
                if torch.distributed.get_world_size() > 1:
                    torch.distributed.broadcast(
                        torch.tensor([0], dtype=torch.long, device="cuda"), src=0
                    )
                    broadcast_list(prompts, src=0)
                    broadcast_list(
                        data=[
                            temperature,
                            top_k,
                            top_p,
                            num_tokens_to_generate,
                            output_logits,
                            output_scores,
                        ],
                        src=0,
                    )

            output = self.generate(
                text_inputs=prompts,
                do_sample=False,  # do_sample=False for greedy decoding
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=num_tokens_to_generate,
                output_logits=output_logits,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                echo=False,
            )

            if isinstance(output, dict):
                output_infer = {"sentences": cast_output(output["sentences"], np.bytes_)}

                if "scores" in output:
                    output_scores = []
                    for r in output["scores"]:
                        lp = torch.tensor(r).cpu().detach().numpy()
                        if len(lp) == 0:
                            output_scores.append([0])
                        else:
                            output_scores.append(lp)
                    output_infer["scores"] = np.array(output_scores).transpose(1, 0, 2)

                if "logits" in output:
                    output_logits = []
                    for r in output["logits"]:
                        lp = torch.tensor(r).cpu().detach().numpy()
                        if len(lp) == 0:
                            output_logits.append([0])
                        else:
                            output_logits.append(lp)
                    output_infer["logits"] = np.array(output_logits).transpose(1, 0, 2)
            else:
                output_infer = {"sentences": cast_output(output, np.bytes_)}

        except Exception as error:
            err_msg = "An error occurred: {}".format(str(error))
            output_infer["sentences"] = cast_output([err_msg], np.bytes_)

        return output_infer

    def _compute_logprobs(
        self,
        prompts: list[str],
        output_infer: dict[str, Any],
        compute_logprob: bool,
        n_top_logprobs: int,
        echo: bool,
    ):
        """Compute log probabilities and top log probabilities from model scores.
        Used by ray_infer_fn to provide OAI API compatible output for evaluations.

        This method processes the raw scores from model generation to compute:
        - Log probabilities for chosen tokens
        - Top-k log probabilities for each position (if requested)
        - Handles both prompt tokens (when echo=True) and generated tokens

        Args:
            prompts: List of input prompts
            output_infer: Dictionary containing model outputs including scores, sequences, and input_lengths
            compute_logprob: Whether to compute log probabilities
            n_top_logprobs: Number of top log probabilities to return (0 to disable)
            echo: Whether to include prompt token log probabilities

        Returns:
            Tuple[Optional[List], Optional[List]]:
                - log_probs_list: List of log probabilities for each sample (None if not computed)
                - top_logprobs_list: List of top-k log probabilities for each sample (None if not computed)
        """
        # Tokenize the prompts to get prompt token IDs (needed for echo)
        prompt_token_ids = None
        prompt_inputs = None
        if echo:
            prompt_inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=self.tokenizer_padding,
                truncation=self.tokenizer_truncation,
            )
            prompt_token_ids = prompt_inputs["input_ids"]
            # Move to same device as model
            for key, val in prompt_inputs.items():
                if torch.is_tensor(val):
                    prompt_inputs[key] = val.cuda()

        # Process each sample
        log_probs_list = []
        top_logprobs_list = []

        for sample_idx in range(len(prompts)):
            sample_log_probs = []
            sample_top_logprobs = []

            # Get the generated sequence for this sample
            sequences = output_infer["sequences"][sample_idx]

            # For echo, compute prompt token logprobs by running forward pass
            if echo and prompt_token_ids is not None:
                prompt_len = len(prompt_token_ids[sample_idx])

                # Run forward pass on prompt to get logits for prompt tokens as scores in output_infer contains
                # logits only for generated tokens.
                with torch.no_grad():
                    # Create input for this specific sample
                    sample_prompt_input = {
                        key: val[sample_idx : sample_idx + 1] for key, val in prompt_inputs.items()
                    }
                    prompt_outputs = self.model(**sample_prompt_input)
                    prompt_logits = prompt_outputs.logits[0]  # Shape: [seq_len, vocab_size]

                # Calculate log probs for each prompt token (except the first BOS token)
                for token_pos in range(1, prompt_len):  # Start from 1 to skip BOS
                    # The logit at position i-1 predicts token at position i
                    logit_for_current_token = prompt_logits[token_pos - 1]
                    current_token_id = prompt_token_ids[sample_idx][token_pos].item()

                    # Calculate log probabilities
                    log_probs = torch.nn.functional.log_softmax(logit_for_current_token, dim=-1)
                    chosen_log_prob = log_probs[current_token_id].item()
                    sample_log_probs.append(chosen_log_prob)

                    # Calculate top log probabilities if requested
                    if n_top_logprobs > 0:
                        top_log_probs_dict = {}
                        top_k_values, top_k_indices = torch.topk(
                            log_probs, min(n_top_logprobs, len(log_probs))
                        )
                        for k_idx in range(len(top_k_indices)):
                            token_id = top_k_indices[k_idx].item()
                            token_str = self.tokenizer.decode([token_id])
                            top_log_probs_dict[token_str] = top_k_values[k_idx].item()
                        sample_top_logprobs.append(top_log_probs_dict)

            # Process the scores for generated tokens
            for token_idx, score_tensor in enumerate(output_infer["scores"]):
                # Get the chosen token ID from the sequence
                # Scores start after the prompt, so we need to offset
                input_len = (
                    output_infer.get("input_lengths", [0])[sample_idx]
                    if "input_lengths" in output_infer
                    else 0
                )
                seq_idx = input_len + token_idx

                if seq_idx < len(sequences):
                    chosen_token_id = (
                        sequences[seq_idx].item()
                        if hasattr(sequences[seq_idx], "item")
                        else sequences[seq_idx]
                    )

                    # Calculate log probabilities
                    log_probs = torch.nn.functional.log_softmax(score_tensor[sample_idx], dim=-1)
                    chosen_log_prob = log_probs[chosen_token_id].item()
                    sample_log_probs.append(chosen_log_prob)

                    # Calculate top log probabilities if requested
                    if n_top_logprobs > 0:
                        top_log_probs_dict = {}
                        top_k_values, top_k_indices = torch.topk(
                            log_probs, min(n_top_logprobs, len(log_probs))
                        )
                        for k_idx in range(len(top_k_indices)):
                            token_id = top_k_indices[k_idx].item()
                            token_str = self.tokenizer.decode([token_id])
                            top_log_probs_dict[token_str] = top_k_values[k_idx].item()
                        sample_top_logprobs.append(top_log_probs_dict)

            log_probs_list.append(sample_log_probs)
            if n_top_logprobs > 0:
                top_logprobs_list.append(sample_top_logprobs)

        # Return log probs and top logprobs
        return_log_probs = log_probs_list if compute_logprob else None
        return_top_logprobs = top_logprobs_list if n_top_logprobs > 0 else None

        return return_log_probs, return_top_logprobs

    def ray_infer_fn(self, inputs: dict[Any, Any]):
        """Perform inference using Ray with dictionary inputs and outputs.

        Args:
            inputs (Dict[Any, Any]): Dictionary containing input parameters:
                - prompts: List of input prompts
                - temperature: Sampling temperature (optional)
                - top_k: Number of highest probability tokens to consider (optional)
                - top_p: Cumulative probability threshold for token sampling (optional)
                - max_tokens: Maximum number of tokens to generate (optional)
                - compute_logprob: Whether to compute log probabilities (optional)
                - n_top_logprobs: Number of top log probabilities to return (optional)
                - echo: Whether to echo the prompt in output (optional)

        Returns:
            Dict[str, Any]: Dictionary containing:
                - sentences: List of generated texts
                - log_probs: Optional list of log probabilities if compute_logprob is True
                - top_logprobs: Optional list of top log probabilities if n_top_logprobs > 0
        """
        import json

        try:
            prompts = inputs.pop("prompts")
            temperature = inputs.pop("temperature", 1.0)
            top_k = int(inputs.pop("top_k", 1))
            top_p = inputs.pop("top_p", 0.0)
            num_tokens_to_generate = inputs.pop("max_tokens", 256)
            output_logits = inputs.pop("output_logits", False)
            output_scores = inputs.pop("output_scores", False)
            compute_logprob = inputs.pop("compute_logprob", False)
            n_top_logprobs = inputs.pop("n_top_logprobs", 0)
            echo = inputs.pop("echo", False)

            output_infer = self._infer_fn_ray(
                prompts=prompts,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_tokens_to_generate=num_tokens_to_generate,
                output_logits=output_logits,
                output_scores=output_scores,
                compute_logprob=compute_logprob,
                n_top_logprobs=n_top_logprobs,
                echo=echo,
            )
            # Code to get logprobs (required in OAI API format for eval) from the scores in output_infer.
            if (
                (compute_logprob or n_top_logprobs > 0)
                and "scores" in output_infer
                and output_infer["scores"]
            ):
                log_probs_list, top_logprobs_list = self._compute_logprobs(
                    prompts=prompts,
                    output_infer=output_infer,
                    compute_logprob=compute_logprob,
                    n_top_logprobs=n_top_logprobs,
                    echo=echo,
                )

                # Add to output
                if log_probs_list is not None:
                    output_infer["log_probs"] = log_probs_list
                if top_logprobs_list is not None:
                    # Convert to JSON strings for compatibility
                    output_infer["top_logprobs"] = [
                        json.dumps(top_logprobs) for top_logprobs in top_logprobs_list
                    ]

                # Remove raw outputs that are not needed in the final response
                output_infer.pop("scores", None)
                output_infer.pop("sequences", None)
                output_infer.pop("input_lengths", None)
            return output_infer
        except Exception as error:
            err_msg = "An error occurred: {}".format(str(error))
            return {"sentences": [err_msg]}

    def _infer_fn_ray(
        self,
        prompts,
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        num_tokens_to_generate=256,
        output_logits=False,
        output_scores=False,
        compute_logprob=False,
        n_top_logprobs=0,
        echo=False,
        cast_output_func=None,
    ):
        """Common internal function for inference operations.

        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to consider
            top_p: Cumulative probability threshold for token sampling
            num_tokens_to_generate: Maximum number of tokens to generate
            output_logits: Whether to output logits
            output_scores: Whether to output scores
            compute_logprob: Whether to compute log probabilities
            n_top_logprobs: Number of top log probabilities to return
            echo: Whether to echo the prompt in output
            cast_output_func: Optional function to cast output values

        Returns:
            Dict containing inference results with raw outputs
        """
        # Enable return_dict if we need scores for logprobs or if output_logits/scores are requested
        return_dict_in_generate = (
            output_logits or output_scores or compute_logprob or n_top_logprobs > 0
        )
        # Enable output_scores if we need to compute logprobs. scores and logits from generate are both identical in
        # case of greedy decoding. Hence setting output_scores to True when compute_logprob or n_top_logprobs > 0.
        if compute_logprob or n_top_logprobs > 0:
            output_scores = True

        if torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                torch.distributed.broadcast(
                    torch.tensor([0], dtype=torch.long, device="cuda"), src=0
                )
                broadcast_list(prompts, src=0)
                broadcast_list(
                    data=[
                        temperature,
                        top_k,
                        top_p,
                        num_tokens_to_generate,
                        output_logits,
                        output_scores,
                    ],
                    src=0,
                )

        output = self.generate(
            text_inputs=prompts,
            do_sample=False,  # do_sample=False for greedy decoding
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=num_tokens_to_generate,
            output_logits=output_logits,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            echo=echo,
        )

        if isinstance(output, dict):
            return output

        else:
            return {"sentences": output}
