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

"""Bounded full-vocabulary capture of generated prediction rows in pinned vLLM V1.

Reference log-probabilities are an ephemeral per-example FP32 mmap, never top-k.
Candidate scoring reduces each vocabulary chunk to per-position KL immediately.
"""

import hashlib
import inspect
from pathlib import Path

__all__ = ["KLWorkerExtension"]


class KLWorkerExtension:
    """Capture full-vocabulary generated predictions in a source-pinned vLLM V1 runner."""

    def full_install(self, expected_runner_sha256):
        """Verify and intercept the reviewed runner's prompt-scoring callback."""
        import torch

        runner = self.model_runner
        if (
            f"{type(runner).__module__}.{type(runner).__name__}"
            != "vllm.v1.worker.gpu_model_runner.GPUModelRunner"
        ):
            raise RuntimeError("Unreviewed vLLM model runner")
        actual = hashlib.sha256(Path(inspect.getfile(type(runner))).read_bytes()).hexdigest()
        if actual != expected_runner_sha256:
            raise RuntimeError(f"Pinned runner changed: {actual}")
        if tuple(inspect.signature(runner._get_prompt_logprobs_dict).parameters) != (
            "hidden_states",
            "num_scheduled_tokens",
        ):
            raise RuntimeError("vLLM callback contract changed")
        if self.parallel_config.pipeline_parallel_size != 1:
            raise RuntimeError("Pipeline parallelism must be one")
        self._full_original = runner._get_prompt_logprobs_dict
        self._full_request = None
        self._full_result = None
        self._full_torch = torch
        runner._get_prompt_logprobs_dict = self._full_callback
        return {
            "rank": self.rank,
            "runner_sha256": actual,
            "vocab_size": self.model_config.get_vocab_size(),
        }

    def full_arm(self, request):
        """Arm one frozen sequence and its reference buffer or output destination."""
        import numpy as np

        if self._full_request is not None:
            raise RuntimeError("Capture already armed")
        prompt, sequence = request["prompt_tokens"], request["sequence"]
        if not 0 < prompt <= len(sequence):
            raise ValueError("Need prompt and continuation")
        vocab = self.model_config.get_vocab_size()
        # The final target is not fed back as input. Every input row can predict
        # a frozen target, including the final input row (also for one-token EOS).
        n = len(sequence) - prompt + 1
        self._full_request, self._full_result = request, None
        self._full_computed, self._full_next = 0, 0
        self._full_id = None
        self._full_bound = None
        self._full_chunks, self._full_kl = [], []
        self._full_heads = {str(k): [] for k in request["top_ks"]}
        self._full_mass = {str(k): [] for k in request["top_ks"]}
        self._full_cpu_error = 0.0
        self._full_normalization_error = 0.0
        self._full_writer = False
        self._full_mmap = None
        self._full_shape = (n, vocab)
        if request.get("reference") and self.rank == 0:
            reference = Path(request["reference"])
            if reference.stat().st_size != n * vocab * 4:
                raise ValueError("Reference mmap size mismatch")
            self._full_mmap = np.memmap(reference, mode="r", dtype=np.float32, shape=(n, vocab))

    def _full_callback(self, hidden_states, scheduled):
        if self._full_request is not None:
            self._full_capture(hidden_states, scheduled)
        return self._full_original(hidden_states, scheduled)

    def _full_capture(self, hidden_states, scheduled):
        import numpy as np
        from kl_eval import token_kl

        torch, runner, request = self._full_torch, self.model_runner, self._full_request
        assert request is not None
        if self._full_result is not None or len(scheduled) != 1:
            raise RuntimeError("Ambiguous/repeated capture")
        req_id, count = next(iter(scheduled.items()))
        state = runner.requests[req_id]
        computed = state.num_computed_tokens
        if (
            runner.input_batch.req_id_to_index != {req_id: 0}
            or computed != self._full_computed
            or hidden_states.shape[0] != count
            or int(runner.query_start_loc.np[0]) != 0
            or int(runner.query_start_loc.np[1]) != count
        ):
            raise RuntimeError("Cached, preempted or noncontiguous prefill")
        if self._full_id is None:
            if state.prompt_token_ids != request["sequence"]:
                raise RuntimeError("Frozen input changed")
            self._full_id, self._full_bound = req_id, state.prompt_token_ids
        elif req_id != self._full_id or state.prompt_token_ids is not self._full_bound:
            raise RuntimeError("Request identity changed")
        if runner.num_prompt_logprobs and not request.get("audit"):
            raise RuntimeError("Dense prompt output prohibited for full runs")
        first = max(computed, request["prompt_tokens"] - 1)
        stop = min(computed + count, len(request["sequence"]))
        # Input row j predicts target j+1; include the final target without feeding it back.
        local_start, local_stop = first - computed, max(first, stop) - computed
        generated_start = max(0, first - request["prompt_tokens"] + 1)
        generated_stop = max(generated_start, stop - request["prompt_tokens"] + 1)
        self._full_computed += count
        self._full_chunks.append([computed, count, generated_start, generated_stop])
        if generated_start != generated_stop:
            if generated_start != self._full_next:
                raise RuntimeError("Missing or duplicate prediction positions")
            selected = hidden_states[local_start:local_stop]
            for offset in range(0, len(selected), 64):
                block = selected[offset : offset + 64]
                logits = runner.model.compute_logits(block)
                # vLLM's TP logits processor may return global logits on every
                # rank. All ranks must enter its collectives, but only rank zero
                # owns file output and metric reduction.
                if logits is None or self.rank != 0:
                    continue
                if logits.shape != (len(block), self._full_shape[1]):
                    raise RuntimeError(
                        f"Global vocabulary mismatch: {logits.shape} versus {self._full_shape[1]}"
                    )
                logp = torch.log_softmax(logits.float(), dim=-1)
                if not torch.isfinite(logp).all():
                    raise RuntimeError("Nonfinite full-vocabulary log-probabilities")
                normalization = float(torch.logsumexp(logp, -1).abs().max().item())
                self._full_normalization_error = max(self._full_normalization_error, normalization)
                if normalization > 1e-5:
                    raise RuntimeError("Full-vocabulary probabilities do not normalize")
                start = generated_start + offset
                stop = start + len(block)
                self._full_writer = True
                if request.get("reference"):
                    assert self._full_mmap is not None
                    ref = torch.from_numpy(np.array(self._full_mmap[start:stop], copy=True)).to(
                        logp.device
                    )
                    metrics = token_kl(ref, logp, request["top_ks"], chunk_size=64)
                    self._full_kl.extend(metrics["full_vocab_kl"])
                    for k in self._full_heads:
                        self._full_heads[k].extend(metrics["conditional_topk_kl"][k])
                        self._full_mass[k].extend(metrics["reference_topk_mass"][k])
                    # Independent float64 CPU audit at first, middle and final position.
                    for index in (0, self._full_shape[0] // 2, self._full_shape[0] - 1):
                        if start <= index < stop:
                            p = np.asarray(self._full_mmap[index], dtype=np.float64)
                            q = logp[index - start].cpu().numpy().astype(np.float64)
                            expected = float(np.sum(np.exp(p) * (p - q), dtype=np.float64))
                            error = abs(expected - self._full_kl[index])
                            self._full_cpu_error = max(self._full_cpu_error, error)
                            if error > 1e-5:
                                raise RuntimeError("Independent CPU/GPU KL audit failed")
                    del ref
                else:
                    if self._full_mmap is None:
                        output = Path(request["output"])
                        output.parent.mkdir(parents=True, exist_ok=True)
                        self._full_mmap = np.memmap(
                            output, mode="w+", dtype=np.float32, shape=self._full_shape
                        )
                    self._full_mmap[start:stop] = logp.cpu().numpy()
                del logits, logp
            self._full_next = generated_stop
        if self._full_computed == len(request["sequence"]):
            self._full_finish()

    def _full_finish(self):
        assert self._full_request is not None
        n, vocab = self._full_shape
        if self._full_next != n:
            raise RuntimeError("Incomplete prediction coverage")
        if self._full_mmap is not None and not self._full_request.get("reference"):
            self._full_mmap.flush()
        self._full_mmap = None
        if self._full_writer and self._full_request.get("reference") and len(self._full_kl) != n:
            raise RuntimeError("Incomplete per-position KL")
        self._full_result = {
            "rank": self.rank,
            "writer": self._full_writer,
            "generated_tokens": n,
            "vocab_size": vocab,
            "chunks": self._full_chunks,
            "normalization_max_error": self._full_normalization_error,
            "cpu_gpu_max_error": self._full_cpu_error,
            "metrics": {
                "full_vocab_kl": self._full_kl,
                "conditional_topk_kl": self._full_heads,
                "reference_topk_mass": self._full_mass,
            },
        }

    def full_collect(self):
        """Return a complete capture and release the armed request."""
        if self._full_result is None:
            raise RuntimeError("No complete capture")
        result = self._full_result
        self._full_request = self._full_result = self._full_mmap = None
        return result
