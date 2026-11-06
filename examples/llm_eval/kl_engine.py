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

"""Isolated resident vLLM engines for bounded, per-example KL capture."""

import multiprocessing
import os
import time
import traceback
from contextlib import suppress
from types import SimpleNamespace

__all__ = ["Engine"]


def _worker(pipe, args, model, role, devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    for name in ("VLLM_CACHE_ROOT", "TRITON_CACHE_DIR"):
        os.environ[name] = str(args.work_dir / f"{name.lower()}-{role}")
    llm = None
    try:
        # Optional CUDA dependencies must initialize only after device isolation.
        import numpy as np
        import torch
        import vllm
        from kl_eval import _continuation_logprobs, token_kl
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(model),
            dtype="bfloat16",
            tensor_parallel_size=len(devices.split(",")),
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.context_limit,
            max_num_seqs=1,
            max_num_batched_tokens=args.prefill_chunk_size,
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            async_scheduling=False,
            generation_config="vllm",
            seed=0,
            max_logprobs=-1,
            logprobs_mode="raw_logprobs",
            kv_cache_dtype="bfloat16" if role == "reference" else args.kv_cache_dtype,
            trust_remote_code=args.trust_remote_code,
            worker_extension_cls="kl_worker.KLWorkerExtension",
        )
        config = llm.llm_engine.vllm_config
        if role == "reference" and config.model_config.quantization is not None:
            raise ValueError("Reference must be an unquantized BF16 checkpoint.")
        contracts = llm.collective_rpc("full_install", args=(args.runner_sha256,))
        vocab_size = config.model_config.get_vocab_size()
        if args.top_ks and max(args.top_ks) > vocab_size:
            raise ValueError("Requested top-k exceeds the output vocabulary.")
        runtime = {
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "quantization": config.model_config.quantization,
            "contracts": contracts,
            "kv_cache_dtype": str(config.cache_config.cache_dtype),
            "vocab_size": vocab_size,
        }
        pipe.send(runtime)
        while True:
            command = pipe.recv()
            if command["op"] == "shutdown":
                break
            start = time.monotonic()
            row = command["row"]
            if command["op"] == "generate":
                cap = min(args.max_new_tokens, args.context_limit - len(row["prompt_ids"]))
                raw = llm.generate(
                    [{"prompt_token_ids": row["prompt_ids"]}],
                    SamplingParams(
                        temperature=0,
                        max_tokens=cap,
                        ignore_eos=True,
                        stop_token_ids=args.eos_token_ids,
                        detokenize=False,
                    ),
                    use_tqdm=False,
                )[0]
                if raw.prompt_token_ids != row["prompt_ids"]:
                    raise ValueError("Generation changed the frozen prompt.")
                ids = list(raw.outputs[0].token_ids)
                if not 1 <= len(ids) <= cap or any(t in args.eos_token_ids for t in ids[:-1]):
                    raise ValueError("Invalid generated length or tokens after EOS.")
                if len(ids) < cap and ids[-1] not in args.eos_token_ids:
                    raise ValueError("Generation stopped before the cap without EOS.")
                result = {
                    **row,
                    "generated_ids": list(raw.outputs[0].token_ids),
                    "finish_reason": raw.outputs[0].finish_reason,
                }
            elif command["op"] == "score":
                request = {
                    "sequence": row["prompt_ids"] + row["generated_ids"][:-1],
                    "prompt_tokens": len(row["prompt_ids"]),
                    "output": command["output"],
                    "reference": command.get("reference"),
                    "top_ks": args.top_ks,
                    "audit": args.audit,
                }
                llm.collective_rpc("full_arm", args=(request,))
                raw = llm.generate(
                    [{"prompt_token_ids": request["sequence"]}],
                    SamplingParams(
                        temperature=0,
                        max_tokens=1,
                        detokenize=False,
                        prompt_logprobs=-1 if args.audit else None,
                        logprobs=-1 if args.audit else None,
                        flat_logprobs=True,
                    ),
                    use_tqdm=False,
                )[0]
                if raw.prompt_token_ids != request["sequence"]:
                    raise ValueError("Scoring changed the frozen input.")
                replicas = llm.collective_rpc("full_collect")
                writers = [r for r in replicas if r["writer"]]
                if len(writers) != 1 or writers[0]["generated_tokens"] != len(row["generated_ids"]):
                    raise ValueError("Incomplete prediction capture.")
                result = writers[0]
                if args.audit:
                    # Prompt scores cover targets except the last; the single sampled
                    # output exposes that last prediction's entire raw distribution.
                    last = raw.outputs[0].logprobs
                    if hasattr(last, "start_indices"):
                        last = SimpleNamespace(
                            start_indices=[0, *last.start_indices],
                            end_indices=[0, *last.end_indices],
                            token_ids=last.token_ids,
                            logprobs=last.logprobs,
                        )
                    else:
                        last = [None, *last]
                    final = _continuation_logprobs(
                        SimpleNamespace(prompt_token_ids=[0, 0], prompt_logprobs=last),
                        1,
                        vocab_size,
                    )
                    dense = (
                        torch.cat(
                            (_continuation_logprobs(raw, len(row["prompt_ids"]), vocab_size), final)
                        )
                        if len(row["generated_ids"]) > 1
                        else final
                    )
                    path = command.get("reference") or command["output"]
                    ref = torch.from_numpy(
                        np.array(
                            np.memmap(
                                path,
                                mode="r",
                                dtype=np.float32,
                                shape=(len(row["generated_ids"]), vocab_size),
                            ),
                            copy=True,
                        )
                    )
                    if command.get("reference"):
                        expected = token_kl(ref, dense, args.top_ks)
                        errors = {
                            "full_vocab_kl": max(
                                abs(a - b)
                                for a, b in zip(
                                    expected["full_vocab_kl"], result["metrics"]["full_vocab_kl"]
                                )
                            )
                        }
                        for field in ("conditional_topk_kl", "reference_topk_mass"):
                            for k in expected[field]:
                                errors[f"{field}.{k}"] = max(
                                    abs(a - b)
                                    for a, b in zip(expected[field][k], result["metrics"][field][k])
                                )
                        error = max(errors.values())
                    else:
                        error = (ref - dense).abs().max().item()
                        errors = {"logprobs": error}
                    if error > 2e-5:
                        raise ValueError(f"Dense capture audit failed: {errors}")
                    result["dense_max_error"] = error
            else:
                raise ValueError("Unknown engine command.")
            result["wall_seconds"] = time.monotonic() - start
            pipe.send(result)
    except BaseException:
        traceback.print_exc()
        pipe.send({"error": traceback.format_exc()})
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        pipe.close()


class Engine:
    """Own a model process and propagate worker failures to the study controller."""

    def __init__(self, args, model, role, devices):
        context = multiprocessing.get_context("spawn")
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, args, model, role, devices))
        self.process.start()
        child.close()
        try:
            self.runtime = self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        while not self.pipe.poll(1):
            if not self.process.is_alive():
                raise RuntimeError(f"Engine exited with code {self.process.exitcode}.")
        result = self.pipe.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def call(self, **command):
        """Generate a continuation or score a frozen example."""
        self.pipe.send(command)
        return self._receive()

    def close(self):
        """Shut down model workers, terminating an unresponsive process."""
        if self.process.is_alive():
            with suppress(BrokenPipeError, EOFError):
                self.pipe.send({"op": "shutdown"})
            self.process.join(30)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(15)
        self.pipe.close()
