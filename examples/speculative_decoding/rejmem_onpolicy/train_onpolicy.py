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

"""Synchronous on-policy training of a DFlash drafter with rejection memory, on vLLM rollouts.

One process per node under torchrun, next to that node's rollout servers (see launch/node.sh)::

    torchrun --nnodes N --nproc_per_node 1 ... train_onpolicy.py --config CONFIG \\
        [--resume_from CHECKPOINT] [key.path=value ...]

Step s on every rank: generate the rank's next prompts on its servers with the current drafter, load
the records the servers captured, take one optimizer step on them (DDP), push the new drafter to the
servers. The drafter is also pushed before the first rollout, so a record must come from the last
version pushed, otherwise the run aborts.
"""

import argparse
import contextlib
import dataclasses
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import rollout
import torch
import torch.distributed as dist
import transformers
from torch.nn.parallel import DistributedDataParallel

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.plugins.hf_training_args import (
    TrainingArguments as SpecTrainingArgs,
)
from modelopt.torch.speculative.plugins.master_weight_adamw import MasterWeightAdamW
from modelopt.torch.speculative.plugins.rejmem.batch import build_batch
from modelopt.torch.speculative.plugins.rejmem.weight_push import DraftWeightPusher
from modelopt.torch.speculative.utils import load_vlm_or_llm

# HF TrainingArguments carrying the recipe's speculative-decoding fields, as in ../main.py.
HfTrainingArguments = dataclasses.make_dataclass(
    "HfTrainingArguments",
    [
        (name, fi.annotation, dataclasses.field(default=fi.default))
        for name, fi in SpecTrainingArgs.model_fields.items()
    ],
    bases=(transformers.TrainingArguments,),
)


# Per-step sums over ranks that the log reports as they are.
_COUNTS = ("prompts", "think_prompts", "truncated", "prompts_skipped", "gen_tokens", "record_bytes")


class OffPolicyError(RuntimeError):
    """A rollout record that the current drafter did not generate."""


def main(connect=DraftWeightPusher):
    """Parse the CLI and train; ``connect(server_urls)`` gives the drafter's push to the servers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="DFlash recipe with an onpolicy section")
    parser.add_argument("--resume_from", type=Path, help="checkpoint whose export the servers run")
    args, overrides = parser.parse_known_args()
    recipe, cfg = rollout.load_config(args.config, overrides)
    OnPolicyTrainer(recipe, cfg, args.resume_from, connect).run()


class OnPolicyTrainer:
    """One rank of the synchronous loop: its node's servers, its prompt shard, a DDP drafter replica."""

    def __init__(self, recipe, cfg, resume_from, connect):
        self.recipe, self.cfg = recipe, cfg
        # Under a multi-process torchrun, building the arguments initializes torch.distributed.
        self.targs = targs = HfTrainingArguments(**recipe.training.model_dump())
        _check_config(recipe, cfg, targs)
        if not dist.is_initialized():
            dist.init_process_group("nccl" if targs.device.type == "cuda" else "gloo")
        self.device, self.rank, self.world = targs.device, dist.get_rank(), dist.get_world_size()
        torch.manual_seed(targs.seed)

        if resume_from is not None:
            # The checkpoint's drafter is its export; _load restores the rest.
            recipe.dflash.dflash_init_checkpoint = str(resume_from / "export")
        model = load_vlm_or_llm(
            recipe.model.model_name_or_path,
            use_fake_base=recipe.model.use_fake_base_for_offline,
            use_offline_training=True,
            dtype="auto",
            device_map="cpu",
            trust_remote_code=recipe.model.trust_remote_code,
        )
        mtsp.convert(model, [("dflash", recipe.dflash.model_dump())])
        self.model = model.to(self.device)
        self.ddp = DistributedDataParallel(
            model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
            find_unused_parameters=bool(targs.ddp_find_unused_parameters),
        )
        self.optimizer = _build_optimizer(model, targs)
        self.scheduler = transformers.get_scheduler(
            targs.lr_scheduler_type,
            self.optimizer,
            num_warmup_steps=targs.get_warmup_steps(targs.max_steps),
            num_training_steps=targs.max_steps,
            scheduler_specific_kwargs=targs.lr_scheduler_kwargs,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            recipe.model.model_name_or_path, trust_remote_code=recipe.model.trust_remote_code
        )
        seed = targs.seed if targs.data_seed is None else targs.data_seed
        self.source = rollout.PromptSource(
            cfg.data_dir, tokenizer, cfg.max_prompt_len, seed, self.rank, self.world
        )
        self.start = 0 if resume_from is None else self._load(resume_from)
        self.launch_ckpt = None if resume_from is None else resume_from.resolve()
        self.out_dir = Path(targs.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if resume_from is not None and self.rank == 0:
            _drop_metrics_after(self.out_dir / "metrics.jsonl", self.start)
        # Tags name the record files; the nonce keeps another run's leftovers from matching.
        self.tag_prefix = f"{uuid.uuid4().hex[:8]}-r{self.rank}"
        self.pusher = _on_all_ranks(self.device, lambda: connect(cfg.server_urls))
        # Whatever export the servers were launched from, the first rollout uses this drafter.
        _on_all_ranks(self.device, lambda: self.pusher.push(self.model))

    def run(self):
        """Train from ``self.start`` to ``max_steps``, checkpointing every ``save_steps``."""
        for step in range(self.start, self.targs.max_steps):
            # Everything that can fail locally runs before the step's first collective and is
            # agreed on, so an error on one rank stops all ranks instead of leaving them waiting.
            batches, counts, times = _on_all_ranks(self.device, lambda: self._collect(step))
            t0 = time.perf_counter()
            counts |= self._train_step(batches)
            t1 = time.perf_counter()
            _on_all_ranks(self.device, lambda: self.pusher.push(self.model))
            times |= {"train": t1 - t0, "push": time.perf_counter() - t1}
            self._log(step + 1, counts, times)
            if (step + 1) % int(self.targs.save_steps) == 0 or step + 1 == self.targs.max_steps:
                self._save(step + 1)
        self.pusher.close()
        dist.destroy_process_group()

    def _collect(self, step):
        """Roll out this rank's prompts of ``step`` and turn the records into micro-batches."""
        cfg, dflash, seq_len = self.cfg, self.recipe.dflash, self.targs.training_seq_len
        t0 = time.perf_counter()
        skipped = self.source.num_skipped
        prompts = self.source.take(cfg.prompts_per_step)
        tags = [f"{self.tag_prefix}-s{step}-{i}" for i in range(len(prompts))]
        max_tokens = [seq_len - len(p.token_ids) for p in prompts]
        # The records load as their requests finish, so t_rollout holds most of their loading.
        rollouts, records, nbytes = rollout.generate_and_load(
            cfg, self.recipe.model.model_name_or_path, prompts, max_tokens, tags
        )
        rollout.check_memory_rows(cfg.server_urls, cfg.request_timeout)
        t1 = time.perf_counter()
        _check_on_policy(records, version=self.pusher.weight_version)
        seed = np.random.SeedSequence((self.targs.seed, self.rank, step)).generate_state(1).item()
        generator = torch.Generator().manual_seed(seed)
        micro = self.targs.per_device_train_batch_size
        batches = [
            build_batch(
                records[i : i + micro],
                [r.token_ids for r in rollouts[i : i + micro]],
                [r.prompt_len for r in rollouts[i : i + micro]],
                seq_len=seq_len,
                num_anchors=dflash.dflash_num_anchors,
                block_size=dflash.dflash_block_size,
                aux_layer_ids=self.model.target_layer_ids,
                memory=cfg.memory,
                generator=generator,
            )
            for i in range(0, len(records), micro)
        ]
        accepted = torch.cat([rec.tensors["round_num_accepted"] for rec in records])
        verified = accepted[accepted >= 0]
        counts = {
            "prompts": len(prompts),
            "think_prompts": sum(p.enable_thinking for p in prompts),
            "truncated": sum(r.finish_reason == "length" for r in rollouts),
            "prompts_skipped": self.source.num_skipped - skipped,
            "gen_tokens": sum(len(r.token_ids) - r.prompt_len for r in rollouts),
            "verified_rounds": verified.numel(),
            "accepted_plus_one": int(verified.sum()) + verified.numel(),
            "record_bytes": nbytes,
        }
        return batches, counts, {"rollout": t1 - t0, "records": time.perf_counter() - t1}

    def _train_step(self, batches):
        """One optimizer step over the micro-batches: the mean of their mean losses, DDP-averaged."""
        self.ddp.train()
        loss_sum = acc_sum = 0.0
        acc_batches = 0
        for i, batch in enumerate(batches):
            last = i == len(batches) - 1
            with contextlib.nullcontext() if last else self.ddp.no_sync():
                with torch.autocast(self.device.type, torch.bfloat16, enabled=self.targs.bf16):
                    out = self.ddp(**_to_device(batch, self.device))
                (out.loss / len(batches)).backward()
            loss_sum += out.loss.item()
            # A micro-batch without a round to train has no accuracy.
            if batch["rejmem_anchor_valid"].any():
                acc_sum += out.train_acc[0][0]
                acc_batches += 1
        grad_norm = 0.0
        if self.targs.max_grad_norm:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.ddp.parameters(), self.targs.max_grad_norm)
            )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return {
            "loss_sum": loss_sum,
            "acc_sum": acc_sum,
            "acc_batches": acc_batches,
            "micro_batches": len(batches),
            "grad_norm_sum": grad_norm,
        }

    def _log(self, step, counts, times):
        if step % self.targs.logging_steps:
            return
        totals = torch.tensor(list(counts.values()), dtype=torch.float64, device=self.device)
        dist.all_reduce(totals)
        worst = torch.tensor(list(times.values()), dtype=torch.float64, device=self.device)
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
        if self.rank != 0:
            return
        c = dict(zip(counts, totals.tolist()))
        entry = {
            "step": step,
            "loss": c["loss_sum"] / c["micro_batches"],
            "acc": c["acc_sum"] / max(c["acc_batches"], 1),
            "grad_norm": c["grad_norm_sum"] / self.world,
            "lr": self.scheduler.get_last_lr()[0],
            "rollout_al": c["accepted_plus_one"] / max(c["verified_rounds"], 1),
            **{k: int(c[k]) for k in _COUNTS},
            **{f"t_{k}": round(v, 3) for k, v in zip(times, worst.tolist())},
        }
        print(json.dumps(entry), flush=True)
        with (self.out_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def _save(self, step):
        """Write ``checkpoint-<step>``: the export the servers relaunch from, optimizer, rank states."""
        final = self.out_dir / f"checkpoint-{step}"
        tmp = final.with_name(final.name + ".tmp")

        def start():
            if self.rank == 0:
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir(parents=True)

        def write():
            rank_state = {"prompts": self.source.state_dict(), "rng": torch.get_rng_state()}
            if self.device.type == "cuda":
                rank_state["cuda_rng"] = torch.cuda.get_rng_state(self.device)
            torch.save(rank_state, tmp / f"rank{self.rank}.pt")
            if self.rank == 0:
                self.model.get_exporter().export(tmp / "export")
                state = {
                    "step": step,
                    "world_size": self.world,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                }
                torch.save(state, tmp / "trainer_state.pt")

        def publish():
            if self.rank == 0:
                # Only complete checkpoints get their final name: the launch script resumes the latest.
                tmp.rename(final)
                limit = self.targs.save_total_limit
                done = sorted(
                    self.out_dir.glob("checkpoint-*[0-9]"), key=lambda p: int(p.name[11:])
                )
                for old in done[: -limit if limit else 0]:
                    # The servers run this checkpoint's export: vLLM's DFlash rereads it on every push.
                    if old.resolve() != self.launch_ckpt:
                        shutil.rmtree(old)

        # The phases end together, so a failure on one rank (e.g. a full disk) stops every rank.
        for phase in (start, write, publish):
            _on_all_ranks(self.device, phase)

    def _load(self, path):
        state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=True)
        if state["world_size"] != self.world:
            raise ValueError(f"{path} was written by {state['world_size']} ranks, not {self.world}")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        rank_state = torch.load(path / f"rank{self.rank}.pt", weights_only=True)
        self.source.load_state_dict(rank_state["prompts"])
        torch.set_rng_state(rank_state["rng"])
        if "cuda_rng" in rank_state:
            torch.cuda.set_rng_state(rank_state["cuda_rng"], self.device)
        return state["step"]


def _check_config(recipe, cfg, targs):
    if targs.max_steps <= 0:
        raise ValueError(
            "set training.max_steps: every step generates new rollouts, epochs do not apply"
        )
    if cfg.prompts_per_step <= 0 or cfg.prompts_per_step % targs.per_device_train_batch_size:
        raise ValueError("onpolicy.prompts_per_step must be a positive multiple of the micro-batch")
    if not cfg.server_urls or not cfg.capture_dir:
        raise ValueError("set onpolicy.server_urls and onpolicy.capture_dir (launch/node.sh does)")
    if recipe.dflash.dflash_mask_token_id is None:
        raise ValueError("set dflash.dflash_mask_token_id")
    # A step trains one rollout of prompts_per_step prompts, with plain DDP in bf16 or fp32.
    unsupported = {
        "gradient_accumulation_steps": targs.gradient_accumulation_steps != 1,
        "fp16": targs.fp16,
        "gradient_checkpointing": targs.gradient_checkpointing,
        "cp_size": targs.cp_size != 1,
        "dp_shard_size": targs.dp_shard_size not in (None, 1),
        "estimate_ar": targs.estimate_ar,
    }
    if bad := [name for name, value in unsupported.items() if value]:
        raise ValueError(f"the on-policy loop does not support these training settings: {bad}")
    if min(targs.save_steps, targs.logging_steps) < 1:
        raise ValueError("training.save_steps and logging_steps must be numbers of steps")
    if cfg.request_timeout + cfg.record_timeout >= targs.ddp_timeout:
        raise ValueError(
            "onpolicy.request_timeout + record_timeout must be below training.ddp_timeout, so that "
            "a stuck server fails its own rank before the other ranks time out"
        )


def _build_optimizer(model, targs):
    """AdamW as ``transformers.Trainer`` resolves ``training.optim``, with fp32 masters if configured."""
    cls, kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(targs, model)
    if not issubclass(cls, torch.optim.AdamW):
        raise ValueError(f"training.optim must be an AdamW variant, got {targs.optim}")
    if targs.weight_decay:
        raise ValueError("training.weight_decay is not supported by the on-policy loop")
    params = [p for p in model.parameters() if p.requires_grad]
    if model.dflash_fp32_master_weights:
        # The fused kernel writes the update into the bf16 parameter, not the master.
        kwargs.pop("fused", None)
        return MasterWeightAdamW(params, weight_decay=0.0, foreach=True, **kwargs)
    return cls(params, weight_decay=0.0, **kwargs)


def _check_on_policy(records, version):
    for rec in records:
        start, end = rec.meta["weight_version_start"], rec.meta["weight_version_end"]
        if not start == end == version:
            raise OffPolicyError(
                f"record {rec.meta['tag']} was generated by drafter versions {start}..{end}, but "
                f"the trainer last pushed version {version}"
            )


def _drop_metrics_after(path, step):
    """Keep the metrics of steps up to ``step``: a resumed run logs the later ones again."""
    if path.exists():
        lines = [ln for ln in path.read_text().splitlines() if json.loads(ln)["step"] <= step]
        path.write_text("".join(f"{ln}\n" for ln in lines))


def _on_all_ranks(device, fn):
    """``fn()`` on every rank; if it raised anywhere, raise everywhere, so no rank blocks in a collective."""
    try:
        result, error = fn(), None
    except Exception as e:
        result, error = None, e
    failed = torch.tensor(int(error is not None), device=device)
    dist.all_reduce(failed)
    if error is not None:
        raise error
    if failed.item():
        raise RuntimeError("another rank failed this step; see its log")
    return result


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


if __name__ == "__main__":
    main()
