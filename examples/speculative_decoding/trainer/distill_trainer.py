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
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from abc import abstractmethod
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.utils import ModelOutput

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import EAGLE3_DEFAULT_CFG
from modelopt.torch.utils import print_rank_0

from .sgl_wrapper import SglangTargetModel

try:
    import wandb
except ImportError:
    wandb = None


mto.enable_huggingface_checkpointing()

# Shape and dtype description of the distillation signal
DistillMetadata = dict[str, tuple[torch.Size, torch.dtype]]


class BaseDistillTrainer:
    """
    Base distill trainer.
    Designed as a placement for HF trainer for several purposes:
    1. Allow separate placement and parallelism for teacher and student.
    2. Allow overlapped teacher and student steps.
    3. Clean, minimal training loop to reduce compatibility issues.
    Args:
        rank: rank of the current process
        args: arguments
        teacher_step: teacher step function.
        student_step: student step function.
    """

    def __init__(self, rank, args, tokenizer, dataloader):
        self.rank = rank
        self.args = args
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.logs = {}

        # Prepare models
        if rank in args.student_ranks:
            self.model = self._prepare_student_model()
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
            # Same scheduler as HF trainer default
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=0, num_training_steps=self.args.total_steps
            )
        else:
            self.model = self._prepare_teacher_model()
        dist.barrier()
        self._print_model_placement(self.model)

    def _print_model_placement(self, module):
        for name, param in module.named_parameters():
            print(f"(Rank {self.rank}) {name:<60} --> {param.device}")

    def _reset_all_mem_stats(self):
        torch.cuda.reset_max_memory_allocated(self.current_rank_device)

    def _print_mem_stats(self):
        max_mem = torch.cuda.max_memory_allocated(self.current_rank_device)
        print(f"GPU {self.current_rank_device}: Max memory allocated: {max_mem / 1024**3:.2f} GB")

    @property
    def current_rank_device(self):
        """Return device of the current rank."""

    @property
    def distill_metadata(self):
        """Return a DistillMetadata that describe the distillation message received by student."""

    @abstractmethod
    def _prepare_teacher_model(self):
        """Return coverted teacher model with correct parallelization."""

    @abstractmethod
    def _prepare_student_model(self):
        """Return coverted student model with correct parallelization."""

    @abstractmethod
    def teacher_step(self, *args, **kwargs) -> list[dict[str, torch.Tensor]]:
        """Run one student step and return distillation messages for each student rank."""

    @abstractmethod
    def student_step(self, *args, **kwargs) -> ModelOutput:
        """Run forward of student step, return a modeloutput object."""

    def save(self, save_path):
        """Save training ckpt on first student rank."""
        if self.rank != self.args.student_ranks[0]:
            return
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        torch.save(self.optimizer.state_dict(), f"{save_path}/optimizer.pt")
        torch.save(self.scheduler.state_dict(), f"{save_path}/scheduler.pt")
        print_rank_0(f"Training ckpt saved to {save_path}")

    def _check_valid_message(self, message: dict[str, torch.Tensor]):
        """Check if message in the format of distill_metadata."""
        if set(message.keys()) != set(self.distill_metadata.keys()):
            raise ValueError(
                f"Message keys: {set(message.keys())} \n"
                f"do not match expected keys {set(self.distill_metadata.keys())}"
            )
        if len(message) != len(self.distill_metadata):
            raise ValueError(
                f"Message length: {len(message)} \n"
                f"does not match expected {len(self.distill_metadata)}"
            )
        for k, v in message.items():
            if v.shape != self.distill_metadata[k][0] or v.dtype != self.distill_metadata[k][1]:
                raise ValueError(
                    f"Invalid message. {k} has shape {v.shape} and dtype {v.dtype}, \n"
                    f"expected {self.distill_metadata[k]}"
                )

    def _init_student_recv_buffer(self):
        """Init buffer for receiving distillation messages from teacher."""
        self.student_recv_buffer = {
            k: torch.empty(v[0], device=self.current_rank_device, dtype=v[1])
            for k, v in self.distill_metadata.items()
        }

    def _recv_from_teacher(self):
        reqs = [
            dist.irecv(buffer, src=self.args.teacher_ranks[0])
            for buffer in self.student_recv_buffer.values()
        ]
        for req in reqs:
            req.wait()

    def _clone_recv_buffer(self):
        """Return a copy of received tensors for student step input."""
        return {k: v.clone().detach() for k, v in self.student_recv_buffer.items()}

    def _send_to_student(self, teacher_outputs):
        if self.rank != self.args.teacher_ranks[0]:
            return
        # TODO: use broadcast
        assert len(teacher_outputs) == len(self.args.student_ranks), (
            f"Number of teacher outputs {len(teacher_outputs)} does not \
            match number of student ranks {len(self.args.student_ranks)}"
        )
        for idx, s in enumerate(self.args.student_ranks):
            self._check_valid_message(teacher_outputs[idx])
            reqs = [dist.isend(buffer, dst=s) for buffer in teacher_outputs[idx].values()]
            for req in reqs:
                req.wait()

    def _get_logging_context(self):
        if wandb is not None and self.rank == self.args.student_ranks[0]:
            return wandb.init(
                config={
                    "epochs": self.args.epoch,
                    "lr": self.args.lr,
                    "batch_size": self.args.batch_size,
                },
            )
        return nullcontext()

    def train(self):
        """Main training entrance of the composed model."""
        self._reset_all_mem_stats()

        if self.rank in self.args.student_ranks:
            with self._get_logging_context() as run:
                self._init_student_recv_buffer()

                # Student training loop
                for epoch in range(self.args.epoch):
                    pbar = (
                        tqdm(self.dataloader)
                        if self.rank == self.args.student_ranks[0]
                        else self.dataloader
                    )
                    for i, batch in enumerate(pbar):
                        global_step = epoch * len(self.dataloader) + i
                        if global_step >= self.args.total_steps:
                            break
                        inputs = {k: v.to(self.model.device) for k, v in batch.items()}

                        # Receive distill messages from teacher
                        self._recv_from_teacher()

                        # Run forward of student step
                        output = self.student_step(inputs, **self._clone_recv_buffer())
                        loss = output.loss

                        # Run backward step
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()

                        # Log and save only on student rank 0
                        if self.rank != self.args.student_ranks[0]:
                            continue

                        train_metrics = {
                            "loss": round(loss.item(), 3),
                            "lr": self.optimizer.param_groups[0]["lr"],
                            # Attach all float metrics
                            **{k: round(v, 3) for k, v in output.items() if isinstance(v, float)},
                        }
                        if "train_acc" in output:
                            train_metrics.update(
                                {
                                    f"train_acc_step{i}": output["train_acc"][i].item()
                                    for i in range(len(output["train_acc"]))
                                }
                            )

                        pbar.set_description(f"Epoch {epoch} Loss {train_metrics['loss']}")

                        # Add train_metrics into self.logs as a dict of lists of metrics since last log
                        for key, value in train_metrics.items():
                            if key not in self.logs:
                                self.logs[key] = []
                            self.logs[key].append(value)

                        if global_step % self.args.log_interval == 0:
                            run.log(
                                {k: sum(v) / len(v) for k, v in self.logs.items()}, step=global_step
                            )
                            self.logs = {}
                        if global_step > 0 and global_step % self.args.save_interval == 0:
                            self.save(f"{self.args.out_path}/epoch_{epoch}_step_{global_step}")

        else:
            # Inference Loop
            for epoch in range(self.args.epoch):
                for i, batch in enumerate(self.dataloader):
                    global_step = epoch * len(self.dataloader) + i
                    if global_step >= self.args.total_steps:
                        break
                    inputs = {k: v.to(self.model.device) for k, v in batch.items()}
                    with torch.inference_mode():
                        self._send_to_student(self.teacher_step(self.model, inputs))

        self._print_mem_stats()
        # Makesure all processes finished before destroy.
        dist.barrier()
        # clean up processess
        dist.destroy_process_group()


class EagleTPTrainer(BaseDistillTrainer):
    """A subclass of BaseDistillTrainer for online eagle training, with base model TP and student DDP."""

    def __init__(self, rank, args, tokenizer, dataloader):
        # Load eagle config
        args.eagle_config = EAGLE3_DEFAULT_CFG["config"]
        if args.eagle_config_path:
            with open(args.eagle_config_path) as f:
                custom_config = json.load(f)
            args.eagle_config["eagle_architecture_config"].update(custom_config)

        super().__init__(rank, args, tokenizer, dataloader)

    @property
    def current_rank_device(self):
        if self.rank in self.args.teacher_ranks:
            return self.args.teacher_devices[self.rank]
        else:
            return self.args.student_devices[self.rank - len(self.args.teacher_ranks)]

    def _prepare_teacher_model(self):
        # Load model with TP among teacher ranks.
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            torch_dtype="auto",
            tp_plan="auto",
            device_mesh=DeviceMesh.from_group(self.args.teacher_pgroup, "cuda"),
        )
        # load eagle config and convert.
        self.args.eagle_config["eagle_architecture_config"].update(
            {
                "hidden_size": model.config.hidden_size,
                "vocab_size": model.config.vocab_size,
                "draft_vocab_size": model.config.vocab_size,
            }
        )
        mtsp.convert(model, [("eagle", self.args.eagle_config)])
        model.eval()
        return model

    def _prepare_student_model(self):
        # Load to CPU first to avoid OOM
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path, torch_dtype="auto", device_map="cpu"
        )
        # Hidden size and vocab size must match base model
        self.args.eagle_config["eagle_architecture_config"].update(
            {
                "hidden_size": model.config.hidden_size,
                "vocab_size": model.config.vocab_size,
                "draft_vocab_size": model.config.vocab_size,
            }
        )
        mtsp.convert(
            model,
            [("eagle", self.args.eagle_config)],
        )

        # TODO:copy needed modules and del the rest
        model.model._modules.pop("layers")
        model.to(self.current_rank_device)

        model.train()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.current_rank_device],
            process_group=self.args.student_pgroup,
            find_unused_parameters=True,
        )
        return model

    @property
    def distill_metadata(self) -> DistillMetadata:
        """Description of the distillation signal received by student."""
        return {
            "base_model_hidden_states": (
                torch.Size(
                    [
                        int(self.args.batch_size / len(self.args.student_ranks)),
                        self.args.training_seq_len,
                        self.args.eagle_config["eagle_architecture_config"]["hidden_size"],
                    ]
                ),
                torch.bfloat16,
            ),
            "aux_hidden_states": (
                torch.Size(
                    [
                        int(self.args.batch_size / len(self.args.student_ranks)),
                        self.args.training_seq_len,
                        self.args.eagle_config["eagle_architecture_config"]["hidden_size"] * 3,
                    ]
                ),
                torch.bfloat16,
            ),
            "base_model_logits": (
                torch.Size(
                    [
                        int(self.args.batch_size / len(self.args.student_ranks)),
                        self.args.training_seq_len,
                        self.args.eagle_config["eagle_architecture_config"]["draft_vocab_size"],
                    ]
                ),
                torch.bfloat16,
            ),
        }

    def teacher_step(self, model, inputs):
        # Collect base model outputs.
        base_model_hidden_states, base_model_logits, _, _ = model._base_model_forward(
            **inputs,
            freeze_base_model=True,
            past_key_values=None,
        )

        # Aux_hidden_states could be on multiple devices. Gather before cat.
        aux_hidden_states = torch.cat(
            [t.to(base_model_logits.device) for t in model.pop_and_gather_aux_hiddens()], dim=-1
        )

        # Chunk the tensors for each student rank.
        base_model_hidden_states = base_model_hidden_states.chunk(len(self.args.student_ranks))
        base_model_logits = base_model_logits.chunk(len(self.args.student_ranks))
        aux_hidden_states = aux_hidden_states.chunk(len(self.args.student_ranks))

        return [
            {
                "base_model_hidden_states": base_model_hidden_states[i],
                "aux_hidden_states": aux_hidden_states[i],
                "base_model_logits": base_model_logits[i],
            }
            for i in range(len(self.args.student_ranks))
        ]

    def student_step(
        self,
        inputs,
        **distill_msgs,
    ) -> ModelOutput:
        self.optimizer.zero_grad()

        # Chunk input_ids and attention_mask for each student rank.
        student_idx = self.rank - len(self.args.teacher_ranks)
        inputs = {k: v.chunk(len(self.args.student_ranks))[student_idx] for k, v in inputs.items()}

        # Second stage forward with provided base model outputs.
        output = self.model(**inputs, base_model_outputs=distill_msgs)

        return output


class EagleSGLTrainer(EagleTPTrainer):
    """A subclass of EagleTPTrainer for online eagle training, with base model SGL and student DDP."""

    def _prepare_teacher_model(self):
        args = self.args
        args.tp_size = len(self.args.teacher_devices)
        args.max_length = self.args.training_seq_len

        teacher_config = AutoConfig.from_pretrained(args.model_path)
        self.args.eagle_config["eagle_architecture_config"].update(
            {
                "hidden_size": teacher_config.hidden_size,
                "vocab_size": teacher_config.vocab_size,
                "draft_vocab_size": teacher_config.vocab_size,
            }
        )

        # patch torch.distributed functions to use only partial ranks
        original_get_world_size = torch.distributed.get_world_size
        original_barrier = torch.distributed.barrier
        torch.distributed.get_world_size = lambda *args, **kwargs: len(self.args.teacher_devices)

        def barrier_patch(*args, **kwargs):
            if not args and not kwargs:
                original_barrier(group=self.args.teacher_pgroup)
            else:
                original_barrier(*args, **kwargs)

        torch.distributed.barrier = barrier_patch

        # load SGL model with patches
        model = SglangTargetModel(
            args=args,
            tp_group=self.args.teacher_pgroup,
            return_full_logits=True,
            gpu_id=self.current_rank_device,
        )

        # retore patches
        torch.distributed.get_world_size = original_get_world_size
        torch.distributed.barrier = original_barrier

        model.set_aux_hidden_states_layers()
        print("rank", self.rank, "SGL base model loaded")
        model.device = self.current_rank_device
        return model

    def teacher_step(self, model, inputs):
        # TODO: handle data loading in preprocess
        sgl_inputs = [
            {
                "input_ids": inputs["input_ids"][i],
                "attention_mask": inputs["attention_mask"][i],
                "loss_mask": inputs["loss_mask"][i],
            }
            for i in range(inputs["input_ids"].shape[0])
        ]

        logits, h, aux_h = model.forward(sgl_inputs)

        # Chunk the tensors for each student rank.
        base_model_logits = logits.chunk(len(self.args.student_ranks))
        base_model_hidden_states = h.chunk(len(self.args.student_ranks))
        aux_hidden_states = aux_h.chunk(len(self.args.student_ranks))

        seq_len = inputs["input_ids"].shape[1]
        vocab_size = logits.shape[-1]
        hid_size = h.shape[-1]

        return [
            {
                "base_model_hidden_states": base_model_hidden_states[i].view(-1, seq_len, hid_size),
                "aux_hidden_states": aux_hidden_states[i].view(-1, seq_len, hid_size * 3),
                "base_model_logits": base_model_logits[i]
                .view(-1, seq_len, vocab_size)
                .to(dtype=torch.bfloat16),
            }
            for i in range(len(self.args.student_ranks))
        ]
