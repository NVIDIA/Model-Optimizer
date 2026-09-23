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

"""AdamW with fp32 master weights, for models trained in a lower precision.

The DFlash draft trains alongside a frozen bf16 target and is cast to match it, so AdamW
allocates its moments with ``zeros_like(p)`` in bf16 -- and bf16 cannot represent the
updates Adam's second moment accumulates. At ``beta2=0.999`` a single step changes ``v`` by
at most 0.1%, while the smallest change bf16 can represent near ``v`` is about 0.4%, so
every decrease rounds back to the same number, ``v`` can only grow, and since the update is
divided by ``sqrt(v)`` the effective step size shrinks on its own from step 1, at any
learning rate.

This keeps the model in its training dtype and holds the extra precision in the optimizer
instead: an fp32 master copy of each low-precision parameter, fp32 moments, the AdamW
update applied to the master, and the result copied back down. Nothing has to reconcile
dtypes at forward time, because the model never holds a dtype the rest of it does not.
"""

import torch
from torch.optim.adamw import adamw
from transformers import TrainerCallback

__all__ = ["MasterWeightAdamW", "VerifyMasterWeightsCallback"]

# State keys this optimizer adds or owns, all of which must stay fp32 across a resume.
_FP32_STATE_KEYS = ("master", "exp_avg", "exp_avg_sq", "max_exp_avg_sq")


class MasterWeightAdamW(torch.optim.AdamW):
    """AdamW that keeps an fp32 master copy of every non-fp32 parameter.

    A drop-in replacement: it subclasses ``AdamW``, so param groups, weight decay and any
    LR scheduler behave unchanged. Parameters that are already fp32 are updated in place
    with no master copy, so a mixed-dtype model pays memory only where it buys precision.

    The master and both moments live in ``self.state[p]``, which is what ``state_dict()``
    serialises, so HF ``Trainer``'s ``optimizer.pt`` carries them across a resume.
    """

    @torch.no_grad()
    def step(self, closure=None):
        """Run one AdamW step in fp32 and write the result back at the parameter's dtype.

        This is ``AdamW.step`` with one substitution: the tensors handed to the functional
        update are the fp32 masters rather than the parameters. Everything else -- the
        group options, the lazy state init, the update itself -- is torch's.

        Swapping ``p.data`` to the master and calling ``super().step()`` would be shorter,
        but it is silently wrong under FSDP2: for a ``DTensor`` parameter the assignment
        updates the wrapper's reported dtype while the local shard stays in the model's, so
        ``zeros_like(p)`` allocates the moments in bf16 after all.
        """
        loss = None
        if closure is not None:
            # `step(closure)` is documented to re-evaluate the loss, which means calling
            # `backward()`; the `no_grad` above would make any such closure raise instead.
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("fused"):
                raise ValueError(
                    "fused AdamW writes through to the parameters it is given, which is not "
                    "compatible with fp32 master weights. Use foreach instead."
                )
            if group.get("differentiable"):
                raise ValueError(
                    "differentiable AdamW builds a graph through the update, but this "
                    "optimizer applies the update to a master copy and then copies the result "
                    "into the parameter, so the graph would stop at that copy. Accepting the "
                    "option would leave it silently inert."
                )
            amsgrad = group.get("amsgrad", False)
            capturable = group.get("capturable", False)
            downcast, targets, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                # Repair first, so that everything below reads fp32. A restore that never
                # reaches `load_state_dict` -- DeepSpeed replaces the optimizer outright and
                # never calls it -- leaves the base class's cast to the parameter's dtype in
                # place, and a bf16 master reads as present, so it would survive the creation
                # below and then be updated in bf16. Keyed off `_FP32_STATE_KEYS` rather than
                # a list written out here, so it cannot go back out of step with it.
                for key in _FP32_STATE_KEYS:
                    value = state.get(key)
                    if value is not None and value.dtype != torch.float32:
                        state[key] = value.float()
                needs_master = p.dtype != torch.float32
                # Per key rather than `if not state`: a resume restores the moments without a
                # master -- from plain AdamW, or from the fp32-draft implementation this
                # replaces -- and an all-or-nothing guard falls straight through to
                # `state["master"]` below and raises a bare KeyError.
                if needs_master and "master" not in state:
                    state["master"] = p.detach().float().clone()
                reference = state.get("master", p)
                for key in ("exp_avg", "exp_avg_sq", *(("max_exp_avg_sq",) if amsgrad else ())):
                    if key not in state:
                        state[key] = torch.zeros_like(reference)
                if "step" not in state:
                    # torch keeps this on the parameter's device under `capturable`, and the
                    # multi-tensor update asserts on it; everywhere else a CPU scalar is the
                    # documented fast path.
                    state["step"] = torch.zeros(
                        (), dtype=torch.float32, device=p.device if capturable else "cpu"
                    )
                elif capturable and state["step"].device != p.device:
                    state["step"] = state["step"].to(p.device)
                target = state["master"] if needs_master else p
                if needs_master:
                    downcast.append((p, target))
                targets.append(target)
                grads.append(p.grad if p.grad.dtype == torch.float32 else p.grad.float())
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                if amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                steps.append(state["step"])

            if not targets:
                continue

            beta1, beta2 = group["betas"]
            adamw(
                targets,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group.get("maximize", False),
                foreach=group.get("foreach"),
                capturable=capturable,
            )
            for param, master in downcast:
                param.copy_(master)

        return loss

    def load_state_dict(self, state_dict):
        """Restore, without letting the base class round the fp32 state to the parameters.

        ``Optimizer.load_state_dict`` casts every restored state tensor except ``step`` to
        its parameter's dtype. For a bf16 model that silently rounds the master copy and both
        moments on every resume -- the exact loss this optimizer exists to avoid, and
        invisible, since training continues and the loss keeps falling. The incoming
        ``state_dict`` still holds the saved tensors, so they are put back afterwards, at
        fp32: a checkpoint written by plain ``AdamW`` saved bf16 moments, and restoring those
        verbatim would leave the feature inert for the whole resumed run.
        """
        super().load_state_dict(state_dict)
        # The positional id -> parameter map the base class builds, rather than an integer
        # index into a flattened parameter list: FSDP2 restores through torch's distributed
        # checkpoint, which keys the state by module FQN and deliberately does not convert
        # back, so an integer-only lookup skips every entry and leaves the downcast in place.
        id_map = dict(
            zip(
                [ident for group in state_dict["param_groups"] for ident in group["params"]],
                [p for group in self.param_groups for p in group["params"]],
            )
        )
        for param_id, saved in state_dict["state"].items():
            param = id_map.get(param_id)
            if param is None:
                continue
            for key in _FP32_STATE_KEYS:
                value = saved.get(key)
                if isinstance(value, torch.Tensor):
                    self.state[param][key] = value.detach().to(
                        device=param.device, dtype=torch.float32, copy=True
                    )
        for group in self.param_groups:
            if group.get("fused"):
                # Every hyperparameter comes from the checkpoint, not from this run, so a
                # checkpoint saved under the default `adamw_torch_fused` restores `fused` here
                # and undoes what built this optimizer in the first place -- and `step()`
                # refuses fused before it ever reaches the master.
                group["fused"], group["foreach"] = False, True
            for param in group["params"]:
                state = self.state.get(param)
                if state is not None and param.dtype == torch.float32 and "master" in state:
                    # Saved by a run whose draft was bf16. On an fp32 parameter `step()` updates
                    # the parameter directly, so a master left here never advances, is written
                    # back out, and silently rolls the weights back on the next bf16 resume.
                    state.pop("master")


class VerifyMasterWeightsCallback(TrainerCallback):
    """Fail loudly when master weights were asked for but the optimizer does not keep them.

    Wiring the optimizer is the caller's job, so a training loop that builds its own
    ``AdamW`` gets bf16 moments and no error -- the feature is simply absent, and the only
    symptom is a drafter that trains a little worse. Checked after the first step this
    process takes, which is when the moments exist.
    """

    _checked = False

    def on_train_begin(self, args, state, control, **kwargs):
        """Arm the check for this ``train()`` call."""
        self._checked = False
        return control

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        """Check the optimizer's moment dtypes once, at the end of the first step taken.

        The first step *taken*, not ``global_step == 1``: a resume restores ``global_step``
        from the checkpoint, so an exact step number never matches again and the check is
        dead on precisely the path that can silently lose the fp32 state -- a checkpoint
        restored through anything that does not go through ``MasterWeightAdamW``'s own
        ``load_state_dict`` comes back at the parameter's dtype.
        """
        if self._checked or optimizer is None:
            return control
        inner = getattr(optimizer, "optimizer", optimizer)  # unwrap accelerate
        dtypes = {
            value.dtype
            for param_state in inner.state.values()
            for key, value in param_state.items()
            if key in _FP32_STATE_KEYS and isinstance(value, torch.Tensor)
        }
        if not dtypes:
            # Nothing allocated, so no step landed -- a ``GradScaler`` can skip the first one
            # on a non-finite gradient. Disarming here would pass the run on an empty check.
            return control
        self._checked = True
        if dtypes != {torch.float32}:
            raise RuntimeError(
                f"dflash_fp32_master_weights is set, but the optimizer's Adam moments are "
                f"{dtypes} rather than fp32, so the flag is doing nothing. The training loop "
                f"has to build {MasterWeightAdamW.__name__}; see "
                f"examples/speculative_decoding/eagle_utils.py for how the shipped one does it."
            )
        return control
