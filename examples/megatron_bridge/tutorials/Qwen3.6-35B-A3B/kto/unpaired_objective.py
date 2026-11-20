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

"""KTO objective used with NeMo RL's Megatron preference-training loop.

KTO: https://arxiv.org/abs/2402.01306
The batch KL estimate z uses mismatched prompt-response pairs, as in TRL's KTOTrainer:
https://github.com/huggingface/trl/blob/main/trl/trainer/kto_trainer.py
No generation during training; z is treated as constant when computing gradients.
"""

import nemo_rl.algorithms.dpo as dpo
import torch
from nemo_rl.algorithms.loss import DPOLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class KTOLossFn(DPOLossFn):
    r"""KTO loss for independently labeled responses using NeMo RL's DPO interface.

    The implemented objective is:

    .. math::

        r_i = \log \frac{\pi_\theta(y_i \mid x_i)}{\pi_{\mathrm{ref}}(y_i \mid x_i)}

        \mathcal{L} = \frac{1}{N}\sum_i w_i
        \left[1 - \sigma\!\left(\beta s_i
        \left[r_i - z\right]\right)\right].

    Log probabilities are summed over response tokens only. Batches alternate
    correct responses (desirable, s_i = +1) and looping responses (undesirable, s_i = -1).
    Here w_i is the sample mask, N is global_valid_seqs clamped to at least one,
    and beta is NeMo RL's reference_policy_kl_penalty.
    pi_theta is the model being fine-tuned; pi_ref is the frozen QAD baseline.
    The batch KL estimate z supplies a shared reference level for the model's
    overall likelihood shift from the frozen QAD baseline. It is treated as
    constant when computing gradients. For an unpadded batch of B responses,
    w_i = 1 and N = B.

    See the KTO paper: https://arxiv.org/abs/2402.01306.
    """

    def _dpo_loss(self, next_token_logprobs, data, global_valid_seqs):
        # Sum policy-versus-reference log probabilities over response tokens only to obtain r_i.
        mask = data["token_mask"][:, 1:]
        reference = data["reference_policy_logprobs"][:, :-1]
        ratios = ((next_token_logprobs - reference) * mask).sum(-1)

        # Hold the shared batch KL estimate z constant during this update.
        z = data["kto_kl_reference"].detach()
        beta = self.reference_policy_kl_penalty

        # NeMo RL interleaves [correct, looping, correct, looping, ...].
        # Even positions get s_i = +1; odd positions (1::2) get s_i = -1.
        signs = torch.ones_like(ratios)
        signs[1::2] = -1
        # Minimizing this loss raises r_i for correct responses and lowers it for looping ones.
        losses = 1 - torch.sigmoid(beta * signs * (ratios - z))

        # Ignore padded samples and normalize by the global number of valid responses.
        weights = data["sample_mask"]
        normalizer = torch.as_tensor(global_valid_seqs, device=ratios.device).clamp(min=1)
        loss = (losses * weights).sum() / normalizer

        # These are NeMo RL training diagnostics; accuracy measures agreement with KTO labels.
        accuracy = ((signs * (ratios - z) > 0).float() * weights).sum() / normalizer
        chosen = (ratios[::2].detach() * weights[::2]).sum() / (normalizer / 2)
        rejected = (ratios[1::2].detach() * weights[1::2]).sum() / (normalizer / 2)
        return (loss, accuracy, chosen, rejected)


def _mismatch_batch(batch, pad_id, max_length=32768):
    """Pair each prompt with another response to compute the batch KL estimate z.

    The mismatched prompt-response pairs are used only to estimate KL.
    KTO still trains on each response with its original prompt and label.
    """
    ids, token_mask = (batch["input_ids"], batch["token_mask"])
    prompts, responses = ([], [])
    for i, length in enumerate(batch["input_lengths"].tolist()):
        positions = torch.nonzero(token_mask[i, :length], as_tuple=False).flatten()
        cut = int(positions[0])
        prompts.append(ids[i, :cut])
        responses.append(ids[i, cut:length])
    sequences, masks = ([], [])
    for i, prompt in enumerate(prompts):
        response = responses[(i - 1) % len(responses)]
        available = max_length - len(prompt)
        # Truncate only the temporary KL examples, preserving their prompts.
        response = response[:available]
        sequences.append(torch.cat([prompt, response]))
        masks.append(torch.cat([torch.zeros_like(prompt), torch.ones_like(response)]))
    lengths = torch.tensor([len(x) for x in sequences], dtype=batch["input_lengths"].dtype)
    output = BatchedDataDict(
        {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_value=pad_id
            ),
            "input_lengths": lengths,
            "token_mask": torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0),
            "sample_mask": batch["sample_mask"].clone(),
        }
    )
    return output


def install(pad_id, *, max_length=32768):
    """Install KTO loss and estimate the shared reference level z.

    The batch KL estimate z measures the model's overall likelihood shift from
    the frozen QAD baseline and is treated as constant when computing gradients.
    Call before nemo_rl.algorithms.dpo.setup(), which creates the loss object.
    kto_train.py already calls install() before that setup function.
    Batches must alternate correct and looping responses.
    """
    if getattr(dpo, "_kto_installed", False):
        return
    original = dpo.add_ref_logprobs_to_data

    def augmented(dataloader, policy, master_config, is_val=False):
        """Attach the batch KL estimate z to each training or validation batch.

        The original iterator adds frozen-reference log probabilities. This
        wrapper also scores mismatched prompt-response pairs to estimate z,
        then yields the original examples with z stored in kto_kl_reference
        for the KTO loss.
        """
        for batch in original(dataloader, policy, master_config, is_val=is_val):
            kl_batch = _mismatch_batch(batch, pad_id, max_length)
            policy.prepare_for_lp_inference()
            pi = policy.get_logprobs(kl_batch)["logprobs"]
            reference = policy.get_reference_policy_logprobs(kl_batch, micro_batch_size=2)[
                "reference_logprobs"
            ]
            delta = ((pi - reference) * kl_batch["token_mask"]).sum(-1)
            weight = kl_batch["sample_mask"]
            # Treat the batch's shared level z as constant when computing gradients.
            z = ((delta * weight).sum() / weight.sum().clamp(min=1)).detach().clamp(min=0)
            policy.prepare_for_training()
            batch["kto_kl_reference"] = torch.full_like(
                batch["sample_mask"], float(z), dtype=torch.float32
            )
            yield batch

    # NeMo RL's dpo.setup() builds this loss; dpo_train() uses the wrapped iterator.
    dpo.DPOLossFn = KTOLossFn
    # Replace DPO's batch iterator so training and validation receive z for the KTO loss.
    dpo.add_ref_logprobs_to_data = augmented
    dpo._kto_installed = True
