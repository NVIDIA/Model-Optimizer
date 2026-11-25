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

"""Fine-tune the Qwen3.6 QAD checkpoint with KTO.

Reuse NeMo RL's examples/run_dpo.py setup and training loop with a custom KTO loss
and a response processor that allows correct and looping responses to have different prompts.

With kto.yaml, training keeps the weights in BF16 and simulates NVFP4
quantization during each forward pass, so KTO learns with quantization error
present. PRETRAINED_CHECKPOINT must contain the model weights and saved
ModelOpt quantization state.
"""

import argparse
from pathlib import Path

import nemo_rl.data.utils as data_utils
from nemo_rl.algorithms.dpo import MasterConfig, dpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import init_ray
from omegaconf import OmegaConf
from preference_processor import preference_processor
from qwen36_checkpoint import register_qwen36_worker, validate_checkpoint
from unpaired_objective import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dpo-config",
        type=Path,
        required=True,
        help="Path to NeMo RL's examples/configs/dpo.yaml",
    )
    parser.add_argument(
        "--kto-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "kto.yaml",
    )
    args = parser.parse_args()

    # KTO: load the objective settings alongside the model and runtime configuration.
    cfg = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(args.dpo_config), OmegaConf.load(args.kto_config)),
        resolve=True,
        throw_on_missing=True,
    )
    # NeMo RL passes this checkpoint config to its training workers.
    checkpoint = cfg["checkpointing"]["pretrained_checkpoint"]
    # Check for saved ModelOpt state and give the workers an absolute path.
    checkpoint["path"] = str(validate_checkpoint(Path(checkpoint["path"])))
    # - Megatron Bridge QAD saves only Qwen's language weights. The default NeMo RL
    #   worker builds the full VLM, so the custom worker builds only the language model.
    # - The default worker looks for ModelOpt state under iter_0000000. The custom
    #   worker uses the supplied QAD iteration directory, such as iter_0000500.
    # - The VLM wrapper normally computes rotary positions. The custom worker calls
    #   the language model directly, so it repeats each text position across three axes.
    register_qwen36_worker()

    config = MasterConfig(**cfg)
    tokenizer = get_tokenizer(config.policy["tokenizer"])

    # KTO: replace DPO loss with KTO loss and add the batch KL estimate z.
    install(tokenizer.pad_token_id, max_length=config.policy["max_total_sequence_length"])
    # KTO: allow correct and looping responses to have different prompts.
    # Keep the response-only likelihood masks already used by the DPO path.
    data_utils.preference_preprocessor = preference_processor

    # Start or connect to Ray, which NeMo RL uses to manage training workers.
    init_ray()
    # Build training and validation datasets using the registered response processor.
    dataset, val_dataset = data_utils.setup_preference_data(tokenizer, config.data)
    # NeMo RL's dpo.setup() now constructs the KTO loss installed above.
    (policy, _cluster, train_dl, val_dl, loss_fn, logger, checkpointer, save_state, master) = setup(
        config, tokenizer, dataset, val_dataset
    )
    # Reuse dpo_train unchanged: it consumes the KTO loss and KL-augmented batches above.
    dpo_train(
        policy, train_dl, val_dl, tokenizer, loss_fn, master, logger, checkpointer, save_state
    )


if __name__ == "__main__":
    main()
