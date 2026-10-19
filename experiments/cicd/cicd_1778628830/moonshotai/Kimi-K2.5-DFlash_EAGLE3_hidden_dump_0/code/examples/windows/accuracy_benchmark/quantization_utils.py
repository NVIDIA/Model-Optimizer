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

from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import modelopt.torch.quantization as mtq
    import modelopt.torch.utils.dataset_utils as dataset_utils
    from modelopt.torch.quantization.plugins import register_hf_attentions_on_the_fly
except ImportError:
    dataset_utils = None
    mtq = None
    register_hf_attentions_on_the_fly = None
MAX_SEQ_LEN = 2048
MAX_OUTPUT_LEN = 512


def get_tokenizer(ckpt_path, max_seq_len=MAX_SEQ_LEN, trust_remote_code=False):
    """Returns the tokenizer from the model ckpt_path."""
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_len,
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )

    # can't set attribute 'pad_token' for "<unk>"
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _quantize_model_with_dataset(lm, quant_cfg: str, calib_dataset):
    atq_cfg = getattr(mtq, quant_cfg)

    net = lm.gpt2 if hasattr(lm, "gpt2") else lm.model

    def calibrate_loop(model):
        print("Calibrating model...")
        for data in tqdm(calib_dataset):
            model(data)
        print("Calibration complete.")

    def is_dynamic(atq_cfg):
        def _not_dynamic(cfg):
            return cfg.get("enable", True) and cfg.get("type", "") != "dynamic"

        for cfg in atq_cfg.get("quant_cfg", {}).values():
            # quantization like W4A8 has a list of weight quantizers
            if isinstance(cfg, list):
                for config in cfg:
                    if _not_dynamic(config):
                        return False
            elif _not_dynamic(cfg):
                return False

        return True

    use_calibration = not is_dynamic(atq_cfg)
    if not use_calibration:
        print("Dynamic quantization. Calibration skipped.")

    quantize_bmm_attention = False
    for key in atq_cfg["quant_cfg"]:
        if "bmm_quantizer" in key:
            quantize_bmm_attention = True
    if quantize_bmm_attention:
        register_hf_attentions_on_the_fly(net)

    net = mtq.quantize(net, atq_cfg, calibrate_loop if use_calibration else None)
    # Fold weights for faster evaluation.
    mtq.fold_weight(net)

    if hasattr(lm, "gpt2"):
        lm.gpt2 = net
    else:
        lm.model = net


def quantize_model(
    model,
    quant_cfg: str,
    tokenizer,
    batch_size,
    calib_size,
    data="cnn_dailymail",
    test_generated=True,
):
    """Quantizes the model with the provided calibration dataset.

    Args:
        model: the model to be quantized.
        quant_cfg: the quantization algorithm config name.
        tokenizer: the tokenizer.
        batch_size: the calibration batch size for each calibration inference run.
        calib_size: the total calibration dataset size.
        data: the name of the calibration dataset.
        test_generated:  If ``True``, test the generated text before and after quantization.
    """
    if "AWQ" in quant_cfg:
        print(
            "\n####\nAWQ calibration could take longer than other calibration methods. "
            "Consider reducing calib_size to reduce calibration time.\n####\n"
        )

    device = model.device
    if hasattr(model, "model"):
        device = model.model.device

    if batch_size == 0:
        net = model.gpt2 if hasattr(model, "gpt2") else model.model

        # We let the system to determine the max data batch for each forward.
        batch_size = dataset_utils.get_max_batch_size(net)
        print(f"Update calib batch {batch_size}")

    calib_dataloader = dataset_utils.get_dataset_dataloader(
        dataset_name=data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=calib_size,
        device=device,
    )

    if test_generated:
        input_str = tokenizer.decode(next(iter(calib_dataloader))[0])
        generated_str_before_ptq = model.run(input_str)

    _quantize_model_with_dataset(model, quant_cfg, calib_dataloader)

    if test_generated:
        generated_str_after_ptq = model.run(input_str)

        print("--------")
        print(f"example test input: {input_str}")
        print("--------")
        print(f"example outputs before ptq: {generated_str_before_ptq}")
        print("--------")
        print(f"example outputs after ptq: {generated_str_after_ptq}")
