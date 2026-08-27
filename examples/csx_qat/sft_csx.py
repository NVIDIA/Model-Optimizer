#!/usr/bin/env python
"""QAT with SFT to recover accuracy lost to CSX PTQ.

Runs the NVIDIA Model-Optimizer gpt-oss SFT recipe unchanged, except that the
fake-quantization in the forward is QLab's CSX 4-/3-bit block-codebook scheme
instead of MXFP4/NVFP4. Weights train through a straight-through estimator while
the codebook and qscale stay frozen at their PTQ values, so the forward sees
exactly the weights the deployed CSX kernel will produce.

    accelerate launch --config_file <zero3.yaml|configs/fsdp2.yaml> sft_csx.py \
        --config <ModelOpt>/examples/gpt-oss/configs/sft_full.yaml \
        --model_name_or_path openai/gpt-oss-20b \
        --csx_artifacts artifacts/csx_4bit.pt \
        --output_dir <out>

Modes for --csx_artifacts:
    <path>   CSX QAT (the point of this script)
    mxfp4    keep ModelOpt's MXFP4 quantizers -- the upstream baseline, useful as
             a control since it isolates CSX from everything else in the pipeline
    none     no quantization at all: high-precision SFT, i.e. the accuracy
             CEILING that QAT is trying to reach

Quantization happens through QATSFTTrainer's lazy path, i.e. after the
parallelism engine has partitioned the parameters. See qlab.qat.csx_modelopt for
why that ordering is load-bearing.
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser

import modelopt.torch.opt as mto
from modelopt.torch.quantization.plugins import QATSFTTrainer, QuantizationArguments

from qlab.qat import CARRIER_RECIPE, install_csx_quantizers

mto.enable_huggingface_checkpointing()


class CSXQATSFTTrainer(QATSFTTrainer):
    """QATSFTTrainer that swaps ModelOpt's weight quantizers for CSX ones.

    The swap happens inside ``_quantize_model``, which the parent calls lazily
    from ``training_step``/``prediction_step`` -- after ZeRO-3/FSDP has
    partitioned the parameters. Quantizing before the engine is built turns the
    expert weights into ModelOpt dynamic attributes the sharding cannot
    partition, which on gpt-oss-20B pinned every rank at 74 GiB and OOMed.
    """

    def __init__(self, *args, csx_blob=None, csx_trainable_scale=False, **kwargs):
        self._csx_blob = csx_blob
        self._csx_trainable_scale = csx_trainable_scale
        self._csx_installed = False
        super().__init__(*args, **kwargs)

    def _quantize_model(self):
        super()._quantize_model()
        if self._csx_blob is None or self._csx_installed:
            return
        blob = self._csx_blob
        n = install_csx_quantizers(
            self.model, blob["artifacts"], blob["geometry"],
            trainable_scale=self._csx_trainable_scale, num_bits=blob["num_bits"],
        )
        self._csx_installed = True
        if int(os.environ.get("RANK", "0")) == 0:
            sq = blob.get("sqnr_db", {})
            gu = [v for k, v in sq.items() if "gate_up" in k]
            dp = [v for k, v in sq.items() if "down_proj" in k]
            print(f"[csx-qat] {n} CSX {blob['num_bits']}-bit quantizers installed "
                  f"(scale {'trainable' if self._csx_trainable_scale else 'frozen'}); "
                  f"PTQ fit SQNR gate_up={sum(gu)/max(len(gu),1):.2f} dB "
                  f"down_proj={sum(dp)/max(len(dp),1):.2f} dB", flush=True)


def _load_dataset(script_args, training_args):
    from datasets import load_dataset, load_from_disk
    try:
        ds = load_from_disk(script_args.dataset_name)
    except Exception:
        ds = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    if training_args.eval_strategy != "no" and script_args.dataset_test_split not in ds:
        ds = ds[script_args.dataset_train_split].train_test_split(test_size=0.1, seed=42)
    return ds


def main(script_args, training_args, model_args, csx_artifacts, trainable_scale):
    kw = {
        "revision": model_args.model_revision,
        "trust_remote_code": getattr(model_args, "trust_remote_code", False),
        "attn_implementation": model_args.attn_implementation,
        "dtype": getattr(model_args, "dtype", "bfloat16"),
        "use_cache": not training_args.gradient_checkpointing,
    }
    from transformers import AutoConfig
    qc = getattr(AutoConfig.from_pretrained(model_args.model_name_or_path),
                 "quantization_config", None)
    if qc and qc.get("quant_method") == "mxfp4":
        kw["quantization_config"] = Mxfp4Config(dequantize=True)
    if not (os.environ.get("WORLD_SIZE") or os.environ.get("RANK")):
        kw["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **kw)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    dataset = _load_dataset(script_args, training_args)
    rank0 = int(os.environ.get("RANK", "0")) == 0

    common = dict(
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
    )

    if csx_artifacts == "none":
        if rank0:
            print("[csx-qat] no quantization: high-precision SFT (the ceiling)",
                  flush=True)
        trainer = SFTTrainer(model=model, **common)
    else:
        quant_args = QuantizationArguments(recipe=CARRIER_RECIPE)
        if csx_artifacts == "mxfp4":
            if rank0:
                print("[csx-qat] MXFP4 QAT via QATSFTTrainer (no CSX)", flush=True)
            trainer = QATSFTTrainer(model=model, quant_args=quant_args, **common)
        else:
            blob = torch.load(csx_artifacts, map_location="cpu", weights_only=False)
            trainer = CSXQATSFTTrainer(
                model=model, quant_args=quant_args, csx_blob=blob,
                csx_trainable_scale=trainable_scale, **common)

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if rank0:
        print("[csx-qat] NOTE: QAT weights are co-adapted to the quantizer and are "
              "WORSE than PTQ when evaluated unquantized. Export with "
              "examples/qat_sft/export_qlab.py before use.", flush=True)


if __name__ == "__main__":
    argv, csx_artifacts, trainable_scale = sys.argv[1:], None, False
    rest, i = [], 0
    while i < len(argv):
        if argv[i] == "--csx_artifacts":
            csx_artifacts = argv[i + 1]; i += 2
        elif argv[i] == "--csx_trainable_scale":
            trainable_scale = argv[i + 1].lower() in ("1", "true", "yes"); i += 2
        else:
            rest.append(argv[i]); i += 1
    sys.argv = [sys.argv[0]] + rest
    if not csx_artifacts:
        raise SystemExit("--csx_artifacts is required (a path, 'mxfp4', or 'none')")

    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    sa, ta, ma, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(sa, ta, ma, csx_artifacts, trainable_scale)
