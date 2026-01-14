# Chaining Multiple Optimizations Techniques

This directory demonstrates how to chain multiple optimization techniques like Pruning, Distillation, and Quantization together to
achieve the best performance on a given model.

## HuggingFace BERT Pruning + Distillation + Quantization

This example shows how to compress a [Hugging Face Bert large model for Question Answering](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad)
using the combination of `modelopt.torch.prune`, `modelopt.torch.distill` and `modelopt.torch.quantize`. More specifically, we will:

1. Prune the Bert large model to 50% FLOPs with GradNAS algorithm and fine-tune with distillation
1. Quantize the fine-tuned model to INT8 precision with Post-Training Quantization (PTQ) and Quantize Aware Training (QAT) with distillation
1. Export the quantized model to ONNX format for deployment with TensorRT

The main python file is [bert_prune_distill_quantize.py](./bert_prune_distill_quantize.py) and scripts for running it
for all 3 steps are available in the [scripts](./scripts/) directory.

NOTE: This example has been tested on 8 x 24GB A5000 GPUs with PyTorch 2.4 and CUDA 12.4. It takes
about 2 hours to complete all the stages of the optimization. Most of the time is spent
on fine-tuning and QAT.

### Pre-requisites

Install Model Optimizer with optional torch and huggingface dependencies:

```bash
pip install "nvidia-modelopt[hf]"
pip install -r requirements.txt
```

### Running the example

To run the example, execute the following scripts in order:

1. First we prune the Bert large model to 50% FLOPs with GradNAS algorithm. Then, we fine-tune the pruned
   model with distillation from unpruned teacher model to recover 99+% of the initial F1 score (93.15).
   We recommend using multiple GPUs for fine-tuning. Note that we use more epochs
   for fine-tuning, which is different from the 2 epochs used originally in fine-tuning Bert without distillation since
   distillation requires more epochs to converge but achieves much better results.

   ```bash
   bash scripts/1_prune.sh
   ```

1. Quantize the fine-tuned model to INT8 precision and run calibration (PTQ).
   Note that PTQ will result in a slight drop in F1 score but we will be able to recover the F1 score with QAT.
   We run QAT with distillation as well from unpruned teacher model.

   ```bash
   bash scripts/2_int8_quantize.sh
   ```

1. Export the quantized model to ONNX format for deployment with TensorRT.

   ```bash
   bash scripts/3_onnx_export.sh
   ```
