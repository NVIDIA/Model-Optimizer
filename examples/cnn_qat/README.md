# Quantization-Aware Training (QAT) for CNNs

Quantization-Aware Training (QAT) with NVIDIA ModelOpt injects simulated quantization noise during training to recover accuracy lost by Post-Training Quantization (PTQ). A CNN model quantized via `mtq.quantize()` can be fine-tuned using your existing training loop. During QAT, the quantizer scales are frozen while the model weights adapt.

Learn more in the [ModelOpt QAT guide](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html#quantization-aware-training-qat).

> **_NOTE:_** This example uses a TorchVision ResNet-50 on an ImageNet-style dataset, but you can extend the same steps to any CNN and computer-vision dataset.

## System Requirements

- GPU: ≥1 CUDA-capable NVIDIA GPU
- Memory & Performance: Varies by model, batch size, and image resolution

## QAT Workflow

1. Load and evaluate your full-precision (FP32/FP16) model on the target task.

1. Quantize the FP32/FP16 model via

```python
model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, calibrate_fn)
```

then re-evaluate to establish a quantized baseline.

1. Fine-tune the quantized model with a small learning rate to recover accuracy.

> **_NOTES:_**
>
> - Optimal hyperparameters (learning rate, epochs, etc.) depend on your model and data.
> - If you already have a PTQ-quantized model, you can skip straight to step 3.

Here is an example code structure for performing QAT with a CNN:

```python
from modelopt.torch.quantization import mtq
from modelopt.torch.opt import mto

# ... build model, loaders, optimizer, scheduler ...

def calibrate_fn(m):
    m.eval()
    seen = 0
    for x, _ in calib_loader:
        m(x.to(device))
        seen += x.size(0)
        if seen >= 512:
            break

# 1. PTQ quantization + calibration
model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, calibrate_fn)

# 2. QAT fine-tuning
for epoch in range(1, epochs + 1):
    train(model, train_loader, ...)
    scheduler.step()

# 3. Save final QAT model (weights + quantizer state)
mto.save(model, "cnn_qat_best.pth")

# 4. To reload for inference or further training:
model = build_model()
mto.restore(model, "cnn_qat_best.pth")
model.to(device)
```

See the full script [torchvision_qat.py](./torchvision_qat.py) for all boilerplate (argument parsing, DDP setup, logging, etc.).

> **_NOTE:_** The example above uses [mto.save](https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html#saving-modelopt-models) and [mto.restore](https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html#restoring-modelopt-models) for saving and restoring ModelOpt modified models. These functions handle the model weights as well as the quantizer states. Please see [saving & restoring](https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html) to learn more.

### End-to-end QAT Example

This folder contains an end-to-end runnable QAT pipeline for a ResNet50 model on an ImageNet-style dataset using the [torchvision_qat.py](./torchvision_qat.py) script.

The script performs the following steps:

- Loads a pre-trained ResNet50 model from TorchVision.
- Evaluates its FP32 accuracy on the validation set.
- Performs PTQ (INT8 quantization by default) using a calibration subset of the validation data.
- Evaluates the PTQ model accuracy.
- Performs QAT for a specified number of epochs.
- Evaluates the QAT model accuracy after each epoch and saves the best performing model.

Here is an example command for multi-GPU QAT using `torchrun`:

```sh
torchrun --nproc_per_node <num_gpus> torchvision_qat.py \
    --train-data-path /path/to/your/imagenet/train \
    --val-data-path /path/to/your/imagenet/val \
    --batch-size 64 \
    --num-workers 8 \
    --epochs 5 \
    --lr 1e-4 \
    --print-freq 50 \
    --output-dir ./resnet50_qat_output
```

For single-GPU training, you can run:

```sh
python torchvision_qat.py \
    --train-data-path /path/to/your/imagenet/train \
    --val-data-path /path/to/your/imagenet/val \
    --batch-size 64 \
    --num-workers 8 \
    --epochs 5 \
    --lr 1e-4 \
    --print-freq 50 \
    --output-dir ./resnet50_qat_output
    --gpu 0 # Specify the GPU ID
```

> **_TIP:_** For single-GPU runs, you can also use the `CUDA_VISIBLE_DEVICES` environment variable to control GPU visibility. For instance, `CUDA_VISIBLE_DEVICES=1 python torchvision_qat.py ... --gpu 0` will run the script on physical GPU 1, as PyTorch will see it as `cuda:0`.

Customize flags— `--epochs`, `--lr`, `--batch-size`, etc. to fit your hardware and data. Also you may use other quantization formats from **ModelOpt**. Please see more details on the supported quantization formats and how to use them as shown below:

```python
import modelopt.torch.quantization as mtq

# Learn about quantization formats and configs
help(mtq.config)
```

You can then modify the `quant_cfg` in `torchvision_qat.py` accordingly.

> **_NOTES:_**
>
> - QAT can sometimes require more memory than full-precision fine-tuning due to the storage of quantization parameters and potentially different optimizer states.
> - Like any other model training, the QAT model accuracy can be further improved by optimizing the training hyper-parameters such as learning rate, training duration, weight decay, and choice of optimizer and scheduler.

## Example Results

| Model Stage | Accuracy (Top-1) |
|-----------------|------------------|
| FP32 ResNet50 | ~76.1% |
| PTQ INT8 | ~75.5% |
| QAT INT8 | ~75.9% |

Your actual results will vary based on the dataset, specific hyperparameters, and training duration. Typically, you should observe:

- PTQ accuracy may be slightly lower than FP32 accuracy.
- QAT should help recover some or all of the accuracy lost during PTQ, and potentially even exceed the FP32 baseline in some cases, or get very close to it.

## Deployment with TensorRT

The final model after QAT, saved using `mto.save()`, contains both the model weights and the quantization metadata. This model can be deployed to TensorRT for inference after ONNX export. The process is generally similar to [deploying a ONNX PTQ](../onnx_ptq/README.md#evaluate-the-quantized-onnx-model) model from ModelOpt.
