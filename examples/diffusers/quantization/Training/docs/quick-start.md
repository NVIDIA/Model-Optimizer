# Quick Start Guide

Get up and running with LTX-Video training in just a few steps!

## ⚡ Installation

First, install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.
Then clone the repository and install the dependencies:

```bash
git clone https://github.com/Lightricks/LTX-Video-Trainer
cd LTX-Video-Trainer
uv sync
source .venv/bin/activate
```

## 💻 Command Line Training

For step-by-step command-line training:

1. **Prepare your dataset** - See [Dataset Preparation](dataset-preparation.md) for splitting videos, generating captions, and preprocessing
2. **Configure training** - Check [Configuration Reference](configuration-reference.md) for setting up your training parameters
3. **Start training** - Follow the [Training Guide](training-guide.md) to run your training job

## 🎨 Web Interface (Gradio)

For users who prefer a graphical interface, you can use our Gradio web UI.
Note that while more user-friendly, it's less flexible configuration-wise than the CLI approach:

```bash
# Install dependencies if you haven't already
uv sync
source .venv/bin/activate

cd scripts
# Launch the Gradio interface
python app_gradio.py
```

This will open a web interface at `http://localhost:7860` that provides all training functionality in a user-friendly way.
The interface provides the same functionality as the command-line tools but in a more intuitive way.

## Next Steps

Once you've completed your first training run, you can:

- Learn more about [Dataset Preparation](dataset-preparation.md) for advanced preprocessing
- Explore different [Training Modes](training-modes.md) (LoRA, Full fine-tuning, IC-LoRA)
- Dive deeper into [Training Configuration](configuration-reference.md)

## Need Help?

If you run into issues at any step, see the [Troubleshooting Guide](troubleshooting.md) for solutions to common problems.
