# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LTX-Video-Trainer** is an open-source video model training toolkit for fine-tuning the Lightricks LTX-Video (LTXV) model. It supports:
- **LoRA training** - Efficient fine-tuning with adapters
- **Full fine-tuning** - Complete model training
- **IC-LoRA training** - In-context control adapters for video-to-video transformations

Key resources:
- **Main repo**: [LTX-Video](https://github.com/Lightricks/LTX-Video)
- **Documentation**: Comprehensive guides in [docs/](docs/) directory
- **Example configs**: [configs/](configs/) contains working training configurations
- **Community**: [Discord server](https://discord.gg/Mn8BRgUKKy)

## Architecture Overview

### Core Structure

```
src/
├── ltx_core/              # In-house LTX model implementations (LTXV + LTX-2/AV)
│   ├── model_loader.py    # Unified model loading (supports local paths and HTTP URLs)
│   └── models/            # Model definitions (transformers, VAEs, vocoders)
├── ltxv_trainer/          # Training framework and utilities
│   ├── trainer.py         # Main training orchestration with Accelerate
│   ├── config.py          # Pydantic configuration models (CRITICAL: schema changes affect all configs)
│   ├── training_strategies.py  # TrainingStrategy base class + implementations
│   ├── datasets.py        # PrecomputedDataset for latent-based training
│   ├── ltxv_pipeline.py   # Inference pipeline for video generation
│   └── [encoding modules] # T5/Gemma encoders, VAE utilities, etc.
scripts/                   # User-facing CLI tools
```

### Key Architectural Patterns

**Model Loading Architecture:**
- `ltx_core.model_loader.load_model()` is the central loading function
- Supports both local paths and HTTP(S) URLs for checkpoint sources
- Has a built-in validator in `ModelConfig` that ensures model_source is either a valid URL or existing local path
- `load_single_file_checkpoint()` from diffusers handles URL fetching natively
- When passing URLs, no `.exists()` checks are performed (config validator guarantees validity)

**Training Flow:**
1. Configuration loaded via Pydantic models in `config.py`
2. Trainer class (`trainer.py`) orchestrates the training loop
3. Training strategies implement different conditioning modes (reference_video, none, ic_lora)
4. Accelerate handles distributed training and device placement
5. Data flows as precomputed latents through `PrecomputedDataset`

**Configuration System:**
- **Pydantic-based**: All configuration in `src/ltxv_trainer/config.py`
- **Field validators**: Complex validation using `@field_validator` and `@model_validator`
- **Example configs**: In `configs/` directory with clear naming:
  - `*_lora.yaml` - LoRA training configurations
  - `*_full.yaml` - Full fine-tuning configurations
  - `*_low_vram.yaml` - Memory-optimized variants

**⚠️ CRITICAL - Schema Changes Protocol:**
When modifying `src/ltxv_trainer/config.py`:
1. Update ALL config files in `configs/` (not just git-tracked ones)
2. Test that all configs remain valid after changes
3. Update `docs/configuration-reference.md`
4. This is non-negotiable - configurations are user-facing

## Development Commands

### Setup and Installation

```bash
# Install dependencies with development tools
pip install -e ".[dev]"

# Install pre-commit hooks (runs ruff formatting/linting)
pre-commit install
```

### Code Quality

```bash
# Run ruff linting and formatting
ruff check .
ruff format .

# Run pre-commit checks (what will run on commit)
pre-commit run --all-files

# Format specific file
ruff format src/ltxv_trainer/config.py
```

### Running Training

```bash
# Basic training with configuration file
python scripts/train.py --config configs/ltxv_2b_lora.yaml

# With distributed training (multi-GPU)
accelerate launch scripts/train_distributed.py --config configs/ltxv_2b_lora.yaml

# Interactive training with Gradio UI
python scripts/app_gradio.py
```

### Dataset Preparation

```bash
# Preprocess dataset (compute latents, etc.)
python scripts/preprocess_dataset.py --data-dir /path/to/videos --output-dir /path/to/preprocessed

# Split videos by scene
python scripts/split_scenes.py --video /path/to/video.mp4 --output-dir /output

# Caption videos (requires model)
python scripts/caption_videos.py --video-dir /path/to/videos --output-dir /captions
```

### Utility Scripts

```bash
# Decode latents from trained model
python scripts/decode_latents.py --latents-dir /path/to/latents --output-dir /output

# Compute reference conditions for IC-LoRA
python scripts/compute_condition.py --video /path/to/video.mp4 --condition-type depth
```

## Code Standards (from .cursor/rules/)

### Type Hints & Code Quality
- **Always use type hints** for all function arguments and return values
- Use Python 3.9+ type hints: `list[str]` not `List[str]`, `str | Path` not `Union[str, Path]`
- **Avoid importing from `typing` module** - use built-in types instead
  - ✅ `def process(items: list[str]) -> dict[str, int]:`
  - ❌ `from typing import List, Dict, Union` and `def process(items: List[str]) -> Dict[str, int]:`
- Always use `pathlib.Path` for file operations
- Keep lines under 120 characters
- Type hint all function signatures (public and private)

### Class Methods
- **Mark methods as `@staticmethod` if they don't access instance or class state**
  - ✅ `@staticmethod` for utility methods that don't use `self` or `cls`
  - ❌ Don't create instance methods that ignore `self`
- Improves readability and enables potential optimizations

### Path & URL Handling
- Always use `Path` for local file operations
- For checkpoint loading, strings are passed to `load_single_file_checkpoint()` which handles both URLs and local paths
- Config validator ensures model sources are valid before runtime
- Never do existence checks for URLs (config validator handles this)

### AI/ML Specific
- Use `@torch.inference_mode()` for inference (prefer over `@torch.no_grad()`)
- Be explicit about device placement (use `accelerator.device` for distributed compatibility)
- Use gradient checkpointing for memory-intensive models
- Support mixed precision (bfloat16 explicitly, via dtype parameters)
- Ensure torch.compile compatibility when enabled

### Logging
- Use `from ltxv_trainer import logger` (rich logger) for all messages
- Avoid print statements in production code
- Logger provides colored, structured output

### Configuration & Validation
- Inherit from `ConfigBaseModel` for all config classes
- Use Pydantic field validators for per-field validation
- Use model validators for cross-field validation
- All config fields must have clear descriptions
- Configuration is user-facing - clear defaults are essential

### Training Framework
- Implement `TrainingStrategy` abstract base class for new conditioning modes
- Ensure Accelerate compatibility with `accelerator.prepare()` for models/dataloaders
- Support mixed precision modes explicitly
- Move models to CPU when not in use to save GPU memory

## Important Files & Modules

**Configuration (CRITICAL):**
- `src/ltxv_trainer/config.py` - Master config definitions (Pydantic models)
  - Modify only with full config audit
  - All validators here ensure runtime guarantees

**Training Core:**
- `src/ltxv_trainer/trainer.py` - Main training loop with Accelerate
  - Implements distributed training orchestration
  - Handles mixed precision, gradient accumulation, checkpointing
  - ~500+ lines, well-structured training state management

**Model Loading:**
- `src/ltx_core/model_loader.py` - Unified model loading (supports URLs and local paths)
  - Central entry point for all model loading
  - Handles transformer, VAE, text encoder, vocoder loading
  - Model type detection (LTXV vs LTX-2/AV)

**Training Strategies:**
- `src/ltxv_trainer/training_strategies.py` - Abstract base + implementations
  - `TrainingStrategy` base class for extensibility
  - Different conditioning modes: none, reference_video, ic_lora

**Datasets:**
- `src/ltxv_trainer/datasets.py` - Precomputed dataset for latent training
  - Loads pre-computed VAE latents (not raw videos)
  - Handles conditioning, frame selection, augmentation

**Text Encoders:**
- `src/ltxv_trainer/t5_encoder.py` - T5 text encoder wrapper (LTXV models)
- `src/ltxv_trainer/gemma_encoder.py` - Gemma text encoder (LTX-2 models)

**Utilities:**
- `src/ltxv_trainer/video_utils.py` - Video processing utilities
- `src/ltxv_trainer/ltxv_utils.py` - LTX-specific utilities
- `src/ltxv_trainer/hf_hub_utils.py` - HuggingFace Hub integration & model cards

## Common Development Tasks

### Adding a New Configuration Parameter
1. Add field to appropriate config class in `src/ltxv_trainer/config.py`
2. Add validator if needed (e.g., for path existence checks)
3. Update ALL config files in `configs/` with the new parameter
4. Update `docs/configuration-reference.md`
5. Use in trainer via `self._config.path.to.parameter`

### Implementing a New Training Strategy
1. Create new class inheriting `TrainingStrategy` in `src/ltxv_trainer/training_strategies.py`
2. Implement abstract methods: `prepare_latents()`, `forward_pass()`, `get_loss()`
3. Add to `_get_strategy()` factory in trainer
4. Create example config file in `configs/`
5. Document in `docs/training-modes.md`

### Supporting a New Model Type
1. Add model definitions to `src/ltx_core/models/`
2. Update `load_model()` in `src/ltx_core/model_loader.py` to detect and load it
3. Update model type detection in `_detect_model_type()`
4. Test with both local paths and URLs

### Fixing URL/Path Handling
- Model source loading: Use `str()` to convert paths to strings for `load_single_file_checkpoint()`
- Config validation: Check `is_url = path_str.startswith(("http://", "https://"))`
- For local paths only: Call `.exists()` after confirming not a URL
- For URLs: Trust config validator that paths are valid

## Debugging Tips

**Training Issues:**
- Check logs first (rich logger provides context)
- GPU memory: Look for OOM errors, may need `enable_gradient_checkpointing=True`
- Mixed precision issues: Verify dtype casting in model forward passes
- Distributed training: Check `accelerator.state` and device placement

**Model Loading:**
- If URL loading fails: Check network connectivity and URL validity
- Local paths: Config validator ensures they exist before runtime
- Model mismatch: Check `_detect_model_type()` logic for correct detection

**Configuration:**
- Validation errors: Check `src/ltxv_trainer/config.py` for validator implementations
- Unknown fields: Config uses `extra="forbid"` - all fields must be defined
- Type mismatches: Pydantic will show clear error messages with field paths

## Testing & Validation

While there's no formal test suite, validate changes with:
1. **Config validation**: Load all configs with `LtxvTrainerConfig.model_validate_yaml(path)`
2. **Import validation**: `python -c "from ltxv_trainer import ..."`
3. **Manual testing**: Run training scripts with test configs in `configs/`
4. **Pre-commit**: Run `pre-commit run --all-files` before committing

## Project Standards (from .cursor/rules/)

### User Experience
- CLI tools designed for users unfamiliar with internals
- Clear error messages with actionable suggestions
- Sensible defaults that work for most users
- Comprehensive help text and examples

### Documentation
- Write from user's perspective (not developer's)
- Keep docs synchronized with code
- Provide working, copy-pasteable examples
- Cover common use cases and troubleshooting
