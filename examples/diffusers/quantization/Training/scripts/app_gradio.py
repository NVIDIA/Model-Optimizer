"""Gradio interface for LTX Video Trainer."""

import datetime
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

import gradio as gr
import torch
import yaml
from huggingface_hub import login

from ltxv_trainer.captioning import (
    DEFAULT_VLM_CAPTION_INSTRUCTION,
    CaptionerType,
    create_captioner,
)
from ltxv_trainer.hf_hub_utils import convert_video_to_gif
from ltxv_trainer.model_loader import (
    LtxModelVersion,
)
from ltxv_trainer.trainer import LtxvTrainer, LtxvTrainerConfig
from scripts.preprocess_dataset import preprocess_dataset
from scripts.process_videos import parse_resolution_buckets

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set PyTorch memory allocator configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Clear CUDA cache before training
torch.cuda.empty_cache()

# Define base directories
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
TRAINING_DATA_DIR = BASE_DIR / "training_data"
VALIDATION_SAMPLES_DIR = OUTPUTS_DIR / "validation_samples"

# Create necessary directories
OUTPUTS_DIR.mkdir(exist_ok=True)
TRAINING_DATA_DIR.mkdir(exist_ok=True)
VALIDATION_SAMPLES_DIR.mkdir(exist_ok=True)


@dataclass
class TrainingConfigParams:
    """Parameters for generating training configuration."""

    model_source: str
    learning_rate: float
    steps: int
    lora_rank: int
    batch_size: int
    validation_prompt: str
    video_dims: tuple[int, int, int]  # width, height, num_frames
    validation_interval: int = 100  # Default validation interval
    push_to_hub: bool = False
    hub_model_id: str | None = None


@dataclass
class TrainingState:
    """State for tracking training progress."""

    status: str | None = None
    progress: str | None = None
    validation: str | None = None
    download: str | None = None
    error: str | None = None
    hf_repo: str | None = None
    checkpoint_path: str | None = None

    def reset(self) -> None:
        """Reset state to initial values."""
        self.status = "running"
        self.progress = None
        self.validation = None
        self.download = None
        self.error = None
        self.hf_repo = None
        self.checkpoint_path = None

    def update(self, **kwargs) -> None:
        """Update state with provided values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class TrainingParams:
    videos: list[str]
    validation_prompt: str
    learning_rate: float
    steps: int
    lora_rank: int
    batch_size: int
    model_source: str
    width: int
    height: int
    num_frames: int
    push_to_hub: bool
    hf_model_id: str
    hf_token: str | None = None
    id_token: str | None = None
    validation_interval: int = 100
    captions_json: str | None = None


def process_video(videos: list, caption_text: str) -> str:
    """Process uploaded videos and generate/edit captions.

    Args:
        videos: List of video file paths
        caption_text: Existing caption text (if any)

    Returns:
        Dataset content as JSON string
    """

    if not videos:
        return ""

    # Create captions dictionary and dataset entries
    captions_data = {}

    if caption_text:
        # Use provided caption for all videos
        for video in videos:
            video_name = str(Path(video).name)
            captions_data[video_name] = caption_text
    else:
        # Generate captions for each video
        device = "cuda" if torch.cuda.is_available() else "cpu"
        captioner = create_captioner(
            captioner_type=CaptionerType.QWEN_25_VL,
            use_8bit=True,
            vlm_instruction=DEFAULT_VLM_CAPTION_INSTRUCTION,
            device=device,
        )

        # Process each video
        for video in videos:
            video_name = str(Path(video).name)
            caption = captioner.caption(video)
            captions_data[video_name] = caption

    # Save both captions and dataset files
    data_dir = TRAINING_DATA_DIR
    # Remove the directory if it exists (compatibility with Python <3.12)
    if data_dir.exists():
        shutil.rmtree(data_dir)

    data_dir.mkdir()

    captions_file = data_dir / "dataset.json"

    with open(captions_file, "w") as f:
        json.dump(captions_data, f, indent=2)

    # Convert the dictionary to a list of objects for Gradio JSON/code display
    dataset_display = [{"media_path": k, "caption": v} for k, v in captions_data.items()]

    return json.dumps(dataset_display, indent=2)


def _handle_validation_sample(step: int, video_path: Path) -> str | None:
    """Handle validation sample conversion and storage.

    Args:
        step: Current training step
        video_path: Path to the validation video

    Returns:
        Path to the GIF file if successful, None otherwise
    """
    gif_path = VALIDATION_SAMPLES_DIR / f"sample_step_{step}.gif"
    try:
        convert_video_to_gif(video_path, gif_path)
        logger.info(f"New validation sample converted to GIF at step {step}: {gif_path}")
        return str(gif_path)
    except Exception as e:
        logger.error(f"Failed to convert validation video to GIF: {e}")
        return None


def generate_training_config(params: TrainingConfigParams, training_data_dir: str) -> dict:
    """Generate training configuration from parameters.

    Args:
        params: Training configuration parameters
        training_data_dir: Directory containing training data

    Returns:
        Dictionary containing the complete training configuration
    """
    # Load the template config
    template_path = Path(__file__).parent.parent / "configs" / "ltxv_13b_lora_template.yaml"
    with open(template_path) as f:
        config = yaml.safe_load(f)

    # Update with UI parameters
    config["model"]["model_source"] = params.model_source
    config["lora"]["rank"] = params.lora_rank
    config["lora"]["alpha"] = params.lora_rank  # Usually alpha = rank
    config["optimization"]["learning_rate"] = params.learning_rate
    config["optimization"]["steps"] = params.steps
    config["optimization"]["batch_size"] = params.batch_size
    config["data"]["preprocessed_data_root"] = str(training_data_dir)
    config["output_dir"] = str(OUTPUTS_DIR / f"lora_r{params.lora_rank}")

    # Update HuggingFace Hub settings
    config["hub"] = {
        "push_to_hub": params.push_to_hub,
        "hub_model_id": params.hub_model_id if params.push_to_hub else None,
    }

    width, height, num_frames = params.video_dims
    # Use the user's validation prompt, resolution and interval
    config["validation"] = {
        "prompts": [params.validation_prompt],
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "video_dims": [width, height, num_frames],
        "seed": 42,
        "inference_steps": 30,
        "interval": params.validation_interval,
        "videos_per_prompt": 1,
        "guidance_scale": 3.5,
    }

    # Ensure validation.images is None if not explicitly set
    if "validation" in config and "images" not in config["validation"]:
        config["validation"]["images"] = None

    return config


class GradioUI:
    """Class to manage Gradio UI components and state."""

    def __init__(self):
        self.training_state = TrainingState()

        # Initialize UI components as None
        self.video_upload = None
        self.caption_output = None
        self.dataset_display = None
        self.validation_prompt = None
        self.status_output = None
        self.progress_output = None
        self.validation_sample = None
        self.training_output = None
        self.download_btn = None
        self.hf_repo_link = None

    def reset_interface(self) -> dict:
        """Reset the interface and clean up all training data.

        Returns:
            Dictionary of Gradio component updates
        """
        # Reset training state
        self.training_state.reset()

        # Clean up training data directory
        if TRAINING_DATA_DIR.exists():
            shutil.rmtree(TRAINING_DATA_DIR)
        TRAINING_DATA_DIR.mkdir(exist_ok=True)

        # Clean up outputs directory
        if OUTPUTS_DIR.exists():
            shutil.rmtree(OUTPUTS_DIR)
        OUTPUTS_DIR.mkdir(exist_ok=True)
        VALIDATION_SAMPLES_DIR.mkdir(exist_ok=True)

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Return empty/default values for all components
        return {
            self.video_upload: gr.update(value=None),
            self.caption_output: gr.update(value=""),
            self.dataset_display: gr.update(value=""),
            self.validation_prompt: gr.update(
                value="a professional portrait video of a person with blurry bokeh background",
                info="Include the LoRA ID token (e.g., &lt;lora&gt;) in this prompt if desired.",
            ),
            self.status_output: gr.update(value=""),
            self.progress_output: gr.update(value=""),
            self.validation_sample: gr.update(value=None),
            self.training_output: gr.update(value=""),
            self.download_btn: gr.update(visible=False),
            self.hf_repo_link: gr.update(visible=False, value=""),
        }

    def get_model_path(self) -> str | None:
        """Get the path to the trained model file."""
        if self.training_state.download and Path(self.training_state.download).exists():
            return self.training_state.download
        return None

    def update_progress(self) -> tuple[str, str | None, str, gr.update, str, gr.update]:
        """Update the UI with current training progress."""
        if self.training_state.status is not None:
            status = self.training_state.status
            progress = self.training_state.progress
            validation_path = self.training_state.validation
            training_log = ""

            # Update based on training status
            if status == "running":
                training_log = f"Training in progress...\n{progress}"
                return (
                    progress,
                    validation_path,
                    training_log,
                    gr.update(visible=False),  # Hide download button
                    "",  # Empty HF link
                    gr.update(visible=False),  # Hide HF link
                )
            elif status == "complete":
                training_log = "Training completed successfully!"
                download_path = self.training_state.download

                # Check if model was pushed to HF Hub
                if self.training_state.hf_repo:
                    # Show HF link, hide download button
                    hf_url = self.training_state.hf_repo
                    hf_html = f'<a href="{hf_url}" target="_blank">View model on HuggingFace Hub</a>'
                    return (
                        progress,
                        validation_path,
                        training_log,
                        gr.update(visible=False),  # Hide download button
                        hf_html,
                        gr.update(visible=True),  # Show HF link
                    )
                elif download_path and Path(download_path).exists():
                    # Show download button, hide HF link
                    return (
                        progress,
                        validation_path,
                        training_log,
                        gr.update(value=download_path, visible=True, label=f"Download {Path(download_path).name}"),
                        "",  # Empty HF link
                        gr.update(visible=False),  # Hide HF link
                    )
            elif status == "failed":
                training_log = f"Training failed: {self.training_state.error}"
                return (
                    progress,
                    validation_path,
                    training_log,
                    gr.update(visible=False),  # Hide download button
                    "",  # Empty HF link
                    gr.update(visible=False),  # Hide HF link
                )

            # Default return for other states
            return (
                progress,
                validation_path,
                training_log,
                gr.update(visible=False),  # Hide download button
                "",  # Empty HF link
                gr.update(visible=False),  # Hide HF link
            )

        # No job running
        return (
            "",
            None,
            "",
            gr.update(visible=False),  # Hide download button
            "",  # Empty HF link
            gr.update(visible=False),  # Hide HF link
        )

    def _save_checkpoint(self, saved_path: Path, trainer_config: LtxvTrainerConfig) -> tuple[Path, str | None]:
        """Save and copy the checkpoint to a permanent location.

        Args:
            saved_path: Path where the checkpoint was initially saved
            trainer_config: Training configuration

        Returns:
            Tuple of (permanent checkpoint path, HF repo URL if applicable)
        """
        permanent_checkpoint_dir = OUTPUTS_DIR / "checkpoints"
        permanent_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename for the checkpoint using UTC timezone
        timestamp = datetime.datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        checkpoint_filename = f"lora_checkpoint_{timestamp}.safetensors"
        permanent_checkpoint_path = permanent_checkpoint_dir / checkpoint_filename

        try:
            shutil.copy2(saved_path, permanent_checkpoint_path)
            logger.info(f"Checkpoint copied to permanent location: {permanent_checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to copy checkpoint: {e}")
            permanent_checkpoint_path = saved_path

        # Return HF repo URL if applicable
        hf_repo = (
            f"https://huggingface.co/{trainer_config.hub.hub_model_id}" if trainer_config.hub.hub_model_id else None
        )

        return permanent_checkpoint_path, hf_repo

    def _preprocess_dataset(
        self,
        dataset_file: Path,
        model_source: str,
        width: int,
        height: int,
        num_frames: int,
        id_token: str | None = None,
    ) -> tuple[bool, str | None]:
        """Preprocess the dataset by computing video latents and text embeddings.

        Args:
            dataset_file: Path to the dataset.json file
            model_source: Model source identifier
            width: Video width
            height: Video height
            num_frames: Number of frames
            id_token: Optional token to prepend to captions (for LoRA training)

        Returns:
            Tuple of (success, error_message)
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Clean up existing precomputed data
            precomputed_dir = TRAINING_DATA_DIR / ".precomputed"
            if precomputed_dir.exists():
                shutil.rmtree(precomputed_dir)

            resolution_buckets = f"{width}x{height}x{num_frames}"
            parsed_buckets = parse_resolution_buckets(resolution_buckets)

            # Run preprocessing using the function directly
            preprocess_dataset(
                dataset_file=str(dataset_file),
                caption_column="caption",
                video_column="media_path",
                resolution_buckets=parsed_buckets,
                batch_size=1,
                output_dir=None,
                id_token=id_token,
                vae_tiling=False,
                decode_videos=True,
                model_source=model_source,
                device="cuda" if torch.cuda.is_available() else "cpu",
                load_text_encoder_in_8bit=False,
            )

            # Clean up preprocessor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True, None

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False, f"Error preprocessing dataset: {e!s}"

    def _should_preprocess_data(
        self,
        width: int,
        height: int,
        num_frames: int,
        videos: list[str],
    ) -> bool:
        """Check if data needs to be preprocessed based on resolution changes.

        Args:
            width: Video width
            height: Video height
            num_frames: Number of frames
            videos: List of video file paths

        Returns:
            True if preprocessing is needed, False otherwise
        """
        resolution_file = TRAINING_DATA_DIR / ".resolution_config"
        current_resolution = f"{width}x{height}x{num_frames}"
        needs_to_copy = False
        for video in videos:
            if Path(video).exists():
                needs_to_copy = True
        if needs_to_copy:
            logger.info("Videos provided, will copy them to training directory.")
            return True, needs_to_copy

        # If no previous resolution or dataset, preprocessing is needed
        if not resolution_file.exists() or not (TRAINING_DATA_DIR / "captions.json").exists():
            return True, needs_to_copy

        # Check if resolution has changed
        try:
            with open(resolution_file) as f:
                previous_resolution = f.read().strip()
            return previous_resolution != current_resolution, needs_to_copy
        except Exception:
            return True, needs_to_copy

    def _save_resolution_config(
        self,
        width: int,
        height: int,
        num_frames: int,
    ) -> None:
        """Save current resolution configuration.

        Args:
            width: Video width
            height: Video height
            num_frames: Number of frames
        """
        resolution_file = TRAINING_DATA_DIR / ".resolution_config"
        current_resolution = f"{width}x{height}x{num_frames}"

        with open(resolution_file, "w") as f:
            f.write(current_resolution)

    def _sync_captions_from_ui(
        self, params: TrainingParams, training_captions_file: Path
    ) -> tuple[dict[str, str] | None, str | None]:
        """Sync captions from the UI to captions.json. Returns (captions_data, error_message)."""
        if params.captions_json:
            try:
                dataset = json.loads(params.captions_json)
                # Convert list of dicts to captions_data dict
                captions_data = {item["media_path"]: item["caption"] for item in dataset}
                # Save to captions.json (overwrite every time)
                with open(training_captions_file, "w") as f:
                    json.dump(captions_data, f, indent=2)
                return captions_data, None
            except Exception as e:
                return None, f"Invalid captions JSON: {e!s}"
        else:
            return None, "No captions found in the UI. Please process videos first."

    # ruff: noqa: PLR0912
    def start_training(
        self,
        params: TrainingParams,
    ) -> tuple[str, gr.update]:
        """Start the training process."""
        if params.hf_token:
            login(token=params.hf_token)

        try:
            # Clear any existing CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Set training status
            self.training_state.reset()  # This sets status to "running"

            # Prepare data directory
            data_dir = TRAINING_DATA_DIR
            data_dir.mkdir(exist_ok=True)

            # Check if we need to copy and process data (first training session)
            training_captions_file = data_dir / "captions.json"
            needs_preprocessing, needs_to_copy = self._should_preprocess_data(
                params.width, params.height, params.num_frames, params.videos
            )

            # Sync captions from UI
            captions_data, error_message = self._sync_captions_from_ui(params, training_captions_file)
            if error_message:
                return error_message, gr.update(interactive=True)

            # Copy videos and create dataset entries
            if needs_to_copy:
                dataset = []
                for video in params.videos:
                    video_path = Path(video)
                    video_name = video_path.name
                    if video_name not in captions_data:
                        return f"No caption found for video {video_name}. Please process videos first.", gr.update(
                            interactive=True
                        )
                    # Copy video to training directory
                    target_path = data_dir / video_name
                    try:
                        shutil.copy2(video, target_path)  # Copy with metadata
                        video_path.unlink()  # Remove original after successful copy
                    except Exception as e:
                        return f"Error copying video {video_path.name}: {e!s}", gr.update(interactive=True)
                    # Add dataset entry with relative path
                    dataset.append(
                        {"caption": captions_data[video_name], "media_path": str(target_path.relative_to(data_dir))}
                    )
                # Save dataset.json with updated paths
                with open(training_captions_file, "w") as f:
                    json.dump(dataset, f, indent=2)

            # Preprocess if needed (first time or resolution changed)
            if needs_preprocessing:
                # Clean up existing precomputed data
                precomputed_dir = TRAINING_DATA_DIR / ".precomputed"
                if precomputed_dir.exists():
                    shutil.rmtree(precomputed_dir)

                success, error_msg = self._preprocess_dataset(
                    dataset_file=training_captions_file,
                    model_source=params.model_source,
                    width=params.width,
                    height=params.height,
                    num_frames=params.num_frames,
                    id_token=params.id_token,
                )
                if not success:
                    return error_msg, gr.update(interactive=True)

                # Save current resolution config after successful preprocessing
                self._save_resolution_config(params.width, params.height, params.num_frames)

            # Generate training config
            config_params = TrainingConfigParams(
                model_source=params.model_source,
                learning_rate=params.learning_rate,
                steps=params.steps,
                lora_rank=params.lora_rank,
                batch_size=params.batch_size,
                validation_prompt=params.validation_prompt,
                video_dims=(params.width, params.height, params.num_frames),
                validation_interval=params.validation_interval,
                push_to_hub=params.push_to_hub,
                hub_model_id=params.hf_model_id if params.push_to_hub else None,
            )

            config = generate_training_config(config_params, str(data_dir))
            config_path = OUTPUTS_DIR / "train_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f, indent=4)

            # Run training
            self.run_training(config_path)

            return "Training completed!", gr.update(interactive=True)

        except Exception as e:
            return f"Error during training: {e!s}", gr.update(interactive=True)
        finally:
            # Clean up CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

    def run_training(self, config_path: Path) -> None:
        """Run the training process and update progress."""
        # Reset training state at the start
        self.training_state.reset()

        try:
            # Load config from YAML
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)

            # Initialize trainer config and trainer
            trainer_config = LtxvTrainerConfig(**config_dict)
            trainer = LtxvTrainer(trainer_config)

            def training_callback(step: int, total_steps: int, sampled_videos: list[Path] | None = None) -> None:
                """Callback function to update training progress and show samples."""
                # Update progress
                progress_pct = (step / total_steps) * 100
                self.training_state.update(progress=f"Step {step}/{total_steps} ({progress_pct:.1f}%)")

                # Update validation video at validation intervals
                if step % trainer_config.validation.interval == 0 and sampled_videos:
                    # Convert the first sample to GIF
                    gif_path = _handle_validation_sample(step, sampled_videos[0])
                    if gif_path:
                        self.training_state.update(validation=gif_path)

            logger.info("Starting training...")

            # Start training with callback
            saved_path, stats = trainer.train(disable_progress_bars=False, step_callback=training_callback)

            # Save checkpoint and get paths
            permanent_checkpoint_path, hf_repo = self._save_checkpoint(saved_path, trainer_config)

            # Update training outputs with completion status
            self.training_state.update(
                status="complete",
                download=str(permanent_checkpoint_path),
                hf_repo=hf_repo,
                checkpoint_path=str(permanent_checkpoint_path),
            )

            logger.info(f"Training completed. Model saved to {permanent_checkpoint_path}")
            logger.info(f"Training stats: {stats}")

        except Exception as e:
            logger.error(f"Training failed: {e!s}", exc_info=True)
            self.training_state.update(status="failed", error=str(e))
            raise
        finally:
            # Don't reset current_job here - let the UI handle it
            if self.training_state.status == "running":
                self.training_state.update(status="failed")

    def create_ui(self) -> gr.Blocks:
        """Create the Gradio UI."""
        with gr.Blocks() as blocks:
            gr.Markdown("# LTX-Video Trainer")

            with gr.Tab("Training"):
                with gr.Row():
                    with gr.Column():
                        # Video upload and caption section
                        self.video_upload = gr.File(label="Upload Videos", file_count="multiple", file_types=["video"])
                        self.caption_output = gr.Textbox(label="Generated/Edited Caption", interactive=True)
                        self.dataset_display = gr.Code(
                            label="Captions JSON",
                            language="json",
                            interactive=True,
                        )

                        generate_btn = gr.Button("Generate Caption")
                        reset_btn = gr.Button("Reset Everything", variant="secondary")

                        # Add validation prompt input
                        self.validation_prompt = gr.Textbox(
                            label="Validation Prompt",
                            placeholder="Enter the prompt to use for validation samples",
                            value="a professional portrait video of a person with blurry bokeh background",
                            interactive=True,
                            info="Include the LoRA ID token (e.g., &lt;lora&gt;) in this prompt if desired.",
                        )

                    with gr.Column():
                        # Basic training settings
                        model_source = gr.Dropdown(
                            choices=[str(v) for v in LtxModelVersion],
                            value=str(LtxModelVersion.latest()),
                            label="Model Version",
                            info="Select the model version to use for training",
                        )
                        lr = gr.Number(value=2e-4, label="Learning Rate")
                        steps = gr.Number(
                            value=1500, label="Training Steps", precision=0, info="Total number of training steps"
                        )
                        validation_interval = gr.Number(
                            value=100,
                            label="Validation Interval",
                            precision=0,
                            info="Number of steps between validation samples",
                            minimum=1,
                        )
                        lora_rank = gr.Dropdown(
                            choices=list(range(8, 257, 8)),
                            value=128,
                            label="LoRA Rank",
                            info="Higher rank = more capacity but more VRAM usage",
                        )
                        batch_size = gr.Number(value=1, label="Batch Size", precision=0)

                        # Add LoRA ID token input
                        id_token = gr.Textbox(
                            label="LoRA ID Token",
                            placeholder="Optional: Enter token to prepend to captions (e.g., <lora>)",
                            value="",
                            info="This token will be prepended to all training captions during training",
                        )

                        # Resolution inputs
                        with gr.Row():
                            width = gr.Dropdown(
                                choices=list(range(256, 1025, 32)),
                                value=768,
                                label="Video Width",
                                info="Width in pixels (multiple of 32)",
                            )
                            height = gr.Dropdown(
                                choices=list(range(256, 1025, 32)),
                                value=768,
                                label="Video Height",
                                info="Height in pixels (multiple of 32)",
                            )
                            num_frames = gr.Dropdown(
                                choices=list(range(9, 129, 8)),
                                value=25,
                                label="Number of Frames",
                                info="Number of frames in the video (multiple of 8)",
                            )

                        # HuggingFace Hub settings
                        with gr.Group(visible=True):
                            gr.Markdown("### HuggingFace Hub Settings")
                            push_to_hub = gr.Checkbox(
                                label="Push to HuggingFace Hub",
                                value=False,
                                info="Enable to push the model to HuggingFace Hub",
                            )
                            hf_token = gr.Textbox(
                                label="HuggingFace Token",
                                type="password",
                                visible=True,
                                info="Your HuggingFace API token",
                            )
                            hf_model_id = gr.Textbox(
                                label="Model ID",
                                placeholder="username/model-name",
                                visible=True,
                                info="Format: username/model-name",
                            )

                # Training control and output
                train_btn = gr.Button("Start Training", variant="primary")
                self.status_output = gr.Textbox(label="Status")
                self.progress_output = gr.Textbox(label="Progress")

                with gr.Row():
                    self.validation_sample = gr.Image(
                        label="Latest Validation Sample",
                        show_label=True,
                        interactive=False,
                        type="filepath",  # Use filepath to support GIFs
                        scale=1,  # Scale to fit container
                        container=True,  # Use container for proper scaling
                        height=512,  # Max height
                    )
                    self.training_output = gr.Textbox(label="Training Log", max_lines=20, show_copy_button=True)

                # Results section
                with gr.Group(visible=True):
                    gr.Markdown("### Training Results")
                    with gr.Row():
                        self.download_btn = gr.DownloadButton(
                            label="Download LoRA Weights",
                            visible=False,
                            interactive=True,
                        )
                        self.hf_repo_link = gr.HTML(
                            value="",
                            visible=False,
                            label="HuggingFace Hub",
                        )

            # Event handlers
            generate_btn.click(
                process_video, inputs=[self.video_upload, self.caption_output], outputs=[self.dataset_display]
            )

            # Update HF fields visibility based on push_to_hub checkbox
            push_to_hub.change(
                lambda x: {
                    hf_token: gr.update(visible=x),
                    hf_model_id: gr.update(visible=x),
                },
                inputs=[push_to_hub],
                outputs=[hf_token, hf_model_id],
            )

            train_btn.click(
                lambda videos,
                validation_prompt,
                lr,
                steps,
                lora_rank,
                batch_size,
                model_source,
                width,
                height,
                num_frames,
                push_to_hub,
                hf_model_id,
                hf_token,
                id_token,
                validation_interval,
                captions_json: self.start_training(
                    TrainingParams(
                        videos=videos,
                        validation_prompt=validation_prompt,
                        learning_rate=lr,
                        steps=steps,
                        lora_rank=lora_rank,
                        batch_size=batch_size,
                        model_source=model_source,
                        width=width,
                        height=height,
                        num_frames=num_frames,
                        push_to_hub=push_to_hub,
                        hf_model_id=hf_model_id,
                        hf_token=hf_token,
                        id_token=id_token,
                        validation_interval=validation_interval,
                        captions_json=captions_json,
                    )
                ),
                inputs=[
                    self.video_upload,
                    self.validation_prompt,
                    lr,
                    steps,
                    lora_rank,
                    batch_size,
                    model_source,
                    width,
                    height,
                    num_frames,
                    push_to_hub,
                    hf_model_id,
                    hf_token,
                    id_token,
                    validation_interval,
                    self.dataset_display,
                ],
                outputs=[self.status_output, train_btn],
            )

            # Update timer to use class method
            timer = gr.Timer(value=10)  # 1 second interval
            timer.tick(
                fn=self.update_progress,
                inputs=None,
                outputs=[
                    self.progress_output,
                    self.validation_sample,
                    self.training_output,
                    self.download_btn,
                    self.hf_repo_link,
                    self.hf_repo_link,
                ],
                show_progress=True,
            )

            # Handle download button click
            self.download_btn.click(self.get_model_path, inputs=None, outputs=[self.download_btn])

            # Handle reset button click
            reset_btn.click(
                self.reset_interface,
                inputs=None,
                outputs=[
                    self.video_upload,
                    self.caption_output,
                    self.dataset_display,
                    self.validation_prompt,
                    self.status_output,
                    self.progress_output,
                    self.validation_sample,
                    self.training_output,
                    self.download_btn,
                    self.hf_repo_link,
                ],
            )

        return blocks


def main() -> None:
    """Main entry point."""
    ui = GradioUI()
    demo = ui.create_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
