#!/usr/bin/env python3

"""
Preprocess a video dataset by computing video clips latents and text captions embeddings.

This script provides a command-line interface for preprocessing video datasets by computing
latent representations of video clips and text embeddings of their captions. The preprocessed
data can be used to accelerate training of video generation models and to save GPU memory.

Basic usage:
    preprocess_dataset.py /path/to/dataset.json --resolution-buckets 768x768x49

The dataset must be a CSV, JSON, or JSONL file with columns for captions and video paths.
"""

from pathlib import Path

import typer
from decode_latents import LatentsDecoder
from rich.console import Console

from ltxv_trainer import logger
from scripts.process_captions import compute_captions_embeddings
from scripts.process_videos import compute_video_latents, parse_resolution_buckets

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Preprocess a video dataset by computing video clips latents and text captions embeddings. "
    "The dataset must be a CSV, JSON, or JSONL file with columns for captions and video paths.",
)


def preprocess_dataset(  # noqa: PLR0913
    dataset_file: str,
    caption_column: str,
    video_column: str,
    resolution_buckets: list[tuple[int, int, int]],
    batch_size: int,
    output_dir: str | None,
    id_token: str | None,
    vae_tiling: bool,
    decode: bool,
    model_source: str,
    device: str,
    load_text_encoder_in_8bit: bool,
    remove_llm_prefixes: bool = False,
    reference_column: str | None = None,
    text_encoder_path: str | None = None,
    encoder_type: str = "t5",
    with_audio: bool = False,
) -> None:
    """Run the preprocessing pipeline with the given arguments"""
    # Validate dataset file
    _validate_dataset_file(dataset_file)

    # Set up output directories
    output_base = Path(output_dir) if output_dir else Path(dataset_file).parent / ".precomputed"
    conditions_dir = output_base / "conditions"
    latents_dir = output_base / "latents"

    if id_token:
        logger.info(f'LoRA trigger word "{id_token}" will be prepended to all captions')

    # Process captions using the dedicated function
    compute_captions_embeddings(
        dataset_file=dataset_file,
        output_dir=str(conditions_dir),
        caption_column=caption_column,
        media_column=video_column,
        id_token=id_token,
        remove_llm_prefixes=remove_llm_prefixes,
        batch_size=batch_size,
        device=device,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        model_source=model_source,
        text_encoder_path=text_encoder_path,
        encoder_type=encoder_type,
    )

    # Process videos using the dedicated function
    if with_audio:
        logger.info("Audio preprocessing enabled - will extract and encode audio from videos")
        audio_latents_dir = output_base / "audio_latents"
    else:
        audio_latents_dir = None

    compute_video_latents(
        dataset_file=dataset_file,
        video_column=video_column,
        resolution_buckets=resolution_buckets,
        output_dir=str(latents_dir),
        model_source=model_source,
        batch_size=batch_size,
        device=device,
        vae_tiling=vae_tiling,
        with_audio=with_audio,
        audio_output_dir=str(audio_latents_dir) if audio_latents_dir else None,
    )

    # Process reference videos if reference_column is provided
    if reference_column:
        logger.info("Processing reference videos for IC-LoRA training...")
        reference_latents_dir = output_base / "reference_latents"

        compute_video_latents(
            dataset_file=dataset_file,
            main_media_column=video_column,
            video_column=reference_column,
            resolution_buckets=resolution_buckets,
            output_dir=str(reference_latents_dir),
            model_source=model_source,
            batch_size=batch_size,
            device=device,
            vae_tiling=vae_tiling,
        )

    # Handle decoding if requested (for verification)
    if decode:
        logger.info("Decoding latents for verification...")

        decoder = LatentsDecoder(
            model_source=model_source,
            device=device,
            vae_tiling=vae_tiling,
            with_audio=with_audio,
        )
        decoder.decode(latents_dir, output_base / "decoded_videos")

        # Also decode reference videos if they exist
        if reference_column:
            reference_latents_dir = output_base / "reference_latents"
            if reference_latents_dir.exists():
                logger.info("Decoding reference videos...")
                decoder.decode(reference_latents_dir, output_base / "decoded_reference_videos")

        # Decode audio latents if they exist
        if with_audio and audio_latents_dir and audio_latents_dir.exists():
            logger.info("Decoding audio latents...")
            decoder.decode_audio(audio_latents_dir, output_base / "decoded_audio")

    # Print summary
    logger.info(f"Dataset preprocessing complete! Results saved to {output_base}")
    if reference_column:
        logger.info("Reference videos processed and saved to reference_latents/ directory for IC-LoRA training")
    if with_audio:
        logger.info("Audio latents saved to audio_latents/ directory for audio-video training")


def _validate_dataset_file(dataset_path: str) -> None:
    """Validate that the dataset file exists and has the correct format"""
    dataset_file = Path(dataset_path)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_file}")

    if not dataset_file.is_file():
        raise ValueError(f"Dataset path must be a file, not a directory: {dataset_file}")

    if dataset_file.suffix.lower() not in [".csv", ".json", ".jsonl"]:
        raise ValueError(f"Dataset file must be CSV, JSON, or JSONL format: {dataset_file}")


@app.command()
def main(  # noqa: PLR0913
    dataset_path: str = typer.Argument(
        ...,
        help="Path to metadata file (CSV/JSON/JSONL) containing captions and video paths",
    ),
    resolution_buckets: str = typer.Option(
        ...,
        help='Resolution buckets in format "WxHxF;WxHxF;..." (e.g. "768x768x25;512x512x49")',
    ),
    caption_column: str = typer.Option(
        default="caption",
        help="Column name containing captions in the dataset JSON/JSONL/CSV file",
    ),
    video_column: str = typer.Option(
        default="media_path",
        help="Column name containing video paths in the dataset JSON/JSONL/CSV file",
    ),
    batch_size: int = typer.Option(
        default=1,
        help="Batch size for preprocessing",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    load_text_encoder_in_8bit: bool = typer.Option(
        default=False,
        help="Load the text encoder in 8-bit precision to save memory",
    ),
    vae_tiling: bool = typer.Option(
        default=False,
        help="Enable VAE tiling for larger video resolutions",
    ),
    output_dir: str | None = typer.Option(
        default=None,
        help="Output directory (defaults to .precomputed in dataset directory)",
    ),
    model_source: str = typer.Option(
        ...,
        help="Path to LTX checkpoint (.safetensors file)",
    ),
    text_encoder_path: str | None = typer.Option(
        default=None,
        help="Path to text encoder directory. For T5: optional (None=load from HF). "
        "For Gemma: required (Gemma model path)",
    ),
    encoder_type: str = typer.Option(
        default="t5",
        help="Text encoder type: 't5' (for LTXV models) or 'gemma' (for LTX-2 models)",
    ),
    id_token: str | None = typer.Option(
        default=None,
        help="Optional token to prepend to each caption (acts as a trigger word when training a LoRA)",
    ),
    decode: bool = typer.Option(
        default=False,
        help="Decode and save latents after encoding (videos and audio) for verification",
    ),
    remove_llm_prefixes: bool = typer.Option(
        default=False,
        help="Remove LLM prefixes from captions",
    ),
    reference_column: str | None = typer.Option(
        default=None,
        help="Column name containing reference video paths in the dataset JSON/JSONL/CSV file",
    ),
    with_audio: bool = typer.Option(
        default=False,
        help="Extract and encode audio from video files (requires LTX-2 model with AudioVAE)",
    ),
) -> None:
    """Preprocess a video dataset by computing and saving latents and text embeddings.

    The dataset must be a CSV, JSON, or JSONL file with columns for captions and video paths.

    Examples:
        # Process a CSV dataset
        python preprocess_dataset.py dataset.csv --resolution-buckets 768x768x25

        # Process a JSON dataset with custom column names
        python preprocess_dataset.py dataset.json
            --resolution-buckets 768x768x25 --caption-column "text" --video-column "video_path"

        # Process dataset with reference videos for IC-LoRA training
        python preprocess_dataset.py dataset.json
            --resolution-buckets 768x768x25 --caption-column "caption"
            --video-column "media_path" --reference-column "reference_path"

        # Process dataset with audio for audio-video training (requires LTX-2 model)
        python preprocess_dataset.py dataset.json
            --resolution-buckets 768x512x97 --model-source /path/to/ltx2.safetensors
            --text-encoder-path /path/to/gemma --encoder-type gemma --with-audio
    """
    parsed_resolution_buckets = parse_resolution_buckets(resolution_buckets)

    if len(parsed_resolution_buckets) > 1:
        logger.warning(
            "Using multiple resolution buckets. "
            "When training with multiple resolution buckets, you must use a batch size of 1."
        )

    preprocess_dataset(
        dataset_file=dataset_path,
        caption_column=caption_column,
        video_column=video_column,
        resolution_buckets=parsed_resolution_buckets,
        batch_size=batch_size,
        output_dir=output_dir,
        id_token=id_token,
        vae_tiling=vae_tiling,
        decode=decode,
        model_source=model_source,
        device=device,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        remove_llm_prefixes=remove_llm_prefixes,
        reference_column=reference_column,
        text_encoder_path=text_encoder_path,
        encoder_type=encoder_type,
        with_audio=with_audio,
    )


if __name__ == "__main__":
    app()
