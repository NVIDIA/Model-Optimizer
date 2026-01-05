#!/usr/bin/env python3

"""
Auto-caption videos using vision-language models.

This script provides a command-line interface for generating captions for videos using
a vision-language model. It supports processing individual videos or entire directories,
customizing the captioning model, and saving the results to various formats.

The paths to videos in the generated dataset/captions file will be RELATIVE to the
directory where the output file is stored. This makes the dataset more portable and
easier to use in different environments.

Basic usage:
    # Caption a single video
    caption_videos.py video.mp4 --output captions.txt

    # Caption all videos in a directory
    caption_videos.py videos_dir/ --output captions.csv

    # Caption with custom instruction
    caption_videos.py video.mp4 --instruction "Describe what happens in this video in detail."

Advanced usage:
    # Use specific captioner type and device
    caption_videos.py videos_dir/ --captioner-type llava_next_7b --device cuda:0

    # Process videos with specific extensions and save as JSON
    caption_videos.py videos_dir/ --extensions mp4,mov,avi --output captions.json
"""

import csv
import json
from enum import Enum
from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers.utils.logging import disable_progress_bar

from ltxv_trainer.captioning import (
    DEFAULT_VLM_CAPTION_INSTRUCTION,
    CaptionerType,
    MediaCaptioningModel,
    create_captioner,
)

VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "webm"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS + IMAGE_EXTENSIONS
SAVE_INTERVAL = 10

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Auto-caption videos using vision-language models.",
)

disable_progress_bar()


class OutputFormat(str, Enum):
    """Available output formats for captions."""

    TXT = "txt"  # Separate files for captions and video paths, one caption / video path per line
    CSV = "csv"  # CSV file with video path and caption columns
    JSON = "json"  # JSON file with video paths as keys and captions as values
    JSONL = "jsonl"  # JSON Lines file with one JSON object per line


def caption_media(
    input_path: Path,
    output_path: Path,
    captioner: MediaCaptioningModel,
    extensions: list[str],
    recursive: bool,
    fps: int,
    clean_caption: bool,
    output_format: OutputFormat,
    override: bool,
) -> None:
    """Caption videos and images using the provided captioning model.
    Args:
        input_path: Path to input video file or directory
        output_path: Path to output caption file
        captioner: Video captioning model
        extensions: List of video file extensions to include
        recursive: Whether to search subdirectories recursively
        fps: Frames per second to sample from videos (ignored for images)
        clean_caption: Whether to clean up captions
        output_format: Format to save the captions in
        override: Whether to override existing captions
    """

    # Get list of media files to process
    media_files = _get_media_files(input_path, extensions, recursive)

    if not media_files:
        console.print("[bold yellow]No media files found to process.[/]")
        return

    console.print(f"Found [bold]{len(media_files)}[/] media files to process.")

    # Get the base directory for relative paths (the directory containing the output file)
    base_dir = output_path.parent.resolve()
    console.print(f"Using [bold blue]{base_dir}[/] as base directory for relative paths")

    # Load existing captions if the output file exists
    existing_captions = _load_existing_captions(output_path, output_format)

    # Convert existing captions keys to absolute paths for comparison
    existing_captions_abs = {}
    for rel_path, caption in existing_captions.items():
        abs_path = str((base_dir / rel_path).resolve())
        existing_captions_abs[abs_path] = caption

    # Filter out media that already have captions if not overriding
    media_to_process = []
    skipped_media = []

    for media_file in media_files:
        media_path_str = str(media_file.resolve())
        if not override and media_path_str in existing_captions_abs:
            skipped_media.append(media_file)
        else:
            media_to_process.append(media_file)

    if skipped_media:
        console.print(f"[bold yellow]Skipping [bold]{len(skipped_media)}[/] media that already have captions.[/]")

    if not media_to_process:
        console.print("[bold yellow]No media to process. All media already have captions.[/]")
        console.print("[bold yellow]Use --override to recaption all media.[/]")
        return

    console.print(f"Processing [bold]{len(media_to_process)}[/] media.")

    # Create progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )

    # Start with existing captions
    captions = existing_captions.copy()

    with progress:
        task = progress.add_task("Generating captions", total=len(media_to_process))

        for i, media_file in enumerate(media_to_process):
            # Update progress description to show current file
            progress.update(task, description=f"Captioning [bold blue]{media_file.name}[/]")

            try:
                # Generate caption for the media
                caption = captioner.caption(
                    path=media_file,
                    fps=fps,
                    clean_caption=clean_caption,
                )

                # Convert absolute path to relative path (relative to the output file's directory)
                rel_path = str(media_file.resolve().relative_to(base_dir))
                # Store the caption with the relative path as key
                captions[rel_path] = caption

            except Exception as e:
                console.print(f"[bold red]Error captioning [bold blue]{media_file}[/]: {e}[/]")

            if i % SAVE_INTERVAL == 0:
                _save_captions(captions, output_path, output_format)

            # Advance progress bar
            progress.advance(task)

    # Save captions to file
    _save_captions(captions, output_path, output_format)

    # Print summary
    processed_media = len(captions) - len(existing_captions)
    total_to_process = len(media_files) - len(skipped_media)
    console.print(
        f"[bold green]✓[/] Captioned [bold]{processed_media}/{total_to_process}[/] media successfully.",
    )


def _get_media_files(
    input_path: Path,
    extensions: list[str] = MEDIA_EXTENSIONS,
    recursive: bool = False,
) -> list[Path]:
    """Get all media files from the input path."""
    input_path = Path(input_path)
    # Normalize extensions to lowercase without dots
    extensions = [ext.lower().lstrip(".") for ext in extensions]

    if input_path.is_file():
        # If input is a file, check if it has a valid extension
        if input_path.suffix.lstrip(".").lower() in extensions:
            return [input_path]
        else:
            typer.echo(f"Warning: {input_path} is not a recognized media file. Skipping.")
            return []
    elif input_path.is_dir():
        # If input is a directory, find all media files
        media_files = []

        # Define the glob pattern based on whether we're searching recursively
        glob_pattern = "**/*" if recursive else "*"

        # Find all files with the specified extensions
        for ext in extensions:
            media_files.extend(input_path.glob(f"{glob_pattern}.{ext}"))

        return sorted(media_files)
    else:
        typer.echo(f"Error: {input_path} does not exist.")
        raise typer.Exit(code=1)


def _save_captions(
    captions: dict[str, str],
    output_path: Path,
    format_type: OutputFormat,
) -> None:
    """Save captions to a file in the specified format.

    Args:
        captions: Dictionary mapping media paths to captions
        output_path: Path to save the output file
        format_type: Format to save the captions in
    """
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]Saving captions...[/]")

    match format_type:
        case OutputFormat.TXT:
            # Create two separate files for captions and media paths
            captions_file = output_path.with_stem(f"{output_path.stem}_captions")
            paths_file = output_path.with_stem(f"{output_path.stem}_paths")

            with captions_file.open("w", encoding="utf-8") as f:
                for caption in captions.values():
                    f.write(f"{caption}\n")

            with paths_file.open("w", encoding="utf-8") as f:
                for media_path in captions:
                    f.write(f"{media_path}\n")

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{captions_file}[/]")
            console.print(f"[bold green]✓[/] Media paths saved to [cyan]{paths_file}[/]")
            console.print("[bold yellow]Note:[/] Use these files with ImageOrVideoDataset by setting:")
            console.print(f"  caption_column='{captions_file.name}'")
            console.print(f"  video_column='{paths_file.name}'")

        case OutputFormat.CSV:
            with output_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["caption", "media_path"])
                for media_path, caption in captions.items():
                    writer.writerow([caption, media_path])

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{output_path}[/]")
            console.print("[bold yellow]Note:[/] Use these files with ImageOrVideoDataset by setting:")
            console.print("  caption_column='[cyan]caption[/]'")
            console.print("  video_column='[cyan]media_path[/]'")

        case OutputFormat.JSON:
            # Format as list of dictionaries with caption and media_path keys
            json_data = [{"caption": caption, "media_path": media_path} for media_path, caption in captions.items()]

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{output_path}[/]")
            console.print("[bold yellow]Note:[/] Use these files with ImageOrVideoDataset by setting:")
            console.print("  caption_column='[cyan]caption[/]'")
            console.print("  video_column='[cyan]media_path[/]'")

        case OutputFormat.JSONL:
            with output_path.open("w", encoding="utf-8") as f:
                for media_path, caption in captions.items():
                    f.write(json.dumps({"caption": caption, "media_path": media_path}, ensure_ascii=False) + "\n")

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{output_path}[/]")
            console.print("[bold yellow]Note:[/] Use these files with ImageOrVideoDataset by setting:")
            console.print("  caption_column='[cyan]caption[/]'")
            console.print("  video_column='[cyan]media_path[/]'")

        case _:
            raise ValueError(f"Unsupported output format: {format_type}")


def _load_existing_captions(  # noqa: PLR0912
    output_path: Path,
    format_type: OutputFormat,
) -> dict[str, str]:
    """Load existing captions from a file.

    Args:
        output_path: Path to the captions file
        format_type: Format of the captions file

    Returns:
        Dictionary mapping media paths to captions, or empty dict if file doesn't exist
    """
    if not output_path.exists():
        return {}

    console.print(f"[bold blue]Loading existing captions from [cyan]{output_path}[/]...[/]")

    existing_captions = {}

    try:
        match format_type:
            case OutputFormat.TXT:
                # For TXT format, we have two separate files
                captions_file = output_path.with_stem(f"{output_path.stem}_captions")
                paths_file = output_path.with_stem(f"{output_path.stem}_paths")

                if captions_file.exists() and paths_file.exists():
                    captions = captions_file.read_text(encoding="utf-8").splitlines()
                    paths = paths_file.read_text(encoding="utf-8").splitlines()

                    if len(captions) == len(paths):
                        existing_captions = dict(zip(paths, captions, strict=False))

            case OutputFormat.CSV:
                with output_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    # Skip header
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            caption, media_path = row[0], row[1]
                            existing_captions[media_path] = caption

            case OutputFormat.JSON:
                with output_path.open("r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    for item in json_data:
                        if "caption" in item and "media_path" in item:
                            existing_captions[item["media_path"]] = item["caption"]

            case OutputFormat.JSONL:
                with output_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line)
                        if "caption" in item and "media_path" in item:
                            existing_captions[item["media_path"]] = item["caption"]

            case _:
                raise ValueError(f"Unsupported output format: {format_type}")

        console.print(f"[bold green]✓[/] Loaded [bold]{len(existing_captions)}[/] existing captions")
        return existing_captions

    except Exception as e:
        console.print(f"[bold yellow]Warning: Could not load existing captions: {e}[/]")
        return {}


@app.command()
def main(  # noqa: PLR0913
    input_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to input video/image file or directory containing media files",
        exists=True,
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Path to output file for captions. Format determined by file extension.",
    ),
    captioner_type: CaptionerType = typer.Option(  # noqa: B008
        CaptionerType.QWEN_25_VL,
        "--captioner-type",
        "-c",
        help="Type of captioner to use. Valid values: 'llava_next_7b', 'qwen_25_vl'",
        case_sensitive=False,
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        "-d",
        help="Device to use for inference (e.g., 'cuda', 'cuda:0', 'cpu')",
    ),
    use_8bit: bool = typer.Option(
        False,
        "--use-8bit",
        help="Whether to use 8-bit precision for the captioning model",
    ),
    instruction: str = typer.Option(
        DEFAULT_VLM_CAPTION_INSTRUCTION,
        "--instruction",
        "-i",
        help="Instruction to give to the captioning model",
    ),
    extensions: str = typer.Option(
        ",".join(MEDIA_EXTENSIONS),
        "--extensions",
        "-e",
        help="Comma-separated list of media file extensions to process",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Search for media files in subdirectories recursively",
    ),
    fps: int = typer.Option(
        3,
        "--fps",
        "-f",
        help="Frames per second to sample from videos (ignored for images)",
    ),
    clean_caption: bool = typer.Option(
        True,
        "--clean-caption",
        help="Whether to clean up captions by removing common VLM patterns",
    ),
    override: bool = typer.Option(
        False,
        "--override",
        help="Whether to override existing captions for media",
    ),
) -> None:
    """Auto-caption videos and images using vision-language models.

    This script supports both LLaVA-NeXT and Qwen2.5-VL models for generating captions.
    The paths in the output file will be relative to the output file's directory.

    Examples:
        # Caption using LLaVA-NeXT (default)
        caption_videos.py video.mp4 -o captions.txt

        # Caption using Qwen2.5-VL
        caption_videos.py video.mp4 -o captions.txt -c qwen_25_vl

        # Caption with custom instruction (especially useful for Qwen)
        caption_videos.py video.mp4 -o captions.txt -c qwen_25_vl -i "Describe this video in detail"

    Valid captioner types:
        qwen_25_vl: Qwen2.5-VL-7B model (default)
        llava_next_7b: LLaVA-NeXT-7B model (default)

    """

    # Determine device
    device = device or "cuda" if torch.cuda.is_available() else "cpu"

    # Parse extensions
    ext_list = [ext.strip() for ext in extensions.split(",")]

    # Determine output path and format
    if output is None:
        output_format = OutputFormat.JSON
        if input_path.is_file():  # noqa: SIM108
            # Default to a JSON file with the same name as the input media
            output = input_path.with_suffix(".dataset.json")
        else:
            # Default to a JSON file in the input directory
            output = input_path / "dataset.json"
    else:
        # Determine format from file extension
        output_format = OutputFormat(Path(output).suffix.lstrip(".").lower())

    # Ensure output path is absolute
    output = Path(output).resolve()
    console.print(f"Output will be saved to [bold blue]{output}[/]")

    # Initialize captioning model
    with console.status("Loading captioning model...", spinner="dots"):
        captioner = create_captioner(
            captioner_type=captioner_type,
            device=device,
            use_8bit=use_8bit,
            vlm_instruction=instruction,
        )
        console.print("[bold green]✓[/] Captioning model loaded successfully")

    # Caption media files
    caption_media(
        input_path=input_path,
        output_path=output,
        captioner=captioner,
        extensions=ext_list,
        recursive=recursive,
        fps=fps,
        clean_caption=clean_caption,
        output_format=output_format,
        override=override,
    )


if __name__ == "__main__":
    app()
