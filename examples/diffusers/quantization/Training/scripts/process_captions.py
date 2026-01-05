#!/usr/bin/env python

"""
Compute text embeddings for video generation training.

This module provides functionality for processing text captions, including:
- Loading captions from various file formats (CSV, JSON, JSONL)
- Cleaning and preprocessing text (removing LLM prefixes, adding ID tokens)
- CaptionsDataset for caption-only preprocessing workflows

Can be used as a standalone script:
    python scripts/process_captions.py dataset.json --output-dir /path/to/output --id-token "mytoken"
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Dataset
from transformers.utils.logging import disable_progress_bar

from ltx_core.model_loader import load_text_encoder
from ltxv_trainer import logger

disable_progress_bar()

# Common phrases that LLMs often add to captions that we might want to remove
COMMON_BEGINNING_PHRASES: tuple[str, ...] = (
    "This video",
    "The video",
    "This clip",
    "The clip",
    "The animation",
    "This image",
    "The image",
    "This picture",
    "The picture",
)

COMMON_CONTINUATION_WORDS: tuple[str, ...] = (
    "shows",
    "depicts",
    "features",
    "captures",
    "highlights",
    "introduces",
    "presents",
)

COMMON_LLM_START_PHRASES: tuple[str, ...] = (
    "In the video,",
    "In this video,",
    "In this video clip,",
    "In the clip,",
    "Caption:",
    *(
        f"{beginning} {continuation}"
        for beginning in COMMON_BEGINNING_PHRASES
        for continuation in COMMON_CONTINUATION_WORDS
    ),
)

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Process text captions and save embeddings for video generation training.",
)


class CaptionsDataset(Dataset):
    """
    Dataset for processing text captions only.

    This dataset is designed for caption preprocessing workflows where you only need
    to process text without loading videos. Useful for:
    - Precomputing text embeddings
    - Caption cleaning and preprocessing
    - Text-only preprocessing pipelines
    """

    def __init__(
        self,
        dataset_file: str | Path,
        caption_column: str,
        media_column: str = "media_path",
        id_token: str | None = None,
        remove_llm_prefixes: bool = False,
    ) -> None:
        """
        Initialize the captions dataset.

        Args:
            dataset_file: Path to CSV/JSON/JSONL metadata file
            caption_column: Column name for captions in the metadata file
            media_column: Column name for media paths (used for output naming)
            id_token: Optional token to prepend to each caption
            remove_llm_prefixes: Whether to remove common LLM-generated prefixes
        """
        super().__init__()

        self.dataset_file = Path(dataset_file)
        self.caption_column = caption_column
        self.media_column = media_column
        self.id_token = f"{id_token.strip()} " if id_token else ""

        # Load captions with their corresponding output embedding paths
        self.caption_data = self._load_caption_data()

        # Convert to lists for indexing
        self.output_paths = list(self.caption_data.keys())
        self.prompts = list(self.caption_data.values())

        # Clean LLM start phrases if requested
        if remove_llm_prefixes:
            self._clean_llm_prefixes()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single caption with optional ID token prepended and output path."""
        prompt = self.id_token + self.prompts[index]
        return {
            "prompt": prompt,
            "output_path": self.output_paths[index],
            "index": index,
        }

    def _load_caption_data(self) -> dict[str, str]:
        """Load captions and compute their output embedding paths."""
        if self.dataset_file.suffix == ".csv":
            return self._load_caption_data_from_csv()
        elif self.dataset_file.suffix == ".json":
            return self._load_caption_data_from_json()
        elif self.dataset_file.suffix == ".jsonl":
            return self._load_caption_data_from_jsonl()
        else:
            raise ValueError("Expected `dataset_file` to be a path to a CSV, JSON, or JSONL file.")

    def _load_caption_data_from_csv(self) -> dict[str, str]:
        """Load captions from a CSV file and compute output embedding paths."""
        df = pd.read_csv(self.dataset_file)

        if self.caption_column not in df.columns:
            raise ValueError(f"Column '{self.caption_column}' not found in CSV file")
        if self.media_column not in df.columns:
            raise ValueError(f"Column '{self.media_column}' not found in CSV file")

        caption_data = {}
        for _, row in df.iterrows():
            media_path = Path(row[self.media_column].strip())
            # Convert media path to embedding output path (same structure, .pt extension)
            output_path = str(media_path.with_suffix(".pt"))
            caption_data[output_path] = row[self.caption_column]

        return caption_data

    def _load_caption_data_from_json(self) -> dict[str, str]:
        """Load captions from a JSON file and compute output embedding paths."""
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")

        caption_data = {}
        for entry in data:
            if self.caption_column not in entry:
                raise ValueError(f"Key '{self.caption_column}' not found in JSON entry: {entry}")
            if self.media_column not in entry:
                raise ValueError(f"Key '{self.media_column}' not found in JSON entry: {entry}")

            media_path = Path(entry[self.media_column].strip())
            # Convert media path to embedding output path (same structure, .pt extension)
            output_path = str(media_path.with_suffix(".pt"))
            caption_data[output_path] = entry[self.caption_column]

        return caption_data

    def _load_caption_data_from_jsonl(self) -> dict[str, str]:
        """Load captions from a JSONL file and compute output embedding paths."""
        caption_data = {}
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                if self.caption_column not in entry:
                    raise ValueError(f"Key '{self.caption_column}' not found in JSONL entry: {entry}")
                if self.media_column not in entry:
                    raise ValueError(f"Key '{self.media_column}' not found in JSONL entry: {entry}")

                media_path = Path(entry[self.media_column].strip())
                # Convert media path to embedding output path (same structure, .pt extension)
                output_path = str(media_path.with_suffix(".pt"))
                caption_data[output_path] = entry[self.caption_column]

        return caption_data

    def _clean_llm_prefixes(self) -> None:
        """Remove common LLM-generated prefixes from captions."""
        for i in range(len(self.prompts)):
            self.prompts[i] = self.prompts[i].strip()
            for phrase in COMMON_LLM_START_PHRASES:
                if self.prompts[i].startswith(phrase):
                    self.prompts[i] = self.prompts[i].removeprefix(phrase).strip()
                    break


def compute_captions_embeddings(  # noqa: PLR0913
    dataset_file: str | Path,
    output_dir: str,
    model_source: str | None,
    caption_column: str = "caption",
    media_column: str = "media_path",
    id_token: str | None = None,
    remove_llm_prefixes: bool = False,
    batch_size: int = 8,
    device: str = "cuda",
    load_text_encoder_in_8bit: bool = False,
    text_encoder_path: str | None = None,
    encoder_type: str = "t5",
) -> None:
    """
    Process captions and save text embeddings.

    Args:
        dataset_file: Path to metadata file (CSV/JSON/JSONL) containing captions and media paths
        output_dir: Directory to save embeddings
        model_source: Path to LTX checkpoint (required for Gemma, not used for T5)
        caption_column: Column name containing captions in the metadata file
        media_column: Column name containing media paths (used for output naming)
        id_token: Optional token to prepend to each caption
        remove_llm_prefixes: Whether to remove common LLM-generated prefixes
        batch_size: Batch size for processing
        device: Device to use for computation
        load_text_encoder_in_8bit: Whether to load text encoder in 8-bit
        text_encoder_path: Path to text encoder directory (None for T5=load from HF, required for Gemma)
        encoder_type: Text encoder type ('t5' or 'gemma')
    """

    console = Console()

    # Create dataset
    dataset = CaptionsDataset(
        dataset_file=dataset_file,
        caption_column=caption_column,
        media_column=media_column,
        id_token=id_token,
        remove_llm_prefixes=remove_llm_prefixes,
    )
    logger.info(f"Loaded {len(dataset):,} captions")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load text encoder
    with console.status("[bold]Loading text encoder...", spinner="dots"):
        # For Gemma: need both text_encoder_path (Gemma model) and model_source (LTX checkpoint)
        # For T5: only need text_encoder_path (None = load from HF, or path to T5 directory)
        if encoder_type == "gemma":
            ltx_checkpoint_path = model_source
            encoder_path = text_encoder_path
        else:  # T5
            ltx_checkpoint_path = None
            encoder_path = text_encoder_path  # None = load from HF, or path to T5 directory

        text_encoder = load_text_encoder(
            encoder_path=encoder_path,
            encoder_type=encoder_type,
            ltx_checkpoint_path=ltx_checkpoint_path,
            device=device,
            dtype=torch.bfloat16,
            load_in_8bit=load_text_encoder_in_8bit,
        )

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Process batches
    total_batches = len(dataloader)
    logger.info(f"Processing captions in {total_batches:,} batches...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing captions", total=len(dataloader))
        for batch in dataloader:
            # Encode prompts
            with torch.inference_mode():
                prompt_embeds, prompt_attention_mask = text_encoder.encode_text(batch["prompt"])

            # Save embeddings for each item in batch
            for i in range(len(batch["prompt"])):
                output_rel_path = Path(batch["output_path"][i])

                # Create output directory maintaining structure
                output_dir_path = output_path / output_rel_path.parent
                output_dir_path.mkdir(parents=True, exist_ok=True)

                embedding_data = {
                    "prompt_embeds": prompt_embeds[i].cpu().contiguous(),
                    "prompt_attention_mask": prompt_attention_mask[i].cpu().contiguous(),
                }

                output_file = output_path / output_rel_path
                torch.save(embedding_data, output_file)

            progress.advance(task)

    logger.info(f"Processed {len(dataset):,} captions. Embeddings saved to {output_path}")


@app.command()
def main(  # noqa: PLR0913
    dataset_file: str = typer.Argument(
        ...,
        help="Path to metadata file (CSV/JSON/JSONL) containing captions and media paths",
    ),
    output_dir: str = typer.Option(
        ...,
        help="Output directory to save text embeddings",
    ),
    caption_column: str = typer.Option(
        default="caption",
        help="Column name containing captions in the dataset JSON/JSONL/CSV file",
    ),
    media_column: str = typer.Option(
        default="media_path",
        help="Column name in the dataset JSON/JSONL/CSV file containing media paths "
        "(used for output file naming and folder structure)",
    ),
    batch_size: int = typer.Option(
        default=8,
        help="Batch size for processing",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    load_text_encoder_in_8bit: bool = typer.Option(
        default=False,
        help="Load the text encoder in 8-bit precision to save memory",
    ),
    model_source: str | None = typer.Option(
        default=None,
        help="Path to LTX checkpoint (required for Gemma/LTX-2, not used for T5/LTXV)",
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
    remove_llm_prefixes: bool = typer.Option(
        default=False,
        help="Remove common LLM-generated prefixes from captions",
    ),
) -> None:
    """Process text captions and save embeddings for video generation training.

    This script processes captions from metadata files and saves text embeddings
    that can be used for training video generation models. The output embeddings
    will maintain the same folder structure and naming as the corresponding media files.

    Examples:
        # Process captions from a CSV file with LTXV model (uses T5 text encoder, loads from HF)
        python scripts/process_captions.py dataset.csv --output-dir ./embeddings --caption-column "text"

        # Process captions with LTX-2 model (uses Gemma text encoder)
        python scripts/process_captions.py dataset.json --output-dir ./embeddings \
            --model-source /path/to/ltx2_checkpoint.safetensors \
            --text-encoder-path /path/to/gemma --encoder-type gemma

        # Process captions with custom T5 model directory
        python scripts/process_captions.py dataset.json --output-dir ./embeddings \
            --text-encoder-path /path/to/t5-model-dir

        # Add a trigger token for LoRA training
        python scripts/process_captions.py dataset.json --output-dir ./embeddings --id-token "mytoken"
    """

    # Determine data root from dataset file path
    if not Path(dataset_file).is_file():
        raise typer.BadParameter(f"Dataset file not found: {dataset_file}")

    if id_token:
        logger.info(f'Trigger token "{id_token}" will be prepended to all captions')

    # Process embeddings
    compute_captions_embeddings(
        dataset_file=dataset_file,
        output_dir=output_dir,
        caption_column=caption_column,
        media_column=media_column,
        id_token=id_token,
        remove_llm_prefixes=remove_llm_prefixes,
        batch_size=batch_size,
        device=device,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        model_source=model_source,
        text_encoder_path=text_encoder_path,
        encoder_type=encoder_type,
    )


if __name__ == "__main__":
    app()
