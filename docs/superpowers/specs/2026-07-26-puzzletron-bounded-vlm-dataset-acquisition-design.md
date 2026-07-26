# Puzzletron Bounded VLM Dataset Acquisition and Setup Design

## Scope

This change is intentionally limited to dataset acquisition, normalization, and
setup-v2 configuration. It does not yet change Puzzletron stage recipe wiring or
the implementation of evaluation, importance estimation, bypass, or global KD.

Setup v2 will expose four dataset choices:

1. the dataset supplied by `sepehr_defaults.yaml`, when present;
2. NVIDIA Puzzle-KD Nemotron Post-Training Dataset v2;
3. NVIDIA Nemotron VLM Dataset v2;
4. a custom local path or Hugging Face dataset.

When the configured default already names one of the first-class NVIDIA
datasets, the wizard de-duplicates the entries while preserving the configured
default's provenance.

Nemotron Image Training v3 is explicitly out of scope because its repository
contains conversation metadata but requires separately acquired upstream media.

## Source contracts

### Puzzle-KD text

- Repository: `nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2`
- Modality: `text`
- Expected content field: `messages`
- The setup answer records the immutable repository revision when it can be
  resolved.
- Existing text loading and token-cache behavior remains unchanged.

### Nemotron VLM v2

- Repository: `nvidia/Nemotron-VLM-Dataset-v2`
- Modality: `multimodal`
- Default subsets: `sparsetables`, `plotqa_cot`, and `wiki_en`, matching the
  Model Optimizer image-calibration defaults.
- Acquisition is bounded by:
  - an explicit list of named subsets;
  - a requested number of valid normalized rows;
  - a deterministic seed;
  - a maximum number of media tar shards per subset; and
  - a destination directory.
- The downloader joins each subset's JSONL conversations to images in that
  subset's media tar shards. It does not clone or snapshot the full repository.
- Rows are normalized with the full user/assistant conversation. The image
  reference in the message is replaced by the matched image before
  materialization.
- Invalid rows, missing media, unsupported media types, and exhausted shard
  budgets are reported explicitly. A requested row count that cannot be
  satisfied is an error rather than a silently smaller dataset.

## Considered approaches

### Direct streaming in every Puzzletron stage

This minimizes initial disk use, but every stage would repeat network access and
tar scans. It also makes row identity depend on transient repository state and
worker connectivity. This is not selected.

### Full or pattern-based repository snapshot

Hugging Face allow-pattern downloads can restrict files, but media tar shards
remain large and the resulting dataset size is controlled only indirectly. It
also encourages accidentally downloading all selected subsets. This is not the
default acquisition path.

### Bounded normalized materialization

The selected design streams repository metadata, opens at most the configured
number of media shards per subset, selects a deterministic number of valid
rows, and writes Puzzletron's existing offline conversation format plus a
content manifest. This gives a hard row bound, an explicit shard bound, stable
reuse across cluster stages, and compatibility with the existing AutoModel VLM
collator and sequence packer.

## Materialized dataset

The existing materialized conversation format remains canonical:

- `samples.json` contains normalized full conversations and relative media
  paths;
- `images/` contains only media referenced by accepted rows; and
- `manifest.json` contains dataset repository, immutable revision, selected
  subsets, seed, shard limit, requested and realized sample counts, rejected-row
  diagnostics, and image content hashes.

Materialization is idempotent only when the existing manifest matches the
requested acquisition identity. A mismatched request must fail with a clear
message instead of reusing unrelated cached data.

The normalized dataset loader remains source-agnostic: InterSyn and Nemotron
materializations both load through the same conversation-dataset factory.

## Setup-v2 interaction

The dataset screen first presents the available source choices. Choosing a
first-class source records its repository, modality, and adapter without asking
the user to re-enter them.

Choosing Nemotron VLM v2 additionally asks for:

- comma-separated subset names, defaulting to the three Model Optimizer
  calibration subsets;
- total valid rows to materialize;
- deterministic seed;
- maximum media shards per subset; and
- local materialization directory.

The wizard validates that counts and shard limits are positive, subsets are
non-empty and unique, and the selected model is multimodal. It records these
answers in the v2 state and renders them into dataset acquisition configuration.
The wizard only generates files; it does not perform a large network download
while answering prompts.

Custom dataset behavior is preserved. The existing default dataset remains
available and retains its current provenance.

## Compatibility boundaries

- Text-only configurations render the same `data` and `tokenize_data` behavior
  as before.
- Existing custom and InterSyn-style materialized multimodal datasets continue
  to load unchanged.
- No model-forward, VLM processor, packing, PP, CP, DP, or KD code is modified
  in this slice.
- The new Nemotron materialization preserves full assistant responses so it can
  later be wired safely into evaluation and KD; it does not reuse the
  prompt-only PTQ calibration collator.
- Sequence packing is not implemented in the downloader. The materialized
  samples preserve the existing conversation contract consumed by AutoModel's
  `PreTokenizedDatasetWrapper` and `neat_pack_dataset_vlm`.

## Validation and tests

Focused tests will cover:

- source-choice ordering, labels, default de-duplication, and modality;
- setup-v2 prompts and persisted acquisition fields;
- rendering of the bounded acquisition configuration;
- normalization of Nemotron full conversations and image replacement;
- deterministic selection across named subsets;
- maximum-shard enforcement;
- explicit failure when the requested valid-row count cannot be met;
- manifest acquisition identity and cache mismatch rejection;
- loading both existing InterSyn and new Nemotron materializations through the
  common factory; and
- regression snapshots for existing text/custom setup output.

Tests will use synthetic JSONL and tar fixtures or injected repository adapters;
they will not require network access.
