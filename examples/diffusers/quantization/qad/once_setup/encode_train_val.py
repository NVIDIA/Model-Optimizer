import torch
from diffusers import DiffusionPipeline
from datasets import load_dataset
import time

def log(msg):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    DEVICE = "cuda"

    teacher_pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image-2512",
        torch_dtype=torch.bfloat16,
    )
    TRAIN_SAMPLES = 40_000
    NUM_VAL_ENCODE = 100
    PROMPT_TEMPLATE_DROP_IDX = 34  # system prompt 
    TOKENIZER_MAX_LENGTH = 1024
    PROMPT_TEMPLATE = (
        "<|im_start|>system\n"
        "Describe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    TOKENIZER_MAX_LENGTH = 1024

    tokenizer = teacher_pipe.tokenizer
    text_encoder = teacher_pipe.text_encoder.to(DEVICE).eval()

    #load dataset, split, and encode training and validation prompts
    ds = load_dataset("Gustavosta/Stable-Diffusion-Prompts")
    all_prompts = ds["train"]["Prompt"]
    total_prompts = len(all_prompts)
    log(f"Total prompts in dataset: {total_prompts:,}")

    train_prompts = all_prompts[:TRAIN_SAMPLES]
    val_prompts = all_prompts[TRAIN_SAMPLES:TRAIN_SAMPLES + NUM_VAL_ENCODE]

    batch_size = 8
    train_embeds = []
    val_embeds = []

    for i in range(0, len(train_prompts), batch_size):
        batch_prompts = train_prompts[i : i + batch_size]
        txt = [PROMPT_TEMPLATE.format(p) for p in batch_prompts]

        txt_tokens = tokenizer(
            txt,
            max_length=TOKENIZER_MAX_LENGTH + PROMPT_TEMPLATE_DROP_IDX,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            encoder_out = text_encoder(
                input_ids=txt_tokens.input_ids,
                attention_mask=txt_tokens.attention_mask,
                output_hidden_states=True,
            )

        hidden_states = encoder_out.hidden_states[-1]
        mask = txt_tokens.attention_mask
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_hidden = torch.split(selected, valid_lengths.tolist(), dim=0)
        # Drop the system prompt tokens (first drop_idx tokens)
        split_hidden = [e[PROMPT_TEMPLATE_DROP_IDX:].cpu() for e in split_hidden]
        train_embeds.extend(split_hidden)

    for i in range(0, len(val_prompts), batch_size):
        batch_prompts = val_prompts[i : i + batch_size]
        txt = [PROMPT_TEMPLATE.format(p) for p in batch_prompts]

        txt_tokens = tokenizer(
            txt,
            max_length=TOKENIZER_MAX_LENGTH + PROMPT_TEMPLATE_DROP_IDX,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            encoder_out = text_encoder(
                input_ids=txt_tokens.input_ids,
                attention_mask=txt_tokens.attention_mask,
                output_hidden_states=True,
            )
        hidden_states = encoder_out.hidden_states[-1]
        mask = txt_tokens.attention_mask
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_hidden = torch.split(selected, valid_lengths.tolist(), dim=0)
        # Drop the system prompt tokens (first drop_idx tokens)
        split_hidden = [e[PROMPT_TEMPLATE_DROP_IDX:].cpu() for e in split_hidden]
        val_embeds.extend(split_hidden)

    torch.save({i: e for i, e in enumerate(train_embeds)}, "train_embeds.pt")
    torch.save({i: e for i, e in enumerate(val_embeds)}, "val_embeds.pt")

if __name__ == "__main__":
    main()

# === Truncation analysis (raw prompts, no template) ===
# Train (70000 prompts):
#   min=2, max=441, mean=61.2
#   Truncated (raw tokens > 1024): 0 / 70000  (0.00%)
# Val (100 prompts):
#   min=11, max=124, mean=60.1
#   Truncated (raw tokens > 1024): 0 / 100  (0.00%)

#   Length distribution (train):
#     (   0,  128]:  68976  (98.5%)
#     ( 128,  256]:    933  (1.3%)
#     ( 256,  512]:     91  (0.1%)
#     ( 512,  768]:      0  (0.0%)
#     ( 768, 1024]:      0  (0.0%)
#     (1024, 1280]:      0  (0.0%)
#     (1280, 1752]:      0  (0.0%)