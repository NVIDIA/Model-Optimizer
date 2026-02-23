import torch
from diffusers import DiffusionPipeline
import modelopt.torch.opt as mto
import numpy as np
import torch.nn.functional as F
import gc
import shutil
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DRY_RUN = False   # quick sanity check: 1 val pass + 1 train batch, then exit
NUM_STEPS = 30
PIXEL_H, PIXEL_W = 1024, 1024
VAE_SCALE_FACTOR = 8
PATCH_SIZE = 2
IMG_H = PIXEL_H // VAE_SCALE_FACTOR // PATCH_SIZE   # 64
IMG_W = PIXEL_W // VAE_SCALE_FACTOR // PATCH_SIZE   # 64
SEQ_LEN = IMG_H * IMG_W                             # 4096
IN_CHANNELS = 64
DEVICE = "cuda"

# per-sample img_shapes argument expected by the transformer
IMG_SHAPES_ONE = [[(1, IMG_H, IMG_W)]]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def calculate_mu(seq_len, base_seq=256, max_seq=8192, base_shift=0.5, max_shift=0.9):
    #returns, by default, 0.693548, for 1024x1024
    #m=y2-y1/x2-x1
    #b=y1-m*x1
    #return y=mx+b
    y2,y1 = max_shift, base_shift
    x, x2,x1 = seq_len, max_seq, base_seq
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b
# load the train and validation text embeddings 
train_embed_map = torch.load('weights/train_embeds.pt', weights_only=True) #40000, seq_len, embed dim = 3584
val_embed_map = torch.load('weights/val_embeds.pt', weights_only=True) #100, seq_len, embed dim = 3584

# input latents, noise seeds, step indices, target latents, timestep values
# 5714 × 6 files + 5716 file-7 = 40000 total, matching train_embed_map
teacher_train_latents_1 = torch.load('weights/teacher_train_latents_gpu1.pt', weights_only=True)
teacher_train_latents_2 = torch.load('weights/teacher_train_latents_gpu2.pt', weights_only=True)
teacher_train_latents_3 = torch.load('weights/teacher_train_latents_gpu3.pt', weights_only=True)
teacher_train_latents_4 = torch.load('weights/teacher_train_latents_gpu4.pt', weights_only=True)
teacher_train_latents_5 = torch.load('weights/teacher_train_latents_gpu5.pt', weights_only=True)
teacher_train_latents_6 = torch.load('weights/teacher_train_latents_gpu6.pt', weights_only=True)
teacher_train_latents_7 = torch.load('weights/teacher_train_latents_gpu7.pt', weights_only=True)
# 100 validation samples
teacher_val_latents = torch.load('weights/teacher_val_latents_gpu1.pt', weights_only=True)
# structure is:
# input_latents = [5714, 4096, 64]
# noise_seeds = [5714]
# step_indices = [5714]
# target_latents = [5714, 4096, 64]
# timestep_values = [5714]
# so 190 complete cycles (190 images × 30 steps) + a partial 14-step cycle at the end (hence the last step_indices value being 13)
    
student_pipeline = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-2512", torch_dtype=torch.bfloat16)
backbone = student_pipeline.transformer
mto.restore(backbone, "qwen_ptq/qwen-sym.pt")
student_pipeline.transformer = backbone
backbone.to("cuda")
BASE_LR = 1e-5                     # base LR at reference batch size
TRAIN_BATCH_SIZE=150
LEARNING_RATE = BASE_LR * (TRAIN_BATCH_SIZE / 8)  # linear scaling rule

student_scheduler = student_pipeline.scheduler
raw_sigmas = np.linspace(1.0, 1.0 / NUM_STEPS, NUM_STEPS)
mu = calculate_mu(SEQ_LEN)
student_scheduler.set_timesteps(sigmas=raw_sigmas.tolist(), mu=mu, device=DEVICE)
timesteps = student_scheduler.timesteps.clone()   # [30] — values passed to model
sigmas = student_scheduler.sigmas.clone()          # [31] — noise levels + terminal 0.0

#important for memory
backbone.gradient_checkpointing = True
backbone._gradient_checkpointing_func = (
    lambda module, *args: torch.utils.checkpoint.checkpoint(
        module.__call__, *args, use_reentrant=True,
    )
)
log("Gradient checkpointing enabled")

parameters = [p for p in backbone.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE, betas=(0.9, 0.999), foreach=False)


def pad_text_batch(embed_list, device="cuda"):
    """Pad a list of text embeddings to the same sequence length.

    encoder_hidden_states_mask is accepted by the transformer but never used —
    the attention processor ignores it and SDPA runs with attn_mask=None.
    txt_seq_lens is used only to size the RoPE frequency slice, not as a mask.

    Returns:
        enc_hs:       [N, max_txt_len, 3584]
        txt_seq_lens: list[int]  original lengths, for RoPE only
    """
    txt_seq_lens = [e.size(0) for e in embed_list]
    max_txt_len  = max(txt_seq_lens)
    padded = []
    for embed, orig_len in zip(embed_list, txt_seq_lens):
        e = embed.to(device=device, dtype=torch.bfloat16)
        pad_len = max_txt_len - orig_len
        if pad_len > 0:
            padding = torch.zeros(pad_len, e.size(1), device=device, dtype=torch.bfloat16)
            e = torch.cat([e, padding], dim=0)
        padded.append(e)
    enc_hs = torch.stack(padded, dim=0)  # [N, max_txt_len, 3584]
    return enc_hs, txt_seq_lens


def save_checkpoint(
    step, student_backbone, optimizer,
    train_log, val_log, OUTPUT_DIR, RUN_NAME, is_final=False,
):
    """Save model + training state for resumption."""
    model_path = f"{OUTPUT_DIR}/{RUN_NAME}_latest.pt"
    state_path = f"{OUTPUT_DIR}/{RUN_NAME}_latest_training_state.pt"
    log(f"  Saving checkpoint at step {step}...")
    mto.save(student_backbone, model_path)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "step": step,
        "train_log": train_log,
        "val_log": val_log,
    }, state_path)
    log(f"  Saved: {model_path} + {state_path} (step {step})")
    if is_final:
        final_model = f"{OUTPUT_DIR}/{RUN_NAME}.pt"
        shutil.copy2(model_path, final_model)
        log(f"  Final checkpoint: {final_model}")


dict_teacher_train_latents = {
    'teacher_train_latents_1': teacher_train_latents_1,
    'teacher_train_latents_2': teacher_train_latents_2,
    'teacher_train_latents_3': teacher_train_latents_3,
    'teacher_train_latents_4': teacher_train_latents_4,
    'teacher_train_latents_5': teacher_train_latents_5,
    'teacher_train_latents_6': teacher_train_latents_6,
    'teacher_train_latents_7': teacher_train_latents_7,
}

OUTPUT_DIR = "qad-sym"
RUN_NAME   = "qad"
PLOT_PATH  = "qad-sym/loss_curve.png"
TIMESTEP_GROUPS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]
train_log  = []
val_log    = []


def plot_losses():
    if not train_log:
        return
    n = len(train_log)
    fig, (ax_t, ax_v) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    x = list(range(1, n + 1))
    for lo, hi in TIMESTEP_GROUPS:
        y = []
        for e in train_log:
            tl = e["timestep_losses"]
            y.append(sum(tl[lo:hi]) / (hi - lo))
        ax_t.plot(x, y, "o-", markersize=2, label=f"t={lo}-{hi-1}")
    avg_y = [e["loss"] for e in train_log]
    ax_t.plot(x, avg_y, "k-", linewidth=2, alpha=0.5, label="avg")
    ax_t.set_ylabel("MSE loss")
    ax_t.set_title("Train loss by timestep group")
    ax_t.legend(fontsize=7, ncol=4)
    ax_t.grid(True, alpha=0.3)

    if val_log:
        vx = list(range(len(val_log)))
        vy = [e["val_loss"] for e in val_log]
        ax_v.plot(vx, vy, "s-", color="tab:orange", markersize=3)
    ax_v.set_xlabel("optimizer step")
    ax_v.set_ylabel("MSE loss")
    ax_v.set_title("Validation loss")
    ax_v.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    plt.close(fig)
    log(f"  Plot saved to {PLOT_PATH}")

# validation tensors (fixed across all epochs)
latents_val          = teacher_val_latents['input_latents']     # [100, 4096, 64]
latents_val_target   = teacher_val_latents['target_latents']    # [100, 4096, 64]
val_step_indices     = teacher_val_latents['step_indices']      # [100]
val_timestep_values  = teacher_val_latents['timestep_values']   # [100]
text_embeds_val      = list(val_embed_map.values())             # list of 100 tensors

# ── Initial validation pass ───────────────────────────────────────────────
log("Running initial validation...")
backbone.eval()
n_val = len(latents_val)
val_losses = []
with torch.no_grad():
    for vi in range(n_val):
        si = val_step_indices[vi].item()
        t_normed = (val_timestep_values[vi] / 1000.0).to(DEVICE, dtype=torch.bfloat16).unsqueeze(0)

        hs = latents_val[vi:vi+1].to(DEVICE, dtype=torch.bfloat16)
        embed = text_embeds_val[vi]
        enc_hs_v, txt_lens_v = pad_text_batch([embed], device=DEVICE)

        vel = backbone(
            hidden_states=hs,
            timestep=t_normed,
            encoder_hidden_states=enc_hs_v,
            img_shapes=IMG_SHAPES_ONE,
            txt_seq_lens=txt_lens_v,
            return_dict=False,
        )[0]

        student_scheduler._step_index = si
        y_val_pred = student_scheduler.step(
            vel, timesteps[si], hs, return_dict=False
        )[0]

        target = latents_val_target[vi:vi+1].to(DEVICE, dtype=torch.bfloat16)
        val_losses.append(F.mse_loss(y_val_pred, target).item())

val_loss = sum(val_losses) / len(val_losses)
val_log.append({"step": -1, "val_loss": val_loss})
log(f"  initial val_loss={val_loss:.6f}")

# ── Training loop ─────────────────────────────────────────────────────────
global_embed_offset = 0
train_batch_count = 0

for key, teacher_train_latents in dict_teacher_train_latents.items():
    latents_in       = teacher_train_latents['input_latents']   # [N, 4096, 64]
    latents_out_true = teacher_train_latents['target_latents']  # [N, 4096, 64]
    # timestep_values:
    #   1000.0, 982.25, 963.89, 944.88, 925.20, 904.80, 883.64, 861.68, 838.88, 815.19,
    #    790.54, 764.89, 738.18, 710.32, 681.25, 650.89, 619.15, 585.93, 551.13, 514.63,
    #    476.30, 436.00, 393.59, 348.88, 301.68, 251.79, 198.96, 142.92,  83.38,  20.00
    timestep_values_scheduler = teacher_train_latents['timestep_values'][:30]
    for i in range(0, len(latents_in), 150):
        input_latents = latents_in[i:i+150]

        # skip the trailing partial batch
        if len(input_latents) < 150:
            continue

        backbone.train()
        backbone.requires_grad_(True)

        # Build per-timestep sub-batches.
        # 150 latents = 5 images × 30 timesteps.
        # input_latents_batch_list[j] → 5 input  latents all at timestep j
        # train_embed_batch_list[j]   → 5 matching text embeddings
        # y_true_batch_list[j]        → 5 target latents at timestep j
        input_latents_batch_list = []
        train_embed_batch_list   = []
        y_true_batch_list        = []
        for j in range(30):
            in_b, emb_b, y_b = [], [], []
            for k in range(5):
                local_offset = k * 30 + j          # 0-149 within the 150-slice
                global_idx   = global_embed_offset + i + local_offset
                in_b.append(input_latents[local_offset])
                emb_b.append(train_embed_map[global_idx])
                y_b.append(latents_out_true[i + local_offset])
            input_latents_batch_list.append(in_b)
            train_embed_batch_list.append(emb_b)
            y_true_batch_list.append(y_b)

        # ── gradient accumulation over 30 timestep forward passes ────────────
        # Each pass is 5 samples; effective batch = 5 × 30 = 150.
        # We scale the loss by 1/NUM_STEPS so the update magnitude is the same
        # as doing a single forward on all 150 samples at once.
        # backward() frees the graph immediately (no retain_graph), so peak
        # activation memory equals one timestep forward pass, not 30.
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        accum_loss = 0.0
        timestep_losses = []

        for timestep_index, (latent_batch, embed_batch) in enumerate(
            zip(input_latents_batch_list, train_embed_batch_list)
        ):
            hidden_states = torch.stack(
                [lat.to(DEVICE, dtype=torch.bfloat16) for lat in latent_batch]
            )  # [5, 4096, 64]
            B = hidden_states.shape[0]
            t_val = (timestep_values_scheduler[timestep_index] / 1000.0).expand(B).to(DEVICE, dtype=torch.bfloat16)

            enc_hs, txt_seq_lens = pad_text_batch(embed_batch, device=DEVICE)
            img_shapes_batch = IMG_SHAPES_ONE * 5

            velocity = backbone(
                hidden_states=hidden_states,
                timestep=t_val,
                encoder_hidden_states=enc_hs,
                img_shapes=img_shapes_batch,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]  # [5, 4096, 64]

            dt = sigmas[timestep_index + 1] - sigmas[timestep_index]
            y_pred = hidden_states + dt * velocity  # Euler step, fully differentiable

            y_true = torch.stack(
                [t.to(DEVICE, dtype=torch.bfloat16) for t in y_true_batch_list[timestep_index]]
            )  # [5, 4096, 64]

            loss = F.mse_loss(y_pred, y_true)
            (loss / NUM_STEPS).backward()
            step_loss = loss.item()
            accum_loss += step_loss
            timestep_losses.append(step_loss)

            del hidden_states, enc_hs, velocity, y_pred, y_true, loss
            torch.cuda.empty_cache()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        train_batch_count += 1
        train_log.append({"step": i, "loss": accum_loss / NUM_STEPS, "timestep_losses": timestep_losses, "lr": LEARNING_RATE})
        log(f"[{key}] step {i}: loss={accum_loss / NUM_STEPS:.6f}")

        if i % 2 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            backbone.eval()
            n_val = len(latents_val)
            val_losses = []

            with torch.no_grad():
                for vi in range(n_val):
                    si = val_step_indices[vi].item()
                    t_normed = (val_timestep_values[vi] / 1000.0).to(DEVICE, dtype=torch.bfloat16).unsqueeze(0)

                    hs = latents_val[vi:vi+1].to(DEVICE, dtype=torch.bfloat16)
                    embed = text_embeds_val[vi]
                    enc_hs_v, txt_lens_v = pad_text_batch([embed], device=DEVICE)

                    vel = backbone(
                        hidden_states=hs,
                        timestep=t_normed,
                        encoder_hidden_states=enc_hs_v,
                        img_shapes=IMG_SHAPES_ONE,
                        txt_seq_lens=txt_lens_v,
                        return_dict=False,
                    )[0]

                    student_scheduler._step_index = si
                    y_val_pred = student_scheduler.step(
                        vel, timesteps[si], hs, return_dict=False
                    )[0]

                    target = latents_val_target[vi:vi+1].to(DEVICE, dtype=torch.bfloat16)
                    val_losses.append(F.mse_loss(y_val_pred, target).item())

            val_loss = sum(val_losses) / len(val_losses)
            val_log.append({"step": i, "val_loss": val_loss})
            log(f"  val_loss={val_loss:.6f}")

        plot_losses()
        save_checkpoint(i, backbone, optimizer, train_log, val_log, OUTPUT_DIR, RUN_NAME)

    global_embed_offset += len(latents_in)

save_checkpoint(train_batch_count, backbone, optimizer, train_log, val_log, OUTPUT_DIR, RUN_NAME, is_final=True)
