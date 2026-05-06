# Credentials Setup

Tokens and registry credentials that ModelOpt workflows need across local and cluster environments. Not SLURM-specific — referenced from PTQ, deployment, evaluation, and slurm-setup skills.

## Check what's already set first

Before configuring anything, check what the user already has — many of these are likely in place from prior `hf auth login`, `docker login`, or previous SLURM work. Skip any section below for which credentials are already present.

```bash
# HF token: env var or persisted from `hf auth login`
[ -n "$HF_TOKEN" ] && echo "✓ HF_TOKEN set in env"
[ -s ~/.cache/huggingface/token ] && echo "✓ HF token at ~/.cache/huggingface/token (from 'hf auth login')"

# Docker / NGC registry credentials
grep -qE '"(nvcr\.io|https://index\.docker\.io)"' ~/.docker/config.json 2>/dev/null && echo "✓ Docker login present"

# Enroot / pyxis credentials (on cluster login node, for SLURM users)
grep -qE '^machine nvcr\.io ' ~/.config/enroot/.credentials 2>/dev/null && echo "✓ Enroot NGC entry present"
```

For remote clusters, run the same checks via SSH (`ssh <cluster-login> '<check>'`) — credentials live on the cluster, not your workstation.

## HuggingFace token (`HF_TOKEN`)

Required for gated models (e.g., Llama, Mistral, some Nemotron variants) and gated datasets (e.g., GPQA, HLE).

Generate at <https://huggingface.co/settings/tokens>. Two persistence options (you can use either or both):

1. **`hf auth login`** (recommended for interactive use) — stores the token at `~/.cache/huggingface/token`. The HF Python client picks it up automatically; `transformers`, `datasets`, and the `hf` CLI all read this file without needing `HF_TOKEN` in the env.

   ```bash
   pip install -U huggingface_hub
   hf auth login   # paste the token interactively
   ```

2. **Environment variable** (good for scripts, CI, and remote sessions):

   ```bash
   export HF_TOKEN=hf_...
   ```

   Persist in `~/.bashrc` or a project-local `.env` file. `HF_TOKEN` takes precedence when both are present.

## NGC API key (for `nvcr.io`)

Required for pulling NGC images (`nvcr.io/nvidia/pytorch:...`, `nvcr.io/nvidia/vllm:...`) via Docker, `srun --container-image`, or enroot.

Generate at <https://ngc.nvidia.com/setup/api-key>.

### Docker

```bash
docker login nvcr.io -u '$oauthtoken' -p <NGC_API_KEY>
```

### Enroot (SLURM / pyxis)

Add an entry to `~/.config/enroot/.credentials` on the cluster. The file may already hold credentials for other registries — **append rather than overwrite**:

```bash
mkdir -p ~/.config/enroot
CREDS=~/.config/enroot/.credentials
touch "$CREDS"
grep -q '^machine nvcr.io ' "$CREDS" || \
    echo 'machine nvcr.io login $oauthtoken password <NGC_API_KEY>' >> "$CREDS"
chmod 600 "$CREDS"
```

> **Note**: `$oauthtoken` is a **literal string** required by NGC, not a shell variable. Do not replace it and do not let your shell expand it — the single quotes above keep it literal.

Without this, `srun --container-image=nvcr.io/...` fails with `401 Unauthorized` when the compute node tries to pull.

## Docker Hub login

Only needed if you hit rate limits pulling public images:

```bash
docker login
```

## Summary

| Credential | Used for | Set via |
|---|---|---|
| `HF_TOKEN` | Gated HF models / datasets | Env var (`export HF_TOKEN=...`) or `.env` |
| NGC API key | `nvcr.io` image pulls | `docker login` or `~/.config/enroot/.credentials` |
| Docker Hub | Rate-limited public image pulls | `docker login` |
