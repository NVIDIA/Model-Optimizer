# Credentials Setup

Tokens and registry credentials that ModelOpt workflows need across local and cluster environments. Not SLURM-specific — referenced from PTQ, deployment, evaluation, and slurm-setup skills.

## HuggingFace token (`HF_TOKEN`)

Required for gated models (e.g., Llama, Mistral, some Nemotron variants) and gated datasets (e.g., GPQA, HLE).

Generate at <https://huggingface.co/settings/tokens>, then export:

```bash
export HF_TOKEN=hf_...
```

Persist in `~/.bashrc` or a project-local `.env` file. For remote clusters, check whether the cluster's shell config already sets it: `ssh <cluster-login> 'env | grep -c HF_TOKEN'`.

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
