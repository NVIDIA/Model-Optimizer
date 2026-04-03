# ModelOpt Slack Bot

Centralized Slack bot for ModelOpt agent skills (PTQ, deployment, evaluation). Shared bot with per-user authentication, isolated job directories, and remote cluster support.

## Architecture

```
Slack                      Bot Server                         Per-Job Execution
┌──────────┐  event      ┌──────────────────┐               ┌──────────────────┐
│ User A   │ ──────────> │  Slack Bot       │  fresh copy   │ Job dir (User A) │
│ @modelopt│             │  (slack-bolt)    │ ────────────> │ Model-Optimizer/ │
│ quantize │ <────────── │                  │  claude --cwd │ .claude/skills/  │
│ Qwen3... │  response   │  ┌────────────┐  │               │ clusters.yaml    │
└──────────┘             │  │ UserStore  │  │               └──────────────────┘
                         │  │ JobManager │  │
┌──────────┐             │  │ KeyStore   │  │               ┌──────────────────┐
│ User B   │ ──────────> │  └────────────┘  │  fresh copy   │ Job dir (User B) │
│ @modelopt│             │                  │ ────────────> │ Model-Optimizer/ │
│ deploy   │ <────────── │                  │               │ .claude/skills/  │
└──────────┘             └──────────────────┘               └──────────────────┘
```

**Key design:**
- Single shared upstream repo (read-only)
- Each job gets a fresh copy (no `.git`) — agent can freely modify code
- User's `clusters.yaml` is injected into each job copy
- Claude CLI runs with user's own auth credentials

## Setup

### 1. Create Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and create a new app
2. Enable **Socket Mode** (Settings > Socket Mode > Enable)
3. Generate an **App-Level Token** with `connections:write` scope → `SLACK_APP_TOKEN` (xapp-...)
4. Add **Bot Token Scopes** (OAuth & Permissions):
   - `app_mentions:read`
   - `chat:write`
   - `commands`
   - `files:write`
   - `im:history`
   - `im:read`
   - `im:write`
5. **Subscribe to Events** (Event Subscriptions):
   - `app_mention`
   - `message.im`
6. **Slash Commands**: Create `/modelopt` command
7. Install the app to your workspace
8. Copy the **Bot User OAuth Token** → `SLACK_BOT_TOKEN` (xoxb-...)

### 2. Install Dependencies

```bash
cd slack-bot
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your tokens and settings
```

### 4. Create Data Directory

```bash
mkdir -p /data/modelopt
```

### 5. Run

```bash
source .env
python bot.py
```

## User Onboarding

First-time users are guided through setup automatically:

1. **Auth choice**: shared team key, own API key, or browser OAuth
2. **Cluster config** (optional): interactive setup of remote SLURM cluster

## Commands

| Command | Description |
|---------|-------------|
| `@modelopt <prompt>` | Run a ModelOpt task |
| `/modelopt setup` | Onboard (auth + cluster config) |
| `/modelopt set-key <key>` | Set own API key (DM only) |
| `/modelopt add-cluster` | Configure a remote cluster |
| `/modelopt clusters` | List configured clusters |
| `/modelopt jobs` | List recent jobs |
| `/modelopt cleanup` | Remove old job directories |
| `/modelopt status` | Show your current status |
| `/modelopt help` | Show available commands |

## Data Layout

```
/data/modelopt/
  keys/                              ← encrypted key store
  users/<slack_uid>/
    auth.json                        ← auth method
    clusters.yaml                    ← user's cluster config
    jobs/
      ptq-20260318-143022/           ← fresh repo copy per job
      ptq-20260318-160511/
```

## Examples

```
@modelopt quantize Qwen3-0.6B with nvfp4
@modelopt quantize and evaluate Llama-3.1-8B with fp8 on cw-dfw
@modelopt deploy ./my-checkpoint with vLLM
```
