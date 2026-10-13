# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HuggingFace-coupled MoE expert-parallel (EP) helpers.

Capability detection and EP degree selection: deciding how a model distributes, from the model,
this GPU's memory and the knobs.
"""


def require_transformers(min_version: str, feature: str) -> None:
    """Fail loudly if the installed transformers is older than ``min_version`` (the floor for ``feature``).

    Callers enforce the version where the needed APIs first exist, so a too-old transformers errors
    clearly instead of silently mis-behaving (e.g. transformers 5.5 silently shards nothing for EP).
    See each call site for the empirically-determined floor.
    """
    import transformers
    from packaging.version import parse

    have = transformers.__version__
    if parse(have) < parse(min_version):
        raise RuntimeError(
            f"{feature} requires transformers >= {min_version}, but {have} is installed. "
            f"Upgrade with: pip install -U 'transformers>={min_version}'."
        )


def _ep_config(hf_config):
    """Return the config that carries the EP plan / expert count.

    For a plain MoE causal LM this is the root config. For a VLM whose nested text submodel
    is the MoE (e.g. ``KimiK25.text_config`` -> deepseek_v3, or ``NemotronH_Omni.llm_config``
    -> nemotron_h), this is that nested config.
    """
    if getattr(hf_config, "base_model_ep_plan", None):
        return hf_config
    for _attr in ("text_config", "llm_config"):
        sub = getattr(hf_config, _attr, None)
        if sub is not None and getattr(sub, "base_model_ep_plan", None):
            return sub
    # Fall through: no plan anywhere; let callers decide.
    return hf_config


def num_experts_from_config(hf_config) -> int:
    """Return the MoE expert count from an HF config (0 if the model is not MoE).

    Recurses into ``.text_config`` / ``.llm_config`` so VLMs whose nested text submodel is the MoE
    (Kimi-K2.7, Llama-4 multimodal, NemotronH-Omni, ...) report the LM's expert count.
    """
    for cfg in (
        hf_config,
        getattr(hf_config, "text_config", None),
        getattr(hf_config, "llm_config", None),
    ):
        if cfg is None:
            continue
        for attr in ("num_experts", "num_local_experts", "n_routed_experts", "num_routed_experts"):
            value = getattr(cfg, attr, None)
            if value:
                return int(value)
    return 0


def is_expert_parallel_capable(hf_config) -> bool:
    """True if the model can be expert-parallel sharded.

    Transformers exposes ``base_model_ep_plan`` on MoE configs whose experts support
    expert parallelism (e.g. gpt_oss, llama4; qwen3_moe in transformers>=5.7). Dense models
    and MoE configs without the plan return False -> use plain FSDP2. For VLMs the plan
    lives on ``text_config`` (the nested LM submodel), so check both.
    """
    return (
        bool(getattr(_ep_config(hf_config), "base_model_ep_plan", None))
        and num_experts_from_config(hf_config) > 0
    )


def choose_ep_dp_degrees(world_size: int, num_experts: int) -> tuple[int, int]:
    """Pick ``(expert_parallel, expert_data_parallel)`` degrees with ``EP * EDP == world_size``.

    EP must divide both ``num_experts`` (whole experts per rank) and ``world_size`` (a valid mesh
    factorization); we take the largest such EP, and each expert shard is then replicated over the
    remaining ``EDP = world_size // EP`` ranks. Falls back to ``(1, world_size)`` when the model is not
    MoE or no valid EP degree exists. Heuristic -- callers may override.
    """
    if num_experts <= 0 or world_size <= 1:
        return 1, world_size
    for ep in range(world_size, 0, -1):
        if world_size % ep == 0 and num_experts % ep == 0:
            return ep, world_size // ep
    return 1, world_size


def _estimate_dense_param_bytes(hf_config) -> "int | None":
    """Best-effort estimate of the DENSE (non-expert) parameter bytes from an HF config.

    embeddings + lm_head + per-layer attention/norms (+ the dense FFN when the model is NOT MoE, since a MoE's FFN is
    the expert weights). Returns ``None`` when the config lacks the fields to estimate (the caller then treats the dense
    part as 'fits'). A deliberately CONSERVATIVE upper bound (ignores GQA and tied embeddings) -- it exists to catch a
    dense part that clearly cannot be replicated, not to be exact.
    """
    cfg = getattr(hf_config, "text_config", None) or hf_config
    h = getattr(cfg, "hidden_size", None)
    n_layers = getattr(cfg, "num_hidden_layers", None)
    if not h or not n_layers:
        return None
    vocab = getattr(cfg, "vocab_size", None) or getattr(hf_config, "vocab_size", None) or 0
    inter = getattr(cfg, "intermediate_size", None) or 4 * h
    attn = 4 * h * h  # q,k,v,o (upper bound; ignores GQA head reduction)
    norms = 4 * h  # ~2 layernorms per layer
    ffn = (
        0 if is_expert_parallel_capable(hf_config) else 3 * h * inter
    )  # gated FFN; a MoE's FFN is expert
    per_layer = attn + norms + ffn
    embed = 2 * vocab * h  # embed + lm_head (untied upper bound)
    bytes_per = 2  # bf16/fp16 default (the pre-quant load dtype)
    return (embed + n_layers * per_layer) * bytes_per


def _dense_fits_per_gpu(
    hf_config, device_mem_bytes: "int | None" = None, headroom: float = 0.6
) -> bool:
    """Fit predicate for the dense knob.

    does the DENSE footprint fit ONE GPU (leaving ``headroom`` for activations + comm buffers + any EP-local experts)?
    CONSERVATIVE -- returns ``True`` ('fits' -> replicate / data-parallel) whenever it cannot estimate the dense size or
    the device memory, so the data-parallel default is preserved. This is the pluggable decision point of the auto-
    resolver; a heuristic or a recipe hint may substitute for it without changing callers.
    """
    dense_bytes = _estimate_dense_param_bytes(hf_config)
    if not dense_bytes:
        return True
    if device_mem_bytes is None:
        try:
            import torch

            device_mem_bytes = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory
        except Exception:
            return True
    return dense_bytes < headroom * device_mem_bytes


def _resolve_ep_size(hf_config, world_size: int, ep_size) -> int:
    """Resolve the expert-parallel degree.

    An explicit ``ep_size`` wins; ``auto`` -> the largest EP degree dividing both the expert count and
    ``world_size`` for an EP-capable model, else ``1``.
    """
    if ep_size not in (None, "auto", "0", 0):
        return int(ep_size)
    if world_size > 1 and is_expert_parallel_capable(hf_config):
        ep, _ = choose_ep_dp_degrees(world_size, num_experts_from_config(hf_config))
        return ep
    return 1


def decide_distribution_plan(
    hf_config,
    world_size: int,
    *,
    ep_size="auto",
    fsdp_size="auto",
    device_mem_bytes: "int | None" = None,
):
    """Decide how a HuggingFace MoE model is distributed for PTQ: a :class:`DistributionPlan`.

    Two knobs, each an explicit value or ``"auto"`` (explicit always wins):

    * ``ep_size`` -- ``auto`` -> the largest EP degree dividing both the expert count and
      ``world_size`` (``1`` for a model that cannot be expert-parallel).
    * ``fsdp_size`` -- ``auto`` -> ``1`` (replicate the dense part) when it fits one GPU, else
      ``world_size`` (FSDP2-shard it); ``"world"`` -> ``world_size``.

    Inside a :class:`~modelopt.torch.utils.distributed.ModelOptSession` the session decides it and
    keeps it as ``session.plan``: on a distributed run once, on rank 0, and every other rank adopts
    rank 0's decision instead of making its own (:meth:`~modelopt.torch.utils.distributed.ModelOptSession.decide`).
    The inputs are all read locally -- the env knobs, this GPU's memory, the config on this rank's
    filesystem -- so ranks deciding independently could disagree, and a disagreement surfaces as a
    hang at mesh creation rather than as an error. Collective: every rank must call it. If rank 0
    fails to decide, every rank raises. Without a session, this process decides alone.
    """
    from .distributed import get_active_mo_session

    def decide():
        return _decide_distribution_plan_on_rank0(
            hf_config,
            world_size,
            ep_size=ep_size,
            fsdp_size=fsdp_size,
            device_mem_bytes=device_mem_bytes,
        )

    session = get_active_mo_session(required=False)
    if not session.initialized:
        return decide()
    return session.decide("plan", decide)


def _decide_distribution_plan_on_rank0(
    hf_config, world_size: int, *, ep_size, fsdp_size, device_mem_bytes: "int | None"
):
    """The decision itself (see :func:`decide_distribution_plan`); run on rank 0 alone when distributed."""
    from .distributed import DistributionPlan

    ep = _resolve_ep_size(hf_config, world_size, ep_size)
    if fsdp_size in (None, "auto"):
        fsdp = 1 if _dense_fits_per_gpu(hf_config, device_mem_bytes) else world_size
    elif str(fsdp_size).lower() in ("world", "full"):
        fsdp = world_size
    else:
        fsdp = int(fsdp_size)
    fsdp = max(1, min(fsdp, world_size))
    return DistributionPlan(world_size=world_size, ep_size=ep, fsdp_size=fsdp)


# ---------------------------------------------------------------------------------------------------
# DP=EP all-to-all expert transport (`torch_a2a`)
#
# The throughput-correct MoE parallelism for calibration: dense+router run data-parallel (each rank its
# own batch), experts are EP-sharded, and tokens are routed to the expert-owning rank by an all-to-all
# (dispatch) / reduce (combine). The expert matmul stays on transformers' `_grouped_linear`, so
# ModelOpt's per-expert quantizers attach and collect amax over the *global* (dispatched) token pool.
# Transport is plain `torch.distributed.all_to_all_single` (NCCL) -- works cross-node at any rank count.
