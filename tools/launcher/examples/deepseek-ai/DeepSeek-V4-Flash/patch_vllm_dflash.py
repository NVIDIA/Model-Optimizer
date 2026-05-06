#!/usr/bin/env python3
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
"""Patch vLLM for DeepSeek-V4-Flash + DFlash speculative decoding.

Applies all patches needed to run DeepSeek-V4-Flash as the DFlash target model in
vLLM >= 0.1.dev15833. Idempotent — safe to run multiple times; each patch checks a
sentinel string before applying.

Patches applied:
  P0  speculative.py         — add "deepseek_v4" to DFlash allowed target models
  P1  deepseek_v4.py         — aux hidden state collection in inner model forward loop
  P2  deepseek_v4.py         — return aux hidden states alongside hidden_states
  P3  deepseek_v4.py         — add set/get EAGLE3 interface methods to outer model
  P4  gpu_model_runner.py    — allow hasattr-based EAGLE3 interface check
  P5  kv_cache_utils.py      — leftover KV cache group for unassigned DFlash draft layers
  P6  eagle.py               — skip missing layers in validate_same_kv_cache_group
  P7  kv_cache_utils.py      — get_uniform_page_size: return min instead of asserting ==1
  P8  kv_cache_utils.py      — _max_memory_usage_bytes_from_groups: handle mixed page sizes
  P9  gpu_model_runner.py    — _reshape_kv_cache_tensors: allow heterogeneous page sizes
  P10 flash_attn.py          — treat fp8_ds_mla as float8_e4m3fn in get_fp8_dtype_for_flashattn
  P11 sparse_attn_indexer.py — bypass fp8_fp4_paged_mqa_logits (smem overflow on H100 w/ MLA)

----------------------------------------------------------------------------
Upstream PR strategy
----------------------------------------------------------------------------
The patches split into two groups with different upstream paths:

GROUP A — Core model support (P0, P1–P3, P4, P10): ~50 lines, PR-ready
  These belong in a single vLLM PR: "Add DFlash speculative decoding support for
  DeepSeek-V4 target model."
    P0   One-liner: register "deepseek_v4" in the DFlash allowed-target list.
    P1–P3 Add the aux hidden state interface (set/get_eagle3_aux_hidden_state_layers)
          to DeepseekV4ForCausalLM — the same interface EAGLE3 requires.
    P4   Make gpu_model_runner.py accept the hasattr-based interface in addition to
          the formal supports_eagle3() check, so new models don't need to subclass.
    P10  Fix fp8_ds_mla → float8_e4m3fn mapping in get_fp8_dtype_for_flashattn.

GROUP B — KV cache heterogeneity (P5–P9): dissolve into proper draft architecture
  These patches work around the fact that vLLM doesn't yet know that DFlash
  cross-attention layers (which attend to the target's hidden states, not a separate
  draft KV cache) are KV-cache-free. A proper upstream implementation would classify
  those layers correctly at the KV cache spec level, eliminating the "leftover"
  layers, the mixed page-size mismatch, and the reshape assert — all without needing
  the five individual patches.

P11 — Kernel fallback (sparse_attn_indexer.py): needs a kernel-level fix
  The fp8_fp4_paged_mqa_logits DeepGEMM kernel exceeds H100 shared memory limits
  (228 KB) when block_size=256 and MLA head_dim=576 bytes. The bypass here attends
  all cached pages instead of running top-k selection — correct but suboptimal.
  The right upstream fix is either:
    (a) tile the DeepGEMM kernel so it fits in smem for large page sizes, or
    (b) add an explicit runtime smem check in sparse_attn_indexer.py with a
        documented fallback path (attend-all) and a one-time warning.
  Option (b) is essentially this patch, just made explicit rather than silent.
----------------------------------------------------------------------------
"""
import pathlib
import re
import sys

VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")


def _delete_pyc(stem: str) -> None:
    for pyc in VLLM.rglob(f"{stem}*.pyc"):
        pyc.unlink(missing_ok=True)


def _patch_file(path: pathlib.Path, old: str, new: str, sentinel: str, label: str) -> bool:
    src = path.read_text()
    if sentinel in src:
        print(f"{label}: already patched")
        return True
    if old not in src:
        print(f"WARNING: {label}: pattern not found — skipping")
        return False
    path.write_text(src.replace(old, new, 1))
    _delete_pyc(path.stem)
    print(f"{label}: OK")
    return True


# ---------------------------------------------------------------------------
# P0: speculative.py — add deepseek_v4 to DFlash allowed target models
# ---------------------------------------------------------------------------
_spec = VLLM / "config" / "speculative.py"
_spec_src = _spec.read_text()
_p0_sentinel = '"deepseek_v4"'
if _p0_sentinel in _spec_src:
    print("P0 speculative.py: already patched")
else:
    _target = None
    for i, line in enumerate(_spec_src.splitlines()):
        if '"deepseek_v3"' in line:
            _target = i
            break
    if _target is None:
        print("WARNING: P0 speculative.py: deepseek_v3 line not found — skipping")
    else:
        lines = _spec_src.splitlines(keepends=True)
        indent = len(lines[_target]) - len(lines[_target].lstrip())
        lines.insert(_target + 1, " " * indent + '"deepseek_v4",\n')
        _spec.write_text("".join(lines))
        _delete_pyc("speculative")
        print("P0 speculative.py: added deepseek_v4 after deepseek_v3 — OK")

# ---------------------------------------------------------------------------
# P1-P3: deepseek_v4.py — EAGLE3/DFlash aux hidden state interface
# ---------------------------------------------------------------------------
_v4 = VLLM / "model_executor" / "models" / "deepseek_v4.py"
_v4_src = _v4.read_text()

if "aux_hidden_state_layers" in _v4_src:
    print("P1-P3 deepseek_v4.py: already patched")
else:
    _old1 = (
        "        for layer in islice(self.layers, self.start_layer, self.end_layer):\n"
        "            hidden_states = layer(\n"
        "                hidden_states,\n"
        "                positions,\n"
        "                input_ids,\n"
        "            )\n"
    )
    _new1 = (
        "        if not hasattr(self, 'aux_hidden_state_layers'):\n"
        "            self.aux_hidden_state_layers = ()\n"
        "        aux_hidden_states = []\n"
        "        for idx, layer in enumerate(\n"
        "            islice(self.layers, self.start_layer, self.end_layer),\n"
        "            start=self.start_layer,\n"
        "        ):\n"
        "            if idx in self.aux_hidden_state_layers:\n"
        "                aux_hidden_states.append(hidden_states.mean(dim=-2))\n"
        "            hidden_states = layer(\n"
        "                hidden_states,\n"
        "                positions,\n"
        "                input_ids,\n"
        "            )\n"
    )
    _old2 = (
        "        hidden_states = self.norm(hidden_states)\n"
        "        return hidden_states\n"
        "\n"
        "    def load_weights("
    )
    _new2 = (
        "        hidden_states = self.norm(hidden_states)\n"
        "        if aux_hidden_states:\n"
        "            return hidden_states, aux_hidden_states\n"
        "        return hidden_states\n"
        "\n"
        "    def load_weights("
    )
    _eagle3_methods = (
        "    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:\n"
        "        self.model.aux_hidden_state_layers = layers\n"
        "\n"
        "    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:\n"
        "        num_layers = len(self.model.layers)\n"
        "        return (2, num_layers // 2, num_layers - 3)\n"
        "\n"
    )
    ok = True
    if _old1 not in _v4_src:
        print("WARNING: P1 deepseek_v4.py: inner loop pattern not found"); ok = False
    if _old2 not in _v4_src:
        print("WARNING: P2 deepseek_v4.py: return pattern not found"); ok = False
    if ok:
        _v4_src = _v4_src.replace(_old1, _new1, 1)
        _v4_src = _v4_src.replace(_old2, _new2, 1)
        # Insert methods before the first of these anchors in DeepseekV4ForCausalLM
        _outer_idx = _v4_src.find("class DeepseekV4ForCausalLM(")
        _outer = _v4_src[_outer_idx:]
        _inserted = False
        for _anchor in ["    def compute_logits(", "    def forward(", "    def load_weights("]:
            if _anchor in _outer:
                _outer = _outer.replace(_anchor, _eagle3_methods + _anchor, 1)
                _inserted = True
                break
        if not _inserted:
            print("WARNING: P3 deepseek_v4.py: no anchor found for methods")
        else:
            _v4_src = _v4_src[:_outer_idx] + _outer
            _v4.write_text(_v4_src)
            _delete_pyc("deepseek_v4")
            print("P1-P3 deepseek_v4.py: OK")

# ---------------------------------------------------------------------------
# P4: gpu_model_runner.py — allow hasattr-based EAGLE3 interface
# ---------------------------------------------------------------------------
_gmr = VLLM / "v1" / "worker" / "gpu_model_runner.py"
_patch_file(
    _gmr,
    old=(
        "                    if not supports_eagle3(self.get_model()):\n"
        "                        raise RuntimeError(\n"
        "                            \"Model does not support EAGLE3 interface but \"\n"
        "                            \"aux_hidden_state_outputs was requested\"\n"
        "                        )"
    ),
    new=(
        "                    _m = self.get_model()  # _eagle3_hasattr_patch\n"
        "                    if not (supports_eagle3(_m) or\n"
        "                            (hasattr(_m, 'set_aux_hidden_state_layers') and\n"
        "                             hasattr(_m, 'get_eagle3_aux_hidden_state_layers'))):\n"
        "                        raise RuntimeError(\n"
        "                            \"Model does not support EAGLE3 interface but \"\n"
        "                            \"aux_hidden_state_outputs was requested\"\n"
        "                        )"
    ),
    sentinel="_eagle3_hasattr_patch",
    label="P4 gpu_model_runner.py EAGLE3 check",
)

# ---------------------------------------------------------------------------
# P5: kv_cache_utils.py — leftover KV cache group for unassigned DFlash layers
# ---------------------------------------------------------------------------
_kvu = VLLM / "v1" / "core" / "kv_cache_utils.py"
_patch_file(
    _kvu,
    old=(
        "    elif grouped_specs := group_and_unify_kv_cache_specs(kv_cache_spec):\n"
        "        # DeepseekV4 case: All layers need the same number of token slots,\n"
        "        # yet some layers are full attention while others are sliding window\n"
        "        # attention in different sizes. Need to group layers into multiple\n"
        "        # UniformTypeKVCacheSpecs.\n"
        "        kv_cache_groups = _get_kv_cache_groups_uniform_groups(grouped_specs)\n"
        "        _annotate_eagle_groups_deepseek_v4(vllm_config, kv_cache_spec, kv_cache_groups)\n"
        "        return kv_cache_groups"
    ),
    new=(
        "    elif grouped_specs := group_and_unify_kv_cache_specs(kv_cache_spec):\n"
        "        # DeepseekV4 case: All layers need the same number of token slots,\n"
        "        # yet some layers are full attention while others are sliding window\n"
        "        # attention in different sizes. Need to group layers into multiple\n"
        "        # UniformTypeKVCacheSpecs.\n"
        "        kv_cache_groups = _get_kv_cache_groups_uniform_groups(grouped_specs)\n"
        "        _annotate_eagle_groups_deepseek_v4(vllm_config, kv_cache_spec, kv_cache_groups)\n"
        "        # _dflash_leftover_patch: collect unassigned layers (e.g., Qwen3 GQA draft)\n"
        "        # and group them by page_size_bytes so each group is uniform.\n"
        "        _assigned = set(n for g in kv_cache_groups for n in g.layer_names)\n"
        "        _leftover = {k: v for k, v in kv_cache_spec.items() if k not in _assigned}\n"
        "        if _leftover:\n"
        "            print(f'kv_cache: creating leftover group for {len(_leftover)} unassigned layers')\n"
        "            from collections import defaultdict as _dd\n"
        "            _by_size = _dd(list)\n"
        "            for _ln, _sp in _leftover.items():\n"
        "                _by_size[_sp.page_size_bytes].append(_ln)\n"
        "            for _lnames in _by_size.values():\n"
        "                _g = {k: _leftover[k] for k in _lnames}\n"
        "                kv_cache_groups += create_kv_cache_group_specs(_g, [_lnames])\n"
        "        return kv_cache_groups"
    ),
    sentinel="_dflash_leftover_patch",
    label="P5 kv_cache_utils.py leftover group",
)

# ---------------------------------------------------------------------------
# P6: eagle.py — skip missing layers in validate_same_kv_cache_group
# ---------------------------------------------------------------------------
_eagle = VLLM / "v1" / "spec_decode" / "eagle.py"
_patch_file(
    _eagle,
    old=(
        "        assert (\n"
        "            len(\n"
        "                set(\n"
        "                    [\n"
        "                        kv_cache_groups[layer_name]\n"
        "                        for layer_name in self._draft_attn_layer_names\n"
        "                    ]\n"
        "                )\n"
        "            )\n"
        "            == 1\n"
        "        ), \"All drafting layers should belong to the same kv cache group\""
    ),
    new=(
        "        # _dflash_group_patch: skip layers missing from kv_cache_groups (e.g., DFlash cross-attn)\n"
        "        _dgroup = set(\n"
        "            kv_cache_groups[n] for n in self._draft_attn_layer_names\n"
        "            if n in kv_cache_groups\n"
        "        )\n"
        "        assert len(_dgroup) <= 1, \"All drafting layers should belong to the same kv cache group\""
    ),
    sentinel="_dflash_group_patch",
    label="P6 eagle.py validate_same_kv_cache_group",
)

# ---------------------------------------------------------------------------
# P7-P8: kv_cache_utils.py — mixed page size support
# ---------------------------------------------------------------------------
_kvu_src = _kvu.read_text()
if "_dflash_page_size_patch" in _kvu_src:
    print("P7-P8 kv_cache_utils.py mixed page sizes: already patched")
else:
    _changed = False
    # P7: get_uniform_page_size — return min(page_sizes) instead of asserting len == 1
    _old7 = (
        "    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}\n"
        "    assert len(page_sizes) == 1\n"
        "    return page_sizes.pop()"
    )
    _new7 = (
        "    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}\n"
        "    # _dflash_page_size_patch: allow mixed page sizes for DFlash heterogeneous draft\n"
        "    if not page_sizes:\n"
        "        return 0\n"
        "    return min(page_sizes)"
    )
    if _old7 in _kvu_src:
        _kvu_src = _kvu_src.replace(_old7, _new7, 1)
        _changed = True
        print("P7 kv_cache_utils.py get_uniform_page_size: OK")
    else:
        print("WARNING: P7 kv_cache_utils.py get_uniform_page_size: pattern not found")

    # P8: _max_memory_usage_bytes_from_groups — handle mixed page sizes
    _old8 = (
        "    # General case: group_size pools, each shared by one layer per group\n"
        "    # Memory = group_size * page_size * blocks_for_max_len\n"
        "    group_size = max(len(group.layer_names) for group in kv_cache_groups)\n"
        "    page_size = get_uniform_page_size(\n"
        "        [group.kv_cache_spec for group in kv_cache_groups]\n"
        "    )\n"
        "    blocks_needed = sum(\n"
        "        cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), page_size)\n"
        "        for group in kv_cache_groups\n"
        "    )\n"
        "\n"
        "    return group_size * page_size * blocks_needed"
    )
    _new8 = (
        "    # General case: group_size pools, each shared by one layer per group\n"
        "    # Memory = group_size * page_size * blocks_for_max_len\n"
        "    # _dflash_page_size_patch: handle mixed page sizes (DFlash heterogeneous draft)\n"
        "    _ps_set = set(g.kv_cache_spec.page_size_bytes for g in kv_cache_groups)\n"
        "    if len(_ps_set) == 1:\n"
        "        group_size = max(len(group.layer_names) for group in kv_cache_groups)\n"
        "        page_size = _ps_set.pop()\n"
        "        blocks_needed = sum(\n"
        "            cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), page_size)\n"
        "            for group in kv_cache_groups\n"
        "        )\n"
        "        return group_size * page_size * blocks_needed\n"
        "    else:\n"
        "        # Mixed page sizes: sum per-group memory independently\n"
        "        return sum(\n"
        "            group.kv_cache_spec.max_memory_usage_bytes(vllm_config)\n"
        "            for group in kv_cache_groups\n"
        "        )"
    )
    if _old8 in _kvu_src:
        _kvu_src = _kvu_src.replace(_old8, _new8, 1)
        _changed = True
        print("P8 kv_cache_utils.py _max_memory_usage_bytes_from_groups: OK")
    else:
        print("WARNING: P8 kv_cache_utils.py _max_memory_usage_bytes_from_groups: pattern not found")

    if _changed:
        _kvu.write_text(_kvu_src)
        _delete_pyc("kv_cache_utils")

# ---------------------------------------------------------------------------
# P9: gpu_model_runner.py — heterogeneous page sizes in _reshape_kv_cache_tensors
# ---------------------------------------------------------------------------
_patch_file(
    _gmr,
    old=(
        "                raw_tensor = kv_cache_raw_tensors[layer_name]\n"
        "                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0\n"
        "                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes"
    ),
    new=(
        "                raw_tensor = kv_cache_raw_tensors[layer_name]\n"
        "                # _dflash_reshape_patch: tolerate non-multiple sizes (heterogeneous draft)\n"
        "                _pg = kv_cache_spec.page_size_bytes\n"
        "                if raw_tensor.numel() % _pg != 0:\n"
        "                    _nb = max(1, raw_tensor.numel() // _pg)\n"
        "                    raw_tensor = raw_tensor[:_nb * _pg]\n"
        "                    kv_cache_raw_tensors[layer_name] = raw_tensor\n"
        "                num_blocks = raw_tensor.numel() // _pg"
    ),
    sentinel="_dflash_reshape_patch",
    label="P9 gpu_model_runner.py _reshape_kv_cache_tensors",
)

# ---------------------------------------------------------------------------
# P10: flash_attn.py — treat fp8_ds_mla as float8_e4m3fn
# ---------------------------------------------------------------------------
_fa = VLLM / "v1" / "attention" / "backends" / "flash_attn.py"
_fa_src = _fa.read_text()
if "_dflash_fp8_ds_mla_patch" in _fa_src:
    print("P10 flash_attn.py fp8_ds_mla: already patched")
else:
    _fa_new = re.sub(
        r"([ \t]*)raise ValueError\(f\"Unrecognized FP8 dtype: \{kv_cache_dtype\}\"\)",
        lambda m: (
            m.group(1) + "# _dflash_fp8_ds_mla_patch: fp8_ds_mla is e4m3fn stored by the compressor\n"
            + m.group(1) + "if kv_cache_dtype == \"fp8_ds_mla\":\n"
            + m.group(1) + "    import torch as _t; return _t.float8_e4m3fn\n"
            + m.group(1) + "raise ValueError(f\"Unrecognized FP8 dtype: {kv_cache_dtype}\")"
        ),
        _fa_src,
        count=1,
    )
    if _fa_new != _fa_src:
        _fa.write_text(_fa_new)
        _delete_pyc("flash_attn")
        print("P10 flash_attn.py fp8_ds_mla: OK")
    else:
        print("WARNING: P10 flash_attn.py: raise ValueError pattern not found — skipping")

# ---------------------------------------------------------------------------
# P11: sparse_attn_indexer.py — bypass fp8_fp4_paged_mqa_logits (smem overflow)
#
# DeepSeek V4 Flash MLA uses block_size=256, head_dim=576 bytes/token. The
# fp8_fp4_paged_mqa_logits DeepGEMM kernel exceeds H100 shared memory limits
# (228 KB) with this configuration. Replace the logits + top_k path with a
# direct fill of topk_indices from the block_table, which attends to all cached
# pages and avoids the large intermediate logits tensor.
# ---------------------------------------------------------------------------
_sai = VLLM / "model_executor" / "layers" / "sparse_attn_indexer.py"
_patch_file(
    _sai,
    old=(
        "        logits = fp8_fp4_paged_mqa_logits(\n"
        "            (padded_q_quant_cast, padded_q_scale),\n"
        "            kv_cache,\n"
        "            weights[:num_padded_tokens],\n"
        "            seq_lens,\n"
        "            decode_metadata.block_table,\n"
        "            decode_metadata.schedule_metadata,\n"
        "            max_model_len=max_model_len,\n"
        "            clean_logits=False,\n"
        "        )\n"
        "        num_rows = logits.shape[0]\n"
        "        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]\n"
        "\n"
        "        if current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):\n"
        "            workspace_manager = current_workspace_manager()\n"
        "            (topk_workspace,) = workspace_manager.get_simultaneous(\n"
        "                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),\n"
        "            )\n"
        "            torch.ops._C.persistent_topk(\n"
        "                logits,\n"
        "                seq_lens,\n"
        "                topk_indices,\n"
        "                topk_workspace,\n"
        "                topk_tokens,\n"
        "                attn_metadata.max_seq_len,\n"
        "            )\n"
        "        else:\n"
        "            if current_platform.is_xpu():\n"
        "                ops.top_k_per_row_decode(\n"
        "                    logits,\n"
        "                    next_n,\n"
        "                    seq_lens,\n"
        "                    topk_indices,\n"
        "                    num_rows,\n"
        "                    logits.stride(0),\n"
        "                    logits.stride(1),\n"
        "                    topk_tokens,\n"
        "                )\n"
        "            else:\n"
        "                torch.ops._C.top_k_per_row_decode(\n"
        "                    logits,\n"
        "                    next_n,\n"
        "                    seq_lens,\n"
        "                    topk_indices,\n"
        "                    num_rows,\n"
        "                    logits.stride(0),\n"
        "                    logits.stride(1),\n"
        "                    topk_tokens,\n"
        "                )"
    ),
    new=(
        "        # _dflash_smem_fallback_patch: bypass fp8_fp4_paged_mqa_logits + top_k\n"
        "        # Directly fill topk_indices with block_table entries (attend to all pages).\n"
        "        # This avoids the (num_tokens x num_total_blocks) logits tensor and the\n"
        "        # fp8_fp4_paged_mqa_logits kernel that overflows H100 smem with MLA block_size=256.\n"
        "        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]\n"
        "        topk_indices.fill_(-1)\n"
        "        _bt_flat = (\n"
        "            decode_metadata.block_table[:batch_size]\n"
        "            .unsqueeze(1)\n"
        "            .expand(-1, next_n, -1)\n"
        "            .reshape(num_padded_tokens, -1)\n"
        "        )\n"
        "        _max_bl = min(_bt_flat.shape[1], topk_tokens)\n"
        "        topk_indices[:, :_max_bl] = _bt_flat[:, :_max_bl]"
    ),
    sentinel="_dflash_smem_fallback_patch",
    label="P11 sparse_attn_indexer.py smem fallback",
)

print("All patches done!")
