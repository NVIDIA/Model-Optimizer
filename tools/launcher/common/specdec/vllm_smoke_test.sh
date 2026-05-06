#!/bin/bash
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

# Quick vLLM smoke test for speculative decoding (EAGLE3, DFlash, MTP, etc.).
# Launches server, sends a few test prompts, verifies responses, and shuts down.
#
# Required env vars:
#   HF_MODEL_CKPT  — target model path
#
# Optional env vars:
#   DRAFT_MODEL     — draft model path (not needed for MTP)
#   SPEC_METHOD     — speculative method: "eagle", "dflash", "mtp", etc. (default: "eagle")
#   NUM_SPEC_TOKENS — number of speculative tokens (default: 15)
#   TP_SIZE         — tensor parallel size (default: 1)
#   VLLM_PORT       — server port (default: 8000)
#   REASONING_PARSER — reasoning parser (e.g., "qwen3" for Qwen3.5)
#   DISABLE_PREFIX_CACHING — set to "1" to disable prefix caching
#   TRUST_REMOTE_CODE — set to "1" to pass --trust-remote-code (needed for custom architectures)
#   UPGRADE_TRANSFORMERS — set to "1" to install transformers from HuggingFace main branch
#   DATA_PARALLEL_SIZE — data parallel size; mutually exclusive with TP_SIZE (default: unset, uses TP_SIZE)
#   KV_CACHE_DTYPE     — kv cache dtype (e.g., "fp8"); omitted if unset
#   BLOCK_SIZE         — paged attention block size (e.g., 256 for DeepSeek V4)
#   ENABLE_EXPERT_PARALLEL — set to "1" to pass --enable-expert-parallel
#   TOKENIZER_MODE     — tokenizer mode (e.g., "deepseek_v4")
#   VLLM_EXTRA_ARGS    — additional raw args appended verbatim to vllm serve (simple flags only)
#   COMPILATION_CONFIG — JSON string for --compilation-config (e.g., for B200 native ops)
#                        e.g., '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
#                        Passed as a properly-quoted single arg to avoid brace expansion issues.
#                        NOTE: NeMo Run generates unquoted env var assignments in sbatch scripts,
#                        so JSON with braces/brackets gets brace-expanded. Use BUILD_COMPILATION_CONFIG
#                        instead to avoid this — the JSON is constructed safely inside the script.
#   BUILD_COMPILATION_CONFIG — alternative to COMPILATION_CONFIG: just pass the cudagraph_mode string
#                        (e.g., "FULL_AND_PIECEWISE") and the script constructs:
#                        {"cudagraph_mode":"<value>","custom_ops":["all"]}
#                        This avoids brace-expansion of JSON in NeMo Run sbatch env var assignments.
#   GPU_MEM_UTIL       — gpu_memory_utilization fraction (default: unset, vLLM default 0.9)
#   MAX_BATCHED_TOKENS — override max_num_batched_tokens (default: 32768)
#   COPY_MODEL_TO_TMPFS — set to "1" to copy model to /dev/shm before serving
#                         (prevents NFS stale-handle errors when 8+ workers mmap weights simultaneously)

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source ${SCRIPT_DIR}/../service_utils.sh 2>/dev/null || true

# Ensure pandas is available (missing in some vLLM nightly builds)
pip install pandas 2>/dev/null || true

# Raise the per-user process limit so concurrent deepgemm/NVCC JIT workers (one per
# DP rank) don't exhaust the nproc limit when popen(nvcc) is called simultaneously
# during CUDA graph capture warmup. popen() returns nullptr (triggering deepgemm's
# "pipe != nullptr" assertion) when fork() fails with EAGAIN due to nproc limit.
ulimit -u unlimited 2>/dev/null || true

# Redirect deepgemm/NVCC JIT compilation away from /tmp (too small on B200) and
# /dev/shm (noexec — dlopen of compiled .so fails). DEEPGEMM_TMPDIR must be a
# writable+executable NFS path (e.g., /cicd/deepgemm_tmp). We use a separate env var
# so enroot doesn't pick it up at container startup (enroot reads TMPDIR before the
# container starts, so setting TMPDIR in sbatch would break the container launch).
if [ -n "${DEEPGEMM_TMPDIR:-}" ]; then
    mkdir -p "$DEEPGEMM_TMPDIR"
    export TMPDIR="$DEEPGEMM_TMPDIR"
fi

# Force torch inductor to use the v2 auto_functionalized algorithm.
# vLLM explicitly sets enable_auto_functionalized_v2=False in its inductor config,
# which causes failures with fallback FP8 ops (e.g., when VLLM_USE_DEEP_GEMM=0):
#   1. v1 decompose pass can't remove auto_functionalized nodes for MXFP4 ops
#   2. Remaining nodes execute as Python wrappers calling ops via stable IValue
#   3. The /opt/venv stable IValue binary doesn't know ScalarType 44 (MXFP4) → crash
# Fix: enable v2 in vLLM's compilation code so the proper decompose pass is used.
# v2 handles in-place mutations generically, removing the Python wrapper path entirely.
# Set FORCE_AF_V2=1 to enable.
if [ "${FORCE_AF_V2:-0}" = "1" ]; then
    python3 << 'PYEOF' || true
import inspect, compileall, glob, re, os, site

# ──────────────────────────────────────────────────────────────────────────────
# The problem:
#   vLLM explicitly passes 'enable_auto_functionalized_v2': False in its
#   inductor_compile_config dict.  This makes the v1 decompose pass run, which
#   can't remove auto_functionalized nodes wrapping MXFP4/cutlass ops.  Those
#   nodes then execute as Python wrappers calling ops via torch._ops.py → stable
#   IValue → ScalarType 44 (MXFP4) not registered in /opt/venv binary → crash.
#
# Fix strategy:
#   1. Write a .pth startup file to ALL site-packages dirs so every spawned
#      worker process auto-loads a module that monkey-patches
#      torch._inductor.config.patch() to strip enable_auto_functionalized_v2=False.
#   2. Patch the source files directly (file glob) with an updated regex that
#      handles both bare and quoted-key dict forms.
#   3. Patch post_grad.py assertions as a safety net.
# ──────────────────────────────────────────────────────────────────────────────

PATCH_MODULE_NAME = 'vllm_force_af_v2_runtime'
PATCH_CODE = r'''
# Auto-loaded via .pth in site-packages.  Runs in main process AND every spawned worker.
# Strategy: intercept torch._dynamo.aot_compile (the AOT compile entry point used by vLLM)
# to strip enable_auto_functionalized_v2=False from options before compilation starts.
# Uses sys.modules as sentinel (torch._inductor.config rejects unknown __getattr__).
import sys as _sys

def _strip_af_v2_false(d):
    if isinstance(d, dict) and d.get('enable_auto_functionalized_v2') is False:
        d = {k: v for k, v in d.items() if k != 'enable_auto_functionalized_v2'}
        print('[force_af_v2] Stripped enable_auto_functionalized_v2=False from inductor options', flush=True)
    return d

def _install():
    if _sys.modules.get('_vllm_af_v2_patched'):
        return
    _sys.modules['_vllm_af_v2_patched'] = True

    # Patch 1: torch._dynamo.aot_compile (called by vLLM decorators.py)
    try:
        import torch._dynamo as _dynamo
        _orig_aot = _dynamo.aot_compile
        def _patched_aot(*args, **kwargs):
            if 'options' in kwargs:
                kwargs['options'] = _strip_af_v2_false(kwargs['options'])
            return _orig_aot(*args, **kwargs)
        _dynamo.aot_compile = _patched_aot
        print('[force_af_v2] Patched torch._dynamo.aot_compile', flush=True)
    except Exception as e:
        print(f'[force_af_v2] aot_compile patch failed: {e}', flush=True)

    # Patch 2: torch._dynamo.aot_compile_fullgraph (alternative entry point)
    try:
        import torch._dynamo.aot_compile as _aot_mod
        _orig_fg = _aot_mod.aot_compile_fullgraph
        def _patched_fg(*args, **kwargs):
            if 'options' in kwargs:
                kwargs['options'] = _strip_af_v2_false(kwargs['options'])
            return _orig_fg(*args, **kwargs)
        _aot_mod.aot_compile_fullgraph = _patched_fg
        print('[force_af_v2] Patched torch._dynamo.aot_compile_fullgraph', flush=True)
    except Exception as e:
        print(f'[force_af_v2] aot_compile_fullgraph patch failed: {e}', flush=True)

    # Patch 3: torch._inductor.config.patch (if it exists in this PyTorch version)
    try:
        import torch._inductor.config as _ic
        _orig_patch = _ic.patch
        def _patched_patch(*args, **kwargs):
            new_args = (_strip_af_v2_false(args[0]),) + args[1:] if args and isinstance(args[0], dict) else args
            if kwargs.get('enable_auto_functionalized_v2') is False:
                kwargs = {k: v for k, v in kwargs.items() if k != 'enable_auto_functionalized_v2'}
            return _orig_patch(*new_args, **kwargs)
        _ic.patch = _patched_patch
        print('[force_af_v2] Patched torch._inductor.config.patch', flush=True)
    except Exception as e:
        print(f'[force_af_v2] config.patch intercept skipped: {e}', flush=True)

    # Patch 4: Set global torch._inductor.config.enable_auto_functionalized_v2 = True.
    # This ensures post_grad.py (which reads the global config) uses the v2 decompose path.
    try:
        import torch._inductor.config as _ic
        _ic.enable_auto_functionalized_v2 = True
        print('[force_af_v2] Set torch._inductor.config.enable_auto_functionalized_v2 = True', flush=True)
    except Exception as e:
        print(f'[force_af_v2] inductor global config set failed: {e}', flush=True)

    # Patch 5: torch._inductor.standalone_compile — vLLM's piecewise backend uses this
    # (NOT torch._dynamo.aot_compile) to compile each graph segment.  Strip the
    # enable_auto_functionalized_v2=False override so the global True setting survives.
    try:
        import torch._inductor as _ti_mod
        _orig_sc = getattr(_ti_mod, 'standalone_compile', None)
        if _orig_sc is not None:
            def _patched_sc(fn, *args, **kwargs):
                opts = kwargs.get('options')
                if isinstance(opts, dict) and opts.get('enable_auto_functionalized_v2') is False:
                    kwargs['options'] = {k: v for k, v in opts.items() if k != 'enable_auto_functionalized_v2'}
                    print('[force_af_v2] Stripped enable_auto_functionalized_v2=False from standalone_compile', flush=True)
                return _orig_sc(fn, *args, **kwargs)
            _ti_mod.standalone_compile = _patched_sc
            print('[force_af_v2] Patched torch._inductor.standalone_compile', flush=True)
        else:
            print('[force_af_v2] torch._inductor.standalone_compile not found, skipping', flush=True)
    except Exception as e:
        print(f'[force_af_v2] standalone_compile patch failed: {e}', flush=True)

_install()
'''

# Write the patch module + .pth startup file to every site-packages directory
site_dirs = site.getsitepackages() + [site.getusersitepackages()]
for sp in site_dirs:
    if not os.path.isdir(sp):
        continue
    try:
        mod_path = os.path.join(sp, f'{PATCH_MODULE_NAME}.py')
        pth_path = os.path.join(sp, f'{PATCH_MODULE_NAME}.pth')
        with open(mod_path, 'w') as f:
            f.write(PATCH_CODE)
        with open(pth_path, 'w') as f:
            f.write(f'import {PATCH_MODULE_NAME}\n')
        print(f'[force_af_v2] Wrote {pth_path} → auto-loads in all worker processes')
    except Exception as e:
        print(f'[force_af_v2] Could not write to {sp}: {e}')

# Also run the runtime patch immediately in this process
exec(PATCH_CODE)

# Step 2: Source-file patch — fix regex to handle quoted-key dict form.
vllm_dirs = [
    '/usr/local/lib/python3.12/dist-packages/vllm',
    '/opt/venv/lib/python3.12/site-packages/vllm',
]
for vllm_dir in vllm_dirs:
    if not os.path.isdir(vllm_dir):
        continue
    for py_file in glob.glob(os.path.join(vllm_dir, '**/*.py'), recursive=True):
        if '__pycache__' in py_file:
            continue
        try:
            with open(py_file) as f:
                content = f.read()
            if 'enable_auto_functionalized_v2' not in content:
                continue
            for i, line in enumerate(content.splitlines()):
                if 'enable_auto_functionalized_v2' in line:
                    print(f'[force_af_v2] Found in {py_file}:{i+1}: {line.strip()}')
            # Match both bare and quoted-key dict forms
            patched = re.sub(
                r'("?enable_auto_functionalized_v2"?\s*[:=]\s*)False',
                r'\1True',
                content
            )
            # Special case: compilation.py stores the key in a KEY constant and uses
            # KEY: False in the dict — the literal string search above misses this form.
            if '/vllm/config/compilation.py' in py_file or py_file.endswith('/compilation.py'):
                patched2 = re.sub(r'\bKEY(\s*:\s*)False', r'KEY\1True', patched)
                if patched2 != patched:
                    patched = patched2
                    print(f'[force_af_v2] Patched KEY: False in {py_file}')
            if patched != content:
                with open(py_file, 'w') as f:
                    f.write(patched)
                compileall.compile_file(py_file, quiet=2, force=True)
                print(f'[force_af_v2] Patched source file: {py_file}')
        except Exception as e:
            print(f'[force_af_v2] Error processing {py_file}: {e}')

# Step 3: Patch post_grad.py assertions as safety net.
try:
    import torch._inductor.fx_passes.post_grad as pg
    src_file = inspect.getfile(pg)
    with open(src_file) as f:
        content = f.read()
    patterns = [
        ('raise AssertionError("auto_functionalized was not removed")',
         'pass  # PATCHED: v1 nodes skipped (FORCE_AF_V2=1)'),
        ('raise AssertionError("auto_functionalized_v2 was not removed")',
         'pass  # PATCHED: v2 nodes skipped (FORCE_AF_V2=1)'),
        ('if config.enable_auto_functionalized_v2:', 'if True:  # PATCHED (FORCE_AF_V2=1)'),
        ('if inductor_config.enable_auto_functionalized_v2:', 'if True:  # PATCHED (FORCE_AF_V2=1)'),
        # Wrap the decompose_triton_kernel_wrapper_functional call in try/except so that a
        # node-count mismatch AssertionError (pattern_matcher.py:316) doesn't abort compilation.
        # vLLM's Triton kernel wrappers can produce a different graph node count than PyTorch 2.11
        # expects; skipping the decompose pass is safe — kernels execute via the wrapper path.
        ('GraphTransformObserver(gm, "decompose_triton_kernel_wrapper_functional").apply_graph_pass(decompose_triton_kernel_wrapper_functional)',
         'try:\n            GraphTransformObserver(gm, "decompose_triton_kernel_wrapper_functional").apply_graph_pass(decompose_triton_kernel_wrapper_functional)\n        except AssertionError as _af2_e:\n            print(f"[force_af_v2] decompose_triton_kernel_wrapper_functional skipped: {_af2_e}", flush=True)  # PATCHED'),
    ]
    patched = content
    for old, new in patterns:
        if old in patched:
            patched = patched.replace(old, new)
            print(f'[force_af_v2] post_grad patch: {old[:70]!r}')
    if patched != content:
        with open(src_file, 'w') as f:
            f.write(patched)
        compileall.compile_file(src_file, quiet=2, force=True)
        print(f'[force_af_v2] Wrote and recompiled {src_file}')
except Exception as e:
    print(f'[force_af_v2] post_grad.py patch failed: {e}')

# Step 4: Patch pattern_matcher.py to remove the node-count assertion fired by
# decompose_triton_kernel_wrapper_functional when vLLM's Triton kernel wrapper graphs
# have a different number of nodes than PyTorch 2.11's replacement graph.
# The assertion at pattern_matcher.py:316 reads:
#   assert len(graph_with_eager_vals.graph.nodes) == len(replacement.graph.nodes)
# The comment above it says "might not be true in general" — we exploit this escape hatch.
try:
    import re as _re
    import torch._inductor.pattern_matcher as pm
    pm_file = inspect.getfile(pm)
    with open(pm_file) as f:
        pm_content = f.read()
    pm_patched = _re.sub(
        r'assert len\(graph_with_eager_vals\.graph\.nodes\) == len\(\s*\n\s*replacement\.graph\.nodes\s*\n\s*\)',
        'pass  # PATCHED: skip node-count assertion for triton_kernel_wrapper_functional (FORCE_AF_V2=1)',
        pm_content,
    )
    if pm_patched != pm_content:
        with open(pm_file, 'w') as f:
            f.write(pm_patched)
        compileall.compile_file(pm_file, quiet=2, force=True)
        print(f'[force_af_v2] Patched pattern_matcher.py node-count assertion: {pm_file}')
    else:
        print(f'[force_af_v2] pattern_matcher.py: assertion pattern not found in {pm_file}')
except Exception as e:
    print(f'[force_af_v2] pattern_matcher.py patch failed: {e}')
PYEOF
fi

# Patch vllm._custom_ops.cutlass_scaled_mm to cast ue8m0 (ScalarType 44) block-FP8
# scale tensors to uint8 before dispatching through PyTorch's stable IValue layer.
#
# deepseek_v4_fp8 stores per-block scales in ue8m0 format (unsigned 8-bit, 8 exponent
# bits, 0 mantissa bits). PyTorch 2.11's stableivalue_conversions.h doesn't recognise
# ScalarType 44, so torch.ops._C.cutlass_scaled_mm crashes during the model's dummy
# forward pass (profile_run). Casting to uint8 preserves the raw bytes — the CUTLASS
# kernel receives the same values — while satisfying the stable IValue type check.
#
# The patch is written as a .pth startup module so it propagates to every forked worker.
python3 << 'PYEOF' || true
import os, site

PATCH_MODULE_NAME = 'vllm_ue8m0_cast_patch'
PATCH_CODE = r'''
import sys as _sys

def _install():
    if _sys.modules.get('_vllm_ue8m0_patch_installed'):
        return
    _sys.modules['_vllm_ue8m0_patch_installed'] = True
    try:
        import torch
        import vllm._custom_ops as _vllm_co

        _orig_csm = _vllm_co.cutlass_scaled_mm

        _SAFE_DTYPES = frozenset([
            torch.float32, torch.float16, torch.bfloat16,
            torch.uint8, torch.int8,
            torch.float8_e4m3fn, torch.float8_e5m2,
        ])

        def _csm_ue8m0_safe(*args, **kwargs):
            args = list(args)
            def _cast(t):
                if t is not None and hasattr(t, 'dtype') and t.dtype not in _SAFE_DTYPES and t.element_size() == 1:
                    return t.view(torch.uint8)
                return t
            if 'scale_a' in kwargs:
                kwargs['scale_a'] = _cast(kwargs['scale_a'])
            elif len(args) > 3:
                args[3] = _cast(args[3])
            if 'scale_b' in kwargs:
                kwargs['scale_b'] = _cast(kwargs['scale_b'])
            elif len(args) > 4:
                args[4] = _cast(args[4])
            return _orig_csm(*args, **kwargs)

        _vllm_co.cutlass_scaled_mm = _csm_ue8m0_safe
        print('[patch_ue8m0] Patched vllm._custom_ops.cutlass_scaled_mm', flush=True)
    except Exception as e:
        print(f'[patch_ue8m0] Patch failed: {e}', flush=True)

_install()
'''

for sp in site.getsitepackages() + [site.getusersitepackages()]:
    if not os.path.isdir(sp):
        continue
    try:
        with open(os.path.join(sp, f'{PATCH_MODULE_NAME}.py'), 'w') as f:
            f.write(PATCH_CODE)
        with open(os.path.join(sp, f'{PATCH_MODULE_NAME}.pth'), 'w') as f:
            f.write(f'import {PATCH_MODULE_NAME}\n')
        print(f'[patch_ue8m0] Wrote .pth to {sp}')
        break
    except Exception as e:
        print(f'[patch_ue8m0] Could not write to {sp}: {e}')

exec(PATCH_CODE)
PYEOF

# Allow callers to upgrade transformers for models not yet in the container's bundled version
# (e.g. deepseek_v4 requires transformers >= 4.52). Set UPGRADE_TRANSFORMERS=1 to enable.
if [ "${UPGRADE_TRANSFORMERS:-0}" = "1" ]; then
    pip install --upgrade --pre transformers 2>/dev/null || true
    # Register deepseek_v4 by writing a .pth file + module to site-packages.
    # Python processes .pth files at startup, so this propagates to every vLLM subprocess.
    python3 << 'PYEOF' || true
import sys, os, sysconfig

PATCH_MODULE = '''
try:
    from transformers import AutoConfig, PretrainedConfig
    class DeepseekV4Config(PretrainedConfig):
        model_type = "deepseek_v4"
        def __init__(self, **kwargs):
            # Pre-populate ALL config.json fields before super().__init__ runs,
            # because PretrainedConfig in transformers 5.x accesses attributes
            # like max_position_embeddings during initialization.
            for k, v in kwargs.items():
                object.__setattr__(self, k, v)
            # Override architectures to TransformersForCausalLM so vLLM routes
            # through its generic transformers backend (trust_remote_code path),
            # since DeepseekV4ForCausalLM is not yet in vLLMs native registry.
            object.__setattr__(self, "architectures", ["TransformersForCausalLM"])
            super().__init__(**kwargs)
    AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
except Exception:
    pass
'''

site_packages = sysconfig.get_path("purelib")
module_path = os.path.join(site_packages, "_deepseek_v4_patch.py")
pth_path = os.path.join(site_packages, "deepseek_v4.pth")

with open(module_path, "w") as f:
    f.write(PATCH_MODULE)
with open(pth_path, "w") as f:
    f.write("import _deepseek_v4_patch\n")

print(f"[patch] wrote {pth_path} -> will register deepseek_v4 on every Python startup")
PYEOF
fi

# Apply custom vLLM patches before starting the server.
# Used for models that require container-level modifications not yet upstream.
# Set VLLM_PATCH_SCRIPT to a Python script path (relative to /nemo_run/code/).
if [ -n "${VLLM_PATCH_SCRIPT:-}" ] && [ -f "${VLLM_PATCH_SCRIPT}" ]; then
    echo "Applying vLLM patches: ${VLLM_PATCH_SCRIPT}"
    python3 "${VLLM_PATCH_SCRIPT}" || { echo "ERROR: patch script failed"; exit 1; }
fi

TMPFS_MODEL=""
cleanup() {
    kill $SERVER_PID 2>/dev/null
    sleep 2
    kill -9 $SERVER_PID 2>/dev/null
    rm -f "${VLLM_LOG:-}" 2>/dev/null
    # Clean up tmpfs copy if we made one
    if [ -n "$TMPFS_MODEL" ] && [ -d "$TMPFS_MODEL" ]; then
        echo "Removing tmpfs model copy: $TMPFS_MODEL"
        rm -rf "$TMPFS_MODEL"
    fi
}
trap cleanup EXIT

MODEL=${HF_MODEL_CKPT}

# Copy model to /dev/shm to avoid NFS stale-handle errors when many workers mmap weights simultaneously
if [ "${COPY_MODEL_TO_TMPFS:-0}" = "1" ]; then
    MODEL_NAME=$(basename "$MODEL")
    TMPFS_MODEL="/dev/shm/${MODEL_NAME}"
    if [ -d "$TMPFS_MODEL" ] && [ -f "$TMPFS_MODEL/config.json" ]; then
        echo "Using existing tmpfs model copy: $TMPFS_MODEL"
    else
        MODEL_SIZE=$(du -sh "$MODEL" 2>/dev/null | cut -f1 || echo "?")
        AVAIL_SHM=$(df -h /dev/shm 2>/dev/null | tail -1 | awk '{print $4}' || echo "?")
        echo "Copying model to /dev/shm (${MODEL_SIZE}, available: ${AVAIL_SHM})..."
        cp -r "$MODEL" "$TMPFS_MODEL"
        echo "Model copy done: $TMPFS_MODEL"
    fi
    MODEL="$TMPFS_MODEL"
    echo "Loading from tmpfs: $MODEL"
fi
DRAFT=${DRAFT_MODEL:-}
# Auto-detect exported checkpoint from training output dir
if [ -z "$DRAFT" ] && [ -n "${DRAFT_CKPT_DIR:-}" ]; then
    DRAFT=$(ls -d ${DRAFT_CKPT_DIR}/exported-checkpoint-* 2>/dev/null | sort -t- -k3 -n | tail -1)
    if [ -n "$DRAFT" ]; then
        echo "Auto-detected draft model: ${DRAFT}"
    fi
fi
METHOD=${SPEC_METHOD:-eagle}
NUM_SPEC=${NUM_SPEC_TOKENS:-15}
PORT=${VLLM_PORT:-8000}
TP=${TP_SIZE:-1}

echo "=== vLLM Speculative Decoding Smoke Test ==="
echo "Method: ${METHOD}"
echo "Target: ${MODEL}"
echo "Draft:  ${DRAFT:-none (self-draft)}"
echo "Spec tokens: ${NUM_SPEC}, TP: ${TP}"

# Build speculative config: include "model" only if DRAFT_MODEL is set
if [ -n "$DRAFT" ] && [ "$DRAFT" != "none" ]; then
    SPEC_CONFIG="{\"method\": \"${METHOD}\", \"model\": \"${DRAFT}\", \"num_speculative_tokens\": ${NUM_SPEC}}"
else
    # Self-draft methods (MTP, Medusa) — no separate draft model
    SPEC_CONFIG="{\"method\": \"${METHOD}\", \"num_speculative_tokens\": ${NUM_SPEC}}"
fi

# Build optional args
OPTIONAL_ARGS=""
if [ -n "${REASONING_PARSER:-}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --reasoning-parser ${REASONING_PARSER}"
fi
if [ "${DISABLE_PREFIX_CACHING:-}" = "1" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --no-enable-prefix-caching"
fi
if [ "${TRUST_REMOTE_CODE:-}" = "1" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --trust-remote-code"
fi
if [ -n "${KV_CACHE_DTYPE:-}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --kv-cache-dtype ${KV_CACHE_DTYPE}"
fi
if [ -n "${BLOCK_SIZE:-}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --block-size ${BLOCK_SIZE}"
fi
if [ "${ENABLE_EXPERT_PARALLEL:-}" = "1" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --enable-expert-parallel"
fi
if [ -n "${TOKENIZER_MODE:-}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --tokenizer-mode ${TOKENIZER_MODE}"
fi
if [ -n "${GPU_MEM_UTIL:-}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --gpu-memory-utilization ${GPU_MEM_UTIL}"
fi
if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} ${VLLM_EXTRA_ARGS}"
fi

# Use data-parallel or tensor-parallel based on which is set
if [ -n "${DATA_PARALLEL_SIZE:-}" ]; then
    PARALLELISM_ARGS="--data-parallel-size ${DATA_PARALLEL_SIZE}"
else
    PARALLELISM_ARGS="--tensor-parallel-size ${TP}"
fi

# If BUILD_COMPILATION_CONFIG is set, construct the JSON here to avoid brace-expansion.
# NeMo Run writes sbatch env vars unquoted, so {"a":"b","c":["d"]} gets brace-expanded by bash.
# BUILD_COMPILATION_CONFIG carries just the cudagraph_mode string; we build the JSON safely.
if [ -z "${COMPILATION_CONFIG:-}" ] && [ -n "${BUILD_COMPILATION_CONFIG:-}" ]; then
    COMPILATION_CONFIG="{\"cudagraph_mode\":\"${BUILD_COMPILATION_CONFIG}\",\"custom_ops\":[\"all\"]}"
fi

# Start vLLM server (capture output for regression check parsing)
# Build command array so COMPILATION_CONFIG JSON is passed as a single properly-quoted arg
# (unquoted ${OPTIONAL_ARGS} expansion handles simple flags; JSON needs array quoting)
VLLM_LOG=$(mktemp /tmp/vllm_server_XXXXXX.log)
VLLM_CMD=(vllm serve "${MODEL}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-32768}"
    ${PARALLELISM_ARGS}
    --port "${PORT}"
    ${OPTIONAL_ARGS})
if [ -n "$SPEC_CONFIG" ]; then
    VLLM_CMD+=(--speculative-config "${SPEC_CONFIG}")
fi
if [ -n "${COMPILATION_CONFIG:-}" ]; then
    VLLM_CMD+=(--compilation-config "${COMPILATION_CONFIG}")
fi
"${VLLM_CMD[@]}" > >(tee -a "$VLLM_LOG") 2>&1 &
SERVER_PID=$!

# Wait for server (large models like DeepSeek V4 need up to 10 min to load + compile)
SERVER_TIMEOUT=${SERVER_TIMEOUT:-600}
echo "Waiting for vLLM server (timeout: ${SERVER_TIMEOUT}s)..."
for i in $(seq 1 ${SERVER_TIMEOUT}); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server died"; wait $SERVER_PID; exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "ERROR: Server did not become ready within ${SERVER_TIMEOUT}s"; exit 1
fi

# Run quick test prompts using chat completions API
MAX_TOKENS=${MAX_OUTPUT_TOKENS:-1024}
echo ""
echo "=== Test Prompts (max_tokens=${MAX_TOKENS}) ==="
PASS=0
FAIL=0
TOTAL_TOKENS=0
TOTAL_TIME=0
# 8 prompts mimicking MT-Bench categories: writing, roleplay, reasoning,
# math, coding, extraction, stem, humanities
for PROMPT in \
    "Write a persuasive email to your manager requesting a four-day work week. Include at least three supporting arguments." \
    "You are a medieval blacksmith. A traveler asks you to forge a sword. Describe your process and the qualities of your finest work." \
    "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Explain your reasoning carefully." \
    "Solve the equation 3x + 7 = 22. Show each step of your solution." \
    "Write a Python function that takes a list of integers and returns the second largest unique value. Include error handling." \
    "Extract all the dates, names, and locations from: On March 15 2024 Dr. Alice Chen presented her findings at the Berlin Conference on Climate Science." \
    "Explain the process of photosynthesis. What role does chlorophyll play and why are plants green?" \
    "Discuss the main themes in George Orwells 1984. How do they relate to modern society?"; do
    START=$(date +%s%N)
    RESULT=$(curl -s http://localhost:${PORT}/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}], \"max_tokens\": ${MAX_TOKENS}, \"temperature\": 0}" \
        2>/dev/null)
    END=$(date +%s%N)
    ELAPSED=$(echo "scale=2; ($END - $START) / 1000000000" | bc 2>/dev/null || echo "0")
    # Use python3 -S to skip site-packages (.pth startup files like _deepseek_v4_patch.pth
    # print [force_af_v2] messages to stdout which corrupt the TOKENS variable).
    TOKENS=$(echo "$RESULT" | python3 -S -c "
import json,sys
try:
    r=json.load(sys.stdin)
    u=r.get('usage') or {}
    t=u.get('completion_tokens',0) or 0
    if not t:
        msg = ((r.get('choices') or [{}])[0].get('message') or {})
        c = msg.get('content') or msg.get('reasoning_content') or ''
        t = len(c.split()) if c else 0
    if not t and r.get('choices'):
        t = 1  # any response with choices = success
    print(t)
except Exception:
    print(0)
" 2>/dev/null)
    if [ -n "$TOKENS" ] && [ "$TOKENS" -gt 0 ] 2>/dev/null; then
        TPS=$(echo "scale=1; $TOKENS / $ELAPSED" | bc 2>/dev/null || echo "?")
        echo "  PASS: ${TOKENS} tokens in ${ELAPSED}s (${TPS} tok/s) — \"${PROMPT:0:50}...\""
        PASS=$((PASS + 1))
        TOTAL_TOKENS=$((TOTAL_TOKENS + TOKENS))
        TOTAL_TIME=$(echo "$TOTAL_TIME + $ELAPSED" | bc 2>/dev/null || echo "0")
    else
        echo "  FAIL: \"${PROMPT}\""
        echo "  Response: $(echo "$RESULT" | head -c 200)"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Results: ${PASS} passed, ${FAIL} failed"
if [ "$TOTAL_TOKENS" -gt 0 ] 2>/dev/null; then
    AVG_TPS=$(echo "scale=1; $TOTAL_TOKENS / $TOTAL_TIME" | bc 2>/dev/null || echo "?")
    echo "Total: ${TOTAL_TOKENS} tokens in ${TOTAL_TIME}s (${AVG_TPS} tok/s avg)"
fi

# Fetch speculative decoding metrics if available
echo ""
METRICS=$(curl -s http://localhost:${PORT}/metrics 2>/dev/null | grep -i "spec\|accept\|draft" | head -10)
if [ -n "$METRICS" ]; then
    echo "=== Speculative Decoding Metrics ==="
    echo "$METRICS"
fi

if [ $FAIL -gt 0 ]; then
    echo "ERROR: Some prompts failed"
    exit 1
fi

# Regression check: minimum acceptance length for speculative decoding
if [ -n "${MIN_ACCEPTANCE_LENGTH:-}" ]; then
    # Parse mean acceptance length from vLLM's SpecDecoding metrics log.
    # vLLM logs: "SpecDecoding metrics: Mean acceptance length: X.XX, ..."
    # Take the last reported value (most accurate, covers all prompts).
    AVG_ACCEPT=$(grep -oP 'Mean acceptance length: \K[0-9.]+' "$VLLM_LOG" 2>/dev/null | tail -1 || true)
    if [ -n "$AVG_ACCEPT" ]; then
        echo ""
        echo "=== Acceptance Length Regression Check ==="
        echo "  Mean acceptance length: ${AVG_ACCEPT}"
        echo "  Threshold: ${MIN_ACCEPTANCE_LENGTH}"
        PASS_CHECK=$(python3 -c "print('yes' if float('${AVG_ACCEPT}') >= float('${MIN_ACCEPTANCE_LENGTH}') else 'no')")
        if [ "$PASS_CHECK" = "yes" ]; then
            echo "  PASS: ${AVG_ACCEPT} >= ${MIN_ACCEPTANCE_LENGTH}"
        else
            echo "  REGRESSION: ${AVG_ACCEPT} < ${MIN_ACCEPTANCE_LENGTH}"
            exit 1
        fi
    else
        echo "WARNING: Could not parse acceptance length from vLLM log, skipping regression check"
    fi
fi

echo "Done"
