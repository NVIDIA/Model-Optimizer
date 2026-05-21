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

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

source ${SCRIPT_DIR}/../service_utils.sh

###################################################################################################
# Usage:
#   query.sh --model MODEL [SERVE_ARGS...] -- [QUERY_ARGS...]
#
# Launches vllm serve with the given model, waits for it to be ready,
# then runs common/query.py against the server.
#
# --model MODEL is required and is consumed by this script. It is used as the
# positional model argument for both vllm serve and common/query.py.
#
# Remaining arguments are split on "--":
#   - Args BEFORE "--" are appended to the vllm serve command (SERVE_ARGS).
#   - Args AFTER  "--" are passed to common/query.py (QUERY_ARGS).
#   - If "--" is absent, all remaining args go to common/query.py.
#
# Environment variables (optional, set by Slurm):
#   SLURM_ARRAY_TASK_ID     Used to shard query.py work across array jobs.
#   SLURM_ARRAY_TASK_COUNT  Total number of array tasks for sharding.
#
# vLLM notes:
#   - vLLM manages GPU distribution internally; run with ntasks_per_node: 1
#     in slurm_config and pass --tensor-parallel-size to match gpus_per_node.
#   - NVFP4 models require vllm/vllm-openai:v0.15.0+ on Blackwell GPUs.
#   - Use --trust-remote-code for models with custom architectures (e.g. Kimi).
#
# In a pipeline YAML task config:
#   args:
#     - --model /hf-local/Qwen/Qwen3-8B  # required
#     - --tensor-parallel-size 4           # vllm serve args (before --)
#     - --max-num-seqs 32
#     - --trust-remote-code
#     - --                                 # separator
#     - --data /hf-local/dataset           # query.py args (after --)
#     - --save /scratchspace/data
#   slurm_config:
#     ntasks_per_node: 1                   # vLLM is single-process
#     gpus_per_node: 4
###################################################################################################

# Ensure pandas is available (missing in some vLLM nightly builds)
pip install pandas 2>/dev/null || true

export OPENAI_API_KEY="token-abc123"

if [ -z ${SLURM_ARRAY_TASK_ID} ]; then
    TASK_ID=0
else
    echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID}"
    TASK_ID=${SLURM_ARRAY_TASK_ID}
fi

if [ -z ${SLURM_ARRAY_TASK_COUNT} ]; then
    TASK_COUNT=1
else
    echo "SLURM_ARRAY_TASK_COUNT ${SLURM_ARRAY_TASK_COUNT}"
    TASK_COUNT=${SLURM_ARRAY_TASK_COUNT}
fi

# Parse --model and split remaining args on "--".
# --model is consumed here; args before "--" go to vllm serve, args after go to query.py.
MODEL=""
SERVE_EXTRA_ARGS=()
QUERY_ARGS=(--shard-id-begin ${TASK_ID} --shard-id-step ${TASK_COUNT})
past_separator=false
skip_next=false

for arg in "$@"; do
    if $skip_next; then
        MODEL="$arg"
        skip_next=false
    elif [ "$arg" = "--model" ]; then
        skip_next=true
    elif [ "$arg" = "--" ]; then
        past_separator=true
    elif [ "$past_separator" = false ]; then
        SERVE_EXTRA_ARGS+=("$arg")
    else
        QUERY_ARGS+=("$arg")
    fi
done

# B200: raise per-user process limit so concurrent deepgemm/NVCC JIT workers don't exhaust
# nproc when popen(nvcc) is called simultaneously across DP ranks during CUDA graph capture.
ulimit -u unlimited 2>/dev/null || true

# B200: redirect deepgemm NVCC JIT to a writable+executable NFS path. /tmp (container tmpfs)
# is too small; /dev/shm is noexec. Use DEEPGEMM_TMPDIR (not TMPDIR) so enroot doesn't read
# it at container startup before the container starts.
if [ -n "${DEEPGEMM_TMPDIR:-}" ]; then
    mkdir -p "$DEEPGEMM_TMPDIR"
    export TMPDIR="$DEEPGEMM_TMPDIR"
fi

# Copy model to /dev/shm to avoid NFS stale-handle errors when many workers mmap weights
# simultaneously during a long data synthesis run. Reuses existing copy if present.
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

# Force torch inductor to use the v2 auto_functionalized algorithm.
# vLLM explicitly sets enable_auto_functionalized_v2=False in its inductor config,
# which causes failures with fallback FP8 ops (e.g., when VLLM_USE_DEEP_GEMM=0):
#   aten::as_strided() Expected a value of type 'List[int]' for argument 'stride'
#   but instead found type 'list'.
# Set FORCE_AF_V2=1 to enable. Ported from common/specdec/vllm_smoke_test.sh.
if [ "${FORCE_AF_V2:-0}" = "1" ]; then
    python3 << 'PYEOF' || true
import inspect, compileall, glob, re, os, site

PATCH_MODULE_NAME = 'vllm_force_af_v2_runtime'
PATCH_CODE = r'''
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

    try:
        import torch._inductor.config as _ic
        _ic.enable_auto_functionalized_v2 = True
        print('[force_af_v2] Set torch._inductor.config.enable_auto_functionalized_v2 = True', flush=True)
    except Exception as e:
        print(f'[force_af_v2] inductor global config set failed: {e}', flush=True)

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
    except Exception as e:
        print(f'[force_af_v2] standalone_compile patch failed: {e}', flush=True)

_install()
'''

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
        print(f'[force_af_v2] Wrote {pth_path} -> auto-loads in all worker processes')
    except Exception as e:
        print(f'[force_af_v2] Could not write to {sp}: {e}')

exec(PATCH_CODE)

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
            patched = re.sub(
                r'("?enable_auto_functionalized_v2"?\s*[:=]\s*)False',
                r'\1True',
                content
            )
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
        ('GraphTransformObserver(gm, "decompose_triton_kernel_wrapper_functional").apply_graph_pass(decompose_triton_kernel_wrapper_functional)',
         'try:\n            GraphTransformObserver(gm, "decompose_triton_kernel_wrapper_functional").apply_graph_pass(decompose_triton_kernel_wrapper_functional)\n        except AssertionError as _af2_e:\n            print(f"[force_af_v2] decompose_triton_kernel_wrapper_functional skipped: {_af2_e}", flush=True)  # PATCHED'),
    ]
    patched = content
    for old, new in patterns:
        if old in patched:
            patched = patched.replace(old, new)
    if patched != content:
        with open(src_file, 'w') as f:
            f.write(patched)
        compileall.compile_file(src_file, quiet=2, force=True)
        print(f'[force_af_v2] Wrote and recompiled {src_file}')
except Exception as e:
    print(f'[force_af_v2] post_grad.py patch failed: {e}')

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
        print(f'[force_af_v2] Patched pattern_matcher.py: {pm_file}')
except Exception as e:
    print(f'[force_af_v2] pattern_matcher.py patch failed: {e}')
PYEOF
fi

# vLLM is single-process: GPU parallelism is handled internally via --tensor-parallel-size.
# No MPI multi-rank logic needed; this script always runs as a single task.
vllm serve \
    ${MODEL} \
    "${SERVE_EXTRA_ARGS[@]}" \
    &
SERVER_PID=$!


# Wait for server to start up by polling the health endpoint
echo "Waiting for server to start..."
MAX_WAIT=${VLLM_STARTUP_TIMEOUT:-600}
WAITED=0
while true; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died during startup"
        wait $SERVER_PID 2>/dev/null
        exit 1
    fi
    response=$(curl -s -o /dev/null -w "%{http_code}" "http://$(hostname -f):8000/health" || true)
    if [ "$response" -eq 200 ]; then
        echo "Server is up! (waited ${WAITED}s)"
        break
    fi
    WAITED=$((WAITED + 10))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server failed to start within ${MAX_WAIT}s"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    echo "Server not ready yet (${WAITED}/${MAX_WAIT}s), retrying in 10 seconds..."
    sleep 10
done

pip3 install -q datasets openai 2>/dev/null || true
echo "Running: python3 common/query.py http://localhost:8000/v1 ${MODEL} ${QUERY_ARGS[*]}"
python3 common/query.py http://localhost:8000/v1 "${MODEL}" "${QUERY_ARGS[@]}"
echo "Main process exit"

kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

exit 0
