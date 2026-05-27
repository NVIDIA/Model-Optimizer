#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

source ${SCRIPT_DIR}/../../service_utils.sh

pip install "speculators<0.5.0" --no-deps 2>/dev/null || true
pip install datasets 2>/dev/null || true

# vLLM API compatibility: speculators 0.4.0.1 uses Request(eos_token_id=...) which
# was removed in newer vLLM. Patch to remove the unsupported kwarg.
python3 -c "
import site, os
for d in site.getsitepackages():
    path = os.path.join(d, 'speculators', 'data_generation', 'vllm_hidden_states_generator.py')
    if not os.path.exists(path):
        continue
    with open(path) as f:
        c = f.read()
    old = '                eos_token_id=self.tokenizer.eos_token_id,\n'
    if old in c:
        with open(path, 'w') as f:
            f.write(c.replace(old, ''))
        print('Patched vllm_hidden_states_generator.py: removed eos_token_id from Request()')
    else:
        print('vllm_hidden_states_generator.py: eos_token_id already removed or not found')
    break
" 2>/dev/null || true

# Pydantic 2.13 compatibility: speculators.ReloadableBaseModel.reload_schema() calls
# model_rebuild(force=True) without a types_namespace. In pydantic 2.13+, inherited
# torch.dtype annotations from transformers.PretrainedConfig cannot be resolved in
# subclass modules that don't import torch. Fix by injecting torch into the namespace.
python3 -c "
import site, os
for d in site.getsitepackages():
    path = os.path.join(d, 'speculators', 'utils', 'pydantic_utils.py')
    if not os.path.exists(path):
        continue
    with open(path) as f:
        c = f.read()
    old = 'cls.model_rebuild(force=True)'
    new = 'import torch as _torch; cls.model_rebuild(force=True, _types_namespace={\"torch\": _torch})'
    if old in c and new not in c:
        with open(path, 'w') as f:
            f.write(c.replace(old, new))
        print('Patched pydantic_utils.py: model_rebuild now passes torch namespace')
    else:
        print('pydantic_utils.py already patched or pattern not found')
    break
" 2>/dev/null || true

# vLLM scheduler compatibility: speculators 0.4.0.1 generate() loop never calls
# scheduler.update_from_output(), so KV blocks are never freed and the scheduler
# stops admitting new requests after MAX_NUM_SEQS=32. Fix by injecting the call
# inside the loop, plus aborting completed requests to free KV capacity.
python3 << 'PYEOF' 2>/dev/null || true
import site, os

old = (
    '            model_output = self.executor.execute_model(scheduler_output)\n'
    '            self.executor.sample_tokens(model_output)\n'
)
new = (
    '            model_output = self.executor.execute_model(scheduler_output)\n'
    '            sampled_output = self.executor.sample_tokens(model_output)\n'
    '            # Advance scheduler state so KV blocks are freed after each batch.\n'
    '            # Without this, newer vLLM never admits requests beyond MAX_NUM_SEQS.\n'
    '            if hasattr(self.scheduler, \'update_from_output\'):\n'
    '                try:\n'
    '                    self.scheduler.update_from_output(scheduler_output, sampled_output)\n'
    '                except Exception:\n'
    '                    try:\n'
    '                        self.scheduler.update_from_output(scheduler_output, model_output)\n'
    '                    except Exception:\n'
    '                        pass\n'
    '            # Abort completed-prefill requests this iteration to free KV capacity.\n'
    '            _just_done = [\n'
    '                _r for _r in scheduler_output.num_scheduled_tokens\n'
    '                if request_num_computed.get(_r, 0) >= request_id_to_prompt_len.get(_r, 0)\n'
    '            ]\n'
    '            if _just_done:\n'
    '                try:\n'
    '                    self.scheduler.finish_requests(_just_done, RequestStatus.FINISHED_ABORTED)\n'
    '                except Exception:\n'
    '                    pass\n'
)

for d in site.getsitepackages():
    path = os.path.join(d, 'speculators', 'data_generation', 'vllm_hidden_states_generator.py')
    if not os.path.exists(path):
        continue
    with open(path) as f:
        c = f.read()
    if old in c and new not in c:
        with open(path, 'w') as f:
            f.write(c.replace(old, new))
        print('Patched vllm_hidden_states_generator.py: added update_from_output + finish_requests in generate() loop')
    else:
        print('vllm_hidden_states_generator.py: scheduler patch already applied or pattern not found')
    break
PYEOF

if [ -z ${SLURM_ARRAY_TASK_ID} ]; then
    TASK_ID=0
else
    TASK_ID=${SLURM_ARRAY_TASK_ID}
fi

if [ -z ${SLURM_ARRAY_TASK_COUNT} ]; then
    TASK_COUNT=1
else
    TASK_COUNT=${SLURM_ARRAY_TASK_COUNT}
fi

python3 modules/Model-Optimizer/examples/speculative_decoding/collect_hidden_states/compute_hidden_states_vllm.py \
    --model ${HF_MODEL_CKPT} \
    --dp-rank ${TASK_ID} \
    --dp-world-size ${TASK_COUNT} \
    --trust_remote_code \
    ${@}
