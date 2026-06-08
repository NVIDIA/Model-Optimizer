# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class Model:
    """Base class for inference-engine wrappers.

    Cross-engine kwarg conventions (read by run.py, set on **kwargs):

    - ``sampling_kwargs``: dict-shaped sampling config (``temperature``,
      etc.). Universal; every engine consumes it.
    - ``max_model_len`` / ``max_seq_len`` / ``context_length``: max
      input+output sequence length the engine should reserve. The CLI
      flag ``--max_seq_len`` in run.py is generic; it is translated to
      one of these three engine-specific kwargs at the run_simple()
      seam based on ``--engine``. New engine wrappers should read one
      of these names and add the mapping in run.py's
      ``_MAX_SEQ_LEN_KEY``.

    Engine-specific kwargs (``mem_fraction_static`` for SGLang,
    ``enable_chunked_prefill`` for TRT-LLM, etc.) are passed through
    ``**kwargs`` from ``--runtime_params engine_args`` without
    translation — those are the engine's own surface, not part of the
    cross-engine contract.
    """

    def __init__(self, model_dir, tokenizer, max_draft_length):
        raise NotImplementedError

    async def run(self, prompt_ids, sampling_params, request_id, turn_id):
        """
        prompt_ids is list of tokens
        output is list of list of tokens
            len(output) = beam width
            len(output[i]) = tokens produced per step?
        """
        raise NotImplementedError

    def get_serving_config(self):
        """Return a JSON-serializable dict describing the engine's effective config.

        Captured into configuration.json's `serving_config` for reproducibility.
        Subclasses override to surface engine-specific defaults (max_model_len,
        kv_cache_dtype, etc.) that don't appear in the CLI args. Default: empty.
        """
        return {}

    def stop(self):
        pass
