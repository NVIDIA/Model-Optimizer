FROM nvcr.io/nvidia/nemo:26.04.00

# Install Jupyter - pin to new version as needed
RUN pip install --root-user-action ignore \
    jupyterlab==4.4.10 \
    ipywidgets==8.1.7 \
    huggingface_hub==1.20.1 && \
    jupyter labextension disable "@jupyterlab/apputils-extension:announcements"
ENV SHELL /bin/bash

# Install git-lfs and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git-lfs

# Disable python warnings to improve student experience
ENV PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning,ignore::SyntaxWarning"
ENV TRANSFORMERS_VERBOSITY=error


# Install Model-Optimizer from source, using branch lmikaelyan/compress-vlms
RUN git clone https://github.com/NVIDIA/Model-Optimizer.git /dli/Model-Optimizer && \
    cd /dli/Model-Optimizer && git checkout lmikaelyan/compress-vlms && \
    python -m pip uninstall -y nvidia-modelopt && \
    pip install -e .

# Install lmms-eval for VLM evaluation (§5.6 — MathVista / BLINK / RealWorldQA)
RUN git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git /dli/lmms-eval && \
    cd /dli/lmms-eval && git checkout v0.7.1 && \
    pip install --root-user-action ignore -e . && \
    pip install --root-user-action ignore python-Levenshtein && \
    # Fix bug in v0.7.1: OpenAIProvider default OPENAI_API_URL includes
    # /chat/completions causing the OpenAI client to double-append that path.
    sed -i 's|os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions/v1")|os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")|' \
        /dli/lmms-eval/lmms_eval/llm_judge/providers/openai.py