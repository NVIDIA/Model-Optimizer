FROM vllm/vllm-openai:v0.26.0

# Set environment variables
ENV PIP_NO_CACHE_DIR=off \
    PIP_CONSTRAINT=

WORKDIR /workspace

# Install system dependencies needed for modelopt
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire Model-Optimizer source code
COPY . Model-Optimizer

# Remove .git directory to reduce image size
RUN rm -rf Model-Optimizer/.git

# Install modelopt from local source with all dependencies. `mlflow` is the optional
# tracking client used by --mlflow; it is not part of `all`.
RUN cd Model-Optimizer && \
    pip install -e ".[all,dev-test,mlflow]"

# Llama4 requires this
RUN pip install flash-attn==2.7.4.post1 --no-build-isolation

# Pre-compile CUDA extensions into a world-accessible directory so the vllm
# user can use the cache at runtime.
ENV TORCH_EXTENSIONS_DIR=/workspace/torch_extensions
RUN python3 -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()" || true

# Allow the non-root vllm user to access the workspace
RUN chmod -R 777 /workspace

USER vllm

# Override the ENTRYPOINT from the base image to allow flexible usage
ENTRYPOINT []

# Set the default command
CMD ["/bin/bash"]
