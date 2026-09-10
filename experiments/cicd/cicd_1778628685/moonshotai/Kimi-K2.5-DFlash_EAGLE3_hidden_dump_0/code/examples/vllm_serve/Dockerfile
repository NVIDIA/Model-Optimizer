FROM vllm/vllm-openai:v0.10.2

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

# Install modelopt from local source with all dependencies
RUN cd Model-Optimizer && \
    pip install -e ".[all,dev-test]"

# Llama4 requires this
RUN pip install flash-attn==2.7.4.post1

# Pre-compile CUDA extensions to avoid compilation time during runtime
RUN python3 -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()" || true

# Install requirements from examples (excluding windows examples)
RUN find Model-Optimizer/examples -name "requirements.txt" | grep -v "windows" | while read req_file; do \
        echo "Installing from $req_file"; \
        pip install -r "$req_file" || echo "Warning: Failed to install from $req_file"; \
    done

# Allow users to run without root
RUN chmod -R 777 /workspace

# Override the ENTRYPOINT from the base image to allow flexible usage
ENTRYPOINT []

# Set the default command
CMD ["/bin/bash"]
