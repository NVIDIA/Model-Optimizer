# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install --yes --no-install-recommends \
    python3 python3-venv python3-dev git build-essential ca-certificates \
    libjpeg-dev libpng-dev && rm -rf /var/lib/apt/lists/*
ARG MODELOPT_REF
RUN test "${#MODELOPT_REF}" = 40 && git init /opt/modelopt && \
    cd /opt/modelopt && git remote add origin https://github.com/NVIDIA/Model-Optimizer.git && \
    git fetch --depth 1 origin "${MODELOPT_REF}" && git checkout --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${MODELOPT_REF}"
COPY install_environment.sh verify_environment.py /opt/ptq/
ARG TORCH_VERSION
ARG TRANSFORMERS_VERSION
ARG TORCHVISION_VERSION
ARG MODEL_CLASS="AutoModelForCausalLM"
ARG PACKAGE_CHECK_FLAGS=""
RUN bash /opt/ptq/install_environment.sh /opt/modelopt /opt/ptq/venv \
    "${TORCH_VERSION}" "${TRANSFORMERS_VERSION}" "${TORCHVISION_VERSION}" ${PACKAGE_CHECK_FLAGS}
ENV PATH="/opt/ptq/venv/bin:${PATH}" PYTHONNOUSERSITE=1
LABEL org.opencontainers.image.source="https://github.com/NVIDIA/Model-Optimizer" \
    org.opencontainers.image.revision="${MODELOPT_REF}"
RUN python /opt/ptq/verify_environment.py \
    --source /opt/modelopt --ref "${MODELOPT_REF}" \
    --model-class "${MODEL_CLASS}" ${PACKAGE_CHECK_FLAGS} \
    > /opt/ptq/environment.json && \
    python /opt/modelopt/examples/hf_ptq/hf_ptq.py --help > /opt/ptq/hf-ptq-help.txt
WORKDIR /opt/ptq
