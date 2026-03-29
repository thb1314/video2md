ARG CUDA_BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g; s|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 \
       python3-venv \
       python3-pip \
       ffmpeg \
       ca-certificates \
       curl \
       git \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir uv==0.10.11 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

ARG INSTALL_EXTRAS="asr"
ARG INSTALL_ONNXRUNTIME_GPU="0"
ARG UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG HTTPS_PROXY=""
ARG HTTP_PROXY=""

ENV UV_INDEX_URL=${UV_INDEX_URL} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    HTTP_PROXY=${HTTP_PROXY}

RUN if [ -n "$INSTALL_EXTRAS" ]; then \
      extras_args=""; \
      for extra in $INSTALL_EXTRAS; do extras_args="$extras_args --extra $extra"; done; \
      uv sync --frozen --no-dev $extras_args; \
    else \
      uv sync --frozen --no-dev; \
    fi

RUN if [ "$INSTALL_ONNXRUNTIME_GPU" = "1" ]; then \
      uv pip install --python /app/.venv/bin/python --upgrade onnxruntime-gpu \
      || uv pip install --python /app/.venv/bin/python --upgrade --index-url https://pypi.org/simple onnxruntime-gpu; \
    fi

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["video2md"]
CMD ["--help"]
