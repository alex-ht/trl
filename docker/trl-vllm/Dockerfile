FROM nvcr.io/nvidia/vllm:26.05.post1-py3

# Target NVIDIA H100 (compute capability 9.0)
ENV TORCH_CUDA_ARCH_LIST=9.0
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libaio-dev \
    nano \
    vim-tiny \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv

# Ubuntu 24.04 marks the system Python as externally managed (PEP 668)
ENV UV_BREAK_SYSTEM_PACKAGES=1

ENV DS_SKIP_CUDA_CHECK=1
ENV DS_BUILD_FUSED_ADAM=1
ENV DS_BUILD_AIO=1
ENV DS_BUILD_QUANTIZER=1
ENV DS_BUILD_TRANSFORMER=1
ENV DS_BUILD_TRANSFORMER_INFERENCE=1
ENV DS_BUILD_STOCHASTIC_TRANSFORMER=1
ENV DS_BUILD_SPARSE_ATTN=1
ENV DS_BUILD_RANDOM_LTD=1
ENV DS_BUILD_CPU_ADAM=1

# DeepSpeed needs torch at build time and setuptools_scm trips over uv's sdist git cache
RUN git clone --branch 0.19.3 --depth 1 https://github.com/alex-ht/DeepSpeed.git /tmp/DeepSpeed && \
    cd /tmp/DeepSpeed && \
    pip install --break-system-packages --no-build-isolation --no-cache-dir . && \
    pip install --break-system-packages --no-build-isolation --no-cache-dir "transformers!=5.1.0" && \
    rm -rf /tmp/DeepSpeed

# Flash Attention 3 for H100 — build from source against the container's PyTorch
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE
ENV FLASH_ATTENTION_DISABLE_SM80=TRUE
ENV MAX_JOBS=4
ENV NVCC_THREADS=4
RUN pip install --break-system-packages --no-cache-dir einops ninja packaging psutil && \
    git clone --recursive https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
    cd /tmp/flash-attention/hopper && \
    pip install --break-system-packages --no-build-isolation --no-cache-dir . && \
    rm -rf /tmp/flash-attention
RUN python -c "import flash_attn_3; print('flash_attn_3 OK')"

ADD . /workspace/trl
RUN cd /workspace/trl && uv pip install --system --no-build-isolation \
    ".[kernels,peft,quantization,scikit,vlm,math_verify]" && \
    pip install --break-system-packages --no-build-isolation --no-cache-dir \
    "git+https://github.com/linkedin/Liger-Kernel.git" \
    wandb tensorboard
