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

ADD . /workspace/trl
# DeepSpeed needs torch at build time and setuptools_scm trips over uv's sdist git cache
RUN pip install --break-system-packages --no-build-isolation --no-cache-dir \
    "deepspeed>=0.14.4" \
    "transformers!=5.1.0"
RUN cd /workspace/trl && uv pip install --system --no-build-isolation \
    ".[kernels,liger,peft,quantization,scikit,vlm,math_verify]"
