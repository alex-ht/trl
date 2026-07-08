# TRL vLLM Docker image

Docker image for TRL training with [vLLM](https://github.com/vllm-project/vllm) generation and [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) optimization, built on NVIDIA's vLLM container.

Target GPU: **NVIDIA H100** (compute capability 9.0).

## Base image

[`nvcr.io/nvidia/vllm:26.05.post1-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm)

Includes (among others):

| Component | Version |
| --- | --- |
| OS | Ubuntu 24.04 |
| Python | 3.12 |
| PyTorch | 2.12.0a0+5aff3928d8 |
| CUDA | 13.2.1.009 |
| cuDNN | 9.22.0.52 |
| NCCL | 2.30.4 |
| TensorRT | 10.16.1.11 |
| TransformerEngine | v2.15 |
| vLLM | (bundled) |
| JupyterLab | 4.3.6 |

## Additions

This Dockerfile installs on top of the base image:

- **TRL** (from the local repository at `/workspace/trl`) with extras: `deepspeed`, `kernels`, `peft`, `quantization`, `scikit`, `vlm`, `math_verify`
- **Liger Kernel** installed from the latest [`linkedin/Liger-Kernel`](https://github.com/linkedin/Liger-Kernel) on GitHub
- **`kernels`** pinned to `>=0.11.0,<0.13` (required kernel API version)
- **Flash Attention 3** for H100, built from source in [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) (`hopper/`) against the container's PyTorch

  The build targets Hopper only (`sm_90a`) and uses the C++11 ABI required by NVIDIA containers. Hub kernels are not used because pre-built wheels fail to load on NVIDIA's custom PyTorch build.
- **nano** and **vi** (`vim-tiny`) text editors
- **git** and **libaio-dev** (required for DeepSpeed ZeRO offload)

The `vllm` extra is omitted because vLLM is already provided by the base image.

## Build

From the repository root:

```bash
docker build -t trl-vllm:h100 -f docker/trl-vllm/Dockerfile .
```

The build context must be the repository root so the Dockerfile can `ADD` the TRL source tree into the image.

The first build can take 1–2 hours because DeepSpeed and Flash Attention 3 CUDA kernels are compiled for H100 during installation.

## Run

```bash
docker run --gpus all -it --rm trl-vllm:h100
```

Mount the Hugging Face cache:

```bash
docker run --gpus all -it --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace/trl \
  trl-vllm:h100
```

To iterate on local code without rebuilding the image, mount the repository over `/workspace/trl`:

```bash
docker run --gpus all -it --rm \
  -v "$(pwd):/workspace/trl" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace/trl \
  trl-vllm:h100
```

## Flash Attention 3 training

Use the installed `flash_attn_3` package when launching TRL scripts on H100:

```bash
accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_gpt_oss.py \
    --attn_implementation flash_attention_3 \
    --dtype bfloat16
```

## DeepSpeed training

Use an Accelerate config with DeepSpeed, for example ZeRO Stage 2:

```bash
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml train.py
```

Ready-to-use configs are in [`examples/accelerate_configs/`](../../examples/accelerate_configs/).

## Environment variables

| Variable | Value | Purpose |
| --- | --- | --- |
| `TORCH_CUDA_ARCH_LIST` | `9.0` | Compile CUDA extensions for H100 |
| `DS_SKIP_CUDA_CHECK` | `1` | Skip DeepSpeed CUDA version check during build |
| `DS_BUILD_*` | `1` | Build selected DeepSpeed CUDA ops at image build time |
| `FLASH_ATTENTION_FORCE_BUILD` | `TRUE` | Build FA3 from source instead of downloading wheels |
| `FLASH_ATTENTION_FORCE_CXX11_ABI` | `TRUE` | Match NVIDIA container PyTorch ABI |
| `FLASH_ATTENTION_DISABLE_SM80` | `TRUE` | Skip Ampere kernels; build Hopper (H100) only |
| `MAX_JOBS` / `NVCC_THREADS` | `4` | Parallel FA3 compilation |