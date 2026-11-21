# Why GGUF, not diffusers, on Kaggle's free T4s

The obvious way to run Qwen-Image on Kaggle is `diffusers`. It doesn't work
on the free T4 tier — and no amount of quantization or memory juggling fixes
it, because the problem is the hardware's dtype support, not memory.

## The symptom

Running Qwen-Image through diffusers on a Kaggle T4 (2× T4, 16 GB each):

- **Qwen-Image 2.1**: images come out **magenta** — color channels blown out,
  detail destroyed.
- **Qwen-Image 1.0**: the run collapses to **NaN** entirely.

## The diagnosis

The T4 (Turing, sm_75) has **no native BF16**. diffusers runs the model in
FP16/BF16 mixed precision; without hardware BF16, FP16 activations overflow
in the attention and norm layers and poison the latent space. The corruption
happens in the forward pass itself, so it cannot be fixed by:

- quantizing the weights (the overflow is in *activations*, not weights),
- lowering resolution or batch size (same overflow, smaller tensor),
- gradient checkpointing / CPU offload (memory tricks don't touch dtype).

## The fix: GGUF + stable-diffusion.cpp

[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
runs the model as a GGUF-quantized graph (Q8_0 here) with its own CUDA
kernels, computing in dtypes the T4 actually supports. Same weights, same
prompts — clean images, no magenta, no NaN.

The stack this repo uses:

| Piece | File | Size |
|---|---|---|
| Diffusion model (DiT) | `Qwen-Image-2.1-Turbo-Q8_0.gguf` (AtomicChat) | 7.59 GB |
| Text encoder | `Qwen3-VL-8B-Instruct-Q8_0.gguf` (unsloth) | 8.71 GB |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` (Comfy-Org) | 675 MB |
| Runtime | `sd-cli` from stable-diffusion.cpp, CUDA build for sm_75 | 122 MB |

## The real bottleneck is the build, not the download

Timed phases of a cold run (from the kernel log):

| Phase | Time |
|---|---|
| Clone stable-diffusion.cpp + cmake configure | ~45 s |
| **Build sd-cli from source** | **36 min 05 s** |
| Download models from Hugging Face (Kaggle→HF is fast, ~150–200 MB/s) | ~1 min 33 s |
| Generate one 1024×1024 image (8 steps) | ~1 min 31 s |
| **Cold total** | **~40 min 41 s** |

Caching the models alone saves under 2 minutes. Caching the **binary**
saves 36. That's why the
[cache dataset](https://www.kaggle.com/datasets/hankyin88/qwen-image-21-turbo-gguf-cache)
ships the pre-built `sd-cli` alongside the weights — with it, a single image
takes **3 min 09 s** end to end, and a 4-image batch about 7–8 minutes.

One caveat: the binary links CUDA 13 libraries, so it stays valid as long as
Kaggle's kernel image ships CUDA 13. If Kaggle upgrades the CUDA major
version, the binary must be rebuilt — `generate.py` falls back to building
from source automatically, so the pipeline keeps working, just slower that
once.

## Requirements

- stable-diffusion.cpp from **2026-10-06 or later** (Qwen-Image-2.1 support;
  the cached binary was built from commit
  `32d516c0197ed5ca31305b8575e14eb1451fad33`).
- CUDA arch 75 (T4) at build time: `-DCMAKE_CUDA_ARCHITECTURES=75`.
- Kaggle mounts the NVIDIA driver libs outside the standard path, so cmake
  can't find `libcuda.so` on its own — `generate.py` locates it via
  `ldconfig` and points CMake at a stub symlink in `/tmp` (note:
  `/usr/local/cuda` is read-only on Kaggle, so the stub can't live there).
