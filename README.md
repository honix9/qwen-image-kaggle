# Qwen-Image on free Kaggle GPUs

Run **Qwen-Image-2.1-Turbo** on Kaggle's free T4 GPUs — a single 1024×1024
image in **~3 minutes** end to end, no paid GPU needed.

The trick is a [public cache dataset](https://www.kaggle.com/datasets/hankyin88/qwen-image-21-turbo-gguf-cache)
holding a pre-built `sd-cli` binary plus the GGUF model files. Without it,
every run pays a ~36-minute source build of
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
before generating anything.

## Why not diffusers?

The T4 has no native BF16, so diffusers' FP16 activations overflow and
poison the latent space — magenta images on Qwen-Image 2.1, NaN on 1.0. No
quantization or memory setting fixes it, because the overflow is in the
forward pass itself. The GGUF route ([stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
+ Q8_0 weights) sidesteps the problem entirely. Full write-up:
[docs/why-gguf.md](docs/why-gguf.md).

## Benchmarks

Measured on Kaggle's free tier (2× T4, 16 GB each), 1024×1024, 8 steps:

| Run | Time |
|---|---|
| Cold run (build sd-cli from source + download models + 1 image) | ~40 min 41 s |
| — of which: sd-cli build | 36 min 05 s |
| — of which: model downloads (Kaggle→HF, ~150–200 MB/s) | ~1 min 33 s |
| — of which: generation | ~1 min 31 s |
| **Cached run, single image** | **3 min 09 s** |
| Cached run, 4-image batch | ~7–8 min |

Kaggle's free GPU quota is 30 hours/week (phone-verified account required),
so a cached run costs roughly 5 minutes of quota per image.

## What's in the cache dataset

[hankyin88/qwen-image-21-turbo-gguf-cache](https://www.kaggle.com/datasets/hankyin88/qwen-image-21-turbo-gguf-cache)
(~17.1 GB total, CC0):

| File | Size | Source |
|---|---|---|
| `Qwen-Image-2.1-Turbo-Q8_0.gguf` | 7.59 GB | [AtomicChat/Qwen-Image-2.1-Turbo-GGUF](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF) |
| `Qwen3-VL-8B-Instruct-Q8_0.gguf` | 8.71 GB | [unsloth/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF) |
| `qwen_image_2.1_vae_bf16.safetensors` | 675 MB | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) |
| `sd-cli` (CUDA, sm_75, from stable-diffusion.cpp `32d516c`) | 122 MB | Built on Kaggle, see [docs/why-gguf.md](docs/why-gguf.md) |

All three model files were verified against their Hugging Face SHA-256
checksums before upload.

## Quickstart

### Prerequisites

- A Kaggle account with **phone verification** (required for free GPU).
- The [Kaggle CLI](https://github.com/Kaggle/kaggle-api) installed and
  authenticated (`~/.kaggle/kaggle.json`).

### 1. Clone and configure

```bash
git clone https://github.com/honix9/qwen-image-kaggle.git
cd qwen-image-kaggle
```

Edit `kernel/kernel-metadata.json` and replace `YOUR_KAGGLE_USERNAME` with
your Kaggle username.

### 2. Edit the prompts

`kernel/generate.py` has a `JOBS` list — `(prompt, seed, output file)`.
Edit the four demo prompts or add your own rows. Each job is one `sd-cli`
call (~1–2 min per image from the cache).

Generation parameters (verified): 8 steps, CFG scale 1.0, euler sampler,
1024×1024, `--rng cpu`.

### 3. Run

```bash
python tools/run.py kernel/
```

This pushes the kernel, polls until it finishes, and downloads the images
to `outputs/`. Or push directly:

```bash
kaggle kernels push -p kernel/
```

### Without the CLI

Create a Kaggle notebook (Script, GPU on, Internet on), paste
`kernel/generate.py`, and add the dataset
`hankyin88/qwen-image-21-turbo-gguf-cache` as an input. Run it.

## How it works

`kernel/generate.py` prefers the cached fast path and falls back
automatically:

1. **sd-cli**: uses the pre-built binary from the dataset (copied to `/tmp`
   and chmod'd, since dataset mounts may be noexec). If missing, clones
   stable-diffusion.cpp and builds with CUDA for sm_75. The build needs a
   `libcuda.so` stub — Kaggle mounts the driver libs outside the standard
   path, so the script locates `libcuda.so.1` via `ldconfig` and points
   CMake at a stub symlink in `/tmp` (`/usr/local/cuda` is read-only).
2. **Models**: uses the dataset files when present, otherwise downloads
   from Hugging Face.
3. **Generate**: loops over `JOBS`, one `sd-cli` call each. Everything
   builds in `/tmp`; only the images land in `/kaggle/working`, so they are
   the only kernel outputs.

If Kaggle upgrades its CUDA major version, the cached binary may stop
working — the script then falls back to a source build automatically.

## Tools

- `tools/run.py` — push a kernel from your machine, poll for completion,
  download outputs. `run.py --resume <kernel-id>` re-attaches to a running
  kernel.
- `tools/chunked_dataset_version.py` — upload multi-GB files to a Kaggle
  dataset in 256 MB resumable chunks. The stock CLI uploads each file as a
  single PUT, which dies after ~1 GB / ~7 min on a slow connection (and its
  resume query gets HTTP 411). This one probes the committed offset and
  resumes where it left off. Built the 17 GB cache dataset this way.

## Troubleshooting

- **Kernel falls back to the 40-minute cold path**: the dataset wasn't
  attached. Check `dataset_sources` in `kernel-metadata.json` (web
  notebooks: add the dataset under *Input*).
- **`libcuda.so` / cmake errors on the cold path**: the stub fix in
  `generate.py` covers the known Kaggle layout; if Kaggle changes its
  image, check the kernel log's `libcuda found:` line first.
- **Only some images in the output**: each `sd-cli` call reloads the
  models, so a failure mid-batch still keeps the earlier images in
  `/kaggle/working`.
- **Quota**: Kaggle allows at most 2 concurrent GPU sessions; a second
  `run.py` while one is active will queue.

## Roadmap

- Wan2.2-5B video generation (working in private testing, to be packaged).
- More samplers / aspect ratios as presets.

## Credits & licenses

- [Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image) by the Qwen team
  (Apache-2.0).
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
  by leejet (MIT).
- GGUF quantizations by
  [AtomicChat](https://huggingface.co/AtomicChat) and
  [unsloth](https://huggingface.co/unsloth); VAE by
  [Comfy-Org](https://huggingface.co/Comfy-Org).
- This repo's code is MIT (see [LICENSE](LICENSE)). Model weights carry
  their own licenses — check the Hugging Face pages above.
