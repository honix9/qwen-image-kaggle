# Qwen-Image-2.1-Turbo GGUF on free Kaggle T4 GPUs via stable-diffusion.cpp (sd-cli)
#
# Two paths, fastest first:
# 1. Cached fast path — models + pre-built sd-cli binary come from the public
#    Kaggle dataset hankyin88/qwen-image-21-turbo-gguf-cache (mounted at
#    /kaggle/input/qwen-image-21-turbo-gguf-cache when listed in
#    kernel-metadata.json dataset_sources). Single image: ~3 min end to end.
# 2. Cold fallback — if any dataset file is missing, sd-cli is built from
#    source (needs stable-diffusion.cpp from 2026-10-06 or later for
#    Qwen-Image-2.1 support) and models are downloaded from Hugging Face.
#    Cold run: ~40 min, of which the sd-cli build alone is ~36 min.
#
# Why GGUF instead of diffusers? See docs/why-gguf.md — short version: the
# T4 has no native BF16, so diffusers' FP16 activations overflow and poison
# the latent space (magenta images on 2.1, NaN on 1.0). The GGUF route
# sidesteps the problem entirely.
#
# Build + models live in /tmp (keeps /kaggle/working clean so only the
# generated images are saved as kernel output; cloning the repo into
# /working tripped Kaggle's API rate limit with the giant .git output).
#
# Run with `kaggle kernels push -p kernel/` (see README), or use
# tools/run.py from your own machine to push, poll, and download outputs.
import os
import shutil
import subprocess
import sys
import time

JOBS = [
    # (prompt, seed, output file) — edit these, or add more rows.
    ("Photorealistic still life of a single glossy red apple on a rustic wooden table, soft morning window light, shallow depth of field, highly detailed", 42, "apple.png"),
    ("A serene mountain lake at sunrise, thin mist over the water, pine trees along the shore, distant snow peaks, photorealistic landscape photography, highly detailed", 7, "lake.png"),
    ("Studio portrait of a tabby cat sitting upright and looking at the camera, soft studio lighting, photorealistic, highly detailed fur", 123, "cat.png"),
    ("A cozy reading nook beside a rainy window, warm lamp light, a stack of old books and a cup of tea on a wooden table, cinematic, photorealistic interior photography", 2026, "nook.png"),
]
WIDTH = 1024
HEIGHT = 1024

WORK = "/kaggle/working"
TMP = "/tmp/gguf"
SD_DIR = os.path.join(TMP, "stable-diffusion.cpp")
SD_BIN = os.path.join(SD_DIR, "build", "bin", "sd-cli")
MODELS = os.path.join(TMP, "models")

DIT_REPO = "AtomicChat/Qwen-Image-2.1-Turbo-GGUF"
DIT_FILE = "Qwen-Image-2.1-Turbo-Q8_0.gguf"
ENC_REPO = "unsloth/Qwen3-VL-8B-Instruct-GGUF"
ENC_FILE = "Qwen3-VL-8B-Instruct-Q8_0.gguf"
VAE_REPO = "Comfy-Org/Qwen-Image-2.1"
VAE_FILE = "vae/qwen_image_2.1_vae_bf16.safetensors"

# Cached dataset (mounted read-only by Kaggle when listed in
# kernel-metadata.json dataset_sources). Files are stored flat at the
# dataset root, so the VAE has no "vae/" prefix here.
DATASET_DIR = "/kaggle/input/qwen-image-21-turbo-gguf-cache"
VAE_DS_FILE = "qwen_image_2.1_vae_bf16.safetensors"
SD_BIN_DS = os.path.join(DATASET_DIR, "sd-cli")

SIGMAS = "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, cwd=None, check=True):
    log("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if r.stdout:
        print(r.stdout[-3000:], flush=True)
    if r.stderr:
        print(r.stderr[-3000:], flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r


os.makedirs(TMP, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)
os.chdir(WORK)
log("disk /tmp: " + os.popen("df -h /tmp | tail -1").read().strip())
log("disk /kaggle/working: " + os.popen("df -h /kaggle/working | tail -1").read().strip())
log("cpus: " + str(os.cpu_count()))
run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
for tool in ["git", "cmake", "g++", "nvcc"]:
    log(f"{tool}: {shutil.which(tool) or 'MISSING'}")

# ---- Fix CUDA driver stub (CMake's CUDA::cuda_driver target) ----
# cmake configure fails with: "Target stable-diffusion links to
# CUDA::cuda_driver but the target was not found." FindCUDAToolkit needs
# libcuda.so (the driver stub). Kaggle mounts the NVIDIA driver libs
# outside the standard path, so locate libcuda.so.1 via ldconfig/find
# instead of assuming /usr/lib/x86_64-linux-gnu/.
import glob
def find_libcuda():
    # 1) ldconfig cache
    r = subprocess.run(["ldconfig", "-p"], text=True, capture_output=True)
    for line in r.stdout.splitlines():
        if "libcuda.so" in line and "=>" in line:
            p = line.split("=>", 1)[1].strip()
            if os.path.exists(p):
                return p
    # 2) filesystem search in likely roots
    for root in ["/usr/lib", "/usr/local", "/opt", "/lib"]:
        for cand in glob.glob(root + "/**/libcuda.so.1", recursive=True):
            return cand
    return None

libcuda = find_libcuda()
log(f"libcuda found: {libcuda or 'NONE'}")
# /usr/local/cuda is READ-ONLY on Kaggle (OSError Errno 30), so put the
# stub symlink in a writable dir and point CMake at it via
# CMAKE_LIBRARY_PATH (find_library honours it).
stubs_dir = "/tmp/cuda-stubs"
os.makedirs(stubs_dir, exist_ok=True)
stub = os.path.join(stubs_dir, "libcuda.so")
if libcuda and not os.path.exists(stub):
    os.symlink(libcuda, stub)
    log(f"created stub symlink: {stub} -> {libcuda}")
elif os.path.exists(stub):
    log(f"stub already exists: {stub}")
else:
    log("WARNING: no libcuda.so.1 found anywhere; cmake may still fail")

# ---- Step 1: sd-cli (prefer cached binary from dataset, else build) ----
SD_BIN_CACHED = os.path.join(TMP, "bin", "sd-cli")
if os.path.isfile(SD_BIN_DS):
    # Dataset mounts may be noexec, so copy to /tmp and chmod rather
    # than running the binary in place.
    os.makedirs(os.path.join(TMP, "bin"), exist_ok=True)
    shutil.copyfile(SD_BIN_DS, SD_BIN_CACHED)
    os.chmod(SD_BIN_CACHED, 0o755)
    SD_BIN = SD_BIN_CACHED
    log(f"using cached sd-cli from dataset ({os.path.getsize(SD_BIN)/1e6:.1f} MB)")
elif not (os.path.isfile(SD_BIN) and os.access(SD_BIN, os.X_OK)):
    if os.path.isdir(SD_DIR):
        shutil.rmtree(SD_DIR)
    log("cloning stable-diffusion.cpp ...")
    run(["git", "clone", "--depth", "1", "--recursive",
         "https://github.com/leejet/stable-diffusion.cpp", SD_DIR])
    log("configuring cmake (SD_CUDA=ON, CUDA arch 75 for T4) ...")
    run(["cmake", "-B", "build", "-DSD_CUDA=ON",
         "-DCMAKE_CUDA_ARCHITECTURES=75",
         "-DCMAKE_LIBRARY_PATH=/tmp/cuda-stubs",
         "-DCMAKE_BUILD_TYPE=Release"], cwd=SD_DIR)
    log("building sd-cli ...")
    run(["cmake", "--build", "build", "--config", "Release", "-j", str(os.cpu_count())], cwd=SD_DIR)
    log("sd-cli built: " + SD_BIN)
else:
    log("sd-cli already built, skipping build")

run([SD_BIN, "--help"], check=False)

# ---- Step 2: model files (prefer cached dataset, fall back to HF) ----
dit_ds = os.path.join(DATASET_DIR, DIT_FILE)
enc_ds = os.path.join(DATASET_DIR, ENC_FILE)
vae_ds = os.path.join(DATASET_DIR, VAE_DS_FILE)
if os.path.isfile(dit_ds) and os.path.isfile(enc_ds) and os.path.isfile(vae_ds):
    dit, enc, vae = dit_ds, enc_ds, vae_ds
    log(f"DiT (dataset): {dit} ({os.path.getsize(dit)/1e9:.2f} GB)")
    log(f"encoder (dataset): {enc} ({os.path.getsize(enc)/1e9:.2f} GB)")
    log(f"VAE (dataset): {vae} ({os.path.getsize(vae)/1e9:.2f} GB)")
else:
    missing = [p for p in (dit_ds, enc_ds, vae_ds) if not os.path.isfile(p)]
    log(f"dataset cache incomplete, missing: {missing}; downloading from HF ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

    dit = hf_hub_download(repo_id=DIT_REPO, filename=DIT_FILE, local_dir=MODELS)
    log(f"DiT: {dit} ({os.path.getsize(dit)/1e9:.2f} GB)")
    enc = hf_hub_download(repo_id=ENC_REPO, filename=ENC_FILE, local_dir=MODELS)
    log(f"encoder: {enc} ({os.path.getsize(enc)/1e9:.2f} GB)")
    vae = hf_hub_download(repo_id=VAE_REPO, filename=VAE_FILE, local_dir=MODELS)
    log(f"VAE: {vae} ({os.path.getsize(vae)/1e9:.2f} GB)")

# ---- Step 3: generate all jobs (one sd-cli call each; each call reloads
# the models from the dataset mount, ~1-2 min per image) ----
for i, (prompt, seed, fname) in enumerate(JOBS, 1):
    out_path = os.path.join(WORK, fname)
    cmd = [
        SD_BIN,
        "--diffusion-model", dit,
        "--vae", vae,
        "--llm", enc,
        "-p", prompt,
        "--steps", "8",
        "--cfg-scale", "1.0",
        "--sampling-method", "euler",
        "--sigmas", SIGMAS,
        "-W", str(WIDTH), "-H", str(HEIGHT),
        "--rng", "cpu",
        "--seed", str(seed),
        "-o", out_path,
    ]
    log(f"generating {i}/{len(JOBS)}: {fname} (seed {seed}) ...")
    run(cmd)
    if os.path.isfile(out_path):
        log(f"DONE {fname} {os.path.getsize(out_path)/1e6:.2f} MB")
    else:
        raise RuntimeError(f"{fname} was not produced")
log(f"ALL {len(JOBS)} DONE")
