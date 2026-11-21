#!/usr/bin/env python3
"""Run a Kaggle script kernel from your own machine: push, poll, download.

Usage:
    run.py <kernel_dir> [--outdir DIR] [--timeout-min N]
    run.py --resume <kernel-id> [--outdir DIR] [--timeout-min N]

kernel_dir holds generate.py and kernel-metadata.json (see ../kernel/).
Needs the Kaggle CLI installed and authenticated (~/.kaggle/kaggle.json),
and a phone-verified Kaggle account for GPU access.

Note: Kaggle normalizes kernel slugs on push (e.g. "qwen-image-21-generate"
becomes "qwen-image-2-1-generate"), so the real slug is parsed from the
push output instead of trusting the metadata id.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

KAGGLE = shutil.which("kaggle")
if KAGGLE is None:
    print("kaggle CLI not found in PATH. Install it with: pip install kaggle",
          file=sys.stderr)
    sys.exit(2)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def parse_args(argv):
    outdir = None
    timeout_min = 120
    positional = []
    i = 0
    while i < len(argv):
        if argv[i] == "--outdir":
            outdir = argv[i + 1]
            i += 2
        elif argv[i] == "--timeout-min":
            timeout_min = int(argv[i + 1])
            i += 2
        elif argv[i] == "--resume":
            positional.append("--resume")
            positional.append(argv[i + 1])
            i += 2
        else:
            positional.append(argv[i])
            i += 1
    return positional, outdir, timeout_min


def wait_and_download(kid, outdir, timeout_min):
    if outdir is None:
        outdir = os.path.join(os.getcwd(), "outputs", kid.replace("/", "_"))
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        time.sleep(60)
        r = sh([KAGGLE, "kernels", "status", kid])
        status = (r.stdout + r.stderr).strip()
        print(time.strftime("%H:%M:%S"), status, flush=True)
        low = status.lower()
        if "complete" in low:
            break
        if "error" in low or "failed" in low or "cannot access" in low:
            print("kernel failed:", status, file=sys.stderr)
            sys.exit(1)
    else:
        print("timed out waiting for", kid, file=sys.stderr)
        sys.exit(1)

    os.makedirs(outdir, exist_ok=True)
    r = sh([KAGGLE, "kernels", "output", kid, "-p", outdir])
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit(1)
    print("outputs in", outdir)
    for f in sorted(os.listdir(outdir)):
        p = os.path.join(outdir, f)
        print(" -", f, os.path.getsize(p), "bytes")


def main():
    positional, outdir, timeout_min = parse_args(sys.argv[1:])
    if positional and positional[0] == "--resume":
        wait_and_download(positional[1], outdir, timeout_min)
        return

    if len(positional) < 1:
        print(__doc__)
        sys.exit(2)
    kernel_dir = positional[0]

    meta = json.load(open(os.path.join(kernel_dir, "kernel-metadata.json")))
    kid = meta["id"]
    if kid.startswith("YOUR_KAGGLE_USERNAME"):
        print("Edit kernel-metadata.json first: replace YOUR_KAGGLE_USERNAME "
              "with your Kaggle username.", file=sys.stderr)
        sys.exit(2)

    tmp = tempfile.mkdtemp(prefix="kaggle-run-")
    for f in ("generate.py", "kernel-metadata.json"):
        shutil.copy(os.path.join(kernel_dir, f), os.path.join(tmp, f))

    print("pushing", kid, "from", tmp, flush=True)
    r = sh([KAGGLE, "kernels", "push", "-p", tmp])
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit(1)
    # Kaggle normalizes slugs; use the real one from the push output.
    m = re.search(r"code/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", r.stdout + r.stderr)
    if m:
        kid = m.group(1)
        print("real kernel slug:", kid, flush=True)

    wait_and_download(kid, outdir, timeout_min)


if __name__ == "__main__":
    main()
