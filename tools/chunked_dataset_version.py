#!/usr/bin/env python3
"""Create a Kaggle dataset version using CHUNKED resumable uploads.

Why: the stock kaggle CLI uploads each file as ONE long PUT. On a slow or
flaky connection, a single PUT dies after ~1GB / ~7 min (SSLEOFError, then
the CLI's resume query gets HTTP 411 from Google), so multi-GB files can
never finish. This script monkey-patches KaggleApi.upload_complete with a
chunked version:
- Probe the session offset by (re-)sending the first 1MB and reading the
  Range header (the empty resume-status PUT 411s; a real chunk PUT works).
- Upload the rest in 256MB chunks, each a separate short PUT with
  Content-Range. A failed chunk is retried; committed bytes persist
  server-side, so re-running the script resumes instead of restarting.
- All other logic (session initiation, tokens, version creation) is the
  stock CLI's, untouched.

Usage (run with cwd=<dataset folder> and folder="."):
  python chunked_dataset_version.py . "<version notes>"

The CLI keeps resumable-upload state in /tmp, which may not survive a
reboot. This script therefore mirrors the state dir to .upload-state/
in the working directory (persistent): restored at startup, backed up
after each file completes and on SIGTERM/SIGINT.
"""
import glob
import os
import shutil
import signal
import sys
import tempfile
import time

import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ResumableUploadResult

CHUNK = 256 * 1024 * 1024  # 256 MB per PUT
PROBE = 1024 * 1024        # 1 MB probe chunk (multiple of 256 KiB)
MAX_ATTEMPTS = 10          # per chunk
PUT_TIMEOUT = 900          # seconds per chunk PUT

STATE_DIR = os.path.join(
    tempfile.gettempdir(), ".kaggle-uploads-%d" % os.geteuid())
PERSIST_DIR = os.path.join(os.getcwd(), ".upload-state")


def backup_state():
    """Copy /tmp upload state to persistent storage (best effort)."""
    try:
        if not os.path.isdir(STATE_DIR):
            return
        os.makedirs(PERSIST_DIR, exist_ok=True)
        for f in glob.glob(os.path.join(STATE_DIR, ".*.json")):
            shutil.copy2(f, PERSIST_DIR)
    except Exception as e:  # noqa: BLE001 - backup must never break uploads
        print(f"state backup failed: {e}", flush=True)


def restore_state():
    """Restore persistent upload state into /tmp if /tmp has none."""
    try:
        if not os.path.isdir(PERSIST_DIR):
            return
        existing = glob.glob(os.path.join(STATE_DIR, ".*.json")) \
            if os.path.isdir(STATE_DIR) else []
        if existing:
            return
        os.makedirs(STATE_DIR, mode=0o700, exist_ok=True)
        for f in glob.glob(os.path.join(PERSIST_DIR, ".*.json")):
            shutil.copy2(f, STATE_DIR)
        print(f"restored {len(glob.glob(os.path.join(PERSIST_DIR, '.*.json')))} "
              f"state file(s) from {PERSIST_DIR}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"state restore failed: {e}", flush=True)


def _term_handler(signum, frame):
    print(f"signal {signum}: backing up upload state, exiting", flush=True)
    backup_state()
    sys.exit(128 + signum)


def parse_range(headers):
    """Range: bytes=0-1072168959 -> next offset 1072168960. No header -> 0."""
    r = headers.get("Range")
    if not r:
        return 0
    try:
        return int(r.split("-")[1]) + 1
    except (IndexError, ValueError):
        return 0


def put_range(session, url, data, start, total):
    end = start + len(data) - 1
    headers = {
        "Content-Length": str(len(data)),
        "Content-Range": "bytes %d-%d/%d" % (start, end, total),
    }
    return session.put(url, data=data, headers=headers, timeout=PUT_TIMEOUT)


def chunked_upload_complete(self, path, url, quiet, resume=False):
    """Drop-in replacement for KaggleApi.upload_complete (same signature)."""
    file_size = os.path.getsize(path)
    name = os.path.basename(path)
    session = requests.Session()
    backup_state()  # session was just initiated; persist its token + URL

    def log(msg):
        if not quiet:
            print(msg, flush=True)

    # --- Probe: (re-)send the first 1MB; Google answers 308 + Range header
    # telling us how many bytes the session already holds. ---
    with open(path, "rb") as fp:
        probe_data = fp.read(min(PROBE, file_size))
    start_at = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = put_range(session, url, probe_data, 0, file_size)
            if resp.status_code in (200, 201):
                log(f"{name}: complete (single probe chunk)")
                return ResumableUploadResult.COMPLETE
            if resp.status_code == 308:
                start_at = parse_range(resp.headers)
                log(f"{name}: session holds {start_at}/{file_size} bytes "
                    f"({100.0 * start_at / file_size:.1f}%)")
                break
            log(f"{name}: probe HTTP {resp.status_code}, retry {attempt + 1}")
        except Exception as e:  # noqa: BLE001 - retry on any transport error
            log(f"{name}: probe error {e}, retry {attempt + 1}")
        time.sleep(min(5 * (attempt + 1), 60))
    if start_at is None:
        return ResumableUploadResult.INCOMPLETE

    # --- Upload remaining chunks. ---
    uploaded = start_at
    with open(path, "rb") as fp:
        while uploaded < file_size:
            fp.seek(uploaded)
            data = fp.read(min(CHUNK, file_size - uploaded))
            progressed = False
            for attempt in range(MAX_ATTEMPTS):
                try:
                    resp = put_range(session, url, data, uploaded, file_size)
                    if resp.status_code in (200, 201):
                        log(f"{name}: complete ({file_size} bytes)")
                        backup_state()
                        return ResumableUploadResult.COMPLETE
                    if resp.status_code == 308:
                        new_off = parse_range(resp.headers)
                        if new_off > uploaded:
                            uploaded = new_off
                            progressed = True
                            break
                        log(f"{name}: chunk @{uploaded} 308 no progress, "
                            f"retry {attempt + 1}")
                    else:
                        log(f"{name}: chunk @{uploaded} HTTP "
                            f"{resp.status_code}, retry {attempt + 1}")
                except Exception as e:  # noqa: BLE001
                    log(f"{name}: chunk @{uploaded} error {e}, "
                        f"retry {attempt + 1}")
                time.sleep(min(5 * (attempt + 1), 60))
            if not progressed:
                log(f"{name}: chunk @{uploaded} failed after {MAX_ATTEMPTS} "
                    f"attempts; will resume on next pass")
                return ResumableUploadResult.INCOMPLETE
            log(f"{name}: {uploaded}/{file_size} "
                f"({100.0 * uploaded / file_size:.1f}%)")
    return ResumableUploadResult.COMPLETE


def main():
    folder = sys.argv[1]
    notes = sys.argv[2] if len(sys.argv) > 2 else "chunked upload"
    restore_state()
    signal.signal(signal.SIGTERM, _term_handler)
    signal.signal(signal.SIGINT, _term_handler)
    KaggleApi.upload_complete = chunked_upload_complete
    api = KaggleApi()
    api.authenticate()
    try:
        api.dataset_create_version_cli(folder, notes, quiet=False)
    finally:
        backup_state()


if __name__ == "__main__":
    main()
