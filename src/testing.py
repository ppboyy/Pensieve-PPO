#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path
import re

# ── CONFIG ───────────────────────────────────────────────────────────
TEST_TRACE_DIR  = Path("test")                # folder with new test traces
TRAIN_TRACE_DIR = Path("posttrain_trace")     # folder train.py reads from
TRAIN_SCRIPT    = "train.py"
TOP_N           = 3

CKPT_DIR     = Path("models")                 # where your .pth files live
CKPT_PATTERN = re.compile(r"nn_model_ep_(\d+)\.pth")
# ────────────────────────────────────────────────────────────────────

def latest_checkpoint():
    candidates = list(CKPT_DIR.glob("nn_model_ep_*.pth"))
    if not candidates:
        return None
    def epoch_of(p):                          # pull number out of the filename
        m = CKPT_PATTERN.search(p.name)
        return int(m.group(1)) if m else -1
    return max(candidates, key=epoch_of)

def main():
    # 1) move top‑N traces
    TEST_TRACE_DIR.mkdir(exist_ok=True)
    TRAIN_TRACE_DIR.mkdir(exist_ok=True)
    traces = sorted([f for f in TEST_TRACE_DIR.iterdir() if f.is_file()])
    to_move = traces[:TOP_N]
    if not to_move:
        print("[INFO] No test traces to move.")
    else:
        for f in to_move:
            dest = TRAIN_TRACE_DIR / f.name
            print(f"[MOVE] {f.name} → {dest}")
            shutil.move(str(f), str(dest))

    # 2) find latest checkpoint
    ckpt = latest_checkpoint()
    if ckpt is None:
        print("[WARN] No checkpoint found; train.py will use its default.")
        cmd = [sys.executable, TRAIN_SCRIPT]
    else:
        print(f"[INFO] Using checkpoint {ckpt.name}")
        cmd = [sys.executable, TRAIN_SCRIPT, str(ckpt)]

    # 3) call train.py and return its exit code
    rc = subprocess.call(cmd)
    return rc

if __name__ == "__main__":
    # keep going until TEST_TRACE_DIR is empty
    while True:
        # if no more files to process, we're done
        if not any(TEST_TRACE_DIR.iterdir()):
            print("[DONE] No more test traces. Exiting.")
            break

        rc = main()
        if rc != 0:
            print(f"[ERROR] train.py exited with {rc}. Aborting.")
            sys.exit(rc)

        # otherwise loop and try to move the next batch of traces
        print("--- Next iteration ---\n")
