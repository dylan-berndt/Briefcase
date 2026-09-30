"""
Copies a training run's encoder checkpoint aside every --every steps, so a good mid-run state can still be probed later
(train.py overwrites checkpoint.pt every 500 steps).

Written to be unable to disturb the run:
  * standard library only -- no torch, no CUDA, so it cannot touch GPU memory; a few MB of host RAM; files are streamed.
  * it acts only when progress.json shows a new multiple of --every. train.py writes progress.json LAST in its save
    (checkpoint.pt, config.json, train_state.pt, then progress.json), and the next save is ~500 steps (~15 min) away, so
    the copy happens in a quiet window. On Windows a reader that overlaps train.py's atomic os.replace would make that
    save fail, so it waits --settle seconds after the step changes and re-checks the step before and after copying.
  * it never writes into the run directory; snapshots go to checkpoints/pretrain/<run>-snapshots/step<N>/
    (checkpoint.pt + config.json; the optimizer state is not copied).
  * it skips a snapshot if free disk is below --minFreeGB, copies into a .tmp folder, verifies a SHA-256 of the copy
    against the source, and only then renames it.
  * every error is logged and retried on the next poll; it never raises out of the loop.

    python -u experiments/levjepa/snapshot_watcher.py --run levjepa-d256-e200 --total 59000
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

_repoRoot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def readStep(progressPath):
    try:
        with open(progressPath) as f:
            return int(json.load(f)["step"])
    except Exception:
        return None


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


def wanted(step, every, total):
    return step is not None and step > 0 and (step % every == 0 or step >= total)


def snapshot(runDir, outDir, step, progressPath, minFreeBytes):
    final = os.path.join(outDir, f"step{step}")
    if os.path.exists(os.path.join(final, "config.json")):
        return "exists"
    os.makedirs(outDir, exist_ok=True)
    free = shutil.disk_usage(outDir).free
    if free < minFreeBytes:
        log(f"step {step}: only {free / 2 ** 30:.1f} GiB free, skipping")
        return "skipped"
    tmp = final + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for name in ("checkpoint.pt", "config.json"):
        shutil.copyfile(os.path.join(runDir, name), os.path.join(tmp, name))
    if readStep(progressPath) != step:            # the trainer saved again while we copied: discard, try later
        shutil.rmtree(tmp, ignore_errors=True)
        return "raced"
    for name in ("checkpoint.pt", "config.json"):
        if sha256(os.path.join(runDir, name)) != sha256(os.path.join(tmp, name)):
            shutil.rmtree(tmp, ignore_errors=True)
            return "mismatch"
    if readStep(progressPath) != step:
        shutil.rmtree(tmp, ignore_errors=True)
        return "raced"
    os.replace(tmp, final)
    return "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="levjepa-d256-e200")
    p.add_argument("--root", default=os.path.join("checkpoints", "pretrain"))
    p.add_argument("--every", type=int, default=5000)
    p.add_argument("--total", type=int, default=59000, help="also snapshot the final step and then exit")
    p.add_argument("--poll", type=float, default=60.0)
    p.add_argument("--settle", type=float, default=20.0)
    p.add_argument("--minFreeGB", type=float, default=5.0)
    args = p.parse_args()

    os.chdir(_repoRoot)
    runDir = os.path.join(args.root, args.run)
    outDir = os.path.join(args.root, args.run + "-snapshots")
    progressPath = os.path.join(runDir, "progress.json")
    log(f"watching {progressPath}; snapshots every {args.every} steps and at {args.total} -> {outDir}")

    handled = set()
    while True:
        try:
            step = readStep(progressPath)
            if wanted(step, args.every, args.total) and step not in handled:
                time.sleep(args.settle)
                if readStep(progressPath) == step:
                    result = snapshot(runDir, outDir, step, progressPath, args.minFreeGB * 2 ** 30)
                    log(f"step {step}: {result}")
                    if result in ("ok", "exists"):
                        handled.add(step)
                    elif result == "skipped":       # low disk: don't retry (and log) every poll
                        time.sleep(600)
                    if result in ("ok", "exists") and step >= args.total:
                        log("final step snapshotted, exiting")
                        return
        except Exception as error:      # never let the watcher die or spin
            log(f"error: {error!r}")
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
