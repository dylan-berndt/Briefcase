"""
Restarts retrain_strong_sigreg.py whenever it exits abnormally (it resumes from its own checkpoint).
Gives up only after MAX_FAST_FAILURES consecutive runs that made no checkpoint progress.

    python -u experiments/embedding-geometry/run_supervised.py
"""
import json
import os
import subprocess
import sys
import time

scriptDir = os.path.dirname(os.path.abspath(__file__))
repoRoot = os.path.dirname(os.path.dirname(scriptDir))
os.chdir(repoRoot)

progressPath = os.path.join("checkpoints", "pretrain", os.environ.get("RUN_NAME", "strong-sigreg-64px-patch8"), "progress.json")
MAX_FAST_FAILURES = 30


def savedStep():
    try:
        with open(progressPath) as f:
            return json.load(f)["step"]
    except Exception:
        return -1


failures = 0
attempt = 0
while failures < MAX_FAST_FAILURES:
    attempt += 1
    before = savedStep()
    print(f"[supervisor] attempt {attempt}, saved step {before}, {time.strftime('%H:%M:%S')}", flush=True)
    code = subprocess.call([sys.executable, "-u", os.path.join(scriptDir, "retrain_strong_sigreg.py")])
    after = savedStep()
    print(f"\n[supervisor] exit code {code}, saved step {after}", flush=True)
    if code == 0:
        break
    failures = 0 if after > before else failures + 1
    time.sleep(min(60, 10 * (failures + 1)))
print("[supervisor] finished", flush=True)
