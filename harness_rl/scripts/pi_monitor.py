"""Invariant monitor for Pi training runs: per-step summary of PiAgentLoop's metrics rows, plus operational counters.

Per training step (the rollouts' weight version, `global_steps`): rollouts, valid rate, rollouts with != 1 model call,
prompt-id mismatches vs the direct rendering, Pi overhead p50/p99, truncation rate. Any invalid rollout, extra call or
prompt mismatch is a violation. With --watch it also samples, every --every seconds, the live Pi processes, leftover
per-rollout dirs and free disk, appending to <trace_dir>/monitor.jsonl and printing violations as they appear.

Usage: python -m harness_rl.scripts.pi_monitor RUN_NAME [--watch] [--every 60]
"""

import argparse
import glob
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from harness_rl.core.env_paths import HARNESS_RL

ROLLOUT_DIRS = Path(tempfile.gettempdir()) / "harness_rl_rollouts"


def read_rows(trace_dir: Path) -> list[dict]:
    rows = []
    for p in glob.glob(str(trace_dir / "metrics_*.jsonl")):
        with open(p) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # a line still being written
                    pass
    return rows


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def per_step(rows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[r.get("global_steps")].append(r)
    out = []
    for step in sorted(by, key=lambda s: (s is None, s)):
        rs = by[step]
        ov = [r["overhead_s"] for r in rs if r["calls"] == 1]
        out.append({
            "step": step, "rollouts": len(rs), "valid_rate": round(sum(r["valid"] for r in rs) / len(rs), 5),
            "invalid": sum(not r["valid"] for r in rs), "calls_ne_1": sum(r["calls"] != 1 for r in rs),
            "prompt_mismatch": sum(r["prompt_match"] is False for r in rs),
            "overhead_p50_s": round(statistics.median(ov), 3) if ov else None, "overhead_p99_s": round(pct(ov, 0.99), 3),
            "truncated_rate": round(sum(r["finish"] == "length" for r in rs) / len(rs), 4),
            "problems": sorted({r["problems"] for r in rs if r["problems"]})[:5],
        })
    return out


def system_sample() -> dict:
    procs = subprocess.run(["pgrep", "-fc", "pi_runtime/node_modules"], capture_output=True, text=True).stdout.strip()
    dirs = len(list(ROLLOUT_DIRS.iterdir())) if ROLLOUT_DIRS.exists() else 0
    return {"t": time.time(), "pi_processes": int(procs or 0), "rollout_dirs": dirs,
            "disk_free_gb": round(shutil.disk_usage("/").free / 1e9, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--trace-root", default=str(HARNESS_RL / "outputs" / "traces"))
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=float, default=60.0)
    args = ap.parse_args()
    trace_dir = Path(args.trace_root) / args.run
    reported: set = set()
    while True:
        steps = per_step(read_rows(trace_dir))
        for s in steps:
            bad = s["invalid"] or s["calls_ne_1"] or s["prompt_mismatch"]
            key = (s["step"], s["rollouts"])
            if bad and key not in reported:
                print(f"VIOLATION step={s['step']}: {json.dumps(s)}", flush=True)
                reported.add(key)
        if not args.watch:
            for s in steps:
                print(json.dumps(s))
            break
        sample = {**system_sample(), "steps_seen": len(steps), "last_step": steps[-1] if steps else None}
        with open(trace_dir / "monitor.jsonl", "a") as f:
            f.write(json.dumps(sample) + "\n")
        time.sleep(args.every)


if __name__ == "__main__":
    main()
