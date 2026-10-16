"""1.5 calibration report: difficulty map, closed-book check and twin-consistency baseline from evals/run.py outputs.

Reads outputs/evals/calib_{grid,closedbook,twins}/results.jsonl plus the calibration sets, writes
docs/difficulty_map.md and docs/difficulty_map.csv, and prints the Phase 1 gate checks it can answer:
  - the map spans the range: some cell <= 10 % and some cell >= 90 % base accuracy, and accuracy falls with hops
  - closed-book accuracy <= chance + 2 points
Chance per task = the probes' chance: 1/2 for comparison, else 1/(number of answer-type candidates in the context).
"""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from harness_rl.core.env_paths import HARNESS_RL
from harness_rl.evals.loader import load_set

SETS = HARNESS_RL / "outputs/calibration/sets"
RUNS = HARNESS_RL / "outputs/evals"
DOCS = HARNESS_RL / "docs"
KNOBS = ("hops", "op", "distractors", "ordering")
LEVEL_ABBR = {"none": "n", "low": "l", "high": "h", "very_high": "vh"}


def chance(task: dict) -> float:
    return 0.5 if task["meta"]["cell"]["op"] == "comparison" else 1.0 / task["meta"]["n_candidates"]


def results(run: str) -> list[dict]:
    return [json.loads(l) for l in open(RUNS / run / "results.jsonl")]


def mean_se(xs: list[float]) -> tuple[float, float]:
    m = statistics.mean(xs)
    return m, (statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0)


def difficulty_map() -> tuple[list[dict], dict]:
    tasks = {t["id"]: t for t in load_set("calib_grid", root=SETS)}
    by_task = defaultdict(list)
    meta = defaultdict(lambda: defaultdict(int))
    for r in results("calib_grid"):
        by_task[r["id"]].append(r["score"])
        meta[r["id"]]["invalid"] += int(not r["valid"])
        meta[r["id"]]["trunc"] += int(r["finish_reason"] == "length")
        meta[r["id"]]["abstain"] += int("cannot be determined" in (r["response"].rsplit("</think>", 1)[-1].lower()))
    cells = defaultdict(list)
    for tid, scores in by_task.items():
        t = tasks[tid]
        cells[tuple(t["meta"]["cell"][k] for k in KNOBS)].append((statistics.mean(scores), chance(t), tid))
    rows = []
    for key in sorted(cells):
        accs = [a for a, _, _ in cells[key]]
        m, se = mean_se(accs)
        n_samples = sum(len(by_task[tid]) for _, _, tid in cells[key])
        rows.append({**dict(zip(KNOBS, key)), "n_tasks": len(accs), "acc": round(m, 4), "se": round(se, 4),
                     "chance": round(statistics.mean(c for _, c, _ in cells[key]), 4),
                     "trunc": round(sum(meta[tid]["trunc"] for _, _, tid in cells[key]) / n_samples, 4),
                     "abstain": round(sum(meta[tid]["abstain"] for _, _, tid in cells[key]) / n_samples, 4),
                     "invalid": sum(meta[tid]["invalid"] for _, _, tid in cells[key])})
    by_hops = defaultdict(list)
    for r in rows:
        by_hops[(r["op"], r["hops"])].append(r["acc"])
    hops_curve = {op: [round(statistics.mean(by_hops[(op, h)]), 4) for h in sorted({k[1] for k in by_hops})]
                  for op in sorted({k[0] for k in by_hops})}
    gate = {"min_cell": min(r["acc"] for r in rows), "max_cell": max(r["acc"] for r in rows),
            "has_cell_le_10": any(r["acc"] <= 0.10 for r in rows), "has_cell_ge_90": any(r["acc"] >= 0.90 for r in rows),
            "acc_by_hops_per_operator": hops_curve,
            "falls_with_hops": all(all(a >= b for a, b in zip(c, c[1:])) for c in hops_curve.values()),
            "invalid_total": sum(r["invalid"] for r in rows)}
    return rows, gate


def closed_book() -> dict:
    tasks = {t["id"]: t for t in load_set("calib_closedbook", root=SETS)}
    rs = results("calib_closedbook")
    acc = statistics.mean(r["score"] for r in rs)
    ch = statistics.mean(chance(tasks[r["id"]]) for r in rs)
    return {"n_tasks": len(tasks), "samples": len(rs), "acc": round(acc, 4), "chance": round(ch, 4),
            "pass": acc <= ch + 0.02, "invalid": sum(not r["valid"] for r in rs)}


def twin_consistency() -> dict:
    tasks = {t["id"]: t for t in load_set("calib_twins", root=SETS)}
    score = {(r["id"], r["sample"]): r["score"] for r in results("calib_twins")}
    base = {t["meta"]["pair"]: t["id"] for t in tasks.values() if t["meta"]["role"] == "base"}
    twin = {t["meta"]["pair"]: t["id"] for t in tasks.values() if t["meta"]["role"] == "counterfactual"}
    correct = both = 0
    samples = max(k for _, k in score) + 1
    for pair, b in base.items():
        for k in range(samples):
            if score.get((b, k)) == 1.0:
                correct += 1
                both += int(score.get((twin[pair], k)) == 1.0)
    base_acc = statistics.mean(score[(b, k)] for b in base.values() for k in range(samples))
    twin_acc = statistics.mean(score[(t, k)] for t in twin.values() for k in range(samples))
    return {"pairs": len(base), "samples": samples, "base_acc": round(base_acc, 4), "twin_acc": round(twin_acc, 4),
            "base_correct": correct, "twin_consistency": round(both / correct, 4) if correct else None}


def write_map(rows: list[dict], gate: dict, cb: dict, tw: dict) -> None:
    with open(DOCS / "difficulty_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    levels = [d for d in ("none", "low", "high", "very_high") if any(r["distractors"] == d for r in rows)]
    cols = [(d, o) for d in levels for o in ("path", "reversed", "shuffled")]
    idx = {(r["op"], r["hops"], r["distractors"], r["ordering"]): r for r in rows}
    lines = ["# Multihop difficulty map (plan 1.5)", "",
             "Base Qwen3-8B through Pi-vanilla; 20 held-out tasks per cell x 8 samples (T 0.6, top_p 0.95, top_k 20, "
             "16k budget). Cell value = mean accuracy (chance in the CSV). Generated by "
             "`adapters/multihop/calibration_report.py`; full table with stderr, chance, truncation and abstention "
             "rates in `difficulty_map.csv`.", "",
             "| op | hops | " + " | ".join(f"{LEVEL_ABBR[d]}/{o[0]}" for d, o in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for op in ("chain", "comparison", "intersection"):
        for h in sorted({r["hops"] for r in rows}):
            lines.append(f"| {op} | {h} | " + " | ".join(f"{idx[(op, h, d, o)]['acc']:.2f}" for d, o in cols) + " |")
    lines += ["", "Columns: distractors (n/l/h/vh) / ordering (p/r/s).", "", "## Gate checks", "",
              f"- min cell {gate['min_cell']:.3f}, max cell {gate['max_cell']:.3f}; "
              f"cell <= 10 %: {gate['has_cell_le_10']}; cell >= 90 %: {gate['has_cell_ge_90']}",
              f"- mean accuracy by hops: {json.dumps(gate['acc_by_hops_per_operator'])}; monotone: {gate['falls_with_hops']}",
              f"- closed-book: acc {cb['acc']:.3f} vs chance {cb['chance']:.3f} (pass if <= chance + 0.02): {cb['pass']}",
              f"- twin consistency baseline: {tw['twin_consistency']} (base acc {tw['base_acc']:.3f}, "
              f"twin acc {tw['twin_acc']:.3f}, {tw['base_correct']} correct base samples)",
              f"- invalid rollouts: grid {gate['invalid_total']}, closed-book {cb['invalid']}"]
    (DOCS / "difficulty_map.md").write_text("\n".join(lines) + "\n")


def main():
    rows, gate = difficulty_map()
    cb, tw = closed_book(), twin_consistency()
    write_map(rows, gate, cb, tw)
    print(json.dumps({"gate": gate, "closed_book": cb, "twins": tw}, indent=1))


if __name__ == "__main__":
    main()
