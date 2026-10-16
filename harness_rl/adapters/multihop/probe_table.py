"""Probe table (plan 1.3, CPU only): per-cell probe accuracy against chance on N tasks per cell.

    python -m harness_rl.adapters.multihop.probe_table --n 2000 --workers 64 [--csv out.csv] [--seed0 0]

Tasks are make_task(adapter, cell, seed0 + i), i < N (training-range specs). The gate (plan 1.5/Phase 1) is
|acc - chance| <= 0.05 for every probe on every cell; the summary prints the max gap per probe and every cell over
the gate, with the binomial standard error so noise and bias can be told apart.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor

GATE = 0.05


def _cell_rows(args) -> list[dict]:
    from harness_rl.adapters.multihop import ADAPTER as A
    from harness_rl.core.adapter import Cell, make_task

    knobs, n, seed0 = args
    cell = Cell(knobs)
    probes = A.probes()
    acc = {p.name: 0.0 for p in probes}
    chance = 0.0
    for i in range(n):
        t = make_task(A, cell, seed0 + i)
        rng = random.Random(seed0 + i)
        chance += probes[0].chance(t)
        for p in probes:
            acc[p.name] += A.verify(t, p.solve(t, rng)).score
    c = chance / n
    return [{**knobs, "probe": name, "n": n, "acc": a / n, "chance": c, "gap": a / n - c,
             "se": math.sqrt(max(c * (1 - c), 1e-9) / n)} for name, a in acc.items()]


def table(n: int, workers: int, seed0: int = 0) -> list[dict]:
    from harness_rl.adapters.multihop import ADAPTER as A

    jobs = [(c.knobs, n, seed0) for c in A.cells()]
    with ProcessPoolExecutor(workers) as ex:
        return [r for rows in ex.map(_cell_rows, jobs) for r in rows]


def summary(rows: list[dict]) -> str:
    out = [f"{len({tuple(sorted((k, r[k]) for k in ('hops', 'op', 'distractors', 'ordering'))) for r in rows})} "
           f"cells, n={rows[0]['n']} tasks per cell", "probe                 max|gap|  cell (acc vs chance)"]
    for p in sorted({r["probe"] for r in rows}):
        r = max((r for r in rows if r["probe"] == p), key=lambda r: abs(r["gap"]))
        out.append(f"{p:21s} {abs(r['gap']):.3f}     h={r['hops']} {r['op']} {r['distractors']} {r['ordering']} "
                   f"({r['acc']:.3f} vs {r['chance']:.3f}, se {r['se']:.3f})")
    over = [r for r in rows if abs(r["gap"]) > GATE]
    out.append(f"cells over the {GATE:.0%} gate: {len(over)}")
    out += [f"  {r['probe']} h={r['hops']} {r['op']} {r['distractors']} {r['ordering']}: {r['acc']:.3f} vs "
            f"{r['chance']:.3f}" for r in over]
    for op in ("chain", "comparison", "intersection"):
        cs = [r["chance"] for r in rows if r["op"] == op and r["probe"] == "random_typed"]
        out.append(f"mean chance {op}: {sum(cs) / len(cs):.3f} (min {min(cs):.3f}, max {max(cs):.3f})")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--csv")
    args = ap.parse_args(argv)
    rows = table(args.n, args.workers, args.seed0)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    print(summary(rows))
    return 0 if all(abs(r["gap"]) <= GATE for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
