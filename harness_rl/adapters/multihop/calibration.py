"""1.5 calibration sets for the multihop adapter, in the eval-set format, for evals/run.py --sets-root.

  calib_grid        the full grid (288 cells in v2), 20 tasks per cell (difficulty map; run with --samples 8)
  calib_closedbook  300 tasks with the facts removed and abstention not offered, so the model must guess;
                    accuracy must stay within chance + 2 points (no answer is recoverable from the question alone)
  calib_twins       300 base tasks plus their counterfactual twins (twin-consistency baseline)

All seeds come from the held-out range at indices CALIB_OFFSET.. (the eval grid uses indices 0..1799), so calibration
never overlaps training or the eval grid. Writes <root>/<name>/{data.jsonl,manifest.json}.
Usage: python -m harness_rl.adapters.multihop.calibration --root harness_rl/outputs/calibration/sets
"""

import argparse
import hashlib
from pathlib import Path

from harness_rl.core.adapter import rng_for
from harness_rl.evals import common

from . import ADAPTER
from .adapter import heldout_spec
from .render import ANSWER_INSTRUCTION

CALIB_OFFSET = 1_000_000
GRID_PER_CELL = 20
N_CLOSED, N_TWINS = 300, 300
CLOSED_INTRO = ("Answer the question. No facts are given; all names are fictional. Give your best guess: "
                "you must name one entity.")
CLOSED_LINE = "Answer: <a single name>"


def _item(set_name: str, task, prompt: str | None = None, extra: dict | None = None) -> dict:
    return {"id": task.task_id, "set": set_name, "split": "dev", "prompt": prompt or task.prompt,
            "answer": task.meta["answer_text"], "aliases": [], "scorer": "multihop",
            "meta": {"spec": task.spec, "cell": task.meta["cell"], "n_candidates": len(task.meta["candidates"]),
                     **(extra or {})}}


def closed_book_prompt(task) -> str:
    return f"{CLOSED_INTRO}\n\nQuestion: {task.meta['question']}\n\n{ANSWER_INSTRUCTION}\n{CLOSED_LINE}"


def build() -> dict[str, list[dict]]:
    cells = ADAPTER.cells()
    idx = CALIB_OFFSET
    grid = []
    for c in cells:
        for _ in range(GRID_PER_CELL):
            grid.append(_item("calib_grid", ADAPTER.instantiate(heldout_spec(c, idx))))
            idx += 1
    closed, twins = [], []
    for i in range(N_CLOSED):
        t = ADAPTER.instantiate(heldout_spec(cells[i % len(cells)], idx))
        idx += 1
        closed.append(_item("calib_closedbook", t, closed_book_prompt(t), {"closed_book": True}))
    for i in range(N_TWINS):
        t = ADAPTER.instantiate(heldout_spec(cells[i % len(cells)], idx))
        idx += 1
        cf = next(tw for tw in ADAPTER.twins(t, rng_for("calib-twin", t.task_id)) if tw.kind == "counterfactual")
        assert ADAPTER.verify(cf.task, cf.expected).score == 1.0 and cf.expected != t.meta["answer_text"]
        twins.append(_item("calib_twins", t, extra={"pair": t.task_id, "role": "base"}))
        twins.append(_item("calib_twins", cf.task, extra={"pair": t.task_id, "role": "counterfactual"}))
    return {"calib_grid": grid, "calib_closedbook": closed, "calib_twins": twins}


def write(root: Path) -> None:
    for name, rows in build().items():
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        data = "".join(common.dumps_line(r) + "\n" for r in rows).encode()
        (d / "data.jsonl").write_bytes(data)
        manifest = {"name": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(rows),
                    "splits": {"dev": len(rows)}, "adapter": ADAPTER.name, "adapter_version": ADAPTER.version,
                    "seed_indices_from": CALIB_OFFSET, "scorer": "multihop",
                    "note": "calibration set (plan 1.5); held-out seed range, disjoint from training and multihop_grid"}
        (d / "manifest.json").write_text(common.dumps_manifest(manifest))
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    write(Path(ap.parse_args().root))
