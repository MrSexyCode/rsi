"""Build the held-out eval grid and the verl training pool (plan 1.4). CPU only.

    python -m harness_rl.adapters.multihop.build_sets --eval            # evals/sets/multihop_grid/
    python -m harness_rl.adapters.multihop.build_sets --pool [--train 20000 --val 500 --seed 0]
    python -m harness_rl.adapters.multihop.build_sets --eval --check    # rebuild in memory, compare bytes

Eval grid: hops {2,4,6,8,12} x 3 ops x 4 distractor levels, shuffled ordering, 30 per cell (1,800); item j of the
grid (cell-major) uses spec seed HELDOUT_SEEDS[j], a range training never draws from. Byte-identical rebuilds.
Pool: uniform over the full 288-cell grid (row i -> cell i mod 288, then shuffled), each row the canonical draw
make_task(adapter, cell, derive_int(seed, split, i)), whose spec seeds lie in TRAIN_SEEDS. Rows are deterministic by
seed; the parquet bytes are too for a fixed pyarrow version (written without statistics timestamps), so the manifest
records both the file sha256 and a sha256 over the canonical JSON of the rows.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from harness_rl.adapters.multihop import ADAPTER
from harness_rl.adapters.multihop.adapter import heldout_spec
from harness_rl.adapters.multihop.dsl import ADAPTER_RELEASE, HELDOUT_SEEDS, TRAIN_SEEDS
from harness_rl.adapters.multihop.render import ANSWER_LINE, INTRO
from harness_rl.core.adapter import Cell, make_task
from harness_rl.core.versioning import file_hash
from harness_rl.evals import common
from harness_rl.evals.common import SETS_DIR, dumps_line, dumps_manifest, sha256_bytes

SET_NAME = "multihop_grid"
EVAL_HOPS = (2, 4, 6, 8, 12)
EVAL_PER_CELL = 30
POOL_DIR = Path("/home/ubuntu/data/harness_rl/multihop_pool")
WORKERS = 64


def eval_cells() -> list[Cell]:
    return [c for c in ADAPTER.cells() if c.knobs["hops"] in EVAL_HOPS and c.knobs["ordering"] == "shuffled"]


def _cell_key(knobs: dict) -> str:
    return f"h{knobs['hops']}/{knobs['op']}/{knobs['distractors']}/{knobs['ordering']}"


def _token_stats(rows: list[tuple[dict, int]]) -> dict:
    by = defaultdict(list)
    for knobs, n in rows:
        by[_cell_key(knobs)].append(n)
    allt = [n for _, n in rows]
    return {"all": {"mean": round(statistics.mean(allt), 1), "max": max(allt), "min": min(allt)},
            "per_cell": {k: {"mean": round(statistics.mean(v), 1), "max": max(v)} for k, v in sorted(by.items())}}


def _map(fn, jobs: list, workers: int) -> list:
    if workers <= 1:
        return list(map(fn, jobs))
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, jobs, chunksize=16))


# ---------------------------------------------------------------- eval grid

def _eval_item(job) -> dict:
    j, knobs = job
    t = ADAPTER.instantiate(heldout_spec(Cell(knobs), j))
    assert ADAPTER.validity(t).ok, t.task_id
    return {"id": t.task_id, "set": SET_NAME, "split": "dev", "prompt": t.prompt, "answer": t.meta["answer_text"],
            "aliases": [], "scorer": "multihop",
            "meta": {"spec": t.spec, "cell": t.meta["cell"], "question": t.meta["question"],
                     "answer_type": t.meta["answer_type"], "n_candidates": len(t.meta["candidates"]),
                     "prompt_tokens": common.prompt_tokens(t.prompt)}}


def build_eval(workers: int = WORKERS) -> tuple[bytes, bytes]:
    jobs = [(i * EVAL_PER_CELL + k, c.knobs) for i, c in enumerate(eval_cells()) for k in range(EVAL_PER_CELL)]
    rows = _map(_eval_item, jobs, workers)
    data = "".join(dumps_line(r) + "\n" for r in rows).encode()
    assert len({r["id"] for r in rows}) == len(rows)
    toks = _token_stats([(r["meta"]["cell"], r["meta"]["prompt_tokens"]) for r in rows])
    here = Path(__file__).resolve().parent
    manifest = {
        "file_format": "one JSON object per line: id, set, split, prompt, answer, aliases, scorer, meta; sorted keys, "
                       "separators (',', ':'), ensure_ascii=False, '\\n' line ends",
        "generator": {"command": "python -m harness_rl.adapters.multihop.build_sets --eval",
                      "adapter": ADAPTER.name, "adapter_version": ADAPTER.version,
                      "files": {common.rel(p): file_hash(p) for p in
                                [here / "build_sets.py", Path(common.__file__)]}},
        "name": SET_NAME,
        "prompt_adapter": {"name": "multihop_v1", "intro": INTRO, "answer_line": ANSWER_LINE,
                           "description": "rendered by adapters/multihop/render.py; meta.spec rebuilds the task"},
        "prompt_tokens": toks["all"],
        "prompt_tokens_per_cell": toks["per_cell"],
        "sampling": {"method": "item j (cell-major over the cells below) = instantiate(heldout_spec(cell, j))",
                     "cells": [_cell_key(c.knobs) for c in eval_cells()], "per_cell": EVAL_PER_CELL,
                     "seed_range": [HELDOUT_SEEDS.start, HELDOUT_SEEDS.stop],
                     "train_seed_range": [TRAIN_SEEDS.start, TRAIN_SEEDS.stop], "n": len(rows)},
        "schema_version": common.SCHEMA_VERSION,
        "scorer": "multihop",
        "seed": 0,
        "sha256": sha256_bytes(data),
        "size": len(rows),
        "source": {"kind": "generated", "adapter": ADAPTER.name, "adapter_version": ADAPTER.version,
                   "adapter_release": ADAPTER_RELEASE},
        "source_split": "held-out seeds of the multihop world compiler (dsl.HELDOUT_SEEDS)",
        "split_rule": {"rule": "none: all dev",
                       "note": "Synthetic training-distribution set: the dev/final rule applies to natural sets "
                               "only, so there is no final split and every item is dev."},
        "splits": {"dev": {"n": len(rows), "sha256": sha256_bytes(data)},
                   "final": {"n": 0, "sha256": sha256_bytes(b"")}},
        "tokenizer": {"counted": "Qwen3 chat template, one user message, add_generation_prompt, "
                                 "enable_thinking=True", "id": common.TOKENIZER_ID,
                      "revision": common.TOKENIZER_REVISION},
    }
    return data, dumps_manifest(manifest).encode()


# ---------------------------------------------------------------- training pool

def _pool_row(job) -> dict:
    split, i, knobs, draw = job
    t = make_task(ADAPTER, Cell(knobs), draw)
    assert ADAPTER.validity(t).ok, t.task_id
    return {"data_source": "multihop", "prompt": [{"role": "user", "content": t.prompt}],
            "ability": "multihop_ctx",
            "reward_model": {"style": "rule", "ground_truth": t.meta["answer_text"]},
            "extra_info": {"split": split, "index": i, "task_id": t.task_id, "spec": json.dumps(t.spec, sort_keys=True),
                           "cell": t.meta["cell"], "adapter_version": ADAPTER.version,
                           "prompt_tokens": common.prompt_tokens(t.prompt)}}


def pool_rows(split: str, n: int, seed: int, workers: int = WORKERS) -> list[dict]:
    import random

    cells = ADAPTER.cells()
    jobs = [(split, i, cells[i % len(cells)].knobs, common.derive_int(seed, "multihop_pool", split, i))
            for i in range(n)]
    random.Random(f"{seed}|{split}").shuffle(jobs)
    jobs = [(split, i, k, d) for i, (_, _, k, d) in enumerate(jobs)]
    return _map(_pool_row, jobs, workers)


def write_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd", write_statistics=False)


def rows_hash(rows: list[dict]) -> str:
    return sha256_bytes("".join(dumps_line(r) + "\n" for r in rows).encode())


def build_pool(n_train: int, n_val: int, seed: int, out: Path, workers: int = WORKERS) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": pool_rows("train", n_train, seed, workers), "val": pool_rows("val", n_val, seed, workers)}
    ids = [r["extra_info"]["task_id"] for rs in splits.values() for r in rs]
    assert len(set(ids)) == len(ids), "duplicate specs in the pool"
    for rs in splits.values():
        for r in rs:
            assert json.loads(r["extra_info"]["spec"])["seed"] in TRAIN_SEEDS
    manifest = {"adapter": ADAPTER.name, "adapter_version": ADAPTER.version, "adapter_release": ADAPTER_RELEASE,
                "seed": seed,
                "command": f"python -m harness_rl.adapters.multihop.build_sets --pool --train {n_train} "
                           f"--val {n_val} --seed {seed}",
                "row_schema": {"data_source": "multihop", "prompt": "[{role: user, content}]",
                               "ability": "multihop_ctx", "reward_model": "{style: rule, ground_truth: answer_text}",
                               "extra_info": "{split, index, task_id, spec (JSON str), cell, adapter_version, "
                                             "prompt_tokens}"},
                "reward": "harness_rl/adapters/multihop/reward.py::compute_score",
                "sampling": f"row i -> cell i mod {len(ADAPTER.cells())} (uniform over the full grid), then shuffled; "
                            "make_task(adapter, cell, derive_int(seed, 'multihop_pool', split, i))",
                "determinism": "rows deterministic by seed (rows_sha256); parquet bytes identical for a fixed "
                               "pyarrow version (file_sha256)",
                "splits": {}}
    for split, rs in splits.items():
        path = out / f"{split}.parquet"
        write_parquet(rs, path)
        toks = _token_stats([(r["extra_info"]["cell"], r["extra_info"]["prompt_tokens"]) for r in rs])
        manifest["splits"][split] = {"n": len(rs), "file": str(path), "file_sha256": file_hash(path),
                                     "rows_sha256": rows_hash(rs), "prompt_tokens": toks["all"],
                                     "prompt_tokens_per_cell": toks["per_cell"]}
    (out / "manifest.json").write_text(dumps_manifest(manifest))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--check", action="store_true", help="eval: compare with the files on disk instead of writing")
    ap.add_argument("--train", type=int, default=20_000)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=POOL_DIR)
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args(argv)
    rc = 0
    if args.eval:
        data, manifest = build_eval(args.workers)
        d = SETS_DIR / SET_NAME
        if args.check:
            same = (d / "data.jsonl").read_bytes() == data and (d / "manifest.json").read_bytes() == manifest
            print(f"{SET_NAME}: {'up to date' if same else 'STALE'}")
            rc |= not same
        else:
            d.mkdir(parents=True, exist_ok=True)
            (d / "data.jsonl").write_bytes(data)
            (d / "manifest.json").write_bytes(manifest)
            m = json.loads(manifest)
            print(f"{SET_NAME}: {m['size']} items, sha256 {m['sha256']}, prompt tokens {m['prompt_tokens']}")
    if args.pool:
        m = build_pool(args.train, args.val, args.seed, args.out, args.workers)
        for s, v in m["splits"].items():
            print(f"{s}: {v['n']} rows -> {v['file']} ({v['file_sha256'][:12]}), prompt tokens {v['prompt_tokens']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
