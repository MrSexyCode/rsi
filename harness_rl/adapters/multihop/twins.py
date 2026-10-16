"""Counterfactual and ablation twins (plan 1.2). A twin is the base spec plus an `edit`, applied to the base task's
rendered fact list (same order, same templates), so the twin differs from its base in exactly one fact line.

  counterfactual  replace the object of one gold-path fact with another entity of the same type that is already in
                  the context, such that the executor's answer on the edited context changes and the twin is valid.
                  Facts and replacements are tried in random order; the decoys guarantee one exists.
  ablation        delete one gold-path fact; the executor must return UNDETERMINED (expected answer: the abstention
                  string). Not a valid task by design (expect_valid=False).
Candidate edits are screened with the executor on the parsed context before the twin is instantiated.
"""

from __future__ import annotations

import random

from harness_rl.adapters.multihop.executor import execute
from harness_rl.adapters.multihop.render import ABSTAIN
from harness_rl.core.adapter import AMBIGUOUS, UNDETERMINED, Task, Twin


def base_spec(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if k != "edit"}


def counterfactual(adapter, task: Task, rng: random.Random) -> Twin | None:
    m = task.meta
    ctx = [tuple(t) for t in m["context"]]
    program = adapter.program(m["program"])
    pairs = {frozenset((s, o)) for s, r, o in ctx if isinstance(o, str)}
    gold = [i for i in m["gold_lines"] if isinstance(ctx[i][2], str)]
    rng.shuffle(gold)
    for i in gold:
        s, r, o = ctx[i]
        zs = sorted(z for z, t in m["entity_types"].items()
                    if t == m["entity_types"][o] and z not in (o, s) and z not in m["question_entities"]
                    and frozenset((s, z)) not in pairs)
        rng.shuffle(zs)
        for z in zs:
            ans = execute(program, ctx[:i] + [(s, r, z)] + ctx[i + 1:]).answer
            if ans in (UNDETERMINED, AMBIGUOUS) or ans == task.answer:
                continue
            spec = {**base_spec(task.spec), "edit": {"kind": "counterfactual", "fact": m["fact_ids"][i], "obj": z}}
            twin = adapter.instantiate(spec)
            if twin.answer == ans and adapter.validity(twin).ok:
                return Twin("counterfactual", twin, twin.meta["answer_text"])
    return None


def ablation(adapter, task: Task, rng: random.Random) -> Twin | None:
    lines = list(task.meta["gold_lines"])
    rng.shuffle(lines)
    for i in lines:
        spec = {**base_spec(task.spec), "edit": {"kind": "ablation", "fact": task.meta["fact_ids"][i]}}
        twin = adapter.instantiate(spec)
        if twin.answer == UNDETERMINED:
            return Twin("ablation", twin, ABSTAIN, expect_valid=False)
    return None
