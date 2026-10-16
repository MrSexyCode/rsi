"""Near-miss distractors (plan 1.2). Each one is inserted next to a gold branch (entities e_0..e_L, relations
r_1..r_L, fact r_i at slot i) and never changes the answer; world.build re-runs the executor after every insertion
and rolls an insertion back if it did.

  same_rel_other_subj  r_i(x) = y for a new subject x of e_{i-1}'s type; y is e_i (a converging near miss, i < L,
                       half the time) or a new entity of e_i's type. Slot i.
  same_subj_other_rel  r'(e_i) = y with r' != r_{i+1}, y new. Slot i + 1.
  dead_end             a 2-3 fact chain leaving e_k by r' != r_{k+1} that never reaches a reserved type.
  branch               leaves e_k (k >= 1, i.e. after hop k; k = 0 when L = 1) by r' != r_{k+1} and continues
                       (reusing the gold relations where the types allow) to a new entity of the branch's end type at
                       slot L: a full-length wrong answer.

Reserved types (the answer type of chain/comparison ends and intersection joins) only ever appear at the end slot,
so every answer-type candidate sits at the same slot with the same number of mentions; that symmetry is what keeps
the positional and frequency probes at chance (see probes.py).
"""

from __future__ import annotations

import random
from functools import lru_cache

from harness_rl.adapters.multihop.schema import OUT, RELATIONS


@lru_cache(maxsize=None)
def reach(t: str, n: int, target: str, reserved: frozenset) -> bool:
    """A relation path of exactly n steps from type t to `target`, intermediate types not reserved."""
    if n == 0:
        return t == target
    return any(step_ok(r, n, target, reserved) for r in OUT[t])


def step_ok(r: str, n: int, target: str, reserved: frozenset) -> bool:
    """Relation r may be the first of the n remaining steps towards `target`."""
    o = RELATIONS[r].obj
    return o == target if n == 1 else (o not in reserved and reach(o, n - 1, target, reserved))


def _free_rels(t: str, avoid: str, reserved: frozenset) -> list[str]:
    return [r for r in OUT[t] if r != avoid and RELATIONS[r].obj not in reserved]


def branch_ks(L: int) -> list[int]:
    return list(range(1, L)) if L >= 2 else [0]


def feasible(dtype: str, rels: list[str], reserved: frozenset) -> bool:
    """Type-level check that a distractor of this type can be attached to a branch with these relations."""
    types = [RELATIONS[rels[0]].subj] + [RELATIONS[r].obj for r in rels]
    L = len(rels)
    if dtype == "same_rel_other_subj":
        return True
    if dtype in ("same_subj_other_rel", "dead_end"):
        return any(_free_rels(types[i], rels[i], reserved) for i in range(L))
    if dtype == "branch":
        return any(step_ok(r, L - k, types[L], reserved) for k in branch_ks(L) for r in OUT[types[k]] if r != rels[k])
    raise ValueError(dtype)


# ---------------------------------------------------------------- insertion (w is a world.World)

def same_rel_other_subj(w, ents: list[str], rels: list[str], reserved: frozenset, rng: random.Random) -> bool:
    L = len(rels)
    i = rng.randint(1, L)
    x = w.new(w.type(ents[i - 1]))
    y = ents[i] if i < L and rng.random() < 0.5 else w.new(w.type(ents[i]))
    return w.add(x, rels[i - 1], y, i, "same_rel_other_subj")


def same_subj_other_rel(w, ents, rels, reserved, rng) -> bool:
    ks = list(range(len(rels)))
    rng.shuffle(ks)
    for i in ks:
        opts = [r for r in _free_rels(w.type(ents[i]), rels[i], reserved) if w.free(ents[i], r)]
        if opts:
            r = rng.choice(opts)
            return w.add(ents[i], r, w.new(RELATIONS[r].obj), i + 1, "same_subj_other_rel")
    return False


def dead_end(w, ents, rels, reserved, rng) -> bool:
    ks = list(range(len(rels)))
    rng.shuffle(ks)
    for k in ks:
        opts = [r for r in _free_rels(w.type(ents[k]), rels[k], reserved) if w.free(ents[k], r)]
        if not opts:
            continue
        x, slot = ents[k], k + 1
        for step in range(rng.randint(2, 3)):
            r = rng.choice(opts)
            y = w.new(RELATIONS[r].obj)
            w.add(x, r, y, slot, "dead_end")
            x, slot = y, slot + 1
            opts = _free_rels(w.type(x), "", reserved)
            if not opts:
                break
        return True
    return False


def branch(w, ents, rels, reserved, rng) -> bool:
    L = len(rels)
    target = w.type(ents[L])
    ks = branch_ks(L)
    rng.shuffle(ks)
    for k in ks:
        first = [r for r in OUT[w.type(ents[k])]
                 if r != rels[k] and w.free(ents[k], r) and step_ok(r, L - k, target, reserved)]
        if not first:
            continue
        x, r = ents[k], rng.choice(first)
        for j in range(k, L):
            y = w.new(RELATIONS[r].obj)
            w.add(x, r, y, j + 1, "branch")
            x = y
            if j + 1 < L:
                n = L - j - 1
                gold = rels[j + 1]
                if RELATIONS[gold].subj == w.type(x) and step_ok(gold, n, target, reserved):
                    r = gold
                else:
                    r = rng.choice([q for q in OUT[w.type(x)] if step_ok(q, n, target, reserved)])
        return True
    return False


INSERT = {"same_rel_other_subj": same_rel_other_subj, "same_subj_other_rel": same_subj_other_rel,
          "dead_end": dead_end, "branch": branch}
