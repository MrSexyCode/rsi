"""Evaluate a query program over the facts rendered into the context (plan 1.2), never over the generator's world.

Facts are (subject name, relation, object) triples, where the object is an entity name or, for attributes, an int;
the adapter feeds the triples parsed back from the rendered prompt lines. A lookup with no matching fact gives
UNDETERMINED; a lookup with two different objects, a comparison tie, a comparison of an entity with itself, or an
intersection with several matches gives AMBIGUOUS. AMBIGUOUS wins over UNDETERMINED when both occur.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from harness_rl.adapters.multihop.dsl import Chain, Comparison, Intersection, Program
from harness_rl.core.adapter import AMBIGUOUS, UNDETERMINED

Triple = tuple[str, str, "str | int"]


@dataclass
class Result:
    answer: str  # entity name, UNDETERMINED or AMBIGUOUS
    used: list[Triple] = field(default_factory=list)  # facts the derivation used (the gold path)
    bridges: list[str] = field(default_factory=list)  # entities the derivation passed through, answer excluded
    ends: list[str] = field(default_factory=list)  # chain ends (comparison / intersection)


class Index:
    def __init__(self, facts: list[Triple]):
        self.fwd: dict[tuple[str, str], list] = defaultdict(list)
        self.rev: dict[tuple[str, object], list[str]] = defaultdict(list)
        for s, r, o in facts:
            if o not in self.fwd[(s, r)]:
                self.fwd[(s, r)].append(o)
            self.rev[(r, o)].append(s)

    def get(self, s: str, r: str):
        objs = self.fwd.get((s, r), [])
        if not objs:
            return UNDETERMINED
        return objs[0] if len(objs) == 1 else AMBIGUOUS


def _worst(*xs: str) -> str | None:
    if AMBIGUOUS in xs:
        return AMBIGUOUS
    if UNDETERMINED in xs:
        return UNDETERMINED
    return None


def _chain(idx: Index, c: Chain, res: Result) -> str:
    x = c.start
    for r in c.rels:
        y = idx.get(x, r)
        if y in (UNDETERMINED, AMBIGUOUS):
            return y
        res.used.append((x, r, y))
        res.bridges.append(y)
        x = y
    return x


def execute(program: Program, facts: list[Triple]) -> Result:
    idx = Index(facts)
    res = Result(UNDETERMINED)
    if isinstance(program, Chain):
        end = _chain(idx, program, res)
        res.answer = end
        if end not in (UNDETERMINED, AMBIGUOUS):
            res.bridges.pop()
        return res
    ea, eb = _chain(idx, program.a, res), _chain(idx, program.b, res)
    if bad := _worst(ea, eb):
        res.answer = bad
        return res
    res.ends = [ea, eb]
    if isinstance(program, Comparison):
        va, vb = idx.get(ea, program.attr), idx.get(eb, program.attr)
        if bad := _worst(va, vb):
            res.answer = bad
            return res
        res.used += [(ea, program.attr, va), (eb, program.attr, vb)]
        if ea == eb or va == vb:
            res.answer = AMBIGUOUS
            return res
        res.answer = ea if (va > vb) == (program.cmp == "max") else eb
        res.bridges = [b for b in res.bridges if b != res.answer]
        return res
    assert isinstance(program, Intersection)
    cands = sorted(set(idx.rev.get((program.rel_a, ea), [])) & set(idx.rev.get((program.rel_b, eb), [])))
    if not cands:
        res.answer = UNDETERMINED
    elif len(cands) > 1 or idx.get(cands[0], program.rel_a) != ea or idx.get(cands[0], program.rel_b) != eb:
        res.answer = AMBIGUOUS
    else:
        res.answer = cands[0]
        res.used += [(cands[0], program.rel_a, ea), (cands[0], program.rel_b, eb)]
    return res
