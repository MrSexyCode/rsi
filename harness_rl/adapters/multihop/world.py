"""The world graph behind one task: entities, functional facts, and the construction of the query structure,
decoys, distractors and attributes (plan 1.2).

The world *is* the context: every fact here gets rendered (twins then edit the rendered list). Constraints enforced
by World.add: subject/object types match the relation, at most one object per (subject, relation) (functional), and
at most one fact per unordered entity pair (so no fact ever restates or inverts another, which would let a model
recover an ablated fact).

Structure, per operator (L = dsl.branch_len; every reserved-type entity sits at slot L, or L + 1 for joins and
attributes, with the same number of mentions, so positional and frequency shortcuts stay at chance):
  chain         gold chain S -r1..rL-> answer, plus CHAIN_DECOYS[level] chains with the same relations from new starts.
  comparison    gold chains from S_a and S_b to ends of type T, COMPARISON_DECOYS[level] decoy chains per side, then one
                attribute fact on every T entity. Values are distinct; the answer's rank among them is uniform over
                all but the extreme loser-side position, so "pick the global extreme" is not a shortcut.
  intersection  n = INTERSECTION_GRID[level] chain ends per side (gold + decoys), and 2n candidates c_i = (a_i, b_i),
                d_i = (a_i, b_{i+1}) joined to them; the answer is c_0, the only candidate joined to both gold ends.
Reserved types (the answer type) never occur inside a branch, so the answer type is only found at answer slots.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from harness_rl.adapters.multihop import distractors as D
from harness_rl.adapters.multihop.dsl import (CHAIN_DECOYS, COMPARISON_DECOYS, INTERSECTION_GRID, Chain, Comparison,
                                              Intersection, Program, distractors_per_type)
from harness_rl.adapters.multihop.executor import execute
from harness_rl.adapters.multihop.names import NameGen
from harness_rl.adapters.multihop.schema import ATTR_OF_TYPE, ATTRIBUTES, RELATIONS


@dataclass
class Fact:
    id: int
    subj: str  # entity id
    rel: str
    obj: str | int  # entity id, or the value of an attribute
    slot: int
    role: str  # gold | decoy | join | attr | <distractor type>


@dataclass
class Entity:
    id: str
    type: str
    name: str


class World:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.namegen = NameGen(rng)
        self.entities: dict[str, Entity] = {}
        self.facts: list[Fact] = []
        self._fn: set[tuple[str, str]] = set()
        self._pairs: set[frozenset] = set()

    def new(self, etype: str) -> str:
        eid = f"e{len(self.entities)}"
        self.entities[eid] = Entity(eid, etype, self.namegen.new(etype))
        return eid

    def type(self, eid: str) -> str:
        return self.entities[eid].type

    def name(self, eid: str) -> str:
        return self.entities[eid].name

    def free(self, subj: str, rel: str) -> bool:
        return (subj, rel) not in self._fn

    def add(self, subj: str, rel: str, obj, slot: int, role: str) -> bool:
        if rel in RELATIONS:
            r = RELATIONS[rel]
            assert self.type(subj) == r.subj and self.type(obj) == r.obj, (rel, subj, obj)
            pair = frozenset((subj, obj))
            if subj == obj or pair in self._pairs:
                return False
        else:
            assert self.type(subj) == ATTRIBUTES[rel].subj
            pair = None
        if (subj, rel) in self._fn:
            return False
        self._fn.add((subj, rel))
        if pair:
            self._pairs.add(pair)
        self.facts.append(Fact(len(self.facts), subj, rel, obj, slot, role))
        return True

    def walk(self, start: str, rels: list[str], role: str) -> list[str]:
        ents = [start]
        for i, r in enumerate(rels):
            ents.append(self.new(RELATIONS[r].obj))
            assert self.add(ents[-2], r, ents[-1], i + 1, role)
        return ents

    def mark(self) -> tuple[int, int]:
        return len(self.facts), len(self.entities)

    def rollback(self, m: tuple[int, int]) -> None:
        nf, ne = m
        for f in self.facts[nf:]:
            self._fn.discard((f.subj, f.rel))
            if f.rel in RELATIONS:
                self._pairs.discard(frozenset((f.subj, f.obj)))
        del self.facts[nf:]
        for eid in list(self.entities)[ne:]:
            del self.entities[eid]

    def triple(self, f: Fact) -> tuple:
        return self.name(f.subj), f.rel, (self.name(f.obj) if f.rel in RELATIONS else f.obj)

    def triples(self) -> list[tuple]:
        return [self.triple(f) for f in self.facts]


@dataclass
class Built:
    world: World
    program: Program
    answer: str  # entity id
    starts: list[str]  # question entity ids
    branches: list[tuple[list[str], list[str]]]  # gold (entities, relations)
    answer_type: str
    placed: dict = field(default_factory=dict)  # distractor type -> count actually inserted


def _type_of_rels(rels: list[str]) -> str:
    return RELATIONS[rels[0]].subj


def _sub_answers(w: World, program: Program) -> tuple:
    """What a distractor must not change: the chain ends, and the full answer once it is computable."""
    t = w.triples()
    if isinstance(program, Chain):
        return (execute(program, t).answer,)
    ends = (execute(program.a, t).answer, execute(program.b, t).answer)
    return ends if isinstance(program, Comparison) else ends + (execute(program, t).answer,)


def build(spec: dict, rng: random.Random) -> Built:
    op, h, prog = spec["op"], spec["hops"], spec["program"]
    w = World(rng)
    if op == "chain":
        rels = prog["rels"]
        T = RELATIONS[rels[-1]].obj
        gold = w.walk(w.new(_type_of_rels(rels)), rels, "gold")
        for _ in range(CHAIN_DECOYS[spec["distractors"]]):
            w.walk(w.new(_type_of_rels(rels)), rels, "decoy")
        program = Chain(w.name(gold[0]), tuple(rels))
        built = Built(w, program, gold[-1], [gold[0]], [(gold, rels)], T)
    elif op == "comparison":
        a, b, attr = prog["a"], prog["b"], prog["attr"]
        T = RELATIONS[a[-1]].obj
        ga = w.walk(w.new(_type_of_rels(a)), a, "gold")
        gb = w.walk(w.new(_type_of_rels(b)), b, "gold")
        for rels in (a, b):
            for _ in range(COMPARISON_DECOYS[spec["distractors"]]):
                w.walk(w.new(_type_of_rels(rels)), rels, "decoy")
        program = Comparison(Chain(w.name(ga[0]), tuple(a)), Chain(w.name(gb[0]), tuple(b)), attr, prog["cmp"])
        built = Built(w, program, "", [ga[0], gb[0]], [(ga, a), (gb, b)], T)
    else:
        a, b, ja, jb = prog["a"], prog["b"], prog["join_a"], prog["join_b"]
        T = RELATIONS[ja].subj
        n = INTERSECTION_GRID[spec["distractors"]]
        ga = w.walk(w.new(_type_of_rels(a)), a, "gold")
        gb = w.walk(w.new(_type_of_rels(b)), b, "gold")
        ends_a = [ga[-1]] + [w.walk(w.new(_type_of_rels(a)), a, "decoy")[-1] for _ in range(n - 1)]
        ends_b = [gb[-1]] + [w.walk(w.new(_type_of_rels(b)), b, "decoy")[-1] for _ in range(n - 1)]
        cands = []
        for i in range(n):
            for j in (i, (i + 1) % n):
                e = w.new(T)
                assert w.add(e, ja, ends_a[i], h, "join") and w.add(e, jb, ends_b[j], h, "join")
                cands.append(e)
        program = Intersection(Chain(w.name(ga[0]), tuple(a)), ja, Chain(w.name(gb[0]), tuple(b)), jb)
        built = Built(w, program, cands[0], [ga[0], gb[0]], [(ga, a), (gb, b)], T)

    reserved = frozenset([T])
    ref = _sub_answers(w, program)
    per = distractors_per_type(spec["distractors"], h)
    jobs = [t for t in spec["distractor_types"] for _ in range(per)]
    rng.shuffle(jobs)
    jobs.sort(key=lambda t: t != "branch")  # branches need a free relation on a path entity: place them first
    built.placed = {t: 0 for t in spec["distractor_types"]}
    for dtype in jobs:
        for bi in rng.sample(range(len(built.branches)), len(built.branches)):
            ents, rels = built.branches[bi]
            m = w.mark()
            if D.INSERT[dtype](w, ents, rels, reserved, rng) and _sub_answers(w, program) == ref:
                built.placed[dtype] += 1
                break
            w.rollback(m)

    if op == "comparison":
        _attributes(w, built, prog["attr"], prog["cmp"], T, h, rng)
    ans = execute(program, w.triples()).answer
    assert ans == w.name(built.answer), (ans, spec)
    return built


def _attributes(w: World, built: Built, attr: str, cmp: str, T: str, slot: int, rng: random.Random) -> None:
    holders = [e for e in w.entities if w.type(e) == T]
    end_a, end_b = built.branches[0][0][-1], built.branches[1][0][-1]
    win, lose = (end_a, end_b) if rng.random() < 0.5 else (end_b, end_a)
    a = ATTRIBUTES[attr]
    vals = sorted(rng.sample(range(a.lo, a.hi + 1), len(holders)))
    N = len(vals)
    if cmp == "max":
        wi = rng.randint(1, N - 1)
        li = rng.randint(0, wi - 1)
    else:
        wi = rng.randint(0, N - 2)
        li = rng.randint(wi + 1, N - 1)
    rest = [v for i, v in enumerate(vals) if i not in (wi, li)]
    rng.shuffle(rest)
    value = {win: vals[wi], lose: vals[li]}
    for e in holders:
        if e not in value:
            value[e] = rest.pop()
    for e in holders:
        assert w.add(e, attr, value[e], slot, "attr")
    built.answer = win
    assert ATTR_OF_TYPE[T] == attr
