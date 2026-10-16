"""MultihopAdapter: the in-context multi-hop world compiler v1 (plan 1.2-1.4).

instantiate(spec): build the world (world.build), order and render the facts, render the question, then parse the
rendered prompt back into triples and run the executor on *those*: the gold answer is what the context determines.
The world RNG is rng_for_spec(spec without "edit"), which equals the rng argument for every non-twin spec; a twin
therefore shares its base's world, order and templates, and only its edit differs.
"""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass

from harness_rl.adapters.multihop import probes as P
from harness_rl.adapters.multihop import twins as TW
from harness_rl.adapters.multihop.distractors import feasible, reach, step_ok
from harness_rl.adapters.multihop.dsl import (DISTRACTOR_TYPES, HELDOUT_SEEDS, HOPS, LEVELS, OPS, ORDERINGS,
                                              TRAIN_SEEDS, Chain, Comparison, Intersection, Program, branch_len,
                                              check_spec, program_relations)
from harness_rl.adapters.multihop.executor import execute
from harness_rl.adapters.multihop.names import TYPES, core_words
from harness_rl.adapters.multihop.render import (ABSTAIN, fact_sentence, order_facts, parse_context, pick, prompt,
                                                 question, question_of)
from harness_rl.adapters.multihop.schema import ATTR_OF_TYPE, OUT, RELATIONS, fact_templates
from harness_rl.adapters.multihop.world import Built, build
from harness_rl.core.adapter import (AMBIGUOUS, UNDETERMINED, Cell, Reward, Task, TrustLevel, ValidityReport,
                                     package_version, rng_for, rng_for_spec, spec_id)
from harness_rl.evals.scoring import normalize_answer

_ABSTAIN_NORM = normalize_answer(ABSTAIN)


def sample_path(n: int, target: str, reserved: frozenset, rng: random.Random) -> list[str]:
    """n relations from some non-reserved start type to `target`, intermediate types not reserved, uniform per step."""
    starts = [t for t in TYPES if t not in reserved and reach(t, n, target, reserved)]
    t, rels = rng.choice(starts), []
    for k in range(n, 0, -1):
        r = rng.choice([r for r in OUT[t] if step_ok(r, k, target, reserved)])
        rels.append(r)
        t = RELATIONS[r].obj
    return rels


def sample_program(op: str, hops: int, dtypes: list[str], rng: random.Random) -> dict:
    """A program whose every gold branch admits every requested distractor type (resampled until it does)."""
    L = branch_len(op, hops)
    for _ in range(1000):
        T = rng.choice(TYPES)
        res = frozenset([T])
        if op == "chain":
            prog = {"rels": sample_path(L, T, res, rng)}
            branches = [prog["rels"]]
        elif op == "comparison":
            prog = {"a": sample_path(L, T, res, rng), "b": sample_path(L, T, res, rng), "attr": ATTR_OF_TYPE[T],
                    "cmp": rng.choice(("max", "min"))}
            branches = [prog["a"], prog["b"]]
        else:
            ja, jb = rng.sample([r for r in OUT[T] if RELATIONS[r].obj != T], 2)
            prog = {"a": sample_path(L, RELATIONS[ja].obj, res, rng), "join_a": ja,
                    "b": sample_path(L, RELATIONS[jb].obj, res, rng), "join_b": jb}
            branches = [prog["a"], prog["b"]]
        if all(feasible(d, b, res) for d in dtypes for b in branches):
            return prog
    raise RuntimeError(f"no feasible program for {op} hops={hops} {dtypes}")


def make_spec(cell: Cell, seed: int, rng: random.Random) -> dict:
    k = cell.knobs
    level = k["distractors"]
    dtypes = {"none": [], "low": sorted(rng.sample(DISTRACTOR_TYPES, 2)), "high": list(DISTRACTOR_TYPES),
              "very_high": list(DISTRACTOR_TYPES)}[level]
    return {"seed": seed, "op": k["op"], "hops": k["hops"], "distractors": level, "distractor_types": dtypes,
            "ordering": k["ordering"], "template_mix": "uniform",
            "program": sample_program(k["op"], k["hops"], dtypes, rng)}


def heldout_spec(cell: Cell, index: int) -> dict:
    """The index-th held-out spec overall; seeds come from the reserved HELDOUT_SEEDS range."""
    seed = HELDOUT_SEEDS[index]
    return make_spec(cell, seed, rng_for("multihop", "heldout", seed, cell.key))


def _mentions(text: str, name: str) -> bool:
    """`name` occurs in `text` as whole words, case-insensitively ("Esto" is not in "restored")."""
    return re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE) is not None


def _mention_order(sentence: str, names: list[str]) -> list[str]:
    return [n for _, n in sorted((sentence.index(n), n) for n in names)]


@dataclass
class Rendered:
    built: Built
    keep: list  # (Fact, template index) in display order, after the edit
    triples: dict  # fact id -> (subject name, relation, object name or value), after the edit
    sentences: list[str]
    question: str
    text: str
    edit: dict | None


def render_spec(spec: dict) -> Rendered:
    """World, display order, templates, question and prompt text of a spec (twins: base task plus the edit)."""
    check_spec(spec)
    base = TW.base_spec(spec)
    rng = rng_for_spec(base)
    b = build(base, rng)
    w = b.world
    order = order_facts(w.facts, spec["ordering"], rng)
    mix = spec["template_mix"]
    tmpl = [pick(rng, range(len(fact_templates(f.rel))), mix) for f in order]
    q = question(b.program, b.answer_type, rng, mix)
    triples = {f.id: w.triple(f) for f in order}
    edit = spec.get("edit")
    if edit and edit["kind"] == "counterfactual":
        s, r, _ = triples[edit["fact"]]
        triples[edit["fact"]] = (s, r, edit["obj"])
    keep = [(f, t) for f, t in zip(order, tmpl) if not (edit and edit["kind"] == "ablation" and f.id == edit["fact"])]
    sentences = [fact_sentence(*triples[f.id], t) for f, t in keep]
    return Rendered(b, keep, triples, sentences, q, prompt(sentences, q), edit)


class MultihopAdapter:
    name = "multihop"
    trust = TrustLevel.COMPUTED
    version = package_version(__file__)

    # ---------------------------------------------------------------- grid and specs

    def cells(self) -> list[Cell]:
        return [Cell({"hops": h, "op": op, "distractors": d, "ordering": o})
                for h, op, d, o in itertools.product(HOPS, OPS, LEVELS, ORDERINGS)]

    def sample_spec(self, cell: Cell, rng: random.Random) -> dict:
        return make_spec(cell, rng.randrange(TRAIN_SEEDS.start, TRAIN_SEEDS.stop), rng)

    # ---------------------------------------------------------------- programs

    @staticmethod
    def program(d: dict) -> Program:
        c = lambda x: Chain(x["start"], tuple(x["rels"]))  # noqa: E731
        if "start" in d:
            return c(d)
        if "attr" in d:
            return Comparison(c(d["a"]), c(d["b"]), d["attr"], d["cmp"])
        return Intersection(c(d["a"]), d["rel_a"], c(d["b"]), d["rel_b"])

    @staticmethod
    def program_dict(p: Program) -> dict:
        c = lambda x: {"start": x.start, "rels": list(x.rels)}  # noqa: E731
        if isinstance(p, Chain):
            return c(p)
        if isinstance(p, Comparison):
            return {"a": c(p.a), "b": c(p.b), "attr": p.attr, "cmp": p.cmp}
        return {"a": c(p.a), "rel_a": p.rel_a, "b": c(p.b), "rel_b": p.rel_b}

    # ---------------------------------------------------------------- instantiate

    def instantiate(self, spec: dict, rng: random.Random | None = None) -> Task:
        r = render_spec(spec)
        b, w, keep, sentences, q, text, edit = r.built, r.built.world, r.keep, r.sentences, r.question, r.text, r.edit
        ctx = parse_context(text)
        res = execute(b.program, ctx)
        answer = res.answer
        by_name = {e.name: e for e in w.entities.values()}
        in_ctx = sorted({x for s, _, o in ctx for x in (s, o) if isinstance(x, str)})
        types = {n: by_name[n].type for n in in_ctx}
        if isinstance(b.program, Comparison):
            cands = sorted(res.ends) if res.ends else []
        else:
            cands = sorted(n for n in in_ctx if types[n] == b.answer_type)
        line_of = {tuple(t): i for i, t in enumerate(ctx)}
        meta = {
            "cell": {"hops": spec["hops"], "op": spec["op"], "distractors": spec["distractors"],
                     "ordering": spec["ordering"]},
            "answer_text": ABSTAIN if answer == UNDETERMINED else answer,
            "answer_type": b.answer_type,
            "question": q,
            "program": self.program_dict(b.program),
            "question_entities": [w.name(e) for e in b.starts],
            "bridges": res.bridges,
            "gold_lines": sorted(line_of[t] for t in res.used),
            "candidates": cands,
            "context": [list(t) for t in ctx],
            "fact_ids": [f.id for f, _ in keep],
            "slots": [f.slot for f, _ in keep],
            "roles": [f.role for f, _ in keep],
            "mentions": [_mention_order(s, [x for x in (t[0], t[2]) if isinstance(x, str)])
                         for s, t in zip(sentences, ctx)],
            "entity_types": types,
            "distractors_placed": b.placed,
            "edit": edit,
        }
        if not edit:
            assert answer == w.name(b.answer), (answer, spec)
        return Task(task_id=spec_id(spec, "mh-"), spec=spec, prompt=text, answer=answer, meta=meta)

    # ---------------------------------------------------------------- scoring and checks

    def verify(self, task: Task, final_answer: str | None) -> Reward:
        pred = normalize_answer(final_answer or "")
        abstained = pred == _ABSTAIN_NORM
        if task.answer == UNDETERMINED:
            em = abstained
        else:
            em = task.answer != AMBIGUOUS and pred == normalize_answer(task.meta["answer_text"])
        in_ctx = pred in {normalize_answer(n) for n in task.meta["entity_types"]}
        return Reward(float(em), {"em": float(em), "abstained": float(abstained),
                                  "gold_undetermined": float(task.answer == UNDETERMINED),
                                  "in_context": float(in_ctx)})

    def validity(self, task: Task) -> ValidityReport:
        reasons = []
        try:
            ctx = parse_context(task.prompt)
        except (ValueError, AssertionError, IndexError) as e:
            return ValidityReport(False, [f"context does not parse: {e}"])
        res = execute(self.program(task.meta["program"]), ctx)
        if res.answer in (UNDETERMINED, AMBIGUOUS):
            reasons.append(f"executor answer is {res.answer}")
        elif res.answer != task.answer:
            reasons.append(f"executor answer {res.answer!r} != task answer {task.answer!r}")
        q = question_of(task.prompt)
        if res.answer not in (UNDETERMINED, AMBIGUOUS) and _mentions(q, res.answer):
            reasons.append("answer in question")
        reasons += [f"bridge in question: {x}" for x in res.bridges if _mentions(q, x)]
        seen, pairs = set(), set()
        for s, r, o in ctx:
            if (s, r) in seen:
                reasons.append(f"not functional: {s} {r}")
            seen.add((s, r))
            if isinstance(o, str):
                if frozenset((s, o)) in pairs:
                    reasons.append(f"two facts on one pair: {s}, {o}")
                pairs.add(frozenset((s, o)))
        return ValidityReport(not reasons, reasons)

    def twins(self, task: Task, rng: random.Random) -> list:
        out = [TW.counterfactual(self, task, rng), TW.ablation(self, task, rng)]
        missing = [k for k, t in zip(("counterfactual", "ablation"), out) if t is None]
        if missing:
            raise RuntimeError(f"{task.task_id}: no valid {missing} twin")
        return out

    def probes(self) -> list:
        return list(P.PROBES)

    def signature(self, spec: dict) -> tuple:
        return spec["op"], spec["hops"], program_relations(spec), tuple(sorted(spec["distractor_types"]))

    def leakage(self, scaffold: str, task: Task) -> bool:
        words = set(re.findall(r"[a-z]+", scaffold.lower()))
        asked = set(task.meta["question_entities"])
        return any(w in words for n in task.meta["entity_types"] if n not in asked for w in core_words(n))
