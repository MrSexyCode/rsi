"""Program-output prediction (plan 1.6): the second adapter, a stub that keeps the adapter interface domain-agnostic.

A task shows a small generated Python program (generator.py) and asks what it prints. Trust level EXECUTABLE: the
gold answer is the program's stdout, obtained by running it in the sandbox (sandbox.py) once in instantiate() and
cached in task.meta["exec"], so verify / validity / probes never execute code.

Spec: {stmts, depth, list_len, seed, rename, perturb}. sample_spec draws seeds until the instantiated task is
valid and stores the accepted seed, so instantiate is a single generate + run and the spec alone reproduces the task
(the rng instantiate receives is unused: everything derives from spec fields, which lets twins share the base program).

Design choices that differ from multihop, documented here because the contract leaves them to the adapter:
  - validity's "answer not in the question" rule applies to list outputs only (a printed list must not appear verbatim
    in the code). For ints it does not apply: answers are small ints and the program is full of small literals, so
    rejecting collisions would skew answers towards "numbers absent from the program", itself a shortcut. The
    literal-copying shortcut is instead measured by the probes below.
  - probes: last_literal (the plan's) and random_literal. chance(task) = 1/k if the output is an int equal to one of
    the program's k distinct int literals, else 0, i.e. the accuracy of guessing a uniformly random literal.
    random_literal matches it by construction; last_literal near it means the last literal is not privileged.
  - leakage: the whole program is the question, so the plan's "literal not in the question" rule is vacuous here and
    small literals are too common to flag. A scaffold leaks if it contains the printed output as a standalone token,
    or any statement of the program verbatim (>= 8 non-space chars), i.e. task-specific content.
"""

from __future__ import annotations

import ast
import random
import re

from harness_rl.adapters.progout import sandbox
from harness_rl.adapters.progout.generator import generate, min_stmts
from harness_rl.core.adapter import (Cell, Probe, Reward, Task, TrustLevel, Twin, ValidityReport, package_version,
                                     rng_for, rng_for_spec, spec_id)
from harness_rl.evals.prompts import answer_suffix

STMTS = (6, 9, 12)
DEPTHS = (0, 1, 2)
LIST_LENS = (3, 5, 7)
MAX_ABS = 999
MAX_LIST_OUT = 12
MAX_TRIES = 200

ANSWER_LINE = "Answer: <the exact printed output>"
QUESTION = "What does this program print?"


def render_prompt(code: str) -> str:
    return f"{QUESTION}\n\n```python\n{code}\n```\n\n{answer_suffix(ANSWER_LINE)}"


def parse_output(answer: str):
    """The printed value as an int or list of ints, or None if it is neither."""
    try:
        v = ast.literal_eval(answer)
    except (ValueError, SyntaxError):
        return None
    if type(v) is int or (type(v) is list and all(type(x) is int for x in v)):
        return v
    return None


def _token_re(s: str) -> re.Pattern:
    return re.compile(rf"(?<![\w\-]){re.escape(s)}(?![\w])")


class ProgoutAdapter:
    name = "progout"
    version = package_version(__file__)
    trust = TrustLevel.EXECUTABLE

    def cells(self) -> list[Cell]:
        return [Cell({"stmts": n, "depth": d, "list_len": L}) for n in STMTS for d in DEPTHS for L in LIST_LENS
                if n >= min_stmts(d)]

    def sample_spec(self, cell: Cell, rng: random.Random) -> dict:
        k = cell.knobs
        for _ in range(MAX_TRIES):
            spec = {"stmts": k["stmts"], "depth": k["depth"], "list_len": k["list_len"],
                    "seed": rng.randrange(2**31), "rename": 0, "perturb": None}
            task = self.instantiate(spec, rng_for_spec(spec))
            # regenerate rather than ship a task whose output no constant can change (dead or self-cancelling code)
            if self.validity(task).ok and self.perturb_twin(task, rng_for("progout", "perturb-check")):
                return spec
        raise RuntimeError(f"no valid program in {MAX_TRIES} tries for {cell.key}")

    def instantiate(self, spec: dict, rng: random.Random) -> Task:
        prog = generate(spec)
        res = sandbox.run(prog.code)
        answer = res.stdout[:-1] if res.status == "ok" and res.stdout.count("\n") == 1 and res.stdout.endswith("\n") \
            else None
        meta = {"answer_text": answer or "", "cell": {k: spec[k] for k in ("stmts", "depth", "list_len")},
                "code": prog.code, "exec": res.to_dict(), "kinds": prog.kinds, "literals": prog.literals,
                "perturbable": prog.perturbable}
        return Task(task_id=spec_id(spec, "progout-"), spec=spec, prompt=render_prompt(prog.code), answer=answer,
                    meta=meta)

    def verify(self, task: Task, final_answer: str) -> Reward:
        gold, pred = task.answer or "", (final_answer or "").strip()
        em = float(bool(gold) and pred == gold)
        em_nospace = float(bool(gold) and "".join(pred.split()) == "".join(gold.split()))
        return Reward(em, {"em": em, "em_nospace": em_nospace})

    def validity(self, task: Task) -> ValidityReport:
        reasons = []
        code, res = task.meta["code"], task.meta["exec"]
        if errs := sandbox.validate(code):
            reasons.append(f"ast: {errs}")
        if res["status"] != "ok":
            reasons.append(f"exec {res['status']}: {res['stderr'][-200:]}")
        elif task.answer is None:
            reasons.append(f"must print exactly one line, got {res['stdout'][:80]!r}")
        else:
            v = parse_output(task.answer)
            if v is None:
                reasons.append(f"output {task.answer!r} is not an int or a list of ints")
            elif any(abs(x) > MAX_ABS for x in (v if isinstance(v, list) else [v])):
                reasons.append(f"output value outside [-{MAX_ABS}, {MAX_ABS}]")
            elif isinstance(v, list) and (len(v) > MAX_LIST_OUT or _token_re(task.answer).search(code)):
                reasons.append("list output too long or verbatim in the program")
        if f"\n{code}\n" not in task.prompt:
            reasons.append("program missing from prompt")
        return ValidityReport(not reasons, reasons)

    def twins(self, task: Task, rng: random.Random) -> list[Twin]:
        out = []
        spec = {**task.spec, "rename": rng.randrange(1, 2**31)}
        t = self.instantiate(spec, rng_for_spec(spec))
        if t.answer == task.answer and self.validity(t).ok:
            out.append(Twin("rename", t, t.answer))
        if pt := self.perturb_twin(task, rng):
            out.append(pt)
        return out

    def perturb_twin(self, task: Task, rng: random.Random) -> Twin | None:
        """Try every (value literal, new value 0-9) in random order until the output changes; None if none does."""
        cands = [(i, v) for i in task.meta["perturbable"] for v in range(10) if v != task.meta["literals"][i]]
        for i, v in rng.sample(cands, len(cands)):
            spec = {**task.spec, "perturb": [i, v]}
            t = self.instantiate(spec, rng_for_spec(spec))
            if t.answer is not None and t.answer != task.answer and self.validity(t).ok:
                return Twin("perturb", t, t.answer)
        return None

    def probes(self) -> list[Probe]:
        def last_literal(t: Task, rng: random.Random) -> str:
            return str(t.meta["literals"][-1])

        def random_literal(t: Task, rng: random.Random) -> str:
            return str(rng.choice(sorted(set(t.meta["literals"]))))

        return [Probe("last_literal", last_literal, literal_chance), Probe("random_literal", random_literal,
                                                                            literal_chance)]

    def signature(self, spec: dict) -> tuple:
        kinds = generate(spec).kinds
        return (spec["stmts"], spec["depth"], spec["list_len"], tuple(sorted(kinds)))

    def leakage(self, scaffold: str, task: Task) -> bool:
        if not scaffold or not task.answer:
            return False
        if _token_re(task.answer).search(scaffold):
            return True
        flat = " ".join(scaffold.split())
        return any(len(s.replace(" ", "")) >= 8 and s in flat for s in (ln.strip() for ln in task.meta["code"].splitlines()))


def literal_chance(task: Task) -> float:
    """P(a uniformly random distinct int literal of the program equals the printed int); 0 for list outputs."""
    lits = set(task.meta["literals"])
    v = parse_output(task.answer) if task.answer else None
    return 1 / len(lits) if type(v) is int and v in lits else 0.0
