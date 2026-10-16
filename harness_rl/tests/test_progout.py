"""1.6 progout adapter: sandbox rejection and limits, executor vs in-process output, determinism, twins, probes per
cell, verify / leakage / reward. The generic contract is in test_adapter_contract.py (parametrized, includes progout).
"""

import ast
import builtins
import contextlib
import dataclasses
import io
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from harness_rl.adapters.progout import ADAPTER, sandbox
from harness_rl.adapters.progout.generator import generate
from harness_rl.adapters.progout.reward import compute_score
from harness_rl.core import registry
from harness_rl.core.adapter import TrustLevel, make_task, rng_for

PROBE_TASKS_PER_CELL = 16


def pmap(fn, xs):
    with ThreadPoolExecutor(16) as ex:  # subprocess-bound; the sandbox does not use preexec_fn, so threads are safe
        return list(ex.map(fn, xs))


@pytest.fixture(scope="module")
def tasks():
    return pmap(lambda cs: make_task(ADAPTER, *cs), [(c, s) for c in ADAPTER.cells() for s in range(3)])


def test_registered():
    assert registry.get("progout") is ADAPTER
    assert ADAPTER.trust is TrustLevel.EXECUTABLE
    assert len(ADAPTER.cells()) == 27


# ---- sandbox

ADVERSARIAL = {
    "import": "import os\nprint(1)",
    "from_import": "from os import system\nprint(1)",
    "open": "print(open(0))",
    "open_name": "f = open\nprint(1)",
    "attribute": "xs = [1]\nxs.append(2)\nprint(xs)",
    "class_trick": "x = ().__class__.__bases__[0].__subclasses__()\nprint(x)",
    "dunder_name": "print(__builtins__)",
    "dunder_import": "print(__import__(1))",
    "eval": "print(eval(1))",
    "exec": "exec(1)\nprint(1)",
    "getattr": "print(getattr(1, 1))",
    "globals": "print(globals())",
    "string": "print('hi')",
    "while_true": "while True:\n    x = 1\nprint(1)",
    "huge_range": "for i in range(1000):\n    x = i\nprint(1)",
    "range_expr": "for i in range(10 ** 9):\n    x = i\nprint(1)",
    "range_var": "n = 9\nfor i in range(n):\n    x = i\nprint(1)",
    "huge_const": "x = 10000000\nprint(x)",
    "pow": "x = 9 ** 9\nprint(x)",
    "recursion": "def f(n):\n    return f(n + 1)\nprint(f(0))",
    "lambda": "f = lambda n: n\nprint(1)",
    "comprehension": "print([i for i in range(10)])",
    "generator": "print(sum((i for i in range(10))))",
    "two_prints": "print(1)\nprint(2)",
    "no_print": "x = 1",
    "print_not_last": "print(1)\nx = 2",
    "print_in_loop": "for i in range(3):\n    print(i)\nprint(1)",
    "print_kwargs": "print(1, end='')",
    "print_starred": "xs = [1]\nprint(*xs)",
    "alias_builtin": "p = print\np(1)",
    "shadow_builtin": "print = 3\nprint(1)",
    "other_builtin": "print(type(1))",
    "with": "with x:\n    y = 1\nprint(1)",
    "try": "try:\n    x = 1\nexcept:\n    x = 2\nprint(x)",
    "class": "class A:\n    x = 1\nprint(1)",
    "del": "x = 1\ndel x\nprint(1)",
    "global": "global x\nprint(1)",
    "walrus": "print((x := 1))",
    "deep_loops": "for a in range(2):\n    for b in range(2):\n        for c in range(2):\n            for d in range(2):\n"
                  "                x = 1\nprint(1)",
    "syntax": "print(",
}


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_sandbox_rejects_adversarial(name):
    assert sandbox.validate(ADVERSARIAL[name]), name
    assert sandbox.execute(ADVERSARIAL[name]).status == "rejected"  # never executed


HEAVY = "xs = [1] * 1000\nfor a in xs:\n    for b in xs:\n        for c in xs:\n            d = a + b\nprint(d)"
MEMORY = "xs = [1] * 1000\nfor i in range(20):\n    xs = xs + xs\nprint(len(xs))"


def test_sandbox_limits():
    assert sandbox.validate(HEAVY) == [] and sandbox.validate(MEMORY) == []  # valid, but must be stopped
    assert sandbox.execute(HEAVY, timeout=0.5).status == "timeout"
    cpu = sandbox.execute(HEAVY, timeout=20, cpu_s=1)
    assert cpu.status == "error" and cpu.returncode != 0  # killed by RLIMIT_CPU before the wall timeout
    mem = sandbox.execute(MEMORY)
    assert mem.status == "error" and "MemoryError" in mem.stderr


def _in_process(code: str) -> str:
    """Trusted generated programs only: exec in this process for comparison with the sandbox."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<program>", "exec"), {"__builtins__": builtins})
    return buf.getvalue()


def test_generated_programs_validate_and_match_in_process():
    specs = [{"stmts": c.knobs["stmts"], "depth": c.knobs["depth"], "list_len": c.knobs["list_len"],
              "seed": rng_for("raw", c.key, s).randrange(2**31), "rename": 0, "perturb": None}
             for c in ADAPTER.cells() for s in range(8)]  # raw generator output, before sample_spec's filtering
    progs = [generate(s) for s in specs]
    for p in progs:
        assert sandbox.validate(p.code) == [], p.code
    results = pmap(lambda p: sandbox.execute(p.code), progs)
    for p, r in zip(progs, results):
        assert r.status == "ok", (p.code, r)
        assert r.stdout == _in_process(p.code)


# ---- adapter

def test_determinism_by_seed():
    c = ADAPTER.cells()[13]
    assert make_task(ADAPTER, c, 5).to_json() == make_task(ADAPTER, c, 5).to_json()
    assert make_task(ADAPTER, c, 5).to_json() != make_task(ADAPTER, c, 6).to_json()
    sandbox.run.cache_clear()  # same bytes when the program is actually re-executed
    t = make_task(ADAPTER, c, 5)
    assert generate(t.spec).code == t.meta["code"]
    assert ADAPTER.instantiate(t.spec, random.Random(123)).to_json() == t.to_json()  # rng unused: spec suffices


def test_tasks_shape(tasks):
    for t in tasks:
        assert t.prompt.startswith("What does this program print?\n\n```python\n")
        assert t.prompt.endswith("Reason step by step. Then end your response with exactly this line:\n"
                                 "Answer: <the exact printed output>")
        assert t.meta["exec"]["stdout"] == t.answer + "\n"
        k = t.meta["cell"]
        assert ADAPTER.signature(t.spec)[:3] == (k["stmts"], k["depth"], k["list_len"])
        assert sandbox._loop_depth(ast.parse(t.meta["code"]).body) == k["depth"]
        assert len(t.meta["kinds"]) == k["stmts"] + 1  # + the print


def test_twins(tasks):
    twins = pmap(lambda t: ADAPTER.twins(t, rng_for("twins", t.task_id)), tasks)
    for t, tws in zip(tasks, twins):
        by = {tw.kind: tw for tw in tws}
        assert set(by) == {"rename", "perturb"}, t.task_id
        r, p = by["rename"], by["perturb"]
        assert r.expected == t.answer and r.task.meta["code"] != t.meta["code"]
        assert r.task.meta["literals"] == t.meta["literals"]
        assert ADAPTER.signature(r.task.spec) == ADAPTER.signature(t.spec)
        assert p.expected != t.answer and p.expected == p.task.answer
        diff = [i for i, (a, b) in enumerate(zip(t.meta["literals"], p.task.meta["literals"])) if a != b]
        assert len(diff) == 1 and diff[0] in t.meta["perturbable"]
        for tw in tws:
            assert ADAPTER.validity(tw.task).ok and ADAPTER.verify(tw.task, tw.expected).score == 1.0


def test_probes_near_chance_per_cell():
    cells = ADAPTER.cells()
    tasks = pmap(lambda cs: make_task(ADAPTER, *cs), [(c, 7000 + s) for c in cells for s in range(PROBE_TASKS_PER_CELL)])
    n = PROBE_TASKS_PER_CELL
    for probe in ADAPTER.probes():
        for i, c in enumerate(cells):
            ts, rng = tasks[i * n:(i + 1) * n], random.Random(0)
            acc = sum(ADAPTER.verify(t, probe.solve(t, rng)).score for t in ts) / n
            chance = sum(probe.chance(t) for t in ts) / n
            tol = 0.05 + 3 * math.sqrt(max(chance * (1 - chance), 1e-4) / n)
            assert abs(acc - chance) <= tol, (probe.name, c.key, acc, chance)


def test_validity_rejects_bad_tasks(tasks):
    t = tasks[0]
    bad_exec = dataclasses.replace(t, meta={**t.meta, "exec": {**t.meta["exec"], "status": "error"}})
    assert not ADAPTER.validity(bad_exec).ok
    bad_ast = dataclasses.replace(t, meta={**t.meta, "code": "import os\n" + t.meta["code"]})
    assert not ADAPTER.validity(bad_ast).ok
    big = dataclasses.replace(t, answer="12345")
    assert not ADAPTER.validity(big).ok
    lst = next(x for x in tasks if x.answer.startswith("["))
    verbatim = dataclasses.replace(lst, meta={**lst.meta, "code": f"q = {lst.answer}\n" + lst.meta["code"]})
    assert not ADAPTER.validity(verbatim).ok


def test_verify(tasks):
    t = next(x for x in tasks if x.answer.startswith("[") and "," in x.answer)
    assert ADAPTER.verify(t, f"  {t.answer}\n").score == 1.0
    squeezed = ADAPTER.verify(t, t.answer.replace(" ", ""))
    assert squeezed.score == 0.0 and squeezed.components["em_nospace"] == 1.0  # lists exactly as printed
    i = next(x for x in tasks if not x.answer.startswith("["))
    assert ADAPTER.verify(i, str(int(i.answer) + 1)).score == 0.0
    assert ADAPTER.verify(i, "").score == 0.0


def test_leakage(tasks):
    t = next(x for x in tasks if not x.answer.startswith("[") and abs(int(x.answer)) >= 10)
    assert not ADAPTER.leakage("", t)
    assert not ADAPTER.leakage("Trace each variable line by line; track the list after every loop iteration.", t)
    assert ADAPTER.leakage(f"I expect the output to be {t.answer}.", t)
    assert not ADAPTER.leakage(f"I expect the output to be {t.answer}7.", t)  # standalone token only
    line = max(t.meta["code"].splitlines(), key=len).strip()
    assert ADAPTER.leakage(f"Watch out for:   {line}", t)


def test_reward(tasks):
    t = tasks[5]
    info = {"spec": json.dumps(t.spec)}
    good = f"<think>maybe Answer: 0</think>\nIt prints this.\nAnswer: {t.answer}"
    assert compute_score("progout", good, t.answer, info)["score"] == 1.0
    assert compute_score("progout", good, None, {"spec": t.spec})["score"] == 1.0
    assert compute_score("progout", "Answer: nope", t.answer, info)["score"] == 0.0
    assert compute_score("progout", "no answer line", t.answer, info)["format_ok"] == 0.0
    with pytest.raises(ValueError):
        compute_score("progout", good, t.answer + "0", info)
